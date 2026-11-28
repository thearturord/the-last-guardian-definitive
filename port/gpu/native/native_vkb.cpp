// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_vkb.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>

#include "core/libraries/kernel/process.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "native/native_debug_server.h"
#include "native/native_display.h"
#include "native/native_frame_events.h"
#include "native/native_frame_gate.h"
#include "native/native_framewalk.h"
#include "native/native_timeline.h"
#include "native/native_vkb_compute.h"
#include "native/native_vkb_graphics.h"
#include "native/native_vkb_upload.h"
#include "native/native_vkb_objects.h"
#include "native/native_vkb_stream.h"
#include "native/native_vkb_side_census.h"
#include "native/native_vkb_work_census.h"

namespace Core::Vkb {

namespace {

/// One sync record of a queue, as its hook typed it
struct Entry {
    enum class Kind : u8 { Fence, Write, Dma } kind{};
    NativeSync::Fence fence{};
    u64 address{};
    std::array<u32, 4> data{};
    u32 bytes{};
    NativeSync::DmaCopy dma{};
    u64 unit{}; ///< K1: the submit unit (JsComputeQueue_Submit) this record ends, on a shared label (0: none)
};

/// api-contract.md K1: the engine's compute submit unit (JsComputeQueue_Submit 02472ed0) on a shared label (e.g. the AABB readback's, RenderDev +0x2c0): the CPU sets
/// the label to 0x11, the unit's end sets 1 then 0x10, and the reader (JsComputeContext_StallUntilSignaled) takes 0x10 as "the unit last submitted on it is done". On the
/// console an older unit's end comes before the newer submit (the queue is idle within the frame); ours runs on the completion thread after the GPU, so it could land after
/// the newer unit's 0x11 and say 0x10 while the newer unit's copies are still to run (the reader then read half-copied records: the crash after the shield cutscene,
/// 01c4c570). The newest unit owns the label; an older unit's end records are dropped. LEGACY(label_units): TLG_LEGACY_LABEL_UNITS=1 writes them all.
std::mutex label_mutex;
std::unordered_map<u64, u64> label_units; ///< label -> its newest unit
u64 label_unit_count = 0;
std::atomic<u64> c_label_units{0}, c_label_stale_dropped{0};
bool LabelUnitsOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_LEGACY_LABEL_UNITS"); return e == nullptr || e[0] != '1'; }();
    return on;
}
/// True when `entry` ends a unit that is no longer its label's newest (call under label_mutex)
bool StaleUnitLocked(const Entry& entry, u64 address) {
    if (entry.unit == 0) {
        return false;
    }
    const auto it = label_units.find(address);
    return it != label_units.end() && it->second != entry.unit;
}

constexpr u32 NumQueues = 65; ///< graphics + the 64 Gnm virtual compute queues
std::mutex mutex;
std::array<std::vector<Entry>, NumQueues> lists;

std::atomic<u64> c_fences[2]{}; ///< [graphics, compute] fences run
std::atomic<u64> c_interrupts[2]{}, c_writes{0}, c_dma{0}, c_dma_bytes{0}, c_dma_ignored{0}, c_flips{0}, c_submits[2]{}, c_bad_queue{0}, c_most_pending{0};

u32 Clamp(u32 queue) {
    if (queue >= NumQueues) {
        c_bad_queue.fetch_add(1, std::memory_order_relaxed);
        return GraphicsQueue;
    }
    return queue;
}

void Push(u32 queue, Entry&& entry) {
    std::scoped_lock lock{mutex};
    auto& list = lists[Clamp(queue)];
    list.push_back(std::move(entry));
    if (list.size() > c_most_pending.load(std::memory_order_relaxed)) {
        c_most_pending.store(list.size(), std::memory_order_relaxed);
    }
}

void WriteGuest(void* address, u64 data, u32 bytes) {
    if (address != nullptr && bytes != 0) {
        std::memcpy(address, &data, bytes);
    }
}

/// Runs one record. Graphics: the end-of-pipe interrupt is the frame's (frames in flight gate, the game's EOP event); compute: the release interrupt of the queue's pipe.
void Run(const Entry& entry, bool graphics, u32 pipe_id) {
    switch (entry.kind) {
    case Entry::Kind::Fence: {
        std::unique_lock unit_lock{label_mutex, std::defer_lock};
        bool stale = false;
        if (entry.unit != 0) {
            unit_lock.lock(); // K1: held over the write (BeginLabelUnit restores the engine's 0x11 under it)
            stale = StaleUnitLocked(entry, entry.fence.address);
            if (stale) {
                c_label_stale_dropped.fetch_add(1, std::memory_order_relaxed);
            }
        }
        const auto irq = [&] {
            c_interrupts[graphics ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
            if (graphics) {
                FrameGate::NoteRasterReached(); // our side has executed everything the frame recorded (nothing, until V2)
                FrameGate::NoteEndOfPipe();
                NativeFrameEvents::EndOfPipe();
            } else {
                NativeFrameEvents::ComputeRelease(pipe_id);
            }
        };
        if (stale) {
            entry.fence.Signal([](void*, u64, u32) {}, irq); // K1: an older unit's end: its interrupt, not its label value
        } else {
            entry.fence.Signal(WriteGuest, irq);
        }
        c_fences[graphics ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
        break;
    }
    case Entry::Kind::Write: {
        std::unique_lock unit_lock{label_mutex, std::defer_lock};
        if (entry.unit != 0) {
            unit_lock.lock();
            if (StaleUnitLocked(entry, entry.address)) {
                c_label_stale_dropped.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
        std::memcpy(reinterpret_cast<void*>(entry.address), entry.data.data(), entry.bytes);
        c_writes.fetch_add(1, std::memory_order_relaxed);
        break;
    }
    case Entry::Kind::Dma: {
        const auto& dma = entry.dma;
        using K = NativeSync::DmaCopy::Kind;
        if (dma.kind == K::MemoryToMemory && dma.destination != 0 && dma.source != 0) {
            std::memmove(reinterpret_cast<void*>(dma.destination), reinterpret_cast<const void*>(dma.source), dma.bytes);
            c_dma.fetch_add(1, std::memory_order_relaxed);
            c_dma_bytes.fetch_add(dma.bytes, std::memory_order_relaxed);
        } else if (dma.kind == K::FillGuest && dma.destination != 0) {
            u32* to = reinterpret_cast<u32*>(dma.destination);
            for (u32 i = 0; i < dma.bytes / 4; ++i) {
                to[i] = dma.fill;
            }
            c_dma.fetch_add(1, std::memory_order_relaxed);
            c_dma_bytes.fetch_add(dma.bytes, std::memory_order_relaxed);
        } else {
            c_dma_ignored.fetch_add(1, std::memory_order_relaxed); // GDS: no GPU work reads it yet
        }
        break;
    }
    }
}

void RunList(u32 queue, u32 pipe_id) {
    std::vector<Entry> taken;
    {
        std::scoped_lock lock{mutex};
        taken.swap(lists[Clamp(queue)]);
    }
    for (const Entry& entry : taken) {
        Run(entry, queue == GraphicsQueue, pipe_id);
    }
}

} // namespace

bool SkipEnginePs4() {
    static const bool on = [] {
        return On() && !WorkCensus::On() && !SideCensus::On();
    }();
    return on;
}
std::array<std::atomic<u64>, static_cast<size_t>(EngineCall::Count)> engine_calls_skipped{};
void NoteEngineCallSkipped(EngineCall call) {
    engine_calls_skipped[static_cast<size_t>(call)].fetch_add(1, std::memory_order_relaxed);
}
nlohmann::json EngineCallsSkipped() {
    static constexpr const char* names[] = {"clear_rtv", "clear_dsv", "copy_texture", "target_eliminate", "clear_quad", "htile_clear", "fast_clear_eliminate"};
    nlohmann::json j = nlohmann::json::object();
    j["on"] = SkipEnginePs4();
    for (size_t i = 0; i < engine_calls_skipped.size(); ++i) {
        j[names[i]] = engine_calls_skipped[i].load(std::memory_order_relaxed);
    }
    return j;
}

bool On() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_VKB");
        return e != nullptr && *e == '1';
    }();
    return on;
}

namespace DevLog {
namespace {
std::array<std::atomic<u64>, static_cast<size_t>(Kind::Count)> counts{};
} // namespace
bool On() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_VKB_DEV_LOG");
        return (e != nullptr && e[0] == '1') || !Core::Vkb::On();
    }();
    return on;
}
void Count(Kind kind) {
    counts[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}
nlohmann::json Report() {
    static constexpr const char* names[] = {"scene_slot_groups_seen", "compute_shaders", "staging_reads", "copies_into_staging_textures",
                                            "begin_recordings", "aabb_read_checks_guard_on", "aabb_read_checks_guard_off"};
    static_assert(std::size(names) == static_cast<size_t>(Kind::Count));
    nlohmann::json j = nlohmann::json::object();
    j["on"] = On();
    j["switch"] = "TLG_VKB_DEV_LOG=1 writes the developer dumps";
    for (size_t i = 0; i < counts.size(); ++i) {
        j[names[i]] = counts[i].load(std::memory_order_relaxed);
    }
    return j;
}
} // namespace DevLog

namespace Census {
namespace {
u64 Key(u64 address) {
    return address & 0xfffffffffffcull;
}
struct Mark {
    bool async{};
    u64 g_draws{}, g_disp{}, a_disp{}, batch{}; ///< positions at the signal; batch: async dispatches in the batch the label closes
    u64 frame{};
    bool waited{};
};
std::mutex m;
u64 g_draws = 0, g_disp = 0, a_disp = 0, a_since_unit = 0, frame = 0;
std::unordered_map<u64, Mark> marks;
std::deque<u64> order; ///< labels, oldest first (bounded)
struct Agg {
    u64 n{}, draws{}, disp{}, a_disp{}, batch{}, min_draws{~0ull}, zero_draws{};
};
Agg g_wait_async, g_wait_graphics, a_wait_graphics, a_wait_async;
u64 g_wait_unknown = 0, a_wait_unknown = 0, batches = 0, batch_disp = 0, batches_never_waited = 0, batch_disp_never_waited = 0, frames = 0;
FILE* log_file = nullptr;
u32 log_frames = 0;
void Log(const char* what, u64 label, const Mark* mk) {
    static const u64 from = [] { const char* e = std::getenv("TLG_VKB_ASYNC_CENSUS_FROM"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 1600ull; }();
    if (frame < from || log_frames >= 3) {
        return;
    }
    if (log_file == nullptr) {
        log_file = std::fopen((Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_async_census.txt").string().c_str(), "w");
        if (log_file == nullptr) {
            return;
        }
    }
    std::fprintf(log_file, "f%llu t%s %s label %llx | g_draws %llu g_disp %llu a_disp %llu", static_cast<unsigned long long>(frame), Common::GetCurrentThreadName().c_str(), what,
                 static_cast<unsigned long long>(label), static_cast<unsigned long long>(g_draws), static_cast<unsigned long long>(g_disp), static_cast<unsigned long long>(a_disp));
    if (mk != nullptr) {
        std::fprintf(log_file, " | signal: %s g_draws %llu g_disp %llu a_disp %llu batch %llu", mk->async ? "async" : "graphics", static_cast<unsigned long long>(mk->g_draws),
                     static_cast<unsigned long long>(mk->g_disp), static_cast<unsigned long long>(mk->a_disp), static_cast<unsigned long long>(mk->batch));
    }
    std::fputc(10, log_file);
}
void Waited(Agg& agg, const Mark& mk) {
    ++agg.n;
    const u64 draws = g_draws - mk.g_draws;
    agg.draws += draws;
    agg.disp += g_disp - mk.g_disp;
    agg.a_disp += a_disp - mk.a_disp;
    agg.batch += mk.batch;
    agg.min_draws = std::min(agg.min_draws, draws);
    agg.zero_draws += draws == 0 ? 1 : 0;
}
} // namespace

bool On() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_ASYNC_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
void Signal(u64 label, bool async) {
    std::scoped_lock lock{m};
    const u64 key = Key(label);
    auto [it, fresh] = marks.try_emplace(key);
    if (!fresh && it->second.async == async && it->second.frame == frame && !it->second.waited) {
        return; // the unit's release and its 0x10 write name the same label
    }
    if (fresh) {
        order.push_back(key);
        if (order.size() > 4096) {
            marks.erase(order.front());
            order.pop_front();
        }
    }
    Mark& mk = it->second;
    mk = Mark{async, g_draws, g_disp, a_disp, async ? a_since_unit : 0, frame, false};
    if (async) {
        ++batches;
        batch_disp += a_since_unit;
        a_since_unit = 0;
    }
    Log(async ? "async signal" : "graphics signal", label, nullptr);
}
void Work(u32 kind, u64 hash, u32 x, u32 y, u32 z) {
    if (!On()) {
        return;
    }
    std::scoped_lock lock{m};
    if (kind != 0 && log_file != nullptr && log_frames < 3) {
        // note 37: each dispatch of the logged frames, with its shader and groups
        std::fprintf(log_file, "    %s dispatch %llx groups %ux%ux%u", kind == 2 ? "async" : "graphics", static_cast<unsigned long long>(hash), x, y, z);
        std::fputc(10, log_file);
    }
    if (kind == 0) {
        ++g_draws;
    } else if (kind == 1) {
        ++g_disp;
    } else {
        ++a_disp;
        ++a_since_unit;
    }
}
void GraphicsWait(u64 label) {
    if (!On()) {
        return;
    }
    std::scoped_lock lock{m};
    const auto it = marks.find(Key(label));
    if (it == marks.end()) {
        ++g_wait_unknown;
        Log("graphics WAIT unknown", label, nullptr);
        return;
    }
    Waited(it->second.async ? g_wait_async : g_wait_graphics, it->second);
    it->second.waited = true;
    Log("graphics WAIT", label, &it->second);
}
void AsyncWait(u64 label) {
    if (!On()) {
        return;
    }
    std::scoped_lock lock{m};
    const auto it = marks.find(Key(label));
    if (it == marks.end()) {
        ++a_wait_unknown;
        Log("async WAIT unknown", label, nullptr);
        return;
    }
    Waited(it->second.async ? a_wait_async : a_wait_graphics, it->second);
    it->second.waited = true;
    Log("async WAIT", label, &it->second);
}
void Frame() {
    std::scoped_lock lock{m};
    // the async batches of the frame before last that graphics never waited for
    for (auto& [key, mk] : marks) {
        if (mk.async && mk.frame + 2 == frame && !mk.waited) {
            ++batches_never_waited;
            batch_disp_never_waited += mk.batch;
            mk.waited = true;
        }
    }
    if (log_frames < 3 && log_file != nullptr) {
        std::fprintf(log_file, "---- end of frame %llu", static_cast<unsigned long long>(frame));
        std::fputc(10, log_file);
        ++log_frames;
        std::fflush(log_file);
    }
    ++frame;
    ++frames;
}
nlohmann::json Report() {
    std::scoped_lock lock{m};
    const auto agg = [](const Agg& a) {
        const double n = static_cast<double>(std::max<u64>(a.n, 1));
        return nlohmann::json{{"count", a.n},
                              {"mean_graphics_draws_between", static_cast<double>(a.draws) / n},
                              {"mean_graphics_dispatches_between", static_cast<double>(a.disp) / n},
                              {"mean_async_dispatches_between", static_cast<double>(a.a_disp) / n},
                              {"mean_async_batch_dispatches", static_cast<double>(a.batch) / n},
                              {"min_graphics_draws_between", a.n ? a.min_draws : 0},
                              {"waits_with_no_graphics_draw_between", a.zero_draws}};
    };
    return {{"on", On()},
            {"frames", frames},
            {"recorded", {{"graphics_draws", g_draws}, {"graphics_dispatches", g_disp}, {"async_dispatches", a_disp}}},
            {"async_batches", batches},
            {"async_batch_dispatches", batch_disp},
            {"async_batches_never_waited_by_graphics", batches_never_waited},
            {"async_dispatches_in_those", batch_disp_never_waited},
            {"graphics_waits_on_async", agg(g_wait_async)},
            {"graphics_waits_on_graphics", agg(g_wait_graphics)},
            {"graphics_waits_unknown", g_wait_unknown},
            {"async_waits_on_graphics", agg(a_wait_graphics)},
            {"async_waits_on_async", agg(a_wait_async)},
            {"async_waits_unknown", a_wait_unknown}};
}
} // namespace Census

void Fence(const NativeSync::Fence& fence, u32 queue, u64 unit) {
    if (!fence.valid) {
        return;
    }
    if (Census::On() && fence.address != 0) {
        Census::Signal(fence.address, queue != GraphicsQueue);
    }
    Entry entry;
    entry.kind = Entry::Kind::Fence;
    entry.fence = fence;
    entry.unit = unit;
    Push(queue, std::move(entry));
}

void Write(u64 address, const u32* data, u32 bytes, u32 queue, u64 unit) {
    if (address == 0 || data == nullptr || bytes == 0 || bytes > 16) {
        return;
    }
    Entry entry;
    entry.kind = Entry::Kind::Write;
    entry.address = address;
    entry.bytes = bytes;
    entry.unit = unit;
    std::memcpy(entry.data.data(), data, bytes);
    Push(queue, std::move(entry));
}

u64 BeginLabelUnit(u64 address) {
    if (address == 0 || !LabelUnitsOn()) {
        return 0;
    }
    std::scoped_lock lock{label_mutex};
    const u64 unit = ++label_unit_count;
    label_units[address] = unit;
    // the engine set the label to 0x11 at the start of its submit; an older unit's end record may have run since (before this unit was known): its 0x11 stands
    const u64 pending = 0x11;
    std::memcpy(reinterpret_cast<void*>(address), &pending, sizeof(pending));
    c_label_units.fetch_add(1, std::memory_order_relaxed);
    return unit;
}

void Dma(const NativeSync::DmaCopy& dma, u32 queue) {
    Entry entry;
    entry.kind = Entry::Kind::Dma;
    entry.dma = dma;
    Push(queue, std::move(entry));
}

/// Kill-experiment TLG_VKB_LAG=1 (fps bound of a pipelined frame; fences then run one frame before the GPU is done): a submit waits for the previous graphics frame
/// only, not for the work it just submitted
bool LagOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_LAG"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::atomic<u64> lag_previous{0};

/// V4f (TLG_VKB_OVERLAP=1): a submit does not wait for its work. Its records (fences, label writes, copies) go to the completion thread, which runs them, in submission order,
/// once our timeline has passed the submission: a fence still means the GPU work before it is done, and the game's thread records the next frame meanwhile. The flip (its
/// display refresh uses the scheduler of the presenting side) stays on the submitting thread; only its trailing fence goes to the completion thread. At most two graphics
/// frames are on the GPU: a graphics submit waits for the one two frames back (our objects are destroyed three graphics submits after their delete).
bool OverlapOn() {
    // default on since 2026-11-14 (gameplay 30.7 -> 50.1 fps); =0 LEGACY(vkb_overlap): every submit waits for its work and runs its records itself
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_OVERLAP"); return e == nullptr || e[0] != '0'; }();
    return on;
}
struct Pending {
    u64 value;
    std::vector<Entry> entries;
    bool graphics;
    u32 pipe_id;
    std::chrono::steady_clock::time_point made{}; ///< when the engine made the submit (V4g: the latency to its records)
};
std::atomic<u64> c_latency_us[2]{}, c_latency_n[2]{}, c_latency_max_us[2]{}, c_queue_us[2]{};
void NoteQueued(u32 k, std::chrono::steady_clock::time_point made) {
    c_queue_us[k].fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - made).count()), std::memory_order_relaxed);
}
std::mutex pending_mutex;
std::condition_variable pending_changed;
std::deque<Pending> pending;
std::array<u64, 2> frame_values{}; ///< the values of the last two graphics submits
std::atomic<u64> c_overlap_waits{0}, c_overlap_wait_us{0}, c_overlap_most_pending{0}, c_completion_runs{0};

void CompletionThread() {
    Common::SetCurrentThreadName("vkb:Completion");
    for (;;) {
        Pending job;
        {
            std::unique_lock lock{pending_mutex};
            pending_changed.wait(lock, [] { return !pending.empty(); });
            job = std::move(pending.front());
            pending.pop_front();
        }
        Stream::Wait(job.value);
        // sensitivity experiment: TLG_VKB_COMPUTE_DELAY_US=N runs the records of a compute submit N us later (is the compute round trip on the frame's critical path)
        static const u64 compute_delay_us = [] { const char* e = std::getenv("TLG_VKB_COMPUTE_DELAY_US"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull; }();
        if (compute_delay_us != 0 && !job.graphics && !job.entries.empty()) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(compute_delay_us);
            while (std::chrono::steady_clock::now() < until) {
            }
        }
        for (const Entry& entry : job.entries) {
            Run(entry, job.graphics, job.pipe_id);
        }
        if (job.made != std::chrono::steady_clock::time_point{}) {
            const u64 us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - job.made).count());
            const u32 k = job.graphics ? 0 : 1;
            c_latency_us[k].fetch_add(us, std::memory_order_relaxed);
            c_latency_n[k].fetch_add(1, std::memory_order_relaxed);
            if (us > c_latency_max_us[k].load(std::memory_order_relaxed)) {
                c_latency_max_us[k].store(us, std::memory_order_relaxed);
            }
        }
        c_completion_runs.fetch_add(1, std::memory_order_relaxed);
    }
}

void Complete(u64 value, std::vector<Entry>&& entries, bool graphics, u32 pipe_id, std::chrono::steady_clock::time_point made = {}) {
    static std::once_flag started;
    std::call_once(started, [] { std::thread{CompletionThread}.detach(); });
    std::scoped_lock lock{pending_mutex};
    pending.push_back({value, std::move(entries), graphics, pipe_id, made});
    c_overlap_most_pending.store(std::max<u64>(c_overlap_most_pending.load(std::memory_order_relaxed), pending.size()), std::memory_order_relaxed);
    pending_changed.notify_one();
}

/// V4g: our timeline value of each graphics submit (by its count), set when the recording thread submits it
constexpr u64 SubmitValueRing = 256;
std::array<std::atomic<u64>, SubmitValueRing> submit_values{};
std::atomic<u64> submitted_index{0};

std::vector<Entry> Take(u32 queue) {
    std::vector<Entry> taken;
    std::scoped_lock lock{mutex};
    taken.swap(lists[Clamp(queue)]);
    return taken;
}

std::atomic<u64> c_lap_waits{0}, c_lap_wait_us{0}, c_lap_gave_up{0};

void SubmitGraphics() {
    c_submits[0].fetch_add(1, std::memory_order_relaxed);
    // note 44 test (TLG_VKB_LAP_FENCE=<submits>, default 0 = off): the engine's thread waits here until the GPU has finished the graphics submit that many submits back.
    // The game rewrites its per-object constants (lights, scene slots, instance data) in rings of three a few frames later and reads them in place on our GPU; if the
    // GPU runs more than two frames behind (the load), a frame draws with the next lap's data (entry 9 of known-visual-bugs.md: unlit frames, objects out of place)
    static const u32 lap = [] { const char* e = std::getenv("TLG_VKB_LAP_FENCE"); return e != nullptr ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u; }();
    if (lap != 0 && OverlapOn() && Stream::Ready()) {
        const u64 now = c_submits[0].load(std::memory_order_relaxed);
        if (now > lap && !GraphicsSubmitDone(now - lap)) {
            const auto t0 = std::chrono::steady_clock::now();
            while (!GraphicsSubmitDone(now - lap)) {
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
                    c_lap_gave_up.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            c_lap_waits.fetch_add(1, std::memory_order_relaxed);
            c_lap_wait_us.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
        }
    }
    if (CbInPlaceOn() && OverlapOn() && Stream::Ready()) { // note 98: the engine rewrites a pool slot of constants 3 frames later; the GPU must not lag that far
        const u64 now = c_submits[0].load(std::memory_order_relaxed);
        if (now > 2) {
            PoolSubmitCheck(GraphicsSubmitDone(now - 2));
        }
    }
    if (Census::On()) {
        Census::Frame();
    }
    if (OverlapOn() && Stream::Ready()) {
        Stream::DeferUnlocked([](vk::CommandBuffer) { ObjectsTick(); }); // the uploads go to the queue first (the stream's first barrier orders them before its work)
    } else {
        ObjectsTick();
    }
    if (!Stream::Ready()) {
        RunList(GraphicsQueue, 0);
        Publish();
        return;
    }
    // V3b: the frame's recorded work goes to the GPU; its fences, label writes and copies run on this thread once the GPU has finished it (what a fence starts, the display
    // refresh of a flip among it, must stay off other threads: it uses the scheduler the raster thread uses)
    if (OverlapOn()) {
        // V4g: the submit is a job of the recording thread (after every job the frame made); the uploads, the submit and the frame's records go from there
        const u64 index = c_submits[0].load(std::memory_order_relaxed);
        Stream::DeferUnlocked([entries = Take(GraphicsQueue), index, made = std::chrono::steady_clock::now()](vk::CommandBuffer) mutable {
            NoteQueued(0, made);
            Stream::PollQueries(); // V5b: the answers of the queries the GPU has finished
            const u64 value = Stream::Submit();
            Stream::NewFrame();
            submit_values[index % SubmitValueRing].store(value, std::memory_order_release);
            submitted_index.store(index, std::memory_order_release);
            Compute::FrameDone();
            Complete(value, std::move(entries), true, 0, made);
            // two graphics frames on the GPU at most
            const u64 two_back = frame_values[0];
            frame_values[0] = frame_values[1];
            frame_values[1] = value;
            if (two_back != 0) {
                const auto t0 = std::chrono::steady_clock::now();
                Stream::Wait(two_back);
                c_overlap_waits.fetch_add(1, std::memory_order_relaxed);
                c_overlap_wait_us.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()),
                                            std::memory_order_relaxed);
            }
            Publish();
        });
        return;
    }
    if (LagOn()) {
        const u64 value = Stream::Submit();
        Stream::Wait(lag_previous.exchange(value));
    } else {
        Stream::Wait(Stream::Submit());
    }
    Compute::FrameDone();
    RunList(GraphicsQueue, 0);
    Publish();
}

void SubmitCompute(u32 queue, u32 pipe_id) {
    c_submits[1].fetch_add(1, std::memory_order_relaxed);
    if (Stream::Ready() && OverlapOn()) {
        Stream::DeferUnlocked([entries = Take(queue), pipe_id, made = std::chrono::steady_clock::now()](vk::CommandBuffer) mutable {
            NoteQueued(1, made);
            const u64 value = Stream::Submit();
            Compute::FrameDone();
            static const bool early = [] { const char* e = std::getenv("TLG_VKB_EARLY_COMPUTE_FENCE"); return e != nullptr && e[0] == '1'; }(); // kill-experiment only
            if (early) {
                for (const Entry& entry : entries) {
                    Run(entry, false, pipe_id);
                }
                return;
            }
            Complete(value, std::move(entries), false, pipe_id, made);
        });
        return;
    }
    if (Stream::Ready()) {
        // V3a part 2: the async queue's work is in the frame stream (in call order with the device context's); its fences run once the GPU has done it
        if (LagOn()) {
            (void)Stream::Submit();
            Stream::Wait(lag_previous.load());
        } else {
            Stream::Wait(Stream::Submit());
        }
        Compute::FrameDone();
    }
    RunList(queue, pipe_id);
}

namespace {
void FlipNow(const NativeFlip::Request& request);
}

void Flip(const NativeFlip::Request& request) {
    Stream::Drain(); // V4g: the frame is submitted (by the recording thread) before the presenter takes its picture
    FlipNow(request);
}

namespace {
void FlipNow(const NativeFlip::Request& request) {
    // the four things of the typed flip tail, in the order the command processor did them (native_executor_direct.cpp, EntryKind::Flip)
    const u32 one = 1;
    NativeDisplay::Lock(request.buffer_index);
    if (request.display_label != 0) {
        std::memcpy(reinterpret_cast<void*>(request.display_label), &one, sizeof(one));
    }
    NativeFrameEvents::FlipReady();
    if (request.has_label && request.label_address != 0) {
        std::memcpy(reinterpret_cast<void*>(request.label_address), &request.label_value, sizeof(u32));
    }
    if (request.has_fence) {
        Entry entry;
        entry.kind = Entry::Kind::Fence;
        entry.fence = request.fence;
        if (OverlapOn() && Stream::Ready()) {
            std::vector<Entry> one_fence{entry};
            Complete(Stream::LastValue(), std::move(one_fence), true, 0); // after the frame's work, in order with its other fences
        } else {
            Run(entry, true, 0);
        }
    }
    c_flips.fetch_add(1, std::memory_order_relaxed);
}
} // namespace

u64 GraphicsSubmits() {
    return c_submits[0].load(std::memory_order_relaxed);
}

u64 PS4_SYSV_ABI OcclusionBeginHook(u64 ctx, u64 address, u64, u64, u64, u64, u64) {
    if (!On() || !Stream::Ready() || address == 0) {
        return 0;
    }
    static const u32 pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    if (!OverlapOn()) {
        // open item D4-09: the answers are polled only on the overlap path (native_vkb.cpp, Submit); with TLG_VKB_OVERLAP=0 a block keeps "visible"
        Core::RareEvents::OpenItem("D4-09", fmt::format("an occlusion query (block {:#x}) with TLG_VKB_OVERLAP=0: no measured answer", address));
    }
    Stream::FirstAnswer(address, pairs);
    Stream::DeferQuery(1, [address](vk::CommandBuffer cmd) { Stream::QueryBegin(cmd, address); });
    (void)ctx;
    return 0; // the engine's function runs (its packets go unread)
}

u64 PS4_SYSV_ABI OcclusionEndHook(u64 ctx, u64 address, u64, u64, u64, u64, u64) {
    if (!On() || !Stream::Ready() || address == 0) {
        return 0;
    }
    Stream::DeferQuery(-1, [address](vk::CommandBuffer cmd) { Stream::QueryEnd(cmd, address); });
    (void)ctx;
    return 0;
}

bool GraphicsSubmitDone(u64 index) {
    if (index == 0) {
        return true;
    }
    if (!OverlapOn() || !Stream::Ready()) {
        return true; // every submit waited for its work
    }
    const u64 last = submitted_index.load(std::memory_order_acquire);
    if (index > last) {
        return false;
    }
    if (last - index >= SubmitValueRing) {
        return true; // long done (two frames on the GPU at most)
    }
    return Stream::CompletedValue() >= submit_values[index % SubmitValueRing].load(std::memory_order_acquire);
}

u64 GraphicsSubmitted() {
    if (!OverlapOn() || !Stream::Ready()) {
        const u64 n = c_submits[0].load(std::memory_order_acquire);
        return n != 0 ? n - 1 : 0; // the newest may still be waiting for its work on the engine thread
    }
    return submitted_index.load(std::memory_order_acquire);
}

u64 GraphicsSubmitsDone() {
    if (!OverlapOn() || !Stream::Ready()) {
        return GraphicsSubmitted(); // every submit before the newest waited for its work
    }
    const u64 last = submitted_index.load(std::memory_order_acquire);
    const u64 completed = Stream::CompletedValue();
    for (u64 i = last; i != 0 && last - i < SubmitValueRing; --i) {
        if (completed >= submit_values[i % SubmitValueRing].load(std::memory_order_acquire)) {
            return i; // the queue finishes in order: everything up to i is done
        }
    }
    return last >= SubmitValueRing ? last - SubmitValueRing + 1 : 0;
}

// note 44, debugger mode (Shift+F5 recording): every graphics submit snapshots all of the backend's counters (graphics, compute with the stream, upload, objects);
// the freeze writes the per-frame changes into the frame walk folder (vkb_frames.json), on the thumbnails' clock, so a bad frame can be read against its neighbours
struct FrameSnap {
    u64 ns{};
    u64 submit{};
    std::vector<std::pair<u32, double>> values;
    std::vector<std::string> notes; ///< what happened in this submit beyond the counters (new pipeline variants)
};
std::vector<std::string> pending_notes;
std::mutex snap_mutex;
std::deque<FrameSnap> snaps;
std::unordered_map<std::string, u32> snap_keys;
std::vector<std::string> snap_names;
constexpr size_t SnapFrames = 1200;

void Flatten(const nlohmann::json& j, const std::string& prefix, std::vector<std::pair<u32, double>>& out) {
    const auto add = [&](const std::string& key, double v) {
        auto [it, fresh] = snap_keys.try_emplace(key, static_cast<u32>(snap_names.size()));
        if (fresh) {
            snap_names.push_back(key);
        }
        out.push_back({it->second, v});
    };
    if (j.is_object()) {
        for (auto it = j.begin(); it != j.end(); ++it) {
            Flatten(it.value(), prefix.empty() ? it.key() : prefix + "." + it.key(), out);
        }
    } else if (j.is_array()) {
        if (j.size() <= 16) {
            for (size_t i = 0; i < j.size(); ++i) {
                if (j[i].is_number() || j[i].is_boolean()) {
                    add(prefix + "[" + std::to_string(i) + "]", j[i].is_boolean() ? (j[i].get<bool>() ? 1.0 : 0.0) : j[i].get<double>());
                }
            }
        }
    } else if (j.is_number()) {
        add(prefix, j.get<double>());
    } else if (j.is_boolean()) {
        add(prefix, j.get<bool>() ? 1.0 : 0.0);
    }
}

void SnapshotFrame() {
    if (Core::FrameWalk::Current() != Core::FrameWalk::State::Recording) {
        return;
    }
    static std::atomic<u64> submit{0};
    const nlohmann::json j = {{"graphics", Graphics::Report()}, {"compute", Compute::Report()}, {"upload", Upload::Report()}, {"objects", ObjectsReport()}};
    FrameSnap f{Core::FrameTimeline::Now(), submit.fetch_add(1, std::memory_order_relaxed), {}};
    std::scoped_lock lock{snap_mutex};
    Flatten(j, "", f.values);
    f.notes = std::move(pending_notes);
    pending_notes.clear();
    snaps.push_back(std::move(f));
    while (snaps.size() > SnapFrames) {
        snaps.pop_front();
    }
}

void Publish() {
    SnapshotFrame();
    static std::atomic<u64> calls{0};
    if ((calls.fetch_add(1, std::memory_order_relaxed) & 63) != 0) {
        return;
    }
    DebugServer::Publish("vkb_sync", {{"lap_fence_waits", c_lap_waits.load()}, {"lap_fence_wait_ms", c_lap_wait_us.load() / 1000}, {"lap_fence_gave_up", c_lap_gave_up.load()},
                                      {"graphics_fences", c_fences[0].load()},
                                      {"compute_fences", c_fences[1].load()},
                                      {"graphics_interrupts", c_interrupts[0].load()},
                                      {"compute_interrupts", c_interrupts[1].load()},
                                      {"label_writes", c_writes.load()},
                                      {"label_units", {{"submitted", c_label_units.load()}, {"older_unit_records_dropped", c_label_stale_dropped.load()}}},
                                      {"dma_copies", c_dma.load()},
                                      {"dma_bytes", c_dma_bytes.load()},
                                      {"dma_ignored_gds", c_dma_ignored.load()},
                                      {"flips", c_flips.load()},
                                      {"graphics_submits", c_submits[0].load()},
                                      {"compute_submits", c_submits[1].load()},
                                      {"records_of_an_unknown_queue", c_bad_queue.load()},
                                      {"most_pending_in_one_queue", c_most_pending.load()},
                                      {"overlap", {{"on", OverlapOn()}, {"completion_runs", c_completion_runs.load()}, {"most_pending", c_overlap_most_pending.load()},
                                                   {"two_frames_back_waits", c_overlap_waits.load()},
                                                   {"submit_to_records_ms_mean_graphics_compute",
                                                    {c_latency_n[0].load() ? static_cast<double>(c_latency_us[0].load()) / 1e3 / static_cast<double>(c_latency_n[0].load()) : 0.0,
                                                     c_latency_n[1].load() ? static_cast<double>(c_latency_us[1].load()) / 1e3 / static_cast<double>(c_latency_n[1].load()) : 0.0}},
                                                   {"submit_to_records_ms_max_graphics_compute", {c_latency_max_us[0].load() / 1000, c_latency_max_us[1].load() / 1000}},
                                                   {"compute_submits_with_records", c_latency_n[1].load()},
                                                   {"submit_waiting_for_the_recording_thread_ms_mean_graphics_compute",
                                                    {c_latency_n[0].load() ? static_cast<double>(c_queue_us[0].load()) / 1e3 / static_cast<double>(c_latency_n[0].load()) : 0.0,
                                                     c_latency_n[1].load() ? static_cast<double>(c_queue_us[1].load()) / 1e3 / static_cast<double>(c_latency_n[1].load()) : 0.0}},
                                                   {"two_frames_back_wait_ms_mean", c_overlap_waits.load() ? static_cast<double>(c_overlap_wait_us.load()) / 1e3 / static_cast<double>(c_overlap_waits.load()) : 0.0}}}});
}

} // namespace Core::Vkb

void Core::Vkb::FrameNote(const std::string& text) {
    if (Core::FrameWalk::Current() != Core::FrameWalk::State::Recording) {
        return;
    }
    std::scoped_lock lock{snap_mutex};
    if (pending_notes.size() < 64) {
        pending_notes.push_back(text);
    }
}

nlohmann::json Core::Vkb::FrameSnapshots() {
    std::scoped_lock lock{snap_mutex};
    nlohmann::json frames = nlohmann::json::array();
    std::vector<double> previous(snap_names.size(), 0.0);
    std::vector<bool> seen(snap_names.size(), false);
    for (const auto& f : snaps) {
        nlohmann::json d = nlohmann::json::object();
        for (const auto& [k, v] : f.values) {
            if (!seen[k] || v != previous[k]) {
                d[snap_names[k]] = seen[k] ? v - previous[k] : v;
            }
            previous[k] = v, seen[k] = true;
        }
        nlohmann::json entry = {{"ns", f.ns}, {"submit", f.submit}, {"changed", std::move(d)}};
        if (!f.notes.empty()) {
            entry["notes"] = f.notes;
        }
        frames.push_back(std::move(entry));
    }
    return {{"note", "note 44: one entry per graphics submit while Shift+F5 recorded; changed = the counters' change since the entry before (the first entry: their values); "
                     "ns is the clock of thumbs.bin (FrameTimeline::Now). Read with documents/tlg/tools/vkb_frames.py"},
            {"keys", snap_names.size()},
            {"frames", std::move(frames)}};
}
