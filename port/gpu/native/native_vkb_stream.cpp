// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_vkb_objects.h"
#include "native/native_debug_server.h" // RareEvents::OpenItem
#include "native/native_flight.h"
#include "native/native_timeline.h"
#include "native/native_framewalk.h"
#include "native/native_vkb_stream.h"
#include "native/native_vkb.h"

#include <tuple>
#include <map>
#include <algorithm>
#include <array>
#include <atomic>
#include <immintrin.h>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_instance.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_scheduler.h"

#include <vk_mem_alloc.h>
#ifdef _WIN32
#include <windows.h> // QueryPerformanceFrequency: the timeline capture's GPU clock calibration
#endif

namespace Core::Vkb::Stream {
namespace {
std::atomic<bool> frame_capture{false};    ///< note 91: draws made now belong to the frame capture
std::atomic<u64> capture_jobs_left{0};      ///< note 91: captured draws not yet recorded
std::atomic<u32> capture_passes{0};         ///< note 91: passes of the capture
std::mutex capture_mutex;                   ///< note 91: the capture's tables
nlohmann::json capture_rows = nlohmann::json::array();
std::map<std::tuple<VkBuffer, u64, u64>, u32> capture_buffer_index;
std::unordered_map<u64, double> capture_draw_ms; ///< draw id -> GPU ms
u64 capture_buffer_bytes = 0;
constexpr u64 CaptureBufferLimit = 2048ull << 20, CaptureBufferMax = 128ull << 20;
u64 SteadyNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
} // namespace
bool StampsForcedFull();

namespace {

const VkbVulkan::Instance* instance = nullptr;
std::mutex mutex; ///< the command buffer, the pool and the ring
vk::Semaphore timeline{};
std::atomic<u64> recording_generation{0}; ///< note 60: one per command buffer begun (a recycled handle is a new recording)
std::atomic<u64> next_value{0}; ///< R4a: the last timeline value handed out (a submission closed); last_value: the last one submitted
std::atomic<u64> last_value{0};

struct OwnedBuffer {
    VkBuffer buffer{};
    VmaAllocation allocation{};
    void* mapped{};
};
OwnedBuffer ring_buffer, gds_buffer, null_buffer, scratch_buffer;
// V4h staging: a device ring the submissions' read-only in-place ranges are copied into
OwnedBuffer stage_buffer;
constexpr u64 StageBytes = 512ull << 20, StageMaxRange = 8ull << 20;
u64 stage_head = 0;
struct StageKey {
    u64 handle, offset, size;
    bool operator==(const StageKey&) const = default;
};
struct StageKeyHash {
    size_t operator()(const StageKey& k) const noexcept {
        return std::hash<u64>{}(k.handle * 0x9E3779B97F4A7C15ull ^ k.offset * 0xC2B2AE3D27D4EB4Full ^ k.size);
    }
};
struct StageRegion {
    u64 begin, end, value;
};
std::deque<StageRegion> stage_in_flight; ///< R4a: registered at the copy, in ring order, with the value of the submission that carries it
std::mutex stage_mutex; ///< R4a: the staging ring, its regions and its stamps (several recording threads)
std::atomic<u64> c_stage_ring{0}, c_stage_ring_bytes{0}; ///< R3 census: copies whose source is the frame ring
std::atomic<u64> c_stage_own_overlap{0};
std::atomic<u64> c_staged{0}, c_staged_bytes{0}, c_stage_reused{0}, c_stage_written{0}, c_stage_too_big{0}, c_stage_waits{0};
std::atomic<u64> c_stage_covered{0}, c_stage_merged{0}, c_stage_merge_too_big{0}; ///< issue 111: reads inside a staged region, copies that took in overlapping regions, unions over the size limit
// V5b occlusion queries
constexpr u32 QuerySlots = 2048;
vk::QueryPool occlusion_pool{};
struct QuerySlot {
    u64 address{};
    u64 value{}; ///< the timeline value of the submission that ended it
    bool pending{};
    bool cut{};     ///< D4-10: ended at a submission's end while the engine's query goes on (its count is part of the interval's)
    bool chained{}; ///< D4-10: begun at the next submission's start to carry an interval on (its answer adds the earlier parts)
};
/// D4-10 (open-items.md): the engine's occlusion query counts every draw between its begin and its end, whatever is submitted in between (on PS4 the Z pass
/// counters are dumped at both ends, across submits). A Vulkan query lives in one command buffer, so an interval cut by a submission goes on in a new query at
/// the next submission's start; the block is "seen" when any part was. LEGACY(query_cut): TLG_LEGACY_QUERY_CUT=1 ends the query at the submit (a part count).
bool QueryCarryOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_LEGACY_QUERY_CUT"); return !(e != nullptr && e[0] == '1'); }();
    return on;
}
std::unordered_map<u64, bool> query_chain_seen; ///< D4-10: per block, "seen" so far over the cut parts of its open interval (blocks_mutex)
std::atomic<u64> c_q_carried{0}, c_q_chained_answers{0};
std::array<std::atomic<u64>, 3> c_q_carried_by{}; ///< D4-10: submissions closed inside a query, by who asked for the submit (other, graphics, async compute)
std::array<QuerySlot, QuerySlots> query_slots{};
u32 query_next = 0;
struct QueryBlock {
    u64 base{};
    u32 pairs{8};
    bool answered{};
};
std::mutex blocks_mutex;
std::unordered_map<u64, QueryBlock> query_blocks;
std::atomic<u64> c_q_begun{0}, c_q_ended{0}, c_q_seen{0}, c_q_hidden{0}, c_q_nested{0}, c_q_dropped{0}, c_q_cut_by_submit{0}, c_q_first{0}, c_q_unmatched_end{0};
constexpr u64 QueryValidBit = 1ull << 63, PlainlyVisible = 0x2FFFFFFull;
void WriteAnswerLocked(u64 address, bool seen) {
    QueryBlock& block = query_blocks[address];
    block.base += 2 * PlainlyVisible;
    auto* words = reinterpret_cast<u64*>(address);
    for (u32 pair = 0; pair < block.pairs; ++pair) {
        words[2 * pair + 1] = ((block.base + (seen ? PlainlyVisible : 0)) & ~QueryValidBit) | QueryValidBit;
    }
    for (u32 pair = 0; pair < block.pairs; ++pair) {
        words[2 * pair] = (block.base & ~QueryValidBit) | QueryValidBit;
    }
}
bool StageOn() {
    // default on since 2026-11-14 (gameplay 58.6 -> 84.2 fps); =0 LEGACY(vkb_stage): read in place over the bus
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_STAGE"); return e == nullptr || e[0] != '0'; }();
    return on;
}
/// Issue 111: a staged read reuses this submission's copy of the memory around it, not only a copy of the very same range. All copies of a submission run in front
/// of it, so a covering copy holds the same bytes as a copy of its own. =0 LEGACY(vkb_stage_merge): a copy is reused only for the same offset and size.
bool StageMergeOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_STAGE_MERGE"); return e == nullptr || e[0] != '0'; }();
    return on;
}
bool HostScratchOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_HOST_SCRATCH"); return e != nullptr && e[0] == '1'; }();
    return on;
}
} // namespace
bool HostCacheOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_HOST_CACHE"); return e != nullptr && e[0] == '1'; }();
    return on;
}
namespace {
OwnedBuffer cache_arena;
bool NoPassBarrier() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_NO_PASS_BARRIER"); return e != nullptr && e[0] == '1'; }();
    return on;
}
constexpr u64 RingBytes = 64ull << 20;
u64 ring_head = 0;
u64 frame_ring_bytes = 0;
/// V4d: what the commands since the last barrier wrote and read, by handle (ranges; more than 32 ranges of one handle merge into one)
struct Range {
    u64 b, e;
    u8 cls; ///< V4h: the kinds of work (Agent bits) that touched it
};
using RangeSet = std::unordered_map<u64, std::vector<Range>>;
RangeSet frame_written; ///< V4i: engine memory our GPU work wrote in place this frame
std::mutex frame_written_mutex; ///< R4a: noted by the recording threads
std::mutex ring_mutex; ///< V4g: the host ring (filled where calls are resolved, reset at the submit)
thread_local std::vector<Transition> pending_transitions;
thread_local Splice* t_sink = nullptr; ///< P1: this thread's jobs go into a placeholder
std::atomic<u64> c_splices{0}, c_splice_jobs{0}, c_splice_waits{0}, c_splice_wait_us{0};
/// note 60: the hand-over from the engine's threads to the recording thread is a lock-free multi-producer queue (Vyukov's intrusive MPSC: a producer links its job
/// with one atomic exchange, call order kept); the recorder is woken only when it sleeps, and the flip's Drain waits on an idle counter. Before: a mutex, a deque and
/// two condition variables, locked and notified per job by both threads (note 59: about 9 % of the render thread in lock waits and SRW locks).
struct StubJob final : Job {
    void Run(vk::CommandBuffer) override {}
};
StubJob queue_stub;
std::atomic<Job*> queue_head{&queue_stub}; ///< producers: the newest job
Job* queue_tail = &queue_stub;              ///< the recorder only: the oldest
std::atomic<u64> queue_size{0};
std::atomic<bool> recorder_sleeping{false}, job_running{false};
std::atomic<u64> recorder_idle_gen{0};
void QueuePush(Job* j) {
    j->queue_next.store(nullptr, std::memory_order_relaxed);
    Job* prev = queue_head.exchange(j, std::memory_order_acq_rel);
    prev->queue_next.store(j, std::memory_order_release);
}
/// The recorder: the oldest job, or null (empty, or a producer between its exchange and its link: the caller tries again)
Job* QueuePop() {
    Job* tail = queue_tail;
    Job* next = tail->queue_next.load(std::memory_order_acquire);
    if (tail == &queue_stub) {
        if (next == nullptr) {
            return nullptr;
        }
        queue_tail = next;
        tail = next;
        next = next->queue_next.load(std::memory_order_acquire);
    }
    if (next != nullptr) {
        queue_tail = next;
        return tail;
    }
    if (tail != queue_head.load(std::memory_order_acquire)) {
        return nullptr; // a producer is linking
    }
    QueuePush(&queue_stub);
    next = tail->queue_next.load(std::memory_order_acquire);
    if (next != nullptr) {
        queue_tail = next;
        return tail;
    }
    return nullptr;
}
thread_local bool on_recorder = false;
thread_local Job* t_preparing = nullptr; ///< issue 111: the job whose Prepare runs now on the resolver (CopyInStream goes with it)
thread_local bool t_in_locked_run = false; ///< issue 111: a locked job's Run on this recording thread (its lock held, its command buffer begun)
std::atomic<u64> c_copies_first{0}, c_copies_first_mb{0}, c_copies_first_in_run{0}, c_copies_first_own_job{0};
std::atomic<u64> c_jobs{0}, c_job_waits_full{0}, c_drains{0}, c_drain_us{0}, c_jobs_most{0};
/// note 45: the recording thread's time: running jobs by kind (unlocked jobs = submits under "submit"), asleep on an empty queue, and the producers' enqueue
std::atomic<u64> c_rec_run_ns[static_cast<u32>(Work::Count) + 1]{}, c_rec_run_n[static_cast<u32>(Work::Count) + 1]{}, c_rec_sleep_ns{0}, c_rec_sleeps{0}, c_enqueue_ns{0};
std::atomic<u64> c_rec_spin_hits{0}; ///< note 82: jobs that arrived while the resolver spun (no sleep, no wake-up)
bool RecTimesOn() { static const bool on = [] { const char* e = std::getenv("TLG_VKB_TIMES"); return e != nullptr && e[0] == '1'; }(); return on; } ///< TLG_VKB_TIMES=1 only
u64 RecNs() { return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); }
constexpr size_t MaxJobs = 16384;
std::atomic<u64> c_hazard_checks{0}, c_hazard_barriers{0}, c_begin_barriers{0};
/// what made a barrier: [the command's kind][0 read after write, 1 write after write, 2 write after read][0 GDS, 1 null buffer, 2 other buffer, 3 image]
std::atomic<u64> c_hazard_cause[6][3][4];
bool LegacyBarriers() {
    static const bool on = [] { const char* e = std::getenv("TLG_LEGACY_VKB_BARRIERS"); return e != nullptr && e[0] == '1'; }(); // LEGACY(vkb_barriers)
    return on;
}
/// The kinds of work that touched a range overlapping `u` (0: none)
u8 Touches(const RangeSet& set, const Use& u) {
    const auto it = set.find(u.handle);
    if (it == set.end()) {
        return 0;
    }
    u8 cls = 0;
    for (const Range& r : it->second) {
        if (u.begin < r.e && r.b < u.end) {
            cls |= r.cls;
        }
    }
    return cls;
}
void Note(RangeSet& set, const Use& u, u8 cls) {
    auto& ranges = set[u.handle];
    for (Range& r : ranges) {
        if (r.b <= u.begin && u.end <= r.e) {
            r.cls |= cls;
            return;
        }
    }
    if (ranges.size() >= 32) {
        u64 b = u.begin, e = u.end;
        u8 all = cls;
        for (const Range& r : ranges) {
            b = std::min(b, r.b), e = std::max(e, r.e), all |= r.cls;
        }
        ranges.assign(1, Range{b, e, all});
        return;
    }
    ranges.push_back(Range{u.begin, u.end, cls});
}
/// After a barrier that waited for the kinds `covered`: what only they touched is synchronized
void Release(RangeSet& set, u8 covered) {
    for (auto it = set.begin(); it != set.end();) {
        auto& ranges = it->second;
        std::erase_if(ranges, [covered](Range& r) {
            r.cls &= static_cast<u8>(~covered);
            return r.cls == 0;
        });
        it = ranges.empty() ? set.erase(it) : std::next(it);
    }
}
vk::PipelineStageFlags2 StagesOf(u8 cls) {
    vk::PipelineStageFlags2 stages{};
    if (cls & 1) {
        stages |= vk::PipelineStageFlagBits2::eAllGraphics;
    }
    if (cls & 2) {
        stages |= vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eDrawIndirect;
    }
    if (cls & 4) {
        stages |= vk::PipelineStageFlagBits2::eAllTransfer;
    }
    return stages;
}
bool FullBarriers() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_FULL_BARRIERS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::atomic<u64> c_barrier_src[8]{}; ///< barriers by the kinds they waited for
std::atomic<u64> c_passes{0}; ///< used since the last submit (the frame before it has finished: SubmitGraphics waits for it)
VkImage null_image{};
VmaAllocation null_allocation{};
vk::ImageView null_view{};
constexpr u32 MaxStamps = 4096, StampPools = 48; // V4g: split submits keep many in flight
std::array<vk::QueryPool, StampPools> stamp_pools{};
std::array<bool, StampPools> pool_busy{}; ///< submitted and not yet read
/// R5 TLG_VKB_PASS_STATS=1: one pipeline statistics query around each timed render pass (vertex shader invocations, clipping primitives, fragment shader invocations)
constexpr u32 MaxPassStats = 1024;
std::array<vk::QueryPool, StampPools> stat_pools{};
bool PassStatsOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_PASS_STATS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::atomic<u64> generations{0}; ///< R4a: command buffers begun (each recording thread takes the next one: its stamp pool)
std::mutex stamp_mutex; ///< R4a: the stamp pools in use and the stamped submissions
struct Segment {
    u32 stamp;
    Work work;
};
/// note 91: debugger mode: a timestamp after every draw; the draw's GPU time runs from the stamp before it in the same command buffer
struct DrawStamp {
    u32 stamp, prev;
    u64 id, vs, ps;
    u32 count, instances;
};
/// R5 meter: one render pass of the command buffer, between its two timestamps (begin before beginRendering, end after endRendering)
struct PassStamp {
    u32 begin{~0u}, end{~0u};
    u32 width{}, height{};
    u32 draws{};
    u8 kind{}; ///< 0 depth only, 1 color (1 to 3 targets), 2 g-buffer (4 or more)
    u32 flags{}, format{};
    u64 ps{};
    u32 colors{};
    vk::Viewport viewport{};
    vk::Rect2D scissor{};
    u32 stat{~0u}; ///< R5 TLG_VKB_PASS_STATS=1: its pipeline statistics query in the pool of its command buffer
};
/// R4a (note 65): what one recording thread records into: its command pool, the command buffer of the submission it records, that buffer's pass, barrier tracking,
/// stamps, staging copies and open query. The single recorder (and every thread before the recorder starts) shares one; each R4a recording thread has its own.
struct Rec {
    vk::CommandPool pool{};
    vk::CommandBuffer current{};
    bool current_has_work = false;
    std::deque<std::pair<u64, vk::CommandBuffer>> in_flight;
    vk::CommandBuffer stage_cmd{}; ///< the copies for `current`, submitted in front of it
    std::unordered_map<StageKey, u64, StageKeyHash> stage_map; ///< this submission's copies
    /// issue 111 (TLG_VKB_STAGE_MERGE): this submission's copies as regions that do not overlap: (buffer handle, start in it) -> (end, offset in the ring)
    std::map<std::pair<u64, u64>, std::pair<u64, u64>> stage_regions;
    u32 stage_stamp_open = ~0u;
    bool query_open = false;
    u32 query_open_index = 0;
    u64 query_open_address = 0;
    bool pass_open = false;
    Pass open_pass{};
    RangeSet written_since, read_since; ///< V4d: what the commands since the last barrier wrote and read
    bool quiet_chain = false;           ///< every write noted since the last full release came from a QuietChain shader (lantern waits)
    std::vector<std::pair<u32, u32>> chain_records; ///< exact rule: the record ranges the chain's dispatches since the last full release wrote
    std::vector<u32> chain_slots;                  ///< exact rule: the particle slots (not 0) they wrote
    RangeSet cmd_written;               ///< V4h: what this command buffer's GPU work wrote (not cleared by barriers)
    u32 stat_count = 0;
    bool stat_open = false;
    u32 stamp_count = 0;
    bool stamping = false; ///< the current command buffer has a free pool
    bool lean = true;      ///< the current command buffer's stamps are lean (decided at its begin: full with TLG_VKB_STAMPS=1 or in debugger mode, note 87)
    u64 generation = 0;
    std::vector<Segment> segments; ///< the current command buffer's: where each run of one kind of work begins (its timestamp)
    Work current_work = Work::Count;
    std::vector<PassStamp> pass_stamps; ///< the current command buffer's
    std::vector<DrawStamp> draw_stamps; ///< note 91: the current command buffer's (debugger mode)
    u32 last_stamp = ~0u;               ///< note 91: the newest stamp written in the current command buffer
    std::vector<vk::Image> pass_images; ///< note 91: the open pass's attachments (colours, then depth)
    u32 capture_pass = ~0u;             ///< note 91: the open pass's index in the frame capture (~0: no captured draw in it yet)
    std::vector<std::tuple<vk::Buffer, u64, u64, u32>> capture_buffers; ///< note 91: buffer ranges of captured draws, copied at the pass end (with their file index)
    u32 jobs_in_cmd = 0;
    std::unordered_set<u64> census_touched; ///< census: handles the current command buffer's commands touched
    std::vector<std::pair<u64, u64>> touched; ///< U1: (handle, generation) of what the current command buffer's commands touched (open addressing, reset by generation)
    u64 census_generation = 0;
    u64 value = 0; ///< R4a: the timeline value of the submission this thread records (0: the shared recorder, which takes the next value at its submit)
};
Rec shared_rec;
thread_local Rec* t_rec = &shared_rec;
/// the timeline value the command buffer being recorded will signal
u64 RecValue() {
    return t_rec->value != 0 ? t_rec->value : next_value.load(std::memory_order_relaxed) + 1;
}
void FullBarrierLocked(vk::CommandBuffer cmd) {
    const vk::MemoryBarrier2 barrier{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                     .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands, .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    t_rec->written_since.clear();
    t_rec->read_since.clear();
    t_rec->quiet_chain = true; // nothing is outstanding
    t_rec->chain_records.clear();
    t_rec->chain_slots.clear();
}
struct PassKey {
    u8 kind;
    u32 width, height;
    bool operator==(const PassKey&) const = default;
};
struct PassKeyHash {
    size_t operator()(const PassKey& k) const noexcept {
        return (static_cast<size_t>(k.kind) << 40) ^ (static_cast<size_t>(k.width) << 20) ^ k.height;
    }
};
struct PassTotal {
    u64 ns{}, passes{}, draws{};
    u64 vs{}, prims{}, fs{}, counted{}; ///< TLG_VKB_PASS_STATS=1: vertex shader invocations, primitives that reached the rasterizer, fragment shader invocations; passes counted
};
std::unordered_map<PassKey, PassTotal, PassKeyHash> pass_totals; ///< finished passes by kind and size (under the stream lock)
struct Submitted {
    u64 value, generation;
    u32 pool, count;
    std::vector<Segment> segments; ///< closed by the stamp at index count - 1 (written at the submit)
    std::vector<PassStamp> passes;
    bool lean = true;
    std::vector<DrawStamp> draws;
};
constexpr u32 WorkKinds = static_cast<u32>(Work::Count);
constexpr const char* WorkNames[WorkKinds] = {"draws", "dispatches", "copies_texture", "clears", "other", "uploads_first_bind", "copies_buffer", "copies_gds", "fills",
                                              "uploads_initial", "uploads_unmap"};
std::array<std::atomic<u64>, WorkKinds> work_bytes{}, work_jobs{};
bool IsCopyKind(Work w) {
    return w == Work::Copy || w == Work::CopyBuffer || w == Work::CopyGds || w == Work::Fill;
}
bool IsUploadKind(Work w) {
    return w == Work::Upload || w == Work::UploadInitial || w == Work::UploadCpuWritten;
}
constexpr u64 WorkWindow = 500; ///< finished submissions per window of the per-kind report
std::array<u64, WorkKinds> work_ns{}, work_window_ns{}, work_last_window_ns{};
u64 work_window_first_ns = 0, work_window_last_ns = 0, work_last_window_span_ns = 0; ///< GPU clock: the window's first stamp and last end
u64 work_total_ns = 0, work_window_total_ns = 0, work_last_window_total_ns = 0, work_submits = 0, work_window_submits = 0, work_window_max_ns = 0, work_last_window_max_ns = 0;
std::deque<Submitted> stamped;
double stamp_period_ns = 1.0;
// V4l meter: the staging command buffer's own GPU time (it is submitted in front of the stamped one, so the per-kind meter never saw it)
constexpr u32 StageStampPairs = 512;
vk::QueryPool stage_stamp_pool{};
u32 stage_stamp_next = 0;
struct StageStamped {
    u64 value;
    u32 index;
};
std::deque<StageStamped> stage_stamped;
u64 stage_gpu_ns_total = 0, stage_gpu_timed = 0;

std::atomic<u64> c_submits{0}, c_empty_submits{0}, c_waits{0}, c_wait_us_total{0}, c_ring_bytes{0}, c_ring_wraps{0}, c_ring_overrun{0}, c_wait_ms_max{0};

bool MakeBuffer(u64 size, VkBufferUsageFlags usage, bool host, OwnedBuffer& out) {
    const VkBufferCreateInfo ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo alloc{};
    alloc.usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (host) {
        alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance->GetAllocator(), &ci, &alloc, &out.buffer, &out.allocation, &info) != VK_SUCCESS) {
        return false;
    }
    out.mapped = info.pMappedData;
    return true;
}

u64 SubmitLocked();
void MakePool() {
    if (!t_rec->pool) {
        t_rec->pool = instance->GetDevice().createCommandPool({.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = instance->GetGraphicsQueueFamilyIndex()}).value;
    }
}
vk::CommandBuffer Begin() {
    if (!t_rec->current) {
        MakePool();
        const vk::CommandBufferAllocateInfo ai{.commandPool = t_rec->pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
        t_rec->current = instance->GetDevice().allocateCommandBuffers(ai).value[0];
        (void)t_rec->current.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        recording_generation.fetch_add(1, std::memory_order_relaxed); // note 60: the draw state cache starts over
        t_rec->generation = generations.fetch_add(1, std::memory_order_relaxed) + 1;
        const u32 pool = static_cast<u32>(t_rec->generation % StampPools);
        static const bool no_stamps = [] { const char* e = std::getenv("TLG_VKB_NO_STAMPS"); return e != nullptr && e[0] == '1'; }(); // kill-experiment: no GPU timestamps
        {
            std::scoped_lock stamp_lock{stamp_mutex};
            t_rec->stamping = !no_stamps && stamp_pools[pool] && !pool_busy[pool];
            if (t_rec->stamping) {
                pool_busy[pool] = true; // R4a: taken now (a second recording thread must not begin in it); its stamps are read after the submission
            }
        }
        if (t_rec->stamping) {
            t_rec->current.resetQueryPool(stamp_pools[pool], 0, MaxStamps);
            if (stat_pools[pool]) {
                t_rec->current.resetQueryPool(stat_pools[pool], 0, MaxPassStats);
            }
        }
        t_rec->lean = !StampsForcedFull() && !Core::FrameWalk::DebuggerMode(); // note 87: debugger mode records full GPU stamps (one relaxed load otherwise)
        t_rec->stamp_count = 0;
        t_rec->stat_count = 0;
        t_rec->stat_open = false;
        t_rec->segments.clear();
        t_rec->pass_stamps.clear();
        t_rec->draw_stamps.clear();
        t_rec->last_stamp = ~0u;
        t_rec->current_work = Work::Count;
        t_rec->cmd_written.clear();
        // V4d: what earlier submissions (the upload batch, the frame before) wrote is visible to this one
        FullBarrierLocked(t_rec->current);
        c_begin_barriers.fetch_add(1, std::memory_order_relaxed);
    }
    return t_rec->current;
}

u32 StampLocked(vk::CommandBuffer cmd);

void MarkWork(vk::CommandBuffer cmd, Work work) {
    if (work == t_rec->current_work) {
        return;
    }
    // note 32: the kind is kept whether or not a stamp is written; Hazard takes the stages its barriers wait for from it (lean stamps left it stale: a compute
    // write could be noted as graphics, and a later barrier then waited for the graphics stages only)
    t_rec->current_work = work;
    if (!t_rec->stamping || (StampsLean() && !t_rec->segments.empty())) {
        return; // lean: the first segment runs to the submission's end stamp
    }
    const u32 stamp = StampLocked(cmd);
    if (stamp != ~0u) {
        t_rec->segments.push_back({stamp, work});
    }
}

void Collect() {
    const u64 done = instance->GetDevice().getSemaphoreCounterValue(timeline).value;
    while (!t_rec->in_flight.empty() && t_rec->in_flight.front().first <= done) {
        instance->GetDevice().freeCommandBuffers(t_rec->pool, t_rec->in_flight.front().second);
        t_rec->in_flight.pop_front();
    }
}

} // namespace

void Init(const VkbVulkan::Instance& instance_) {
    std::scoped_lock lock{mutex};
    if (instance != nullptr) {
        return;
    }
    instance = &instance_;
    const vk::Device device = instance->GetDevice();
    MakePool();
    const vk::SemaphoreTypeCreateInfo type{.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
    timeline = device.createSemaphore({.pNext = &type}).value;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    MakeBuffer(RingBytes, usage, true, ring_buffer);
    MakeBuffer(64 << 10, usage, false, gds_buffer);
    MakeBuffer(64 << 10, usage, false, null_buffer);
    if (StageOn()) {
        MakeBuffer(StageBytes, usage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false, stage_buffer);
    }
    if (HostCacheOn()) {
        MakeBuffer(HostCacheBytes, usage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false, cache_arena);
    }
    if (HostScratchOn()) {
        MakeBuffer(HostScratchBytes, usage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false, scratch_buffer);
    }
    // the 1x1 image for empty image slots, kept in the general layout
    const VkImageCreateInfo ici{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {1, 1, 1},
                                .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    const VmaAllocationCreateInfo ialloc{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    vmaCreateImage(instance->GetAllocator(), &ici, &ialloc, &null_image, &null_allocation, nullptr);
    null_view = device.createImageView({.image = vk::Image{null_image}, .viewType = vk::ImageViewType::e2D, .format = vk::Format::eR8G8B8A8Unorm,
                                        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}).value;
    occlusion_pool = device.createQueryPool({.queryType = vk::QueryType::eOcclusion, .queryCount = QuerySlots}).value;
    for (auto& pool : stamp_pools) {
        pool = device.createQueryPool({.queryType = vk::QueryType::eTimestamp, .queryCount = MaxStamps}).value;
    }
    if (PassStatsOn() && instance->GetPhysicalDevice().getFeatures().pipelineStatisticsQuery) {
        for (auto& pool : stat_pools) {
            pool = device.createQueryPool({.queryType = vk::QueryType::ePipelineStatistics,
                                           .queryCount = MaxPassStats,
                                           .pipelineStatistics = vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations | vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
                                                                 vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations})
                       .value;
        }
    }
    stage_stamp_pool = device.createQueryPool({.queryType = vk::QueryType::eTimestamp, .queryCount = 2 * StageStampPairs}).value;
    stamp_period_ns = instance->GetPhysicalDevice().getProperties().limits.timestampPeriod;
    const vk::CommandBuffer cmd = Begin();
    cmd.fillBuffer(vk::Buffer{gds_buffer.buffer}, 0, VK_WHOLE_SIZE, 0);
    cmd.fillBuffer(vk::Buffer{null_buffer.buffer}, 0, VK_WHOLE_SIZE, 0);
    if (scratch_buffer.buffer) {
        cmd.fillBuffer(vk::Buffer{scratch_buffer.buffer}, 0, VK_WHOLE_SIZE, 0);
    }
    const vk::ImageMemoryBarrier2 to_general{.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                             .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                                             .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral, .image = vk::Image{null_image},
                                             .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_general});
    t_rec->current_has_work = true;
    (void)SubmitLocked(); // R4a: these first commands go now (the recording threads never submit the shared state)
    LOG_INFO(Render_Vulkan, "vkb: frame stream ready (ring {} MiB)", RingBytes >> 20);
}

bool Ready() {
    return instance != nullptr;
}

/// R5 census (note 34): why open passes end, [reason][0 all, 1 shadow: depth only, 2048x2048]
namespace EndWhy {
constexpr u32 Break = 0, Barrier = 1, OtherDepth = 2, OtherSize = 3, OtherColors = 4, Submit = 5, Transition = 6, Job = 7, Count = Job + static_cast<u32>(Work::Count) + 1;
} // namespace EndWhy
std::atomic<u64> c_pass_ends[EndWhy::Count][2]{};
void EndPassLocked(u32 why = EndWhy::Break) {
    if (t_rec->pass_open && t_rec->current) {
        const bool shadow = t_rec->open_pass.color_count == 0 && t_rec->open_pass.depth && t_rec->open_pass.extent.width == 2048 && t_rec->open_pass.extent.height == 2048;
        c_pass_ends[std::min<u32>(why, EndWhy::Count - 1)][0].fetch_add(1, std::memory_order_relaxed);
        if (shadow) {
            c_pass_ends[std::min<u32>(why, EndWhy::Count - 1)][1].fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (t_rec->pass_open && t_rec->current) {
        t_rec->current.endRendering();
        if (t_rec->stat_open && !t_rec->pass_stamps.empty()) {
            t_rec->current.endQuery(stat_pools[t_rec->generation % StampPools], t_rec->pass_stamps.back().stat);
        }
        t_rec->stat_open = false;
        if (!t_rec->pass_stamps.empty() && t_rec->pass_stamps.back().end == ~0u && t_rec->pass_stamps.back().begin != ~0u) {
            t_rec->pass_stamps.back().end = StampLocked(t_rec->current);
        }
        if (t_rec->capture_pass != ~0u) { // note 91: the frame capture: this pass's targets as they are after it, and the buffers its draws read
            for (u32 a = 0; a < t_rec->pass_images.size(); ++a) {
                Core::Vkb::DumpImageHandle(t_rec->current, t_rec->pass_images[a],
                                           fmt::format("capture/pass{:04}_{}{}", t_rec->capture_pass, a < t_rec->open_pass.color_count ? "color" : "depth", a));
            }
            for (const auto& [buffer, offset, range, index] : t_rec->capture_buffers) {
                Core::Vkb::DumpBuffer(t_rec->current, buffer, offset, range, fmt::format("capture/buffers/b{:05}", index));
            }
            t_rec->capture_buffers.clear();
            t_rec->capture_pass = ~0u;
        }
    }
    t_rec->pass_open = false;
}

Recording Record(Work work) {
    Recording r{std::unique_lock{mutex}, {}};
    r.cmd = Begin();
    EndPassLocked(EndWhy::Job + static_cast<u32>(work));
    MarkWork(r.cmd, work);
    t_rec->current_has_work = true;
    return r;
}

Recording RecordDraw() {
    Recording r{std::unique_lock{mutex}, {}};
    r.cmd = Begin();
    MarkWork(r.cmd, Work::Draw);
    t_rec->current_has_work = true;
    return r;
}

void BreakPass(vk::CommandBuffer) {
    EndPassLocked(EndWhy::Break);
}

bool LegacyBarriersOn() {
    return LegacyBarriers();
}

thread_local u64 hazard_tag = 0;
void SetHazardTag(u64 tag) {
    hazard_tag = tag;
}
thread_local bool t_job_async = false;
void SetJobAsync(bool async) {
    t_job_async = async;
}
/// GPU census (TLG_VKB_OVERLAP_CENSUS=1, run with one recording thread: TLG_VKB_RECORDERS=1, so the hazard checks come in stream order):
/// (1) overlap: for each run of async compute dispatches (a batch), the ranges it writes and reads; then the graphics work after it, job by job, until the first job that
///     reads or writes what the batch wrote, or writes what it read (that job and all after it depend on the batch: the rest could run beside it on a second queue);
/// (2) uploads: a CPU-written buffer copy whose destination no earlier command of its command buffer touched could move to the front of that command buffer.
namespace OverlapCensus {
bool On() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_OVERLAP_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
struct Span {
    u64 begin{~0ull}, end{0};
};
struct Sample {
    u32 batch_dispatches{}, draw_jobs{}, dispatch_jobs{}, other_jobs{};
    bool conflict{};
    u32 conflict_kind{}; // 0 reads a batch write, 1 writes a batch write, 2 writes a batch read
    u32 conflict_work{};
    u64 conflict_tag{};
    bool conflict_image{};
};
std::mutex m;
std::unordered_map<u64, Span> written, read;
bool in_batch = false, in_prefix = false;
Sample cur;
std::vector<Sample> samples;
u64 batches = 0, big_batches = 0, big_draw_jobs = 0, big_dispatch_jobs = 0, big_other_jobs = 0, big_conflicts = 0;
u64 up_total = 0, up_front = 0, up_bytes = 0, up_front_bytes = 0;
std::unordered_set<u64> images; // handles noted as images (ImageUse: the whole range)
void Close() {
    ++batches;
    if (cur.batch_dispatches >= 50) {
        ++big_batches;
        big_draw_jobs += cur.draw_jobs;
        big_dispatch_jobs += cur.dispatch_jobs;
        big_other_jobs += cur.other_jobs;
        big_conflicts += cur.conflict ? 1 : 0;
        if (samples.size() < 64) {
            samples.push_back(cur);
        }
    }
    in_prefix = false;
}
bool Hit(const std::unordered_map<u64, Span>& set, const Use& u) {
    const auto it = set.find(u.handle);
    return it != set.end() && u.begin < it->second.end && it->second.begin < u.end;
}
// v2: per frame, every range each side wrote (a few spans per buffer, merged); each job is checked against what the other side wrote earlier in the frame
struct Spans {
    std::vector<Span> v;
    void Add(u64 b, u64 e) {
        for (Span& x : v) {
            if (b <= x.end && x.begin <= e) {
                x.begin = std::min(x.begin, b);
                x.end = std::max(x.end, e);
                return;
            }
        }
        if (v.size() < 256) {
            v.push_back({b, e});
        } else {
            v[0].begin = std::min(v[0].begin, b), v[0].end = std::max(v[0].end, e);
        }
    }
    bool Hit(u64 b, u64 e) const {
        for (const Span& x : v) {
            if (b < x.end && x.begin < e) {
                return true;
            }
        }
        return false;
    }
};
std::unordered_map<u64, Spans> fa_written, fg_written; // this frame: async writes, graphics writes
u64 frame_seen = 0;
std::atomic<u64> frame_now{0};
u64 v2_frames = 0, g_jobs = 0, g_dep = 0, g_dep_first_job = 0, a_jobs = 0, a_dep = 0, g_jobs_this = 0, g_first_dep_this = ~0ull;
std::map<u64, u64> g_dep_by_tag, a_dep_by_tag;
void NoteV2(std::span<const Use> uses, u32 work, bool async, u64 tag) {
    const u64 f = frame_now.load(std::memory_order_relaxed);
    if (f != frame_seen) {
        frame_seen = f;
        fa_written.clear();
        fg_written.clear();
        ++v2_frames;
        if (g_first_dep_this != ~0ull) {
            g_dep_first_job += g_first_dep_this;
        }
        g_jobs_this = 0;
        g_first_dep_this = ~0ull;
    }
    auto& other = async ? fg_written : fa_written;
    bool dep = false;
    for (const Use& u : uses) {
        if (u.handle == 0) {
            continue;
        }
        if (const auto it = other.find(u.handle); it != other.end() && it->second.Hit(u.begin, u.end)) {
            dep = true;
            break;
        }
    }
    if (async) {
        ++a_jobs;
        if (dep) {
            ++a_dep;
            ++a_dep_by_tag[tag];
        }
    } else {
        ++g_jobs;
        ++g_jobs_this;
        if (dep) {
            ++g_dep;
            ++g_dep_by_tag[work == static_cast<u32>(Work::Dispatch) ? tag : (u64{work} << 56)];
            if (g_first_dep_this == ~0ull) {
                g_first_dep_this = g_jobs_this;
            }
        }
    }
    auto& mine = async ? fa_written : fg_written;
    for (const Use& u : uses) {
        if (u.handle != 0 && u.write) {
            mine[u.handle].Add(u.begin, u.end);
        }
    }
}
void Note(std::span<const Use> uses, u32 work, bool async, u64 tag) {
    std::scoped_lock lock{m};
    NoteV2(uses, work, async, tag);
    if (async) {
        if (!in_batch) {
            if (in_prefix) {
                Close(); // the next batch came first: everything since the last one was independent of it
            }
            written.clear();
            read.clear();
            cur = Sample{};
            in_batch = true;
        }
        ++cur.batch_dispatches;
        for (const Use& u : uses) {
            if (u.handle == 0) {
                continue;
            }
            Span& sp = (u.write ? written : read)[u.handle];
            sp.begin = std::min(sp.begin, u.begin);
            sp.end = std::max(sp.end, u.end);
        }
        return;
    }
    if (in_batch) {
        in_batch = false;
        in_prefix = true;
    }
    if (!in_prefix) {
        return;
    }
    for (const Use& u : uses) {
        if (u.handle == 0) {
            continue;
        }
        const bool raw = Hit(written, u);
        const bool war = u.write && Hit(read, u);
        if (raw || war) {
            cur.conflict = true;
            cur.conflict_kind = raw ? (u.write ? 1 : 0) : 2;
            cur.conflict_work = work;
            cur.conflict_tag = tag;
            cur.conflict_image = u.end == ~0ull && u.begin == 0;
            Close();
            return;
        }
    }
    (work == static_cast<u32>(Work::Draw) ? cur.draw_jobs : work == static_cast<u32>(Work::Dispatch) ? cur.dispatch_jobs : cur.other_jobs) += 1;
}
nlohmann::json Report() {
    std::scoped_lock lock{m};
    nlohmann::json list = nlohmann::json::array();
    for (const Sample& x : samples) {
        list.push_back({{"batch", x.batch_dispatches}, {"independent_draw_jobs", x.draw_jobs}, {"independent_dispatch_jobs", x.dispatch_jobs}, {"independent_other_jobs", x.other_jobs},
                        {"ended_by", x.conflict ? (x.conflict_kind == 0 ? "reads a batch write" : x.conflict_kind == 1 ? "writes a batch write" : "writes a batch read") : "next batch"},
                        {"by_work", x.conflict_work}, {"by_tag", fmt::format("{:#x}", x.conflict_tag)}, {"whole_resource", x.conflict_image}});
    }
    const auto top = [](const std::map<u64, u64>& by) {
        std::vector<std::pair<u64, u64>> v(by.begin(), by.end());
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        nlohmann::json out = nlohmann::json::object();
        for (size_t i = 0; i < v.size() && i < 12; ++i) {
            out[fmt::format("{:#x}", v[i].first)] = v[i].second;
        }
        return out;
    };
    const double fr = static_cast<double>(std::max<u64>(v2_frames, 1));
    nlohmann::json v2 = {{"frames", v2_frames}, {"graphics_jobs_per_frame", g_jobs / fr}, {"graphics_jobs_reading_or_writing_async_writes_per_frame", g_dep / fr},
                         {"first_such_graphics_job_mean_index", g_dep_first_job / fr}, {"async_jobs_per_frame", a_jobs / fr},
                         {"async_jobs_reading_or_writing_graphics_writes_per_frame", a_dep / fr}, {"graphics_dep_by_kind_or_shader", top(g_dep_by_tag)},
                         {"async_dep_by_shader", top(a_dep_by_tag)}};
    return {{"v2", v2}, {"batches", batches}, {"big_batches", big_batches}, {"big_independent_draw_jobs", big_draw_jobs}, {"big_independent_dispatch_jobs", big_dispatch_jobs},
            {"big_independent_other_jobs", big_other_jobs}, {"big_ended_by_conflict", big_conflicts}, {"samples", list},
            {"uploads", {{"cpu_written_copies", up_total}, {"movable_to_front", up_front}, {"mb", up_bytes >> 20}, {"movable_mb", up_front_bytes >> 20}}}};
}
} // namespace OverlapCensus
void SetLanternFootprint(const LanternFootprint* footprint);
namespace {
/// Lantern waits (live issue #98): the game updates the lantern flames with ~150 tiny dispatches of these shaders per frame on the terrace
/// (s04 FA_b01). Each one writes buffers the ones before it wrote, so each got a full wait, ~2.7 ms per frame on an RTX 5090.
/// Measured 11-21: the dispatches never touch the same particle records or the per-particle side records (readback of every
/// dispatch's range, 451 submissions, 0 shared), so among themselves they need no wait. What they do need is the work before
/// them: the first one after any other command keeps its wait, made full so that only these shaders' writes stay outstanding.
/// After that, a write-after-write between two of them is skipped. TLG_VKB_LANTERN_WAITS=0 turns it off.
constexpr std::array<u64, 4> kQuietChain{0x1171c360, 0xa4f2ed2d, 0x82c47ba4, 0x7a83ef95};
bool QuietChainOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_LANTERN_WAITS"); return e == nullptr || e[0] != '0'; }();
    return on;
}
bool IsQuietChain(u64 tag) {
    return std::find(kQuietChain.begin(), kQuietChain.end(), tag) != kQuietChain.end();
}
std::atomic<u64> c_quiet_skipped{0}, c_quiet_overlap{0}, c_quiet_unknown{0};
thread_local const LanternFootprint* t_lantern = nullptr;
/// Exact rule: does this dispatch's footprint miss everything the chain wrote since its last full release?
bool LanternDisjoint(const LanternFootprint& f) {
    for (const auto& [b, e] : t_rec->chain_records) {
        if (f.record_begin < e && b < f.record_end) {
            return false;
        }
    }
    for (u32 i = 0; i < f.slot_count; ++i) {
        if (std::find(t_rec->chain_slots.begin(), t_rec->chain_slots.end(), f.slots[i]) != t_rec->chain_slots.end()) {
            return false;
        }
    }
    return true;
}
/// R5 census (note 70, `TLG_VKB_BARRIER_BY_SHADER=1`, debug only): per dispatch shader (the hazard tag), the commands checked, the barriers they got and the first cause
struct ShaderBarriers {
    u64 checked{}, barriers{};
    std::array<u64, 3> how{}; ///< read after write, write after write, write after read
};
bool BarrierByShaderOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_BARRIER_BY_SHADER"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::mutex bbs_mutex;
std::unordered_map<u64, ShaderBarriers> bbs;
} // namespace
namespace {
/// R5 census (note 32): barriers 150,000 to 150,400 of a run (gameplay), what made them
void LogHazard(std::span<const Use> uses) {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_HAZARD_LOG"); return e != nullptr && e[0] == '1'; }();
    const u64 n = c_hazard_barriers.load(std::memory_order_relaxed);
    if (!on || n < 150000 || n >= 150400) {
        return;
    }
    static const char* kinds[] = {"draw", "dispatch", "copy", "clear", "other", "upload", "copy_buffer", "copy_gds", "fill", "upload_initial", "upload_unmap", "none"};
    std::string text = fmt::format("{} {} tag {:#x}:", n, kinds[std::min<u32>(static_cast<u32>(t_rec->current_work), 11)], hazard_tag);
    const auto ranges_of = [](const RangeSet& set, const Use& u) {
        std::string t;
        if (const auto it = set.find(u.handle); it != set.end()) {
            for (const Range& r : it->second) {
                if (u.begin < r.e && r.b < u.end) {
                    t += fmt::format(" [{:#x},{:#x}) cls{}", r.b, r.e == ~0ull ? 0ull : r.e, r.cls);
                }
            }
        }
        return t;
    };
    for (const Use& u : uses) {
        if (u.handle == 0) {
            continue;
        }
        const std::string w = ranges_of(t_rec->written_since, u), r = u.write ? ranges_of(t_rec->read_since, u) : std::string{};
        if (!w.empty() || !r.empty()) {
            text += fmt::format(" | {:#x} [{:#x},{:#x}) {}{}{}{}{}", u.handle, u.begin, u.end == ~0ull ? 0ull : u.end, u.write ? "W" : "R", w.empty() ? "" : " after write", w,
                                r.empty() ? "" : " after read", r);
        }
    }
    if (FILE* f = std::fopen((Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_hazards.txt").string().c_str(), "a")) {
        std::fputs(text.c_str(), f);
        std::fputc(10, f); // newline
        std::fclose(f);
    }
}
} // namespace

void SetLanternFootprint(const LanternFootprint* footprint) {
    t_lantern = footprint;
}
bool IsLanternShader(u64 hash) {
    return IsQuietChain(hash);
}
bool HoistOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_HOIST_UPLOADS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
constexpr u64 TouchedSlots = 1u << 14;
inline u64 TouchHash(u64 h) {
    return ((h >> 4) * 0x9e3779b97f4a7c15ull) >> 50; // 14 bits
}
/// U1: true if the current command buffer touched `handle`; `mark` adds it
bool Touched(u64 handle, bool mark) {
    auto& t = t_rec->touched;
    if (t.empty()) {
        t.assign(TouchedSlots, {0, 0});
    }
    const u64 gen = t_rec->generation;
    for (u64 i = 0, at = TouchHash(handle); i < 16; ++i, at = (at + 1) & (TouchedSlots - 1)) {
        auto& e = t[at];
        if (e.second != gen) { // free in this command buffer
            if (mark) {
                e = {handle, gen};
            }
            return false;
        }
        if (e.first == handle) {
            return true;
        }
    }
    return true; // a full probe run: say touched (the copy stays in place: always correct)
}
void Touch(u64 handle) {
    (void)Touched(handle, true);
}
std::atomic<u64> c_hoisted{0}, c_hoisted_bytes{0}, c_hoist_in_place{0};
void EnsureStageCmd();
bool HoistCopy(vk::Buffer src, u64 src_offset, vk::Buffer dst, u64 dst_offset, u64 size) {
    if (!HoistOn() || !t_rec->current) {
        return false;
    }
    const u64 handle = reinterpret_cast<u64>(static_cast<VkBuffer>(dst));
    if (Touched(handle, false) || Touches(t_rec->cmd_written, BufferUse(src, src_offset, size, false)) != 0) { // (the source: our GPU work in this command buffer wrote it)
        c_hoist_in_place.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    EnsureStageCmd();
    t_rec->stage_cmd.copyBuffer(src, dst, vk::BufferCopy{src_offset, dst_offset, size});
    Touch(handle); // a second copy into it in this command buffer goes in place (after the first, behind the current command buffer's first barrier)
    c_hoisted.fetch_add(1, std::memory_order_relaxed);
    c_hoisted_bytes.fetch_add(size, std::memory_order_relaxed);
    return true;
}
bool Hazard(vk::CommandBuffer cmd, std::span<const Use> uses, Agent agent) {
    if (LegacyBarriers()) {
        EndPassLocked(EndWhy::Barrier);
        FullBarrierLocked(cmd); // the old rule: every command but a draw (draws call UsePass only)
        return true;
    }
    c_hazard_checks.fetch_add(1, std::memory_order_relaxed);
    if (HoistOn()) {
        for (const Use& u : uses) {
            if (u.handle != 0) {
                Touch(u.handle);
            }
        }
    }
    if (OverlapCensus::On()) [[unlikely]] {
        OverlapCensus::Note(uses, static_cast<u32>(t_rec->current_work), t_job_async, hazard_tag);
        if (t_rec->census_generation != t_rec->generation) {
            t_rec->census_generation = t_rec->generation;
            t_rec->census_touched.clear();
        }
        if (t_rec->current_work == Work::UploadCpuWritten) {
            for (const Use& u : uses) {
                if (u.write && u.handle != 0) {
                    const bool front = t_rec->census_touched.count(u.handle) == 0;
                    std::scoped_lock lock{OverlapCensus::m};
                    ++OverlapCensus::up_total;
                    OverlapCensus::up_bytes += u.end - u.begin;
                    if (front) {
                        ++OverlapCensus::up_front;
                        OverlapCensus::up_front_bytes += u.end - u.begin;
                    }
                }
            }
        }
        for (const Use& u : uses) {
            if (u.handle != 0) {
                t_rec->census_touched.insert(u.handle);
            }
        }
    }
    u8 cls = static_cast<u8>(agent);
    if (cls == 0) {
        cls = t_rec->current_work == Work::Draw ? 1 : t_rec->current_work == Work::Dispatch ? 2 : (IsCopyKind(t_rec->current_work) || t_rec->current_work == Work::Clear || IsUploadKind(t_rec->current_work)) ? 4 : 7;
    }
    u8 src = 0;
    bool counted = false;
    bool only_cs_buffer_waw = cls == 2; // note 32 bound test: the only conflict is a dispatch writing a buffer earlier dispatches wrote
    for (const Use& u : uses) {
        if (u.handle == 0) {
            continue;
        }
        const u8 after_write = Touches(t_rec->written_since, u);
        const u8 after_read = u.write ? Touches(t_rec->read_since, u) : 0;
        src |= after_write | after_read;
        if ((after_write | after_read) != 0 &&
            (after_read != 0 || !u.write || after_write != 2 || u.end == ~0ull || u.handle == reinterpret_cast<u64>(gds_buffer.buffer))) {
            only_cs_buffer_waw = false;
        }
        if ((after_write | after_read) != 0 && !counted) {
            counted = true;
            const u32 how = after_write ? (u.write ? 1 : 0) : 2;
            const u32 what = u.handle == reinterpret_cast<u64>(gds_buffer.buffer) ? 0 : u.handle == reinterpret_cast<u64>(null_buffer.buffer) ? 1 : u.end == ~0ull && u.begin == 0 ? 3 : 2;
            c_hazard_cause[std::min<u32>(static_cast<u32>(t_rec->current_work), 5)][how][what].fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (BarrierByShaderOn() && cls == 2) [[unlikely]] {
        u32 how = 3;
        for (const Use& u : uses) {
            if (u.handle != 0 && how == 3) {
                const u8 w = Touches(t_rec->written_since, u), r = u.write ? Touches(t_rec->read_since, u) : 0;
                how = w ? (u.write ? 1 : 0) : r ? 2 : 3;
            }
        }
        std::scoped_lock lock{bbs_mutex};
        ShaderBarriers& s = bbs[hazard_tag];
        ++s.checked;
        if (src != 0) {
            ++s.barriers;
            if (how < 3) {
                ++s.how[how];
            }
        }
    }
    static const bool skip_cs_waw = [] { const char* e = std::getenv("TLG_VKB_TEST_SKIP_CS_WAW"); return e != nullptr && e[0] == '1'; }(); // note 32 bound test only
    if (src != 0 && skip_cs_waw && only_cs_buffer_waw) {
        src = 0;
    }
    const bool quiet = cls == 2 && QuietChainOn() && IsQuietChain(hazard_tag);
    if (src != 0 && quiet && only_cs_buffer_waw && t_rec->quiet_chain) {
        // exact rule: only the earlier lantern dispatches' writes are outstanding; skip the wait when this one's records and slots (read from the engine's records
        // before it ran) miss all of theirs
        if (t_lantern == nullptr || !t_lantern->known) {
            c_quiet_unknown.fetch_add(1, std::memory_order_relaxed);
        } else if (!LanternDisjoint(*t_lantern)) {
            c_quiet_overlap.fetch_add(1, std::memory_order_relaxed);
        } else {
            src = 0;
            c_quiet_skipped.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (src != 0 && quiet) {
        src = 7; // the wait it keeps releases everything, so only lantern writes stay outstanding after it
    }
    const bool released_all = src == 7;
    if (src != 0) {
        LogHazard(uses);
        EndPassLocked(EndWhy::Barrier);
        if (FullBarriers()) {
            FullBarrierLocked(cmd);
        } else {
            // V4h: wait for the kinds of work that touched it; everything after sees their writes
            // note 70 bound test only (TLG_VKB_TEST_NARROW_DST=1): the barrier waits only in the stages of the command that conflicts, so later work of other kinds can
            // overlap; NOT correct as it stands (the release below assumes every later command saw the writes), never a default
            static const bool narrow_dst = [] { const char* e = std::getenv("TLG_VKB_TEST_NARROW_DST"); return e != nullptr && e[0] == '1'; }();
            const vk::PipelineStageFlags2 dst_stages = !narrow_dst ? vk::PipelineStageFlagBits2::eAllCommands
                                                       : cls == 2 ? vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eDrawIndirect
                                                       : cls == 1 ? vk::PipelineStageFlagBits2::eAllGraphics
                                                                  : vk::PipelineStageFlagBits2::eAllCommands;
            const vk::MemoryBarrier2 barrier{.srcStageMask = StagesOf(src), .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                             .dstStageMask = dst_stages,
                                             .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
            cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
            Release(t_rec->written_since, src);
            Release(t_rec->read_since, src);
        }
        c_barrier_src[src & 7].fetch_add(1, std::memory_order_relaxed);
        c_hazard_barriers.fetch_add(1, std::memory_order_relaxed);
    }
    t_rec->quiet_chain = quiet && (t_rec->quiet_chain || released_all) && t_lantern != nullptr && t_lantern->known;
    if (released_all) { // the full release: only this dispatch's writes are outstanding now
        t_rec->chain_records.clear();
        t_rec->chain_slots.clear();
    }
    if (t_rec->quiet_chain) {
        t_rec->chain_records.push_back({t_lantern->record_begin, t_lantern->record_end});
        t_rec->chain_slots.insert(t_rec->chain_slots.end(), t_lantern->slots.begin(), t_lantern->slots.begin() + t_lantern->slot_count);
    } else {
        t_rec->chain_records.clear();
        t_rec->chain_slots.clear();
    }
    for (const Use& u : uses) {
        if (u.handle != 0) {
            Note(u.write ? t_rec->written_since : t_rec->read_since, u, cls);
            if (u.write && (StageOn() || HoistOn())) {
                Note(t_rec->cmd_written, u, cls);
            }
        }
    }
    return src != 0;
}

void NotePassDraw(u32 flags, u64 ps_hash, u32 format, const vk::Viewport& viewport, const vk::Rect2D& scissor) {
    if (!t_rec->pass_open || t_rec->pass_stamps.empty() || t_rec->pass_stamps.back().end != ~0u) {
        return;
    }
    PassStamp& p = t_rec->pass_stamps.back();
    p.flags |= flags;
    if (p.ps == 0) {
        p.ps = ps_hash, p.format = format, p.viewport = viewport, p.scissor = scissor;
    }
}

bool UsePass(vk::CommandBuffer cmd, const Pass& pass, std::span<const Use> attachments) {
    if (t_rec->pass_open && t_rec->open_pass == pass) {
        if (!t_rec->pass_stamps.empty() && t_rec->pass_stamps.back().end == ~0u) {
            ++t_rec->pass_stamps.back().draws;
        }
        return false;
    }
    EndPassLocked(!t_rec->pass_open ? EndWhy::Barrier : t_rec->open_pass.depth != pass.depth ? EndWhy::OtherDepth : t_rec->open_pass.extent != pass.extent || t_rec->open_pass.layers != pass.layers ? EndWhy::OtherSize : EndWhy::OtherColors);
    if (LegacyBarriers()) {
        if (!NoPassBarrier()) {
            FullBarrierLocked(cmd);
        }
    } else {
        (void)Hazard(cmd, attachments); // the pass's draws write its attachments: only an earlier read or write of them since the last barrier needs one
    }
    std::array<vk::RenderingAttachmentInfo, 8> colors{};
    for (u32 i = 0; i < pass.color_count; ++i) {
        colors[i] = {.imageView = pass.colors[i], .imageLayout = vk::ImageLayout::eGeneral, .loadOp = vk::AttachmentLoadOp::eLoad, .storeOp = vk::AttachmentStoreOp::eStore};
    }
    const vk::RenderingAttachmentInfo depth{.imageView = pass.depth, .imageLayout = vk::ImageLayout::eGeneral, .loadOp = vk::AttachmentLoadOp::eLoad,
                                            .storeOp = vk::AttachmentStoreOp::eStore};
    const vk::RenderingInfo info{.renderArea = {{0, 0}, pass.extent},
                                 .layerCount = std::max(pass.layers, 1u),
                                 .colorAttachmentCount = pass.color_count,
                                 .pColorAttachments = colors.data(),
                                 .pDepthAttachment = pass.depth ? &depth : nullptr,
                                 .pStencilAttachment = pass.depth && pass.depth_has_stencil ? &depth : nullptr};
    if (t_rec->stamping && !StampsLean() && t_rec->stamp_count + 8 < MaxStamps) {
        PassStamp ps;
        ps.begin = StampLocked(cmd);
        ps.width = pass.extent.width, ps.height = pass.extent.height, ps.draws = 1;
        ps.kind = pass.color_count == 0 ? 0 : pass.color_count >= 4 ? 2 : 1;
        ps.colors = pass.color_count;
        if (ps.begin != ~0u) {
            if (const vk::QueryPool sp = stat_pools[t_rec->generation % StampPools]; sp && t_rec->stat_count < MaxPassStats) {
                ps.stat = t_rec->stat_count++;
                cmd.beginQuery(sp, ps.stat, {});
                t_rec->stat_open = true;
            }
            t_rec->pass_stamps.push_back(ps);
        }
    }
    cmd.beginRendering(info);
    t_rec->pass_open = true;
    t_rec->open_pass = pass;
    t_rec->pass_images.clear();
    t_rec->capture_pass = ~0u;
    if (frame_capture.load(std::memory_order_relaxed) || capture_jobs_left.load(std::memory_order_relaxed) != 0) { // note 91
        for (const Use& u : attachments) {
            t_rec->pass_images.push_back(vk::Image{reinterpret_cast<VkImage>(u.handle)});
        }
    }
    c_passes.fetch_add(1, std::memory_order_relaxed);
    return true;
}

namespace {
u64 SubmitLocked();
u32 Recorders();
u64 CloseSubmission();
std::atomic<u64> c_splits{0};
/// V4g: TLG_VKB_SPLIT=N (0 off): the recording thread submits what it has recorded after N jobs, so the GPU starts on the frame while the game still makes it.
/// V4l: default 512 (was 128): about 5 submissions a frame instead of 16; each submission's staging costs GPU time of its own (note 26). 0 lost the device (note 25).
u32 SplitJobs() {
    static const u32 n = [] { const char* e = std::getenv("TLG_VKB_SPLIT"); return e != nullptr && e[0] != 0 ? static_cast<u32>(std::atoi(e)) : 512u; }();
    return RecorderOn() ? n : 0u;
}
} // namespace

extern std::atomic<void (*)(u32)> g_before_enqueue; // R3b (defined at Enqueue)
thread_local u32 t_submit_source = 0; ///< D4-10 counts: who asked for the submit being closed (Submit's `source`)
u64 Submit(u32 source) {
    t_submit_source = source;
    if (auto* hook = g_before_enqueue.load(std::memory_order_acquire)) {
        hook(~0u); // R3b: the open batch belongs to this submission
    }
    if (Recorders() > 1 && on_recorder && t_rec == &shared_rec) {
        return CloseSubmission(); // R4a: on the resolver (the engine's submit job)
    }
    std::scoped_lock lock{mutex};
    return SubmitLocked();
}

namespace {
u64 SubmitLocked() {
    t_rec->jobs_in_cmd = 0;
    if (instance == nullptr) {
        return 0;
    }
    Collect();
    const bool worker = t_rec->value != 0; // R4a: a recording thread's submission (its value handed out when the resolver closed it)
    if (!worker && (!t_rec->current || !t_rec->current_has_work)) {
        c_empty_submits.fetch_add(1, std::memory_order_relaxed);
        return next_value.load(std::memory_order_relaxed);
    }
    const u64 value = worker ? t_rec->value : next_value.load(std::memory_order_relaxed) + 1;
    if (t_rec->current) {
        EndPassLocked(EndWhy::Submit);
        if (t_rec->query_open) {
            t_rec->current.endQuery(occlusion_pool, t_rec->query_open_index); // the engine submitted inside the query: it counts what this command buffer drew
            {
                std::scoped_lock blocks_lock{blocks_mutex};
                query_slots[t_rec->query_open_index].pending = true;
                query_slots[t_rec->query_open_index].value = value;
            }
            t_rec->query_open = false;
            c_q_cut_by_submit.fetch_add(1, std::memory_order_relaxed);
            // open item D4-10: the engine's query interval goes on past this submit; our Vulkan query counts only what this command buffer drew
            Core::RareEvents::OpenItem("D4-10", fmt::format("an explicit submit inside an occlusion query (block {:#x}): the query ended at the submit",
                                                            t_rec->query_open_address));
        }
        if (t_rec->stamping && !t_rec->segments.empty()) {
            (void)StampLocked(t_rec->current); // closes the last segment
        }
        (void)t_rec->current.end();
    }
    if (t_rec->stage_cmd) {
        if (t_rec->stage_stamp_open != ~0u) {
            t_rec->stage_cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, stage_stamp_pool, t_rec->stage_stamp_open + 1);
        }
        (void)t_rec->stage_cmd.end();
    }
    if (worker) {
        // R4a: the submissions go to the queue in value order (each one's first barrier covers the ones before it)
        for (u64 last = last_value.load(std::memory_order_acquire); last + 1 != value; last = last_value.load(std::memory_order_acquire)) {
            last_value.wait(last, std::memory_order_acquire);
        }
    } else {
        next_value.store(value, std::memory_order_relaxed);
    }
    std::array<vk::CommandBufferSubmitInfo, 2> cmd_infos{};
    u32 cmd_count = 0;
    if (t_rec->stage_cmd) {
        if (t_rec->stage_stamp_open != ~0u) {
            std::scoped_lock stage_lock{stage_mutex};
            stage_stamped.push_back({value, t_rec->stage_stamp_open});
        }
        t_rec->stage_stamp_open = ~0u;
#if defined(__clang__)
        cmd_infos[cmd_count++] = {.commandBuffer = t_rec->stage_cmd};
#else // GCC: {..} also matches operator=(VkCommandBufferSubmitInfo const&) and is ambiguous
        cmd_infos[cmd_count++] = vk::CommandBufferSubmitInfo{.commandBuffer = t_rec->stage_cmd};
#endif
        t_rec->in_flight.emplace_back(value, t_rec->stage_cmd);
        t_rec->stage_map.clear(); // a copy serves its submission only: the CPU may add to a bound range before a later draw of the same frame
        t_rec->stage_regions.clear();
        t_rec->stage_cmd = vk::CommandBuffer{};
    }
    if (t_rec->current) {
#if defined(__clang__)
        cmd_infos[cmd_count++] = {.commandBuffer = t_rec->current};
#else // GCC: {..} also matches operator=(VkCommandBufferSubmitInfo const&) and is ambiguous
        cmd_infos[cmd_count++] = vk::CommandBufferSubmitInfo{.commandBuffer = t_rec->current};
#endif
    }
    const vk::SemaphoreSubmitInfo signal{.semaphore = timeline, .value = value, .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
    const vk::SubmitInfo2 submit{.commandBufferInfoCount = cmd_count, .pCommandBufferInfos = cmd_infos.data(), .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signal};
    {
        std::scoped_lock queue_lock{VkbVulkan::Scheduler::submit_mutex};
        Core::Flight::Add(Core::Flight::Kind::QueueSubmit, 0, value, 0); // flight recorder: the frame stream's submission (timeline value)
        if (const vk::Result result = instance->GetGraphicsQueue().submit2(submit); result != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "vkb: frame stream submit failed: {}", vk::to_string(result));
        }
    }
    if (t_rec->current) {
        t_rec->in_flight.emplace_back(value, t_rec->current);
    }
    {
        std::scoped_lock ring_lock{ring_mutex};
        frame_ring_bytes = 0;
    }
    if (t_rec->stamping) {
        const u32 pool = static_cast<u32>(t_rec->generation % StampPools);
        std::scoped_lock stamp_lock{stamp_mutex};
        if (t_rec->stamp_count != 0) {
            stamped.push_back({value, t_rec->generation, pool, t_rec->stamp_count, std::move(t_rec->segments), std::move(t_rec->pass_stamps), t_rec->lean, std::move(t_rec->draw_stamps)}); // its pool is freed once read
        } else {
            pool_busy[pool] = false;
        }
        t_rec->segments.clear();
        t_rec->pass_stamps.clear();
        t_rec->draw_stamps.clear();
        t_rec->last_stamp = ~0u;
    }
    t_rec->stamping = false;
    t_rec->current = vk::CommandBuffer{};
    t_rec->current_has_work = false;
    last_value.store(value, std::memory_order_release);
    last_value.notify_all();
    c_submits.fetch_add(1, std::memory_order_relaxed);
    return value;
}
} // namespace

u64 LastValue() {
    return last_value.load(std::memory_order_relaxed);
}
u64 RecordingValue() {
    return RecValue();
}

u64 CompletedValue() {
    return instance != nullptr ? instance->GetDevice().getSemaphoreCounterValue(timeline).value : 0;
}

void PendTransition(vk::Image image, vk::ImageLayout old_layout, vk::ImageAspectFlags aspect) {
    pending_transitions.push_back({image, old_layout, aspect});
}

bool RecorderOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_VKB_RECORDER");
        const char* o = std::getenv("TLG_VKB_OVERLAP");
        return (e == nullptr || e[0] != '0') && (o == nullptr || o[0] != '0');
    }();
    return on;
}

bool OnRecorder() {
    return on_recorder || !RecorderOn();
}

namespace {
/// issue 111 (CopyInStream): lock held (the recording's), outside a pass, with the stream's hazard barriers as any copy job
void RecordCopiesLocked(vk::CommandBuffer cmd, std::span<const Job::Copy> copies) {
    EndPassLocked(EndWhy::Job + static_cast<u32>(Work::Upload));
    MarkWork(cmd, Work::Upload);
    t_rec->current_has_work = true;
    for (const Job::Copy& c : copies) {
        if (c.source) {
            (void)Hazard(cmd, {BufferUse(c.source, c.source_offset, c.bytes, false), BufferUse(c.destination, 0, c.bytes, true)});
            cmd.copyBuffer(c.source, c.destination, vk::BufferCopy{c.source_offset, 0, c.bytes});
        } else {
            (void)Hazard(cmd, {BufferUse(c.destination, 0, c.bytes, true)});
            cmd.fillBuffer(c.destination, 0, VK_WHOLE_SIZE, 0);
        }
        c_copies_first.fetch_add(1, std::memory_order_relaxed);
        c_copies_first_mb.fetch_add(c.bytes, std::memory_order_relaxed);
    }
}
} // namespace

void CopyInStream(vk::Buffer source, u64 source_offset, vk::Buffer destination, u64 bytes) {
    const Job::Copy c{source, source_offset, destination, bytes};
    if (t_preparing != nullptr) {
        t_preparing->copies_first.push_back(c);
        return;
    }
    if (on_recorder && RecorderOn() && t_in_locked_run && t_rec != nullptr && t_rec->current) {
        const Work was = t_rec->current_work; // a job's Run (the single recorder): its lock is held, its command buffer begun; its own kind is back after
        RecordCopiesLocked(t_rec->current, std::span<const Job::Copy>{&c, 1});
        MarkWork(t_rec->current, was);
        c_copies_first_in_run.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    c_copies_first_own_job.fetch_add(1, std::memory_order_relaxed); // (not from a job's Prepare or Run: a job of its own, behind what is queued)
    Defer(Work::Upload, [c](vk::CommandBuffer cmd) { RecordCopiesLocked(cmd, std::span<const Job::Copy>{&c, 1}); });
}

namespace {
void RunJob(Job& job) {
    if (job.locked || !job.transitions.empty()) {
        const bool shared = t_rec == &shared_rec; // R4a: a recording thread's state is its own (no lock; its submission is cut by the resolver)
        std::unique_lock lock{mutex, std::defer_lock};
        if (shared) {
            lock.lock();
        }
        const vk::CommandBuffer cmd = Begin();
        if (!job.transitions.empty()) {
            EndPassLocked(EndWhy::Transition); // a barrier: outside a render pass
            std::vector<vk::ImageMemoryBarrier2> barriers;
            barriers.reserve(job.transitions.size());
            for (const Transition& t : job.transitions) {
                barriers.push_back({.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                    .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                    .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                    .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                                    .oldLayout = t.old_layout,
                                    .newLayout = vk::ImageLayout::eGeneral,
                                    .image = t.image,
                                    .subresourceRange = {t.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
            }
            cmd.pipelineBarrier2({.imageMemoryBarrierCount = static_cast<u32>(barriers.size()), .pImageMemoryBarriers = barriers.data()});
            t_rec->current_has_work = true;
        }
        if (job.locked) {
            if (!job.copies_first.empty()) {
                RecordCopiesLocked(cmd, job.copies_first); // issue 111: in front of the job that asked for them
            }
            if (job.work != Work::Draw) {
                EndPassLocked(EndWhy::Job + static_cast<u32>(job.work));
            }
            MarkWork(cmd, job.work);
            t_rec->current_has_work = true;
            t_in_locked_run = true;
            job.Run(cmd);
            t_in_locked_run = false;
            if (const u32 split = SplitJobs(); shared && split != 0 && (t_rec->jobs_in_cmd += job.weight) >= split && !t_rec->query_open) {
                (void)SubmitLocked(); // its records (none: the engine's fences go with the engine's submits) need nothing
                c_splits.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
    }
    job.Run(vk::CommandBuffer{});
}

/// R4a (note 65): TLG_VKB_RECORDERS=N (default 2; N=1: the single recorder). The recorder thread becomes the resolver: it runs the
/// unlocked jobs (submits, deletions, the objects tick) and each job's Prepare (a draw's late buffers) in order, and cuts the stream into submissions (at the engine's
/// submits and every TLG_VKB_SPLIT jobs, never inside an occlusion query). Each submission is recorded whole by one of N recording threads, in turn, into its own
/// command buffer (each begins with a full barrier and ends its pass, as every submission did) and goes to the queue in value order.
u32 Recorders() {
    static const u32 n = [] {
        const char* e = std::getenv("TLG_VKB_RECORDERS");
        return e != nullptr && e[0] != 0 ? std::clamp(static_cast<u32>(std::atoi(e)), 1u, 4u) : 2u;
    }();
    return RecorderOn() ? n : 1u;
}
/// A submission is opened on a recording thread at its first job (its timeline value handed out then) and streamed to it in batches while the resolver
/// works on, so the recording runs beside the engine as it did on the single recorder; the close only marks its end.
struct Item {
    Job* job; ///< null: the end of the submission `value`
    u64 value;
};
struct Worker {
    std::mutex m;
    std::condition_variable cv;
    std::vector<Item> items;
    bool waiting = false;
    Rec rec;
};
std::array<Worker, 4> workers;
std::atomic<u64> closed_value{0}; ///< the last submission the resolver closed (Drain waits until it is on the queue)
Worker* open_worker = nullptr;    ///< the resolver's open submission (none: null)
u64 open_value = 0;
u32 open_jobs = 0;
bool open_query = false;
u64 open_query_address = 0; ///< D4-10: the block of the open query
bool resume_query = false;  ///< D4-10: a submission closed inside a query: the next one begins with its continuation
std::vector<Item> batch; ///< items for open_worker not handed over yet
std::atomic<u64> c_r4a_submissions{0}, c_r4a_jobs{0}, c_r4a_transition_jobs{0}, c_r4a_worker_waits{0}, c_r4a_batches{0};
struct TransitionJob final : Job {
    void Run(vk::CommandBuffer) override {}
};
void FlushBatch() {
    if (batch.empty() || open_worker == nullptr) {
        return;
    }
    Worker& w = *open_worker;
    bool wake = false;
    {
        std::scoped_lock lock{w.m};
        w.items.insert(w.items.end(), batch.begin(), batch.end());
        wake = w.waiting;
    }
    batch.clear();
    if (wake) {
        w.cv.notify_one();
    }
    c_r4a_batches.fetch_add(1, std::memory_order_relaxed);
}
u64 CloseSubmission() {
    if (open_worker == nullptr) {
        c_empty_submits.fetch_add(1, std::memory_order_relaxed);
        return std::max(closed_value.load(std::memory_order_relaxed), last_value.load(std::memory_order_relaxed));
    }
    const u64 value = open_value;
    if (open_query && QueryCarryOn()) {
        // D4-10: the engine submits inside an occlusion query: this submission's part of it ends here, the next submission carries it on
        struct CutQuery {
            void operator()(vk::CommandBuffer cmd) const {
                QueryCut(cmd);
            }
        };
        auto cut = std::make_unique<FnJob<CutQuery>>(CutQuery{});
        cut->work = Work::Other;
        batch.push_back({cut.release(), value});
        resume_query = true;
        const u32 source = std::min<u32>(t_submit_source, 2);
        if (c_q_carried_by[source].fetch_add(1, std::memory_order_relaxed) < 5) {
            static constexpr const char* names[] = {"other", "the engine's graphics submit", "the engine's async compute submit"};
            LOG_WARNING(Render_Vulkan, "D4-10: {} closed a submission inside the occlusion query of block {:#x}; the next submission carries it on", names[source],
                    open_query_address);
        }
    }
    batch.push_back({nullptr, value});
    FlushBatch();
    closed_value.store(value, std::memory_order_release);
    open_worker = nullptr;
    open_jobs = 0;
    c_r4a_submissions.fetch_add(1, std::memory_order_relaxed);
    return value;
}
void AddToSubmission(std::unique_ptr<Job> job) {
    if (open_worker == nullptr) {
        open_value = next_value.fetch_add(1, std::memory_order_relaxed) + 1;
        open_worker = &workers[open_value % Recorders()];
        if (resume_query) { // D4-10: the query cut at the last submission goes on first thing in this one
            resume_query = false;
            if (open_query) {
                struct ResumeQuery {
                    u64 address;
                    void operator()(vk::CommandBuffer cmd) const {
                        QueryBegin(cmd, address, true);
                    }
                };
                auto resume = std::make_unique<FnJob<ResumeQuery>>(ResumeQuery{open_query_address});
                resume->work = Work::Other;
                batch.push_back({resume.release(), open_value});
            }
        }
    }
    if (job->query != 0) {
        open_query = job->query > 0;
        if (open_query) {
            open_query_address = job->query_address;
        }
    }
    const bool counts = job->locked;
    const u32 weight = job->weight;
    batch.push_back({job.release(), open_value});
    c_r4a_jobs.fetch_add(1, std::memory_order_relaxed);
    if (const u32 split = SplitJobs(); counts && split != 0 && (open_jobs += weight) >= split && !open_query) {
        (void)CloseSubmission();
        c_splits.fetch_add(1, std::memory_order_relaxed);
    } else if (batch.size() >= 32) {
        FlushBatch();
    }
}
void Resolve(std::unique_ptr<Job> job) {
    if (!job->locked && job->transitions.empty()) {
        FlushBatch(); // (an unlocked job may wait for the GPU: what is resolved goes to its recording thread first)
        job->Run(vk::CommandBuffer{});
        return;
    }
    if (!job->locked) { // an unlocked job with layout changes: they go into the stream, the job runs here
        auto t = std::make_unique<TransitionJob>();
        t->locked = false;
        t->transitions.swap(job->transitions);
        AddToSubmission(std::move(t));
        c_r4a_transition_jobs.fetch_add(1, std::memory_order_relaxed);
        FlushBatch();
        job->Run(vk::CommandBuffer{});
        return;
    }
    t_preparing = job.get();
    job->Prepare();
    t_preparing = nullptr;
    AddToSubmission(std::move(job));
}
void WorkerThread(u32 index) {
    Common::SetCurrentThreadName(index == 0 ? "vkb:Record1" : index == 1 ? "vkb:Record2" : index == 2 ? "vkb:Record3" : "vkb:Record4");
    on_recorder = true;
    Worker& w = workers[index];
    t_rec = &w.rec;
    std::vector<Item> mine;
    for (;;) {
        {
            std::unique_lock lock{w.m};
            if (w.items.empty()) {
                c_r4a_worker_waits.fetch_add(1, std::memory_order_relaxed);
                w.waiting = true;
                w.cv.wait(lock, [&] { return !w.items.empty(); });
                w.waiting = false;
            }
            mine.swap(w.items);
        }
        for (const Item& item : mine) {
            t_rec->value = item.value;
            if (item.job != nullptr) {
                RunJob(*item.job);
                delete item.job;
            } else {
                (void)SubmitLocked();
            }
        }
        mine.clear();
    }
}

void RecorderThread() {
    Common::SetCurrentThreadName("vkb:Recorder");
    on_recorder = true;
    if (Recorders() > 1) {
        for (u32 i = 0; i < Recorders(); ++i) {
            std::thread{WorkerThread, i}.detach();
        }
    }
    for (;;) {
        std::unique_ptr<Job> job;
        job_running.store(true, std::memory_order_seq_cst);
        if (Job* j = QueuePop(); j != nullptr) {
            job.reset(j);
            queue_size.fetch_sub(1, std::memory_order_seq_cst);
        } else if (queue_size.load(std::memory_order_seq_cst) != 0) {
            std::this_thread::yield(); // a producer is between its exchange and its link
            continue;
        } else {
            if (Recorders() > 1) {
                FlushBatch(); // R4a: nothing resolved waits while the resolver sleeps
            }
            job_running.store(false, std::memory_order_seq_cst);
            recorder_idle_gen.fetch_add(1, std::memory_order_seq_cst);
            recorder_idle_gen.notify_all();
            // note 82: a short spin on the queue (pause instructions, no clock reads) before sleeping. The render thread is the slower side, so the queue runs
            // empty between draws; a sleeping resolver costs the render thread a wake-up system call at its next hand-over (ZwAlertThreadByThreadIdEx, 2.4 % of
            // it). TLG_VKB_RESOLVER_SPIN=<pause iterations> (default 2000, 0 off)
            static const u32 spin = [] { const char* e = std::getenv("TLG_VKB_RESOLVER_SPIN"); return e != nullptr ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 2000u; }();
            bool arrived = false;
            for (u32 i = 0; i < spin; ++i) {
                if (queue_size.load(std::memory_order_relaxed) != 0) {
                    arrived = true;
                    break;
                }
                _mm_pause();
            }
            if (arrived) {
                c_rec_spin_hits.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            recorder_sleeping.store(true, std::memory_order_seq_cst);
            if (queue_size.load(std::memory_order_seq_cst) == 0) {
                const u64 s0 = RecNs();
                recorder_sleeping.wait(true, std::memory_order_seq_cst);
                c_rec_sleep_ns.fetch_add(RecNs() - s0, std::memory_order_relaxed);
                c_rec_sleeps.fetch_add(1, std::memory_order_relaxed);
            }
            recorder_sleeping.store(false, std::memory_order_seq_cst);
            continue;
        }
        // sensitivity experiment: TLG_VKB_RECORDER_SPIN_NS=N spends N ns more per draw job on the recording thread (is it on the frame's critical path)
        static const u64 spin_ns = [] { const char* e = std::getenv("TLG_VKB_RECORDER_SPIN_NS"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull; }();
        if (spin_ns != 0 && job->work == Work::Draw) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(spin_ns);
            while (std::chrono::steady_clock::now() < until) {
            }
        }
        if (job->splice) { // P1 / P2: the pass made on another thread, its jobs in this place, each as soon as it is made
            auto* sp = static_cast<Splice*>(job.get());
            c_splices.fetch_add(1, std::memory_order_relaxed);
            if (!job->transitions.empty()) { // the layout changes the engine's thread decided before the placeholder (they ride on its next job: this one)
                auto t = std::make_unique<TransitionJob>();
                t->transitions.swap(job->transitions);
                if (Recorders() > 1) {
                    Resolve(std::move(t));
                } else {
                    RunJob(*t);
                }
            }
            size_t next = 0;
            for (;;) {
                const u32 state = sp->pushed.load(std::memory_order_acquire);
                const u32 count = state & 0x7fffffffu;
                if (next < count) {
                    std::vector<std::unique_ptr<Job>> take;
                    {
                        std::scoped_lock lock{sp->m};
                        for (; next < sp->jobs.size(); ++next) {
                            take.push_back(std::move(sp->jobs[next]));
                        }
                    }
                    c_splice_jobs.fetch_add(take.size(), std::memory_order_relaxed);
                    for (auto& inner : take) {
                        if (Recorders() > 1) {
                            Resolve(std::move(inner));
                        } else {
                            RunJob(*inner);
                        }
                    }
                    continue;
                }
                if ((state & 0x80000000u) != 0) {
                    break;
                }
                c_splice_waits.fetch_add(1, std::memory_order_relaxed);
                const u64 w0 = RecNs();
                if (Recorders() > 1) {
                    FlushBatch(); // what is resolved so far goes to its recording thread while we wait
                }
                sp->pushed.wait(state, std::memory_order_acquire);
                c_splice_wait_us.fetch_add((RecNs() - w0) / 1000, std::memory_order_relaxed);
            }
            job.release();
            if (sp->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                delete sp;
            }
            continue;
        }
        const u32 kind = job->locked ? static_cast<u32>(job->work) : static_cast<u32>(Work::Count);
        if (Recorders() > 1) {
            Resolve(std::move(job));
            continue;
        }
        if (!RecTimesOn()) {
            RunJob(*job);
            continue;
        }
        const u64 r0 = RecNs();
        RunJob(*job);
        c_rec_run_ns[kind].fetch_add(RecNs() - r0, std::memory_order_relaxed);
        c_rec_run_n[kind].fetch_add(1, std::memory_order_relaxed);
    }
}
} // namespace

u64 RecordingGeneration() {
    return t_rec->generation; // R4a: this thread's command buffer (each one begun takes a new number)
}
bool JobPoolOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_JOB_POOL"); return e == nullptr || e[0] != '0'; }();
    return on;
}
void EnqueueImpl(std::unique_ptr<Job> job);
std::atomic<void (*)(u32)> g_before_enqueue{nullptr};
void SetBeforeEnqueue(void (*hook)(u32)) {
    g_before_enqueue.store(hook, std::memory_order_release);
}
void EnqueueNoHook(std::unique_ptr<Job> job) {
    EnqueueImpl(std::move(job));
}
void Enqueue(std::unique_ptr<Job> job) {
    if (auto* hook = g_before_enqueue.load(std::memory_order_acquire)) {
        hook(static_cast<u32>(job->work));
    }
    if (!RecTimesOn()) {
        EnqueueImpl(std::move(job));
        return;
    }
    const u64 e0 = RecNs();
    EnqueueImpl(std::move(job));
    c_enqueue_ns.fetch_add(RecNs() - e0, std::memory_order_relaxed);
}
std::atomic<u64> g_order_serial{1};
std::atomic<u32> g_open_splices{0};
u64 OrderSerial() {
    return g_order_serial.fetch_add(1, std::memory_order_acq_rel);
}
u64 SinkOpenSerial() {
    return t_sink != nullptr ? t_sink->open_serial : 0;
}
u32 OpenSplices() {
    return g_open_splices.load(std::memory_order_acquire);
}
std::mutex open_splices_mutex;
std::vector<Splice*> open_splices; ///< (open_splices_mutex) placeholders not yet closed, in stream order
thread_local bool t_mover = false; ///< issue 130: this thread runs the engine's heap mover
std::atomic<u64> c_mover_into_splice{0}, c_mover_into_stream{0};
bool MoverSinkBegin() {
    static const bool legacy = [] { const char* e = std::getenv("TLG_LEGACY_MOVER_IN_STREAM"); return e != nullptr && e[0] == '1'; }();
    if (legacy || t_sink != nullptr || t_mover) {
        return false;
    }
    t_mover = true;
    return true;
}
bool MoverIntoSplice() {
    return t_mover && g_open_splices.load(std::memory_order_acquire) != 0;
}
void MoverSinkEnd(bool begun) {
    if (begun) {
        t_mover = false;
    }
}
/// A heap mover job into the oldest placeholder still open (under the registry lock, so the placeholder cannot be freed meanwhile; under its own lock, so it is not
/// closed meanwhile); false: none open
bool PushToOldestSplice(std::unique_ptr<Job>& job) {
    std::scoped_lock lock{open_splices_mutex};
    for (Splice* s : open_splices) {
        std::scoped_lock splice_lock{s->m};
        if (s->closed.load(std::memory_order_acquire) != 0) {
            continue;
        }
        s->jobs.push_back(std::move(job));
        s->pushed.fetch_add(1, std::memory_order_acq_rel);
        s->pushed.notify_all();
        return true;
    }
    return false;
}
Splice* OpenSplice() {
    auto job = std::make_unique<Splice>();
    job->work = Work::Draw;
    job->open_serial = OrderSerial();
    g_open_splices.fetch_add(1, std::memory_order_acq_rel);
    {
        std::scoped_lock lock{open_splices_mutex};
        open_splices.push_back(job.get());
    }
    Splice* p = job.get();
    Enqueue(std::move(job));
    return p;
}
void SinkTo(Splice* splice) {
    t_sink = splice;
}
void CloseSplice(Splice* splice) {
    {
        std::scoped_lock lock{open_splices_mutex};
        std::erase(open_splices, splice);
    }
    g_open_splices.fetch_sub(1, std::memory_order_acq_rel);
    {
        std::scoped_lock splice_lock{splice->m}; // (a heap mover job pushed under this lock is counted before the close)
        splice->closed.store(1, std::memory_order_release);
        splice->pushed.fetch_or(0x80000000u, std::memory_order_acq_rel);
    }
    splice->pushed.notify_all();
    if (splice->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete splice;
    }
}
void EnqueueImpl(std::unique_ptr<Job> job) {
    job->transitions.swap(pending_transitions);
    pending_transitions.clear();
    c_jobs.fetch_add(1, std::memory_order_relaxed);
    if (t_mover) { // issue 130: the heap mover's copies ahead of every draw made after its swap
        if (PushToOldestSplice(job)) {
            c_mover_into_splice.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        c_mover_into_stream.fetch_add(1, std::memory_order_relaxed);
    }
    if (t_sink != nullptr) { // P1: a pass made on another thread (P2: the resolver takes it now)
        {
            std::scoped_lock lock{t_sink->m};
            t_sink->jobs.push_back(std::move(job));
        }
        t_sink->pushed.fetch_add(1, std::memory_order_acq_rel);
        t_sink->pushed.notify_all();
        return;
    }
    if (!RecorderOn()) {
        RunJob(*job);
        return;
    }
    static std::once_flag started;
    std::call_once(started, [] { std::thread{RecorderThread}.detach(); });
    if (queue_size.load(std::memory_order_relaxed) >= MaxJobs) {
        c_job_waits_full.fetch_add(1, std::memory_order_relaxed);
        while (queue_size.load(std::memory_order_relaxed) >= MaxJobs) {
            std::this_thread::yield();
        }
    }
    QueuePush(job.release());
    const u64 size = queue_size.fetch_add(1, std::memory_order_seq_cst) + 1;
    if (size > c_jobs_most.load(std::memory_order_relaxed)) {
        c_jobs_most.store(size, std::memory_order_relaxed);
    }
    if (recorder_sleeping.load(std::memory_order_seq_cst)) {
        recorder_sleeping.store(false, std::memory_order_seq_cst);
        recorder_sleeping.notify_one();
    }
}

void Drain() {
    if (!RecorderOn() || on_recorder) {
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const u64 gen = recorder_idle_gen.load(std::memory_order_seq_cst);
        if (queue_size.load(std::memory_order_seq_cst) == 0 && !job_running.load(std::memory_order_seq_cst)) {
            break;
        }
        recorder_idle_gen.wait(gen, std::memory_order_seq_cst);
    }
    if (Recorders() > 1) { // R4a: and the submissions the resolver closed are on the queue
        const u64 target = closed_value.load(std::memory_order_acquire);
        for (u64 last = last_value.load(std::memory_order_acquire); last < target; last = last_value.load(std::memory_order_acquire)) {
            last_value.wait(last, std::memory_order_acquire);
        }
    }
    c_drains.fetch_add(1, std::memory_order_relaxed);
    c_drain_us.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
}

void Wait(u64 value) {
    if (instance == nullptr || value == 0) {
        return;
    }
    const vk::Device device = instance->GetDevice();
    const auto t0 = std::chrono::steady_clock::now();
    while (device.getSemaphoreCounterValue(timeline).value < value) {
        const vk::SemaphoreWaitInfo wait{.semaphoreCount = 1, .pSemaphores = &timeline, .pValues = &value};
        if (device.waitSemaphores(wait, 100'000'000ull) == vk::Result::eErrorDeviceLost) {
            LOG_ERROR(Render_Vulkan, "vkb: device lost while waiting for the frame stream");
            return;
        }
    }
    const u64 us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    const u64 ms = us / 1000;
    if (ms > c_wait_ms_max.load(std::memory_order_relaxed)) {
        c_wait_ms_max.store(ms, std::memory_order_relaxed);
    }
    c_waits.fetch_add(1, std::memory_order_relaxed);
    c_wait_us_total.fetch_add(us, std::memory_order_relaxed);
}

std::atomic<u64> g_ring_generation{0};
u64 HostRingGeneration() {
    return g_ring_generation.load(std::memory_order_acquire);
}
Slice HostRing(u64 size, u64 alignment) {
    std::scoped_lock lock{ring_mutex};
    alignment = std::max<u64>(alignment, 16);
    u64 at = (ring_head + alignment - 1) & ~(alignment - 1);
    if (at + size > RingBytes) {
        at = 0;
        c_ring_wraps.fetch_add(1, std::memory_order_relaxed);
        g_ring_generation.fetch_add(1, std::memory_order_release); // R3c: per-thread chunks taken before the wrap are not used further
        Wait(last_value.load(std::memory_order_relaxed)); // V4f: frames may still be on the GPU; the ring's start is reused only after they are done (a wrap is rare)
    }
    frame_ring_bytes += size;
    if (frame_ring_bytes > RingBytes) {
        c_ring_overrun.fetch_add(1, std::memory_order_relaxed); // one frame used more than the ring: its own earlier data is overwritten (counted)
    }
    ring_head = at + size;
    c_ring_bytes.fetch_add(size, std::memory_order_relaxed);
    return {vk::Buffer{ring_buffer.buffer}, at, static_cast<u8*>(ring_buffer.mapped) + at};
}

vk::Buffer HostRingBuffer() {
    return vk::Buffer{ring_buffer.buffer};
}
vk::Buffer GdsBuffer() {
    return vk::Buffer{gds_buffer.buffer};
}
vk::Buffer HostCacheArena() {
    return vk::Buffer{cache_arena.buffer};
}
vk::Buffer HostScratch() {
    return vk::Buffer{scratch_buffer.buffer};
}
vk::Buffer NullBuffer() {
    return vk::Buffer{null_buffer.buffer};
}
vk::ImageView NullImageView(bool) {
    return null_view;
}

void NoteBytes(Work work, u64 bytes) {
    const u32 k = static_cast<u32>(work);
    if (k < WorkKinds) {
        work_bytes[k].fetch_add(bytes, std::memory_order_relaxed);
        work_jobs[k].fetch_add(1, std::memory_order_relaxed);
    }
}

u32 Stamp(vk::CommandBuffer cmd) {
    return StampsLean() ? ~0u : StampLocked(cmd);
}
bool StampsForcedFull() {
    // R5 (note 31): lean by default; the detailed meters (per kind, per pass, per dispatch) cost about 0.9 ms of GPU a frame (+9 fps without them).
    // TLG_VKB_STAMPS=1 (or TLG_VKB_PASS_STATS=1 / TLG_VKB_PASS_DUMP=1, which need them) turns them on for the whole run.
    static const bool on = [] {
        for (const char* name : {"TLG_VKB_STAMPS", "TLG_VKB_PASS_STATS", "TLG_VKB_PASS_DUMP"}) {
            if (const char* e = std::getenv(name); e != nullptr && e[0] == '1') {
                return true;
            }
        }
        return false;
    }();
    return on;
}
bool StampsLean() {
    // note 87: decided per command buffer at its begin (full with TLG_VKB_STAMPS=1, and while debugger mode records)
    return t_rec->lean;
}

void NoteInPlaceWrite(vk::Buffer buffer, u64 offset, u64 size) {
    if (StageOn()) {
        std::scoped_lock written_lock{frame_written_mutex};
        Note(frame_written, BufferUse(buffer, offset, size, true), 4);
    }
}

void NewFrame() {
    OverlapCensus::frame_now.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    t_rec->stage_map.clear();
    t_rec->stage_regions.clear();
    std::scoped_lock written_lock{frame_written_mutex};
    frame_written.clear();
}

void FirstAnswer(u64 address, u32 pairs) {
    std::scoped_lock lock{blocks_mutex};
    QueryBlock& block = query_blocks[address];
    block.pairs = pairs;
    if (!block.answered) {
        block.answered = true;
        WriteAnswerLocked(address, true); // nothing measured for this block yet: "visible" until the first real answer
        c_q_first.fetch_add(1, std::memory_order_relaxed);
    }
}

void QueryBegin(vk::CommandBuffer cmd, u64 address, bool chained) {
    if (!occlusion_pool) {
        return;
    }
    if (t_rec->query_open) {
        c_q_nested.fetch_add(1, std::memory_order_relaxed); // (the game does not nest them) the new block keeps its last answer
        return;
    }
    u32 index = 0;
    {
        std::scoped_lock blocks_lock{blocks_mutex}; // R4a: the slots are shared by the recording threads and the poll
        index = query_next;
        query_next = (query_next + 1) % QuerySlots;
        if (query_slots[index].pending) {
            c_q_dropped.fetch_add(1, std::memory_order_relaxed);
        }
        query_slots[index] = {address, 0, false, false, chained};
    }
    cmd.resetQueryPool(occlusion_pool, index, 1);
    cmd.beginQuery(occlusion_pool, index, {});
    t_rec->query_open = true;
    t_rec->query_open_index = index;
    t_rec->query_open_address = address;
    if (!chained) {
        c_q_begun.fetch_add(1, std::memory_order_relaxed);
    }
}

void QueryCut(vk::CommandBuffer cmd) {
    if (!t_rec->query_open) {
        return;
    }
    cmd.endQuery(occlusion_pool, t_rec->query_open_index);
    {
        std::scoped_lock blocks_lock{blocks_mutex};
        QuerySlot& slot = query_slots[t_rec->query_open_index];
        slot.pending = true;
        slot.cut = true;
        slot.value = RecValue(); // this command buffer's
    }
    t_rec->query_open = false;
    c_q_carried.fetch_add(1, std::memory_order_relaxed);
}

void QueryEnd(vk::CommandBuffer cmd, u64 address) {
    if (!t_rec->query_open || t_rec->query_open_address != address) {
        c_q_unmatched_end.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    cmd.endQuery(occlusion_pool, t_rec->query_open_index);
    {
        std::scoped_lock blocks_lock{blocks_mutex};
        query_slots[t_rec->query_open_index].pending = true;
        query_slots[t_rec->query_open_index].value = RecValue(); // this command buffer's
    }
    t_rec->query_open = false;
    c_q_ended.fetch_add(1, std::memory_order_relaxed);
}

void PollQueries() {
    if (!occlusion_pool) {
        return;
    }
    std::scoped_lock lock{mutex};
    const u64 done = CompletedValue();
    std::scoped_lock blocks_lock{blocks_mutex};
    // in issue order: from the oldest slot (the next one to be handed out) around the ring, so a newer answer for the same block is written last
    // (api-contract review B4: index order wrote an older answer over a newer one across the wrap)
    const u32 oldest = query_next;
    for (u32 k = 0; k < QuerySlots; ++k) {
        const u32 index = (oldest + k) % QuerySlots;
        QuerySlot& slot = query_slots[index];
        if (!slot.pending || slot.value > done) {
            continue;
        }
        std::array<u64, 2> result{}; // the sample count, its availability
        const vk::Result status = instance->GetDevice().getQueryPoolResults(occlusion_pool, index, 1, sizeof(result), result.data(), sizeof(result),
                                                                           vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        if (status != vk::Result::eSuccess || result[1] == 0) {
            continue;
        }
        slot.pending = false;
        bool seen = result[0] != 0;
        if (slot.chained) { // D4-10: the earlier parts of this interval (the GPU finishes submissions in order, and this loop goes in issue order)
            if (const auto it = query_chain_seen.find(slot.address); it != query_chain_seen.end()) {
                seen = seen || it->second;
                query_chain_seen.erase(it);
            }
            c_q_chained_answers.fetch_add(1, std::memory_order_relaxed);
        }
        if (slot.cut) {
            query_chain_seen[slot.address] = seen; // the interval goes on in a chained query: no answer yet
            continue;
        }
        (seen ? c_q_seen : c_q_hidden).fetch_add(1, std::memory_order_relaxed);
        WriteAnswerLocked(slot.address, seen);
    }
}

bool StageRead(vk::Buffer& buffer, u64& offset, u64 size) {
    if (!StageOn() || !stage_buffer.buffer || size == 0) {
        return false;
    }
    if (size > StageMaxRange) {
        c_stage_too_big.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const Use u = BufferUse(buffer, offset, size, false);
    if (Touches(t_rec->cmd_written, u) != 0) { // (a write in an earlier submission is before the copy in queue order)
        c_stage_written.fetch_add(1, std::memory_order_relaxed); // our GPU work wrote it earlier in this submission: the copy in front would miss that
        return false;
    }
    const StageKey key{u.handle, offset, size};
    const bool merge = StageMergeOn();
    if (!merge) {
        if (const auto it = t_rec->stage_map.find(key); it != t_rec->stage_map.end()) {
            buffer = vk::Buffer{stage_buffer.buffer};
            offset = it->second;
            c_stage_reused.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    // issue 111: what is copied. Without merging: the read itself. With it: from the read's start rounded down to 256 bytes (the ring offset keeps the read's
    // offset modulo 256, so every alignment the draw met still holds), widened over the regions of this submission it overlaps; a region that holds the read
    // already is used as it is. Before: draws reading the same buffer with a different end (an index range up to the draw's last index) copied it once each.
    u64 copy_from = offset, copy_size = size;
    bool keep = false; // the copy is kept as a region of this submission
    if (merge) {
        auto& regions = t_rec->stage_regions;
        u64 lo = offset & ~255ull, hi = offset + size;
        auto next = regions.upper_bound({u.handle, offset});
        auto first = next;
        if (next != regions.begin()) {
            const auto before = std::prev(next);
            if (before->first.first == u.handle && before->second.first > lo) {
                if (before->second.first >= offset + size) {
                    buffer = vk::Buffer{stage_buffer.buffer};
                    offset = before->second.second + (offset - before->first.second);
                    c_stage_covered.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                first = before;
                lo = std::min(lo, before->first.second);
            }
        }
        auto last = next;
        while (last != regions.end() && last->first.first == u.handle && last->first.second < hi) {
            hi = std::max(hi, last->second.first);
            ++last;
        }
        if (hi - lo <= StageMaxRange) {
            if (first != last) {
                c_stage_merged.fetch_add(1, std::memory_order_relaxed);
            }
            copy_from = lo, copy_size = hi - lo, keep = true;
            regions.erase(first, last); // the union takes their place (draws recorded before keep their own copies)
        } else {
            c_stage_merge_too_big.fetch_add(1, std::memory_order_relaxed); // the union would pass the size limit: the read alone, not kept as a region
        }
    }
    size = copy_size; // the ring space and the copy below are for the copied range
    std::unique_lock stage_lock{stage_mutex}; // R4a: the ring, its regions and its stamps are shared by the recording threads
    const u64 mine = RecValue();
    u64 at = (stage_head + 255) & ~255ull;
    if (at + size > StageBytes) {
        at = 0;
        t_rec->stage_map.clear(); // (copies of this frame before the wrap are not reused after it)
        t_rec->stage_regions.clear();
    }
    // the region is free once the submissions that used it are done on the GPU: the ring is used in order, so only its oldest regions can be in the way
    u64 wait_for = 0;
    while (!stage_in_flight.empty()) {
        const StageRegion& r = stage_in_flight.front();
        if (!(at < r.end && r.begin < at + size)) {
            break;
        }
        if (r.value >= mine) {
            c_stage_own_overlap.fetch_add(1, std::memory_order_relaxed); // R4a: this submission's own (or a later one's): never waited for (a wrap within it)
            break;
        }
        wait_for = std::max(wait_for, r.value);
        stage_in_flight.pop_front();
    }
    stage_head = at + size;
    stage_in_flight.push_back({at, at + size, mine}); // reserved now; the wait below runs without the lock (the submission it waits for may be another recording thread's)
    stage_lock.unlock();
    if (wait_for != 0 && CompletedValue() < wait_for) {
        Wait(wait_for);
        c_stage_waits.fetch_add(1, std::memory_order_relaxed);
    }
    EnsureStageCmd();
    t_rec->stage_cmd.copyBuffer(buffer, vk::Buffer{stage_buffer.buffer}, vk::BufferCopy{copy_from, at, size});
    if (keep) {
        t_rec->stage_regions.insert_or_assign({u.handle, copy_from}, std::pair<u64, u64>{copy_from + size, at});
    } else if (!merge) {
        t_rec->stage_map.emplace(key, at);
    }
    if (u64 rh = 0, rb = 0, re = 0; Core::Vkb::FrameRingInPlace(rh, rb, re) && rh == u.handle && copy_from >= rb && copy_from + size <= re) {
        c_stage_ring.fetch_add(1, std::memory_order_relaxed);
        c_stage_ring_bytes.fetch_add(size, std::memory_order_relaxed);
    }
    buffer = vk::Buffer{stage_buffer.buffer};
    offset = at + (offset - copy_from);
    c_staged.fetch_add(1, std::memory_order_relaxed);
    c_staged_bytes.fetch_add(size, std::memory_order_relaxed);
    return true;
}
/// The command buffer in front of the current one (V4h copies, U1 hoisted uploads): made at its first use, with its barrier
void EnsureStageCmd() {
    if (!t_rec->stage_cmd) {
        MakePool();
        const vk::CommandBufferAllocateInfo ai{.commandPool = t_rec->pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
        t_rec->stage_cmd = instance->GetDevice().allocateCommandBuffers(ai).value[0];
        (void)t_rec->stage_cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        const vk::MemoryBarrier2 barrier{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                         .dstStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
                                         .dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
        t_rec->stage_cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        t_rec->stage_stamp_open = ~0u;
        bool stamp = false;
        {
            std::scoped_lock stamp_lock{stage_mutex};
            stamp = stage_stamp_pool && stage_stamped.size() < StageStampPairs;
            if (stamp) {
                t_rec->stage_stamp_open = 2 * (stage_stamp_next++ % StageStampPairs);
            }
        }
        if (stamp) {
            t_rec->stage_cmd.resetQueryPool(stage_stamp_pool, t_rec->stage_stamp_open, 2);
            t_rec->stage_cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, stage_stamp_pool, t_rec->stage_stamp_open);
        }
    }
}

namespace {
u32 StampLocked(vk::CommandBuffer cmd) {
    // mutex held by the caller (Record)
    if (!t_rec->stamping || t_rec->stamp_count >= MaxStamps) {
        return ~0u;
    }
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, stamp_pools[t_rec->generation % StampPools], t_rec->stamp_count);
    t_rec->last_stamp = t_rec->stamp_count;
    return t_rec->stamp_count++;
}
} // namespace

u64 Generation() {
    return t_rec->generation;
}

namespace {
// The frame timeline capture (F9 frame profiler, F8 stall profiler, timeline_report.py): the GPU time of our submissions, as the old PS4 path's
// GpuTimeline wrote it. Our stamps are in the GPU clock; VK_EXT_calibrated_timestamps maps them to the CPU clock of FrameTimeline::Now (QPC on Windows).
// Recorded only while a capture runs, from the stamps FinishedStamps reads anyway.
struct GpuClock {
    u64 gpu_ns = 0, cpu_ns = 0, taken_ns = 0;
    bool valid = false;
};
GpuClock gpu_clock; ///< under the stream lock
void CalibrateGpuClockLocked() {
    const u64 now = FrameTimeline::Now();
    if (gpu_clock.valid && now - gpu_clock.taken_ns < 500'000'000ull) {
        return;
    }
    gpu_clock.taken_ns = now;
#ifdef _WIN32
    if (!instance->IsCalibratedTimestampsSupported()) {
        return;
    }
    static const long long frequency = [] {
        LARGE_INTEGER f{};
        QueryPerformanceFrequency(&f);
        return static_cast<long long>(f.QuadPart);
    }();
    const std::array<vk::CalibratedTimestampInfoEXT, 2> infos = {vk::CalibratedTimestampInfoEXT{.timeDomain = vk::TimeDomainEXT::eDevice},
                                                                 vk::CalibratedTimestampInfoEXT{.timeDomain = vk::TimeDomainEXT::eQueryPerformanceCounter}};
    std::array<u64, 2> stamps{};
    u64 deviation = 0;
    if (frequency <= 0 ||
        instance->GetDevice().getCalibratedTimestampsEXT(static_cast<u32>(infos.size()), infos.data(), stamps.data(), &deviation) != vk::Result::eSuccess) {
        return;
    }
    gpu_clock.gpu_ns = static_cast<u64>(static_cast<double>(stamps[0]) * stamp_period_ns); // the same conversion as the stamps (FinishedStamps)
    gpu_clock.cpu_ns = static_cast<u64>(static_cast<double>(stamps[1]) * 1e9 / static_cast<double>(frequency));
    gpu_clock.valid = true;
    FrameTimeline::Record(FrameTimeline::GpuClockSync, stamps[0], gpu_clock.cpu_ns, static_cast<u64>(stamp_period_ns * 1000.0));
#endif
}
u64 GpuToCpuNs(u64 gpu_ns) {
    return gpu_clock.cpu_ns + static_cast<u64>(static_cast<s64>(gpu_ns - gpu_clock.gpu_ns));
}
/// GPU pass kinds of the timeline (timeline_report.py PASS_KIND): 0x11 depth only, 0x12 color, 0x13 g-buffer, 0x21 dispatch (the full stamps of TLG_VKB_STAMPS=1);
/// 0x30 a whole submission (lean stamps, the default: begin to end, every kind of work), 0x31 the staging copies submitted in front of a submission.
constexpr u64 GpuKindSubmission = 0x30, GpuKindStaging = 0x31;
void TimelineGpuPass(u64 kind, u64 begin, u64 end, u64 detail) {
    if (end >= begin) {
        FrameTimeline::Record(FrameTimeline::GpuPass, (end - begin) | (kind << 56), GpuToCpuNs(end), detail);
    }
}
/// note 87: GPU passes in debugger mode (Shift+F5 until the freeze). Every command buffer begun then has full stamps (per pass, per dispatch run; about 0.9 ms of
/// GPU a frame, accepted while recording), so the freeze's timeline holds every pass, and this keeps per submission its GPU time, its slowest passes and its
/// dispatch time: the newest RecentSubmissions and the WorstSubmissions slowest since debugger mode started. The freeze writes them (gpu_passes.json).
/// Off debugger mode nothing here runs: the stamps stay lean.
constexpr size_t RecentSubmissions = 600, WorstSubmissions = 16, PassesKept = 8;
struct GpuPass {
    double ms;
    u8 kind;
    u32 width, height, draws, colors, format, flags;
    u64 ps;
};
struct GpuDraw {
    double ms;
    u64 id, vs, ps;
    u32 count, instances;
};
struct GpuSubmission {
    u64 at_ns, value;
    double ms, dispatch_ms;
    u32 passes, dispatch_runs;
    std::vector<GpuPass> top;
    std::vector<GpuDraw> top_draws; ///< note 91: the slowest draws (debugger mode)
};
std::deque<GpuSubmission> gpu_recent;  ///< newest last (under the stamp lock)
std::vector<GpuSubmission> gpu_worst;  ///< slowest first
u64 gpu_noted = 0;
bool gpu_was_debugging = false;
/// note 90: pixel shaders of slow passes waiting for their dump, and the ones already dumped in this recording (under arm_mutex)
std::mutex arm_mutex;
std::vector<u64> armed_ps, dumped_ps;
std::atomic<u32> armed_count{0};
constexpr double SlowPassArmMs = 20.0;

void NoteGpuPassesLocked(const Submitted& s, const std::vector<u64>& ns) {
    const bool on = Core::FrameWalk::DebuggerMode();
    if (on != gpu_was_debugging) {
        gpu_was_debugging = on;
        if (on) {
            gpu_recent.clear(), gpu_worst.clear(), gpu_noted = 0; // a new recording starts empty
            std::scoped_lock arm_lock{arm_mutex};
            armed_ps.clear(), dumped_ps.clear();
            armed_count.store(0, std::memory_order_relaxed);
        }
    }
    if (!on || s.lean || s.count == 0 || s.segments.empty() || s.count > ns.size()) {
        return;
    }
    const u64 first = ns[s.segments.front().stamp], end = ns[s.count - 1];
    if (end < first) {
        return;
    }
    GpuSubmission rec{.at_ns = SteadyNs(), .value = s.value, .ms = static_cast<double>(end - first) / 1e6, .dispatch_ms = 0, .passes = static_cast<u32>(s.passes.size()),
                      .dispatch_runs = 0, .top = {}};
    for (size_t i = 0; i < s.segments.size(); ++i) {
        if (s.segments[i].work == Work::Dispatch) {
            const u64 a = ns[s.segments[i].stamp], b = i + 1 < s.segments.size() ? ns[s.segments[i + 1].stamp] : end;
            if (b >= a) {
                rec.dispatch_ms += static_cast<double>(b - a) / 1e6;
                ++rec.dispatch_runs;
            }
        }
    }
    for (const PassStamp& p : s.passes) {
        if (p.begin < s.count && p.end < s.count && ns[p.end] >= ns[p.begin]) {
            rec.top.push_back({static_cast<double>(ns[p.end] - ns[p.begin]) / 1e6, p.kind, p.width, p.height, p.draws, p.colors, p.format, p.flags, p.ps});
        }
    }
    std::sort(rec.top.begin(), rec.top.end(), [](const GpuPass& a, const GpuPass& b) { return a.ms > b.ms; });
    for (const GpuPass& p : rec.top) { // note 90: arm the slow passes' pixel shaders for a full draw dump
        if (p.ms >= SlowPassArmMs && p.ps != 0) {
            std::scoped_lock arm_lock{arm_mutex};
            if (std::ranges::find(armed_ps, p.ps) == armed_ps.end() && std::ranges::find(dumped_ps, p.ps) == dumped_ps.end()) {
                armed_ps.push_back(p.ps);
                armed_count.fetch_add(1, std::memory_order_relaxed);
                LOG_WARNING(Render_Vulkan, "vkb: debugger mode: pass of {:.1f} ms, pixel shader {:#x} armed: its next draw dumps its images and buffers", p.ms, p.ps);
            }
        }
    }
    for (const DrawStamp& d : s.draws) { // note 91
        if (d.stamp < s.count && d.prev < s.count && ns[d.stamp] >= ns[d.prev]) {
            const double dms = static_cast<double>(ns[d.stamp] - ns[d.prev]) / 1e6;
            rec.top_draws.push_back({dms, d.id, d.vs, d.ps, d.count, d.instances});
            std::scoped_lock cap_lock{capture_mutex};
            if (!capture_rows.empty() || frame_capture.load(std::memory_order_relaxed)) {
                capture_draw_ms[d.id] = dms;
            }
        }
    }
    std::sort(rec.top_draws.begin(), rec.top_draws.end(), [](const GpuDraw& a, const GpuDraw& b) { return a.ms > b.ms; });
    if (rec.top_draws.size() > 16) {
        rec.top_draws.resize(16);
    }
    if (rec.top.size() > PassesKept) {
        rec.top.resize(PassesKept);
    }
    ++gpu_noted;
    if (gpu_worst.size() < WorstSubmissions || rec.ms > gpu_worst.back().ms) {
        gpu_worst.push_back(rec);
        std::sort(gpu_worst.begin(), gpu_worst.end(), [](const GpuSubmission& a, const GpuSubmission& b) { return a.ms > b.ms; });
        if (gpu_worst.size() > WorstSubmissions) {
            gpu_worst.pop_back();
        }
    }
    gpu_recent.push_back(std::move(rec));
    if (gpu_recent.size() > RecentSubmissions) {
        gpu_recent.pop_front();
    }
}

void TimelineSubmissionLocked(const Submitted& s, const std::vector<u64>& ns) {
    if (!gpu_clock.valid || s.count == 0 || s.segments.empty() || s.count > ns.size()) {
        return;
    }
    const u64 first = ns[s.segments.front().stamp];
    const u64 end = ns[s.count - 1];
    if (end < first) {
        return;
    }
    FrameTimeline::Record(FrameTimeline::GpuCommandBuffer, end - first, GpuToCpuNs(end), s.value);
    if (s.lean) {
        TimelineGpuPass(GpuKindSubmission, first, end, s.value);
        return;
    }
    for (const PassStamp& p : s.passes) {
        if (p.begin < s.count && p.end < s.count) {
            TimelineGpuPass(0x11 + p.kind, ns[p.begin], ns[p.end], (p.draws & 0xffffffu) | (u64(p.width & 0xfffffu) << 24) | (u64(p.height) << 44));
        }
    }
    for (size_t i = 0; i < s.segments.size(); ++i) {
        if (s.segments[i].work == Work::Dispatch) {
            TimelineGpuPass(0x21, ns[s.segments[i].stamp], i + 1 < s.segments.size() ? ns[s.segments[i + 1].stamp] : end, 0);
        }
    }
}
} // namespace

void SetFrameCapture(bool on) {
    if (on) {
        std::scoped_lock lock{capture_mutex};
        capture_rows = nlohmann::json::array();
        capture_buffer_index.clear();
        capture_draw_ms.clear();
        capture_buffer_bytes = 0;
        capture_passes.store(0);
    }
    frame_capture.store(on, std::memory_order_relaxed);
}
bool FrameCaptureOn() {
    return frame_capture.load(std::memory_order_relaxed);
}
void CaptureDrawQueued() {
    capture_jobs_left.fetch_add(1, std::memory_order_relaxed);
}
u64 CaptureJobsLeft() {
    return capture_jobs_left.load(std::memory_order_relaxed);
}

u32 NoteCapturedDraw(nlohmann::json row, std::span<const vk::DescriptorBufferInfo> buffers) {
    if (t_rec->capture_pass == ~0u) {
        t_rec->capture_pass = capture_passes.fetch_add(1, std::memory_order_relaxed);
    }
    nlohmann::json files = nlohmann::json::array();
    {
        std::scoped_lock lock{capture_mutex};
        for (const vk::DescriptorBufferInfo& b : buffers) {
            if (!b.buffer || b.range == VK_WHOLE_SIZE || b.range == 0) {
                files.push_back(nullptr);
                continue;
            }
            const auto key = std::make_tuple(static_cast<VkBuffer>(b.buffer), static_cast<u64>(b.offset), static_cast<u64>(b.range));
            auto it = capture_buffer_index.find(key);
            if (it == capture_buffer_index.end()) {
                if (b.range > CaptureBufferMax || capture_buffer_bytes + b.range > CaptureBufferLimit) {
                    files.push_back("skipped (size limit)");
                    continue;
                }
                const u32 index = static_cast<u32>(capture_buffer_index.size());
                it = capture_buffer_index.emplace(key, index).first;
                capture_buffer_bytes += b.range;
                t_rec->capture_buffers.emplace_back(b.buffer, b.offset, b.range, index);
            }
            files.push_back(fmt::format("buffers/b{:05}_{}B.gpu", it->second, b.range));
        }
        row["pass"] = t_rec->capture_pass;
        row["buffer_files"] = std::move(files);
        capture_rows.push_back(std::move(row));
    }
    capture_jobs_left.fetch_sub(1, std::memory_order_relaxed);
    return t_rec->capture_pass;
}

void StampDraw(u64 id, u64 vs, u64 ps, u32 count, u32 instances) {
    if (!t_rec->stamping || t_rec->lean || t_rec->stamp_count + 8 >= MaxStamps) {
        return;
    }
    const u32 prev = t_rec->last_stamp;
    const u32 stamp = StampLocked(t_rec->current);
    if (stamp != ~0u && prev != ~0u) {
        t_rec->draw_stamps.push_back({stamp, prev, id, vs, ps, count, instances});
    }
}

nlohmann::json CaptureReport() {
    std::scoped_lock lock{capture_mutex};
    nlohmann::json rows = capture_rows;
    for (auto& r : rows) {
        if (const auto it = capture_draw_ms.find(r.value("id", 0ull)); it != capture_draw_ms.end()) {
            r["gpu_ms"] = it->second;
        }
    }
    return {{"note", "frame capture (note 91): every draw of one frame, in the order the game made them; pass = the passNNNN_* files (each attachment after that pass, "
                     "full size, raw texels: vkb_dump_png.py); buffer_files = the ranges the draw read as the GPU had them after its pass (buffers/); gpu_ms = the draw's GPU "
                     "time (from the stamp before it in its command buffer); images = the draw's image views (handles)"},
            {"passes", capture_passes.load()}, {"buffers_written", capture_buffer_index.size()}, {"buffer_mb", static_cast<double>(capture_buffer_bytes) / (1 << 20)},
            {"draws", rows}};
}

bool TakeSlowPsArm(u64 ps_hash) {
    if (armed_count.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    std::scoped_lock lock{arm_mutex};
    const auto it = std::ranges::find(armed_ps, ps_hash);
    if (it == armed_ps.end()) {
        return false;
    }
    armed_ps.erase(it);
    armed_count.fetch_sub(1, std::memory_order_relaxed);
    dumped_ps.push_back(ps_hash);
    return true;
}

nlohmann::json GpuPassReport() {
    std::scoped_lock lock{stamp_mutex};
    const u64 now = SteadyNs();
    const auto one = [&](const GpuSubmission& r) {
        nlohmann::json passes = nlohmann::json::array();
        for (const GpuPass& p : r.top) {
            passes.push_back({{"ms", p.ms}, {"kind", p.kind == 0 ? "depth" : p.kind == 1 ? "color" : "gbuffer"}, {"size", fmt::format("{}x{}", p.width, p.height)},
                              {"draws", p.draws}, {"colors", p.colors}, {"format", p.format}, {"flags", p.flags}, {"ps", fmt::format("{:016x}", p.ps)}});
        }
        nlohmann::json draws = nlohmann::json::array();
        for (const GpuDraw& d : r.top_draws) {
            draws.push_back({{"ms", d.ms}, {"draw_id", d.id}, {"vs", fmt::format("{:016x}", d.vs)}, {"ps", fmt::format("{:016x}", d.ps)}, {"count", d.count},
                             {"instances", d.instances}});
        }
        return nlohmann::json{{"seconds_before_the_report", static_cast<double>(now - r.at_ns) / 1e9}, {"timeline_value", r.value}, {"gpu_ms", r.ms}, {"passes", r.passes},
                              {"dispatch_ms", r.dispatch_ms}, {"dispatch_runs", r.dispatch_runs}, {"slowest_passes", passes}, {"slowest_draws", draws}};
    };
    nlohmann::json worst = nlohmann::json::array(), recent = nlohmann::json::array();
    for (const GpuSubmission& r : gpu_worst) {
        worst.push_back(one(r));
    }
    for (const GpuSubmission& r : gpu_recent) {
        recent.push_back(one(r));
    }
    return {{"note", "debugger mode (Shift+F5 .. freeze): every submission of the frame stream with full GPU stamps; worst = the slowest since debugger mode started, "
                     "recent = the newest (oldest first). Per submission its GPU ms, dispatch time and its slowest passes (kind, size, draws, ps = pixel shader hash; "
                     "flags: 1 depth, 2 stencil, 4 depth test, 8 stencil test, 16 depth bounds, 32 depth write). The timeline of the freeze has every pass as well."},
            {"submissions_noted", gpu_noted}, {"worst", worst}, {"recent", recent}};
}

std::vector<Stamps> FinishedStamps() {
    std::scoped_lock lock{mutex};
    std::vector<Stamps> out;
    if (instance == nullptr) {
        return out;
    }
    const u64 done = instance->GetDevice().getSemaphoreCounterValue(timeline).value;
    const bool capture = FrameTimeline::capture_active.load(std::memory_order_relaxed);
    if (capture) {
        CalibrateGpuClockLocked();
    }
    std::unique_lock stage_lock{stage_mutex};
    while (!stage_stamped.empty() && stage_stamped.front().value <= done) {
        std::array<u64, 2> t{};
        if (instance->GetDevice().getQueryPoolResults(stage_stamp_pool, stage_stamped.front().index, 2, sizeof(t), t.data(), sizeof(u64), vk::QueryResultFlagBits::e64) ==
                vk::Result::eSuccess &&
            t[1] >= t[0]) {
            stage_gpu_ns_total += static_cast<u64>(static_cast<double>(t[1] - t[0]) * stamp_period_ns);
            ++stage_gpu_timed;
            if (capture && gpu_clock.valid) {
                TimelineGpuPass(GpuKindStaging, static_cast<u64>(static_cast<double>(t[0]) * stamp_period_ns), static_cast<u64>(static_cast<double>(t[1]) * stamp_period_ns),
                                stage_stamped.front().value);
            }
        }
        stage_stamped.pop_front();
    }
    stage_lock.unlock();
    std::scoped_lock stamp_lock{stamp_mutex};
    while (!stamped.empty() && stamped.front().value <= done) {
        const Submitted& s = stamped.front();
        Stamps& r = out.emplace_back();
        r.generation = s.generation;
        r.ns.resize(s.count);
        (void)instance->GetDevice().getQueryPoolResults(stamp_pools[s.pool], 0, s.count, r.ns.size() * sizeof(u64), r.ns.data(), sizeof(u64), vk::QueryResultFlagBits::e64);
        for (u64& t : r.ns) {
            t = static_cast<u64>(static_cast<double>(t) * stamp_period_ns);
        }
        if (capture) {
            TimelineSubmissionLocked(s, r.ns);
        }
        NoteGpuPassesLocked(s, r.ns);
        static const bool dump = [] { const char* e = std::getenv("TLG_VKB_PASS_DUMP"); return e != nullptr && e[0] == '1'; }();
        static u64 dump_seen = 0;
        static u32 dump_left = 6; // the submissions of about one frame, after 9,000 timed ones (gameplay)
        static FILE* dump_file = nullptr;
        if (dump && ++dump_seen > 9000 && dump_left != 0) {
            if (dump_file == nullptr) {
                dump_file = std::fopen((Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_pass_dump.txt").string().c_str(), "w");
            }
            if (dump_file != nullptr) {
                std::fprintf(dump_file, "submission (timeline value %llu): %zu passes\n", static_cast<unsigned long long>(s.value), s.passes.size());
                for (const PassStamp& p : s.passes) {
                    const double ms = p.begin < s.count && p.end < s.count && r.ns[p.end] >= r.ns[p.begin] ? static_cast<double>(r.ns[p.end] - r.ns[p.begin]) / 1e6 : -1.0;
                    std::array<u64, 3> st{};
                    if (p.stat != ~0u && stat_pools[s.pool]) {
                        (void)instance->GetDevice().getQueryPoolResults(stat_pools[s.pool], p.stat, 1, sizeof(st), st.data(), sizeof(st), vk::QueryResultFlagBits::e64);
                    }
                    std::fprintf(dump_file, "    counts: vs %llu  prims %llu  fs %llu\n", static_cast<unsigned long long>(st[0]), static_cast<unsigned long long>(st[1]),
                                 static_cast<unsigned long long>(st[2]));
                    std::fprintf(dump_file, "  %-7s %4ux%-4u colors %u fmt %3u  %7.3f ms  draws %4u  flags%s%s%s%s%s%s  ps %016llx  vp %.0f,%.0f %.0fx%.0f  sc %d,%d %ux%u\n", p.kind == 0 ? "depth" : p.kind == 1 ? "color" : "gbuffer",
                                 p.width, p.height, p.colors, p.format, ms, p.draws, (p.flags & 1) ? " D" : "", (p.flags & 2) ? " S" : "", (p.flags & 4) ? " dtest" : "",
                                 (p.flags & 8) ? " stest" : "", (p.flags & 16) ? " dbounds" : "", (p.flags & 32) ? " dwrite" : "", static_cast<unsigned long long>(p.ps),
                                 p.viewport.x, p.viewport.y, p.viewport.width, p.viewport.height, p.scissor.offset.x, p.scissor.offset.y, p.scissor.extent.width,
                                 p.scissor.extent.height);
                }
                if (--dump_left == 0) {
                    std::fclose(dump_file);
                    dump_file = nullptr;
                }
            }
        }
        for (const PassStamp& p : s.passes) {
            if (p.begin < s.count && p.end < s.count && r.ns[p.end] >= r.ns[p.begin]) {
                PassTotal& t = pass_totals[PassKey{p.kind, p.width, p.height}];
                t.ns += r.ns[p.end] - r.ns[p.begin];
                ++t.passes;
                t.draws += p.draws;
                std::array<u64, 3> st{};
                if (p.stat != ~0u && stat_pools[s.pool] &&
                    instance->GetDevice().getQueryPoolResults(stat_pools[s.pool], p.stat, 1, sizeof(st), st.data(), sizeof(st), vk::QueryResultFlagBits::e64) == vk::Result::eSuccess) {
                    t.vs += st[0], t.prims += st[1], t.fs += st[2];
                    ++t.counted;
                }
            }
        }
        if (!s.segments.empty() && s.count != 0) {
            // GPU time per kind of work: from each kind change to the next (the last to the stamp at the submit)
            const u64 end = r.ns[s.count - 1];
            const u64 first = r.ns[s.segments.front().stamp];
            for (size_t i = 0; i < s.segments.size(); ++i) {
                const u64 a = r.ns[s.segments[i].stamp];
                const u64 b = i + 1 < s.segments.size() ? r.ns[s.segments[i + 1].stamp] : end;
                if (b >= a) {
                    work_ns[static_cast<u32>(s.segments[i].work)] += b - a;
                    work_window_ns[static_cast<u32>(s.segments[i].work)] += b - a;
                }
            }
            if (work_window_first_ns == 0 || first < work_window_first_ns) {
                work_window_first_ns = first;
            }
            work_window_last_ns = std::max(work_window_last_ns, end);
            if (end >= first) {
                work_total_ns += end - first;
                work_window_total_ns += end - first;
                work_window_max_ns = std::max(work_window_max_ns, end - first);
            }
            ++work_submits;
            if (++work_window_submits == WorkWindow) {
                work_last_window_ns = work_window_ns;
                work_last_window_total_ns = work_window_total_ns;
                work_last_window_max_ns = work_window_max_ns;
                work_last_window_span_ns = work_window_last_ns > work_window_first_ns ? work_window_last_ns - work_window_first_ns : 0;
                work_window_first_ns = work_window_last_ns = 0;
                work_window_ns = {};
                work_window_total_ns = work_window_max_ns = work_window_submits = 0;
            }
        }
        pool_busy[s.pool] = false;
        stamped.pop_front();
    }
    return out;
}

nlohmann::json Report() {
    nlohmann::json work = nlohmann::json::object();
    {
        std::scoped_lock lock{mutex};
        nlohmann::json all = nlohmann::json::object(), last = nlohmann::json::object();
        for (u32 k = 0; k < WorkKinds; ++k) {
            all[WorkNames[k]] = work_submits ? static_cast<double>(work_ns[k]) / 1e6 / static_cast<double>(work_submits) : 0.0;
            last[WorkNames[k]] = static_cast<double>(work_last_window_ns[k]) / 1e6 / static_cast<double>(WorkWindow);
        }
        double all_copies = 0, all_uploads = 0, last_copies = 0, last_uploads = 0;
        nlohmann::json bytes = nlohmann::json::object();
        for (u32 k = 0; k < WorkKinds; ++k) {
            const Work w = static_cast<Work>(k);
            if (IsCopyKind(w)) {
                all_copies += all[WorkNames[k]].get<double>(), last_copies += last[WorkNames[k]].get<double>();
            } else if (IsUploadKind(w)) {
                all_uploads += all[WorkNames[k]].get<double>(), last_uploads += last[WorkNames[k]].get<double>();
            }
            if (work_jobs[k].load() != 0) {
                bytes[WorkNames[k]] = {{"jobs", work_jobs[k].load()}, {"mb", work_bytes[k].load() >> 20}};
            }
        }
        all["copies"] = all_copies, all["uploads"] = all_uploads, last["copies"] = last_copies, last["uploads"] = last_uploads; // the sums (older scripts read these)
        work["bytes_by_kind"] = bytes;
        all["total"] = work_submits ? static_cast<double>(work_total_ns) / 1e6 / static_cast<double>(work_submits) : 0.0;
        last["total"] = static_cast<double>(work_last_window_total_ns) / 1e6 / static_cast<double>(WorkWindow);
        last["worst_submission"] = static_cast<double>(work_last_window_max_ns) / 1e6;
        last["gpu_busy_pct_of_the_window"] = work_last_window_span_ns ? 100.0 * static_cast<double>(work_last_window_total_ns) / static_cast<double>(work_last_window_span_ns) : 0.0;
        last["window_ms"] = static_cast<double>(work_last_window_span_ns) / 1e6;
        work["note"] = "GPU ms per timed submission of the frame stream, by kind of work (timestamps where the kind changes); last_window: the newest 500 finished submissions";
        work["stamps"] = StampsForcedFull() ? "full (TLG_VKB_STAMPS=1)" : Core::FrameWalk::DebuggerMode() ? "full (debugger mode)" : "lean: the total and busy share are right; the whole submission counts as its first kind (TLG_VKB_STAMPS=1 or debugger mode for the split by kind)";
        work["timed_submissions"] = work_submits, work["whole_run"] = all, work["last_window"] = last;
    }
    return {{"occlusion", {{"begun", c_q_begun.load()}, {"ended", c_q_ended.load()}, {"seen", c_q_seen.load()}, {"hidden", c_q_hidden.load()},
                           {"first_answers_visible", c_q_first.load()}, {"nested_ignored", c_q_nested.load()}, {"dropped_unanswered", c_q_dropped.load()},
                           {"ended_by_a_submit", c_q_cut_by_submit.load()}, {"end_without_begin", c_q_unmatched_end.load()},
                           {"carried_across_a_submit", c_q_carried.load()}, {"chained_answers", c_q_chained_answers.load()},
                           {"carry_by_submit", {{"other", c_q_carried_by[0].load()}, {"graphics", c_q_carried_by[1].load()}, {"async_compute", c_q_carried_by[2].load()}}}}},
            {"copies_in_front_of_a_job", {{"copies", c_copies_first.load()}, {"mb", c_copies_first_mb.load() >> 20}, {"recorded_in_a_run", c_copies_first_in_run.load()}, {"as_a_job_of_their_own", c_copies_first_own_job.load()}}},
            {"staging", {{"on", StageOn()}, {"copies", c_staged.load()}, {"mb_copied", c_staged_bytes.load() >> 20}, {"reused_in_a_submission", c_stage_reused.load()}, {"merge", StageMergeOn()}, {"reads_inside_a_staged_region", c_stage_covered.load()},
                          {"copies_that_took_in_overlapping_regions", c_stage_merged.load()}, {"unions_over_the_size_limit", c_stage_merge_too_big.load()}, {"copies_from_the_frame_ring", c_stage_ring.load()},
                         {"mb_from_the_frame_ring", c_stage_ring_bytes.load() >> 20},
                         {"left_in_place_gpu_written", c_stage_written.load()}, {"left_in_place_too_big", c_stage_too_big.load()}, {"waits_for_the_ring", c_stage_waits.load()}}},
            {"recorder", {{"on", RecorderOn()}, {"jobs", c_jobs.load()},
                          {"time_ms_total", [] {
                               static constexpr const char* names[] = {"draw", "dispatch", "copy", "clear", "other", "upload", "copy_buffer", "copy_gds", "fill", "upload_initial", "upload_cpu_written", "submit"};
                               nlohmann::json t = nlohmann::json::object();
                               for (u32 i = 0; i <= static_cast<u32>(Work::Count); ++i) {
                                   t[names[i]] = {c_rec_run_ns[i].load() / 1000000, c_rec_run_n[i].load()};
                               }
                               t["asleep"] = {c_rec_sleep_ns.load() / 1000000, c_rec_sleeps.load()};
                               t["spin_hits"] = c_rec_spin_hits.load();
                               t["enqueue_on_producers"] = c_enqueue_ns.load() / 1000000;
                               return t;
                           }()}, {"most_queued", c_jobs_most.load()}, {"producer_waits_queue_full", c_job_waits_full.load()},
                          {"overlap_census", OverlapCensus::On() ? OverlapCensus::Report() : nlohmann::json{}}, {"hoisted_uploads", {{"on", HoistOn()}, {"hoisted", c_hoisted.load()}, {"mb", c_hoisted_bytes.load() >> 20}, {"in_place", c_hoist_in_place.load()}}}, {"drains", c_drains.load()}, {"splices", c_splices.load()}, {"splice_jobs", c_splice_jobs.load()}, {"splice_waits", c_splice_waits.load()}, {"splice_wait_us", c_splice_wait_us.load()}, {"split_submits", c_splits.load()},
                          {"r4a_recorders_submissions_jobs_transition_jobs_worker_waits_batches", {Recorders(), c_r4a_submissions.load(), c_r4a_jobs.load(), c_r4a_transition_jobs.load(), c_r4a_worker_waits.load(), c_r4a_batches.load()}},
                          {"stage_own_overlap", c_stage_own_overlap.load()},
                          {"drain_wait_ms_mean", c_drains.load() ? static_cast<double>(c_drain_us.load()) / 1e3 / static_cast<double>(c_drains.load()) : 0.0}}},
            {"gpu_ms_by_work", work},
            {"gpu_by_pass", [] {
                 std::scoped_lock lock{mutex};
                 nlohmann::json out = nlohmann::json::array();
                 for (const auto& [k, t] : pass_totals) {
                     out.push_back({{"kind", k.kind == 0 ? "depth" : k.kind == 1 ? "color" : "gbuffer"}, {"width", k.width}, {"height", k.height}, {"ms_total", static_cast<double>(t.ns) / 1e6},
                                    {"passes", t.passes}, {"draws", t.draws}, {"counted", t.counted}, {"vs", t.vs}, {"prims", t.prims}, {"fs", t.fs}});
                 }
                 return out;
             }()},
            {"staging_gpu", {{"note", "the staging command buffer's GPU time (after its barrier): not part of gpu_ms_by_work"},
                             {"ms_total", static_cast<double>(stage_gpu_ns_total) / 1e6}, {"timed_submissions", stage_gpu_timed}}},
            {"barrier_causes", [] {
                 nlohmann::json j = nlohmann::json::object();
                 const char* kinds[6] = {"draw", "dispatch", "copy", "clear", "other", "none"};
                 const char* hows[3] = {"read_after_write", "write_after_write", "write_after_read"};
                 const char* whats[4] = {"gds", "null_buffer", "buffer", "image"};
                 for (u32 k = 0; k < 6; ++k)
                     for (u32 h = 0; h < 3; ++h)
                         for (u32 w = 0; w < 4; ++w)
                             if (const u64 n = c_hazard_cause[k][h][w].load(); n != 0) {
                                 j[std::string(kinds[k]) + "." + hows[h] + "." + whats[w]] = n;
                             }
                 return j;
             }()},
            {"barriers_by_shader", [] {
                 nlohmann::json j = nlohmann::json::array();
                 if (!BarrierByShaderOn()) {
                     return j;
                 }
                 std::scoped_lock lock{bbs_mutex};
                 for (const auto& [tag, s] : bbs) {
                     j.push_back({{"shader", fmt::format("{:#x}", tag)}, {"checked", s.checked}, {"barriers", s.barriers}, {"raw", s.how[0]}, {"waw", s.how[1]}, {"war", s.how[2]}});
                 }
                 return j;
             }()},
            {"barriers_by_what_they_waited_for", [] {
                 nlohmann::json j = nlohmann::json::object();
                 for (u32 k = 1; k < 8; ++k) {
                     j[std::string((k & 1) ? "graphics" : "") + ((k & 2) ? "+compute" : "") + ((k & 4) ? "+transfer" : "")] = c_barrier_src[k].load();
                 }
                 return j;
             }()},
            {"barriers", {{"legacy_rule", LegacyBarriers()}, {"full_barriers", FullBarriers()}, {"commands_checked", c_hazard_checks.load()}, {"barriers_for_a_hazard", c_hazard_barriers.load()}, {"lantern_waits_skipped_overlap_unknown", {c_quiet_skipped.load(), c_quiet_overlap.load(), c_quiet_unknown.load()}},
                          {"barriers_at_command_buffer_begin", c_begin_barriers.load()}}},
            {"submits", c_submits.load()},
            {"submits_with_nothing_recorded", c_empty_submits.load()},
            {"render_passes", c_passes.load()},
            {"pass_ends", [] {
                 static const char* names[] = {"break", "barrier", "other_depth", "other_size", "other_colors", "submit", "transition", "job_draw", "job_dispatch", "job_copy",
                                               "job_clear", "job_other", "job_upload", "job_copy_buffer", "job_copy_gds", "job_fill", "job_upload_initial", "job_upload_unmap", "job_none"};
                 nlohmann::json j, shadow;
                 for (u32 k = 0; k < EndWhy::Count; ++k) {
                     if (const u64 v = c_pass_ends[k][0].load()) {
                         j[names[k]] = v;
                     }
                     if (const u64 v = c_pass_ends[k][1].load()) {
                         shadow[names[k]] = v;
                     }
                 }
                 return nlohmann::json{{"all", j}, {"shadow_2048_depth_only", shadow}};
             }()},
            {"waits", c_waits.load()},
            {"longest_wait_ms", c_wait_ms_max.load()},
            {"wait_ms_mean", c_waits.load() ? static_cast<double>(c_wait_us_total.load()) / 1e3 / static_cast<double>(c_waits.load()) : 0.0},
            {"host_ring_mb_used", c_ring_bytes.load() >> 20},
            {"host_ring_wraps", c_ring_wraps.load()},
            {"host_ring_overruns", c_ring_overrun.load()}};
}

} // namespace Core::Vkb::Stream
