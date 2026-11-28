// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <span>
#include <atomic>
#include <vector>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_common.h"

namespace VkbVulkan {
class Instance;
}

/// V3b of the engine-to-Vulkan backend: the frame's command stream. The engine's work calls record into one command buffer (in the order the game makes them); a graphics submit
/// submits it to the graphics queue with the next value of our timeline. What must wait for the GPU (the queue's fences and interrupts, the flip) runs on the submitting thread
/// once the timeline has passed the frame's value.
namespace Core::Vkb::Stream {

void Init(const VkbVulkan::Instance& instance);
[[nodiscard]] bool Ready();

/// What a recording does, for the GPU time per kind of work (diagnostic: a timestamp where the kind changes)
enum class Work : u8 { Draw, Dispatch, Copy, Clear, Other, Upload, CopyBuffer, CopyGds, Fill, UploadInitial, UploadCpuWritten, Count };
/// R5 itemized copies: Copy = the engine's texture copies, CopyBuffer / CopyGds / Fill = its buffer copies, GDS reads and pattern fills; Upload = our device copy at a static
/// buffer's first bind, UploadInitial = a new buffer's initial data, UploadCpuWritten = the Unmap / UpdateSubresource copies.
/// The bytes each kind moves (CPU side, counted where the job is made).
void NoteBytes(Work work, u64 bytes); ///< Upload (R5 meter): our own copies of engine data into our buffers (first bind, initial data, Unmap / UpdateSubresource)

/// V4g: an image's move to the general layout, decided where the engine's call is resolved and recorded in front of the next job of that thread
struct Transition {
    vk::Image image{};
    vk::ImageLayout old_layout{};
    vk::ImageAspectFlags aspect{};
};
void PendTransition(vk::Image image, vk::ImageLayout old_layout, vk::ImageAspectFlags aspect);

/// V4g: the recording thread. An engine call is resolved where it is made (its engine state read, our objects found) into a job; the job's Vulkan commands are recorded on
/// our recording thread, in the order the jobs were made (the stream lock held, the command buffer begun; a job that is not a draw ends an open pass). An unlocked job (a submit)
/// runs without the lock and gets no command buffer. `TLG_VKB_RECORDER=0` (or `TLG_VKB_OVERLAP=0`) runs every job at once on the calling thread.
struct Job {
    virtual ~Job() = default;
    virtual void Run(vk::CommandBuffer cmd) = 0;
    /// R4a (note 65): run in stream order on the resolver before the job goes to a recording thread (a draw's late buffers)
    virtual void Prepare() {}
    Work work{Work::Other};
    bool locked{true};
    s8 query{}; ///< R4a: +1 opens an occlusion query, -1 ends it (our splits never cut a submission inside one; an engine submit inside one carries it on, D4-10)
    u64 query_address{}; ///< D4-10: the query's block (for the continuation in the next submission)
    u32 weight{1}; ///< G2: the jobs it stands for in the submission split (a batch: its draws)
    bool splice{false}; ///< P1 (parallel passes): a Splice placeholder (the resolver waits for its jobs and handles them in its place)
    std::vector<Transition> transitions;
    /// issue 111: buffer copies its Prepare asked for (CopyInStream): recorded in the stream in front of the job
    struct Copy {
        vk::Buffer source{}; ///< none: the destination is filled with zeros
        u64 source_offset{};
        vk::Buffer destination{};
        u64 bytes{};
    };
    std::vector<Copy> copies_first;
    std::atomic<Job*> queue_next{nullptr}; ///< note 60: the lock-free hand-over queue's link
};
/// A2 (note 46; TLG_VKB_JOB_POOL, default on, =0 off): the memory of jobs of one size is reused (a job is made on the engine's thread and freed on the recording thread,
/// thousands a frame; each was a heap block of several KB)
bool JobPoolOn();
/// note 60: changes whenever the recorder begins a command buffer (the draw state cache is per recording)
u64 RecordingGeneration();
template <size_t Size>
struct JobBlocks {
    static inline std::atomic_flag lock = ATOMIC_FLAG_INIT;
    static inline std::vector<void*> free_blocks;
    static void* Get() {
        if (JobPoolOn()) {
            void* p = nullptr;
            while (lock.test_and_set(std::memory_order_acquire)) {
            }
            if (!free_blocks.empty()) {
                p = free_blocks.back();
                free_blocks.pop_back();
            }
            lock.clear(std::memory_order_release);
            if (p != nullptr) {
                return p;
            }
        }
        return ::operator new(Size);
    }
    static void Put(void* p) {
        if (JobPoolOn()) {
            while (lock.test_and_set(std::memory_order_acquire)) {
            }
            const bool kept = free_blocks.size() < 16384;
            if (kept) {
                free_blocks.push_back(p);
            }
            lock.clear(std::memory_order_release);
            if (kept) {
                return;
            }
        }
        ::operator delete(p);
    }
};
template <class F>
struct FnJob final : Job {
    explicit FnJob(F&& f_) : f(std::move(f_)) {}
    static void* operator new(size_t) {
        return JobBlocks<sizeof(FnJob)>::Get();
    }
    static void operator delete(void* p) {
        JobBlocks<sizeof(FnJob)>::Put(p);
    }
    void Run(vk::CommandBuffer cmd) override {
        f(cmd);
    }
    F f;
};
void Enqueue(std::unique_ptr<Job> job);
/// P1 (parallel passes, TLG_VKB_PAR_ZPASS): a placeholder the engine's thread puts into the stream where a pass belongs; another thread makes the pass's jobs into it
/// (SinkTo), then Close. The resolver waits at the placeholder until it is closed and handles its jobs there, so the GPU order stays the API order.
struct Splice final : Job {
    Splice() {
        splice = true;
    }
    void Run(vk::CommandBuffer) override {}
    std::mutex m;
    std::vector<std::unique_ptr<Job>> jobs; ///< (under m) the filling thread appends; the resolver takes them in order as they come (P2: streamed, not at Close)
    std::atomic<u32> pushed{0};             ///< jobs appended so far (+ (1 << 31) once closed); the resolver waits on it
    std::atomic<u32> closed{0};
    std::atomic<u32> refs{2}; ///< the resolver and the filling thread: the last one done deletes it
    u64 open_serial{};        ///< OrderSerial() when the engine's thread put it into the stream (move census)
};
/// Issue 130 (api-contract.md W5, 2026-11-27): the engine's heap mover (GpuHeap_MoveOneResource / GpuHeap_DropTopMipsOverBudget, on a game worker) copies a resource
/// to new memory and swaps the engine's pointers only once the copy has run on the GPU (on the PS4 the async copy runs ahead of the frame's graphics work). A pass drawn
/// on one of our workers sits in the stream at its placeholder but reads the engine's pointers later, so a draw made after the swap must find the copy done in front
/// of it: while the mover runs, each job it makes goes into the oldest placeholder not yet closed (behind the draws already in it, ahead of every later one), or into
/// the stream when none is open. Nothing waits (a placeholder closing first is skipped). TLG_LEGACY_MOVER_IN_STREAM=1: always the stream (the old order).
bool MoverSinkBegin();
void MoverSinkEnd(bool begun);
bool MoverIntoSplice(); ///< (move census) this thread runs the mover and a placeholder is open
/// Move census (debug, TLG_VKB_MOVE_CENSUS): a serial that orders stream events across threads; the placeholder this thread fills (its open serial, 0: none);
/// placeholders not yet closed
u64 OrderSerial();
u64 SinkOpenSerial();
u32 OpenSplices();
/// The engine's thread: a new placeholder, enqueued now (the caller's open batch goes first). The pointer stays valid until Close.
Splice* OpenSplice();
/// The filling thread: its jobs go into `splice` (nullptr: into the stream again)
void SinkTo(Splice* splice);
/// The filling thread, after its last job (its open batch flushed): the resolver may handle the placeholder
void CloseSplice(Splice* splice);
/// R3b: called (on the enqueuing thread) before any job is enqueued or a submission is made: the caller's open batch of draws goes first (API order)
void SetBeforeEnqueue(void (*hook)(u32 work)); ///< (`work`: the job's Work kind, ~0u at a submission)
/// R3b: enqueue without calling the hook (the batch itself)
void EnqueueNoHook(std::unique_ptr<Job> job);
template <class F>
void Defer(Work work, F&& f) {
    auto job = std::make_unique<FnJob<std::decay_t<F>>>(std::decay_t<F>(std::forward<F>(f)));
    job->work = work;
    Enqueue(std::move(job));
}
/// Issue 111: a job with its own Prepare (run in stream order on the resolver, R4a; not run with the single recorder or without the recording thread)
template <class P, class F>
struct FnPrepJob final : Job {
    FnPrepJob(P&& p_, F&& f_) : p(std::move(p_)), f(std::move(f_)) {}
    static void* operator new(size_t) {
        return JobBlocks<sizeof(FnPrepJob)>::Get();
    }
    static void operator delete(void* ptr) {
        JobBlocks<sizeof(FnPrepJob)>::Put(ptr);
    }
    void Prepare() override {
        p();
    }
    void Run(vk::CommandBuffer cmd) override {
        f(cmd);
    }
    P p;
    F f;
};
template <class P, class F>
void DeferPrepared(Work work, P&& prepare, F&& f) {
    auto job = std::make_unique<FnPrepJob<std::decay_t<P>, std::decay_t<F>>>(std::decay_t<P>(std::forward<P>(prepare)), std::decay_t<F>(std::forward<F>(f)));
    job->work = work;
    Enqueue(std::move(job));
}
/// R4a: a job that begins (+1) or ends (-1) an occlusion query
template <class F>
void DeferQuery(s8 query, u64 address, F&& f) {
    auto job = std::make_unique<FnJob<std::decay_t<F>>>(std::decay_t<F>(std::forward<F>(f)));
    job->work = Work::Other;
    job->query = query;
    job->query_address = address;
    Enqueue(std::move(job));
}
template <class F>
void DeferUnlocked(F&& f) {
    auto job = std::make_unique<FnJob<std::decay_t<F>>>(std::decay_t<F>(std::forward<F>(f)));
    job->locked = false;
    Enqueue(std::move(job));
}
/// Issue 111: a buffer copy that must run in the stream in front of the job being resolved or recorded on our recording side now (a buffer's device copy made at
/// a first bind found there): from a job's Prepare (the resolver) it goes with that job, from a job's Run it is recorded at once. `source` none: zeros.
/// Elsewhere (not on the recording side) it is a job of its own (Defer).
void CopyInStream(vk::Buffer source, u64 source_offset, vk::Buffer destination, u64 bytes);
/// V4h (TLG_VKB_STAGE=1): a range of engine memory a command reads in place (guest memory over the bus) is copied, once per submission, into our device ring by a command
/// buffer that runs in front of the submission's; `buffer` / `offset` then name the copy. Not for a range our GPU work wrote earlier in the same submission. Lock held (a job).
/// False: left in place.
bool StageRead(vk::Buffer& buffer, u64& offset, u64 size);
/// U1 (TLG_VKB_HOIST_UPLOADS, default off while tested): a copy of CPU-written data into one of our buffers, from a job (lock held). When no command of the current
/// command buffer touched `dst` yet, the copy goes into the command buffer that runs in front of it (with the V4h copies, behind its one barrier; the current command
/// buffer's first barrier orders it before everything in it) and true is returned; else false (the caller records it in place, with its hazard check).
bool HoistCopy(vk::Buffer src, u64 src_offset, vk::Buffer dst, u64 dst_offset, u64 size);
/// V4i: our GPU work that writes engine memory in place says so: those ranges stay in place for the rest of the frame (a copy in front of a later submission would
/// miss the write). Lock held (a job). NewFrame: at the engine's graphics submit.
void NoteInPlaceWrite(vk::Buffer buffer, u64 offset, u64 size);
void NewFrame();

/// V5b: occlusion queries. The engine begins and ends counting for a result block (`address`: per counter pair a begin word and an end word, 16 bytes apart, bit 63
/// valid). Begin / End run in jobs (lock held, outside a pass): a Vulkan occlusion query counts the draws in between, in one command buffer (no split while it is open;
/// a submit in between ends it there). The answer (seen or not) is written into the block once the GPU has finished; a block never answered reads "visible".
void FirstAnswer(u64 address, u32 pairs);
void QueryBegin(vk::CommandBuffer cmd, u64 address, bool chained = false);
/// D4-10: ends this recording's open query at a submission's end; its interval goes on in a chained QueryBegin in the next submission
void QueryCut(vk::CommandBuffer cmd);
void QueryEnd(vk::CommandBuffer cmd, u64 address);
/// Writes the answers of the queries the GPU has finished (the recording thread, at a submit)
void PollQueries();

/// Waits until every job made so far has run
void Drain();
[[nodiscard]] bool RecorderOn();
/// True on the recording thread (or wherever jobs run at once)
[[nodiscard]] bool OnRecorder();
/// The value the GPU has passed on our timeline
[[nodiscard]] u64 CompletedValue();

/// The stream's command buffer, locked for the caller while it records
struct Recording {
    std::unique_lock<std::mutex> lock;
    vk::CommandBuffer cmd;
};
[[nodiscard]] Recording Record(Work work = Work::Other);
/// V4: the same for a draw: an open render pass stays open (Record ends it: every other command runs outside a pass)
[[nodiscard]] Recording RecordDraw();

/// V4d: what a command reads or writes, for the barriers: a range of a buffer, or a whole image (by handle)
struct Use {
    u64 handle{};
    u64 begin{}, end{};
    bool write{};
};
[[nodiscard]] inline Use BufferUse(vk::Buffer buffer, u64 offset, u64 size, bool write) {
    return {reinterpret_cast<u64>(static_cast<VkBuffer>(buffer)), offset, size == VK_WHOLE_SIZE ? ~0ull : offset + size, write};
}
[[nodiscard]] inline Use ImageUse(vk::Image image, bool write) {
    return {reinterpret_cast<u64>(static_cast<VkImage>(image)), 0, ~0ull, write};
}
/// V4d: before a command is recorded (lock held: the recording's): one full memory barrier when the command touches a range an earlier command wrote since the last barrier,
/// or writes a range one read since then (an open pass ends first); then the command's uses are noted. True: a barrier was recorded.
/// `TLG_LEGACY_VKB_BARRIERS=1` brings back the old rule (a barrier before every command but a draw, and at every pass change).
/// V4h: the barrier waits only for the kinds of GPU work (graphics, compute, transfer) that wrote what the command touches (or read what it writes); `agent` is the command's
/// kind (Current: the job's). `TLG_VKB_FULL_BARRIERS=1`: every barrier waits for all earlier work (V4d).
enum class Agent : u8 { Current = 0, Graphics = 1, Compute = 2, Transfer = 4 };
bool Hazard(vk::CommandBuffer cmd, std::span<const Use> uses, Agent agent = Agent::Current);
/// R5 census TLG_VKB_HAZARD_LOG=1: the shader hash the next Hazard calls belong to (recording thread), for log/vkb_hazards.txt
void SetHazardTag(u64 tag);
/// census (TLG_VKB_OVERLAP_CENSUS=1): the job being recorded is a dispatch of the engine's async compute context
void SetJobAsync(bool async);
/// Lantern waits, exact rule (our change to tabnul's #98): what one lantern particle dispatch writes, read by the CPU from the engine's own emitter records before it
/// runs: the emitter records it processes ([record_begin, record_end), its per-record output) and the particle slots they name (record dword 6; slot 0 left out: the
/// game's own throwaway slot, written by many emitters at once on the console too). `known` false: not readable (the dispatch keeps its wait).
struct LanternFootprint {
    bool known{false};
    u32 record_begin{}, record_end{};
    u32 slot_count{};
    std::array<u32, 16> slots{};
};
/// Set around the Hazard call of a lantern dispatch (null otherwise)
void SetLanternFootprint(const LanternFootprint* footprint);
/// One of the four lantern particle shaders (by program hash)
[[nodiscard]] bool IsLanternShader(u64 hash);
[[nodiscard]] bool LegacyBarriersOn();
inline bool Hazard(vk::CommandBuffer cmd, std::initializer_list<Use> uses, Agent agent = Agent::Current) {
    return Hazard(cmd, std::span<const Use>{uses.begin(), uses.size()}, agent);
}

/// V4: a render pass (dynamic rendering) of our attachments, in the general layout
struct Pass {
    std::array<vk::ImageView, 8> colors{};
    u32 color_count{};
    vk::ImageView depth{};
    bool depth_has_stencil{};
    vk::Extent2D extent{};
    u32 layers{1};
    bool operator==(const Pass&) const = default;
};
/// Makes `pass` the open pass of the recording (lock held): the same pass stays open; another ends the open one and begins, after a barrier when its attachments (`attachments`,
/// written) need one (V4d). Returns true when a new pass began.
bool UsePass(vk::CommandBuffer cmd, const Pass& pass, std::span<const Use> attachments);
/// R5 meter: what a draw in the open pass tests (bits: 1 depth attachment, 2 stencil attachment, 4 depth test, 8 stencil test, 16 depth bounds, 32 depth write), its pixel
/// shader and first color format (lock held, after UsePass). TLG_VKB_PASS_DUMP=1 writes one frame's passes with these to log/vkb_pass_dump.txt.
void NotePassDraw(u32 flags, u64 ps_hash, u32 format, const vk::Viewport& viewport, const vk::Rect2D& scissor);
/// Ends the open pass, if any (lock held): before a barrier or any command that is not a draw
void BreakPass(vk::CommandBuffer cmd);

/// Submits what was recorded since the last submit; returns the timeline value that is reached when it is done (the last value when nothing was recorded)
u64 Submit();
/// The last value handed out by Submit
[[nodiscard]] u64 LastValue();
/// R4a: the timeline value the command buffer being recorded on this thread goes out with (inside a job; LastValue() + 1 on the single recorder)
u64 RecordingValue();
/// Waits until the GPU has passed `value`
void Wait(u64 value);

/// A slice of the host ring (mapped, coherent; reused after the frames that use it are done)
struct Slice {
    vk::Buffer buffer{};
    u64 offset{};
    void* data{};
};
[[nodiscard]] Slice HostRing(u64 size, u64 alignment);
/// Scene on the GPU R1: the host ring's buffer (the record-mode row table)
vk::Buffer HostRingBuffer();
/// R3c: changes when the host ring wraps (a per-thread chunk taken before is not used further)
u64 HostRingGeneration();
/// The GDS stand-in (64 KiB, device local), a 64 KiB zero-initialised buffer for empty buffer slots, and a 1x1 image view in the general layout for empty image slots
[[nodiscard]] vk::Buffer GdsBuffer();
/// Kill-experiment TLG_VKB_HOST_SCRATCH=1 (wrong picture, fps bound only): a 256 MiB zeroed device-local buffer that stands in for every engine buffer read in place
/// (guest memory over the bus); null when the switch is off
[[nodiscard]] vk::Buffer HostScratch();
constexpr u64 HostScratchBytes = 256ull << 20;
/// Kill-experiment TLG_VKB_HOST_CACHE=1 (fps bound; the data is a snapshot of the first use, stale after): a 2 GiB device-local arena that holds a copy of every engine buffer
/// range read in place; null when the switch is off
[[nodiscard]] vk::Buffer HostCacheArena();
[[nodiscard]] bool HostCacheOn();
constexpr u64 HostCacheBytes = 2048ull << 20;
/// Kill-experiment TLG_VKB_NO_PASS_BARRIER=1: no full barrier when a render pass changes
[[nodiscard]] vk::Buffer NullBuffer();
[[nodiscard]] vk::ImageView NullImageView(bool storage);

/// GPU timestamps (diagnostic): a timestamp written into `cmd` (the recording's, lock held), its index in the current command buffer, or ~0u when there is no room
u32 Stamp(vk::CommandBuffer cmd);
/// R5: one timestamp at each submission's start and end only (the default); TLG_VKB_STAMPS=1 adds the per-kind, per-pass and per-dispatch ones
bool StampsLean();
/// The current command buffer's generation (lock held: the recording's)
[[nodiscard]] u64 Generation();
/// The timestamps (nanoseconds) of every submission the GPU has finished since the last call, with its generation
struct Stamps {
    u64 generation{};
    std::vector<u64> ns;
};
[[nodiscard]] std::vector<Stamps> FinishedStamps();

[[nodiscard]] nlohmann::json Report();
/// note 87: debugger mode's GPU submissions with their slowest passes (the slowest since debugger mode started, the newest); the frame walk freeze writes it (gpu_passes.json).
[[nodiscard]] nlohmann::json GpuPassReport();
/// note 90: debugger mode: a pass of 20 ms or more arms its pixel shader; the next draw with it dumps everything it uses (once per shader per recording). True once.
[[nodiscard]] bool TakeSlowPsArm(u64 ps_hash);
/// note 91: the one-frame capture of the frame walk freeze (debugger mode). The render thread turns it on for one frame (SetFrameCapture); each draw made then is
/// counted (CaptureDrawQueued) and, when recorded, noted with its row and buffers (NoteCapturedDraw, recording lock held); its pass's attachments and those buffers are
/// copied at the pass end. CaptureReport: the draw table (with per-draw GPU times).
void SetFrameCapture(bool on);
[[nodiscard]] bool FrameCaptureOn();
void CaptureDrawQueued();
[[nodiscard]] u64 CaptureJobsLeft();
u32 NoteCapturedDraw(nlohmann::json row, std::span<const vk::DescriptorBufferInfo> buffers);
[[nodiscard]] nlohmann::json CaptureReport();
/// note 91: debugger mode: a timestamp right after a draw (recording lock held), for the per-draw GPU times of gpu_passes.json and the capture
void StampDraw(u64 id, u64 vs, u64 ps, u32 count, u32 instances);

} // namespace Core::Vkb::Stream
