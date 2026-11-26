#include "tlg_write_log.h"
// tlg: glue between the C loader and the vendored shadPS4 video core.
#include "tlg_overlay.h"
#include "tlg_png.h"
#include "tlg_settings.h"
#include "tlg_copy.h"
#ifndef _WIN32
#include <utility>
#include <sys/resource.h>
#endif
#include "tlg_toggles.h"
#include "tlg_freeze.h"
#include "tlg_game_sema.h"
#include "tlg_fun.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <boost/asio/io_context.hpp>
#include "common/polyfill_thread.h"
#include "video_core/renderdoc.h"
#include <cstring>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <SDL3/SDL.h>
#include "../tlggpu.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/rdtsc.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/libs.h"
#include "native/native_timeline.h" // VKB
#include "native/native_vkb.h" // VKB
#include "native/native_renderdoc.h" // VKB: TLG_RENDERDOC
#include "native/native_hooks.h" // VKB
#include "native/native_debug_server.h" // VKB
#include "native/native_dev_keys.h" // VKB
#include "native/shadps4/core/guest_heap.h" // VKB
#include "native/shadps4/core/libraries/gnmdriver/gnmdriver.h" // VKB
#include "native/shadps4/core/libraries/videoout/video_out.h" // VKB
#include "core/memory.h"
#include "core/signals.h"
#include "sdl_window.h"
#include "tlg_trophy.h"
#include "video_core/renderer_vulkan/shader_recipe.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifdef _WIN32
extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentThreadId();
#endif
void TlgFreeze::NoteSlow(const char* what, std::uintptr_t address, double ms) {
#ifdef _WIN32
    const unsigned long thread = GetCurrentThreadId();
#else
    const unsigned long thread = 0;
#endif
    std::printf("SLOW: %.1f ms %s at %#llx, thread %lu\n", ms, what,
                static_cast<unsigned long long>(address), thread);
    std::fflush(stdout);
}

extern "C" {
// runtime_memory.c
int runtime_memory_is_mapped(uintptr_t address, uint64_t size);
int runtime_memory_write_backing(uintptr_t address, const void* data, uint64_t size);
int runtime_memory_read_backing(uintptr_t address, void* data, uint64_t size);
uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size);
int runtime_memory_region(uintptr_t address, uintptr_t* start, uintptr_t* end, int* mapped);
void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write);
typedef void (*RuntimeGpuRange)(uintptr_t address, uint64_t size);
// runtime_kernel.c: one clock for guest and GPU timestamps
uint64_t runtime_process_time_us(void);
uint64_t runtime_process_time_counter(void);
uint64_t runtime_tsc_frequency(void);
int32_t* runtime_errno(void);
void runtime_memory_set_gpu_hooks(RuntimeGpuRange map, RuntimeGpuRange unmap, RuntimeGpuRange invalidate);
// runtime.c: exit counters (and the ISA site report), as on the guest's own exit
void runtime_report(void);
}

Frontend::WindowSDL* g_window = nullptr;

namespace Libraries::GnmDriver { void RegisterLib(Core::Loader::SymbolsResolver* sym); }
namespace Libraries::AvPlayer { void RegisterLib(Core::Loader::SymbolsResolver* sym); }
namespace Libraries::VideoOut { void RegisterLib(Core::Loader::SymbolsResolver* sym); }
namespace Libraries::Kernel { void RegisterEventQueue(Core::Loader::SymbolsResolver* sym); }

namespace {
struct Symbol {
    std::string nid, library, module;
    u64 address;
};
std::vector<Symbol> g_symbols;
u32 g_sdk_version;
} // namespace

namespace Core::Loader {
void SymbolsResolver::AddSymbol(const char* nid, const char* library, const char* module, SymbolType,
                                u64 address) {
    g_symbols.push_back({nid, library, module, address});
}
} // namespace Core::Loader

// ElfInfo's fields are private to Core::Emulator; the port fills them here.
namespace Core {
class Emulator {
public:
    static void FillElfInfo(const TlgGpuConfig& config) {
        auto& info = Common::ElfInfo::Instance();
        info.initialized = true;
        info.game_serial = config.serial ? config.serial : "UNKNOWN";
        info.title = config.title ? config.title : "";
        info.sdk_ver = config.sdk_version;
        info.psf_attributes.raw = config.psf_attributes;
    }
};

static std::atomic<Vulkan::Rasterizer*> g_discard_rasterizer{nullptr};
// The guest is about to overwrite this range (memset, memcpy): skip what the GPU left in the whole
// pages inside it.
extern "C" void tlggpu_discard_overwrite(void* dst, std::uint64_t size) {
    auto* r = g_discard_rasterizer.load(std::memory_order_acquire);
    if (!r) {
        return;
    }
    const std::uint64_t addr = reinterpret_cast<std::uint64_t>(dst);
    const std::uint64_t first = (addr + 4095) & ~std::uint64_t{4095};
    const std::uint64_t last = (addr + size) & ~std::uint64_t{4095};
    if (last > first && r->HasGpuData(first, last - first)) {
        r->DiscardForOverwrite(first, last - first);
    }
}

void MemoryManager::SetRasterizer(Vulkan::Rasterizer* rasterizer_) {
    rasterizer = rasterizer_;
    g_discard_rasterizer.store(rasterizer_, std::memory_order_release);
    // Existing guest mappings are replayed by the runtime when hooks are installed.
    runtime_memory_set_gpu_hooks(
        [](uintptr_t address, uint64_t size) {
            auto* r = Memory::Instance()->GetRasterizer();
            r->RegisterMemory(address, size);
            r->MapMemory(address, size);
        },
        [](uintptr_t address, uint64_t size) { Memory::Instance()->GetRasterizer()->UnmapMemory(address, size); },
        [](uintptr_t address, uint64_t size) {
            Memory::Instance()->GetRasterizer()->InvalidateMemory(address, size);
        });
}
void MemoryManager::InvalidateMemory(VAddr address, u64 size) {
    if (rasterizer) rasterizer->InvalidateMemory(address, size);
}
u64 MemoryManager::ClampRangeSize(VAddr virtual_addr, u64 size) {
    return runtime_memory_clamp(virtual_addr, size);
}
static void CopySparseSerial(VAddr source, u8* dest, u64 size) {
    while (size) {
        uintptr_t start = 0, end = 0;
        int mapped = 0;
        if (!runtime_memory_region(source, &start, &end, &mapped)) {
            end = source + size;
        }
        const u64 n = std::min<u64>(size, end - source);
        // tlg: read through the unprotected backing view so readback protection (Precise
        // readbacks) cannot fault on the recorder/copy threads; direct read as fallback.
        if (!mapped) std::memset(dest, 0, n);
        else if (!runtime_memory_read_backing(source, dest, n))
            std::memcpy(dest, reinterpret_cast<const void*>(source), n);
        source += n; dest += n; size -= n;
    }
}
void MemoryManager::CopySparseMemory(VAddr source, u8* dest, u64 size) {
    // tlg: large uploads (streaming) are split across the copy threads.
    constexpr u64 Chunk = 512 * 1024;
    if (size < 4 * Chunk || !TlgCopy::Enabled()) {
        return CopySparseSerial(source, dest, size);
    }
    TlgCopy::ParallelFor((size + Chunk - 1) / Chunk, [&](std::size_t i) {
        const u64 offset = i * Chunk;
        CopySparseSerial(source + offset, dest + offset, std::min(Chunk, size - offset));
    });
}
bool MemoryManager::IsValidMapping(VAddr virtual_addr, u64 size) { // VKB
    uintptr_t start = 0, end = 0;
    int mapped = 0;
    return runtime_memory_region(virtual_addr, &start, &end, &mapped) && mapped && virtual_addr >= start && virtual_addr + size <= end;
}
bool MemoryManager::TryWriteBacking(void* address, const void* data, u64 size) {
    TlgWriteLog::Note(reinterpret_cast<uintptr_t>(address), data, size, TlgWriteLog::Backing);
    return runtime_memory_write_backing(reinterpret_cast<uintptr_t>(address), data, size) != 0;
}
void AddressSpace::Protect(VAddr virtual_addr, u64 size, MemoryPermission perms) {
    TlgStats::Timer timer{TlgStats::t_protect};
    const u64 pages = (size + 4095) / 4096;
    TlgStats::protect_calls.fetch_add(1, std::memory_order_relaxed);
    TlgStats::protect_pages.fetch_add(pages, std::memory_order_relaxed);
    if (!True(perms & MemoryPermission::Write)) {
        TlgStats::protect_revoke_calls.fetch_add(1, std::memory_order_relaxed);
        TlgStats::protect_revoke_pages.fetch_add(pages, std::memory_order_relaxed);
    }
    runtime_memory_gpu_protect(virtual_addr, size, True(perms & MemoryPermission::Read),
                               True(perms & MemoryPermission::Write));
}
boost::icl::interval_set<VAddr> AddressSpace::GetUsableRegions() {
    boost::icl::interval_set<VAddr> set;
    set += boost::icl::interval<VAddr>::right_open(0x10'0000'0000ULL, 0xfc'0000'0000ULL);
    return set;
}
} // namespace Core

// Kernel services the vendored libraries call directly.
namespace Libraries::Kernel {
u64 PS4_SYSV_ABI sceKernelGetTscFrequency() { return runtime_tsc_frequency(); }
u64 PS4_SYSV_ABI sceKernelReadTsc() { return Common::FencedRDTSC(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTime() { return runtime_process_time_us(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounter() { return runtime_process_time_counter(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounterFrequency() { return 1'000'000'000; }
s32 PS4_SYSV_ABI sceKernelIsNeoMode() { return 0; } // base PS4
s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32* ver) {
    if (!ver) return ORBIS_KERNEL_ERROR_EINVAL;
    *ver = s32(g_sdk_version);
    return ORBIS_OK;
}
s32 PS4_SYSV_ABI sceKernelUsleep(u32 microseconds) {
    std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
    return ORBIS_OK;
}
} // namespace Libraries::Kernel

namespace {
// SDL video must be driven from one thread: the window lives on its own host thread.
std::thread g_window_thread;
std::mutex g_window_mutex;
std::condition_variable g_window_cv;
bool g_window_ready;
} // namespace

u32 TlgDisplayRefreshHz() {
    static const u32 hz = [] {
        const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetPrimaryDisplay());
        const u32 rate = mode && mode->refresh_rate > 0 ? u32(mode->refresh_rate + 0.5f) : 60;
        std::printf("GPU: vblank follows the display refresh rate, %u Hz\n", rate);
        return std::max<u32>(rate, 60);
    }();
    return hz;
}

#if defined(TLG_PGO_GENERATE) && defined(TLG_PGO_CLANG)
// Clang instrumentation (compiler-rt profile runtime): write (merging into the %m file) and
// reset, so the file sums the intervals like the GCC path below.
extern "C" int __llvm_profile_write_file(void);
extern "C" void __llvm_profile_reset_counters(void);
static void __gcov_dump(void) { __llvm_profile_write_file(); }
static void __gcov_reset(void) { __llvm_profile_reset_counters(); }
#elif defined(TLG_PGO_GENERATE)
extern "C" void __gcov_dump(void);
extern "C" void __gcov_reset(void);
#endif
#ifdef TLG_PGO_GENERATE
// Instrumented build (TLG_PGO=generate): the game often ends through _exit (watchdog, guest
// exit), which skips the profile write at exit. Write every 30 s and reset: the files sum the
// intervals.
static void StartProfileWriter() {
    std::thread([] {
        Common::SetCurrentThreadName("tlg:pgo");
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            __gcov_dump();
            __gcov_reset();
            std::printf("PGO: profile written\n");
        }
    }).detach();
}
#endif

bool TlgHdrStartup() {
    return TlgSettings::Get().startup_hdr;
}

extern "C" void tlggpu_flush_cache(void) {
    Vulkan::ShaderRecipe::ReportBuilds(); // TLG_SHADER_PREBUILD / TLG_SHADER_BUILD_STATS
    const char* env = std::getenv("TLG_CACHE_FLUSH_ON_EXIT");
    static const bool enabled = !(env && env[0] == '0');
    auto& cache = Storage::DataBase::Instance();
    if (!enabled || !cache.IsOpened()) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    const bool done = cache.Flush(std::chrono::seconds(2));
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    std::printf("Shader cache: %s at exit in %lld ms\n",
                done ? "queue drained" : "queued writes NOT finished",
                static_cast<long long>(ms));
}

// VKB: guest waits and signals of the runtime for the frame timeline capture (F9 frame profiler, F8 stall profiler), from the old port; the runtime
// calls these through runtime_set_timeline_hooks (src/runtime.h), set in tlggpu_init.
extern "C" {
typedef struct RuntimeTimelineHooks {
    void (*wait_begin)(void);
    void (*wait_end)(const char* kind, const void* object, const char* name, uintptr_t caller);
    void (*signal)(int kind, const void* object);
} RuntimeTimelineHooks;
void runtime_set_timeline_hooks(const RuntimeTimelineHooks* hooks);
}
namespace {
thread_local u64 timeline_wait_start = 0;
u32 TimelineWaitKind(const char* kind) {
    if (!kind) return 6;
    if (!std::strncmp(kind, "condition", 9)) return 1;
    if (!std::strncmp(kind, "semaphore", 9)) return 2;
    if (!std::strncmp(kind, "sleep", 5)) return 3;
    if (!std::strncmp(kind, "event", 5)) return 4;
    if (!std::strncmp(kind, "mutex", 5)) return 5;
    return 6;
}
// The capture's object id. Semaphores (ids), files (descriptors) and sleeps (microseconds) give small numbers, not pointers: they get the wait kind in
// the top byte, so a semaphore and a file of the same number keep their own name and signals (a semaphore post is kind 7, its waits kind 2).
u64 TimelineObjectId(u32 wait_kind, const void* object) {
    const u64 id = reinterpret_cast<u64>(object);
    return id < (u64(1) << 32) ? id | (u64(wait_kind) << 56) : id;
}
void TimelineWaitBegin() {
    timeline_wait_start = Core::FrameTimeline::capture_active.load(std::memory_order_relaxed) ? Core::FrameTimeline::Now() : 0;
}
void TimelineWaitEnd(const char* kind, const void* object, const char* name, uintptr_t caller) {
    const u64 start = std::exchange(timeline_wait_start, 0);
    if (start == 0 || !Core::FrameTimeline::capture_active.load(std::memory_order_relaxed)) {
        return;
    }
    const u64 end = Core::FrameTimeline::Now();
    if (end - start < 50'000) {
        return;
    }
    const u32 wait_kind = TimelineWaitKind(kind);
    const u64 id = TimelineObjectId(wait_kind, object);
    Core::FrameTimeline::Record(Core::FrameTimeline::GameWait, (end - start) | (u64(wait_kind) << 56), end, id);
    Core::FrameTimeline::Record(Core::FrameTimeline::GameWaitSite, static_cast<u64>(caller), end, id); // the stall profiler pairs this with the wait above
    Core::FrameTimeline::CaptureNote(id, name && name[0] ? name : (kind ? kind : ""));
}
void TimelineSignal(int kind, const void* object) {
    if (Core::FrameTimeline::capture_active.load(std::memory_order_relaxed)) {
        const u64 id = TimelineObjectId(kind == 7 ? 2 : 1, object);
        Core::FrameTimeline::Record(Core::FrameTimeline::GameSignal, static_cast<u64>(kind), Core::FrameTimeline::Now(), id);
    }
}
} // namespace

extern "C" int tlggpu_init(const TlgGpuConfig* config) {
    {
        const RuntimeTimelineHooks hooks{&TimelineWaitBegin, &TimelineWaitEnd, &TimelineSignal};
        runtime_set_timeline_hooks(&hooks);
    }
    TlgSettings::Load();
#ifdef TLG_PGO_GENERATE
    StartProfileWriter();
#endif
    g_sdk_version = config->sdk_version;
#ifdef _WIN32
    if (config->user_dir && !std::getenv("TLG_GPU_USER_DIR")) _putenv_s("TLG_GPU_USER_DIR", config->user_dir);
#else
    if (config->user_dir) setenv("TLG_GPU_USER_DIR", config->user_dir, 0);
#endif
    Core::Emulator::FillElfInfo(*config);
    if (Core::Vkb::On()) {
        // VKB: the runtime maps the game's memory from 64 GiB (runtime_memory.c USER_MIN); the backend's heap checks (native/shadps4/core/guest_heap.h) use it
        Core::g_guest_heap = {0x1000000000ull, 0x10000000000ull, 0x1000000000ull, 0x10000000000ull};
        Core::DevKeys::InstallHooks(); // the glitch marker's and the frame walk's screenshots, toasts and thumbnails
    }
    // VKB: the debug server (development; TLG_DEBUG_SERVER_PORT, JSON endpoints, native/native_debug_server.cpp)
    if (const char* port = std::getenv("TLG_DEBUG_SERVER_PORT"); port && std::atoi(port) > 0) {
        Core::DebugServer::Start(static_cast<u16>(std::atoi(port)));
    }
    const std::string title = config->title ? config->title : "The Last Guardian";
    const s32 width = config->width, height = config->height;
    g_window_thread = std::thread([title, width, height] {
        Common::SetCurrentThreadName("tlg:window");
        auto* window = new Frontend::WindowSDL(width, height, title.c_str());
        {
            std::scoped_lock lock{g_window_mutex};
            g_window = window;
            g_window_ready = true;
        }
        g_window_cv.notify_all();
        while (window->PollEvents()) {
            SDL_Delay(2);
        }
        LOG_INFO(Frontend, "Window closed by user");
        runtime_report();
        tlggpu_flush_cache();
        std::fflush(stdout);
        std::_Exit(0);
    });
    g_window_thread.detach();
    {
        std::unique_lock lock{g_window_mutex};
        g_window_cv.wait(lock, [] { return g_window_ready; });
    }
    VideoCore::LoadRenderDoc(); // TLG_RENDERDOC=1 (native/native_renderdoc.cpp): before either renderer makes its Vulkan instance
    Core::Loader::SymbolsResolver resolver;
    if (Core::Vkb::On()) {
        // VKB: the engine-to-Vulkan backend (TLG_VKB=1, merge plan M3 (4)): our Gnm and VideoOut take the game's calls and our presenter
        // shows the backend's image; their renderer (presenter, Liverpool) is never created
        Libraries::VkbGnmDriver::RegisterLib(&resolver);
        Libraries::VkbVideoOut::RegisterLib(&resolver);
        return 0;
    }
    // GnmDriver creates the presenter that the VideoOut present thread uses.
    Libraries::GnmDriver::RegisterLib(&resolver);
    Libraries::VideoOut::RegisterLib(&resolver);
    return 0;
}

namespace Libraries::Kernel { void StartKernelService(); }
extern "C" void tlggpu_register_kernel(void) {
    Libraries::Kernel::StartKernelService();
    Core::Loader::SymbolsResolver resolver;
    Libraries::Kernel::RegisterEventQueue(&resolver);
    Libraries::AvPlayer::RegisterLib(&resolver);
}

extern "C" uintptr_t tlggpu_resolve(const char* scoped_nid) {
    const char* hash = std::strchr(scoped_nid, '#');
    const size_t length = hash ? size_t(hash - scoped_nid) : std::strlen(scoped_nid);
    for (const auto& symbol : g_symbols) {
        if (symbol.nid.size() == length && !std::memcmp(symbol.nid.data(), scoped_nid, length)) {
            return uintptr_t(symbol.address);
        }
    }
    return 0;
}

extern "C" int tlggpu_handle_fault(void* ucontext, void* address) {
    // tlg: the freeze report shows threads stuck in a page-protection fault (and how many).
    TlgFreeze::faults_in_progress.fetch_add(1, std::memory_order_relaxed);
    int handled;
    {
        TlgFreeze::WaitScope wait{"GPU page fault handler", address};
        handled = Core::Signals::Instance()->DispatchAccessViolation(ucontext, address) ? 1 : 0;
    }
    TlgFreeze::faults_in_progress.fetch_sub(1, std::memory_order_relaxed);
    TlgFreeze::faults_handled.fetch_add(handled, std::memory_order_relaxed);
    return handled;
}

extern "C" void tlggpu_set_guest_image(std::uintptr_t base, std::uint64_t size) {
    TlgFreeze::guest_size.store(size);
    TlgFreeze::guest_base.store(base);
}

extern "C" void tlggpu_install_hooks(std::uintptr_t image_base, std::uint64_t image_size) { // VKB
    if (Core::Vkb::On()) {
        Core::GuestHooks::InstallForEboot(image_base, image_size);
    }
}

extern "C" void tlggpu_set_game_sema_signal(TlgGpuGameSemaSignal signal) {
    TlgGameSema::signal.store(signal);
}

extern "C" void tlggpu_set_wait_hooks(TlgGpuWaitBegin begin, TlgGpuWaitEnd end) {
    TlgFreeze::end_hook.store(end);
    TlgFreeze::begin_hook.store(begin);
}

extern "C" void tlggpu_frame_counters(TlgGpuFrameCounters* out) {
    out->submitted = TlgFreeze::flips_submitted.load(std::memory_order_relaxed);
    out->shown = TlgFreeze::flips_shown.load(std::memory_order_relaxed);
    out->presented = TlgFreeze::presents.load(std::memory_order_relaxed);
    out->vblanks = TlgFreeze::vblanks.load(std::memory_order_relaxed);
    out->faults_in_progress = TlgFreeze::faults_in_progress.load(std::memory_order_relaxed);
    out->faults_handled = TlgFreeze::faults_handled.load(std::memory_order_relaxed);
}

extern "C" unsigned tlggpu_symbol_count(void) {
    return unsigned(g_symbols.size());
}

// Kernel service thread: runs boost::asio timers for equeue timer events.
namespace Libraries::Kernel {
boost::asio::io_context io_context;
static std::mutex m_asio_req;
static std::condition_variable_any cv_asio_req;
static std::atomic<u32> asio_requests;
static std::jthread service_thread;

void KernelSignalRequest() {
    std::unique_lock lock{m_asio_req};
    ++asio_requests;
    cv_asio_req.notify_one();
}

static void KernelServiceThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("tlg:kernel_service");
    while (!stoken.stop_requested()) {
        {
            std::unique_lock lock{m_asio_req};
            cv_asio_req.wait(lock, stoken, [] { return asio_requests != 0; });
        }
        if (stoken.stop_requested()) break;
        io_context.run();
        io_context.restart();
        asio_requests = 0;
    }
}

void StartKernelService() {
    service_thread = std::jthread{KernelServiceThread};
}

int* PS4_SYSV_ABI __Error() {
    return runtime_errno();
}
} // namespace Libraries::Kernel

// RenderDoc capture hooks: not used by the port (LoadRenderDoc is native/native_renderdoc.cpp: TLG_RENDERDOC=1).
namespace VideoCore {
void StartCapture() {}
void EndCapture() {}
void TriggerCapture() {}
void SetOutputDir(const std::filesystem::path&, const std::string&) {}
bool IsRenderDocLoaded() { return false; }
void RequestScreenshot(ScreenshotRequest) {}
u32 ConsumeGameOnlyScreenshotRequests() { return 0; }
u32 ConsumeWithOverlaysScreenshotRequests() { return 0; }
ScreenshotRequests ConsumeScreenshotRequests() { return {}; }
} // namespace VideoCore

extern "C" void tlggpu_overlay_notice(const char* text, float seconds) {
    TlgOverlay::Notice(text ? text : "", seconds);
}

// The controller the game reads changed (runtime_pad.c): Xbox-style or not, for Button icons: Automatic.
extern "C" void tlggpu_pad_kind(int xbox) {
    TlgSettings::Get().pad_is_xbox = xbox != 0;
    std::printf("Button icons: controller in use is %s\n", xbox ? "Xbox" : "PlayStation");
}

extern "C" void tlggpu_pad_game_buttons(uint32_t buttons) {
    TlgOverlay::SetGameButtons(buttons);
}

extern "C" int tlggpu_photo_takes_pad(uint8_t lx, uint8_t ly, uint8_t rx, uint8_t ry) {
    return TlgFun::PhotoTakesPad(lx, ly, rx, ry) ? 1 : 0;
}

extern "C" int tlggpu_overlay_captures_input(void) {
    return TlgOverlay::CapturesInput() ? 1 : 0;
}

extern "C" void tlggpu_audio_settings(TlgGpuAudioSettings* out) {
    const auto& s = TlgSettings::Get();
    out->surround = s.audio_surround ? 1 : 0;
    std::snprintf(out->device, sizeof(out->device), "%s", s.audio_device.c_str());
}

extern "C" void tlggpu_input_settings(TlgGpuInputSettings* out) {
    const auto& s = TlgSettings::Get();
    out->mouse_enabled = s.mouse_enabled ? 1 : 0;
    out->invert_y = s.mouse_invert_y ? 1 : 0;
    out->sensitivity = s.mouse_sensitivity;
    out->smoothing = s.mouse_smoothing;
    out->accel = s.mouse_accel;
    out->deadzone = s.mouse_deadzone;
    std::snprintf(out->key_map, sizeof(out->key_map), "%s", s.key_map.c_str());
}

extern "C" int tlggpu_text_input_begin(const char* initial, const char* prompt) {
    if (!g_window) return 0;
    g_window->BeginTextInput(initial ? initial : "", prompt ? prompt : "Text");
    return 1;
}

static TlgOverlay::SaveRow SaveRowOf(const TlgSaveEntry& entry) {
    TlgOverlay::SaveRow row;
    row.title = entry.title ? entry.title : "";
    row.subtitle = entry.subtitle ? entry.subtitle : "";
    row.detail = entry.detail ? entry.detail : "";
    row.mtime = entry.mtime;
    row.bytes = entry.bytes;
    row.is_new = entry.is_new != 0;
    return row;
}

static TlgPng::Image SavePicture(const char* path) {
    TlgPng::Image picture;
    if (path && !TlgPng::LoadRgba(path, picture)) {
        LOG_WARNING(Lib_SaveDataDialog, "save dialog: no picture from {}", path);
    }
    return picture;
}

extern "C" int tlggpu_save_list_begin(int kind, const TlgSaveEntry* entries, int count, int focus) {
    if (!g_window || count <= 0 || !entries) return 0;
    // Pictures are decoded here, on the game's thread, before the list is shown.
    std::vector<TlgOverlay::SaveRow> rows;
    std::vector<TlgPng::Image> pictures;
    for (int i = 0; i < count; ++i) {
        rows.push_back(SaveRowOf(entries[i]));
        pictures.push_back(SavePicture(entries[i].icon));
    }
    TlgOverlay::BeginSaveList(kind, std::move(rows), std::move(pictures), focus);
    return 1;
}

extern "C" int tlggpu_save_list_poll(void) {
    return TlgOverlay::PollSaveList();
}

extern "C" int tlggpu_save_message_begin(int kind, int code, uint64_t value, const char* text,
                                         int buttons, const TlgSaveEntry* entry) {
    if (!g_window) return 0;
    TlgOverlay::SaveRow row;
    if (entry) row = SaveRowOf(*entry);
    TlgOverlay::BeginSaveMessage(kind, code, value, text ? text : "", buttons, entry ? &row : nullptr,
                                 SavePicture(entry ? entry->icon : nullptr));
    return 1;
}

extern "C" int tlggpu_save_message_poll(void) {
    return TlgOverlay::PollSaveMessage();
}

extern "C" void tlggpu_save_dialog_close(void) {
    TlgOverlay::CloseSaveDialog();
}

extern "C" void tlggpu_save_progress(int kind, int active, int percent, const char* text) {
    if (g_window) TlgOverlay::SaveProgress(kind, active, percent, text);
}

extern "C" void tlggpu_save_toast(int kind) {
    if (g_window) TlgOverlay::SaveToast(kind);
}

extern "C" void tlggpu_trophy_setup(const void* trp, uint64_t trp_size, const void* npbind, uint64_t npbind_size,
                                    const char* store, int language) {
    const auto* t = static_cast<const unsigned char*>(trp);
    const auto* n = static_cast<const unsigned char*>(npbind);
    TlgTrophy::Setup(t ? std::vector<unsigned char>(t, t + trp_size) : std::vector<unsigned char>{},
                     n ? std::vector<unsigned char>(n, n + npbind_size) : std::vector<unsigned char>{},
                     store ? store : "trophies.txt", language);
}

extern "C" int tlggpu_trophy_unlock(int id) {
    return TlgTrophy::Unlock(id) ? 1 : 0;
}

extern "C" void tlggpu_trophy_flags(uint32_t flags[4]) {
    TlgTrophy::UnlockFlags(flags);
}

extern "C" int tlggpu_trophy_count(void) {
    return TlgTrophy::Count();
}

extern "C" int tlggpu_trophy_next_locked(void) {
    return TlgTrophy::NextLocked();
}

extern "C" int tlggpu_text_input_poll(char* out, uint64_t size) {
    if (!g_window) return 2;
    std::string text;
    const int state = g_window->PollTextInput(text);
    if (size) {
        const size_t n = std::min<size_t>(text.size(), size - 1);
        std::memcpy(out, text.data(), n);
        out[n] = 0;
    }
    return state;
}
