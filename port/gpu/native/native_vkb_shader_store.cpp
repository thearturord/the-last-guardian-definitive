// SPDX-License-Identifier: GPL-2.0-or-later
// TLG_VKB_SHADER_MEMO and TLG_VKB_SHADER_PREBUILD: the new engine's translated shader stages, kept for the session and on disk (see the header)

#include "native/native_vkb_shader_store.h"
#include "native/native_vkb_shader_list.h"
#include "native/native_vkb_shader_prebuild.h"
#include "native/native_vkb_pipeline_warm.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <xxhash.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "native/shadps4/shader_recompiler/ir/microinstruction.h"
#include "tlg_settings.h"
#include "native/shadps4/shader_recompiler/specialization.h"

namespace Core::Vkb::ShaderStore {
namespace {

// ---- the encoding of an entry (raw bytes: the folder id names the build, so only this build reads them) ----

constexpr u32 Magic = 0x5653474c; // "LGSV"
constexpr u32 Version = 1;

struct Writer {
    std::vector<u8> out;
    void Raw(const void* p, size_t n) {
        const auto* b = static_cast<const u8*>(p);
        out.insert(out.end(), b, b + n);
    }
    template <typename T>
    void Pod(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        Raw(&v, sizeof(T));
    }
    template <typename List>
    void Items(const List& list) { // a static_vector or small_vector of plain structs
        static_assert(std::is_trivially_copyable_v<typename List::value_type>);
        Pod(static_cast<u32>(list.size()));
        Raw(list.data(), list.size() * sizeof(typename List::value_type));
    }
};

struct Reader {
    const u8* p;
    size_t left;
    bool ok = true;
    void Raw(void* dst, size_t n) {
        if (!ok || n > left) {
            ok = false;
            std::memset(dst, 0, n);
            return;
        }
        std::memcpy(dst, p, n);
        p += n;
        left -= n;
    }
    template <typename T>
    void Pod(T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        Raw(&v, sizeof(T));
    }
    template <typename List>
    void Items(List& list) {
        static_assert(std::is_trivially_copyable_v<typename List::value_type>);
        u32 n = 0;
        Pod(n);
        bool fits = true;
        if constexpr (requires { list.max_size(); }) {
            fits = n <= list.max_size();
        }
        if (!ok || !fits || static_cast<size_t>(n) * sizeof(typename List::value_type) > left) {
            ok = false;
            return;
        }
        list.resize(n);
        Raw(list.data(), n * sizeof(typename List::value_type));
    }
    template <typename T>
    void Vector(std::vector<T>& v) {
        u32 n = 0;
        Pod(n);
        if (!ok || static_cast<size_t>(n) * sizeof(T) > left) {
            ok = false;
            return;
        }
        v.resize(n);
        Raw(v.data(), n * sizeof(T));
    }
};

// The members of the translator's Info, in declaration order. The mirror structs below repeat them: a member the engine's author adds,
// removes or moves makes the sizes differ and stops the build here, so the encoding is never silently short of a member.
#define TLG_STORE_PERSISTENT(X)                                                                                                            \
    X(buffers) X(images) X(samplers) X(fmasks) X(fetch_shader_sgpr_base) X(shared_memory_scratch_size) X(pgm_hash) X(tess_consts_dword_offset) \
    X(tess_consts_ptr_base) X(hw_stage) X(sw_stage) X(mrt_mask) X(has_fetch_shader) X(has_bitwise_xor) X(uses_dma) X(indexed) X(constant_offsets)
#define TLG_STORE_INFO(X)                                                                                                                  \
    X(user_data) X(flattened_ud_buf) X(srt_info) X(loads) X(stores) X(readconst_types) X(gs_copy_data) X(uses_patches) X(pgm_base)              \
    X(has_storage_images) X(has_discard) X(has_image_gather) X(has_image_query) X(has_flatbuf) X(uses_buffer_atomic_float_min_max)            \
    X(uses_image_atomic_float_min_max) X(uses_lane_id) X(uses_shader_clock) X(uses_group_quad) X(uses_group_shuffle) X(uses_group_ballot)     \
    X(shared_types) X(uses_fp16) X(uses_fp64) X(uses_pack_10_11_11) X(uses_unpack_10_11_11) X(uses_buffer_int64_atomics)                     \
    X(uses_shared_int64_atomics) X(stores_tess_level_outer) X(stores_tess_level_inner) X(translation_failed) X(fs_interpolation)
#define TLG_STORE_SPEC(X) X(info) X(runtime_info) X(bitset) X(fetch_shader_data) X(vs_attribs) X(buffers) X(images) X(fmasks) X(samplers) X(start)

#define TLG_MIRROR_P(m) decltype(VkbShader::InfoPersistent::m) m;
#define TLG_MIRROR_I(m) decltype(VkbShader::Info::m) m;
#define TLG_MIRROR_S(m) decltype(VkbShader::StageSpecialization::m) m;
struct PersistentMirror {
    TLG_STORE_PERSISTENT(TLG_MIRROR_P)
};
struct InfoMirror : PersistentMirror {
    TLG_STORE_INFO(TLG_MIRROR_I)
};
struct SpecMirror {
    TLG_STORE_SPEC(TLG_MIRROR_S)
};
static_assert(sizeof(PersistentMirror) == sizeof(VkbShader::InfoPersistent), "VkbShader::InfoPersistent changed: update TLG_STORE_PERSISTENT");
static_assert(sizeof(InfoMirror) == sizeof(VkbShader::Info), "VkbShader::Info changed: update TLG_STORE_INFO");
static_assert(sizeof(SpecMirror) == sizeof(VkbShader::StageSpecialization), "VkbShader::StageSpecialization changed: update TLG_STORE_SPEC");
static_assert(std::is_trivially_copyable_v<VkbShader::RuntimeInfo> && std::is_trivially_copyable_v<VkbShader::Backend::Bindings>);
static_assert(std::is_trivially_copyable_v<WriteWindow::Window>);

// One member each. Members that belong to a draw and not to the translation are not kept: the user data (set at the use), the flattened
// user data (made again from the user data), the program base (set at the use).
template <typename T>
void Put(Writer& w, const T& v) {
    w.Pod(v);
}
template <typename T, std::size_t N>
void Put(Writer& w, const boost::container::static_vector<T, N>& v) {
    w.Items(v);
}
template <typename T, std::size_t N, typename S>
void Put(Writer& w, const SmallVector<T, N, S>& v) {
    w.Items(v);
}
void Put(Writer&, const std::span<const u32>&) {}
void Put(Writer&, const std::vector<u32>&) {}
void Put(Writer& w, const VkbShader::PersistentSrtInfo& srt) {
    VkbShader::PersistentSrtInfo copy = srt;
    copy.walker_func = nullptr;
    w.Pod(copy);
    std::vector<u8> code;
    if (srt.walker_func != nullptr && !VkbShader::PristineWalkerCode(srt.walker_func, code)) {
        // not the walker made by this thread's last translation: the bytes in the code buffer (a fault may have patched them)
        code.assign(reinterpret_cast<const u8*>(srt.walker_func), reinterpret_cast<const u8*>(srt.walker_func) + srt.walker_func_size);
    }
    w.Pod(static_cast<u32>(code.size()));
    w.Raw(code.data(), code.size());
}
void Put(Writer&, const VkbShader::CopyShaderData&) {} // geometry stages only (Keep refuses a stage that has one)
void Put(Writer& w, const VkbShader::Gcn::FetchShaderData& f) {
    w.Pod(f.size);
    w.Items(f.attributes);
    w.Pod(f.vertex_offset_sgpr);
    w.Pod(f.instance_offset_sgpr);
}
void Put(Writer&, const VkbShader::Info* const&) {}

template <typename T>
void Get(Reader& r, T& v) {
    r.Pod(v);
}
template <typename T, std::size_t N>
void Get(Reader& r, boost::container::static_vector<T, N>& v) {
    r.Items(v);
}
template <typename T, std::size_t N, typename S>
void Get(Reader& r, SmallVector<T, N, S>& v) {
    r.Items(v);
}
void Get(Reader&, std::span<const u32>&) {}
void Get(Reader&, std::vector<u32>&) {}
void Get(Reader& r, VkbShader::PersistentSrtInfo& srt) {
    r.Pod(srt);
    u32 size = 0;
    r.Pod(size);
    srt.walker_func = nullptr;
    if (!r.ok || size > r.left || size != srt.walker_func_size) {
        r.ok = false;
        return;
    }
    if (size != 0) {
        srt.walker_func = VkbShader::RegisterWalkerCode(r.p, size);
        r.p += size;
        r.left -= size;
    }
}
void Get(Reader&, VkbShader::CopyShaderData&) {}
void Get(Reader& r, VkbShader::Gcn::FetchShaderData& f) {
    r.Pod(f.size);
    r.Items(f.attributes);
    r.Pod(f.vertex_offset_sgpr);
    r.Pod(f.instance_offset_sgpr);
}
void Get(Reader&, const VkbShader::Info*&) {}

void PutInfo(Writer& w, const VkbShader::Info& info) {
#define TLG_PUT(m) Put(w, info.m);
    TLG_STORE_PERSISTENT(TLG_PUT)
    TLG_STORE_INFO(TLG_PUT)
#undef TLG_PUT
}
void GetInfo(Reader& r, VkbShader::Info& info) {
#define TLG_GET(m) Get(r, info.m);
    TLG_STORE_PERSISTENT(TLG_GET)
    TLG_STORE_INFO(TLG_GET)
#undef TLG_GET
}
// The check mode's comparison, member by member (the encoding also holds padding bytes, which differ between two equal translations)
template <typename T>
bool Eq(const T& a, const T& b) {
    if constexpr (requires { a == b; }) {
        return a == b;
    } else {
        static_assert(std::has_unique_object_representations_v<T>, "a member without == and with padding: give it a comparison here");
        return std::memcmp(&a, &b, sizeof(T)) == 0;
    }
}
template <typename List>
bool EqItems(const List& a, const List& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!Eq(a[i], b[i])) {
            return false;
        }
    }
    return true;
}
template <typename T, std::size_t N>
bool Eq(const boost::container::static_vector<T, N>& a, const boost::container::static_vector<T, N>& b) {
    return EqItems(a, b);
}
bool Eq(const std::span<const u32>&, const std::span<const u32>&) {
    return true; // the draw's, set at the use
}
bool Eq(const VkbShader::PersistentSrtInfo& a, const VkbShader::PersistentSrtInfo& b) {
    Writer wa, wb; // the walker by its bytes, as kept
    Put(wa, a);
    Put(wb, b);
    const size_t head = sizeof(VkbShader::PersistentSrtInfo);
    return a.walker_func_size == b.walker_func_size && a.flattened_bufsize_dw == b.flattened_bufsize_dw &&
           std::memcmp(a.table_root_plus1, b.table_root_plus1, sizeof(a.table_root_plus1)) == 0 && std::memcmp(a.table_src, b.table_src, sizeof(a.table_src)) == 0 &&
           std::equal(wa.out.begin() + head, wa.out.end(), wb.out.begin() + head, wb.out.end());
}
template <std::size_t N>
bool Eq(const std::array<VkbShader::Info::Interpolation, N>& a, const std::array<VkbShader::Info::Interpolation, N>& b) {
    static_assert(std::has_unique_object_representations_v<VkbShader::Info::Interpolation>);
    return std::memcmp(a.data(), b.data(), sizeof(a)) == 0;
}
bool Eq(const VkbShader::CopyShaderData&, const VkbShader::CopyShaderData&) {
    return true; // geometry stages only, never kept
}
/// The first member in which two translations' infos differ (nullptr: none)
const char* FirstDifference(const VkbShader::Info& a, const VkbShader::Info& b) {
    const char* first = nullptr;
#define TLG_EQ(m)                                                                                                                          \
    if (first == nullptr && !Eq(a.m, b.m)) {                                                                                               \
        first = #m;                                                                                                                        \
    }
    TLG_STORE_PERSISTENT(TLG_EQ)
    TLG_STORE_INFO(TLG_EQ)
#undef TLG_EQ
    return first;
}

void PutSpec(Writer& w, const VkbShader::StageSpecialization& spec) {
#define TLG_PUT(m) Put(w, spec.m);
    TLG_STORE_SPEC(TLG_PUT)
#undef TLG_PUT
}
void GetSpec(Reader& r, VkbShader::StageSpecialization& spec) {
#define TLG_GET(m) Get(r, spec.m);
    TLG_STORE_SPEC(TLG_GET)
#undef TLG_GET
}

// ---- the entries ----

struct CodeId {
    u64 words{}, lo{}, hi{};
    bool operator==(const CodeId&) const = default;
};
CodeId CodeOf(std::span<const u32> code) {
    const XXH128_hash_t h = XXH3_128bits(code.data(), code.size_bytes());
    return {code.size(), h.low64, h.high64};
}

struct Entry {
    CodeId code;
    VkbShader::RuntimeInfo runtime_in{}; ///< compared byte by byte (the translator's settings start zeroed, so their padding is zero too)
    VkbShader::Backend::Bindings start{};
    u64 salt{};
    VkbShader::StageSpecialization spec; ///< the kinds of the resources the stage read (the normal renderer's rule)
    Translation result;                  ///< result.info: no user data between lookups
    u64 translate_ns{};
    bool used = false;                   ///< build stats: its first use was logged
    bool from_disk = false;
};

u64 KeyOf(u64 pgm_hash, const VkbShader::RuntimeInfo& runtime) {
    return pgm_hash ^ (static_cast<u64>(runtime.hw_stage) << 56) ^ (static_cast<u64>(runtime.sw_stage) << 60);
}

bool SameInputs(const Entry& e, const Inputs& in, const CodeId& code) {
    return e.salt == in.salt && e.start == in.start && e.code == code && std::memcmp(&e.runtime_in, &in.runtime, sizeof(VkbShader::RuntimeInfo)) == 0;
}

std::vector<u8> Encode(const Entry& e) {
    Writer w;
    w.Pod(Magic);
    w.Pod(Version);
    w.Pod(e.code);
    w.Pod(e.runtime_in);
    w.Pod(e.start);
    w.Pod(e.salt);
    w.Pod(e.translate_ns);
    PutSpec(w, e.spec);
    PutInfo(w, *e.result.info);
    w.Pod(e.result.runtime);
    w.Pod(e.result.end);
    w.Pod(static_cast<u32>(e.result.spv.size()));
    w.Raw(e.result.spv.data(), e.result.spv.size() * sizeof(u32));
    w.Pod(static_cast<u32>(e.result.windows.size()));
    w.Raw(e.result.windows.data(), e.result.windows.size() * sizeof(WriteWindow::Window));
    w.Pod(static_cast<u64>(XXH3_64bits(w.out.data(), w.out.size()))); // a damaged or cut file is never read as an entry
    return std::move(w.out);
}

std::unique_ptr<Entry> Decode(const std::vector<u8>& bytes) {
    if (bytes.size() < 16 || XXH3_64bits(bytes.data(), bytes.size() - 8) != [&] { u64 v; std::memcpy(&v, bytes.data() + bytes.size() - 8, 8); return v; }()) {
        return nullptr;
    }
    Reader r{bytes.data(), bytes.size() - 8};
    u32 magic = 0, version = 0;
    r.Pod(magic);
    r.Pod(version);
    if (!r.ok || magic != Magic || version != Version) {
        return nullptr;
    }
    auto e = std::make_unique<Entry>();
    r.Pod(e->code);
    r.Pod(e->runtime_in);
    r.Pod(e->start);
    r.Pod(e->salt);
    r.Pod(e->translate_ns);
    GetSpec(r, e->spec);
    e->result.info = std::make_unique<VkbShader::Info>();
    GetInfo(r, *e->result.info);
    e->spec.info = e->result.info.get();
    r.Pod(e->result.runtime);
    r.Pod(e->result.end);
    r.Vector(e->result.spv);
    r.Vector(e->result.windows);
    if (!r.ok || r.left != 0 || e->result.spv.empty()) {
        return nullptr;
    }
    e->from_disk = true;
    return e;
}

// ---- the store ----

// The store's state lives as long as the process: never destroyed, so the writer thread may still run at exit.
struct State {
    std::mutex mutex;
    std::unordered_map<u64, std::vector<std::unique_ptr<Entry>>> entries; ///< by program hash and stage
    std::unordered_map<u64, std::vector<std::filesystem::path>> on_disk;  ///< files not read yet, by the same key
    std::unordered_set<std::string> indexed;                              ///< names of the files indexed or written so far (Rescan adds the others)
    std::filesystem::path folder;
    std::unordered_set<u64> different_logged;
    std::string open_error;
    // the writer: files are written on a thread of their own, as "<name>.tmp" first and then renamed, so a reader never sees half a file
    std::mutex write_mutex;
    std::condition_variable write_cv;
    struct WriteItem {
        std::filesystem::path path; ///< full path
        std::vector<u8> bytes;
        bool recipe = false; ///< a shader recipe's file (QueueWrite): counted apart from the store's own
    };
    std::deque<WriteItem> write_queue;
    u64 writes_pending = 0;                                                     ///< queued or being written (Flush)
    std::condition_variable flushed_cv;
};
State& S() {
    static State* const state = new State;
    return *state;
}
std::atomic<bool> ready{false};
std::atomic<bool> writer_started{false}; ///< WriterLoop runs (QueueWrite)
std::atomic<u64> c_lookups{0}, c_hits{0}, c_disk_hits{0}, c_kept{0}, c_saved_ns{0}, c_check_same{0}, c_check_different{0}, c_spv_words{0}, c_files_indexed{0},
    c_files_read{0}, c_files_bad{0}, c_written{0}, c_write_failed{0}, c_not_kept_geometry{0}, c_open_ms{0}, c_bypassed{0}, c_kind_misses{0}, c_recipe_written{0}, c_recipe_write_failed{0}, c_rescanned{0},
    c_outside_fixed{0}, c_memory_only{0};

std::string FileName(u64 pgm_hash, const VkbShader::RuntimeInfo& runtime, const std::vector<u8>& bytes) {
    return fmt::format("{:016x}_{}_{}_{:016x}.vst", pgm_hash, static_cast<u32>(runtime.hw_stage), static_cast<u32>(runtime.sw_stage), XXH3_64bits(bytes.data(), bytes.size()));
}

void WriterLoop() {
    State& st = S();
    for (;;) {
        State::WriteItem item;
        {
            std::unique_lock lock{st.write_mutex};
            st.write_cv.wait(lock, [&] { return !st.write_queue.empty(); });
            item = std::move(st.write_queue.front());
            st.write_queue.pop_front();
        }
        const auto& path = item.path;
        auto tmp = path;
        tmp += ".tmp";
        std::error_code ec;
        bool ok = false;
        if (std::ofstream f{tmp, std::ios::binary | std::ios::trunc}) {
            f.write(reinterpret_cast<const char*>(item.bytes.data()), static_cast<std::streamsize>(item.bytes.size()));
            ok = static_cast<bool>(f);
        }
        if (ok) {
            std::filesystem::rename(tmp, path, ec);
            ok = !ec;
        }
        if (!ok) {
            std::filesystem::remove(tmp, ec);
            if (item.recipe) {
                if (c_recipe_write_failed.fetch_add(1) == 0) {
                    LOG_ERROR(Render_Vulkan, "vkb shader recipes: could not write {} (the recipe is lost; the stage itself is in the store; failed recipe writes are counted)",
                              path.string());
                }
            } else if (c_write_failed.fetch_add(1) == 0) {
                LOG_ERROR(Render_Vulkan, "vkb shader store: could not write {} (the stage stays translated in memory for this session; failed writes are counted)",
                          path.string());
            }
        } else {
            (item.recipe ? c_recipe_written : c_written).fetch_add(1, std::memory_order_relaxed);
        }
        {
            std::scoped_lock lock{st.write_mutex};
            --st.writes_pending;
        }
        st.flushed_cv.notify_all();
    }
}

void QueueWriteFile(std::filesystem::path path, std::vector<u8> bytes, bool recipe = false) {
    State& st = S();
    {
        std::scoped_lock lock{st.write_mutex};
        st.write_queue.push_back({std::move(path), std::move(bytes), recipe});
        ++st.writes_pending;
    }
    st.write_cv.notify_one();
}

/// Writes a small file now (tmp, then renamed): the folder's profile and settings
bool WriteNow(const std::filesystem::path& path, const void* data, size_t size) {
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream f{tmp, std::ios::binary | std::ios::trunc};
        f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!f) {
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

std::vector<u8> ReadAll(const std::filesystem::path& path) {
    std::ifstream f{path, std::ios::binary};
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

/// The file of the module that holds this code (libtlggpu.so / tlggpu.dll): its bytes name the build.
std::filesystem::path ModulePath() {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&ModulePath), &module)) {
        return {};
    }
    std::wstring name(32768, L'\0');
    const DWORD n = GetModuleFileNameW(module, name.data(), static_cast<DWORD>(name.size()));
    if (n == 0 || n >= name.size()) {
        return {};
    }
    name.resize(n);
    return std::filesystem::path{name};
#else
    Dl_info dl{};
    if (dladdr(reinterpret_cast<void*>(&ModulePath), &dl) == 0 || dl.dli_fname == nullptr) {
        return {};
    }
    return std::filesystem::path{dl.dli_fname};
#endif
}

/// The build's id: the module file's size and time of its last change (any rebuild changes the time; a copy that keeps the time keeps the
/// id, which is right: the same bytes). Reading the whole file is too slow at start (it holds the debug information too, hundreds of MB).
bool BuildId(const std::filesystem::path& path, u64& out) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        return false;
    }
    const auto time = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return false;
    }
    const u64 parts[2] = {static_cast<u64>(size), static_cast<u64>(time.time_since_epoch().count())};
    out = XXH3_64bits(parts, sizeof(parts));
    return true;
}

/// A hash of the bytes of `v` that hold its members, without its padding bytes (they can differ between two equal values). A byte is a
/// member's when flipping its lowest bit makes the value compare different (a bool stays a valid bool); `T`'s == compares every member.
template <typename T>
u64 HashMembers(const T& v, u64 seed) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::array<u8, sizeof(T)> bytes;
    std::memcpy(bytes.data(), &v, sizeof(T));
    for (size_t i = 0; i < sizeof(T); ++i) {
        T flipped;
        std::memcpy(&flipped, &v, sizeof(T));
        reinterpret_cast<u8*>(&flipped)[i] ^= 1;
        if (flipped == v) {
            bytes[i] = 0; // padding
        }
    }
    return XXH3_64bits_withSeed(bytes.data(), bytes.size(), seed);
}

/// Every setting the translator reads besides its profile; each value is part of the folder id, so a start with another value never takes
/// a stage translated with the old one. port/tests/test_vkb_store_settings.py reads these three lists and fails when the translator reads a
/// setting none of them names.
constexpr const char* TranslatorEnv[] = {"TLG_CONST_OFFSET_FLOATRULE", "TLG_BLOCK1D"};          ///< environment, by value
constexpr const char* TranslatorEmulatorSettings[] = {"IsDirectMemoryAccessEnabled"};            ///< EmulatorSettings, in TranslatorSettings()
/// read, but only to write debug files (IsDumpShaders), or only to choose how fast the same result is made (TLG_VKB_TRANSLATE_FAST: the same
/// module word for word, proven with tlg-probe --vkb-compare-stores)
constexpr const char* TranslatorWithoutEffect[] = {"IsDumpShaders", "TLG_VKB_TRANSLATE_FAST"};
std::string TranslatorSettings() {
    static_assert(std::size(TranslatorEmulatorSettings) == 1 && std::size(TranslatorWithoutEffect) == 2, "name each EmulatorSettings value below");
    std::string out = fmt::format("IsDirectMemoryAccessEnabled={}", EmulatorSettings.IsDirectMemoryAccessEnabled() ? 1 : 0);
    for (const char* name : TranslatorEnv) {
        const char* value = std::getenv(name);
        out += fmt::format(";{}={}", name, value != nullptr ? value : "<unset>");
    }
    return out;
}

/// The folder id of this build for `profile` and `settings`; false when the engine's module file cannot be read
bool FolderId(const VkbShader::Profile& profile, const std::string& settings, u64& id, std::filesystem::path& module) {
    module = ModulePath();
    u64 build = 0;
    if (module.empty() || !BuildId(module, build)) {
        return false;
    }
    id = XXH3_64bits_withSeed(settings.data(), settings.size(), HashMembers(profile, build));
    return true;
}

/// The stored stages of `dir` by key, except those named in `skip`; with `remove_cut`, cut writes are removed (only the game's own opening
/// does that: later, a .tmp file may be another process's write in progress, the shader pre-build's)
void IndexFolder(const std::filesystem::path& dir, std::unordered_map<u64, std::vector<std::filesystem::path>>& found, u64& files, u64& stale, bool remove_cut,
                 std::unordered_set<std::string>& names) {
    std::error_code ec;
    for (const auto& item : std::filesystem::directory_iterator{dir, ec}) {
        const auto name = item.path().filename().string();
        if (name.ends_with(".tmp")) { // a write cut by the end of a session
            if (remove_cut) {
                std::filesystem::remove(item.path(), ec);
                ++stale;
            }
            continue;
        }
        unsigned long long pgm = 0;
        u32 hw = 0, sw = 0;
        if (std::sscanf(name.c_str(), "%16llx_%u_%u_", &pgm, &hw, &sw) != 3 || !name.ends_with(".vst")) {
            continue;
        }
        if (!names.insert(name).second) {
            continue;
        }
        VkbShader::RuntimeInfo key_runtime{};
        key_runtime.hw_stage = static_cast<VkbShader::HwStage>(hw);
        key_runtime.sw_stage = static_cast<VkbShader::SwStage>(sw);
        found[KeyOf(pgm, key_runtime)].push_back(item.path());
        ++files;
    }
}

constexpr const char* ProfileFile = "profile.bin";   ///< the translator profile the folder's stages were made with (raw: this build reads it)
constexpr const char* SettingsFile = "settings.txt"; ///< TranslatorSettings() of the game that opened the folder

void OpenFolder(const VkbShader::Profile& profile) {
    State& st = S();
    const auto t0 = std::chrono::steady_clock::now();
    const std::string settings = TranslatorSettings();
    u64 id = 0;
    std::filesystem::path module;
    if (!FolderId(profile, settings, id, module)) {
        std::scoped_lock lock{st.mutex};
        st.open_error = fmt::format("the engine's module file {} could not be read; the store stays in memory for this session (report: kept_in_memory_only_no_folder)", module.string());
        LOG_ERROR(Render_Vulkan, "vkb shader store: {}", st.open_error);
        return; // never ready: nothing is read or written
    }
    const auto parent = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / (MemoryPatcher::g_game_serial + "-vkb-shaders");
    const auto dir = parent / fmt::format("{:016x}", id);
    std::error_code ec;
    u32 removed = 0;
    for (const auto& other : std::filesystem::directory_iterator{parent, ec}) { // another build, card or setting: never read again
        if (other.path().filename() != dir.filename()) {
            std::filesystem::remove_all(other.path(), ec);
            ++removed;
        }
    }
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::scoped_lock lock{st.mutex};
        st.open_error = fmt::format("could not create {}: {}; the store stays in memory for this session (report: kept_in_memory_only_no_folder)", dir.string(), ec.message());
        LOG_ERROR(Render_Vulkan, "vkb shader store: {}", st.open_error);
        return;
    }
    // the build process (tlg-probe --vkb-build-store) translates with this profile and refuses a folder whose settings are not its own
    if (!WriteNow(dir / ProfileFile, &profile, sizeof(profile)) || !WriteNow(dir / SettingsFile, settings.data(), settings.size())) {
        LOG_ERROR(Render_Vulkan, "vkb shader store: could not write {} or {} in {}; the shader pre-build cannot fill this folder (the store itself works)", ProfileFile,
                  SettingsFile, dir.string());
    }
    std::unordered_map<u64, std::vector<std::filesystem::path>> found;
    std::unordered_set<std::string> names;
    u64 files = 0, stale = 0;
    IndexFolder(dir, found, files, stale, true, names);
    {
        std::scoped_lock lock{st.mutex};
        st.folder = dir;
        st.on_disk = std::move(found);
        st.indexed = std::move(names);
    }
    c_files_indexed.store(files);
    c_open_ms.store(static_cast<u64>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()));
    ready.store(true, std::memory_order_release);
    std::printf("Shader store (new engine): %s, %llu stored stages, %llu cut writes removed, %u folders of other builds or settings removed, %llu ms; "
                "translator settings %s\n",
                dir.string().c_str(), static_cast<unsigned long long>(files), static_cast<unsigned long long>(stale), removed,
                static_cast<unsigned long long>(c_open_ms.load()), settings.c_str());
    Prebuild::Start(dir); // TLG_VKB_SHADER_PREBUILD: the shipped list's stages this folder lacks, built in the background
}

/// Reads the files of `key` not read yet. The files are taken from the index under the lock and read without it (another thread that
/// looks up the same stage meanwhile does not find them and translates).
void LoadKey(u64 key) {
    State& st = S();
    std::vector<std::filesystem::path> files;
    {
        std::scoped_lock lock{st.mutex};
        const auto it = st.on_disk.find(key);
        if (it == st.on_disk.end()) {
            return;
        }
        files = std::move(it->second);
        st.on_disk.erase(it);
    }
    std::vector<std::unique_ptr<Entry>> loaded;
    for (const auto& path : files) {
        std::ifstream f{path, std::ios::binary};
        std::vector<u8> bytes{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
        auto e = f.bad() ? nullptr : Decode(bytes);
        if (!e) {
            if (c_files_bad.fetch_add(1) < 5) { // all are counted (report: files_damaged_removed)
                LOG_ERROR(Render_Vulkan, "vkb shader store: {} is damaged or of another store version; removed (the stage is translated again)", path.string());
            }
            std::error_code ec;
            std::filesystem::remove(path, ec);
            continue;
        }
        c_files_read.fetch_add(1, std::memory_order_relaxed);
        loaded.push_back(std::move(e));
    }
    std::scoped_lock lock{st.mutex};
    auto& list = st.entries[key];
    for (auto& e : loaded) {
        list.push_back(std::move(e));
    }
}

// ---- build statistics ----

bool StatsOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_SHADER_BUILD_STATS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
struct Stats {
    std::mutex mutex;
    std::unordered_map<u64, u32> per_program;
    u32 builds = 0, frame_builds = 0, before_frame = 0, slow = 0, in_slow = 0, listed = 0, while_build = 0;
    std::vector<std::pair<u32, u32>> frames; ///< microseconds, builds
    size_t judged = 0;
    std::chrono::steady_clock::time_point last_frame{}, last_report{};
};
Stats& GetStats() {
    static Stats s;
    return s;
}
void ReportLocked(Stats& s) { // the rule and the text of the normal renderer's ReportLocked (shader_recipe.cpp)
    constexpr size_t Window = 30;
    std::vector<u32> around;
    for (size_t& i = s.judged; i < s.frames.size() && i + Window < s.frames.size(); ++i) {
        around.clear();
        for (size_t j = i > Window ? i - Window : 0; j < std::min(s.frames.size(), i + Window + 1); ++j) {
            around.push_back(s.frames[j].first);
        }
        std::ranges::nth_element(around, around.begin() + around.size() / 2);
        const u32 median = around[around.size() / 2];
        if (s.frames[i].first > 2 * median && s.frames[i].first > median + 16'667) {
            ++s.slow;
            s.in_slow += s.frames[i].second;
        }
    }
    u32 pending = 0;
    for (size_t i = s.judged; i < s.frames.size(); ++i) {
        pending += s.frames[i].second;
    }
    std::printf("Shader builds while playing: %u (%u inside slow frames; %u before the first frame, %u while the background build ran, %u of programs in the shader "
                "list); %zu frames, %u slow (over twice the median of the 61 frames around them), %u builds in the last frames not judged yet; %u waits for the build "
                "gave up; new engine\n",
                s.builds, s.in_slow, s.before_frame, s.while_build, s.listed, s.frames.size(), s.slow, pending, Prebuild::WaitsGivenUp());
}

void NoteUse(u64 pgm_hash, u32 hw_stage, Entry& e) {
    if (!StatsOn() || e.used) {
        return;
    }
    e.used = true;
    std::printf("Shader used from the cache: %#018llx_0 (hw stage %u), found by the lookup\n", static_cast<unsigned long long>(pgm_hash), hw_stage);
}

/// A store switch's value: 1/on, 0/off, check; unset: `fallback`; anything else: off, with a warning
Mode ReadMode(const char* name, Mode fallback) {
    const char* e = std::getenv(name);
    if (e == nullptr) {
        return fallback;
    }
    if (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0) {
        return Mode::Off;
    }
    if (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0) {
        return Mode::On;
    }
    if (std::strcmp(e, "check") == 0) {
        return Mode::Check;
    }
    LOG_WARNING(Render_Vulkan, "{}={}: not 0, off, 1, on or check; taken as off", name, e);
    return Mode::Off;
}

Mode MemoMode() {
    static const Mode mode = ReadMode("TLG_VKB_SHADER_MEMO", Mode::On);
    return mode;
}

Mode DiskMode() {
    // unset: follows Performance > Build shaders ahead (tlg.ini shader_prebuild, on by default), the same setting as the normal renderer's pre-build
    static const Mode mode = ReadMode("TLG_VKB_SHADER_PREBUILD", TlgSettings::Get().startup_shader_prebuild ? Mode::On : Mode::Off);
    return mode;
}
} // namespace

Mode GetMode() {
    // the disk store keeps its stages in memory too, so either switch on keeps them for the session; check in either switch checks
    static const Mode mode = MemoMode() == Mode::Check || DiskMode() == Mode::Check ? Mode::Check
                             : MemoMode() == Mode::On || DiskMode() == Mode::On     ? Mode::On
                                                                                    : Mode::Off;
    return mode;
}

bool OnDisk() {
    return DiskMode() != Mode::Off;
}

void Open(const VkbShader::Profile& profile) {
    static std::once_flag once;
    if (GetMode() == Mode::Off) {
        return;
    }
    std::call_once(once, [&] {
        if (!OnDisk()) { // TLG_VKB_SHADER_MEMO only: kept in memory for this session; no folder, no writer, no shader pre-build
            std::printf("Shader store (new engine): in memory for this session (TLG_VKB_SHADER_MEMO=%s), not on disk (TLG_VKB_SHADER_PREBUILD off)\n",
                        GetMode() == Mode::Check ? "check" : "on");
            return;
        }
        std::thread{WriterLoop}.detach();
        writer_started.store(true, std::memory_order_release);
        OpenFolder(profile); // at the engine's start, before its first draw: a stat, a folder listing (a few ms)
    });
}

bool Find(const Inputs& in, Translation& out) {
    c_lookups.fetch_add(1, std::memory_order_relaxed);
    const u64 key = KeyOf(in.params.hash, in.runtime);
    const CodeId code = CodeOf(in.params.code);
    if (ready.load(std::memory_order_acquire)) {
        LoadKey(key);
    }
    State& st = S();
    std::scoped_lock lock{st.mutex};
    const auto it = st.entries.find(key);
    if (it == st.entries.end()) {
        return false;
    }
    bool same_inputs = false;
    for (const auto& e : it->second) {
        if (!SameInputs(*e, in, code)) {
            continue;
        }
        same_inputs = true;
        // the resource kinds are read through this draw's user data (and the tables it points to), as the normal renderer does
        VkbShader::Info& info = *e->result.info;
        info.user_data = in.params.user_data;
        info.pgm_base = in.params.Base();
        info.RefreshFlatBuf();
        const bool match = e->spec.Matches(info, in.runtime, in.start);
        if (match) {
            out.info = std::make_unique<VkbShader::Info>(info);
            out.runtime = e->result.runtime;
            out.end = e->result.end;
            out.spv = e->result.spv;
            out.windows = e->result.windows;
        }
        info.user_data = {};
        if (match) {
            c_hits.fetch_add(1, std::memory_order_relaxed);
            c_disk_hits.fetch_add(e->from_disk ? 1 : 0, std::memory_order_relaxed);
            if (GetMode() != Mode::Check) { // check mode translates anyway
                c_saved_ns.fetch_add(e->translate_ns, std::memory_order_relaxed);
            }
            NoteUse(in.params.hash, static_cast<u32>(in.runtime.hw_stage), *e);
            return true;
        }
    }
    if (same_inputs) { // stored with the same settings, but the draw's resources are of other kinds: a new variant, translated
        c_kind_misses.fetch_add(1, std::memory_order_relaxed);
    }
    return false;
}

bool Ready() {
    return ready.load(std::memory_order_acquire);
}

bool ForWarm(u64 pgm_hash, const VkbShader::RuntimeInfo& runtime, const VkbShader::Backend::Bindings& start, u64 salt, Translation& out) {
    const u64 key = KeyOf(pgm_hash, runtime);
    if (ready.load(std::memory_order_acquire)) {
        LoadKey(key);
    }
    State& st = S();
    std::scoped_lock lock{st.mutex};
    const auto it = st.entries.find(key);
    if (it == st.entries.end()) {
        return false;
    }
    for (const auto& e : it->second) {
        if (e->salt != salt || !(e->start == start) || std::memcmp(&e->runtime_in, &runtime, sizeof(VkbShader::RuntimeInfo)) != 0) {
            continue;
        }
        out.info = std::make_unique<VkbShader::Info>(*e->result.info);
        out.runtime = e->result.runtime;
        out.end = e->result.end;
        out.spv = e->result.spv;
        return true;
    }
    return false;
}

bool Keep(const Inputs& in, const VkbShader::Profile& profile, const VkbShader::Info& info, const VkbShader::RuntimeInfo& runtime, const VkbShader::Backend::Bindings& end,
          const std::vector<u32>& spv, const std::vector<WriteWindow::Window>& windows, u64 translate_ns) {
    if (!info.gs_copy_data.attr_map.empty() || in.runtime.hw_stage == VkbShader::HwStage::Geometry) { // the new engine has no geometry stages; never kept, counted
        c_not_kept_geometry.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto e = std::make_unique<Entry>();
    e->code = CodeOf(in.params.code);
    std::memcpy(&e->runtime_in, &in.runtime, sizeof(VkbShader::RuntimeInfo));
    e->start = in.start;
    e->salt = in.salt;
    e->result.info = std::make_unique<VkbShader::Info>(info);
    e->result.info->user_data = in.params.user_data;
    e->result.info->pgm_base = in.params.Base();
    e->result.info->RefreshFlatBuf();
    e->spec = VkbShader::StageSpecialization(*e->result.info, in.runtime, profile, in.start);
    e->result.info->user_data = {};
    e->result.info->flattened_ud_buf.clear();
    e->result.runtime = runtime;
    e->result.end = end;
    e->result.spv = spv;
    e->result.windows = windows;
    e->translate_ns = translate_ns;
    e->used = true; // its first use is this draw: logged as a build
    c_kept.fetch_add(1, std::memory_order_relaxed);
    // way A (native_vkb_shader_list.h): the shader list fixes two settings; a stage translated with another value can never come from it
    if (ShaderList::OutsideFixedValues(in.runtime)) {
        if (c_outside_fixed.fetch_add(1, std::memory_order_relaxed) < 3) { // every one is counted (report: built_outside_list_fixed_values)
            LOG_WARNING(Render_Vulkan,
                        "vkb shader store: program {:#x} (hw stage {}) translated with dx10_clamp {} and depth_before_shader {}: the shader list fixes "
                        "them to {} and {}, so it cannot hold this stage",
                        in.params.hash, static_cast<u32>(in.runtime.hw_stage), static_cast<u32>(in.runtime.props.dx10_clamp),
                        in.runtime.hw_stage == VkbShader::HwStage::Fragment ? static_cast<u32>(in.runtime.hw.fs.depth_before_shader) : 0u,
                        static_cast<u32>(ShaderList::FixedDx10Clamp), static_cast<u32>(ShaderList::FixedDepthBeforeShader));
        }
    }
    c_spv_words.fetch_add(spv.size(), std::memory_order_relaxed);
    // on disk only once the folder is open (else the store is in memory only: TLG_VKB_SHADER_MEMO without TLG_VKB_SHADER_PREBUILD, or the folder could not be opened, said
    // at the start)
    const bool to_disk = ready.load(std::memory_order_acquire);
    std::vector<u8> bytes;
    std::string name;
    if (to_disk) {
        bytes = Encode(*e);
        name = FileName(in.params.hash, in.runtime, bytes);
    }
    State& st = S();
    {
        std::scoped_lock lock{st.mutex};
        st.entries[KeyOf(in.params.hash, in.runtime)].push_back(std::move(e));
        if (to_disk) {
            st.indexed.insert(name); // a later Rescan does not read it back
        }
    }
    if (to_disk) {
        QueueWriteFile(st.folder / name, std::move(bytes));
    } else if (OnDisk()) { // the folder could not be opened (said once at the start): kept in memory only, every one counted
        c_memory_only.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

const char* Differs(const Translation& a, const VkbShader::Info& info, const VkbShader::RuntimeInfo& runtime, const VkbShader::Backend::Bindings& end,
                    const std::vector<u32>& spv, const std::vector<WriteWindow::Window>& windows) {
    if (const char* member = FirstDifference(*a.info, info); member != nullptr) {
        return member;
    }
    if (a.spv != spv) {
        return "module";
    }
    if (!(a.end == end)) {
        return "bindings";
    }
    if (!(a.windows == windows)) {
        return "write windows";
    }
    if (std::memcmp(&a.runtime, &runtime, sizeof(VkbShader::RuntimeInfo)) != 0) {
        return "runtime info";
    }
    return nullptr;
}

u64 Rescan() {
    if (!ready.load(std::memory_order_acquire)) {
        return 0;
    }
    State& st = S();
    std::filesystem::path dir;
    std::unordered_set<std::string> names;
    {
        std::scoped_lock lock{st.mutex};
        dir = st.folder;
        names = st.indexed;
    }
    std::unordered_map<u64, std::vector<std::filesystem::path>> found;
    u64 files = 0, stale = 0;
    IndexFolder(dir, found, files, stale, false, names);
    std::scoped_lock lock{st.mutex};
    for (auto& [key, paths] : found) {
        auto& list = st.on_disk[key];
        for (auto& path : paths) {
            if (st.indexed.insert(path.filename().string()).second) {
                list.push_back(std::move(path));
            }
        }
    }
    c_rescanned.fetch_add(files, std::memory_order_relaxed);
    return files;
}

bool Same(const Inputs& in, const Translation& stored, const VkbShader::Info& info, const VkbShader::RuntimeInfo& runtime, const VkbShader::Backend::Bindings& end,
          const std::vector<u32>& spv, const std::vector<WriteWindow::Window>& windows) {
    const char* differs = FirstDifference(*stored.info, info);
    const bool spv_same = stored.spv == spv, end_same = stored.end == end, windows_same = stored.windows == windows,
               runtime_same = std::memcmp(&stored.runtime, &runtime, sizeof(VkbShader::RuntimeInfo)) == 0;
    const bool same = differs == nullptr && spv_same && end_same && windows_same && runtime_same;
    (same ? c_check_same : c_check_different).fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        State& st = S();
        std::scoped_lock lock{st.mutex};
        if (st.different_logged.insert(in.params.hash).second) {
            LOG_ERROR(Render_Vulkan, "vkb shader store: the stored translation of program {:#x} (stage {}) differs from a fresh one: module {} / {} words ({}), bindings {}, "
                      "runtime {}, windows {}, first info member that differs: {}",
                      in.params.hash, static_cast<u32>(in.runtime.sw_stage), stored.spv.size(), spv.size(), spv_same ? "same" : "differ", end_same ? "same" : "differ",
                      runtime_same ? "same" : "differ", windows_same ? "same" : "differ", differs != nullptr ? differs : "none");
        }
    }
    return same;
}

bool QueueWrite(std::filesystem::path path, std::vector<u8> bytes) {
    if (!writer_started.load(std::memory_order_acquire)) { // recipes are recorded only with the store on, which starts the writer: never queued without it
        if (c_recipe_write_failed.fetch_add(1) == 0) {
            LOG_ERROR(Render_Vulkan, "vkb shader recipes: {} not written: the store's writer does not run (the store is off); refused writes are counted", path.string());
        }
        return false;
    }
    QueueWriteFile(std::move(path), std::move(bytes), true);
    return true;
}

void Flush() {
    State& st = S();
    std::unique_lock lock{st.write_mutex};
    st.flushed_cv.wait(lock, [&] { return st.writes_pending == 0; });
}

bool OpenForBuild(const std::filesystem::path& folder_in, VkbShader::Profile& profile, std::string& why) {
    std::error_code ec;
    auto folder = folder_in.lexically_normal();
    if (folder.filename().empty()) { // given with a separator at the end
        folder = folder.parent_path();
    }
    if (!std::filesystem::is_directory(folder, ec)) {
        why = fmt::format("{} is not a folder", folder.string());
        return false;
    }
    const std::vector<u8> profile_bytes = ReadAll(folder / ProfileFile);
    if (profile_bytes.size() != sizeof(VkbShader::Profile)) {
        why = fmt::format("{} has no {} of this build's size ({} bytes, this build {}): not opened by this build's game", folder.string(), ProfileFile,
                          profile_bytes.size(), sizeof(VkbShader::Profile));
        return false;
    }
    std::memcpy(static_cast<void*>(&profile), profile_bytes.data(), sizeof(profile));
    const std::vector<u8> settings_bytes = ReadAll(folder / SettingsFile);
    const std::string game_settings{settings_bytes.begin(), settings_bytes.end()};
    const std::string settings = TranslatorSettings();
    if (game_settings != settings) {
        why = fmt::format("the translator settings differ: the game's \"{}\", this process \"{}\" (start it with the game's environment)", game_settings, settings);
        return false;
    }
    u64 id = 0;
    std::filesystem::path module;
    if (!FolderId(profile, settings, id, module)) {
        why = fmt::format("the engine's module file {} could not be read", module.string());
        return false;
    }
    if (folder.filename() != fmt::format("{:016x}", id)) {
        why = fmt::format("{} is not this build's folder for its profile and settings (this build: {:016x}): another build of the engine made it", folder.string(), id);
        return false;
    }
    std::unordered_map<u64, std::vector<std::filesystem::path>> found;
    std::unordered_set<std::string> names;
    u64 files = 0, stale = 0;
    IndexFolder(folder, found, files, stale, false, names); // other build processes may be writing into the folder meanwhile
    State& st = S();
    {
        std::scoped_lock lock{st.mutex};
        st.folder = folder;
        st.on_disk = std::move(found);
        st.indexed = std::move(names);
    }
    c_files_indexed.store(files);
    std::thread{WriterLoop}.detach();
    writer_started.store(true, std::memory_order_release);
    ready.store(true, std::memory_order_release);
    return true;
}

std::filesystem::path Folder() {
    State& st = S();
    std::scoped_lock lock{st.mutex};
    return st.folder;
}

void NoteBypass() {
    c_bypassed.fetch_add(1, std::memory_order_relaxed);
}

void NoteBuild(u64 pgm_hash, u32 hw_stage) {
    if (!StatsOn()) {
        return;
    }
    auto& s = GetStats();
    std::scoped_lock lock{s.mutex};
    ++s.builds;
    ++s.frame_builds;
    s.before_frame += s.last_frame == std::chrono::steady_clock::time_point{};
    const bool listed = Prebuild::Listed(pgm_hash), running = Prebuild::Running(); // the normal renderer's words, for check_run.py
    s.listed += listed;
    s.while_build += running;
    std::printf("Shader built while playing: %#018llx_%u (hw stage %u)%s\n", static_cast<unsigned long long>(pgm_hash), s.per_program[pgm_hash]++, hw_stage,
                !listed ? (running ? ", background build still running" : "") : running ? ", in the list, background build still running" : ", in the list");
}

void NoteFrame() {
    if (!StatsOn()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    auto& s = GetStats();
    std::scoped_lock lock{s.mutex};
    if (s.last_frame != std::chrono::steady_clock::time_point{}) {
        const u32 us = static_cast<u32>(std::chrono::duration_cast<std::chrono::microseconds>(now - s.last_frame).count());
        std::printf("Frame time: %.1f ms, %u shader builds\n", us / 1000.0, s.frame_builds);
        s.frames.emplace_back(us, std::exchange(s.frame_builds, 0));
    } else {
        s.frame_builds = 0;
    }
    s.last_frame = now;
    if (now - s.last_report >= std::chrono::seconds{5}) {
        s.last_report = now;
        ReportLocked(s);
    }
}

nlohmann::json Report() {
    const auto name = [](Mode m) { return m == Mode::Off ? "off" : m == Mode::Check ? "check" : "on"; };
    const char* mode = name(GetMode());
    size_t programs = 0, waiting = 0;
    std::string error;
    std::string dir;
    State& st = S();
    {
        std::scoped_lock lock{st.mutex};
        programs = st.entries.size();
        error = st.open_error;
        dir = st.folder.string();
    }
    {
        std::scoped_lock lock{st.write_mutex};
        waiting = st.write_queue.size();
    }
    return {{"mode", mode},
            {"memo", name(MemoMode())}, // TLG_VKB_SHADER_MEMO: in memory for the session
            {"disk", name(DiskMode())}, // TLG_VKB_SHADER_PREBUILD: on disk, and the shader pre-build
            {"folder", dir},
            {"open_error", error},
            {"open_ms", c_open_ms.load()},
            {"lookups", c_lookups.load()},
            {"hits", c_hits.load()},
            {"hits_from_disk", c_disk_hits.load()},
            {"kept", c_kept.load()},
            {"kept_in_memory_only_no_folder", c_memory_only.load()},
            {"built_outside_list_fixed_values", c_outside_fixed.load()},
            {"programs_in_memory", programs},
            {"files_indexed_at_open", c_files_indexed.load()},
            {"files_found_by_rescan", c_rescanned.load()},
            {"prebuild", Prebuild::Report()},
            {"pipeline_warm_check", PipelineWarm::Report()},
            {"files_read", c_files_read.load()},
            {"files_damaged_removed", c_files_bad.load()},
            {"files_written", c_written.load()},
            {"files_write_failed", c_write_failed.load()},
            {"recipe_files_written", c_recipe_written.load()},
            {"recipe_files_write_failed", c_recipe_write_failed.load()},
            {"files_waiting", waiting},
            {"not_kept_geometry", c_not_kept_geometry.load()},
            {"misses_other_resource_kinds", c_kind_misses.load()},
            {"bypassed_for_debug_tools", c_bypassed.load()},
            {"module_mb_kept", static_cast<double>(c_spv_words.load()) * 4.0 / 1048576.0},
            {"translate_ms_saved", static_cast<double>(c_saved_ns.load()) / 1e6},
            {"check_same", c_check_same.load()},
            {"check_different", c_check_different.load()},
            {"translator_use_slot_misses", VkbShader::IR::UseSlotMisses()}}; // TLG_VKB_TRANSLATE_FAST: microinstruction.h
}

int CompareStores(const std::filesystem::path& dir_a, const std::filesystem::path& dir_b) {
    // test tool (tlg-probe --vkb-compare-stores A B; issue 160 step 2b): the stages of two stores built from the same recipes (for example by
    // two builds, or with TLG_VKB_TRANSLATE_FAST=0 and 1) are paired by their inputs and compared member by member (Differs)
    const auto load = [](const std::filesystem::path& dir, std::vector<std::unique_ptr<Entry>>& out, u64& bad) {
        std::error_code ec;
        std::vector<std::filesystem::path> files;
        for (std::filesystem::directory_iterator it{dir, ec}, end; !ec && it != end; it.increment(ec)) {
            if (it->path().extension() == ".vst") {
                files.push_back(it->path());
            }
        }
        std::sort(files.begin(), files.end());
        for (const auto& file : files) {
            if (auto e = Decode(ReadAll(file))) {
                out.push_back(std::move(e));
            } else {
                ++bad;
            }
        }
        return !ec;
    };
    std::vector<std::unique_ptr<Entry>> a, b;
    u64 bad_a = 0, bad_b = 0;
    if (!load(dir_a, a, bad_a) || !load(dir_b, b, bad_b)) {
        std::printf("vkb-compare-stores: a folder cannot be read\n");
        return 3;
    }
    const auto same_inputs = [](const Entry& x, const Entry& y) {
        return x.result.info->pgm_hash == y.result.info->pgm_hash && x.code == y.code && x.salt == y.salt && x.start == y.start &&
               std::memcmp(&x.runtime_in, &y.runtime_in, sizeof(VkbShader::RuntimeInfo)) == 0;
    };
    const auto same_kinds = [](const VkbShader::StageSpecialization& x, const VkbShader::StageSpecialization& y) {
        // StageSpecialization's == reads y's lists by x's sizes and uses only y's bits: the sizes and bits are compared first, so a
        // translation that changed the number of resources is reported and never read past a list's end
        return x.bitset == y.bitset && x.vs_attribs.size() == y.vs_attribs.size() && x.buffers.size() == y.buffers.size() &&
               x.images.size() == y.images.size() && x.fmasks.size() == y.fmasks.size() && x.samplers.size() == y.samplers.size() && x == y;
    };
    std::vector<bool> paired(b.size(), false);
    u64 pairs = 0, different = 0, only_a = 0, ns_a = 0, ns_b = 0, words = 0;
    for (const auto& x : a) {
        // A store may hold several stages with the same inputs that differ only in the kinds of the resources the draw used (Find picks
        // one by spec.Matches). Files are read in name order and a name holds a hash of the whole entry, its translation time too, so the
        // order of such variants changes from run to run: pair a stage first with the one of the same kinds, and only when there is
        // none with another one of the same inputs (then the kinds are reported as the difference).
        const auto find = [&](bool by_kinds) {
            size_t j = 0;
            while (j < b.size() && (paired[j] || !same_inputs(*x, *b[j]) || (by_kinds && !same_kinds(x->spec, b[j]->spec)))) {
                ++j;
            }
            return j;
        };
        size_t j = find(true);
        if (j == b.size()) {
            j = find(false);
        }
        if (j == b.size()) {
            ++only_a;
            continue;
        }
        paired[j] = true;
        ++pairs;
        const Entry& y = *b[j];
        ns_a += x->translate_ns;
        ns_b += y.translate_ns;
        words += x->result.spv.size();
        const char* member = Differs(x->result, *y.result.info, y.result.runtime, y.result.end, y.result.spv, y.result.windows);
        if (member == nullptr && !same_kinds(x->spec, y.spec)) {
            member = "resource kinds";
        }
        if (member != nullptr && different++ < 10) {
            std::printf("vkb-compare-stores: %016llx (stage %u) differs: %s\n", static_cast<unsigned long long>(x->result.info->pgm_hash),
                        static_cast<u32>(x->runtime_in.hw_stage), member);
        }
    }
    const u64 only_b = static_cast<u64>(std::count(paired.begin(), paired.end(), false));
    std::printf("vkb-compare-stores: A %zu stages (%llu unreadable), B %zu (%llu unreadable); %llu pairs with the same inputs, %llu different, %llu only in A, "
                "%llu only in B; %llu module words compared; translation A %.1f ms, B %.1f ms (B/A %.3f)\n",
                a.size(), static_cast<unsigned long long>(bad_a), b.size(), static_cast<unsigned long long>(bad_b), static_cast<unsigned long long>(pairs),
                static_cast<unsigned long long>(different), static_cast<unsigned long long>(only_a), static_cast<unsigned long long>(only_b),
                static_cast<unsigned long long>(words), ns_a / 1e6, ns_b / 1e6, ns_a != 0 ? static_cast<double>(ns_b) / static_cast<double>(ns_a) : 0.0);
    return different == 0 && only_a == 0 && only_b == 0 && bad_a == 0 && bad_b == 0 && pairs != 0 ? 0 : 1;
}

} // namespace Core::Vkb::ShaderStore

extern "C" int tlg_vkb_compare_stores(const char* dir_a, const char* dir_b) {
    return Core::Vkb::ShaderStore::CompareStores(std::filesystem::u8path(dir_a), std::filesystem::u8path(dir_b));
}
