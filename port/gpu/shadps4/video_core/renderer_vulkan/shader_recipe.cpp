// SPDX-FileCopyrightText: Copyright 2026 tlg_pc contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// tlg: shader recipes, see shader_recipe.h. Ported from deadinside28/bloodborne_pc, thanks to
// @deadinside28.

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <miniz.h>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "common/hash.h"
#include "common/logging/log.h"
#include "common/serdes.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/amdgpu/regs_shader.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/shader_fields.h"
#include "video_core/renderer_vulkan/shader_list.h"
#include "video_core/renderer_vulkan/shader_recipe.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "tlg_overlay.h"

// The runtime's file layer (src/runtime.h): a guest path on the host.
extern "C" int runtime_file_translate(const char* guest, char* out, size_t size);

namespace Vulkan::ShaderRecipe {

namespace {

constexpr u32 Magic = 0x52474c54; // TLGR
/// 1: struct sizes only; 2: the layout id (LayoutId()) after the sizes. All three are read.
/// 3 (PortableVersion): the portable id (PortableId()) in place of the layout id, the run-time
/// state and the binding start numbers in the portable form (shader_fields.h), the program's
/// length and CRC-32, and a fetch shader as its vertex input layout. The game writes it, without
/// the program, into its log for a shader the list lacked (LogUnlisted); log_recipes.py adds the
/// program from the player's game files back.
constexpr u32 Version = 2, PortableVersion = 3;

/// Guest memory the compiler read on this thread since Begin(), by address.
thread_local bool recording = false;
thread_local std::map<u64, std::vector<u8>> regions;

void Record(const void* address, size_t size) {
    if (!recording || !address) {
        return;
    }
    if (size == 0) {
        // A fetch shader: its length is where it returns (s_setpc_b64), as ParseFetchShader.
        const auto* code = static_cast<const u32*>(address);
        Shader::Gcn::GcnCodeSlice slice(code, code + std::numeric_limits<u32>::max());
        Shader::Gcn::GcnDecodeContext decoder;
        while (!slice.atEnd() && size < 4096) {
            const auto inst = decoder.decodeInstruction(slice);
            size += inst.length;
            if (inst.opcode == Shader::Gcn::Opcode::S_SETPC_B64) {
                break;
            }
        }
    }
    auto& bytes = regions[reinterpret_cast<u64>(address)];
    if (bytes.size() < size) {
        bytes.resize(size);
        AmdGpu::ReadShaderCode(address, bytes.data(), size);
    }
}

struct Writer {
    std::vector<u8> out;
    template <typename T>
    void Pod(const T& value) {
        const auto* p = reinterpret_cast<const u8*>(&value);
        out.insert(out.end(), p, p + sizeof(T));
    }
    void Bytes(const void* data, u64 size) {
        Pod(size);
        const auto* p = static_cast<const u8*>(data);
        out.insert(out.end(), p, p + size);
    }
    void Words(std::span<const u32> words) {
        Bytes(words.data(), words.size_bytes());
    }
};

struct Reader {
    std::span<const u8> in;
    size_t pos = 0;
    bool ok = true;
    template <typename T>
    T Pod() {
        T value{};
        if (pos + sizeof(T) > in.size()) {
            ok = false;
            return value;
        }
        std::memcpy(&value, in.data() + pos, sizeof(T));
        pos += sizeof(T);
        return value;
    }
    std::vector<u8> Bytes() {
        const auto size = Pod<u64>();
        if (!ok || size > in.size() - pos) {
            ok = false;
            return {};
        }
        std::vector<u8> bytes(in.begin() + pos, in.begin() + pos + size);
        pos += size;
        return bytes;
    }
    std::vector<u32> Words() {
        const auto bytes = Bytes();
        std::vector<u32> words(bytes.size() / sizeof(u32));
        std::memcpy(words.data(), bytes.data(), words.size() * sizeof(u32));
        return words;
    }
};

struct Recipe {
    Shader::HwStage hw_stage;
    Shader::SwStage sw_stage;
    u64 pgm_hash;
    u64 perm_idx;
    u64 pgm_base;
    std::vector<u32> user_data;
    std::vector<u32> vs_copy;
    std::vector<u32> flat;
    Shader::RuntimeInfo runtime;
    Shader::Backend::Bindings start;
    std::vector<std::pair<u64, std::vector<u8>>> regions;
    /// tlg: shader list recipes: the vertex input layout in place of the fetch shader's code.
    std::optional<Shader::Gcn::FetchShaderData> fetch;
};

/// A recipe's header (magic, version, struct sizes, layout id); false with the reason in `why`
/// when this build cannot read the recipe. `other_build` (optional) tells a recipe of another
/// build (version, struct sizes or layout) from a damaged one.
bool ReadHeader(Reader& r, const char*& why, bool* other_build = nullptr,
                u32* version_out = nullptr) {
    if (other_build) {
        *other_build = false;
    }
    const u32 magic = r.Pod<u32>();
    const u32 version = r.Pod<u32>();
    if (version_out) {
        *version_out = version;
    }
    const u32 runtime_size = r.Pod<u32>();
    const u32 bindings_size = r.Pod<u32>();
    // Version 1 recipes have no layout id: read as before, on the struct sizes alone.
    // Version 3: the portable id.
    const u64 layout = version >= 2 ? r.Pod<u64>() : 0;
    if (!r.ok) {
        why = "is damaged";
        return false;
    }
    if (magic != Magic) {
        why = "is not a recipe";
        return false;
    }
    if (version != 1 && version != 2 && version != PortableVersion) {
        why = "has a recipe version this build does not read";
        if (other_build) {
            *other_build = true;
        }
        return false;
    }
    if (version == PortableVersion) {
        if (layout != PortableId()) {
            why = "was made with other field lists (portable id differs)";
            if (other_build) {
                *other_build = true;
            }
            return false;
        }
        return true;
    }
    if (runtime_size != sizeof(Shader::RuntimeInfo) ||
        bindings_size != sizeof(Shader::Backend::Bindings)) {
        why = "was made by a build with other struct sizes";
        if (other_build) {
            *other_build = true;
        }
        return false;
    }
    if (version >= 2 && layout != LayoutId() && LayoutCheck()) {
        why = "was made by a build with another struct layout (layout id differs)";
        if (other_build) {
            *other_build = true;
        }
        return false;
    }
    return true;
}

/// A recorded recipe; nullopt with the reason in `why` when it cannot be read by this build.
std::optional<Recipe> Parse(std::span<const u8> data, const char*& why) {
    Reader r{data};
    u32 version = 0;
    if (!ReadHeader(r, why, nullptr, &version)) {
        return std::nullopt;
    }
    Recipe recipe{};
    recipe.hw_stage = r.Pod<Shader::HwStage>();
    recipe.sw_stage = r.Pod<Shader::SwStage>();
    recipe.pgm_hash = r.Pod<u64>();
    recipe.perm_idx = r.Pod<u64>();
    recipe.pgm_base = r.Pod<u64>();
    recipe.user_data = r.Words();
    recipe.vs_copy = r.Words();
    recipe.flat = r.Words();
    u32 length = 0, crc = 0;
    if (version == PortableVersion) {
        const auto runtime = r.Bytes();
        const auto start = r.Bytes();
        if (!r.ok || !RuntimeFromPortable(runtime, recipe.runtime) ||
            !BindingsFromPortable(start, recipe.start)) {
            why = "is damaged";
            return std::nullopt;
        }
        length = r.Pod<u32>();
        crc = r.Pod<u32>();
        if (r.Pod<u8>() != 0) {
            auto& fetch = recipe.fetch.emplace();
            fetch.size = r.Pod<u32>();
            fetch.vertex_offset_sgpr = r.Pod<s8>();
            fetch.instance_offset_sgpr = r.Pod<s8>();
            const u8 attributes = r.Pod<u8>();
            for (u8 i = 0; r.ok && i < attributes; ++i) {
                auto& a = fetch.attributes.emplace_back();
                a.semantic = r.Pod<u8>();
                a.dest_vgpr = r.Pod<u8>();
                a.num_elements = r.Pod<u8>();
                a.sgpr_base = r.Pod<u8>();
                a.dword_offset = r.Pod<u8>();
                a.instance_data = r.Pod<u8>();
                a.inst_offset = r.Pod<u8>();
                a.data_format = r.Pod<u8>();
                a.num_format = r.Pod<u8>();
            }
        }
    } else {
        recipe.runtime = r.Pod<Shader::RuntimeInfo>();
        recipe.start = r.Pod<Shader::Backend::Bindings>();
    }
    const auto count = r.Pod<u64>();
    for (u64 i = 0; r.ok && i < count; ++i) {
        const auto address = r.Pod<u64>();
        recipe.regions.emplace_back(address, r.Bytes());
    }
    if (!r.ok || recipe.user_data.size() != Shader::ShaderParams::NumShaderUserData) {
        why = "is damaged";
        return std::nullopt;
    }
    if (version == PortableVersion) { // the program, as the log line named it
        const auto program = std::ranges::find(recipe.regions, recipe.pgm_base,
                                               &std::pair<u64, std::vector<u8>>::first);
        if (program == recipe.regions.end() || program->second.size() != length ||
            u32(mz_crc32(MZ_CRC32_INIT, program->second.data(), length)) != crc) {
            why = "has no program, or another one than it names (length, CRC-32)";
            return std::nullopt;
        }
    }
    recipe.runtime.hw.gs.vs_copy = {};
    return recipe;
}

/// A shader list recipe with its program read from the game files; nullopt when the game files
/// do not hold it.
std::optional<Recipe> FromList(const ShaderList::Entry& e, ShaderList::GameFiles& files,
                               const char*& why) {
    if (e.user_data.size() != Shader::ShaderParams::NumShaderUserData ||
        e.runtime.size() != sizeof(Shader::RuntimeInfo) ||
        e.start.size() != sizeof(Shader::Backend::Bindings)) {
        why = "has a damaged entry";
        return std::nullopt;
    }
    auto program = files.Program(e);
    if (!program) {
        why = "is not in the game files";
        return std::nullopt;
    }
    Recipe recipe{};
    recipe.hw_stage = Shader::HwStage(e.hw_stage);
    recipe.sw_stage = Shader::SwStage(e.sw_stage);
    recipe.pgm_hash = e.pgm_hash;
    recipe.perm_idx = e.perm_idx;
    recipe.pgm_base = e.pgm_base;
    recipe.user_data = e.user_data;
    recipe.vs_copy = e.vs_copy;
    recipe.flat = e.flat;
    std::memcpy(&recipe.runtime, e.runtime.data(), sizeof(recipe.runtime));
    std::memcpy(&recipe.start, e.start.data(), sizeof(recipe.start));
    recipe.runtime.hw.gs.vs_copy = {};
    recipe.regions = e.regions;
    recipe.regions.emplace_back(e.pgm_base, std::move(*program));
    recipe.fetch = e.fetch;
    return recipe;
}

std::string RecipeName(u64 pgm_hash, size_t perm_idx) {
    return fmt::format("{:#018x}_{}", pgm_hash, perm_idx);
}

/// Guest pages mapped at their own addresses in the child, for one recipe.
class GuestMapping {
public:
    explicit GuestMapping(const std::vector<std::pair<u64, std::vector<u8>>>& regions) {
        constexpr u64 Page = 0x10000; // Windows' allocation granularity, also fine on Linux
        std::map<u64, u64> spans; // start -> end, page aligned and merged
        for (const auto& [address, bytes] : regions) {
            u64 start = address & ~(Page - 1);
            u64 end = (address + bytes.size() + Page - 1) & ~(Page - 1);
            auto it = spans.upper_bound(start);
            if (it != spans.begin() && std::prev(it)->second >= start) {
                --it;
                start = it->first;
            }
            while (it != spans.end() && it->first <= end) {
                end = std::max(end, it->second);
                it = spans.erase(it);
            }
            spans.emplace(start, end);
        }
        for (const auto& [start, end] : spans) {
            if (!Map(start, end - start)) {
                ok = false;
                return;
            }
            mapped.emplace_back(start, end - start);
        }
        for (const auto& [address, bytes] : regions) {
            std::memcpy(reinterpret_cast<void*>(address), bytes.data(), bytes.size());
        }
    }
    ~GuestMapping() {
        for (const auto& [address, size] : mapped) {
#ifdef _WIN32
            VirtualFree(reinterpret_cast<void*>(address), 0, MEM_RELEASE);
#else
            munmap(reinterpret_cast<void*>(address), size);
#endif
        }
    }
    bool ok = true;

private:
    static bool Map(u64 address, u64 size) {
#ifdef _WIN32
        return VirtualAlloc(reinterpret_cast<void*>(address), size, MEM_RESERVE | MEM_COMMIT,
                            PAGE_READWRITE) == reinterpret_cast<void*>(address);
#else
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
        void* p = mmap(reinterpret_cast<void*>(address), size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == MAP_FAILED) {
            return false;
        }
        if (p != reinterpret_cast<void*>(address)) { // a kernel without MAP_FIXED_NOREPLACE
            munmap(p, size);
            return false;
        }
        return true;
#endif
    }
    std::vector<std::pair<u64, u64>> mapped;
};

std::vector<u8> ReadFile(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{file}, {}};
}

bool WriteFile(const std::filesystem::path& path, const void* data, size_t size) {
    auto temp = path;
#ifdef _WIN32
    temp += "." + std::to_string(GetCurrentProcessId()) + ".tmp"; // children write in parallel
#else
    temp += "." + std::to_string(getpid()) + ".tmp"; // children write in parallel
#endif
    {
        std::ofstream file{temp, std::ios::binary};
        file.write(static_cast<const char*>(data), std::streamsize(size));
        if (!file) {
            return false;
        }
    }
    std::error_code error;
    std::filesystem::rename(temp, path, error);
    return !error;
}

/// Programs (pgm_hash) with any shader binary in `dir`.
std::unordered_set<u64> CachedPrograms(const std::filesystem::path& dir) {
    std::unordered_set<u64> programs;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator{dir, error}) {
        if (entry.path().extension() == ".spv") {
            programs.insert(std::strtoull(entry.path().stem().string().c_str(), nullptr, 16));
        }
    }
    return programs;
}

struct RecipeFile {
    u64 pgm_hash;
    size_t perm_idx;
    std::filesystem::path path;
    bool operator<(const RecipeFile& other) const {
        return std::tie(pgm_hash, perm_idx) < std::tie(other.pgm_hash, other.perm_idx);
    }
};

/// Recipes of programs `cache_dir` lacks, per program in permutation order.
std::vector<RecipeFile> MissingRecipes(const std::filesystem::path& cache_dir,
                                       const std::filesystem::path& recipe_dir) {
    const auto cached = CachedPrograms(cache_dir);
    std::vector<RecipeFile> missing;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator{recipe_dir, error}) {
        if (entry.path().extension() != ".in") {
            continue;
        }
        const auto stem = entry.path().stem().string();
        const auto sep = stem.find('_');
        if (sep == std::string::npos) {
            continue;
        }
        const u64 hash = std::strtoull(stem.c_str(), nullptr, 16);
        if (!cached.contains(hash)) {
            missing.push_back({hash, size_t(std::strtoull(stem.c_str() + sep + 1, nullptr, 10)),
                               entry.path()});
        }
    }
    std::sort(missing.begin(), missing.end());
    return missing;
}

/// The recipe folder's pipeline keys are raw GraphicsPipelineKeys of the build that recorded its
/// recipes, and carry no layout id of their own: they are used only when no recipe of the folder
/// is of another build. Empty when they can be used, else the reason.
std::string FolderKeysRefused(const std::filesystem::path& recipe_dir) {
    u32 other = 0;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator{recipe_dir, error}) {
        if (entry.path().extension() != ".in") {
            continue;
        }
        std::array<u8, 32> head{};
        std::ifstream in{entry.path(), std::ios::binary};
        in.read(reinterpret_cast<char*>(head.data()), head.size());
        Reader r{std::span<const u8>{head.data(), size_t(in.gcount())}};
        const char* why = "";
        bool other_build = false;
        other += !ReadHeader(r, why, &other_build) && other_build;
    }
    return other ? fmt::format("{} recipes of the folder were made by another build", other)
                 : std::string{};
}

/// LogUnlisted's line for a shader the shader list lacks (defined with the background state).
void LogUnlisted(const Shader::Info& info, const std::map<u64, std::vector<u8>>& read,
                 const Shader::RuntimeInfo& lookup_runtime, const Shader::Backend::Bindings& start,
                 size_t perm_idx);

} // namespace

void Begin() {
    regions.clear();
    recording = true;
    Shader::guest_read_hook = &Record;
}

std::vector<u8> Take(const Shader::Info& info, std::span<const u32> code,
                     const Shader::RuntimeInfo& lookup_runtime,
                     const Shader::Backend::Bindings& start, size_t perm_idx) {
    Shader::guest_read_hook = nullptr;
    recording = false;
    NoteBuild(info.pgm_hash, perm_idx, u32(info.hw_stage));
    auto read = std::move(regions);
    regions.clear();
    if (info.translation_failed || (Storage::RecipeDir().empty() && !ShaderList::Enabled())) {
        return {};
    }
    // Motion vertex shaders embed session-local buffer device addresses (never cached either).
    if (lookup_runtime.hw_stage == Shader::HwStage::Vertex &&
        lookup_runtime.hw.vs.motion_vectors) {
        return {};
    }
    const auto ud = info.UserData();
    if (ud.size() != Shader::ShaderParams::NumShaderUserData) {
        return {};
    }
    // The program itself, at its guest address: the info's base and inline constants use it.
    auto& program = read[info.pgm_base];
    if (program.size() < code.size_bytes()) {
        program.resize(code.size_bytes());
        std::memcpy(program.data(), code.data(), code.size_bytes());
    }
    LogUnlisted(info, read, lookup_runtime, start, perm_idx);
    if (Storage::RecipeDir().empty()) {
        return {};
    }

    Writer w;
    w.Pod(Magic);
    w.Pod(Version);
    w.Pod(u32(sizeof(Shader::RuntimeInfo)));
    w.Pod(u32(sizeof(Shader::Backend::Bindings)));
    w.Pod(LayoutId());
    w.Pod(info.hw_stage);
    w.Pod(info.sw_stage);
    w.Pod(info.pgm_hash);
    w.Pod(u64(perm_idx));
    w.Pod(u64(info.pgm_base));
    w.Words(ud);
    w.Words(lookup_runtime.hw_stage == Shader::HwStage::Geometry ? lookup_runtime.hw.gs.vs_copy
                                                                  : std::span<const u32>{});
    w.Words(info.FlatUserData());
    auto runtime = lookup_runtime;
    runtime.hw.gs.vs_copy = {};
    w.Pod(runtime);
    w.Pod(start);
    w.Pod(u64(read.size()));
    for (const auto& [address, bytes] : read) {
        w.Pod(address);
        w.Bytes(bytes.data(), bytes.size());
    }
    return std::move(w.out);
}

void Save(u64 pgm_hash, size_t perm_idx, std::vector<u8>&& recipe) {
    if (recipe.empty()) {
        return;
    }
    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderInput,
                                       RecipeName(pgm_hash, perm_idx), std::move(recipe));
}

namespace {

/// The import child's exit code (-1 when it could not start). Arguments after the executable.
/// tlg: the import child's progress (ImportChild): before each shader it writes which one it builds
/// (program hash and permutation) into this file of the cache, so the parent knows the shader a
/// child stopped on.
std::filesystem::path ProgressFile(const std::filesystem::path& cache_dir, u32 part) {
    return cache_dir / fmt::format("import-progress-{}.txt", part);
}

/// Programs a build process stopped on (crash, failed assertion, no progress): one line each
/// (program hash, permutation, how it stopped). The child leaves them out, so a shader that cannot
/// be built costs one restart once per cache, not one per start.
std::filesystem::path LeftOutFile(const std::filesystem::path& cache_dir) {
    return cache_dir / "import-left-out.txt";
}

std::string ReadText(const std::filesystem::path& path) {
    const auto bytes = ReadFile(path);
    return {bytes.begin(), bytes.end()};
}

/// TLG_SHADER_PREBUILD_RESTART (default 1): a build process that stops early (crash, failed
/// assertion, or no progress for TLG_SHADER_PREBUILD_ENTRY_S seconds on one shader, default 30)
/// is started again after the shader it stopped on, which is left out. 0: the old way, the
/// build ends there (the rest of its share stays unbuilt).
bool RestartOn() {
    const char* env = std::getenv("TLG_SHADER_PREBUILD_RESTART");
    return !(env && std::strcmp(env, "0") == 0);
}

long EntryLimitSeconds() {
    long limit_s = 30;
    if (const char* env = std::getenv("TLG_SHADER_PREBUILD_ENTRY_S"); env && *env) {
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end != env && *end == '\0' && value >= 1 && value <= 3600) {
            limit_s = value;
        }
    }
    return limit_s;
}

struct ChildEnd {
    int code = -1;   ///< exit code; -1 when it did not exit by itself
    int signal = 0;  ///< Linux: the signal that ended it
    bool hung = false; ///< killed: no progress for the entry limit
};

/// How far each build process is (its progress file: shaders done and in its share), for the
/// corner note "Building shaders NN%". A restarted process counts on from the runs before it.
struct BuildCount {
    std::atomic<u32> done{0}, total{0};
};
std::array<BuildCount, 32> build_counts;

void ShowBuildProgress() {
    u32 done = 0, total = 0;
    for (const auto& count : build_counts) {
        done += count.done;
        total += count.total;
    }
    TlgOverlay::SetShaderBuild(done, std::max(total, 1u));
}

/// One run of the child. With `limit_s` > 0 it is killed when its progress file names the same
/// shader for longer than that. `count` gets its progress on top of `base` (runs before it).
ChildEnd RunChildOnce(const std::vector<std::filesystem::path>& args,
                      const std::filesystem::path& progress, long limit_s, BuildCount& count,
                      u32 base) {
    ChildEnd end;
    std::string last;
    auto changed = std::chrono::steady_clock::now();
    // True when the child has worked on one shader for longer than the limit.
    const auto stalled = [&] {
        const auto now = std::chrono::steady_clock::now();
        if (auto text = ReadText(progress); text != last) {
            last = std::move(text);
            changed = now;
            unsigned long long hash = 0, perm = 0;
            unsigned done = 0, total = 0;
            if (std::sscanf(last.c_str(), "%llx %llu %u %u", &hash, &perm, &done, &total) == 4) {
                count.total = base + total;
                count.done = base + done;
                ShowBuildProgress();
            }
            return false;
        }
        return limit_s > 0 && !last.empty() && now - changed > std::chrono::seconds{limit_s};
    };
#ifdef _WIN32
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring command = L"\"" + std::wstring{exe} + L"\" --import-shaders";
    for (const auto& arg : args) {
        command += L" \"" + arg.wstring() + L"\"";
    }
    // The game's job object: its build processes end with the game, also when it crashes or is
    // ended from outside (the job closes with the game's last handle).
    static const HANDLE job = [] {
        HANDLE h = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (h != nullptr && !SetInformationJobObject(h, JobObjectExtendedLimitInformation, &limits,
                                                     sizeof(limits))) {
            CloseHandle(h);
            h = nullptr;
        }
        if (h == nullptr) {
            std::printf("Shader prebuild: no job object (error %lu): build processes may outlive "
                        "the game\n",
                        GetLastError());
        }
        return h;
    }();
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    // Suspended until it is in the job: it never runs outside it.
    if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr,
                       nullptr, &si, &pi)) {
        if (job != nullptr && !AssignProcessToJobObject(job, pi.hProcess)) {
            std::printf("Shader prebuild: a build process is not in the game's job object (error "
                        "%lu): it may outlive the game\n",
                        GetLastError());
        }
        if (ResumeThread(pi.hThread) == DWORD(-1)) { // it would wait suspended forever
            std::printf("Shader prebuild: a build process could not be started (error %lu)\n",
                        GetLastError());
            TerminateProcess(pi.hProcess, 1);
        }
        for (;;) {
            const DWORD wait = WaitForSingleObject(pi.hProcess, 100);
            if (wait != WAIT_TIMEOUT) {
                break;
            }
            if (stalled()) {
                TerminateProcess(pi.hProcess, 1);
                WaitForSingleObject(pi.hProcess, INFINITE);
                end.hung = true;
                break;
            }
        }
        DWORD exit_code = 1;
        GetExitCodeProcess(pi.hProcess, &exit_code);
        end.code = end.hung ? -1 : int(exit_code);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
#else
    std::vector<std::string> strings;
    for (const auto& arg : args) {
        strings.push_back(arg.string());
    }
    std::vector<char*> argv{const_cast<char*>("tlg-probe"), const_cast<char*>("--import-shaders")};
    for (auto& arg : strings) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    std::fflush(stdout);
    // posix_spawn, not fork: the game runs meanwhile, and a fork copies its (large) memory map
    // while every other thread of the game stops. In its own process group: a stop signal sent to
    // the game's group (Ctrl+C in a terminal, a launcher or test tool ending the game) does not
    // reach it, so it is not taken for a shader it cannot build; it ends with the game through its
    // parent-death signal (ImportChild).
    posix_spawnattr_t attr;
    const bool attr_ok = posix_spawnattr_init(&attr) == 0;
    if (!attr_ok || posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP) != 0 ||
        posix_spawnattr_setpgroup(&attr, 0) != 0) {
        std::printf("Shader prebuild: a build process gets no process group of its own\n");
    }
    // The game's process id: the child checks that its parent is still this game once its
    // parent-death signal is set (a game that ended in between would not send it).
    std::string parent = fmt::format("TLG_SHADER_PREBUILD_PARENT={}", getpid());
    std::vector<char*> envp{parent.data()};
    for (char** e = environ; *e != nullptr; ++e) {
        if (std::strncmp(*e, "TLG_SHADER_PREBUILD_PARENT=", 27) != 0) {
            envp.push_back(*e);
        }
    }
    envp.push_back(nullptr);
    pid_t pid = 0;
    const int spawned =
        posix_spawn(&pid, "/proc/self/exe", nullptr, attr_ok ? &attr : nullptr, argv.data(), envp.data());
    if (attr_ok) {
        posix_spawnattr_destroy(&attr);
    }
    if (spawned == 0) {
        int status = 0;
        for (;;) {
            const pid_t done = waitpid(pid, &status, WNOHANG);
            if (done == pid || (done < 0 && errno != EINTR)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
            if (stalled()) {
                kill(pid, SIGKILL);
                while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
                }
                end.hung = true;
                break;
            }
        }
        if (!end.hung) {
            end.code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            end.signal = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
        }
    }
#endif
    return end;
}

std::mutex left_out_mutex;

/// Runs the import child (tlg-probe --import-shaders ARGS) for `cache_dir`, share `part`, until it
/// ends well. A child that stops early on a shader (one the compiler cannot take: a failed
/// assertion, a crash, a hang) is started again without it, at most 32 times; the
/// shaders it built before are in the cache and are not built again. 0 when the last run ended
/// well.
int RunChild(const std::vector<std::filesystem::path>& args, const std::filesystem::path& cache_dir,
             u32 part) {
    const bool restart = RestartOn();
    const long limit_s = restart ? EntryLimitSeconds() : 0;
    const auto progress = ProgressFile(cache_dir, part);
    auto& count = build_counts[part % build_counts.size()];
    std::error_code error;
    for (u32 runs = 1;; ++runs) {
        std::filesystem::remove(progress, error);
        const auto end = RunChildOnce(args, progress, limit_s, count, count.done);
        const auto at = ReadText(progress);
        std::filesystem::remove(progress, error);
        if (end.code == 0) {
            count.done = count.total.load();
            ShowBuildProgress();
            return 0;
        }
#ifndef _WIN32
        // Stopped from outside (the machine shuts down, someone ended it): not the shader's fault,
        // nothing is left out.
        if (end.signal == SIGTERM || end.signal == SIGINT || end.signal == SIGHUP) {
            std::printf("Shader prebuild: a build process was stopped from outside (signal %d); "
                        "its share of the build ends here\n",
                        end.signal);
            return 1;
        }
#endif
        const auto how = end.hung     ? fmt::format("no progress for {} s", limit_s)
                         : end.signal ? fmt::format("signal {}", end.signal)
                                      : fmt::format("exit {}", end.code);
        // The progress names the shader only while the child builds one: a child that stops
        // before its first shader or after its last is not started again.
        u64 pgm_hash = 0, perm_idx = 0;
        const bool named = std::sscanf(at.c_str(), "%llx %llu",
                                       reinterpret_cast<unsigned long long*>(&pgm_hash),
                                       reinterpret_cast<unsigned long long*>(&perm_idx)) == 2;
        if (!restart || !named || runs > 32) {
            std::printf("Shader prebuild: a build process stopped (%s)%s; its share of the build "
                        "ends here\n",
                        how.c_str(),
                        !named      ? ""
                        : !restart  ? " (restart off: TLG_SHADER_PREBUILD_RESTART=0)"
                                    : ", 32 restarts already");
            return end.code != 0 ? end.code : 1;
        }
        {
            std::scoped_lock lock{left_out_mutex};
            std::ofstream out{LeftOutFile(cache_dir), std::ios::binary | std::ios::app};
            out << fmt::format("{:#018x} {} {}\n", pgm_hash, perm_idx, how);
        }
        std::printf("Shader prebuild: a build process stopped (%s) on shader %#018llx_%llu; that "
                    "program is left out (import-left-out.txt of the cache) and the build goes on\n",
                    how.c_str(), static_cast<unsigned long long>(pgm_hash),
                    static_cast<unsigned long long>(perm_idx));
        LOG_WARNING(Render_Vulkan,
                    "tlg: shader build process stopped ({}) on {:#018x}_{}; left out, restarted",
                    how, pgm_hash, perm_idx);
    }
}

/// tlg: the background import (shader_prebuild on) and the counts of shaders built while
/// playing. Never freed: the import thread may outlive the game's shutdown.
struct Background {
    std::mutex mutex;
    std::condition_variable done;
    bool running = false;
    std::vector<u64> listed; ///< programs of the shader list, sorted
    std::chrono::steady_clock::time_point started{};
    // Builds while playing:
    u32 builds = 0, builds_listed = 0, builds_while_import = 0, builds_before_frame = 0;
    u32 frame_builds = 0;
    u32 fill_looks = 0, fill_empty = 0, fill_added = 0; ///< NoteFill: looks, of these without
                                                        ///< a permutation, permutations taken
    FillRefused fill_refused{};                         ///< NoteFill: left out, by reason
    std::vector<std::pair<u32, u32>> frames; ///< frame time in us, builds counted in that frame
    std::chrono::steady_clock::time_point last_frame{}, last_report{};
    size_t judged = 0;    ///< frames judged slow or not so far
    u32 slow = 0, in_slow = 0; ///< slow frames, builds counted in them
    u32 waits_given_up = 0;    ///< waits that hit TLG_SHADER_PREBUILD_WAIT_S
    std::set<std::pair<u64, u64>> logged; ///< shaders LogUnlisted wrote a line for
    std::vector<std::string> listed_keys; ///< pipeline key file names of the shader list, sorted
    std::set<std::string> logged_keys;    ///< keys LogUnlistedKey wrote a line for
};

Background& State() {
    static auto* state = new Background;
    return *state;
}

void ReportLocked(Background& state);

// The per-shader cache lines only with TLG_SHADER_BUILD_STATS=1 (test runs), not with the list
// alone (a player's log would get a line for every shader of the list).
bool StatsVerbose() {
    static const bool on = [] {
        const char* env = std::getenv("TLG_SHADER_BUILD_STATS");
        return env && *env == '1';
    }();
    return on;
}

bool StatsOn() {
    static const bool on = [] {
        const char* env = std::getenv("TLG_SHADER_BUILD_STATS");
        return ShaderList::Enabled() || (env && *env == '1');
    }();
    return on;
}

/// The game folder on the host; empty for a packed game (.zar), which the list cannot read yet.
std::filesystem::path HostGameDir() {
    char host[4096];
    if (runtime_file_translate("/app0/eboot.bin", host, sizeof(host)) != 0) {
        return {};
    }
    return std::filesystem::u8path(host).parent_path();
}

} // namespace

namespace {

/// Memory the compiler read that may be game code (make_list.py's rule for the list): more than
/// 64 bytes, the shader header signature, or an s_setpc_b64 (the end of a fetch shader) in an
/// aligned word.
bool LooksLikeCode(u64 address, std::span<const u8> bytes) {
    constexpr std::string_view Signature{"OrbShdr"};
    if (bytes.size() > 64 ||
        std::search(bytes.begin(), bytes.end(), Signature.begin(), Signature.end()) != bytes.end()) {
        return true;
    }
    // Code is 4-byte aligned in memory: every aligned word of the region.
    for (size_t at = (4 - address % 4) % 4; at + 4 <= bytes.size(); at += 4) {
        u32 word;
        std::memcpy(&word, bytes.data() + at, 4);
        if ((word & 0xFFFFFF00u) == 0xBE802000u) {
            return true;
        }
    }
    return false;
}

std::string Base64(std::span<const u8> bytes) {
    static constexpr char Digits[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t at = 0; at < bytes.size(); at += 3) {
        const u32 n = u32(bytes[at]) << 16 | (at + 1 < bytes.size() ? u32(bytes[at + 1]) << 8 : 0) |
                      (at + 2 < bytes.size() ? u32(bytes[at + 2]) : 0);
        out += Digits[n >> 18 & 63];
        out += Digits[n >> 12 & 63];
        out += at + 1 < bytes.size() ? Digits[n >> 6 & 63] : '=';
        out += at + 2 < bytes.size() ? Digits[n & 63] : '=';
    }
    return out;
}

/// tlg: a shader the game built while playing that the shader list lacks: its recipe goes into
/// the game log as one line, once per shader and session, so players' logs can complete the list
/// (port/tools/shader-list/log_recipes.py):
///   Shader recipe (not in list): <program hash>_<permutation> <base64 of the zlib-packed recipe>
/// The recipe is version 3 without its program (PortableVersion). No game code goes in: of the
/// program only its length and CRC-32, a fetch shader as its vertex input layout, and of the other
/// memory the compiler read only regions that do not look like code (LooksLikeCode, the rule of
/// make_list.py) and lie outside the program; a geometry shader's copy shader (code) is never
/// written: a shader with one, or that read other memory, gets a line without its recipe. Only while the
/// shader list is used (Build shaders ahead on, a list read); a program of the list is left out
/// while the background build runs (the build may still bring this very shader).
void LogUnlisted(const Shader::Info& info, const std::map<u64, std::vector<u8>>& read,
                 const Shader::RuntimeInfo& lookup_runtime, const Shader::Backend::Bindings& start,
                 size_t perm_idx) {
    if (!ShaderList::Enabled()) {
        return;
    }
    {
        auto& state = State();
        std::scoped_lock lock{state.mutex};
        if (state.listed.empty() ||
            (state.running && std::ranges::binary_search(state.listed, info.pgm_hash)) ||
            !state.logged.emplace(info.pgm_hash, perm_idx).second) {
            return;
        }
    }
    const auto name = RecipeName(info.pgm_hash, perm_idx);
    const auto program = read.find(info.pgm_base);
    const auto ud = info.UserData();
    u64 fetch_address = 0;
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    const bool copy_shader = lookup_runtime.hw_stage == Shader::HwStage::Geometry &&
                             !lookup_runtime.hw.gs.vs_copy.empty();
    bool code_like = program == read.end() || copy_shader;
    if (info.has_fetch_shader && !code_like) {
        if (info.fetch_shader_sgpr_base + 1 < ud.size()) {
            std::memcpy(&fetch_address, &ud[info.fetch_shader_sgpr_base], sizeof(fetch_address));
            fetch = Shader::Gcn::ParseFetchShader(info);
        }
        code_like = !fetch || fetch->attributes.size() > 255;
    }
    std::vector<std::pair<u64, std::span<const u8>>> data_regions;
    for (const auto& [address, bytes] : read) {
        if (code_like) {
            break;
        }
        if (address == info.pgm_base || (fetch && address == fetch_address)) {
            continue;
        }
        const u64 program_end = info.pgm_base + program->second.size();
        code_like = LooksLikeCode(address, bytes) ||
                    (address < program_end && address + bytes.size() > info.pgm_base);
        data_regions.emplace_back(address, bytes);
    }
    if (code_like) {
        std::printf("Shader recipe (not in list): %s left out: %s\n", name.c_str(),
                    copy_shader ? "a geometry shader's copy shader is game code"
                                : "the compiler read memory that may be game code");
        return;
    }
    Writer w;
    w.Pod(Magic);
    w.Pod(PortableVersion);
    w.Pod(u32{0});
    w.Pod(u32{0});
    w.Pod(PortableId());
    w.Pod(info.hw_stage);
    w.Pod(info.sw_stage);
    w.Pod(info.pgm_hash);
    w.Pod(u64(perm_idx));
    w.Pod(u64(info.pgm_base));
    w.Words(ud);
    w.Words(std::span<const u32>{}); // no copy shader (code_like above)
    w.Words(info.FlatUserData());
    auto runtime = lookup_runtime;
    runtime.hw.gs.vs_copy = {};
    const auto portable_runtime = PortableRuntime(runtime);
    const auto portable_start = PortableBindings(start);
    w.Bytes(portable_runtime.data(), portable_runtime.size());
    w.Bytes(portable_start.data(), portable_start.size());
    const auto& code = program->second;
    w.Pod(u32(code.size()));
    w.Pod(u32(mz_crc32(MZ_CRC32_INIT, code.data(), code.size())));
    w.Pod(u8(fetch ? 1 : 0));
    if (fetch) {
        w.Pod(fetch->size);
        w.Pod(fetch->vertex_offset_sgpr);
        w.Pod(fetch->instance_offset_sgpr);
        w.Pod(u8(fetch->attributes.size()));
        for (const auto& a : fetch->attributes) {
            for (const u8 field : {a.semantic, a.dest_vgpr, a.num_elements, a.sgpr_base,
                                   a.dword_offset, a.instance_data, a.inst_offset, a.data_format,
                                   a.num_format}) {
                w.Pod(field);
            }
        }
    }
    w.Pod(u64(data_regions.size()));
    for (const auto& [address, bytes] : data_regions) {
        w.Pod(address);
        w.Bytes(bytes.data(), bytes.size());
    }
    mz_ulong packed_size = mz_compressBound(mz_ulong(w.out.size()));
    std::vector<u8> packed(packed_size);
    if (mz_compress2(packed.data(), &packed_size, w.out.data(), mz_ulong(w.out.size()), 9) !=
        MZ_OK) {
        std::printf("Shader recipe (not in list): %s left out: it could not be packed\n",
                    name.c_str());
        return;
    }
    packed.resize(packed_size);
    std::printf("Shader recipe (not in list): %s %s\n", name.c_str(), Base64(packed).c_str());
}

} // namespace

void LogUnlistedKey(const std::string& name, std::span<const u8> key_file) {
    if (!ShaderList::Enabled()) {
        return;
    }
    {
        auto& state = State();
        std::scoped_lock lock{state.mutex};
        if (state.listed.empty() || std::ranges::binary_search(state.listed_keys, name) ||
            !state.logged_keys.insert(name).second) {
            return;
        }
    }
    std::string why;
    const auto portable = PortableKey(key_file, why);
    if (!portable) {
        std::printf("Pipeline key (not in list): %s left out: it %s\n", name.c_str(), why.c_str());
        return;
    }
    LogPortableKey(name, *portable);
}

void LogPortableKey(const std::string& name, std::span<const u8> portable) {
    mz_ulong packed_size = mz_compressBound(mz_ulong(portable.size()));
    std::vector<u8> packed(packed_size);
    if (mz_compress2(packed.data(), &packed_size, portable.data(), mz_ulong(portable.size()),
                     9) != MZ_OK) {
        std::printf("Pipeline key (not in list): %s left out: it could not be packed\n",
                    name.c_str());
        return;
    }
    packed.resize(packed_size);
    std::printf("Pipeline key (not in list): %s %s\n", name.c_str(), Base64(packed).c_str());
}

u64 LayoutId() {
    static const u64 id = ShaderFields::ComputeLayoutId();
    return id;
}

u64 PortableId() {
    static const u64 id = ShaderFields::ComputePortableId();
    return id;
}

std::vector<u8> PortableRuntime(const Shader::RuntimeInfo& runtime) {
    ShaderFields::FieldWriter w;
    auto copy = runtime;
    ShaderFields::RuntimeFields(w, copy);
    return std::move(w.out);
}

bool RuntimeFromPortable(std::span<const u8> portable, Shader::RuntimeInfo& runtime) {
    runtime = ShaderFields::ZeroRuntime();
    ShaderFields::FieldReader r{portable};
    ShaderFields::RuntimeFields(r, runtime);
    return r.Done();
}

std::vector<u8> PortableBindings(const Shader::Backend::Bindings& bindings) {
    ShaderFields::FieldWriter w;
    auto copy = bindings;
    ShaderFields::BindingsFields(w, copy);
    return std::move(w.out);
}

bool BindingsFromPortable(std::span<const u8> portable, Shader::Backend::Bindings& bindings) {
    bindings = {};
    ShaderFields::FieldReader r{portable};
    ShaderFields::BindingsFields(r, bindings);
    return r.Done();
}

/// TLG_SHADER_LAYOUT_CHECK=0: recipes and lists of another layout id are read anyway, on the
/// struct sizes alone (the check as it was before the layout id).
bool LayoutCheck() {
    static const bool on = [] {
        const char* env = std::getenv("TLG_SHADER_LAYOUT_CHECK");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return on;
}

/// The import children of the background build: ARGS = CACHE RECIPES [LIST GAME_DIR], in `jobs`
/// parallel shares (each restarted after a shader it stops on, RunChild). 0 when all ended well.
int RunChildren(const std::vector<std::filesystem::path>& args, u32 jobs) {
    if (jobs <= 1) {
        return RunChild(args, args[0], 0);
    }
    std::vector<int> codes(jobs, -1);
    std::vector<std::thread> children;
    for (u32 part = 0; part < jobs; ++part) {
        children.emplace_back([&args, &codes, part, jobs] {
            auto child_args = args;
            child_args.emplace_back(std::to_string(part));
            child_args.emplace_back(std::to_string(jobs));
            codes[part] = RunChild(child_args, args[0], part);
        });
    }
    for (auto& child : children) {
        child.join();
    }
    int code = 0;
    for (const int c : codes) {
        code = code != 0 ? code : c;
    }
    return code;
}

bool ImportMissing(const std::filesystem::path& cache_dir) {
    const auto recipe_dir = Storage::RecipeDir();
    if (!ShaderList::Enabled()) {
        if (recipe_dir.empty() || MissingRecipes(cache_dir, recipe_dir).empty()) {
            return false;
        }
        // The child reads profile.bin, which the IO thread may still be writing.
        Storage::DataBase::Instance().Flush(std::chrono::seconds{5});
        const auto started = std::chrono::steady_clock::now();
        const int code = RunChild({cache_dir, recipe_dir}, cache_dir, 0);
        TlgOverlay::SetShaderBuild(0, 0);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - started)
                            .count();
        LOG_INFO(Render_Vulkan, "tlg: shader recipes imported in {} ms (exit {})", ms, code);
        std::printf("Shader recipes: import took %lld ms (exit %d)\n", static_cast<long long>(ms),
                    code);
        return code == 0;
    }

    // shader_prebuild on: the shader list counts too, and the child runs in the background.
    const auto list = ShaderList::Path();
    const auto game_dir = list.empty() ? std::filesystem::path{} : HostGameDir();
    std::vector<u64> listed;
    std::vector<std::string> listed_keys;
    if (list.empty()) {
        std::printf("Shader prebuild: no shader list for this game\n");
    } else if (game_dir.empty()) {
        std::printf("Shader prebuild: the game folder is packed (.zar); the shader list is not "
                    "read for it yet\n");
    } else {
        std::string error;
        listed = ShaderList::Programs(list, error, &listed_keys);
        if (listed.empty()) {
            std::printf("Shader prebuild: shader list %s not used: %s\n", list.string().c_str(),
                        error.empty() ? "it names no programs" : error.c_str());
        }
    }
    const auto cached = CachedPrograms(cache_dir);
    const auto missing_listed = std::ranges::count_if(
        listed, [&](u64 pgm_hash) { return !cached.contains(pgm_hash); });
    const auto missing_recorded =
        recipe_dir.empty() ? 0 : MissingRecipes(cache_dir, recipe_dir).size();
    auto& state = State();
    {
        std::scoped_lock lock{state.mutex};
        state.listed = listed;
        state.listed_keys = std::move(listed_keys);
    }
    if (missing_listed == 0 && missing_recorded == 0) {
        return false;
    }
    Storage::DataBase::Instance().Flush(std::chrono::seconds{5});
    std::vector<std::filesystem::path> args{cache_dir, recipe_dir};
    if (missing_listed != 0) {
        args.push_back(list);
        args.push_back(game_dir);
    }
    {
        std::scoped_lock lock{state.mutex};
        state.running = true;
        state.started = std::chrono::steady_clock::now();
    }
    std::printf("Shader prebuild: building in the background: %lld programs of the shader list "
                "and %zu recorded recipes the cache lacks\n",
                static_cast<long long>(missing_listed), missing_recorded);
    // Children in parallel: half the CPU threads (the game runs meanwhile), at most 8;
    // TLG_SHADER_PREBUILD_JOBS sets the count.
    u32 jobs = std::clamp(std::max(1u, std::thread::hardware_concurrency()) / 2, 1u, 8u);
    if (const char* env = std::getenv("TLG_SHADER_PREBUILD_JOBS"); env && std::atoi(env) > 0) {
        jobs = u32(std::min(std::atoi(env), 32));
    }
    if (args.size() == 2) {
        jobs = 1; // recorded recipes only: one child, as the plain import
    }
    std::printf("Shader prebuild: %u build processes\n", jobs);
    std::thread{[args = std::move(args), jobs] {
        // Test only: TLG_SHADER_PREBUILD_DELAY_MS holds the build back, to check the waits.
        if (const char* env = std::getenv("TLG_SHADER_PREBUILD_DELAY_MS"); env && *env) {
            std::this_thread::sleep_for(std::chrono::milliseconds{std::atoi(env)});
        }
        const int code = RunChildren(args, jobs);
        TlgOverlay::SetShaderBuild(0, 0);
        auto& state = State();
        std::scoped_lock lock{state.mutex};
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - state.started)
                            .count();
        std::printf("Shader recipes: import took %lld ms (exit %d, in the background)\n",
                    static_cast<long long>(ms), code);
        state.running = false;
        state.done.notify_all();
    }}.detach();
    return true;
}

void WaitImport(const char* why, bool (*stop)(void*), void* stop_arg) {
    auto& state = State();
    std::unique_lock lock{state.mutex};
    if (!state.running) {
        return;
    }
    const auto started = std::chrono::steady_clock::now();
    // A build that hangs must not hold the game: TLG_SHADER_PREBUILD_WAIT_S (default 120).
    long limit_s = 120;
    if (const char* env = std::getenv("TLG_SHADER_PREBUILD_WAIT_S"); env && *env) {
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end != env && *end == '\0' && value >= 0 && value <= 3600) {
            limit_s = value;
        }
    }
    std::printf("Shader prebuild: %s waits for the background build (at most %ld s)\n", why,
                limit_s);
    while (state.running && !(stop && stop(stop_arg)) &&
           std::chrono::steady_clock::now() - started < std::chrono::seconds{limit_s}) {
        state.done.wait_for(lock, std::chrono::milliseconds{100});
    }
    if (state.running) {
        ++state.waits_given_up;
        std::printf("Shader prebuild: %s stops waiting, the build is still running\n", why);
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - started)
                        .count();
    std::printf("Shader prebuild: %s waited %lld ms\n", why, static_cast<long long>(ms));
}

void NoteBuild(u64 pgm_hash, size_t perm_idx, u32 hw_stage) {
    if (!StatsOn()) {
        return;
    }
    auto& state = State();
    std::scoped_lock lock{state.mutex};
    const bool listed = std::ranges::binary_search(state.listed, pgm_hash);
    ++state.builds;
    ++state.frame_builds;
    state.builds_listed += listed;
    state.builds_while_import += state.running;
    state.builds_before_frame += state.last_frame == std::chrono::steady_clock::time_point{};
    std::printf("Shader built while playing: %#018llx_%zu (hw stage %u)%s\n",
                static_cast<unsigned long long>(pgm_hash), perm_idx, hw_stage,
                !listed ? "" : state.running ? ", in the list, background build still running"
                                             : ", in the list");
}

bool StoreLookup() {
    static const bool on = [] {
        const char* env = std::getenv("TLG_SHADER_STORE_LOOKUP");
        const bool value = env && *env ? *env == '1' : ShaderList::Enabled();
        std::printf("Shader cache lookup for programs without a pipeline key: %s "
                    "(TLG_SHADER_STORE_LOOKUP)\n",
                    value ? "on" : "off");
        return value;
    }();
    return on;
}

bool StoreFill() {
    static const bool on = [] {
        const char* env = std::getenv("TLG_SHADER_STORE_FILL");
        const bool value = env && *env ? *env == '1' : StoreLookup();
        std::printf("Shader cache lookup for permutations a known program lacks: %s "
                    "(TLG_SHADER_STORE_FILL)\n",
                    value ? "on" : "off");
        return value;
    }();
    return on;
}

void NoteFill(u64 pgm_hash, size_t perms, const FillRefused& refused, u32 hw_stage) {
    {
        auto& state = State();
        std::scoped_lock lock{state.mutex};
        ++state.fill_looks;
        state.fill_empty += perms == 0;
        state.fill_added += u32(perms);
        state.fill_refused.meta += refused.meta;
        state.fill_refused.spirv += refused.spirv;
        state.fill_refused.module += refused.module;
    }
    if (StatsVerbose()) {
        std::printf("Shader permutations added from the cache to a known program: %#018llx, %zu "
                    "permutations; left out: %u meta refused, %u without SPIR-V, %u module "
                    "refused (hw stage %u)\n",
                    static_cast<unsigned long long>(pgm_hash), perms, refused.meta, refused.spirv,
                    refused.module, hw_stage);
    }
}

void NoteLookup(u64 pgm_hash, size_t perms, u32 hw_stage) {
    if (StatsVerbose()) {
        std::printf("Shader found in the cache without a pipeline key: %#018llx, %zu "
                    "permutations (hw stage %u)\n",
                    static_cast<unsigned long long>(pgm_hash), perms, hw_stage);
    }
}

void NoteCacheUse(u64 pgm_hash, size_t perm_idx, u32 hw_stage, bool lookup) {
    if (StatsVerbose()) {
        std::printf("Shader used from the cache: %#018llx_%zu (hw stage %u), %s\n",
                    static_cast<unsigned long long>(pgm_hash), perm_idx, hw_stage,
                    lookup ? "found by the lookup" : "loaded with a pipeline key");
    }
}

void NoteFrame() {
    if (!StatsOn()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    auto& state = State();
    std::scoped_lock lock{state.mutex};
    if (state.last_frame != std::chrono::steady_clock::time_point{}) {
        const u32 us = u32(
            std::chrono::duration_cast<std::chrono::microseconds>(now - state.last_frame).count());
        // TLG_SHADER_BUILD_STATS=1: every frame's time and the shaders built in it, so a test
        // run counts hitches from a mark in the log.
        if (StatsVerbose()) {
            std::printf("Frame time: %.1f ms, %u shader builds\n", us / 1000.0, state.frame_builds);
        }
        state.frames.emplace_back(us, std::exchange(state.frame_builds, 0));
    } else {
        state.frame_builds = 0; // before the first frame: counted apart
    }
    state.last_frame = now;
    // A test run is often ended by a kill: the counts so far, every 5 s.
    if (now - state.last_report >= std::chrono::seconds{5}) {
        state.last_report = now;
        ReportLocked(state);
    }
}

namespace {
void ReportLocked(Background& state) {
    // A slow frame takes more than twice the median of the 61 frames around it, and at least
    // 1/60 s more (the frame time changes a lot between scenes and cutscenes).
    constexpr size_t Window = 30;
    // Each frame is judged once, when the frames after it are there (the end of a session is
    // judged with what it has).
    const auto& frames = state.frames;
    std::vector<u32> around;
    for (size_t& i = state.judged; i < frames.size() && i + Window < frames.size(); ++i) {
        around.clear();
        for (size_t j = i > Window ? i - Window : 0; j < std::min(frames.size(), i + Window + 1);
             ++j) {
            around.push_back(frames[j].first);
        }
        std::ranges::nth_element(around, around.begin() + around.size() / 2);
        const u32 median = around[around.size() / 2];
        if (frames[i].first > 2 * median && frames[i].first > median + 16'667) {
            ++state.slow;
            state.in_slow += frames[i].second;
        }
    }
    u32 pending = 0; // builds in frames not judged yet
    for (size_t i = state.judged; i < frames.size(); ++i) {
        pending += frames[i].second;
    }
    std::printf("Shader builds while playing: %u (%u inside slow frames; %u before the first "
                "frame, %u while the background build ran, %u of programs in the shader list); "
                "%zu frames, %u slow (over twice the median of the 61 frames around them), %u "
                "builds in the last frames not judged yet; %u waits for the build gave up\n",
                state.builds, state.in_slow, state.builds_before_frame, state.builds_while_import,
                state.builds_listed, frames.size(), state.slow, pending, state.waits_given_up);
    if (state.fill_looks) {
        std::printf("Shader cache looks before a build (TLG_SHADER_STORE_FILL): %u (%u found "
                    "nothing), %u permutations taken; left out: %u meta refused, %u without "
                    "SPIR-V, %u module refused\n",
                    state.fill_looks, state.fill_empty, state.fill_added, state.fill_refused.meta,
                    state.fill_refused.spirv, state.fill_refused.module);
    }
}
} // namespace

void ReportBuilds() {
    if (!StatsOn()) {
        return;
    }
    auto& state = State();
    std::scoped_lock lock{state.mutex};
    ReportLocked(state);
}

int ImportChild(const std::filesystem::path& cache_dir, const std::filesystem::path& recipe_dir,
                const std::filesystem::path& list_path, const std::filesystem::path& game_dir,
                u32 part, u32 parts) {
    // tlg: profile.bin starts with the cache's version header since v0.10.0; reading it as a bare
    // profile made every import stop here (exit 2) without a word in the game's log.
    const auto cached_profile = DecodeCacheHeader(ReadFile(cache_dir / "profile.bin"));
    if (!cached_profile) {
        std::fprintf(stderr, "import-shaders: no profile.bin of this build in %s\n",
                     cache_dir.string().c_str());
        return 2;
    }
    const Shader::Profile profile = *cached_profile;
#ifdef _WIN32
    // A shader the compiler cannot take may crash the child: no error dialog, the parent
    // restarts it (RunChild).
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#else
    // Started by the game (RunChildOnce): it ends with the game, also when the game crashes or is
    // killed (Windows: the game's job object).
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    if (const char* parent = std::getenv("TLG_SHADER_PREBUILD_PARENT");
        parent && std::strtoll(parent, nullptr, 10) != getppid()) {
        std::fprintf(stderr, "import-shaders: the game ended before the build began\n");
        return 1;
    }
    // In a background process group: writing to a terminal set to stop such writers (stty tostop)
    // would stop it, and the parent would take that for a shader it cannot build.
    signal(SIGTTOU, SIG_IGN);
#endif

    const auto started = std::chrono::steady_clock::now();
    // The work: recorded recipes of programs the cache lacks, then the shader list's recipes of
    // programs neither has. A program takes all its permutations from one of the two (the
    // permutation index is only valid within one recording).
    struct Work {
        u64 pgm_hash;
        size_t perm_idx;
        std::filesystem::path file;
        const ShaderList::Entry* entry;
        bool operator<(const Work& other) const {
            return std::tie(pgm_hash, perm_idx) < std::tie(other.pgm_hash, other.perm_idx);
        }
    };
    std::vector<Work> work;
    std::set<u64> recorded;
    std::map<std::string, u32> unread; // recorded recipes this build cannot read, by reason
    for (const auto& file : MissingRecipes(cache_dir, recipe_dir)) {
        // A recipe this build cannot read (another layout, version or size) is left out here, so
        // the shader list can still build its program.
        std::array<u8, 32> head{};
        std::ifstream in{file.path, std::ios::binary};
        in.read(reinterpret_cast<char*>(head.data()), head.size());
        Reader r{std::span<const u8>{head.data(), size_t(in.gcount())}};
        if (const char* why = ""; !ReadHeader(r, why)) {
            ++unread[why];
            continue;
        }
        work.push_back({file.pgm_hash, file.perm_idx, file.path, nullptr});
        recorded.insert(file.pgm_hash);
    }
    std::optional<ShaderList::List> list;
    if (!list_path.empty()) {
        std::string error;
        list = ShaderList::Read(list_path, error);
        if (!list) {
            std::printf("import-shaders: shader list %s not used: %s\n",
                        list_path.string().c_str(), error.c_str());
        }
    }
    if (list) {
        const auto cached = CachedPrograms(cache_dir);
        for (const auto& entry : list->entries) {
            if (!cached.contains(entry.pgm_hash) && !recorded.contains(entry.pgm_hash)) {
                work.push_back({entry.pgm_hash, size_t(entry.perm_idx), {}, &entry});
            }
        }
    }
    std::sort(work.begin(), work.end());
    // Programs a build process stopped on before (RunChild): left out, never tried again.
    std::set<u64> left_out;
    {
        std::istringstream lines{ReadText(LeftOutFile(cache_dir))};
        for (std::string line; std::getline(lines, line);) {
            if (const u64 hash = std::strtoull(line.c_str(), nullptr, 16); hash != 0) {
                left_out.insert(hash);
            }
        }
    }
    const auto left_out_now =
        std::erase_if(work, [&](const Work& item) { return left_out.contains(item.pgm_hash); });
    // Several children share the work by program (each maps guest memory at fixed addresses, so
    // they are processes, not threads); the share is a function of the program hash alone, so a
    // child that starts later and sees what the others wrote meanwhile skips nothing. The first
    // child copies the pipeline keys.
    if (parts > 1) {
        std::erase_if(work, [&](const Work& item) {
            return ((item.pgm_hash * 0x9E3779B97F4A7C15ull) >> 32) % parts != part;
        });
    }
    std::optional<ShaderList::GameFiles> game_files;
    if (list) {
        game_files.emplace(game_dir, *list);
    }

    // Pipelines need their keys; the recipe folder and the shader list keep a copy of every key.
    // First, so a build cut short still leaves them (the next start builds the missing shaders).
    u32 keys = 0;
    std::error_code error;
    auto key_dir = part == 0 ? recipe_dir : std::filesystem::path{};
    if (const auto why = key_dir.empty() ? std::string{} : FolderKeysRefused(key_dir);
        !why.empty()) {
        std::printf("import-shaders: the recipe folder's pipeline keys are not copied: %s\n",
                    why.c_str());
        key_dir.clear();
    }
    for (const auto& entry : std::filesystem::directory_iterator{key_dir, error}) {
        if (entry.path().extension() == ".key" &&
            !std::filesystem::exists(cache_dir / entry.path().filename())) {
            std::filesystem::copy_file(entry.path(), cache_dir / entry.path().filename(), error);
            keys += !error;
        }
    }
    if (list && part == 0) {
        if (list->keys_refused) {
            std::printf(
                "import-shaders: %u pipeline keys of the shader list not used: the key %s\n",
                list->keys_refused, list->keys_refused_why.c_str());
        }
        for (const auto& [key_name, bytes] : list->keys) {
            const auto path = cache_dir / std::filesystem::u8path(key_name);
            if (!std::filesystem::exists(path, error)) {
                keys += WriteFile(path, bytes.data(), bytes.size());
            }
        }
    }
    Shader::Pools pools;
    u32 built = 0, skipped = 0, from_list = 0, not_found = 0, done = 0;
    std::set<u64> failed; // a program whose first permutation failed keeps no later one
    std::map<u64, const char*> unread_program; // the reason, by program (its later permutations)
    // Which shader this child builds, for the parent when the child stops early (RunChild).
    std::FILE* progress = nullptr;
#ifdef _WIN32
    _wfopen_s(&progress, ProgressFile(cache_dir, part).c_str(), L"wb");
#else
    progress = std::fopen(ProgressFile(cache_dir, part).c_str(), "wb");
#endif
    for (const auto& item : work) {
        if (progress) {
            std::fseek(progress, 0, SEEK_SET);
            // Fixed width (rewritten in place): shader, then shaders done and in this share.
            std::fprintf(progress, "%#018llx %20llu %10u %10u\n",
                         static_cast<unsigned long long>(item.pgm_hash),
                         static_cast<unsigned long long>(item.perm_idx), done,
                         static_cast<unsigned>(work.size()));
            std::fflush(progress);
        }
        if (++done % 100 == 0) {
            std::printf("import-shaders: %u of %zu\n", done, work.size());
            std::fflush(stdout);
        }
        if (failed.contains(item.pgm_hash)) {
            ++skipped;
            if (const auto why = unread_program.find(item.pgm_hash);
                why != unread_program.end() && !item.entry) {
                ++unread[why->second];
            }
            continue;
        }
        std::optional<Recipe> recipe;
        if (item.entry) {
            const char* why = "";
            recipe = FromList(*item.entry, *game_files, why);
            if (!recipe) {
                std::printf("import-shaders: shader list program %#018llx (hw stage %u) %s\n",
                            static_cast<unsigned long long>(item.pgm_hash), item.entry->hw_stage,
                            why);
                ++not_found;
            }
        } else {
            const auto data = ReadFile(item.file);
            const char* why = "";
            recipe = Parse(data, why);
            if (!recipe) {
                ++unread[why];
                unread_program.emplace(item.pgm_hash, why);
            }
        }
        if (!recipe || recipe->pgm_hash != item.pgm_hash) {
            ++skipped;
            failed.insert(item.pgm_hash);
            continue;
        }
        GuestMapping mapping{recipe->regions};
        const auto program = std::ranges::find(recipe->regions, recipe->pgm_base,
                                               &std::pair<u64, std::vector<u8>>::first);
        if (!mapping.ok || program == recipe->regions.end()) {
            ++skipped;
            failed.insert(item.pgm_hash);
            continue;
        }
        if (!recipe->vs_copy.empty()) {
            recipe->runtime.hw.gs.vs_copy = recipe->vs_copy;
        }
        const Shader::ShaderParams params{
            .user_data = std::span<const u32, Shader::ShaderParams::NumShaderUserData>{
                recipe->user_data.data(), Shader::ShaderParams::NumShaderUserData},
            .code = {reinterpret_cast<const u32*>(recipe->pgm_base),
                     program->second.size() / sizeof(u32)},
            .hash = recipe->pgm_hash,
        };
        Shader::replay_flat = &recipe->flat;
        Shader::Info info{recipe->hw_stage, recipe->sw_stage, params};
        // A list recipe has the vertex input layout instead of the fetch shader: the compiler
        // reads it from a snapshot, as its own workers do (FetchShaderSnapshot).
        Shader::Gcn::FetchShaderSnapshot fetch_snapshot;
        std::optional<Shader::Gcn::ScopedFetchShaderSnapshot> fetch_scope;
        if (recipe->fetch) {
            fetch_snapshot.info = &info;
            fetch_snapshot.fetch = recipe->fetch;
            for (const auto& attribute : recipe->fetch->attributes) {
                fetch_snapshot.buffers.push_back(attribute.GetSharp(info));
            }
            fetch_scope.emplace(fetch_snapshot);
        }
        auto runtime = recipe->runtime;
        auto binding = recipe->start;
        std::vector<u32> spv;
        std::optional<Shader::StageSpecialization> spec;
        const auto name = RecipeName(item.pgm_hash, item.perm_idx);
        try {
            const auto ir = Shader::TranslateProgram(params.code, pools, info, runtime, profile);
            spec.emplace(info, recipe->runtime, profile, recipe->start);
            spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, ir, binding);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "import-shaders: %s: %s\n", name.c_str(), e.what());
        }
        Shader::replay_flat = nullptr;
        pools.ReleaseContents();
        if (spv.empty() || info.translation_failed) {
            ++skipped;
            failed.insert(item.pgm_hash);
            continue;
        }
        spec->info = &info;
        const auto meta = EncodeShaderMeta(info, *spec, HashCombine(item.pgm_hash, item.perm_idx),
                                           item.perm_idx);
        WriteFile(cache_dir / fmt::format("{:#018x}.meta",
                                          HashCombine(item.pgm_hash, item.perm_idx)),
                  meta.data(), meta.size());
        WriteFile(cache_dir / (name + ".spv"), spv.data(), spv.size() * sizeof(u32));
        ++built;
        from_list += item.entry != nullptr;
    }

    if (progress) {
        std::fclose(progress);
        std::filesystem::remove(ProgressFile(cache_dir, part), error);
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - started)
                        .count();
    if (left_out_now) {
        std::printf("import-shaders: %zu shaders left out: a build process stopped on their "
                    "program before (%s)\n",
                    size_t(left_out_now), LeftOutFile(cache_dir).string().c_str());
    }
    for (const auto& [why, count] : unread) {
        std::printf("import-shaders: %u recorded recipes skipped: the recipe %s\n", count,
                    why.c_str());
    }
    std::printf("import-shaders: %u shaders built (%u from the shader list), %u skipped, %u list "
                "programs not in the game files, %u pipeline keys, layout id %#018llx%s, %lld ms\n",
                built, from_list, skipped, not_found, keys,
                static_cast<unsigned long long>(LayoutId()),
                LayoutCheck() ? "" : " (check off: TLG_SHADER_LAYOUT_CHECK=0)",
                static_cast<long long>(ms));
    return 0;
}

namespace {
/// The parts of the portable recipe file (PortableRecipes).
struct PortableOut {
    std::vector<u8> bytes;
    template <typename T>
    void Pod(T value) {
        const auto* p = reinterpret_cast<const u8*>(&value);
        bytes.insert(bytes.end(), p, p + sizeof(value));
    }
    void Blob(std::span<const u8> data) {
        Pod(u32(data.size()));
        bytes.insert(bytes.end(), data.begin(), data.end());
    }
    void Text(const std::string& text) {
        Blob({reinterpret_cast<const u8*>(text.data()), text.size()});
    }
};
} // namespace

int PortableRecipes(const std::filesystem::path& recipe_dir, const std::filesystem::path& out) {
    // The file: magic, version, the portable id and this build's layout id, then the recipes
    // (count; name, runtime info, bindings) and the pipeline keys (count; file name, portable
    // key); every part with a u32 length first.
    PortableOut head, recipes, keys;
    if (!LayoutCheck()) {
        // The portable form is made by the build that recorded the recipes; a recipe of another
        // layout read anyway would give wrong numbers.
        std::fprintf(stderr, "portable-recipes: not with TLG_SHADER_LAYOUT_CHECK=0\n");
        return 2;
    }

    std::vector<std::filesystem::path> ins, key_files, other_key_files;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator{recipe_dir, error}) {
        if (entry.path().extension() == ".in") {
            ins.push_back(entry.path());
        } else if (entry.path().extension() == ".key") {
            key_files.push_back(entry.path());
        } else if (entry.path().extension() == ".vkey") {
            // a key of another kind, already in its portable form (log_recipes.py): as it is
            other_key_files.push_back(entry.path());
        }
    }
    if (error) {
        std::fprintf(stderr, "portable-recipes: cannot read %s\n", recipe_dir.string().c_str());
        return 2;
    }
    std::ranges::sort(ins);
    std::ranges::sort(key_files);
    std::ranges::sort(other_key_files);

    u32 written = 0, refused = 0, v1 = 0, raw_same = 0;
    for (const auto& path : ins) {
        const auto data = ReadFile(path);
        const char* why = "";
        auto recipe = Parse(data, why);
        if (!recipe) {
            std::printf("portable-recipes: %s skipped: the recipe %s\n",
                        path.filename().string().c_str(), why);
            ++refused;
            continue;
        }
        v1 += data.size() >= 8 && data[4] == 1 && data[5] == 0 && data[6] == 0 && data[7] == 0;
        const auto runtime = PortableRuntime(recipe->runtime);
        const auto bindings = PortableBindings(recipe->start);
        // Read back: an equal struct, and the same portable bytes once more.
        auto runtime_back = ShaderFields::ZeroRuntime();
        Shader::Backend::Bindings bindings_back{};
        if (!RuntimeFromPortable(runtime, runtime_back) ||
            !BindingsFromPortable(bindings, bindings_back) || !(runtime_back == recipe->runtime) ||
            bindings_back != recipe->start || PortableRuntime(runtime_back) != runtime) {
            std::printf("portable-recipes: %s skipped: it does not come back the same from the "
                        "portable form\n",
                        path.filename().string().c_str());
            ++refused;
            continue;
        }
        // For the report: read back byte for byte as recorded (padding and unused union bytes
        // count here, so this may be lower than the number written).
        raw_same += std::memcmp(&runtime_back, &recipe->runtime, sizeof(runtime_back)) == 0;
        recipes.Text(path.stem().string());
        recipes.Blob(runtime);
        recipes.Blob(bindings);
        ++written;
    }

    u32 key_count = 0, keys_refused = 0, keys_same = 0;
    if (const auto why = FolderKeysRefused(recipe_dir); !why.empty()) {
        std::printf("portable-recipes: the folder's %zu pipeline keys are left out: %s\n",
                    key_files.size(), why.c_str());
        keys_refused = u32(key_files.size());
        key_files.clear();
    }
    for (const auto& path : key_files) {
        const auto data = ReadFile(path);
        std::string why;
        const auto portable = PortableKey(data, why);
        if (!portable) {
            std::printf("portable-recipes: key %s skipped: it %s\n",
                        path.filename().string().c_str(), why.c_str());
            ++keys_refused;
            continue;
        }
        // For the report: the key file this build makes from it has the recorded name.
        std::string name;
        KeyFromPortable(*portable, name, why);
        keys_same += name == path.filename().string();
        keys.Text(path.filename().string());
        keys.Blob(*portable);
        ++key_count;
    }
    u32 other_count = 0;
    for (const auto& path : other_key_files) {
        const auto data = ReadFile(path);
        if (ShaderList::PortableKind(data) != ShaderList::OtherKeyKind) {
            std::printf("portable-recipes: key %s skipped: not a key of kind %u\n",
                        path.filename().string().c_str(), ShaderList::OtherKeyKind);
            ++keys_refused;
            continue;
        }
        keys.Text(path.filename().string());
        keys.Blob(data);
        ++key_count;
        ++other_count;
    }

    head.bytes = {'T', 'L', 'G', 'P', 'O', 'R', 'T', 'B'};
    head.Pod(u32{1});
    head.Pod(PortableId());
    head.Pod(LayoutId());
    head.Pod(written);
    head.bytes.insert(head.bytes.end(), recipes.bytes.begin(), recipes.bytes.end());
    head.Pod(key_count);
    head.bytes.insert(head.bytes.end(), keys.bytes.begin(), keys.bytes.end());
    if (!WriteFile(out, head.bytes.data(), head.bytes.size())) {
        std::fprintf(stderr, "portable-recipes: cannot write %s\n", out.string().c_str());
        return 2;
    }
    std::printf("portable-recipes: %u recipes written (%u of them version 1, read on the struct "
                "sizes alone; %u read back byte for byte as recorded), %u skipped; %u pipeline "
                "keys written (%u under the recorded file name, %u of kind 2 as they are), %u skipped; "
                "portable id %#018llx, layout id %#018llx\n",
                written, v1, raw_same, refused, key_count, keys_same, other_count, keys_refused,
                static_cast<unsigned long long>(PortableId()),
                static_cast<unsigned long long>(LayoutId()));
    return refused || keys_refused ? 1 : 0;
}

} // namespace Vulkan::ShaderRecipe

extern "C" int tlg_portable_recipes(const char* recipe_dir, const char* out) {
    return Vulkan::ShaderRecipe::PortableRecipes(std::filesystem::u8path(recipe_dir),
                                                 std::filesystem::u8path(out));
}

// tlg-probe --keys-from-portable DIR: every portable key DIR/<name>.pkey (log_recipes.py, from
// the log lines of LogUnlistedKey) as this build's key file DIR/<name>, for merge_recipes.py. 0
// when every one was written.
extern "C" int tlg_keys_from_portable(const char* dir_name) {
    const auto dir = std::filesystem::u8path(dir_name);
    u32 written = 0, refused = 0;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator{dir, error}) {
        if (entry.path().extension() != ".pkey") {
            continue;
        }
        std::ifstream in{entry.path(), std::ios::binary};
        const std::vector<u8> portable{std::istreambuf_iterator<char>{in}, {}};
        std::string name, why;
        const auto bytes = Vulkan::KeyFromPortable(portable, name, why);
        const auto out = dir / entry.path().stem();
        if (!bytes || name.empty()) {
            std::printf("keys-from-portable: %s skipped: it %s\n",
                        entry.path().filename().string().c_str(), why.c_str());
            ++refused;
            continue;
        }
        std::ofstream file{out, std::ios::binary};
        file.write(reinterpret_cast<const char*>(bytes->data()), std::streamsize(bytes->size()));
        written += bool(file);
        refused += !file;
    }
    if (error) {
        std::fprintf(stderr, "keys-from-portable: cannot read %s\n", dir_name);
        return 2;
    }
    std::printf("keys-from-portable: %u keys written, %u skipped\n", written, refused);
    return refused ? 1 : 0;
}

// tlg-probe --prebuild CACHE RECIPES LIST GAME_DIR JOBS: the game's background build of the
// shader list, run in the foreground (offline check: restarts after a shader the compiler cannot
// take, build time). 0 when every child ended well.
extern "C" int tlg_prebuild(const char* cache_dir, const char* recipe_dir, const char* list,
                            const char* game_dir, unsigned jobs) {
    const auto started = std::chrono::steady_clock::now();
    const int code = Vulkan::ShaderRecipe::RunChildren(
        {std::filesystem::u8path(cache_dir), std::filesystem::u8path(recipe_dir),
         std::filesystem::u8path(list), std::filesystem::u8path(game_dir)},
        std::clamp(jobs, 1u, 32u));
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - started)
                        .count();
    std::printf("prebuild: %lld ms, exit %d\n", static_cast<long long>(ms), code);
    return code;
}

extern "C" int tlg_import_shaders(const char* cache_dir, const char* recipe_dir) {
    return Vulkan::ShaderRecipe::ImportChild(std::filesystem::u8path(cache_dir),
                                             std::filesystem::u8path(recipe_dir));
}

extern "C" int tlg_import_shaders_list(const char* cache_dir, const char* recipe_dir,
                                       const char* list, const char* game_dir, unsigned part,
                                       unsigned parts) {
    return Vulkan::ShaderRecipe::ImportChild(
        std::filesystem::u8path(cache_dir), std::filesystem::u8path(recipe_dir),
        std::filesystem::u8path(list), std::filesystem::u8path(game_dir), part,
        parts ? parts : 1);
}

namespace {
/// One .meta file of a shader cache, read as the game reads it.
bool ReadMetaFile(const std::filesystem::path& p, Shader::StageSpecialization& spec,
                  Shader::Info& info, u64& perm_hash, size_t& perm_idx) {
    std::ifstream file{p, std::ios::binary};
    std::vector<u8> data{std::istreambuf_iterator<char>{file}, {}};
    if (data.empty()) {
        return false;
    }
    Serialization::Archive ar{std::move(data)};
    Serialization::Reader meta{ar};
    u32 meta_version{}, binary_version{};
    meta.Read(meta_version);
    meta.Read(binary_version);
    meta.Read(perm_hash);
    meta.Read(perm_idx);
    spec.Deserialize(ar);
    return info.Deserialize(ar);
}
} // namespace

// tlg: debug check (tlg-probe --compare-meta DIR_A DIR_B): loads every .meta of DIR_A and the
// same file in DIR_B as the game does and compares the fields the game uses, not raw bytes
// (unused list slots and padding hold leftover memory).
extern "C" int tlg_compare_meta(const char* dir_a, const char* dir_b) {
    namespace fs = std::filesystem;
    const auto read = ReadMetaFile;
    // Immediates count only for dwords not loaded from the flat buffer.
    const auto same_fetch = [](const auto& a, const auto& b) {
        if (a.offsets != b.offsets || a.load_mask != b.load_mask) {
            return false;
        }
        for (size_t i = 0; i < a.immediates.size(); ++i) {
            if (!(a.load_mask >> i & 1) && a.immediates[i] != b.immediates[i]) {
                return false;
            }
        }
        return true;
    };
    const auto same = [&](const auto& a, const auto& b) {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, Shader::BufferResource>) {
            return same_fetch(a.sharp_fetch, b.sharp_fetch) && a.used_types == b.used_types &&
                   a.buffer_type == b.buffer_type && a.is_written == b.is_written &&
                   a.is_formatted == b.is_formatted && a.post_op == b.post_op &&
                   a.post_op_dw1_mask == b.post_op_dw1_mask;
        } else if constexpr (std::is_same_v<T, Shader::ImageResource>) {
            return same_fetch(a.sharp_fetch, b.sharp_fetch) && a.is_depth == b.is_depth &&
                   a.is_atomic == b.is_atomic && a.is_array == b.is_array &&
                   a.is_written == b.is_written && a.is_r128 == b.is_r128 &&
                   a.needs_native == b.needs_native &&
                   a.constant_mip_index == b.constant_mip_index &&
                   a.mip_fallback_mode == b.mip_fallback_mode && a.post_op == b.post_op;
        } else if constexpr (std::is_same_v<T, Shader::SamplerResource>) {
            return same_fetch(a.sharp_fetch, b.sharp_fetch) && a.post_op == b.post_op &&
                   a.post_op_tsharp_dw3_off == b.post_op_tsharp_dw3_off &&
                   a.is_depth == b.is_depth;
        } else {
            return a.sharp_idx == b.sharp_idx;
        }
    };
    const auto same_list = [&](const auto& a, const auto& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (!same(a[i], b[i])) {
                return false;
            }
        }
        return true;
    };
    std::map<std::string, u32> diffs;
    u32 files = 0, equal = 0;
    for (const auto& entry : fs::directory_iterator{fs::u8path(dir_a)}) {
        if (entry.path().extension() != ".meta") {
            continue;
        }
        ++files;
        Shader::StageSpecialization sa, sb;
        Shader::Info ia, ib;
        u64 ha{}, hb{};
        size_t pa{}, pb{};
        if (!read(entry.path(), sa, ia, ha, pa) ||
            !read(fs::u8path(dir_b) / entry.path().filename(), sb, ib, hb, pb)) {
            ++diffs["unreadable"];
            continue;
        }
        std::vector<const char*> bad;
        const auto check = [&](bool ok, const char* what) {
            if (!ok) {
                bad.push_back(what);
            }
        };
        check(ha == hb && pa == pb, "perm hash/index");
        check(std::memcmp(&sa.start, &sb.start, sizeof(sa.start)) == 0, "spec.start");
        check(sa.runtime_info == sb.runtime_info, "spec.runtime_info");
        check(sa.bitset == sb.bitset, "spec.bitset");
        check(sa.fetch_shader_data == sb.fetch_shader_data, "spec.fetch_shader_data");
        check(sa.vs_attribs == sb.vs_attribs, "spec.vs_attribs");
        check(sa.buffers == sb.buffers, "spec.buffers");
        check(sa.images == sb.images, "spec.images");
        check(sa.fmasks == sb.fmasks, "spec.fmasks");
        check(sa.samplers == sb.samplers, "spec.samplers");
        check(same_list(ia.buffers, ib.buffers), "info.buffers");
        check(same_list(ia.images, ib.images), "info.images");
        check(same_list(ia.samplers, ib.samplers), "info.samplers");
        check(same_list(ia.fmasks, ib.fmasks), "info.fmasks");
        check(ia.ud_mask.mask == ib.ud_mask.mask, "info.ud_mask");
        check(ia.fetch_shader_sgpr_base == ib.fetch_shader_sgpr_base &&
                  ia.shared_memory_scratch_size == ib.shared_memory_scratch_size &&
                  ia.pgm_hash == ib.pgm_hash && ia.tess_consts_dword_offset == ib.tess_consts_dword_offset &&
                  ia.tess_consts_ptr_base == ib.tess_consts_ptr_base && ia.hw_stage == ib.hw_stage &&
                  ia.sw_stage == ib.sw_stage && ia.mrt_mask == ib.mrt_mask &&
                  ia.has_fetch_shader == ib.has_fetch_shader &&
                  ia.has_bitwise_xor == ib.has_bitwise_xor && ia.uses_dma == ib.uses_dma,
              "info.scalars");
        check(ia.flattened_ud_buf == ib.flattened_ud_buf, "info.flat");
        check(ia.srt_info.walker_func_size == ib.srt_info.walker_func_size &&
                  ia.srt_info.flattened_bufsize_dw == ib.srt_info.flattened_bufsize_dw,
              "info.srt");
        if (bad.empty()) {
            ++equal;
        }
        for (const char* what : bad) {
            if (diffs[what]++ == 0) {
                std::printf("first %s difference: %s\n", what, entry.path().filename().string().c_str());
            }
        }
    }
    std::printf("compare-meta: %u files, %u equal in every field\n", files, equal);
    for (const auto& [what, count] : diffs) {
        std::printf("  %s differs in %u\n", what.c_str(), count);
    }
    return equal == files ? 0 : 1;
}

// tlg: tool for merging the recipe folders of two sessions (tlg-probe --same-permutation
// CACHE_A CACHE_B; port/tools/shader-list/merge_recipes.py). A permutation number is only the
// order in which one session met a program's settings, so the same recipe name can be another
// shader in another session. This prints, for every shader of CACHE_B (the import's .meta files)
// whose program CACHE_A also has, the first permutation of CACHE_A with the same settings, by
// the compare the game itself uses to pick a permutation (StageSpecialization ==, as in
// GetProgram and MergeWarmStage): "same PROGRAM PERM_B PERM_A", or "new PROGRAM PERM_B" when
// none matches. Exit 0 when every file was read.
extern "C" int tlg_same_permutation(const char* dir_a, const char* dir_b) {
    namespace fs = std::filesystem;
    struct Stage {
        Shader::StageSpecialization spec;
        std::unique_ptr<Shader::Info> info; // the spec points to it
        size_t perm_idx{};
    };
    u32 unreadable = 0;
    const auto load = [&](const char* dir) {
        std::map<u64, std::vector<Stage>> programs; // by program, sorted by permutation
        std::error_code error;
        for (const auto& entry : fs::directory_iterator{fs::u8path(dir), error}) {
            if (entry.path().extension() != ".meta") {
                continue;
            }
            Stage stage;
            stage.info = std::make_unique<Shader::Info>();
            u64 perm_hash{};
            if (!ReadMetaFile(entry.path(), stage.spec, *stage.info, perm_hash, stage.perm_idx)) {
                std::fprintf(stderr, "same-permutation: %s unreadable\n",
                             entry.path().string().c_str());
                ++unreadable;
                continue;
            }
            stage.spec.info = stage.info.get();
            programs[stage.info->pgm_hash].push_back(std::move(stage));
        }
        if (error) {
            std::fprintf(stderr, "same-permutation: cannot read %s\n", dir);
            ++unreadable;
        }
        for (auto& [hash, stages] : programs) {
            std::ranges::sort(stages, {}, &Stage::perm_idx);
        }
        return programs;
    };
    const auto a = load(dir_a);
    const auto b = load(dir_b);
    u32 same = 0, fresh = 0;
    for (const auto& [hash, stages] : b) {
        const auto it = a.find(hash);
        if (it == a.end()) {
            continue;
        }
        for (const auto& stage : stages) {
            const auto match = std::ranges::find_if(
                it->second, [&](const Stage& known) { return known.spec == stage.spec; });
            if (match != it->second.end()) {
                std::printf("same %#018llx %zu %zu\n", static_cast<unsigned long long>(hash),
                            stage.perm_idx, match->perm_idx);
                ++same;
            } else {
                std::printf("new %#018llx %zu\n", static_cast<unsigned long long>(hash),
                            stage.perm_idx);
                ++fresh;
            }
        }
    }
    std::fprintf(stderr, "same-permutation: %u same, %u new, %u unreadable\n", same, fresh,
                 unreadable);
    return unreadable ? 1 : 0;
}
