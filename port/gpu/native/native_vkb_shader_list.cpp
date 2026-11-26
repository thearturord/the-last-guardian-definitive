// SPDX-License-Identifier: GPL-2.0-or-later
// TLG_VKB_SHADER_PREBUILD: the new engine's shader list, the normal renderer's list converted (see the header)

#include "native/native_vkb_shader_list.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string_view>
#include <type_traits>
#include <fmt/format.h>
#include <xxhash.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "common/memory_patcher.h"
#include "native/native_vkb_shader_fields.h"
// The normal renderer's list (its entry) and its field lists, read only. These headers come with pull request 149
// (the normal renderer's shader pre-build, the lower pull request); the new engine's own commits do not change
// port/gpu/shadps4, they only include these two headers.
#include "video_core/renderer_vulkan/shader_fields.h"
#include "video_core/renderer_vulkan/shader_list.h"

extern "C" int runtime_file_translate(const char* guest, char* out, size_t size);

namespace Core::Vkb::ShaderList {
namespace {

// The two renderers' translators are copies of one: their stage numbers are the same
static_assert(static_cast<u32>(VkbShader::HwStage::Fragment) == static_cast<u32>(Shader::HwStage::Fragment) &&
              static_cast<u32>(VkbShader::HwStage::Vertex) == static_cast<u32>(Shader::HwStage::Vertex) &&
              static_cast<u32>(VkbShader::HwStage::Geometry) == static_cast<u32>(Shader::HwStage::Geometry) &&
              static_cast<u32>(VkbShader::HwStage::Compute) == static_cast<u32>(Shader::HwStage::Compute));
static_assert(static_cast<u32>(VkbShader::SwStage::Fragment) == static_cast<u32>(Shader::SwStage::Fragment) &&
              static_cast<u32>(VkbShader::SwStage::Vertex) == static_cast<u32>(Shader::SwStage::Vertex) &&
              static_cast<u32>(VkbShader::SwStage::Geometry) == static_cast<u32>(Shader::SwStage::Geometry) &&
              static_cast<u32>(VkbShader::SwStage::Compute) == static_cast<u32>(Shader::SwStage::Compute) &&
              VkbShader::MaxStageTypes == Shader::MaxStageTypes);
// Convert copies the normal entry's runtime info and bindings as bytes (its reader made them this build's structs)
static_assert(std::is_trivially_copyable_v<Shader::RuntimeInfo> && std::is_trivially_copyable_v<Shader::Backend::Bindings>);

std::vector<u8> ReadAll(const std::filesystem::path& path) {
    std::ifstream f{path, std::ios::binary};
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::filesystem::path ExecutableDir() {
#ifdef _WIN32
    wchar_t exe[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    return n ? std::filesystem::path{std::wstring{exe, n}}.parent_path() : std::filesystem::path{};
#else
    std::error_code error;
    return std::filesystem::read_symlink("/proc/self/exe", error).parent_path();
#endif
}

/// A member's name in a field list, with its place among the members of that name ("in.param_index#3": the fourth input's)
struct Names {
    std::map<std::string, u32, std::less<>> seen;
    std::string Next(std::string_view name) {
        auto it = seen.try_emplace(std::string{name}, 0).first;
        return fmt::format("{}#{}", name, it->second++);
    }
};

/// The values of a normal entry's members, by name (the stage's variants only, as its list holds them)
struct Collect {
    static constexpr bool AllVariants = false;
    Names names;
    std::map<std::string, u64> values;
    template <typename Get, typename Set>
    void Field(std::string_view name, Get get, Set) {
        values.emplace(names.Next(name), get());
    }
};

/// The new engine's members, set from the normal entry's by name, or by the rules of the header
struct Assign {
    static constexpr bool AllVariants = false;
    const std::map<std::string, u64>& normal;
    Names names;
    std::set<std::string> used; ///< the normal members taken
    u32 fixed_dx10_clamp = 0, fixed_depth_before_shader = 0;
    std::string failed; ///< the first member without a value or whose value does not fit
    u64 Normal(const std::string& key) const {
        const auto it = normal.find(key);
        return it == normal.end() ? 0 : it->second;
    }
    template <typename Get, typename Set>
    void Field(std::string_view name, Get, Set set) {
        const std::string key = names.Next(name);
        u64 v = 0;
        if (name == "r.props.dx10_clamp") {
            v = FixedDx10Clamp;
            ++fixed_dx10_clamp;
        } else if (name == "fs.depth_before_shader") {
            v = FixedDepthBeforeShader;
            ++fixed_depth_before_shader;
        } else if (name == "r.props.reserved" || name == "b.indexed") {
            v = 0;
        } else if (name == "vs.is_indirect") {
            v = Normal("vs.vertex_sgpr_offset#0") != 0 || Normal("vs.instance_sgpr_offset#0") != 0;
        } else if (const auto it = normal.find(key); it != normal.end()) {
            v = it->second;
            used.insert(key);
        } else {
            if (failed.empty()) {
                failed = fmt::format("the new engine's setting {} is not in the normal entry", key);
            }
            return;
        }
        if (!set(v) && failed.empty()) {
            failed = fmt::format("the value {} of {} does not fit the new engine's setting", v, key);
        }
    }
};

/// The normal members the new engine does not have and drops (the header)
constexpr std::string_view Dropped[] = {"b.user_data#0"};

/// The first normal member that was not taken and is not dropped: empty when there is none
std::string Leftover(const std::map<std::string, u64>& normal, const std::set<std::string>& used) {
    for (const auto& [key, value] : normal) {
        if (!used.contains(key) && std::ranges::find(Dropped, std::string_view{key}) == std::end(Dropped)) {
            return fmt::format("the normal entry's setting {} has no place in the new engine", key);
        }
    }
    return {};
}

} // namespace

std::filesystem::path Path() {
    if (const char* env = std::getenv("TLG_VKB_SHADER_LIST"); env != nullptr && *env != '\0') {
        return std::strcmp(env, "0") == 0 ? std::filesystem::path{} : std::filesystem::u8path(env);
    }
    // The normal renderer's list (ShaderList::Path): one list for every region. The US game (CUSA03627, 1.03) has the same shader archives and
    // eboot.bin as the European one (CUSA03745, 1.03) it was made from; for the other discs each entry's length and CRC-32 check what is read
    // from the player's files.
    const auto dir = ExecutableDir();
    for (const auto& candidate : {dir / "shader-list" / "shaders.tlgsl", dir.parent_path() / "shader-list" / "shaders.tlgsl"}) {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error)) {
            return candidate;
        }
    }
    return {};
}

u64 FileId(const std::filesystem::path& path) {
    const auto bytes = ReadAll(path);
    return bytes.empty() ? 0 : XXH3_64bits_withSeed(bytes.data(), bytes.size(), bytes.size());
}

std::optional<Stage> Convert(const Vulkan::ShaderList::Entry& e, std::string& why) {
    if (e.hw_stage > static_cast<u32>(VkbShader::HwStage::Compute) || e.sw_stage >= VkbShader::MaxStageTypes) {
        why = "names a stage the new engine does not know";
        return std::nullopt;
    }
    if (e.hw_stage == static_cast<u32>(VkbShader::HwStage::Geometry) || !e.vs_copy.empty()) {
        why = "is a geometry stage (the new engine has none)";
        return std::nullopt;
    }
    if (e.runtime.size() != sizeof(Shader::RuntimeInfo) || e.start.size() != sizeof(Shader::Backend::Bindings) ||
        e.user_data.size() != VkbShader::ShaderParams::NumShaderUserData || e.length == 0 || e.length % 4 != 0) {
        why = "is damaged (sizes)";
        return std::nullopt;
    }
    // the normal entry's members by name (its reader made its runtime info and bindings this build's structs)
    Collect normal;
    {
        alignas(Shader::RuntimeInfo) u8 runtime_bytes[sizeof(Shader::RuntimeInfo)];
        std::memcpy(runtime_bytes, e.runtime.data(), sizeof(runtime_bytes));
        auto& runtime = *reinterpret_cast<Shader::RuntimeInfo*>(runtime_bytes);
        Vulkan::ShaderFields::RuntimeFields(normal, runtime);
        Shader::Backend::Bindings start{};
        std::memcpy(&start, e.start.data(), sizeof(start));
        Vulkan::ShaderFields::BindingsFields(normal, start);
    }
    Stage s;
    s.hw_stage = static_cast<VkbShader::HwStage>(e.hw_stage);
    s.sw_stage = static_cast<VkbShader::SwStage>(e.sw_stage);
    s.kind = s.hw_stage == VkbShader::HwStage::Compute ? 1 : 0;
    s.pgm_hash = e.pgm_hash;
    s.pgm_base = e.pgm_base;
    s.code_words = e.length / 4;
    std::memcpy(s.user_data.data(), e.user_data.data(), sizeof(s.user_data));
    s.flat = e.flat;
    s.regions = e.regions;
    s.runtime = ShaderFields::ZeroRuntime();
    std::memset(static_cast<void*>(&s.start), 0, sizeof(s.start));
    Assign assign{.normal = normal.values};
    ShaderFields::RuntimeFields(assign, s.runtime);
    ShaderFields::BindingsFields(assign, s.start);
    if (!assign.failed.empty()) {
        why = assign.failed;
        return std::nullopt;
    }
    if (auto left = Leftover(normal.values, assign.used); !left.empty()) {
        why = std::move(left);
        return std::nullopt;
    }
    if (s.runtime.hw_stage != s.hw_stage || s.runtime.sw_stage != s.sw_stage) {
        why = "names other stages in its runtime info";
        return std::nullopt;
    }
    s.fixed_dx10_clamp = assign.fixed_dx10_clamp != 0;
    s.fixed_depth_before_shader = assign.fixed_depth_before_shader != 0;
    if (s.kind == 1) { // the new engine's compute rule: the salt and the local size are the workgroup's size x (native_vkb_compute.cpp)
        s.salt = s.runtime.hw.cs.workgroup_size[0];
        s.local_x = s.runtime.hw.cs.workgroup_size[0];
    }
    if (e.fetch) {
        auto& fetch = s.fetch.emplace();
        fetch.size = e.fetch->size;
        fetch.vertex_offset_sgpr = e.fetch->vertex_offset_sgpr;
        fetch.instance_offset_sgpr = e.fetch->instance_offset_sgpr;
        for (const auto& a : e.fetch->attributes) { // without `semantic`, which the new engine's layout does not have
            fetch.attributes.push_back({.dest_vgpr = a.dest_vgpr,
                                        .num_elements = a.num_elements,
                                        .sgpr_base = a.sgpr_base,
                                        .dword_offset = a.dword_offset,
                                        .instance_data = a.instance_data,
                                        .inst_offset = a.inst_offset,
                                        .data_format = a.data_format,
                                        .num_format = a.num_format});
        }
    }
    return s;
}

std::filesystem::path HostGameDir() {
    char host[4096];
    if (runtime_file_translate("/app0/eboot.bin", host, sizeof(host)) != 0) {
        return {};
    }
    return std::filesystem::u8path(host).parent_path();
}

} // namespace Core::Vkb::ShaderList
