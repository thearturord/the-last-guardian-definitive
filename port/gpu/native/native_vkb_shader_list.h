// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
// TLG_VKB_SHADER_PREBUILD: the new engine's shader list is the normal renderer's (owner decision 2026-11-24: one shipped list for both
// renderers; way A of 2026-11-25). The list file is shader-list/shaders.tlgsl next to the executable, made by
// port/tools/shader-list/make_list.py and read by the normal renderer's own reader (video_core/renderer_vulkan/shader_list.h: its Read and
// GameFiles, which only read files; port/gpu/shadps4 is not changed: principle 8). Each entry names one shader stage as the normal
// renderer's translator was handed it, its program by where the player's game files hold it; Convert below turns it into the inputs of the
// new engine's translator.
//
// Per field of the new engine's inputs (native_vkb_shader_fields.h, by member name):
// - copied: the stages, the program's hash and base, the 16 user data words, the flattened user data, the recorded memory reads, the vertex
//   input layout (without the normal renderer's `semantic`, which the new engine does not have), every runtime info field both renderers
//   have, and the binding starts `unified` and `buffer`;
// - derived: kind (compute when the stage is compute), the code length (the entry's length / 4), salt and local size x (compute: the
//   workgroup's size x, the new engine's own rule in native_vkb_compute.cpp; graphics: 0), vs.is_indirect (a vertex or instance offset
//   register is named: the normal renderer sets them only for indirect draws, the new engine sets is_indirect with them,
//   native_vkb_graphics.cpp), props.reserved = 0 (spare bits), Bindings.indexed = 0 (the new engine never starts a stage past 0 there);
// - fixed (way A; the normal list does not hold them): props.dx10_clamp = 1 and, for a pixel stage, fs.depth_before_shader = 0, the values
//   of every stage the new engine recorded (60907 stages of 219 scenes). Each fixed value is counted in the build result. A stage the game
//   uses with another value is never built from the list: the store's rule (same inputs) lets the game translate it while playing, and
//   the store counts that (OutsideFixedValues, report: built_outside_list_fixed_values) with a warning line.
// - dropped: the normal renderer's Bindings.user_data (its own push constant counter; the new engine's translation does not have it).
// A field of either side that is in none of these lists stops the conversion of the entry (counted by reason): nothing is guessed.

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/types.h"
#include "native/shadps4/shader_recompiler/backend/bindings.h"
#include "native/shadps4/shader_recompiler/frontend/fetch_shader.h"
#include "native/shadps4/shader_recompiler/params.h"
#include "native/shadps4/shader_recompiler/runtime_info.h"

namespace Vulkan::ShaderList {
struct Entry;
}

namespace Core::Vkb::ShaderList {

/// The fixed values of way A (see above)
inline constexpr bool FixedDx10Clamp = true;
inline constexpr bool FixedDepthBeforeShader = false;

/// A stage whose inputs hold another value than the list's fixed ones: the list can never hold it
[[nodiscard]] inline bool OutsideFixedValues(const VkbShader::RuntimeInfo& runtime) {
    return runtime.props.dx10_clamp != FixedDx10Clamp ||
           (runtime.hw_stage == VkbShader::HwStage::Fragment && runtime.hw.fs.depth_before_shader != FixedDepthBeforeShader);
}

/// The list of this game: TLG_VKB_SHADER_LIST (a path; "0": none), else shaders.tlgsl in shader-list/ next to the executable or one folder
/// above it (a build in port/out); the US game (CUSA03627) without a list of its own reads the European one's (CUSA03745: the same game
/// files), and says so. Empty when there is none.
[[nodiscard]] std::filesystem::path Path();

/// An id of the list file's contents (its size and a hash of its bytes): the shader pre-build's "built" marker names it. 0 when unreadable.
[[nodiscard]] u64 FileId(const std::filesystem::path& path);

/// One list entry in the new engine's form: the inputs of one translation, without the program (the game files hold it)
struct Stage {
    u32 kind{}; ///< 0 graphics, 1 compute
    VkbShader::HwStage hw_stage{};
    VkbShader::SwStage sw_stage{};
    u64 pgm_hash{};
    u64 pgm_base{};
    u64 code_words{};
    std::array<u32, VkbShader::ShaderParams::NumShaderUserData> user_data{};
    std::vector<u32> flat;
    VkbShader::RuntimeInfo runtime{};
    VkbShader::Backend::Bindings start{};
    u64 salt{};
    u32 local_x{};
    std::optional<VkbShader::Gcn::FetchShaderData> fetch; ///< the vertex input layout in place of the fetch shader's code
    std::vector<std::pair<u64, std::vector<u8>>> regions; ///< the recorded memory reads, without the program
    bool fixed_dx10_clamp{};          ///< the conversion set dx10_clamp to its fixed value (every stage)
    bool fixed_depth_before_shader{}; ///< the conversion set depth_before_shader to its fixed value (a pixel stage)
};

/// The new engine's form of a normal list entry; nullopt with the reason in `why` when a field has no rule (see above) or the entry is not
/// one the new engine can build (a geometry stage, a damaged entry).
[[nodiscard]] std::optional<Stage> Convert(const Vulkan::ShaderList::Entry& entry, std::string& why);

/// The files of the list build in the store folder (they go with it: the launcher's "clear shader cache" removes them, and a new build of
/// the engine, card or setting has a new folder without them). Progress: the entry a build process works on. Result: its counts. Left out: a
/// first line "list <id>", then one line per entry a build process stopped on (its number, how it stopped). Built: the marker of a finished
/// build, first line "list <id> game <files>": a start that finds it with the same first line builds nothing.
inline std::filesystem::path ProgressFile(const std::filesystem::path& store, u32 part) {
    return store / ("list-build-progress-" + std::to_string(part) + ".txt");
}
inline std::filesystem::path ResultFile(const std::filesystem::path& store, u32 part) {
    return store / ("list-build-result-" + std::to_string(part) + ".txt");
}
inline std::filesystem::path LeftOutFile(const std::filesystem::path& store) {
    return store / "list-build-left-out.txt";
}
inline std::filesystem::path BuiltMarker(const std::filesystem::path& store) {
    return store / "list-built.txt";
}

/// The game folder on the host; empty for a packed game (.zar), which the list cannot read yet.
[[nodiscard]] std::filesystem::path HostGameDir();

} // namespace Core::Vkb::ShaderList
