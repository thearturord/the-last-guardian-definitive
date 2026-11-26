// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
// TLG_VKB_SHADER_RECIPES: the new engine's shader recipes, the inputs of every fresh translation, so a build process can fill the shader
// store (native_vkb_shader_store.h) again without the game: after a new build (its store folder starts empty) or a cleared store.
//
// A recipe holds what the translator read: the stage, the program's hash, base and code, the 16 user data words, the guest memory the
// translator read through them (descriptor tables through Info::ReadUdReg, the fetch shader's code, tessellation constants; the bytes as
// the translator got them), the flattened user data after translation (the resource tables' walk, replayed instead of walked again), the
// runtime info as handed to the translator, where its bindings start, the caller's salt and the compute stage's local size x. It is the
// normal renderer's way (commit 4e42ca0: video_core/renderer_vulkan/shader_recipe.cpp, ported from deadinside28/bloodborne_pc), on the
// new engine's own translator copy (port/gpu/native/shadps4/shader_recompiler: guest_read_hook, replay_flat, guest_valid_hook in info.h).
//
// Folder: <user>/cache/<serial>-vkb-recipes/<program>_<hw stage>_<sw stage>_<digest>.vkr, written by the store's writer thread. It holds
// program code and game memory: the player's own cache, never shipped (a shipped list names the game file instead). The launcher's "clear
// shader cache" removes cache/<serial>* and so this folder too (port/tests/test_shader_cache.py checks the rule).
// A recipe carries a layout id (native_vkb_shader_fields.h): a build that lays RuntimeInfo or Bindings out otherwise does not read it.
//
// Recording: TLG_VKB_SHADER_RECIPES=1 with the store on disk (TLG_VKB_SHADER_PREBUILD=1 or check; the session memo TLG_VKB_SHADER_MEMO alone records none); default off. A recipe is kept for each
// translation the store keeps (Keep), so one per stage variant the store lacked.

#include <filesystem>
#include <nlohmann/json_fwd.hpp>

#include "native/native_vkb_shader_store.h"

namespace Core::Vkb::ShaderRecipe {

/// TLG_VKB_SHADER_RECIPES=1 (read once)
[[nodiscard]] bool On();

/// Records the guest memory the translator reads on this thread while it lives (when `active`; inactive: nothing happens). Take() keeps
/// the recipe; without it the recording is dropped (a failed translation).
class Recording {
public:
    explicit Recording(bool active);
    ~Recording();
    Recording(const Recording&) = delete;
    Recording& operator=(const Recording&) = delete;
    /// Ends the recording and queues the recipe of the translation since the start: `in` as handed to the store, `info` after
    /// translation (its flattened user data), `local_x` the compute stage's local size x (graphics: 0).
    void Take(const ShaderStore::Inputs& in, const VkbShader::Info& info, u32 local_x);

private:
    bool active_;
};

/// The build process (tlg-probe --vkb-build-store STORE RECIPES): opens the game's store folder `store_folder` (ShaderStore::OpenForBuild:
/// its profile, refused when the settings or the build differ) and, for every recipe of `recipe_dir` whose stage the store lacks (the
/// store's own rule, Find), maps the recorded memory at its guest addresses, translates with the recorded inputs and the replayed
/// flattened user data and keeps the stage through the store's own Keep (its encoder and file names). A recipe it cannot read, map or
/// translate is counted and named. No graphics device is used. 0 when the folder was opened (skipped recipes are reported, not
/// failures), 2 when the store folder was refused, 3 when the recipe folder cannot be read, 4 when stages could not be written.
/// Only the translator's four known guest reads are recorded (Info::ReadUdReg through a pointer, the fetch shader, the tessellation
/// constants, the program); a read by another way would see zeros on a mapped page (or stop the process on an unmapped one). Check mode
/// (TLG_VKB_SHADER_PREBUILD=check) compares every stage the game uses with a fresh translation: the proof for a built store.
int BuildStore(const std::filesystem::path& store_folder, const std::filesystem::path& recipe_dir);

/// The shader pre-build's build process for the list (tlg-probe --vkb-build-store STORE --list LIST GAME [PART PARTS]): opens the store as
/// BuildStore does, reads LIST (the normal renderer's shader list, with its reader) and builds the list's entries of its part (by program
/// hash) the store lacks, each converted to the new engine's inputs (ShaderList::Convert, native_vkb_shader_list.h; the values it fixes are
/// counted: fixed_dx10_clamp, fixed_depth_before_shader) and its program read from the game folder `game_dir` (length and CRC-32 checked).
/// Before each entry it writes the entry's number into list-build-progress-<part>.txt of the store;
/// entries named in list-build-left-out.txt (a process stopped on them before: the game's parent, native_vkb_shader_prebuild.h) are skipped.
/// Every entry of the part ends counted: built, already in the store, or skipped with its reason (list-build-result-<part>.txt, and the
/// first three of each reason in the log). 0 done, 2 refused, 3 list unreadable,
/// 4 writes failed.
int BuildList(const std::filesystem::path& store_folder, const std::filesystem::path& list, const std::filesystem::path& game_dir, u32 part, u32 parts);

/// recipes kept, recordings dropped (the translation failed or the store did not keep the stage), the folder
[[nodiscard]] nlohmann::json Report();

} // namespace Core::Vkb::ShaderRecipe
