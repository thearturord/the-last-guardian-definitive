// SPDX-FileCopyrightText: Copyright 2026 tlg_pc contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// tlg: shader recipes. Every shader translation also stores its inputs (program code, user data,
// runtime info, bindings, the flattened user data and the guest memory the compiler read:
// fetch shaders, resource descriptors) in Storage::RecipeDir(). A cache that lacks a program
// (new install, new GPU or driver, a build with another recompiler) gets it translated again from
// its recipe at start, in a child process (tlg-probe --import-shaders) that maps the recorded
// memory at its guest addresses. The game then loads it like any cached shader.
// Ported from deadinside28/bloodborne_pc (docs/shader_recipes.md, render/shader_capture.cpp,
// render/shader_rebuild.cpp), thanks to @deadinside28.
#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "common/types.h"

namespace Shader {
struct Info;
struct RuntimeInfo;
namespace Backend {
struct Bindings;
}
} // namespace Shader

namespace Vulkan::ShaderRecipe {

/// Starts recording the guest memory the compiler reads on this thread.
void Begin();
/// Ends recording; the recipe of the translation since Begin(). `lookup_runtime` and `start` are
/// the runtime info and bindings as the lookup passed them (before translation adjusted them),
/// `info` the translated info (its flattened user data). Empty when nothing should be kept.
std::vector<u8> Take(const Shader::Info& info, std::span<const u32> code,
                     const Shader::RuntimeInfo& lookup_runtime,
                     const Shader::Backend::Bindings& start, size_t perm_idx);
/// tlg: the layout id of the structs a recipe and a version 2 shader list keep as raw bytes
/// (Shader::RuntimeInfo, Shader::Backend::Bindings): a hash of the bytes every member of their
/// field lists (shader_fields.h) takes, as this build lays them out. Recipes (version 2) and
/// version 2 lists carry it; one with another id is not read (equal struct sizes do not prove an
/// equal layout). TLG_SHADER_LAYOUT_CHECK=0 reads them anyway, on the struct sizes alone.
u64 LayoutId();
/// False with TLG_SHADER_LAYOUT_CHECK=0.
bool LayoutCheck();

/// tlg: the portable form (shader_fields.h) of the two structs: every member as a number, in the
/// order of one field list, so a build of any compiler reads them right (the shader list is
/// made on Linux and read on Windows too, which lays some bit fields out otherwise). The
/// From functions return false when the bytes are damaged or of another field list.
std::vector<u8> PortableRuntime(const Shader::RuntimeInfo& runtime);
bool RuntimeFromPortable(std::span<const u8> portable, Shader::RuntimeInfo& runtime);
std::vector<u8> PortableBindings(const Shader::Backend::Bindings& bindings);
bool BindingsFromPortable(std::span<const u8> portable, Shader::Backend::Bindings& bindings);
/// The id of the portable form's field lists (also of the pipeline keys): a hash of the member
/// names in order, the same in every build of the same source. A shader list carries it; a
/// build with other field lists does not read the list.
u64 PortableId();

/// tlg-probe --portable-recipes RECIPES OUT: writes the recipes of the folder RECIPES (as this
/// build reads them: its own layout id, or version 1 on the struct sizes) and its pipeline keys
/// in the portable form to the file OUT, which port/tools/shader-list/make_list.py turns into the
/// shader list. Every item is read back and compared first. 0 on success.
int PortableRecipes(const std::filesystem::path& recipe_dir, const std::filesystem::path& out);

/// Stores a recipe from Take() (no-op when empty or recipes are off).
void Save(u64 pgm_hash, size_t perm_idx, std::vector<u8>&& recipe);

/// The game's side at start: runs the import child for `cache_dir` when RecipeDir() has
/// programs the cache lacks, and waits for it. True when the child ran without errors.
/// tlg: with shader_prebuild on (shader_list.h) the shipped shader list counts too, and the
/// child runs in the background: true when it was started; WaitImport() waits for it.
bool ImportMissing(const std::filesystem::path& cache_dir);

/// shader_prebuild on: waits until the background import has finished (at once when none
/// runs). `why` names the waiter in the log; `stop` (optional) ends the wait early.
void WaitImport(const char* why, bool (*stop)(void*) = nullptr, void* stop_arg = nullptr);

/// The child (tlg-probe --import-shaders CACHE RECIPES [LIST GAME_DIR [PART PARTS]]): translates every recipe
/// whose program `cache_dir` does not have with the cache's profile.bin and writes its .spv and
/// .meta, then copies the recipe folder's pipeline keys the cache lacks (only when no recipe of
/// the folder was made by another build: the keys are raw structs too). With a shader list,
/// its recipes count too, their programs read from the game folder `game_dir`. PART of PARTS:
/// this child's share of the programs (children run in parallel). 0 on success.
int ImportChild(const std::filesystem::path& cache_dir, const std::filesystem::path& recipe_dir,
                const std::filesystem::path& list_path = {},
                const std::filesystem::path& game_dir = {}, u32 part = 0, u32 parts = 1);

/// A pipeline the game made while playing whose key the shader list lacks: one log line per key
/// and session, so players' logs alone can add the key to the list (log_recipes.py):
///   Pipeline key (not in list): <key file name> <base64 of the zlib-packed portable key>
/// The portable key (vk_pipeline_serialization.h PortableKey) holds stage hashes and pipeline
/// state, no game code. Only while the shader list is used. `key_file`: the key file's bytes.
void LogUnlistedKey(const std::string& name, std::span<const u8> key_file);
/// The same line for a key already in its portable form (a key of another kind; the
/// caller checks that the list lacks it and logs it once).
void LogPortableKey(const std::string& name, std::span<const u8> portable);

/// Counts a shader the game process translated itself (not from the cache): with
/// shader_prebuild on or TLG_SHADER_BUILD_STATS=1 each is logged, so the list can grow.
void NoteBuild(u64 pgm_hash, size_t perm_idx, u32 hw_stage);
/// TLG_SHADER_STORE_LOOKUP (default: on when shader_prebuild is on, else off): a program the
/// game meets that no pipeline key loaded is looked up in the shader cache (vk_pipeline_cache).
bool StoreLookup();
/// TLG_SHADER_STORE_FILL (default: as TLG_SHADER_STORE_LOOKUP): before the game builds a
/// permutation of a program it already has, it adds the cache's permutations of that program it
/// does not hold yet (a program loaded by pipeline keys holds only those its keys name).
bool StoreFill();
/// A program the lookup found in the cache, with its number of permutations (build stats).
void NoteLookup(u64 pgm_hash, size_t perms, u32 hw_stage);
/// Cache permutations a look of StoreFill could not use, by reason: meta refused (another
/// version, a shader rebuilt every session, another program or stage), no SPIR-V, the driver
/// refused the module. Each is counted once per session and program.
struct FillRefused {
    u32 meta = 0, spirv = 0, module = 0;
};
/// A known program's look into the cache before a build (StoreFill): permutations added, and
/// those it left out (build stats; a look that found nothing counts too).
void NoteFill(u64 pgm_hash, size_t perms, const FillRefused& refused, u32 hw_stage);
/// The first use of a permutation that came from the cache: through a pipeline key (the
/// warm-up) or the lookup (build stats: the check of the shader list counts them).
void NoteCacheUse(u64 pgm_hash, size_t perm_idx, u32 hw_stage, bool lookup);
/// One game frame was shown (the slow-frame count of ReportBuilds).
void NoteFrame();
/// At exit: shaders built while playing, and how many of them in slow frames.
void ReportBuilds();

} // namespace Vulkan::ShaderRecipe
