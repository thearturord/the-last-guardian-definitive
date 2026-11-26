// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
// TLG_VKB_SHADER_MEMO and TLG_VKB_SHADER_PREBUILD: the new engine's translated shader stages, kept for the session and on disk.
//
// The new engine translates a stage (program + the settings its translation reads) into a Vulkan module the first time a draw or dispatch
// needs it; the translation is most of the cost of a first draw (issue 111: 87% on this box, the driver 12%). The store keeps every
// translation in the player's cache folder, so a later start takes it instead of translating again, and the shader pre-build fills it
// before the game needs it. A stored translation is used only where its inputs are the same: the program (its length and a 128-bit hash
// of its words), the settings handed to the translator, where its bindings start, a caller's salt, and the kinds of the resources the
// stage reads through this draw's user data (the normal renderer's rule, StageSpecialization). That is the rule of the session store of
// pull request 120 (issue 111), made persistent.
//
// Folder: <user>/cache/<serial>-vkb-shaders/<id>/, id = a hash of the engine's own module file's size and time (any other build: another folder), of
// the translator's profile (another graphics card, driver or setting: another folder) and of every other setting the translator reads
// (TranslatorEnv in the .cpp; port/tests/test_vkb_store_settings.py keeps that list complete). Folders of other ids are removed when a new
// one is opened. The launcher's "clear shader cache" removes cache/<serial>* and so this folder too.
//
// Two switches, each 0/off, 1/on or check (any other value: off, with a warning):
// TLG_VKB_SHADER_MEMO (default on, issue 160 step 1): every translation is kept in memory for the session, so a new settings combination
// (variant) of a stage translated before with the same inputs takes it instead of translating it again (on this box about three quarters of
// all translations were such repeats). Nothing is read from or written to disk. =0: every stage is translated (the old behaviour).
// TLG_VKB_SHADER_PREBUILD (default off): the store is also on disk (the folder below) and the shader pre-build fills it. On or check
// keeps the stages in memory too, whatever TLG_VKB_SHADER_MEMO says.
// check (in either switch): every stage is translated anyway, and a stored entry that differs from the fresh translation is counted and
// logged as an error (the fresh one is used).

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json_fwd.hpp>

#include "common/types.h"
#include "native/native_vkb_write_window.h"
#include "native/shadps4/shader_recompiler/backend/bindings.h"
#include "native/shadps4/shader_recompiler/info.h"
#include "native/shadps4/shader_recompiler/params.h"
#include "native/shadps4/shader_recompiler/profile.h"
#include "native/shadps4/shader_recompiler/runtime_info.h"

namespace Core::Vkb::ShaderStore {

enum class Mode : u8 { Off, On, Check };
/// The store's mode for its callers: off only when both switches are off; check when either is check
[[nodiscard]] Mode GetMode();
/// TLG_VKB_SHADER_PREBUILD is on or check: the store's folder on disk, its writer, the shader recipes and the shader pre-build
[[nodiscard]] bool OnDisk();

/// Opens the store for `profile` (the translator's; once, later calls are ignored): with OnDisk, finds and lists its folder; when that fails
/// nothing is read or written (said once in the log) and translations are kept in memory for the session, as without OnDisk.
void Open(const VkbShader::Profile& profile);

/// What one translation hands back to its caller
struct Translation {
    std::unique_ptr<VkbShader::Info> info; ///< after translation, with the user data of the draw it was asked for
    VkbShader::RuntimeInfo runtime{};       ///< after translation (the translator completes it)
    VkbShader::Backend::Bindings end{};     ///< the binding counter after the stage
    std::vector<u32> spv;
    std::vector<WriteWindow::Window> windows; ///< compute: the write windows of the translated program (graphics: empty)
};

/// The inputs of one translation. `salt`: anything else of the caller's that its results depend on (0: nothing).
struct Inputs {
    const VkbShader::ShaderParams& params;
    const VkbShader::RuntimeInfo& runtime; ///< as handed to the translator
    VkbShader::Backend::Bindings start{};
    u64 salt{};
};

/// A stored translation for these inputs: true and `out` filled (the info is a copy, set to `in`'s user data and program base).
bool Find(const Inputs& in, Translation& out);
/// The store's folder is listed (Open's background part is done): ForWarm finds what is on disk
[[nodiscard]] bool Ready();
/// The pipeline warm-up (native_vkb_pipeline_warm.h): the first stored translation of `pgm_hash` with these settings, start and salt,
/// without the program words and the resource kinds (both need a draw's guest memory). Its info has no user data: the warm-up only builds
/// the pipeline from its binding lists. False: none stored.
bool ForWarm(u64 pgm_hash, const VkbShader::RuntimeInfo& runtime, const VkbShader::Backend::Bindings& start, u64 salt, Translation& out);
/// Keeps a fresh translation (`info` still holds `in`'s user data) in memory and, once the folder is open, on disk. False when the stage is
/// never kept (a geometry stage: counted).
bool Keep(const Inputs& in, const VkbShader::Profile& profile, const VkbShader::Info& info, const VkbShader::RuntimeInfo& runtime, const VkbShader::Backend::Bindings& end,
          const std::vector<u32>& spv, const std::vector<WriteWindow::Window>& windows, u64 translate_ns);
/// Check mode: compares a stored translation with a fresh one; false (counted, one error line per program) when they differ.
bool Same(const Inputs& in, const Translation& stored, const VkbShader::Info& info, const VkbShader::RuntimeInfo& runtime, const VkbShader::Backend::Bindings& end,
          const std::vector<u32>& spv, const std::vector<WriteWindow::Window>& windows);

/// The first member in which a translation differs from `a` (nullptr: the same), compared member by member as Same does, without counting:
/// the shader list's maker checks each entry with it (native_vkb_shader_recipe.h).
[[nodiscard]] const char* Differs(const Translation& a, const VkbShader::Info& info, const VkbShader::RuntimeInfo& runtime, const VkbShader::Backend::Bindings& end,
                                  const std::vector<u32>& spv, const std::vector<WriteWindow::Window>& windows);

/// Test tool (tlg-probe --vkb-compare-stores A B): pairs the stages of two store folders by their inputs and compares each pair as Differs
/// does (and the resource kinds); prints the counts and the summed translation times. 0 = every stage on both sides and all the same.
int CompareStores(const std::filesystem::path& dir_a, const std::filesystem::path& dir_b);

/// Lists the store's folder again and indexes the files it did not know yet (the shader pre-build's build processes write into it while the
/// game runs: native_vkb_shader_prebuild.h). Their .tmp files are left alone. The number of new files.
u64 Rescan();

/// A stage translated without the store although it is on, because a debug tool needs the translated program (census, probe): counted.
void NoteBypass();

/// The store's folder (empty while it is not open, or the store is off).
[[nodiscard]] std::filesystem::path Folder();

/// The store's folder also holds profile.bin (the translator profile, raw) and settings.txt (TranslatorSettings in the .cpp), written at
/// Open: the shader pre-build's build process (tlg-probe --vkb-build-store, native_vkb_shader_recipe.h) translates with that profile.
/// OpenForBuild opens `folder` for that process: reads its profile into `profile`, and refuses (false, the reason in `why`) a folder whose
/// settings are not this process's own (the process must run with the game's environment) or whose id is not this build's for them (another
/// build of the engine made it). Then Find and Keep work on it as in the game; Flush waits until every queued file is written.
bool OpenForBuild(const std::filesystem::path& folder, VkbShader::Profile& profile, std::string& why);
void Flush();
/// Writes a shader recipe's file on the store's writer thread (as "<path>.tmp", then renamed; only with the store on, which starts that
/// thread; false, counted and said once, without it). Counted apart from the store's own files (report: recipe_files_written,
/// recipe_files_write_failed).
bool QueueWrite(std::filesystem::path path, std::vector<u8> bytes);

/// Build statistics (TLG_SHADER_BUILD_STATS=1, any mode, also off): the same log lines as the normal renderer's (shader_recipe.cpp), so
/// one reader (port/tools/shader-list/check_run.py) counts both renderers: "Shader built while playing: <hash>_<n> (hw stage <s>)" per
/// fresh translation, "Shader used from the cache: ..., found by the lookup" at the first use of a stored one, "Frame time: <ms> ms, <n>
/// shader builds" per frame and the "Shader builds while playing: ..." summary every 5 s.
void NoteBuild(u64 pgm_hash, u32 hw_stage);
/// One frame was shown (the new engine's flip).
void NoteFrame();

[[nodiscard]] nlohmann::json Report();

} // namespace Core::Vkb::ShaderStore
