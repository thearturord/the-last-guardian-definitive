// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
// TLG_VKB_SHADER_PREBUILD (on or check): the shader pre-build of the new engine. When the store's folder (native_vkb_shader_store.h) is opened
// in the game and a shader list ships with it (native_vkb_shader_list.h) that was not built into this folder yet (no list-built.txt naming
// this list), the game starts build processes in the background (tlg-probe --vkb-build-store FOLDER --list LIST GAME PART PARTS, its own
// executable; TLG_VKB_SHADER_PREBUILD_JOBS of them, default half the CPU threads, at most 8), which fill the folder from the list while the
// game runs. The store reads their files at each stage load (Rescan) and when the build ends. A stage load waits for the build (at most
// TLG_VKB_SHADER_PREBUILD_WAIT_S, default 120 s), except the first load of the session (the title): the Lua stage-load hook of
// port/gpu/shim/tlg_lua.cpp, shared with the normal renderer's pre-build. A complete build writes list-built.txt into the folder (naming the
// list and the game files), so the next start builds nothing; the launcher's "clear shader cache" removes the folder and with it the marker.
// The build processes end with the game (Linux: a parent-death signal; Windows: a job object).
//
// A build process never stops the game: one that crashes, fails an assertion or makes no progress on one entry for
// TLG_VKB_SHADER_PREBUILD_ENTRY_S (default 30 s) is started again without that entry (list-build-left-out.txt, so it costs one restart once per
// folder), at most 32 times per part; one that runs longer than TLG_VKB_SHADER_PREBUILD_PART_S (default 600 s) is stopped and its part ends
// without blaming an entry (the next start goes on with it: no marker). Everything is
// counted and logged: entries built, already there, and skipped by reason (programs the game files lack, entries the build could not make).
//
// Off (TLG_VKB_SHADER_PREBUILD=0, the default, or TLG_VKB not 1) nothing here runs. TLG_VKB_SHADER_LIST=0: no list (the store alone).

#include <filesystem>
#include <nlohmann/json_fwd.hpp>

#include "common/types.h"

namespace Core::Vkb::Prebuild {

/// The new engine runs with the store on: the stage-load hook is needed (read once)
[[nodiscard]] bool HookWanted();

/// From the store's opening in the game (its folder): starts the background build when the list needs one. Once.
void Start(const std::filesystem::path& store_folder);

/// A stage load (the Lua hook): reads the files the build wrote so far and, unless `first` (the session's first load), waits for the build
/// to end, at most TLG_VKB_SHADER_PREBUILD_WAIT_S. At once when no build runs.
void StageLoad(const char* what, bool first);

/// The background build runs
[[nodiscard]] bool Running();
/// The program is named by the list (build statistics)
[[nodiscard]] bool Listed(u64 pgm_hash);
/// Waits that hit the limit
[[nodiscard]] u32 WaitsGivenUp();

[[nodiscard]] nlohmann::json Report();

} // namespace Core::Vkb::Prebuild
