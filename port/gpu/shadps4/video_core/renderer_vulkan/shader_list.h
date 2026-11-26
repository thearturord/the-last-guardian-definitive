// SPDX-FileCopyrightText: Copyright 2026 tlg_pc contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// tlg: the shader list (Performance > Build shaders ahead, tlg.ini shader_prebuild). A shipped list of shader recipes
// without program bytes: each recipe keeps its pipeline state, and its program is named by where
// the player's own game files hold it (port/tools/shader-list/make_list.py writes the list). The
// recipe import child (shader_recipe.h) reads the programs from the game folder and builds the
// shaders the cache lacks, in the background at start.
#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/frontend/fetch_shader.h"

namespace Vulkan::ShaderList {

/// Build shaders ahead (tlg.ini shader_prebuild, on by default; TLG_SHADER_PREBUILD wins), as it
/// was at start: the list is read and the import runs in the background.
bool Enabled();

/// The list: TLG_SHADER_LIST, else shaders.tlgsl in shader-list/ next to the executable or one
/// folder above it (a build in port/out). One list for every region (made from the European
/// CUSA03745; the US game has the same game files). Empty when there is none.
std::filesystem::path Path();

struct Source {
    u8 kind{}; ///< 1: an entry of a .psarc archive (ident = MD5 of its path); 2: a game file
    std::vector<u8> ident;
};

struct Entry {
    u32 hw_stage{}, sw_stage{};
    u64 pgm_hash{}, perm_idx{}, pgm_base{};
    std::vector<u32> user_data, vs_copy, flat;
    /// Shader::RuntimeInfo and Shader::Backend::Bindings, as this build lays them out
    std::vector<u8> runtime, start;
    u32 source{};
    u64 offset{};
    u32 length{}, crc{};
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    std::vector<std::pair<u64, std::vector<u8>>> regions; ///< descriptors the compiler reads
};

struct List {
    std::vector<Source> sources;
    std::vector<Entry> entries;
    std::vector<std::pair<std::string, std::vector<u8>>> keys; ///< pipeline key files
    u32 keys_refused = 0;         ///< version 3: keys this build does not read (key version)
    std::string keys_refused_why; ///< the reason of the last one
    /// version 3: keys of kind OtherKeyKind (a key form this build does not read), as they are
    std::vector<std::vector<u8>> other_keys;
};

/// A pipeline key kind kept in the list as it is (the second number of a portable key; the kinds
/// this build reads are ShaderFields::KindGraphics and KindCompute).
constexpr u32 OtherKeyKind = 2;

/// The kind of a portable key (its second LEB128 number); ~0u when it is too short.
inline u32 PortableKind(std::span<const u8> portable) {
    size_t pos = 0;
    u64 value = 0;
    for (int field = 0; field < 2; ++field) {
        value = 0;
        for (int shift = 0;; shift += 7) {
            if (pos >= portable.size() || shift > 63) {
                return ~0u;
            }
            const u8 byte = portable[pos++];
            value |= u64(byte & 0x7f) << shift;
            if (!(byte & 0x80)) {
                break;
            }
        }
    }
    return value > 0xffffffffull ? ~0u : u32(value);
}

/// Reads a list; nullopt (with the reason in `error`) when it is damaged, was made with other
/// field lists (version 3, ShaderRecipe::PortableId()) or, a version 2 list, by a build with other
/// struct sizes or another struct layout (ShaderRecipe::LayoutId()). Version 3 runtime infos,
/// bindings and pipeline keys are turned into this build's structs and key files here.
std::optional<List> Read(const std::filesystem::path& path, std::string& error);

/// The program hashes a list names, for the game's side (which shaders are covered); empty with
/// the reason in `error` when the list cannot be read. `key_names` (optional): the file names of
/// its pipeline keys as this build names them, sorted.
std::vector<u64> Programs(const std::filesystem::path& path, std::string& error,
                          std::vector<std::string>* key_names = nullptr);

/// Reads programs out of the game folder: the ps4_shader_cache entries of trico_bootp.psarc and
/// trico_boot.psarc (the update first, as the game does) and files of the folder.
class GameFiles {
public:
    GameFiles(std::filesystem::path game_dir, const List& list);
    ~GameFiles();
    /// The program of `entry`, checked by length and CRC-32; nullopt when it is not there.
    std::optional<std::vector<u8>> Program(const Entry& entry);

private:
    struct Archive;
    std::filesystem::path game_dir;
    const List& list;
    std::vector<std::unique_ptr<Archive>> archives;
    bool archives_open = false;
    std::map<u32, std::vector<std::vector<u8>>> source_data; ///< per source, every candidate
};

} // namespace Vulkan::ShaderList
