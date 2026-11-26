// SPDX-FileCopyrightText: Copyright 2026 tlg_pc contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// tlg: the shader list, see shader_list.h.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#endif

#include <fmt/format.h>
#include <miniz.h>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/shader_list.h"
#include "video_core/renderer_vulkan/shader_recipe.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "tlg_settings.h"

namespace Vulkan::ShaderList {

namespace {

constexpr std::array<char, 8> Magic = {'T', 'L', 'G', 'S', 'L', 'I', 'S', 'T'};
/// 3: the runtime info, bindings and pipeline keys in the portable form (shader_fields.h), the
/// portable id (ShaderRecipe::PortableId()) after the payload size: one list for every build.
/// 2: raw struct bytes, the layout id (ShaderRecipe::LayoutId()) after the payload size: read
/// only by a build that lays the structs out the same way (make_list.py --raw still writes it).
/// Version 1 lists had no id and are not read: their raw struct bytes could be laid out otherwise.
constexpr u32 Version = 3, RawVersion = 2;
constexpr u8 SourcePsarc = 1, SourceGameFile = 2;

std::vector<u8> ReadAll(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{file}, {}};
}

struct Reader {
    std::span<const u8> in;
    size_t pos = 0;
    bool ok = true;
    template <typename T>
    T Pod() {
        T value{};
        if (!ok || pos + sizeof(T) > in.size()) {
            ok = false;
            return value;
        }
        std::memcpy(&value, in.data() + pos, sizeof(T));
        pos += sizeof(T);
        return value;
    }
    std::vector<u8> Raw(size_t size) {
        if (!ok || size > in.size() - pos) {
            ok = false;
            return {};
        }
        std::vector<u8> bytes(in.begin() + pos, in.begin() + pos + size);
        pos += size;
        return bytes;
    }
    std::vector<u8> Blob() {
        return Raw(Pod<u32>());
    }
    std::vector<u32> Words() {
        const auto bytes = Blob();
        std::vector<u32> words(bytes.size() / sizeof(u32));
        std::memcpy(words.data(), bytes.data(), words.size() * sizeof(u32));
        return words;
    }
};

/// The list's payload, unpacked, and its version; empty on any damage.
std::vector<u8> Payload(const std::filesystem::path& path, u32& version, std::string& error) {
    const auto file = ReadAll(path);
    constexpr size_t Header = Magic.size() + 4 + 4 + 8 + 8;
    if (file.size() < Magic.size() + 4 ||
        std::memcmp(file.data(), Magic.data(), Magic.size()) != 0) {
        error = "not a shader list";
        return {};
    }
    std::memcpy(&version, file.data() + Magic.size(), 4);
    if (version != Version && version != RawVersion) {
        error = fmt::format("list version {} (this build reads {} and {})", version, RawVersion,
                            Version);
        return {};
    }
    if (file.size() < Header) {
        error = "damaged list";
        return {};
    }
    u64 size{}, id{};
    std::memcpy(&size, file.data() + Magic.size() + 8, 8);
    std::memcpy(&id, file.data() + Magic.size() + 16, 8);
    if (version != RawVersion && id != ShaderRecipe::PortableId()) {
        error = fmt::format("list made with other field lists (portable id {:#018x}, this build "
                            "{:#018x})",
                            id, ShaderRecipe::PortableId());
        return {};
    }
    if (const u64 layout = id; version == RawVersion && layout != ShaderRecipe::LayoutId() &&
                               ShaderRecipe::LayoutCheck()) {
        error = fmt::format("list made by a build with another struct layout (layout id {:#018x}, "
                            "this build {:#018x})",
                            layout, ShaderRecipe::LayoutId());
        return {};
    }
    if (size > (u64{256} << 20)) {
        error = "damaged list";
        return {};
    }
    std::vector<u8> payload(size);
    mz_ulong length = mz_ulong(size);
    if (mz_uncompress(payload.data(), &length, file.data() + Header,
                      mz_ulong(file.size() - Header)) != MZ_OK ||
        length != size) {
        error = "damaged list";
        return {};
    }
    return payload;
}

bool ReadFetch(Reader& r, Entry& e) {
    if (r.Pod<u8>() == 0) {
        return r.ok;
    }
    auto& fetch = e.fetch.emplace();
    fetch.size = r.Pod<u32>();
    fetch.vertex_offset_sgpr = r.Pod<s8>();
    fetch.instance_offset_sgpr = r.Pod<s8>();
    const u8 count = r.Pod<u8>();
    for (u8 i = 0; r.ok && i < count; ++i) {
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
    return r.ok;
}

/// A version 3 entry's runtime info and bindings: from the portable form into this build's
/// structs, kept as their bytes (as a version 2 list holds them).
bool Portable(Reader& r, Entry& e) {
    const auto runtime_bytes = r.Blob();
    const auto start_bytes = r.Blob();
    Shader::RuntimeInfo runtime{};
    Shader::Backend::Bindings start{};
    if (!r.ok || !ShaderRecipe::RuntimeFromPortable(runtime_bytes, runtime) ||
        !ShaderRecipe::BindingsFromPortable(start_bytes, start)) {
        r.ok = false;
        return false;
    }
    const auto* p = reinterpret_cast<const u8*>(&runtime);
    e.runtime.assign(p, p + sizeof(runtime));
    p = reinterpret_cast<const u8*>(&start);
    e.start.assign(p, p + sizeof(start));
    return true;
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

u32 Crc(std::span<const u8> bytes) {
    return u32(mz_crc32(MZ_CRC32_INIT, bytes.data(), bytes.size()));
}

} // namespace

bool Enabled() {
    // Performance > Build shaders ahead (tlg.ini shader_prebuild, TLG_SHADER_PREBUILD wins), as
    // it was when the game started.
    return TlgSettings::Get().startup_shader_prebuild;
}

std::filesystem::path Path() {
    if (const char* env = std::getenv("TLG_SHADER_LIST"); env && *env) {
        return *env == '0' && env[1] == '\0' ? std::filesystem::path{}
                                             : std::filesystem::u8path(env);
    }
    // One list for every region: the US disc (CUSA03627) has the same shader archives and
    // eboot.bin as the European CUSA03745 1.03 it was made from (SHA-256 compared); for the other
    // discs each entry's length and CRC-32 check every program read from the player's files: one
    // that differs is skipped, never built wrong.
    const auto dir = ExecutableDir();
    for (const auto& candidate : {dir / "shader-list" / "shaders.tlgsl",
                                  dir.parent_path() / "shader-list" / "shaders.tlgsl"}) {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error)) {
            return candidate;
        }
    }
    return {};
}

std::optional<List> Read(const std::filesystem::path& path, std::string& error) {
    u32 version{};
    const auto payload = Payload(path, version, error);
    if (payload.empty()) {
        return std::nullopt;
    }
    Reader r{payload};
    const bool raw = version == RawVersion;
    if (raw && (r.Pod<u32>() != sizeof(Shader::RuntimeInfo) ||
                r.Pod<u32>() != sizeof(Shader::Backend::Bindings))) {
        error = "list made for another shader compiler (struct sizes differ)";
        return std::nullopt;
    }
    List list;
    const u32 sources = r.Pod<u32>();
    for (u32 i = 0; r.ok && i < sources; ++i) {
        auto& s = list.sources.emplace_back();
        s.kind = r.Pod<u8>();
        s.ident = r.Blob();
    }
    const u32 entries = r.Pod<u32>();
    for (u32 i = 0; r.ok && i < entries; ++i) {
        auto& e = list.entries.emplace_back();
        e.hw_stage = r.Pod<u32>();
        e.sw_stage = r.Pod<u32>();
        e.pgm_hash = r.Pod<u64>();
        e.perm_idx = r.Pod<u64>();
        e.pgm_base = r.Pod<u64>();
        e.user_data = r.Words();
        e.vs_copy = r.Words();
        e.flat = r.Words();
        if (raw) {
            e.runtime = r.Raw(sizeof(Shader::RuntimeInfo));
            e.start = r.Raw(sizeof(Shader::Backend::Bindings));
        } else if (!Portable(r, e)) {
            break;
        }
        e.source = r.Pod<u32>();
        e.offset = r.Pod<u64>();
        e.length = r.Pod<u32>();
        e.crc = r.Pod<u32>();
        if (!ReadFetch(r, e)) {
            break;
        }
        const u32 regions = r.Pod<u32>();
        for (u32 j = 0; r.ok && j < regions; ++j) {
            const auto address = r.Pod<u64>();
            e.regions.emplace_back(address, r.Blob());
        }
        if (e.source >= list.sources.size()) {
            r.ok = false;
        }
    }
    const u32 keys = r.Pod<u32>();
    for (u32 i = 0; r.ok && i < keys; ++i) {
        if (!raw) {
            // The key file of this build, named as this build names it.
            const auto portable = r.Blob();
            if (r.ok && PortableKind(portable) == OtherKeyKind) {
                list.other_keys.push_back(portable); // a key of another kind: kept as it is
                continue;
            }
            std::string name, why;
            if (auto bytes = KeyFromPortable(portable, name, why); bytes && r.ok) {
                list.keys.emplace_back(std::move(name), std::move(*bytes));
            } else if (r.ok) {
                ++list.keys_refused;
                list.keys_refused_why = why;
            }
            continue;
        }
        const auto name = r.Blob();
        list.keys.emplace_back(std::string{name.begin(), name.end()}, r.Blob());
        const auto& key = list.keys.back().first;
        if (key.empty() || key.find_first_of("/\\:") != std::string::npos || key[0] == '.') {
            r.ok = false; // a key file name never leaves the cache folder
        }
    }
    if (!r.ok || r.pos != payload.size()) {
        error = "damaged list";
        return std::nullopt;
    }
    return list;
}

std::vector<u64> Programs(const std::filesystem::path& path, std::string& error,
                          std::vector<std::string>* key_names) {
    const auto list = Read(path, error);
    std::vector<u64> programs;
    if (list) {
        for (const auto& e : list->entries) {
            programs.push_back(e.pgm_hash);
        }
        if (key_names) {
            for (const auto& [name, _bytes] : list->keys) {
                key_names->push_back(name);
            }
            std::ranges::sort(*key_names);
        }
    }
    std::ranges::sort(programs);
    programs.erase(std::unique(programs.begin(), programs.end()), programs.end());
    return programs;
}

/// One PSARC archive (big-endian table of contents, zlib blocks): its entries by name MD5.
struct GameFiles::Archive {
    struct Item {
        std::array<u8, 16> md5;
        u32 first_block;
        u64 size, offset;
    };
    std::ifstream file;
    u32 block = 0;
    std::vector<Item> items;
    std::vector<u32> block_sizes;

    static u64 Be(const u8* p, size_t n) {
        u64 v = 0;
        for (size_t i = 0; i < n; ++i) {
            v = v << 8 | p[i];
        }
        return v;
    }

    bool Open(const std::filesystem::path& path) {
        file.open(path, std::ios::binary);
        u8 head[32];
        if (!file.read(reinterpret_cast<char*>(head), sizeof(head)) ||
            std::memcmp(head, "PSAR", 4) != 0 || std::memcmp(head + 8, "zlib", 4) != 0) {
            return false;
        }
        const u64 toc_length = Be(head + 12, 4), entry_size = Be(head + 16, 4),
                  count = Be(head + 20, 4);
        block = u32(Be(head + 24, 4));
        if (toc_length < 32 || entry_size < 30 || toc_length - 32 < count * entry_size ||
            block == 0 || toc_length > (u64{64} << 20)) {
            return false;
        }
        std::vector<u8> toc(toc_length - 32);
        if (!file.read(reinterpret_cast<char*>(toc.data()), std::streamsize(toc.size()))) {
            return false;
        }
        for (u64 i = 0; i < count; ++i) {
            const u8* e = toc.data() + i * entry_size;
            Item item{};
            std::memcpy(item.md5.data(), e, 16);
            item.first_block = u32(Be(e + 16, 4));
            item.size = Be(e + 20, 5);
            item.offset = Be(e + 25, 5);
            items.push_back(item);
        }
        const size_t width = block <= 0x10000 ? 2 : block <= 0x1000000 ? 3 : 4;
        for (size_t at = count * entry_size; at + width <= toc.size(); at += width) {
            block_sizes.push_back(u32(Be(toc.data() + at, width)));
        }
        return true;
    }

    std::optional<std::vector<u8>> Read(std::span<const u8> md5) {
        const auto it = std::ranges::find_if(items, [&](const Item& item) {
            return md5.size() == 16 && std::memcmp(item.md5.data(), md5.data(), 16) == 0;
        });
        if (it == items.end() || it->size > (u64{64} << 20)) {
            return std::nullopt;
        }
        std::vector<u8> out;
        out.reserve(it->size);
        file.clear();
        file.seekg(std::streamoff(it->offset));
        std::vector<u8> packed;
        for (u64 got = 0, b = it->first_block; got < it->size; ++b) {
            if (b >= block_sizes.size()) {
                return std::nullopt;
            }
            const u64 want = std::min<u64>(block, it->size - got);
            const u64 stored = block_sizes[b] ? block_sizes[b] : block;
            packed.resize(stored);
            if (!file.read(reinterpret_cast<char*>(packed.data()), std::streamsize(stored))) {
                return std::nullopt;
            }
            if (block_sizes[b] == 0 || stored == want) {
                out.insert(out.end(), packed.begin(), packed.begin() + want);
            } else {
                const size_t at = out.size();
                out.resize(at + want);
                mz_ulong length = mz_ulong(want);
                if (mz_uncompress(out.data() + at, &length, packed.data(), mz_ulong(stored)) !=
                        MZ_OK ||
                    length != want) {
                    return std::nullopt;
                }
            }
            got += want;
        }
        return out;
    }
};

GameFiles::GameFiles(std::filesystem::path game_dir_, const List& list_)
    : game_dir{std::move(game_dir_)}, list{list_} {}

GameFiles::~GameFiles() = default;

std::optional<std::vector<u8>> GameFiles::Program(const Entry& entry) {
    const auto& source = list.sources[entry.source];
    auto [it, fresh] = source_data.try_emplace(entry.source);
    if (fresh) {
        if (source.kind == SourcePsarc) {
            if (!std::exchange(archives_open, true)) {
                for (const char* name : {"trico_bootp.psarc", "trico_boot.psarc"}) {
                    auto archive = std::make_unique<Archive>();
                    if (archive->Open(game_dir / name)) {
                        archives.push_back(std::move(archive));
                    }
                }
            }
            for (auto& archive : archives) {
                if (auto data = archive->Read(source.ident)) {
                    it->second.push_back(std::move(*data));
                }
            }
        } else if (source.kind == SourceGameFile) {
            // A plain file name of the game folder only (eboot.bin).
            const std::string name{source.ident.begin(), source.ident.end()};
            if (!name.empty() && name.find_first_of("/\\:") == std::string::npos &&
                name.find("..") == std::string::npos) {
                it->second.push_back(ReadAll(game_dir / std::filesystem::u8path(name)));
            }
        }
        // The permutations of a program follow each other, so the archive entry of the last
        // program is kept; whole game files (eboot.bin) stay, they serve several programs.
        for (auto old = source_data.begin(); old != source_data.end();) {
            old = old->first != entry.source && list.sources[old->first].kind == SourcePsarc
                      ? source_data.erase(old)
                      : std::next(old);
        }
    }
    for (const auto& data : source_data[entry.source]) {
        if (entry.offset <= data.size() && entry.length <= data.size() - entry.offset) {
            const std::span<const u8> bytes{data.data() + entry.offset, entry.length};
            if (Crc(bytes) == entry.crc) {
                return std::vector<u8>{bytes.begin(), bytes.end()};
            }
        }
    }
    return std::nullopt;
}

} // namespace Vulkan::ShaderList
