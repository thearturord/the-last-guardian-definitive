// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: the shader recipe import child reads the profile.bin the cache writes. From v0.10.0 to
// v0.12.0 the cache wrote a version header the child did not expect, so every import stopped at
// once (exit 2) and no recipe was ever used. Runs the real child on a temporary cache folder.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <setjmp.h>
#include "shader_recompiler/profile.h"
#include "video_core/renderer_vulkan/shader_recipe.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"

#undef NDEBUG
#include <cassert>

// The renderer's runtime hooks; there is no guest process here, so guest accesses abort. Types as
// gpu/shim/tlg_toggles.h declares them (sigjmp_buf comes from there on Windows).
extern "C" {
__thread sigjmp_buf* runtime_fault_recover;
std::uint64_t runtime_disabled_optimizations = 0;
std::uint64_t runtime_disabled_optimizations2 = 0;
void runtime_report(void) {}
uint64_t runtime_tsc_frequency() { return 1000000000; }
int runtime_file_translate(const char*, char*, size_t) { std::abort(); }
uint64_t runtime_memory_clamp(uintptr_t, uint64_t) { std::abort(); }
uint64_t runtime_process_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
int runtime_memory_region(uintptr_t, uintptr_t*, uintptr_t*, int*) { std::abort(); }
void runtime_memory_set_gpu_hooks(void (*)(uintptr_t, uint64_t), void (*)(uintptr_t, uint64_t),
                                  void (*)(uintptr_t, uint64_t)) { std::abort(); }
void runtime_thread_attach_host(const char*) {}
void runtime_memory_gpu_protect(uintptr_t, uint64_t, int, int) { std::abort(); }
void runtime_memory_describe(uintptr_t, char*, size_t) { std::abort(); }
void runtime_restart() { std::abort(); }
uint64_t runtime_process_time_counter() { return runtime_process_time_us() * 1000; }
int32_t* runtime_errno() { std::abort(); }
int runtime_memory_write_backing(uintptr_t, const void*, uint64_t) { std::abort(); }
int runtime_memory_read_backing(uintptr_t, void*, uint64_t) { std::abort(); }
}

namespace {
void Write(const std::filesystem::path& path, const void* data, size_t size) {
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    file.write(static_cast<const char*>(data), std::streamsize(size));
    assert(file);
}
} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    _set_error_mode(_OUT_TO_STDERR);
#endif
    using namespace Vulkan;
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.support_int64 = true;

    // The layout both sides use: header first, profile after it, nothing else.
    const auto header = EncodeCacheHeader(profile);
    assert(header.size() > sizeof(profile));
    const auto decoded = DecodeCacheHeader(header);
    assert(decoded && *decoded == profile);
    // A bare profile (the layout before v0.10.0), a short file or another build's header: none.
    std::vector<u8> bare(sizeof(profile));
    std::memcpy(bare.data(), &profile, sizeof(profile));
    assert(!DecodeCacheHeader(bare));
    assert(!DecodeCacheHeader(std::span{header}.first(header.size() - 1)));
    auto other = header;
    other[4] ^= 1; // a different shader binary version
    assert(!DecodeCacheHeader(other));

    // The child accepts the cache's own profile.bin and copies the recipe folder's keys.
    const std::filesystem::path root =
        std::filesystem::path{argc > 1 ? argv[1] : "."} / "recipe-import-test";
    std::filesystem::remove_all(root);
    const auto cache = root / "cache", recipes = root / "recipes";
    std::filesystem::create_directories(cache);
    std::filesystem::create_directories(recipes);
    const u8 key[4] = {1, 2, 3, 4};
    Write(recipes / "0x0000000000000001.key", key, sizeof(key));

    Write(cache / "profile.bin", header.data(), header.size());
    assert(ShaderRecipe::ImportChild(cache, recipes) == 0);
    assert(std::filesystem::exists(cache / "0x0000000000000001.key"));

    // A profile.bin the child cannot use stops it before it writes anything.
    std::filesystem::remove(cache / "0x0000000000000001.key");
    Write(cache / "profile.bin", bare.data(), bare.size());
    assert(ShaderRecipe::ImportChild(cache, recipes) == 2);
    assert(!std::filesystem::exists(cache / "0x0000000000000001.key"));

    std::filesystem::remove_all(root);
    std::puts("recipe import test passed");
    return 0;
}
