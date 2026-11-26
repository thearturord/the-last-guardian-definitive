// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <string>

#include "shader_recompiler/frontend/fetch_shader.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace PipelineBinaryCache {
using Key = std::vector<u8>;
struct Binary {
    Key key;
    std::vector<u8> data;
};
constexpr size_t MaxFileSize = 64 * 1024 * 1024;
constexpr u32 MaxBinaries = 32;

std::vector<u8> Encode(const Key& global, const Key& pipeline, u64 graphics_hash,
                       const std::vector<Binary>& binaries);
std::optional<std::vector<Binary>> Decode(std::span<const u8> bytes, const Key& global,
                                          const Key& pipeline, u64 graphics_hash);
bool Save(const std::filesystem::path& file, std::span<const u8> bytes);
std::optional<std::vector<Binary>> Load(const std::filesystem::path& file, const Key& global,
                                        const Key& pipeline, u64 graphics_hash);
} // namespace PipelineBinaryCache

vk::ResultValue<vk::UniquePipeline> CreateCachedGraphicsPipeline(
    vk::Device device, vk::PipelineCache cache, const vk::GraphicsPipelineCreateInfo& info,
    const GraphicsPipelineKey& key, const vk::PipelineBinaryKeyKHR& global_key,
    const std::filesystem::path& directory);
void ReportPipelineBinaryCache();

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata);
void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata);
void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash, size_t perm_idx);
void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx);
/// tlg: profile.bin as the cache stores it: the version header, then the device profile.
std::vector<u8> EncodeCacheHeader(const Shader::Profile& profile);
/// tlg: the profile of a profile.bin from this build; empty for another build's or a damaged one.
/// The shader recipe import reads it too, so both always agree on the layout.
std::optional<Shader::Profile> DecodeCacheHeader(std::span<const u8> data);
/// tlg: shader list: a pipeline key file of this build (as RegisterPipelineData writes it) in the
/// portable form (shader_fields.h), which any build reads; nullopt with the reason in `why` when
/// it cannot be stored so (damaged, another key version, or it does not come back the same).
std::optional<std::vector<u8>> PortableKey(std::span<const u8> key_file, std::string& why);
/// The key file of this build from its portable form, and its file name as the game names it
/// (the graphics key's hash in this build); nullopt with the reason when it cannot be read.
std::optional<std::vector<u8>> KeyFromPortable(std::span<const u8> portable, std::string& name,
                                               std::string& why);
/// tlg: a .meta blob as RegisterShaderMeta stores it (shader recipe import writes it too).
std::vector<u8> EncodeShaderMeta(const Shader::Info& info, const Shader::StageSpecialization& spec,
                                 size_t perm_hash, size_t perm_idx);

} // namespace Vulkan
