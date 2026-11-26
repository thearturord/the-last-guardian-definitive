// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
// The new engine's pipeline warm-up from the shipped shader list (owner decision 2026-11-26: "it has to do the same as the normal
// renderer"; way A: the normal renderer's pipeline keys in shader-list/shaders.tlgsl are converted, no list of our own).
//
// Step 1 (this file): the check. Each pipeline key of the list is converted into the new engine's key (and the depth format its
// pipeline is made for); each pipeline the new engine makes while playing is looked up among them and counted as a hit or a miss.
// For a miss the closest converted key with the same two stages names the members that differ. Nothing is built from the list yet.
//
// Per member of the new engine's key (native_vkb_pipeline.h), from the normal key (video_core/renderer_vulkan/vk_graphics_pipeline.h):
// - copied: color targets (formats, export format, swizzle), write masks, samples, mrt mask, attachment count, depth clamp and clip,
//   primitive type (same numbers), depth and stencil format;
// - converted: stage hashes (the normal key holds HashCombine(program hash, permutation): the list's entries give the program),
//   blend (enabled targets only; the normal key's alpha factors are the colour factors unless separate_alpha_blend; the factors and ops
//   through LiverpoolToVK as the normal pipeline does), logic op (on when rop3 is not Copy), polygon mode, clip space (-w..w),
//   provoking vertex, alpha_not_exported (cb_shader_mask without A), blend_self_scale (the new engine's own rule, native_vkb_graphics.cpp);
// - zero: vertex_buffer_formats (the new engine needs dynamic vertex input), patch_control_points (no tessellation);
// - depth format: LiverpoolToVK::DepthFormat of the depth and stencil format; without D32S8 the stencil format is Invalid
//   (our depth image has no stencil, as in Compile);
// - left out (counted): keys with motion vectors, tessellation or geometry stages, stages the list's entries do not name.

#include <array>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json_fwd.hpp>

#include "common/types.h"

typedef struct VkPipelineCache_T* VkPipelineCache;

namespace Vulkan::ShaderList {
struct List;
}
namespace VkbVulkan::Vkb {
struct GraphicsPipelineKey;
}
namespace VkbShader {
struct RuntimeInfo;
namespace Backend {
struct Bindings;
}
} // namespace VkbShader

namespace Core::Vkb::PipelineWarm {

/// One graphics pipeline key of the normal renderer's list in plain numbers (read by native_vkb_pipeline_list.cpp, which alone sees the
/// normal renderer's structs)
struct ListKey {
    u64 vs = 0, ps = 0; ///< program hashes of the vertex and pixel stage (0: none)
    s64 vs_entry = -1, ps_entry = -1; ///< the list entries of those stages (-1: none)
    u32 z_format = 0, stencil_format = 0, depth_clamp = 0, depth_clip = 0, clip_minus_w = 0, provoking_last = 0;
    u32 prim_type = 0, polygon_mode = 0, rop3 = 0, depth_samples = 0, num_samples = 0, cb_shader_mask = 0, mrt_mask = 0, num_color_attachments = 0;
    struct Color {
        u32 data_format = 0, num_format = 0, num_conversion = 0, export_format = 0;
        std::array<u8, 4> swizzle{};
        u32 blend = 0; ///< BlendControl as its register word
        u32 write_mask = 0, samples = 0;
    };
    std::array<Color, 8> color{};
};

/// The list's graphics keys; `left` counts the keys left out, by reason
std::vector<ListKey> ReadListKeys(const Vulkan::ShaderList::List& list, std::map<std::string, u64>& left);

/// Converts the list's keys (the shader pre-build calls it once it has read the list)
void NoteList(const Vulkan::ShaderList::List& list);

/// The inputs one stage of a pipeline was translated from (the store's inputs; null: no such stage)
struct StageInputs {
    const VkbShader::RuntimeInfo* runtime = nullptr;
    const VkbShader::Backend::Bindings* start = nullptr;
};

/// A pipeline the new engine is about to make: counted as a hit or a miss (Compile, native_vkb_graphics.cpp). A miss whose stages are both
/// list entries is logged once as a key of the list's other kind (Vulkan::ShaderList::OtherKeyKind), so players' logs alone add it:
///   Pipeline key (not in list): v_0x<hash>.vkey <base64 of the zlib-packed key>
/// The key holds the pipeline state field by field and each stage as its list entry (program hash and permutation), no game code; it is
/// logged only when it reads back into the same key. The next list (port/tools/shader-list/add_from_logs.py) carries it, and the warm-up
/// builds it like the converted keys.
void Check(const VkbVulkan::Vkb::GraphicsPipelineKey& key, u32 depth_format, const StageInputs& vs, const StageInputs& ps);

/// Step 2, the warm-up: once the shader pre-build is done (or found its build from before), a background thread makes each converted
/// pipeline once from the stored stages, through the shared pipeline cache below, and drops it again: the driver keeps the compiled
/// pipeline, so the game's own make of the same pipeline is a cache hit. "Building shaders NN%" shows its progress. TLG_VKB_PIPELINE_WARM=0: off.
void Start();

/// The pipeline cache the warm-up and the game's pipelines share (made on first use; the device must be ready)
[[nodiscard]] VkPipelineCache Cache();

[[nodiscard]] nlohmann::json Report();

} // namespace Core::Vkb::PipelineWarm
