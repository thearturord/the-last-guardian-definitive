// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later
// VKB: the backend's recompiler profile, from its own device (merge note 5: the backend has its own instance; this is the old port's pipeline-cache profile
// without its two old-path switches, indexed_resources and constant_offsets, since the backend binds every resource in its own set).
#pragma once

#include <algorithm>
#include <cstdlib>

#include "native/shadps4/shader_recompiler/profile.h"
#include "tlg_settings.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_instance.h"

namespace VideoCore::Vkb {

inline VkbShader::Profile ProfileFor(const VkbVulkan::Instance& instance) {
    const auto& vk12_props = instance.GetVk12Properties();
    const auto driver = instance.GetDriverID();
    VkbShader::Profile profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = 0x00010600U,
        .subgroup_size = instance.SubgroupSize(),
        .sparse_page_shift = 14, // the raw address access's block size (that access is off)
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence = vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence = vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve = bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve = bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve = bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max = instance.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout = instance.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter = instance.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance.IsFragmentShaderBarycentricSupported(),
        .supports_shader_subgroup_clock = instance.IsShaderSubgroupClockSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() && driver == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = driver == vk::DriverId::eNvidiaProprietary || driver == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = driver == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = driver == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance.IsShaderStencilExportSupported(),
        // issue 75: reads with automatic level choice out of per-pixel branches; TLG_VKB_READS_OUT_OF_BRANCH=0 the old translation. Plain SPIR-V, no driver
        // feature. Read once at start; there is no translated-shader cache on disk, so nothing translated the other way is reused.
        .reads_out_of_branch = [] { const char* e = std::getenv("TLG_VKB_READS_OUT_OF_BRANCH"); return e == nullptr || e[0] != '0'; }(),
    };
    // look option (Extras "Realistic tree shade", TLG_LEAF_AO), see leaf_ambient_occlusion_pass.cpp: the material entries are part of the translation,
    // strength and radius are read live per dispatch (native_vkb_compute.cpp). Plain SPIR-V, no driver feature.
    profile.leaf_ao_materials = TlgSettings::LeafAoMaterials();
    return profile;
}

} // namespace VideoCore::Vkb
