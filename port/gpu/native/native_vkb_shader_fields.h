// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
// TLG_VKB_SHADER_RECIPES: the field lists of the two structs a new-engine shader recipe keeps as raw bytes (VkbShader::RuntimeInfo,
// VkbShader::Backend::Bindings), and the layout id made from them. A recipe outlives the build that recorded it (the shader store is
// thrown away with every build; the recipes fill it again), so a recipe is read only by a build that lays those two structs out the same
// way: equal struct sizes do not prove an equal layout. The normal renderer's way (port/gpu/shadps4/video_core/renderer_vulkan/
// shader_fields.h, its LayoutId), for the new engine's own copies of the structs, which differ from the normal renderer's (bit fields in
// the fragment info, Bindings::indexed, dx10_clamp, is_indirect).
// The shader list (native_vkb_shader_list.h) is the normal renderer's: its conversion sets this field list's members by name from the
// normal renderer's field list (RuntimeFields / BindingsFields with a visitor of its own).

#include <cstring>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "common/types.h"
#include "native/shadps4/shader_recompiler/backend/bindings.h"
#include "native/shadps4/shader_recompiler/runtime_info.h"

namespace Core::Vkb::ShaderFields {

/// FNV-1a
struct Hash {
    u64 value = 0xcbf29ce484222325ull;
    void Byte(u8 byte) {
        value = (value ^ byte) * 0x100000001b3ull;
    }
    void Number(u64 number) {
        for (int i = 0; i < 8; ++i) {
            Byte(static_cast<u8>(number >> (8 * i)));
        }
    }
    void Text(std::string_view text) {
        Number(text.size());
        for (const char c : text) {
            Byte(static_cast<u8>(c));
        }
    }
};

/// How this build lays a struct out: per member, the struct all zero with that member set to all ones, every byte hashed (bit fields
/// have no offsetof; this covers them and plain members alike). `object` is the struct the field list runs over.
struct FieldLayout {
    static constexpr bool AllVariants = true;
    Hash hash;
    void* object = nullptr;
    size_t size = 0;
    template <typename Get, typename Set>
    void Field(std::string_view, Get, Set set) {
        std::memset(object, 0, size);
        set(~u64{0});
        const auto* bytes = static_cast<const u8*>(object);
        for (size_t i = 0; i < size; ++i) {
            hash.Byte(bytes[i]);
        }
        std::memset(object, 0, size);
    }
};

} // namespace Core::Vkb::ShaderFields

/// One member in a field list (also a bit field, an enum or a bool). The setter returns false when the number does not fit the member.
#define TLG_VKB_FIELD(io, x)                                                                                                              \
    (io).Field(                                                                                                                            \
        #x, [&]() -> u64 { return static_cast<u64>(x); },                                                                                  \
        [&](u64 v_) -> bool {                                                                                                              \
            using T_ = std::remove_cvref_t<decltype(x)>;                                                                                   \
            const T_ t_ = static_cast<T_>(v_);                                                                                             \
            (x) = t_;                                                                                                                      \
            return static_cast<u64>(t_) == v_ && static_cast<u64>(x) == v_;                                                                \
        })

namespace Core::Vkb::ShaderFields {

// A member added to one of these structs changes its size (unless it fills padding): the build stops here until the member is in its
// field list below and the number is changed. Sizes as GCC lays them out on Linux; a Windows build that stops here has other sizes.
static_assert(sizeof(VkbShader::RuntimeInfo) == 184);
static_assert(sizeof(VkbShader::RuntimeInfo::Properties) == 4);
static_assert(sizeof(VkbShader::SwVertexRuntimeInfo) == 16);
static_assert(sizeof(VkbShader::SwTessControlRuntimeInfo) == 28);
static_assert(sizeof(VkbShader::SwTessEvalRuntimeInfo) == 16);
static_assert(sizeof(VkbShader::HwLocalRuntimeInfo) == 4);
static_assert(sizeof(VkbShader::HwExportRuntimeInfo) == 4);
static_assert(sizeof(VkbShader::HwGeometryRuntimeInfo) == 80);
static_assert(sizeof(VkbShader::HwVertexRuntimeInfo) == 28);
static_assert(sizeof(VkbShader::HwFragmentRuntimeInfo) == 144);
static_assert(sizeof(VkbShader::HwFragmentRuntimeInfo::PsInput) == 2);
static_assert(sizeof(VkbAmdGpu::PsInput) == 4);
static_assert(sizeof(VkbShader::PsColorBuffer) == 8);
static_assert(sizeof(VkbShader::HwComputeRuntimeInfo) == 20);
static_assert(sizeof(VkbShader::Backend::Bindings) == 12);

/// The field list of VkbShader::RuntimeInfo: every member, once. Of its two unions the layout and schema visitors see every variant, the
/// other visitors (the list conversion) only the one its stage uses (as RuntimeInfo::operator== compares them; the stages come first, so a visitor knows them
/// when it reaches the unions). hw.gs.vs_copy (a pointer and a length) is not listed: a geometry stage has no recipe (the new engine has
/// no geometry stages; the store never keeps one).
template <typename IO>
void RuntimeFields(IO& io, VkbShader::RuntimeInfo& r) {
    using VkbShader::HwStage;
    using VkbShader::SwStage;
    constexpr bool all = IO::AllVariants;
    TLG_VKB_FIELD(io, r.sw_stage);
    TLG_VKB_FIELD(io, r.hw_stage);
    TLG_VKB_FIELD(io, r.props.fp_denorm_mode32);
    TLG_VKB_FIELD(io, r.props.fp_denorm_mode16_64);
    TLG_VKB_FIELD(io, r.props.fp_round_mode32);
    TLG_VKB_FIELD(io, r.props.fp_round_mode16_64);
    TLG_VKB_FIELD(io, r.props.num_allocated_vgprs);
    TLG_VKB_FIELD(io, r.props.num_user_data);
    TLG_VKB_FIELD(io, r.props.num_input_vgprs);
    TLG_VKB_FIELD(io, r.props.dx10_clamp);
    TLG_VKB_FIELD(io, r.props.reserved);
    if (all || r.sw_stage == SwStage::Vertex) {
        auto& vs = r.sw.vs;
        TLG_VKB_FIELD(io, vs.step_rate_0);
        TLG_VKB_FIELD(io, vs.step_rate_1);
        TLG_VKB_FIELD(io, vs.vertex_sgpr_offset);
        TLG_VKB_FIELD(io, vs.instance_sgpr_offset);
        TLG_VKB_FIELD(io, vs.is_indirect);
        TLG_VKB_FIELD(io, vs.tess_emulated_primitive);
    }
    if (all || r.sw_stage == SwStage::TessellationControl) {
        auto& tcs = r.sw.tcs;
        TLG_VKB_FIELD(io, tcs.hs_output_cp_stride);
        TLG_VKB_FIELD(io, tcs.num_input_control_points);
        TLG_VKB_FIELD(io, tcs.num_threads);
        TLG_VKB_FIELD(io, tcs.tess_type);
        TLG_VKB_FIELD(io, tcs.offchip_lds_enable);
        TLG_VKB_FIELD(io, tcs.ls_stride);
        TLG_VKB_FIELD(io, tcs.hs_output_base);
    }
    if (all || r.sw_stage == SwStage::TessellationEval) {
        auto& tes = r.sw.tes;
        TLG_VKB_FIELD(io, tes.hs_output_cp_stride);
        TLG_VKB_FIELD(io, tes.tess_type);
        TLG_VKB_FIELD(io, tes.tess_topology);
        TLG_VKB_FIELD(io, tes.tess_partitioning);
    }
    if (all || r.hw_stage == HwStage::Local) {
        TLG_VKB_FIELD(io, r.hw.ls.ls_stride);
    }
    if (all || r.hw_stage == HwStage::Export) {
        TLG_VKB_FIELD(io, r.hw.es.vertex_data_size);
    }
    if (all || r.hw_stage == HwStage::Geometry) {
        auto& gs = r.hw.gs;
        TLG_VKB_FIELD(io, gs.num_outputs);
        for (size_t i = 0; i < gs.outputs.size(); ++i) {
            for (size_t j = 0; j < gs.outputs[i].size(); ++j) {
                TLG_VKB_FIELD(io, gs.outputs[i][j]);
            }
        }
        TLG_VKB_FIELD(io, gs.num_invocations);
        TLG_VKB_FIELD(io, gs.output_vertices);
        TLG_VKB_FIELD(io, gs.in_vertex_data_size);
        TLG_VKB_FIELD(io, gs.out_vertex_data_size);
        TLG_VKB_FIELD(io, gs.in_primitive);
        for (size_t i = 0; i < gs.out_primitive.size(); ++i) {
            TLG_VKB_FIELD(io, gs.out_primitive[i]);
        }
        TLG_VKB_FIELD(io, gs.mode);
        TLG_VKB_FIELD(io, gs.vs_copy_hash);
    }
    if (all || r.hw_stage == HwStage::Vertex) {
        auto& vs = r.hw.vs;
        TLG_VKB_FIELD(io, vs.num_outputs);
        for (size_t i = 0; i < vs.outputs.size(); ++i) {
            for (size_t j = 0; j < vs.outputs[i].size(); ++j) {
                TLG_VKB_FIELD(io, vs.outputs[i][j]);
            }
        }
        TLG_VKB_FIELD(io, vs.emulate_depth_negative_one_to_one);
        TLG_VKB_FIELD(io, vs.clip_disable);
        TLG_VKB_FIELD(io, vs.user_clip_plane_mask);
        TLG_VKB_FIELD(io, vs.motion_vectors);
    }
    if (all || r.hw_stage == HwStage::Fragment) {
        auto& fs = r.hw.fs;
        for (auto* flags : {&fs.en_flags, &fs.addr_flags}) {
            auto& f = *flags;
            TLG_VKB_FIELD(io, f.persp_sample_ena);
            TLG_VKB_FIELD(io, f.persp_center_ena);
            TLG_VKB_FIELD(io, f.persp_centroid_ena);
            TLG_VKB_FIELD(io, f.persp_pull_model_ena);
            TLG_VKB_FIELD(io, f.linear_sample_ena);
            TLG_VKB_FIELD(io, f.linear_center_ena);
            TLG_VKB_FIELD(io, f.linear_centroid_ena);
            TLG_VKB_FIELD(io, f.line_stipple_tex_ena);
            TLG_VKB_FIELD(io, f.pos_x_float_ena);
            TLG_VKB_FIELD(io, f.pos_y_float_ena);
            TLG_VKB_FIELD(io, f.pos_z_float_ena);
            TLG_VKB_FIELD(io, f.pos_w_float_ena);
            TLG_VKB_FIELD(io, f.front_face_ena);
            TLG_VKB_FIELD(io, f.ancillary_ena);
            TLG_VKB_FIELD(io, f.sample_coverage_ena);
            TLG_VKB_FIELD(io, f.pos_fixed_pt_ena);
        }
        TLG_VKB_FIELD(io, fs.num_inputs);
        for (size_t i = 0; i < fs.inputs.size(); ++i) {
            auto& in = fs.inputs[i];
            TLG_VKB_FIELD(io, in.param_index);
            TLG_VKB_FIELD(io, in.is_default);
            TLG_VKB_FIELD(io, in.is_flat);
            TLG_VKB_FIELD(io, in.default_value);
        }
        for (size_t i = 0; i < fs.color_buffers.size(); ++i) {
            auto& c = fs.color_buffers[i];
            TLG_VKB_FIELD(io, c.data_format);
            TLG_VKB_FIELD(io, c.num_format);
            TLG_VKB_FIELD(io, c.num_conversion);
            TLG_VKB_FIELD(io, c.export_format);
            TLG_VKB_FIELD(io, c.blend_self_scale);
            for (size_t j = 0; j < c.swizzle.array.size(); ++j) {
                TLG_VKB_FIELD(io, c.swizzle.array[j]);
            }
        }
        TLG_VKB_FIELD(io, fs.z_export_format);
        TLG_VKB_FIELD(io, fs.num_samples);
        TLG_VKB_FIELD(io, fs.mrtz_mask);
        TLG_VKB_FIELD(io, fs.front_face_all_bits);
        TLG_VKB_FIELD(io, fs.dual_source_blending);
        TLG_VKB_FIELD(io, fs.clip_distance_emulation);
        TLG_VKB_FIELD(io, fs.depth_before_shader);
        TLG_VKB_FIELD(io, fs.motion_vectors);
    }
    if (all || r.hw_stage == HwStage::Compute) {
        auto& cs = r.hw.cs;
        TLG_VKB_FIELD(io, cs.shared_memory_size);
        for (size_t i = 0; i < cs.workgroup_size.size(); ++i) {
            TLG_VKB_FIELD(io, cs.workgroup_size[i]);
        }
        for (size_t i = 0; i < cs.tgid_enable.size(); ++i) {
            TLG_VKB_FIELD(io, cs.tgid_enable[i]);
        }
    }
}

/// The field list of VkbShader::Backend::Bindings
template <typename IO>
void BindingsFields(IO& io, VkbShader::Backend::Bindings& b) {
    TLG_VKB_FIELD(io, b.unified);
    TLG_VKB_FIELD(io, b.buffer);
    TLG_VKB_FIELD(io, b.indexed);
}

/// The layout id: the bytes every member of the two field lists takes in this build, and the structs' sizes and alignments.
inline u64 LayoutId() {
    static const u64 id = [] {
        FieldLayout layout{};
        layout.hash.Number(sizeof(VkbShader::RuntimeInfo));
        layout.hash.Number(alignof(VkbShader::RuntimeInfo));
        layout.hash.Number(sizeof(VkbShader::Backend::Bindings));
        layout.hash.Number(alignof(VkbShader::Backend::Bindings));
        VkbShader::RuntimeInfo runtime{};
        std::memset(static_cast<void*>(&runtime), 0, sizeof(runtime));
        layout.object = &runtime;
        layout.size = sizeof(runtime);
        RuntimeFields(layout, runtime);
        VkbShader::Backend::Bindings bindings{};
        layout.object = &bindings;
        layout.size = sizeof(bindings);
        BindingsFields(layout, bindings);
        return layout.hash.value;
    }();
    return id;
}

/// A RuntimeInfo as the engine starts one: all zero (padding too)
inline VkbShader::RuntimeInfo ZeroRuntime() {
    VkbShader::RuntimeInfo runtime{};
    std::memset(static_cast<void*>(&runtime), 0, sizeof(runtime));
    return runtime;
}

} // namespace Core::Vkb::ShaderFields
