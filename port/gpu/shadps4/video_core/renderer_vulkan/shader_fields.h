// SPDX-FileCopyrightText: Copyright 2026 tlg_pc contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// tlg: field lists of the structs the shader recipes and the shader list keep (at the end of this
// file: Shader::RuntimeInfo, Shader::Backend::Bindings and the pipeline keys). One list per struct
// names every member once (TLG_FIELD); it is run by one of these visitors:
// - FieldWriter / FieldReader: the portable form. Every member as a number (LEB128, little end
//   first), bit fields too, in the order of the list. Any build reads it into its own structs,
//   whatever its compiler does with bit fields and padding (the shader list is made on Linux and
//   read on Windows, whose compiler lays some bit fields out otherwise).
// - FieldSchema: a hash of the member names in list order (PortableId): a list made with another
//   field list is not read.
// - FieldLayout: a hash of the bytes every member takes in this build (LayoutId): raw copies of
//   the structs (recorded recipes) are read only by a build that lays them out the same way.
#pragma once

#include <bit>
#include <cstring>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"

namespace Vulkan::ShaderFields {

/// FNV-1a.
struct Hash {
    u64 value = 0xcbf29ce484222325ull;
    void Byte(u8 byte) {
        value = (value ^ byte) * 0x100000001b3ull;
    }
    void Number(u64 number) {
        for (int i = 0; i < 8; ++i) {
            Byte(u8(number >> (8 * i)));
        }
    }
    void Text(std::string_view text) {
        Number(text.size());
        for (const char c : text) {
            Byte(u8(c));
        }
    }
};

/// The portable form of a struct.
struct FieldWriter {
    static constexpr bool AllVariants = false;
    std::vector<u8> out;
    template <typename Get, typename Set>
    void Field(std::string_view, Get get, Set) {
        for (u64 v = get();; v >>= 7) {
            if (v < 0x80) {
                out.push_back(u8(v));
                break;
            }
            out.push_back(u8(v | 0x80));
        }
    }
    /// A count (of a vector that follows): written as a field.
    template <typename Get, typename Set>
    void Count(std::string_view name, Get get, Set set) {
        Field(name, get, set);
    }
};

/// Reads the portable form into a struct. `ok` turns false when the input ends early or a
/// number does not fit its member (damage, or a list of another field list).
struct FieldReader {
    static constexpr bool AllVariants = false;
    std::span<const u8> in;
    size_t pos = 0;
    bool ok = true;
    template <typename Get, typename Set>
    void Field(std::string_view, Get, Set set) {
        u64 v = 0;
        for (int shift = 0;; shift += 7) {
            if (!ok || pos >= in.size() || shift > 63) {
                ok = false;
                return;
            }
            const u8 byte = in[pos++];
            v |= u64(byte & 0x7f) << shift;
            if (!(byte & 0x80)) {
                break;
            }
        }
        if (!set(v)) {
            ok = false;
        }
    }
    template <typename Get, typename Set>
    void Count(std::string_view name, Get get, Set set) {
        Field(name, get, set);
    }
    /// All of the input was read, and nothing failed.
    bool Done() const {
        return ok && pos == in.size();
    }
};

/// The field list itself (names in order, every variant of a union, one element of a vector).
struct FieldSchema {
    static constexpr bool AllVariants = true;
    Hash hash;
    template <typename Get, typename Set>
    void Field(std::string_view name, Get, Set) {
        hash.Text(name);
    }
    template <typename Get, typename Set>
    void Count(std::string_view name, Get, Set set) {
        hash.Text(name);
        set(1); // the element's fields are named once
    }
};

/// How this build lays a struct out: per member, the struct all zero with that member set to
/// all ones, every byte hashed (bit fields have no offsetof; this covers them and plain members
/// alike). `object` is the struct the field list runs over.
struct FieldLayout {
    static constexpr bool AllVariants = true;
    Hash hash;
    void* object;
    size_t size;
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
    template <typename Get, typename Set>
    void Count(std::string_view, Get, Set) {
        // Vectors are not raw-copied structs; their counts have no layout.
    }
};

} // namespace Vulkan::ShaderFields

/// One member in a field list: `io` a visitor above, `x` the member (also a bit field, an enum or
/// a bool). The text of `x` is its name in the schema, so a list names a member the same way in
/// every build.
#define TLG_FIELD(io, x)                                                                           \
    (io).Field(                                                                                    \
        #x, [&]() -> u64 { return static_cast<u64>(x); },                                          \
        [&](u64 v_) -> bool {                                                                      \
            using T_ = std::remove_cvref_t<decltype(x)>;                                           \
            const T_ t_ = static_cast<T_>(v_);                                                     \
            (x) = t_;                                                                              \
            return static_cast<u64>(t_) == v_ && static_cast<u64>(x) == v_;                        \
        })

/// A Vulkan flags member (vk::Flags has no cast to a number of its own).
#define TLG_FLAGS(io, x)                                                                           \
    (io).Field(                                                                                    \
        #x,                                                                                        \
        [&]() -> u64 {                                                                             \
            return u64(static_cast<typename std::remove_cvref_t<decltype(x)>::MaskType>(x));       \
        },                                                                                         \
        [&](u64 v_) -> bool {                                                                      \
            using M_ = typename std::remove_cvref_t<decltype(x)>::MaskType;                        \
            (x) = std::remove_cvref_t<decltype(x)>(M_(v_));                                        \
            return u64(M_(v_)) == v_;                                                              \
        })

/// A vector (std::vector or the static_vector of the vertex inputs): its count, checked against
/// its capacity when read.
#define TLG_COUNT(io, x)                                                                           \
    (io).Count(                                                                                    \
        #x, [&]() -> u64 { return (x).size(); },                                                   \
        [&](u64 v_) -> bool {                                                                      \
            if (v_ > (x).max_size() || v_ > 0x100000) {                                            \
                return false;                                                                      \
            }                                                                                      \
            (x).resize(size_t(v_));                                                                \
            return true;                                                                           \
        })

namespace Vulkan::ShaderFields {

// A member added to one of the structs below changes its size (unless it fills padding): the
// build stops here until the member is in its field list and the number is changed. The sizes
// are the same on Linux (GCC, Clang) and Windows (MinGW Clang).
static_assert(sizeof(Shader::RuntimeInfo) == 280);
static_assert(sizeof(Shader::RuntimeInfo::Properties) == 28);
static_assert(sizeof(Shader::SwVertexRuntimeInfo) == 16);
static_assert(sizeof(Shader::SwTessControlRuntimeInfo) == 28);
static_assert(sizeof(Shader::SwTessEvalRuntimeInfo) == 16);
static_assert(sizeof(Shader::HwLocalRuntimeInfo) == 4);
static_assert(sizeof(Shader::HwExportRuntimeInfo) == 4);
static_assert(sizeof(Shader::HwGeometryRuntimeInfo) == 80);
static_assert(sizeof(Shader::HwVertexRuntimeInfo) == 28);
static_assert(sizeof(Shader::HwFragmentRuntimeInfo) == 216);
static_assert(sizeof(Shader::HwFragmentRuntimeInfo::PsInput) == 4);
static_assert(sizeof(AmdGpu::PsInput) == 4);
static_assert(sizeof(Shader::PsColorBuffer) == 8);
static_assert(sizeof(AmdGpu::CompMapping) == 4);
static_assert(sizeof(Shader::HwComputeRuntimeInfo) == 20);
static_assert(sizeof(Shader::Backend::Bindings) == 12);
static_assert(sizeof(GraphicsPipelineKey) == 352);
static_assert(sizeof(AmdGpu::BlendControl) == 4);

/// The field list of Shader::RuntimeInfo: every member, once. Of its two unions
/// only the variants its stages use are listed (as RuntimeInfo::operator== compares them); the
/// schema and layout visitors see every variant. hw.gs.vs_copy (a pointer and length) is not
/// listed: recipes carry the words themselves. A member added to one of these structs is added
/// here too, or its value is lost in the portable form and its place in the layout id.
template <typename IO>
void RuntimeFields(IO& io, Shader::RuntimeInfo& r) {
    using namespace Shader;
    constexpr bool all = IO::AllVariants;
    TLG_FIELD(io, r.sw_stage);
    TLG_FIELD(io, r.hw_stage);
    TLG_FIELD(io, r.props.fp_denorm_mode32);
    TLG_FIELD(io, r.props.fp_denorm_mode16_64);
    TLG_FIELD(io, r.props.fp_round_mode32);
    TLG_FIELD(io, r.props.fp_round_mode16_64);
    TLG_FIELD(io, r.props.num_user_data);
    TLG_FIELD(io, r.props.num_input_vgprs);
    TLG_FIELD(io, r.props.num_allocated_vgprs);
    if (all || r.sw_stage == SwStage::Vertex) {
        auto& vs = r.sw.vs;
        TLG_FIELD(io, vs.step_rate_0);
        TLG_FIELD(io, vs.step_rate_1);
        TLG_FIELD(io, vs.vertex_sgpr_offset);
        TLG_FIELD(io, vs.instance_sgpr_offset);
        TLG_FIELD(io, vs.tess_emulated_primitive);
    }
    if (all || r.sw_stage == SwStage::TessellationControl) {
        auto& tcs = r.sw.tcs;
        TLG_FIELD(io, tcs.hs_output_cp_stride);
        TLG_FIELD(io, tcs.num_input_control_points);
        TLG_FIELD(io, tcs.num_threads);
        TLG_FIELD(io, tcs.tess_type);
        TLG_FIELD(io, tcs.offchip_lds_enable);
        TLG_FIELD(io, tcs.ls_stride);
        TLG_FIELD(io, tcs.hs_output_base);
    }
    if (all || r.sw_stage == SwStage::TessellationEval) {
        auto& tes = r.sw.tes;
        TLG_FIELD(io, tes.hs_output_cp_stride);
        TLG_FIELD(io, tes.tess_type);
        TLG_FIELD(io, tes.tess_topology);
        TLG_FIELD(io, tes.tess_partitioning);
    }
    if (all || r.hw_stage == HwStage::Local) {
        TLG_FIELD(io, r.hw.ls.ls_stride);
    }
    if (all || r.hw_stage == HwStage::Export) {
        TLG_FIELD(io, r.hw.es.vertex_data_size);
    }
    if (all || r.hw_stage == HwStage::Geometry) {
        auto& gs = r.hw.gs;
        TLG_FIELD(io, gs.num_outputs);
        for (size_t i = 0; i < gs.outputs.size(); ++i) {
            for (size_t j = 0; j < gs.outputs[i].size(); ++j) {
                TLG_FIELD(io, gs.outputs[i][j]);
            }
        }
        TLG_FIELD(io, gs.num_invocations);
        TLG_FIELD(io, gs.output_vertices);
        TLG_FIELD(io, gs.in_vertex_data_size);
        TLG_FIELD(io, gs.out_vertex_data_size);
        TLG_FIELD(io, gs.in_primitive);
        for (size_t i = 0; i < gs.out_primitive.size(); ++i) {
            TLG_FIELD(io, gs.out_primitive[i]);
        }
        TLG_FIELD(io, gs.mode);
        TLG_FIELD(io, gs.vs_copy_hash);
    }
    if (all || r.hw_stage == HwStage::Vertex) {
        auto& vs = r.hw.vs;
        TLG_FIELD(io, vs.num_outputs);
        for (size_t i = 0; i < vs.outputs.size(); ++i) {
            for (size_t j = 0; j < vs.outputs[i].size(); ++j) {
                TLG_FIELD(io, vs.outputs[i][j]);
            }
        }
        TLG_FIELD(io, vs.emulate_depth_negative_one_to_one);
        TLG_FIELD(io, vs.clip_disable);
        TLG_FIELD(io, vs.user_clip_plane_mask);
        TLG_FIELD(io, vs.motion_vectors);
    }
    if (all || r.hw_stage == HwStage::Fragment) {
        auto& fs = r.hw.fs;
        for (auto* flags : {&fs.en_flags, &fs.addr_flags}) {
            auto& f = *flags;
            TLG_FIELD(io, f.persp_sample_ena);
            TLG_FIELD(io, f.persp_center_ena);
            TLG_FIELD(io, f.persp_centroid_ena);
            TLG_FIELD(io, f.persp_pull_model_ena);
            TLG_FIELD(io, f.linear_sample_ena);
            TLG_FIELD(io, f.linear_center_ena);
            TLG_FIELD(io, f.linear_centroid_ena);
            TLG_FIELD(io, f.line_stipple_tex_ena);
            TLG_FIELD(io, f.pos_x_float_ena);
            TLG_FIELD(io, f.pos_y_float_ena);
            TLG_FIELD(io, f.pos_z_float_ena);
            TLG_FIELD(io, f.pos_w_float_ena);
            TLG_FIELD(io, f.front_face_ena);
            TLG_FIELD(io, f.ancillary_ena);
            TLG_FIELD(io, f.sample_coverage_ena);
            TLG_FIELD(io, f.pos_fixed_pt_ena);
        }
        TLG_FIELD(io, fs.num_inputs);
        for (size_t i = 0; i < fs.inputs.size(); ++i) {
            auto& in = fs.inputs[i];
            TLG_FIELD(io, in.param_index);
            TLG_FIELD(io, in.is_default);
            TLG_FIELD(io, in.is_flat);
            TLG_FIELD(io, in.default_value);
        }
        for (size_t i = 0; i < fs.color_buffers.size(); ++i) {
            auto& c = fs.color_buffers[i];
            TLG_FIELD(io, c.data_format);
            TLG_FIELD(io, c.num_format);
            TLG_FIELD(io, c.num_conversion);
            TLG_FIELD(io, c.export_format);
            TLG_FIELD(io, c.blend_self_scale);
            for (size_t j = 0; j < c.swizzle.array.size(); ++j) {
                TLG_FIELD(io, c.swizzle.array[j]);
            }
        }
        TLG_FIELD(io, fs.z_export_format);
        TLG_FIELD(io, fs.num_samples);
        TLG_FIELD(io, fs.mrtz_mask);
        TLG_FIELD(io, fs.front_face_all_bits);
        TLG_FIELD(io, fs.dual_source_blending);
        TLG_FIELD(io, fs.clip_distance_emulation);
        TLG_FIELD(io, fs.motion_vectors);
    }
    if (all || r.hw_stage == HwStage::Compute) {
        auto& cs = r.hw.cs;
        TLG_FIELD(io, cs.shared_memory_size);
        for (size_t i = 0; i < cs.workgroup_size.size(); ++i) {
            TLG_FIELD(io, cs.workgroup_size[i]);
        }
        for (size_t i = 0; i < cs.tgid_enable.size(); ++i) {
            TLG_FIELD(io, cs.tgid_enable[i]);
        }
    }
}

/// The field list of Shader::Backend::Bindings.
template <typename IO>
void BindingsFields(IO& io, Shader::Backend::Bindings& b) {
    TLG_FIELD(io, b.unified);
    TLG_FIELD(io, b.buffer);
    TLG_FIELD(io, b.user_data);
}

/// Every member of GraphicsPipelineKey (the pad fields are always zero, see the struct).
template <typename IO>
void KeyFields(IO& io, GraphicsPipelineKey& k) {
    for (size_t i = 0; i < k.stage_hashes.size(); ++i) {
        TLG_FIELD(io, k.stage_hashes[i]);
    }
    for (size_t i = 0; i < k.vertex_buffer_formats.size(); ++i) {
        TLG_FIELD(io, k.vertex_buffer_formats[i]);
    }
    TLG_FIELD(io, k.patch_control_points);
    TLG_FIELD(io, k.num_color_attachments);
    for (size_t i = 0; i < k.color_buffers.size(); ++i) {
        auto& c = k.color_buffers[i];
        TLG_FIELD(io, c.data_format);
        TLG_FIELD(io, c.num_format);
        TLG_FIELD(io, c.num_conversion);
        TLG_FIELD(io, c.export_format);
        TLG_FIELD(io, c.blend_self_scale);
        for (size_t j = 0; j < c.swizzle.array.size(); ++j) {
            TLG_FIELD(io, c.swizzle.array[j]);
        }
    }
    for (size_t i = 0; i < k.blend_controls.size(); ++i) {
        auto& b = k.blend_controls[i];
        TLG_FIELD(io, b.color_src_factor);
        TLG_FIELD(io, b.color_func);
        TLG_FIELD(io, b.color_dst_factor);
        TLG_FIELD(io, b.alpha_src_factor);
        TLG_FIELD(io, b.alpha_func);
        TLG_FIELD(io, b.alpha_dst_factor);
        TLG_FIELD(io, b.separate_alpha_blend);
        TLG_FIELD(io, b.enable);
        TLG_FIELD(io, b.disable_rop3);
    }
    for (size_t i = 0; i < k.write_masks.size(); ++i) {
        TLG_FLAGS(io, k.write_masks[i]);
    }
    TLG_FIELD(io, k.cb_shader_mask.raw);
    TLG_FIELD(io, k.logic_op);
    TLG_FIELD(io, k.num_samples);
    TLG_FIELD(io, k.depth_samples);
    for (size_t i = 0; i < k.color_samples.size(); ++i) {
        TLG_FIELD(io, k.color_samples[i]);
    }
    TLG_FIELD(io, k.mrt_mask);
    TLG_FIELD(io, k.motion_vectors);
    TLG_FIELD(io, k.z_format);
    TLG_FIELD(io, k.stencil_format);
    TLG_FIELD(io, k.depth_clamp_enable);
    TLG_FIELD(io, k.prim_type);
    TLG_FIELD(io, k.polygon_mode);
    TLG_FIELD(io, k.clip_space);
    TLG_FIELD(io, k.provoking_vtx_last);
    TLG_FIELD(io, k.depth_clip_enable);
}

/// Every member SerializationSupport::Serialize writes. The multisample state's pointers (pNext,
/// pSampleMask) are not stored: PortableKey refuses a key that has them.
template <typename IO>
void SupportFields(IO& io, GraphicsPipeline::SerializationSupport& s) {
    TLG_COUNT(io, s.vertex_attributes);
    for (size_t i = 0; i < s.vertex_attributes.size(); ++i) {
        auto& a = s.vertex_attributes[i];
        TLG_FIELD(io, a.location);
        TLG_FIELD(io, a.binding);
        TLG_FIELD(io, a.format);
        TLG_FIELD(io, a.offset);
    }
    TLG_COUNT(io, s.vertex_bindings);
    for (size_t i = 0; i < s.vertex_bindings.size(); ++i) {
        auto& b = s.vertex_bindings[i];
        TLG_FIELD(io, b.binding);
        TLG_FIELD(io, b.stride);
        TLG_FIELD(io, b.inputRate);
    }
    TLG_COUNT(io, s.divisors);
    for (size_t i = 0; i < s.divisors.size(); ++i) {
        auto& d = s.divisors[i];
        TLG_FIELD(io, d.binding);
        TLG_FIELD(io, d.divisor);
    }
    auto& m = s.multisampling;
    TLG_FLAGS(io, m.flags);
    TLG_FIELD(io, m.rasterizationSamples);
    TLG_FIELD(io, m.sampleShadingEnable);
    io.Field(
        "m.minSampleShading", [&]() -> u64 { return std::bit_cast<u32>(m.minSampleShading); },
        [&](u64 v) {
            m.minSampleShading = std::bit_cast<float>(u32(v));
            return v <= 0xffffffffu;
        });
    TLG_FIELD(io, m.alphaToCoverageEnable);
    TLG_FIELD(io, m.alphaToOneEnable);
    TLG_COUNT(io, s.tcs);
    for (size_t i = 0; i < s.tcs.size(); ++i) {
        TLG_FIELD(io, s.tcs[i]);
    }
    TLG_COUNT(io, s.tes);
    for (size_t i = 0; i < s.tes.size(); ++i) {
        TLG_FIELD(io, s.tes[i]);
    }
}

/// The kinds of pipeline key (the u32 after the version in a key file).
constexpr u32 KindGraphics = 0, KindCompute = 1;

/// The fields of a portable key: the key version and kind, then the key's own fields.
template <typename IO>
void PortableKeyFields(IO& io, u32& version, u32& kind, GraphicsPipelineKey& graphics,
                       GraphicsPipeline::SerializationSupport& support, u64& compute) {
    TLG_FIELD(io, version);
    TLG_FIELD(io, kind);
    if (IO::AllVariants || kind == KindGraphics) {
        ShaderFields::KeyFields(io, graphics);
        ShaderFields::SupportFields(io, support);
    }
    if (IO::AllVariants || kind == KindCompute) {
        TLG_FIELD(io, compute);
    }
}

/// A RuntimeInfo as Initialize() leaves it before the stages are known: all zero.
inline Shader::RuntimeInfo ZeroRuntime() {
    Shader::RuntimeInfo runtime{};
    std::memset(static_cast<void*>(&runtime), 0, sizeof(runtime));
    return runtime;
}

/// LayoutId(): the bytes every member of the two field lists takes in this build, and the
/// structs' sizes and alignments.
inline u64 ComputeLayoutId() {
    FieldLayout layout{};
    layout.hash.Number(sizeof(Shader::RuntimeInfo));
    layout.hash.Number(alignof(Shader::RuntimeInfo));
    layout.hash.Number(sizeof(Shader::Backend::Bindings));
    layout.hash.Number(alignof(Shader::Backend::Bindings));
    auto runtime = ZeroRuntime();
    layout.object = &runtime;
    layout.size = sizeof(runtime);
    RuntimeFields(layout, runtime);
    Shader::Backend::Bindings bindings{};
    layout.object = &bindings;
    layout.size = sizeof(bindings);
    BindingsFields(layout, bindings);
    return layout.hash.value;
}

/// The field lists of the portable form (PortableId()).
inline u64 ComputePortableId() {
    Hash hash;
    hash.Text("tlg shader list portable form 1");
    FieldSchema schema;
    auto runtime = ZeroRuntime();
    RuntimeFields(schema, runtime);
    Shader::Backend::Bindings bindings{};
    BindingsFields(schema, bindings);
    hash.Number(schema.hash.value);
    FieldSchema keys;
    u32 version{}, kind{};
    GraphicsPipelineKey graphics;
    GraphicsPipeline::SerializationSupport support;
    u64 compute{};
    PortableKeyFields(keys, version, kind, graphics, support, compute);
    hash.Number(keys.hash.value);
    return hash.value;
}

} // namespace Vulkan::ShaderFields
