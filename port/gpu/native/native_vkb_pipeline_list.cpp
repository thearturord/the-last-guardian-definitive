// SPDX-License-Identifier: GPL-2.0-or-later
// The normal renderer's pipeline keys of the shader list in plain numbers (native_vkb_pipeline_warm.h). Only this file sees the normal
// renderer's key structs; port/gpu/shadps4 is not changed, its headers are only read.

#include <cstring>
#include <unordered_map>

#include "common/hash.h"
#include "common/serdes.h"
#include "native/native_vkb_pipeline_warm.h"
#include "video_core/renderer_vulkan/shader_fields.h"
#include "video_core/renderer_vulkan/shader_list.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"

namespace Core::Vkb::PipelineWarm {

static_assert(sizeof(AmdGpu::BlendControl) == 4);

std::vector<ListKey> ReadListKeys(const Vulkan::ShaderList::List& list, std::map<std::string, u64>& left) {
    using Shader::SwStage;
    // the normal key's stage hash is HashCombine(program hash, permutation) (vk_pipeline_cache.cpp, GetProgram)
    std::unordered_map<u64, size_t> entry_of;
    for (size_t i = 0; i < list.entries.size(); ++i) {
        const auto& e = list.entries[i];
        entry_of.emplace(HashCombine(e.pgm_hash, e.perm_idx), i);
    }
    std::vector<ListKey> out;
    for (const auto& [name, bytes] : list.keys) {
        Serialization::Archive ar{std::vector<u8>{bytes.begin(), bytes.end()}};
        u32 version = 0, kind = 0;
        Vulkan::GraphicsPipelineKey key{};
        try {
            Serialization::Reader r{ar};
            r.Read(version);
            r.Read(kind);
            if (kind != Vulkan::ShaderFields::KindGraphics) {
                continue; // compute: the new engine's compute pipelines are not keyed this way
            }
            key.Deserialize(ar);
        } catch (const Serialization::CorruptData&) {
            ++left["damaged"];
            continue;
        }
        if (key.motion_vectors != 0) {
            ++left["motion vectors (the new engine has no such attachment)"];
            continue;
        }
        bool other_stage = false;
        for (u32 s = 0; s < key.stage_hashes.size(); ++s) {
            if (s != static_cast<u32>(SwStage::Vertex) && s != static_cast<u32>(SwStage::Fragment) && key.stage_hashes[s] != 0) {
                other_stage = true;
            }
        }
        if (other_stage || key.patch_control_points != 0) {
            ++left["tessellation or geometry stage"];
            continue;
        }
        ListKey k;
        const auto stage = [&](SwStage s, u64& program, s64& entry) {
            const u64 h = key.stage_hashes[static_cast<u32>(s)];
            if (h == 0) {
                return true;
            }
            const auto it = entry_of.find(h);
            if (it == entry_of.end()) {
                return false;
            }
            program = list.entries[it->second].pgm_hash;
            entry = static_cast<s64>(it->second);
            return true;
        };
        if (!stage(SwStage::Vertex, k.vs, k.vs_entry) || !stage(SwStage::Fragment, k.ps, k.ps_entry) || k.vs == 0) {
            ++left["a stage no entry of the list names"];
            continue;
        }
        k.z_format = static_cast<u32>(key.z_format);
        k.stencil_format = static_cast<u32>(key.stencil_format);
        k.depth_clamp = key.depth_clamp_enable;
        k.depth_clip = key.depth_clip_enable;
        k.clip_minus_w = key.clip_space == AmdGpu::ClipSpace::MinusWToW;
        k.provoking_last = key.provoking_vtx_last == AmdGpu::ProvokingVtxLast::Last;
        k.prim_type = static_cast<u32>(key.prim_type);
        k.polygon_mode = static_cast<u32>(key.polygon_mode);
        k.rop3 = static_cast<u32>(key.logic_op);
        k.depth_samples = key.depth_samples;
        k.num_samples = key.num_samples;
        k.cb_shader_mask = key.cb_shader_mask.raw;
        k.mrt_mask = key.mrt_mask;
        k.num_color_attachments = key.num_color_attachments;
        for (u32 cb = 0; cb < 8; ++cb) {
            const auto& c = key.color_buffers[cb];
            auto& o = k.color[cb];
            o.data_format = static_cast<u32>(c.data_format);
            o.num_format = static_cast<u32>(c.num_format);
            o.num_conversion = static_cast<u32>(c.num_conversion);
            o.export_format = static_cast<u32>(c.export_format);
            for (u32 i = 0; i < 4; ++i) {
                o.swizzle[i] = static_cast<u8>(c.swizzle.array[i]);
            }
            std::memcpy(&o.blend, &key.blend_controls[cb], sizeof(o.blend));
            o.write_mask = static_cast<u32>(static_cast<VkColorComponentFlags>(key.write_masks[cb]));
            o.samples = key.color_samples[cb];
        }
        out.push_back(k);
    }
    return out;
}

} // namespace Core::Vkb::PipelineWarm
