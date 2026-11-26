// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "native/shadps4/common/small_vector.h"
#include "common/types.h"
#include "native/shadps4/shader_recompiler/info.h"

namespace Serialization {
struct Archive;
}

namespace VkbShader::Gcn {

struct VertexAttribute {
    enum InstanceIdType : u8 {
        None = 0,
        OverStepRate0 = 1,
        OverStepRate1 = 2,
        Plain = 3,
    };

    u8 dest_vgpr;     ///< Destination VGPR to load first component.
    u8 num_elements;  ///< Number of components to load
    u8 sgpr_base;     ///< SGPR that contains the pointer to the list of vertex V#
    u8 dword_offset;  ///< The dword offset of the V# that describes this attribute.
    u8 instance_data; ///< Indicates that the buffer will be accessed in instance rate
    u8 inst_offset;   ///< Instruction offset applied on the formatted buffer loads
    u8 data_format;   ///< Data format override when typed buffer loads are used
    u8 num_format;    ///< Number format override when typed buffer loads are used

    InstanceIdType GetStepRate() const {
        return static_cast<InstanceIdType>(instance_data);
    }

    constexpr VkbAmdGpu::Buffer GetSharp(const VkbShader::Info& info) const noexcept {
        auto buffer = info.ReadUdReg<VkbAmdGpu::Buffer>(sgpr_base, dword_offset);
        buffer.base_address += inst_offset;
        if (data_format) {
            buffer.data_format = data_format;
            buffer.num_format = num_format;
        }
        return buffer;
    }

    bool operator==(const VertexAttribute& other) const = default;
};

struct FetchShaderData {
    u32 size = 0;
    SmallVector<VertexAttribute, 8> attributes;
    s8 vertex_offset_sgpr = -1;   ///< SGPR of vertex offset from VADDR
    s8 instance_offset_sgpr = -1; ///< SGPR of instance offset from VADDR

    bool operator==(const FetchShaderData& other) const {
        return attributes == other.attributes && vertex_offset_sgpr == other.vertex_offset_sgpr &&
               instance_offset_sgpr == other.instance_offset_sgpr;
    }

    bool Empty() const {
        return size == 0;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& buffer);
};

/// tlg: the shader list of the new engine (native_vkb_shader_list.h, TLG_VKB_SHADER_PREBUILD): a list entry holds the vertex input layout
/// instead of the fetch shader's code (game code is never copied into the list). While replay_fetch is set (the list's build process,
/// one entry at a time on its thread), ParseFetchShader gives that layout for a stage that has a fetch shader, without reading code.
inline thread_local const FetchShaderData* replay_fetch = nullptr;

const u32* GetFetchShaderCode(const Info& info, u32 sgpr_base);

bool ParseFetchShader(const VkbShader::Info& info, FetchShaderData& out_fetch_data);

} // namespace VkbShader::Gcn
