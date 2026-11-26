// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "native/shadps4/shader_recompiler/ir/type.h"
#include "native/shadps4/video_core/amdgpu/resource.h"

#include <boost/container/static_vector.hpp>

namespace VkbShader {

static constexpr u32 NUM_USER_DATA_REGS = 16;
static constexpr u32 NUM_IMAGES = 64;
static constexpr u32 NUM_BUFFERS = 56;
static constexpr u32 NUM_SAMPLERS = 16;
static constexpr u32 NUM_FMASKS = 8;

using SharpLocation = u16;

constexpr SharpLocation UNKNOWN_LOCATION = std::numeric_limits<u16>::max();

template <typename T>
struct SharpFetch {
    static constexpr std::size_t N = sizeof(T) / sizeof(u32);
    static_assert(N <= 8);

    enum class Summary : u8 {
        SingleLoad,
        MultiLoad,
        Invalid,
    };

    Summary summary{Summary::MultiLoad};
    std::array<u32, N> immediates;
    std::array<SharpLocation, N> offsets;
    u8 load_mask;

    bool operator==(const SharpFetch&) const = default;

    template <u32 num_dwords = N>
        requires(num_dwords <= N)
    constexpr bool Fetch(const u32* flatbuf, T* out) const {
        if (summary == Summary::SingleLoad) [[likely]] {
            std::memcpy(out, flatbuf + offsets[0], num_dwords * sizeof(u32));
            return true;
        } else if (summary == Summary::Invalid) [[unlikely]] {
            return false;
        }
        u8 mask = load_mask;
        std::array<u32, num_dwords> out_dw;
        for (u32 i = 0; i < num_dwords; i++) {
            out_dw[i] = (mask & 1) ? flatbuf[offsets[i]] : immediates[i];
            mask >>= 1;
        }
        std::memcpy(out, out_dw.data(), sizeof(out_dw));
        return true;
    }
};

enum class SharpFetchPostOp : u8 {
    None,
    // For buffers
    BitwiseOrDw1WithImm,
    OffsetByProgramBase,
    ClearStrideAndBitshiftDw2WithImm,
    // For images,
    ConvertCubeTo2DArray,
    // For samplers
    DisableAnisoIfSingleLod,
    ForceRepeatXyzClamp,
    ForceLastTexelXyClamp,
    ClearAnisoRatioAndThreshold,
};

enum class BufferType : u8 {
    Guest,
    Flatbuf,
    BdaPagetable,
    FaultBuffer,
    GdsBuffer,
    SharedMemory,
    ClipPlanes,
};

/// tlg C9 (the indexed data path): graphics shaders find the guest buffers of a draw through descriptor arrays of one global set (set 0) and a per draw row of slot numbers; what stays
/// on the push descriptors (images and samplers while only buffers are indexed, the special buffers) sits in set 1. The classification is shared by the shader backend and the binder.
namespace Indexed {
constexpr u32 GlobalSet = 0;
constexpr u32 LegacySet = 1;
constexpr u32 TableBinding = 0;  ///< the row table: u32 slots of the draws
constexpr u32 BufferBinding = 1; ///< the array of storage buffer descriptors
constexpr u32 SamplerBinding = 2; ///< the array of sampler descriptors
constexpr u32 ImageBinding = 3;   ///< the first of the arrays of image descriptors, one per array key (below)
constexpr u32 NumImageKeys = 12;  ///< number class (float, sint, uint) x dimension class (2D, 2D array/cube, 2D multisampled, 3D)
/// The storage formats an atomic image can have (the SPIR-V image format follows the data and number format of its T#; the backend has the same list in the same order).
constexpr u32 NumAtomicFormats = 18;
/// Array keys: 0..11 sampled images, 12..23 storage images (format unknown), 24 + 4 * format + dimension: storage images of an atomic format.
constexpr u32 NumImageArrays = 2 * NumImageKeys + 4 * NumAtomicFormats;
constexpr u32 FirstAtomicKey = 2 * NumImageKeys;
constexpr u32 NoKey = 0xff;       ///< the image stays on the push descriptors
constexpr u32 MaxBuffers = 1u << 16;
constexpr u32 MaxImages = 1u << 13;        ///< per sampled image array
constexpr u32 MaxStorageImages = 1u << 11; ///< per storage image array (the last RunSlots of it are the runs of the mip level images)
constexpr u32 RunSlots = 1024;             ///< consecutive slots of a storage image array for the images bound once per mip level (MipStorageFallbackMode::DynamicIndex)
constexpr u32 NullRunLength = 16;          ///< the first slots of the runs are null descriptors, for an image of the kind whose T# is invalid
constexpr u32 MaxAtomicImages = 1u << 8;   ///< per atomic format array
constexpr u32 MaxSamplers = 1u << 12;
/// The buffers found through the table: the guest buffers, the special buffers that are one fixed object, the clip planes and the shared memory of compute (null, or a stream copy
/// that is a slot of its own), and the flat user data, which is not a descriptor at all: its dwords are copied into the draw's row, behind the slots.
[[nodiscard]] constexpr bool IsIndexedBufferType(BufferType type) {
    return type == BufferType::Guest || type == BufferType::GdsBuffer || type == BufferType::BdaPagetable || type == BufferType::FaultBuffer ||
           type == BufferType::ClipPlanes || type == BufferType::Flatbuf || type == BufferType::SharedMemory;
}
} // namespace Indexed

struct BufferResource {
    bool operator==(const BufferResource&) const = default; ///< tlg: the shader store's check mode compares translations (TLG_VKB_SHADER_PREBUILD=check)
    SharpFetch<VkbAmdGpu::Buffer> sharp_fetch{};
    IR::Type used_types{};
    BufferType buffer_type{};
    bool is_written{};
    bool is_formatted{};
    SharpFetchPostOp post_op{};
    u32 post_op_dw1_mask{};
    u32 post_op_dw2_shift{};

    bool IsSpecial() const noexcept {
        return buffer_type != BufferType::Guest;
    }

    /// The sharp from four descriptor words the draw's record holds (a single load), with the same post operation as `GetSharp`.
    constexpr VkbAmdGpu::Buffer FromWords(const u32* words, const auto& info) const noexcept {
        VkbAmdGpu::Buffer buffer;
        std::memcpy(&buffer, words, sizeof(buffer));
        return PostOp(buffer, info);
    }

    constexpr VkbAmdGpu::Buffer GetSharp(const auto& info) const noexcept {
        VkbAmdGpu::Buffer buffer;
        if (!sharp_fetch.Fetch(info.FlatUserData().data(), &buffer)) {
            return VkbAmdGpu::Buffer::Null();
        }
        return PostOp(buffer, info);
    }

    constexpr VkbAmdGpu::Buffer PostOp(VkbAmdGpu::Buffer buffer, const auto& info) const noexcept {
        if (post_op == SharpFetchPostOp::BitwiseOrDw1WithImm) {
            reinterpret_cast<u32*>(&buffer)[1] |= post_op_dw1_mask;
        } else if (post_op == SharpFetchPostOp::ClearStrideAndBitshiftDw2WithImm) {
            buffer.stride = 0;
            buffer.num_records <<= post_op_dw2_shift;
        } else if (post_op == SharpFetchPostOp::OffsetByProgramBase) {
            buffer.base_address += info.ProgramBase();
        }
        if (!buffer.Valid()) {
            return VkbAmdGpu::Buffer::Null();
        }
        return buffer;
    }
};
using BufferResourceList = boost::container::static_vector<BufferResource, NUM_BUFFERS>;

enum class MipStorageFallbackMode : u16 {
    None,
    DynamicIndex,
    ConstantIndex,
};

struct ImageResource {
    bool operator==(const ImageResource&) const = default; ///< tlg: the shader store's check mode compares translations (TLG_VKB_SHADER_PREBUILD=check)
    SharpFetch<VkbAmdGpu::Image> sharp_fetch{};
    bool is_depth{};
    bool is_atomic{};
    bool is_array{};
    bool is_written{};
    bool is_r128{};
    /// tlg: read other than by normalized sampling without offsets (texel loads, size
    /// queries, offsets): needs the native-size image, not a reduced scene proxy.
    bool needs_native{};
    u8 constant_mip_index{};
    MipStorageFallbackMode mip_fallback_mode{};
    SharpFetchPostOp post_op{};
    /// tlg C9: the descriptor array the image is read from (Indexed::NoKey: the push descriptors), set by the SPIR-V backend when it declares the image; part of the persistent info.
    u8 array_key{0xff};

    /// The sharp from the eight descriptor words the draw's record holds (a single load), with the same fix ups as `GetSharp`.
    constexpr VkbAmdGpu::Image FromWords(const u32* words) const noexcept {
        VkbAmdGpu::Image image{};
        std::memcpy(&image, words, (is_r128 ? 4 : 8) * sizeof(u32));
        if (is_r128) {
            image.pitch = image.width;
        }
        return Finish(image);
    }

    constexpr VkbAmdGpu::Image GetSharp(const auto& info) const noexcept {
        VkbAmdGpu::Image image{};
        if (!Fetch(info.FlatUserData().data(), &image)) {
            return VkbAmdGpu::Image::Null(is_depth);
        }
        return Finish(image);
    }

    constexpr VkbAmdGpu::Image Finish(VkbAmdGpu::Image image) const noexcept {
        if (post_op == SharpFetchPostOp::ConvertCubeTo2DArray) {
            image.type = u64(VkbAmdGpu::ImageType::Color2DArray);
            image.depth = (image.depth + 1) * 6 - 1;
        }
        if (!image.Valid()) {
            image = VkbAmdGpu::Image::Null(is_depth);
        } else if (is_depth) {
            const auto data_fmt = image.GetDataFmt();
            if (data_fmt != VkbAmdGpu::DataFormat::Format16 &&
                data_fmt != VkbAmdGpu::DataFormat::Format32) {
                image = VkbAmdGpu::Image::Null(true);
            }
        }
        return image;
    }

    /// The raw T# (before the fix ups of `GetSharp`) from the descriptor words of the draw's record.
    constexpr void RawFromWords(const u32* words, VkbAmdGpu::Image* out) const noexcept {
        std::memcpy(out, words, (is_r128 ? 4 : 8) * sizeof(u32));
        if (is_r128) {
            out->pitch = out->width;
        }
    }

    constexpr bool Fetch(const u32* flatbuf, VkbAmdGpu::Image* out) const {
        if (!is_r128) {
            // Fetch full 8 byte T#
            return sharp_fetch.Fetch(flatbuf, out);
        }
        // Fetch r128 T# and fix pitch
        if (!sharp_fetch.Fetch<4>(flatbuf, out)) {
            return false;
        }
        out->pitch = out->width;
        return true;
    }

    u32 NumBindings(const auto& info) const {
        const VkbAmdGpu::Image tsharp = GetSharp(info);
        return (mip_fallback_mode == MipStorageFallbackMode::DynamicIndex)
                   ? (tsharp.last_level - tsharp.base_level + 1)
                   : 1;
    }
};
using ImageResourceList = boost::container::static_vector<ImageResource, NUM_IMAGES>;

namespace Indexed {
/// The images found through the arrays: every image the backend gave a key (all but the mip level fallback with a dynamic index, and atomics of formats other than R32).
[[nodiscard]] inline bool IsIndexedImage(const ImageResource& image) noexcept {
    return image.array_key != NoKey;
}
/// The descriptor array an image of this T# is read from: what the shader declares (the sampled type of its number class, the dimension of its host view type).
[[nodiscard]] inline u32 ImageKey(const VkbAmdGpu::Image& sharp, bool is_array) noexcept {
    u32 number = 0;
    switch (VkbAmdGpu::GetNumberClass(sharp.GetNumberFmt())) {
    case VkbAmdGpu::NumberClass::Sint:
        number = 1;
        break;
    case VkbAmdGpu::NumberClass::Uint:
        number = 2;
        break;
    default:
        break;
    }
    u32 dimension = 0;
    switch (sharp.GetHostViewType(is_array)) {
    case VkbAmdGpu::ImageType::Color1DArray:
    case VkbAmdGpu::ImageType::Color2DArray:
    case VkbAmdGpu::ImageType::Cube:
        dimension = 1;
        break;
    case VkbAmdGpu::ImageType::Color2DMsaa:
        dimension = 2;
        break;
    case VkbAmdGpu::ImageType::Color3D:
        dimension = 3;
        break;
    default:
        break;
    }
    return number * 4 + dimension;
}
/// The index of the format of an atomic image (the order of `AtomicSpvFormats` in the backend), or NumAtomicFormats.
[[nodiscard]] inline u32 AtomicFormatIndex(const VkbAmdGpu::Image& sharp) noexcept {
    using D = VkbAmdGpu::DataFormat;
    using N = VkbAmdGpu::NumberFormat;
    static constexpr std::array<std::pair<D, N>, NumAtomicFormats> formats{{
        {D::Format32, N::Uint},
        {D::Format32, N::Sint},
        {D::Format32, N::Float},
        {D::Format32_32, N::Float},
        {D::Format32_32, N::Uint},
        {D::Format32_32_32_32, N::Uint},
        {D::Format16, N::Float},
        {D::Format16, N::Uint},
        {D::Format16_16, N::Float},
        {D::Format16_16, N::Snorm},
        {D::Format8_8, N::Unorm},
        {D::Format16_16_16_16, N::Float},
        {D::Format16_16_16_16, N::Unorm},
        {D::Format8, N::Unorm},
        {D::Format8_8_8_8, N::Unorm},
        {D::Format8_8_8_8, N::Uint},
        {D::Format10_11_11, N::Float},
        {D::Format32_32_32_32, N::Float},
    }};
    for (u32 i = 0; i < NumAtomicFormats; ++i) {
        if (sharp.GetDataFmt() == formats[i].first && sharp.GetNumberFmt() == formats[i].second) {
            return i;
        }
    }
    return NumAtomicFormats;
}
/// The array of an image resource with the T# the shader was compiled for (NoKey: it stays on the push descriptors).
[[nodiscard]] inline u32 ArrayKey(const ImageResource& image, const VkbAmdGpu::Image& sharp) noexcept {
    const u32 key = ImageKey(sharp, image.is_array);
    if (!image.is_written) {
        return key;
    }
    if (!image.is_atomic) {
        return NumImageKeys + key; // (also the images bound once per mip level: a run of consecutive slots)
    }
    // atomics: the SPIR-V image format follows the data and number format; one array per format and dimension
    const u32 format = AtomicFormatIndex(sharp);
    if (format == NumAtomicFormats || image.mip_fallback_mode == MipStorageFallbackMode::DynamicIndex) {
        return NoKey;
    }
    return FirstAtomicKey + format * 4 + key % 4;
}
} // namespace Indexed

struct SamplerResource {
    bool operator==(const SamplerResource&) const = default; ///< tlg: the shader store's check mode compares translations (TLG_VKB_SHADER_PREBUILD=check)
    SharpFetch<VkbAmdGpu::Sampler> sharp_fetch{};
    SharpFetchPostOp post_op{};
    SharpLocation post_op_tsharp_dw3_off{};
    bool is_depth{};

    /// True when the sampler can be built from its own four descriptor words (the record's): the post operation needs nothing else from the flat user data.
    constexpr bool FromWordsPossible() const noexcept {
        return sharp_fetch.summary == SharpFetch<VkbAmdGpu::Sampler>::Summary::SingleLoad && post_op != SharpFetchPostOp::DisableAnisoIfSingleLod;
    }

    constexpr VkbAmdGpu::Sampler FromWords(const u32* words) const noexcept {
        VkbAmdGpu::Sampler sampler{};
        std::memcpy(&sampler, words, sizeof(sampler));
        return PostOp(sampler, 0);
    }

    constexpr VkbAmdGpu::Sampler GetSharp(const auto& info) const noexcept {
        VkbAmdGpu::Sampler sampler{};
        sharp_fetch.Fetch(info.FlatUserData().data(), &sampler);
        return PostOp(sampler, post_op == SharpFetchPostOp::DisableAnisoIfSingleLod ? info.FlatUserData()[post_op_tsharp_dw3_off] : 0);
    }

    constexpr VkbAmdGpu::Sampler PostOp(VkbAmdGpu::Sampler sampler, u32 tsharp_dw3) const noexcept {
        if (post_op == SharpFetchPostOp::DisableAnisoIfSingleLod) {
            if (((tsharp_dw3 >> 12) & 0xff) == 0) {
                sampler.max_aniso.Assign(VkbAmdGpu::AnisoRatio::One);
            }
        } else if (post_op == SharpFetchPostOp::ForceRepeatXyzClamp) {
            sampler.clamp_x.Assign(VkbAmdGpu::ClampMode::Wrap);
            sampler.clamp_y.Assign(VkbAmdGpu::ClampMode::Wrap);
            sampler.clamp_z.Assign(VkbAmdGpu::ClampMode::Wrap);
        } else if (post_op == SharpFetchPostOp::ForceLastTexelXyClamp) {
            sampler.clamp_x.Assign(VkbAmdGpu::ClampMode::ClampLastTexel);
            sampler.clamp_y.Assign(VkbAmdGpu::ClampMode::ClampLastTexel);
            sampler.clamp_z.Assign(VkbAmdGpu::ClampMode::Wrap);
        } else if (post_op == SharpFetchPostOp::ClearAnisoRatioAndThreshold) {
            sampler.max_aniso.Assign(VkbAmdGpu::AnisoRatio::One);
            sampler.aniso_threshold.Assign(0);
        }
        return sampler;
    }
};
using SamplerResourceList = boost::container::static_vector<SamplerResource, NUM_SAMPLERS>;

struct FMaskResource {
    bool operator==(const FMaskResource&) const = default; ///< tlg: the shader store's check mode compares translations (TLG_VKB_SHADER_PREBUILD=check)
    SharpLocation sharp_idx;

    constexpr VkbAmdGpu::Image GetSharp(const auto& info) const noexcept {
        return info.template ReadUdSharp<VkbAmdGpu::Image>(sharp_idx);
    }
};
using FMaskResourceList = boost::container::static_vector<FMaskResource, NUM_FMASKS>;

struct PushData {
    static constexpr u32 XOffsetIndex = 0;
    static constexpr u32 YOffsetIndex = 1;
    static constexpr u32 XScaleIndex = 2;
    static constexpr u32 YScaleIndex = 3;
    static constexpr u32 BufOffsetIndex = 4;

    float xoffset;
    float yoffset;
    float xscale;
    float yscale;
    std::array<u8, NUM_BUFFERS> buf_offsets;
    u32 motion_param; ///< tlg: object motion parameter index (0: off)
    u32 scene_size; ///< reduced width | height << 16; zero means native pixel coordinates
    u32 row_base;   ///< tlg C9: first entry of this draw's row in the indexed data table
    static constexpr u32 RowBaseIndex = 7;

    void AddOffset(u32 binding, u32 offset) {
        ASSERT(offset < 256 && binding < buf_offsets.size());
        buf_offsets[binding] = offset;
    }
};
static_assert(sizeof(PushData) <= 128,
              "PushData size is greater than minimum size guaranteed by Vulkan spec");

} // namespace VkbShader
