// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/vkb_core/vkb_device.h"
#include "native/native_vkb_objects.h"
#include "native/native_vkb_indexed.h"
#include "native/native_gpu_read_trap.h"
#include "native/native_desc.h"
#include "native/native_tsharp_watch.h"
#include "native/native_vkb_util.h"

#include "native/shadps4/video_core/renderer_vulkan/vk_common.h"
#include "native/shadps4/video_core/host_shaders/fill_pattern_comp.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_shader_util.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_platform.h"
#include <vulkan/vulkan_format_traits.hpp>
#include <condition_variable>
#include <functional>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <thread>
#include <memory>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include <map>
#include <string>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <nlohmann/json.hpp>

#include "common/thread.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "native/native_debug_server.h"
#include "native/native_vkb.h"
#include "native/native_vkb_stream.h"
#include "native/native_vkb_upload.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_host_import.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_instance.h"

#include <vk_mem_alloc.h>
#include <xxhash.h>
#ifndef _WIN32
#include <sys/mman.h>
extern "C" {
// runtime_memory.c: host memory the game can see, in [0x800000000, 0x1000000000) (below 40 bits); null when that range is full
void* runtime_low_map(size_t size, int prot);
}
#endif


namespace Core::Vkb {
void BumpBufferEpoch(); ///< note 46: an A1 memo answer may have changed (definitions below)
void BumpViewEpoch();

/// API contract census (documents/tlg/native plan/engine/api-contract.md, step C8; debug, TLG_VKB_CONTRACT_CENSUS=1): counts of the engine values the contract rows
/// depend on (sampler descriptions, texture and view formats, clear formats, alpha to coverage, depth bias, a texture copied into and then rewritten by the CPU)
namespace {
std::mutex contract_mutex;
std::map<std::string, u64> contract_counts;
std::set<u64> contract_copy_destinations; ///< texture ids a GPU copy wrote (row L7)
} // namespace
bool ContractCensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_CONTRACT_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
void ContractNote(const std::string& key) {
    if (!ContractCensusOn()) {
        return;
    }
    std::scoped_lock lock{contract_mutex};
    ++contract_counts[key];
}
namespace {
std::set<u64> contract_target7; ///< C10: texture ids bound as target 7
} // namespace
void ContractTarget7(u64 texture_id, const std::string& key) {
    if (!ContractCensusOn()) {
        return;
    }
    std::scoped_lock lock{contract_mutex};
    contract_target7.insert(texture_id);
    ++contract_counts[key];
}
void ContractSampled(u64 texture_id) {
    if (!ContractCensusOn() || texture_id == 0) {
        return;
    }
    std::scoped_lock lock{contract_mutex};
    if (contract_target7.contains(texture_id)) {
        ++contract_counts[fmt::format("C10 a shader reads target-7 texture {}", texture_id)];
    }
}
nlohmann::json ContractReport() {
    nlohmann::json j = nlohmann::json::object();
    std::scoped_lock lock{contract_mutex};
    for (const auto& [key, n] : contract_counts) {
        j[key] = n;
    }
    return j;
}

namespace {

using NativeDevice::Kind;
using NativeDevice::Object;

// Direct3D 11 bind flags and misc flags
constexpr u32 BindShaderResource = 0x8, BindRenderTarget = 0x20, BindDepthStencil = 0x40, BindUnorderedAccess = 0x80;
constexpr u32 MiscTextureCube = 0x4;

/// A DXGI format as a Vulkan format (the plain Direct3D 11 to Vulkan mapping; typeless formats take their unorm / float / uint member). eUndefined: not mapped.
vk::Format ColorFormat(u32 dxgi) {
    using F = vk::Format;
    switch (dxgi) {
    case 1: case 3: return F::eR32G32B32A32Uint;
    case 2: return F::eR32G32B32A32Sfloat;
    case 4: return F::eR32G32B32A32Sint;
    case 6: return F::eR32G32B32Sfloat;
    case 9: case 10: return F::eR16G16B16A16Sfloat;
    case 11: return F::eR16G16B16A16Unorm;
    case 12: return F::eR16G16B16A16Uint;
    case 13: return F::eR16G16B16A16Snorm;
    case 14: return F::eR16G16B16A16Sint;
    case 15: case 17: return F::eR32G32Uint;
    case 16: return F::eR32G32Sfloat;
    case 18: return F::eR32G32Sint;
    case 23: case 24: return F::eA2B10G10R10UnormPack32;
    case 25: return F::eA2B10G10R10UintPack32;
    case 26: return F::eB10G11R11UfloatPack32;
    case 27: case 28: return F::eR8G8B8A8Unorm;
    case 29: return F::eR8G8B8A8Srgb;
    case 30: return F::eR8G8B8A8Uint;
    case 31: return F::eR8G8B8A8Snorm;
    case 32: return F::eR8G8B8A8Sint;
    case 33: case 34: return F::eR16G16Sfloat;
    case 35: return F::eR16G16Unorm;
    case 36: return F::eR16G16Uint;
    case 37: return F::eR16G16Snorm;
    case 38: return F::eR16G16Sint;
    case 39: case 42: return F::eR32Uint;
    case 40: case 41: return F::eR32Sfloat;
    case 43: return F::eR32Sint;
    case 44: case 46: return F::eR32Sfloat; // R24G8 / R24_UNORM_X8: the console keeps such depth as 32 bit float
    case 48: case 49: return F::eR8G8Unorm;
    case 50: return F::eR8G8Uint;
    case 51: return F::eR8G8Snorm;
    case 52: return F::eR8G8Sint;
    case 53: case 54: return F::eR16Sfloat;
    case 55: case 56: return F::eR16Unorm;
    case 57: return F::eR16Uint;
    case 58: return F::eR16Snorm;
    case 59: return F::eR16Sint;
    case 60: case 61: case 65: return F::eR8Unorm;
    case 62: return F::eR8Uint;
    case 63: return F::eR8Snorm;
    case 64: return F::eR8Sint;
    case 70: case 71: return F::eBc1RgbaUnormBlock;
    case 72: return F::eBc1RgbaSrgbBlock;
    case 73: case 74: return F::eBc2UnormBlock;
    case 75: return F::eBc2SrgbBlock;
    case 76: case 77: return F::eBc3UnormBlock;
    case 78: return F::eBc3SrgbBlock;
    case 79: case 80: return F::eBc4UnormBlock;
    case 81: return F::eBc4SnormBlock;
    case 82: case 83: return F::eBc5UnormBlock;
    case 84: return F::eBc5SnormBlock;
    case 85: return F::eR5G6B5UnormPack16;
    case 86: return F::eA1R5G5B5UnormPack16;
    case 87: case 88: return F::eB8G8R8A8Unorm;
    // api-contract.md R4 / R6 (2026-11-18): the engine's table maps B8G8R8A8_TYPELESS (90) and the B8G8R8X8 typeless / sRGB pair (92, 93) to its sRGB 8x4
    // format (92 / 93 select BGR1: alpha 1 from the view's T#, R8), and R9G9B9E5 (67) to its 5_9_9_9 float format
    case 90: case 91: case 92: case 93: return F::eB8G8R8A8Srgb;
    case 67: return F::eE5B9G9R9UfloatPack32;
    case 94: case 95: return F::eBc6HUfloatBlock;
    case 96: return F::eBc6HSfloatBlock;
    case 97: case 98: return F::eBc7UnormBlock;
    case 99: return F::eBc7SrgbBlock;
    default: return F::eUndefined;
    }
}

/// The depth format of a texture bound as a depth stencil target (eUndefined: not a depth format)
vk::Format DepthFormat(u32 dxgi) {
    switch (dxgi) {
    case 39: case 40: case 41: return vk::Format::eD32Sfloat;
    case 19: case 20: case 21: case 22: case 44: case 45: case 46: case 47: return vk::Format::eD32SfloatS8Uint;
    case 53: case 55: case 56: return vk::Format::eD16Unorm;
    default: return vk::Format::eUndefined;
    }
}
bool HasStencil(vk::Format f) {
    return f == vk::Format::eD32SfloatS8Uint;
}
bool IsStencilView(u32 dxgi) {
    return dxgi == 22 || dxgi == 47; // X32_TYPELESS_G8X24_UINT, X24_TYPELESS_G8_UINT
}

struct Image {
    vk::Image image{};
    VmaAllocation allocation{};
    vk::Format format{};
    vk::ImageType type{};
    u32 width{}, height{}, depth{}, mips{}, layers{};
    bool is_depth{}, cube{};
    bool cube_compatible{}; ///< square 2D with a multiple of six layers: a module may read it as a cube even when the engine did not flag it one
    u64 address{};          ///< the engine texture's memory (V4c: the display buffers)
    u64 bytes{};
    u64 guest_bytes{}; ///< the engine texture's memory size (+0x38): a file read into it gives the image new contents (MemoryLoaded)
    bool size_tried{};  ///< guest_bytes derived from the description once (note 84)
    bool gpu_written{}; ///< a render target or UAV: its contents live in our image only (the engine's memory under it is stale), never uploaded from a file read
    vk::ImageLayout layout{vk::ImageLayout::eUndefined}; ///< after its upload (V2b); render targets stay undefined until V4 writes them
    bool mutable_format{true}; ///< R5 test TLG_VKB_TARGETS_FIXED=1: false for color render targets made without MUTABLE_FORMAT (views in its own format only)
};
/// R5 bound test (TLG_VKB_TARGETS_FIXED=1, wrong picture where a target is read in another format): color render targets (not UAVs) are made without MUTABLE_FORMAT /
/// EXTENDED_USAGE, which can keep the driver from compressing them
bool TargetsFixed() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_TARGETS_FIXED"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::atomic<u64> c_fixed_targets{0}, c_fixed_view_format_forced{0}, c_fixed_attachment_refused{0};
struct View {
    vk::ImageView view{};
    u64 texture_id{};
    vk::ImageViewType type{};
    vk::Format format{};
    vk::ImageSubresourceRange range{};
    std::array<vk::ImageView, 7> other{}; ///< the same range in another view type, by vk::ImageViewType (made at the first need)
    std::vector<std::pair<vk::Format, vk::ImageView>> formatted; ///< V4: the same range in another format (attachments in the pipeline's format)
    vk::ComponentMapping components{}; ///< R8: the channels the engine's T# selects (sampled views); the `other` view types read the same channels (D4-07)
};
struct Doomed {
    u32 due_submit;
    vk::Image image;
    VmaAllocation allocation;
    vk::ImageView view;
    VkBuffer buffer{};
    VmaAllocation buffer_allocation{};
};
/// V2f: our device buffer for an engine buffer the GPU writes, by the start of the engine buffer's memory
struct DeviceBuffer {
    u64 end{};
    VkBuffer buffer{};
    VmaAllocation allocation{};
    u64 id{};
    bool gpu_writes{}; ///< bound as an unordered access view: the GPU writes it (its bytes are ours alone)
};

std::mutex mutex;
std::unordered_map<u64, Image> images;
std::unordered_map<u64, View> views;
std::unordered_map<u64, u64> image_at_address; ///< V4c: the newest render target image at an engine texture's memory, by address
std::map<u64, DeviceBuffer> device_buffers;
std::unordered_map<u64, u64> devbuf_hash; ///< api-contract.md Y7 census: device buffer start -> hash of its guest memory at the last check (under `mutex`)
std::unordered_map<u64, std::string> devbuf_desc; ///< api-contract.md Y7 census: device buffer start -> its engine description (under `mutex`)
std::unordered_map<u64, u64> device_buffer_start; ///< engine id -> start address
std::deque<Doomed> doomed;
std::unique_ptr<VkbVulkan::HostImport> host_import;
bool device_ready = false;
const VkbVulkan::Instance* instance = nullptr;

std::atomic<u64> c_images_created{0}, c_images_failed{0}, c_images_unmapped_format{0}, c_images_destroyed{0}, c_image_bytes{0}, c_image_ns{0};
std::atomic<u64> c_views_created{0}, c_views_failed{0}, c_views_clamped{0}, c_views_no_image{0}, c_views_destroyed{0}, c_views_buffer{0};
std::atomic<u64> c_dev_buffers{0}, c_dev_failed{0}, c_dev_bytes{0}, c_dev_destroyed{0}, c_dev_no_initial{0}, c_dev_replaced{0}, c_find_straddle{0}, c_find_none{0};
std::map<std::string, u64> refused_pairs; ///< (mutex) our view's type, the type a module wanted
std::atomic<u64> c_other_views{0}, c_other_view_refused{0}, c_tex_copies{0}, c_tex_copy_no_image{0}, c_tex_copy_mismatch{0};
std::atomic<u64> c_display_found{0}, c_display_missed{0}, c_display_not_general{0};
std::atomic<u64> c_attach_views{0}, c_attach_refused{0}, c_clears_color{0}, c_clears_depth{0}, c_clear_no_view{0};
std::atomic<u64> c_samplers_made{0}, c_samplers_invalid{0}, c_sampler_minmax{0}, c_sampler_border_rounded{0}, c_sampler_mirror_once{0};
std::unordered_map<u64, vk::Sampler> samplers; ///< by engine sampler id (mutex)
std::atomic<u64> c_reupload_wanted{0}, c_reupload_done{0}, c_reupload_no_image{0}, c_reupload_in_stream{0}, c_moved_textures{0}, c_moved_no_image{0},
    c_moved_mismatch{0}, c_moved_cut_to_blocks{0};
std::atomic<u64> c_cpu_writes{0}, c_cpu_write_bytes{0}, c_copies{0}, c_copy_bytes{0}, c_copies_device{0}, c_copy_unresolved{0}, c_copy_overlap{0}, c_fills{0}, c_fill_unresolved{0};
std::atomic<u64> c_buffers{0}, c_buffers_in_place{0}, c_buffers_not_in_place{0}, c_buffer_bytes{0}, c_too_early{0}, c_deleted_unknown{0};

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// The device, once the presenter (and with it the rasterizer) exists; false before that (the object is then made at its first use)
bool Ready() {
    if (device_ready) {
        return true;
    }
    if (!Core::Vkb::Device::Ready()) { // merge note 5: the backend's own device, made by its presenter
        c_too_early.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    instance = &Core::Vkb::Device::Instance();
    host_import = std::make_unique<VkbVulkan::HostImport>(*instance);
    Upload::Init(*instance, *host_import);
    Stream::Init(*instance);
    device_ready = true;
    LOG_INFO(Render_Vulkan, "vkb: objects on the device, in place buffers {}", host_import->Available() ? "available" : "NOT available");
    return true;
}

u32 FullMips(u32 w, u32 h, u32 d) {
    u32 m = 1;
    for (u32 s = std::max({w, h, d}); s > 1; s >>= 1) {
        ++m;
    }
    return m;
}

/// note 66 (R2b part 2a): an image gets its one layout, GENERAL, when it is made (a D3D11 driver's resources keep one layout for their life): an image with initial
/// data is left in it by its upload; every other one by one stream job queued right after the creation (in stream order before any work that can use it). Binds,
/// clears and copies never move a layout.
struct NewImage {
    vk::Image image;
    vk::ImageAspectFlags aspect;
    bool batch; ///< a colour texture that is not a render target: its contents can come through the upload batch (Map / Unmap), so its layout is set there, in order with them
};
thread_local std::vector<NewImage> t_new_images; ///< made under this module's lock; their layout job is queued after it (no job is made under the lock)
std::atomic<u64> c_layout_jobs{0}, c_layout_images{0}, c_layout_by_upload{0}, c_layout_in_batch{0};
/// Queues the layout job of the images this thread made (this module's lock not held)
void InitNewImages() {
    if (t_new_images.empty()) {
        return;
    }
    std::vector<NewImage> list;
    for (const NewImage& n : t_new_images) {
        if (n.batch) {
            Upload::InitLayout(n.image);
            c_layout_in_batch.fetch_add(1, std::memory_order_relaxed);
        } else {
            list.push_back(n);
        }
    }
    t_new_images.clear();
    if (list.empty()) {
        return;
    }
    c_layout_jobs.fetch_add(1, std::memory_order_relaxed);
    c_layout_images.fetch_add(list.size(), std::memory_order_relaxed);
    Stream::Defer(Stream::Work::Other, [list = std::move(list)](vk::CommandBuffer cmd) {
        boost::container::small_vector<vk::ImageMemoryBarrier2, 8> barriers;
        for (const NewImage& n : list) {
            barriers.push_back({.srcStageMask = vk::PipelineStageFlagBits2::eNone,
                                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                                .oldLayout = vk::ImageLayout::eUndefined,
                                .newLayout = vk::ImageLayout::eGeneral,
                                .image = n.image,
                                .subresourceRange = {n.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
        }
        cmd.pipelineBarrier2({.imageMemoryBarrierCount = static_cast<u32>(barriers.size()), .pImageMemoryBarriers = barriers.data()});
        // Tint hunt kill experiment (2026-11-16, TLG_VKB_CLEAR_NEW_IMAGES=1): an image with no contents starts as zeros (on the console its memory holds what
        // the engine left there; a Vulkan image starts undefined), in case something reads it before the first write.
        static const bool clear_new = [] { const char* e = std::getenv("TLG_VKB_CLEAR_NEW_IMAGES"); return e != nullptr && e[0] == '1'; }();
        if (clear_new) {
            for (const NewImage& n : list) {
                const vk::ImageSubresourceRange range{n.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
                if (n.aspect & vk::ImageAspectFlagBits::eColor) {
                    cmd.clearColorImage(n.image, vk::ImageLayout::eGeneral, vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}}, range);
                } else {
                    cmd.clearDepthStencilImage(n.image, vk::ImageLayout::eGeneral, vk::ClearDepthStencilValue{0.0f, 0}, range);
                }
            }
            const vk::MemoryBarrier2 cleared{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                             .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                             .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
            cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &cleared});
        }
    });
}

/// Creates the image of `texture` (mutex held)
Image* MakeImage(const Object& texture) {
    if (auto it = images.find(texture.id); it != images.end()) {
        return &it->second;
    }
    const u64 t0 = NowNs();
    Image img;
    img.is_depth = (texture.bind & BindDepthStencil) != 0;
    img.format = img.is_depth ? DepthFormat(texture.format) : ColorFormat(texture.format);
    if (ContractCensusOn()) {
        ContractNote(fmt::format("R texture dxgi {} {} bind {:#x} -> {}", texture.format, img.is_depth ? "depth" : "color", texture.bind, vk::to_string(img.format)));
    }
    if (img.format == vk::Format::eUndefined) {
        c_images_unmapped_format.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 16) {
            LOG_WARNING(Render_Vulkan, "vkb: texture id {} has DXGI format {} (bind {:#x}) with no Vulkan format", texture.id, texture.format, texture.bind);
        }
        return nullptr;
    }
    img.width = std::max(texture.width, 1u);
    img.height = std::max(texture.height, 1u);
    img.depth = std::max(texture.depth, 1u);
    // note 41: a 1D texture is a 2D image of height 1: the recompiled shaders declare every 1D image as 2D (spirv_emit_context.cpp ImageType), and a 1D view in a 2D
    // slot reads zeros (the occlusion pass 0x1930bb5e read its 1D tables as 0: the grey haze)
    img.type = texture.kind == Kind::Texture3D ? vk::ImageType::e3D : vk::ImageType::e2D;
    const u32 full = FullMips(img.width, img.height, img.type == vk::ImageType::e3D ? img.depth : 1);
    img.mips = texture.mips == 0 ? full : std::min(texture.mips, full);
    img.layers = img.type == vk::ImageType::e3D ? 1 : std::max(texture.layers, 1u);
    img.cube_compatible = img.type == vk::ImageType::e2D && img.layers % 6 == 0 && img.width == img.height;
    img.address = texture.address;
    img.guest_bytes = texture.bytes;
    img.gpu_written = (texture.bind & (BindRenderTarget | BindUnorderedAccess)) != 0;
    img.cube = (texture.misc & MiscTextureCube) != 0 && img.cube_compatible;
    // debug (tint hunt, 2026-11-16): TLG_VKB_WATCH_VOLUMES=1 arms a hardware write watch near the end of the first probe volumes' memory (36 x 36 x N 3D textures), to find the
    // engine code that fills them after we copied them (the hits go to the log: "write watch")
    static const bool watch_volumes = [] { const char* e = std::getenv("TLG_VKB_WATCH_VOLUMES"); return e != nullptr && e[0] == '1'; }();
    if (watch_volumes && img.type == vk::ImageType::e3D && img.width == 36 && texture.address != 0) {
        const u64 at = texture.address + static_cast<u64>(img.width) * img.height * img.depth * 8 * 15 / 16;
        Core::NativeTsharpWatch::ArmRotating(at, "vkb probe volume"); // the newest four volumes
    }
    u32 samples = 1;
    if (texture.kind == Kind::Texture2D && texture.object != 0) {
        samples = *reinterpret_cast<const u32*>(texture.object + 0x180 + 5 * 4); // SampleCount
        if (samples != 2 && samples != 4 && samples != 8) {
            samples = 1;
        }
    }
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
    if (img.is_depth) {
        usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
    } else {
        if ((texture.bind & BindRenderTarget) != 0) {
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
        }
        if ((texture.bind & BindUnorderedAccess) != 0) {
            usage |= vk::ImageUsageFlagBits::eStorage;
        }
    }
    vk::ImageCreateFlags flags = img.is_depth ? vk::ImageCreateFlags{} : (vk::ImageCreateFlagBits::eMutableFormat | vk::ImageCreateFlagBits::eExtendedUsage);
    if (TargetsFixed() && !img.is_depth && (texture.bind & BindRenderTarget) != 0 && (texture.bind & BindUnorderedAccess) == 0) {
        flags = {};
        img.mutable_format = false;
        c_fixed_targets.fetch_add(1, std::memory_order_relaxed);
    }
    if (img.cube_compatible && samples == 1) {
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }
    if (img.type == vk::ImageType::e3D && !img.is_depth) {
        flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
    }
    const vk::ImageCreateInfo ci{
        .flags = flags,
        .imageType = img.type,
        .format = img.format,
        .extent = {img.width, img.height, img.depth},
        .mipLevels = img.mips,
        .arrayLayers = img.layers,
        .samples = static_cast<vk::SampleCountFlagBits>(samples),
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    const VmaAllocationCreateInfo alloc_ci{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    const VkImageCreateInfo ci_raw = static_cast<VkImageCreateInfo>(ci);
    VkImage raw{};
    VmaAllocationInfo info{};
    const VkResult result = vmaCreateImage(instance->GetAllocator(), &ci_raw, &alloc_ci, &raw, &img.allocation, &info);
    if (result != VK_SUCCESS) {
        c_images_failed.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 16) {
            LOG_ERROR(Render_Vulkan, "vkb: image of texture id {} ({}x{}x{} mips {} layers {} {}) failed: {}", texture.id, img.width, img.height, img.depth, img.mips,
                      img.layers, vk::to_string(img.format), vk::to_string(vk::Result{result}));
        }
        return nullptr;
    }
    img.image = vk::Image{raw};
    img.bytes = info.size;
    c_images_created.fetch_add(1, std::memory_order_relaxed);
    c_image_bytes.fetch_add(info.size, std::memory_order_relaxed);
    c_image_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    // note 66: GENERAL from here on (the job is queued by the caller after the lock; an upload of its contents takes its place)
    img.layout = vk::ImageLayout::eGeneral;
    // debug: under the validation layer (TLG_VK_VALIDATION=1) each image carries its texture id, size, format and how its layout is set, so a message names it
    static const bool name_images = [] { const char* e = std::getenv("TLG_VK_VALIDATION"); return e != nullptr && e[0] == '1'; }();
    if (name_images) {
        const char* path = img.is_depth ? "depth" : (texture.bind & BindRenderTarget) != 0 ? "target" : "texture (batch layout)";
        VkbVulkan::SetObjectName(instance->GetDevice(), img.image, "vkb tex {} {}x{}x{} mips {} layers {} {} {} usage {} bind {:#x}", texture.id, img.width, img.height, img.depth,
                                 img.mips, img.layers, vk::to_string(img.format), path, texture.usage, texture.bind);
    }
    t_new_images.push_back({img.image, !img.is_depth ? vk::ImageAspectFlags{vk::ImageAspectFlagBits::eColor}
                                      : HasStencil(img.format) ? (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)
                                                               : vk::ImageAspectFlags{vk::ImageAspectFlagBits::eDepth},
                            !img.is_depth && (texture.bind & BindRenderTarget) == 0});
    return &images.emplace(texture.id, img).first->second;
}

/// The view's range from its Direct3D 11 description (`d`, dwords after format and dimension), as (view type, first level, levels, first layer, layers); false: not an image view
struct Range {
    vk::ImageViewType type;
    u32 base_mip, mips, base_layer, layers;
    bool depth_slices{}; ///< a 3D render target or UAV that names depth slices (the view covers the whole volume here)
};
bool ViewRange(Kind kind, const Object& view, const Image& img, Range& r) {
    const u32 all_mips = img.mips, all_layers = img.layers;
    r = {img.type == vk::ImageType::e1D ? (all_layers > 1 ? vk::ImageViewType::e1DArray : vk::ImageViewType::e1D)
         : img.type == vk::ImageType::e3D ? vk::ImageViewType::e3D
         // D4-08 (JsDevice_CreateShaderResourceView 02493620): a null description of a cube texture with more than six faces is the engine's dimension 10 (a cube
         // array of faces / 6 cubes; a single-cube view must have exactly six layers); a render target or UAV of it is a 2D array, as in D3D11
         : img.cube ? (all_layers <= 6 ? vk::ImageViewType::eCube : kind == Kind::ShaderResourceView ? vk::ImageViewType::eCubeArray : vk::ImageViewType::e2DArray)
         : (all_layers > 1 ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D),
         0, all_mips, 0, all_layers};
    if (!view.has_desc) {
        return true; // a null description: the whole resource
    }
    const u32 dim = view.desc[1];
    const u32* d = &view.desc[2];
    const bool srv = kind == Kind::ShaderResourceView;
    const bool dsv = kind == Kind::DepthStencilView;
    if (srv) {
        switch (dim) {
        case 1: case 11: return false; // buffer
        case 2: r = {vk::ImageViewType::e1D, d[0], d[1], 0, 1}; break;
        case 3: r = {vk::ImageViewType::e1DArray, d[0], d[1], d[2], d[3]}; break;
        case 4: r = {vk::ImageViewType::e2D, d[0], d[1], 0, 1}; break;
        case 5: r = {vk::ImageViewType::e2DArray, d[0], d[1], d[2], d[3]}; break;
        case 6: r = {vk::ImageViewType::e2D, 0, 1, 0, 1}; break;
        case 7: r = {vk::ImageViewType::e2DArray, 0, 1, d[0], d[1]}; break;
        case 8: r = {vk::ImageViewType::e3D, d[0], d[1], 0, 1}; break;
        case 9: r = {vk::ImageViewType::eCube, d[0], d[1], 0, 6}; break;
        case 10: r = {vk::ImageViewType::eCubeArray, d[0], d[1], d[2], d[3] * 6}; break;
        default: return true;
        }
    } else if (dsv) {
        // D3D11_DEPTH_STENCIL_VIEW_DESC: format, dimension, flags, then the range
        switch (dim) {
        case 1: r = {vk::ImageViewType::e1D, d[1], 1, 0, 1}; break;
        case 2: r = {vk::ImageViewType::e1DArray, d[1], 1, d[2], d[3]}; break;
        case 3: r = {vk::ImageViewType::e2D, d[1], 1, 0, 1}; break;
        case 4: r = {vk::ImageViewType::e2DArray, d[1], 1, d[2], d[3]}; break;
        case 5: r = {vk::ImageViewType::e2D, 0, 1, 0, 1}; break;
        case 6: r = {vk::ImageViewType::e2DArray, 0, 1, d[1], d[2]}; break; // api-contract.md V7: after the Flags word (it was read as FirstArraySlice)
        default: r.mips = 1; return true;
        }
        if (ContractCensusOn() && (dim == 5 || dim == 6)) {
            ContractNote("V7 multisample depth view");
        }
    } else {
        // render target and unordered access views: format, dimension, mip slice, then the array or depth slices
        switch (dim) {
        case 1: return false; // buffer
        case 2: r = {vk::ImageViewType::e1D, d[0], 1, 0, 1}; break;
        case 3: r = {vk::ImageViewType::e1DArray, d[0], 1, d[1], d[2]}; break;
        case 4: r = {vk::ImageViewType::e2D, d[0], 1, 0, 1}; break;
        case 5: r = {vk::ImageViewType::e2DArray, d[0], 1, d[1], d[2]}; break;
        case 6: r = {vk::ImageViewType::e2D, 0, 1, 0, 1}; break;
        case 7: r = {vk::ImageViewType::e2DArray, 0, 1, d[0], d[1]}; break;
        case 8: r = {vk::ImageViewType::e3D, d[0], 1, 0, 1, true}; break;
        default: r.mips = 1; return true;
        }
        if (ContractCensusOn() && dim == 8) { // api-contract.md V7 (debug): a 3D render target / UAV's depth slices (FirstWSlice, WSize) against the volume at that mip
            const u32 depth = std::max(img.depth >> std::min<u32>(d[0], 31), 1u);
            const bool whole = d[1] == 0 && (d[2] == ~0u || d[2] >= depth);
            ContractNote(fmt::format("V7 3D {} view: {}", kind == Kind::RenderTargetView ? "render target" : "unordered access", whole ? "whole volume" : "a slice range"));
        }
    }
    // -1 (all remaining) and out of range values are clamped to the image
    bool clamped = false;
    if (r.base_mip >= all_mips) {
        r.base_mip = all_mips - 1, clamped = true;
    }
    if (r.mips == ~0u || r.mips == 0 || r.base_mip + r.mips > all_mips) {
        clamped |= r.mips != ~0u && r.mips != 0;
        r.mips = all_mips - r.base_mip;
    }
    if (img.type == vk::ImageType::e3D) {
        r.base_layer = 0, r.layers = 1;
    } else {
        if (r.base_layer >= all_layers) {
            r.base_layer = all_layers - 1, clamped = true;
        }
        if (r.layers == ~0u || r.layers == 0 || r.base_layer + r.layers > all_layers) {
            clamped |= r.layers != ~0u && r.layers != 0;
            r.layers = all_layers - r.base_layer;
        }
    }
    if ((r.type == vk::ImageViewType::eCube && (!img.cube || r.layers != 6)) || (r.type == vk::ImageViewType::eCubeArray && (!img.cube || r.layers % 6 != 0))) {
        r.type = vk::ImageViewType::e2DArray;
    }
    if (r.type == vk::ImageViewType::e2D && r.layers > 1) {
        r.type = vk::ImageViewType::e2DArray;
    }
    if (img.type == vk::ImageType::e3D && r.type != vk::ImageViewType::e3D) {
        r.type = vk::ImageViewType::e3D;
    }
    if (clamped) {
        c_views_clamped.fetch_add(1, std::memory_order_relaxed);
        if (ContractCensusOn()) { // api-contract.md V6 (debug): what a clamped view asked for against what our image has
            ContractNote(fmt::format("V6 view range clamped: kind {} dim {} desc {} {} {} {} | image {}x{}x{} mips {} layers {} {} -> view mips {}+{} layers {}+{}", static_cast<u32>(kind), dim,
                                     d[0], d[1], d[2], d[3], img.width, img.height, img.depth, all_mips, all_layers, vk::to_string(img.format), r.base_mip, r.mips,
                                     r.base_layer, r.layers));
        }
    }
    return true;
}

/// note 44 test (TLG_VKB_DESTROY_DELAY=<submits>, default 3): how many graphics submits after the engine's delete an object of ours lives on. A large value (60) keeps
/// deleted images and buffers alive for about a second: if the post-load glitches stop, an image or buffer was freed while the GPU still used it
u32 DestroyDelay() {
    static const u32 n = [] {
        const char* e = std::getenv("TLG_VKB_DESTROY_DELAY");
        const long v = e != nullptr ? std::strtol(e, nullptr, 10) : 3;
        return static_cast<u32>(std::clamp(v, 3L, 600L));
    }();
    return n;
}

void DestroyDue(u32 now, bool all) {
    std::vector<Doomed> due;
    {
        std::scoped_lock lock{mutex};
        // V4g: an object goes once the GPU has done the graphics submit after its delete (the last one that may use it); the submits are counted where the engine makes them,
        // our recording thread may still be one behind
        while (!doomed.empty() && (all || (doomed.front().due_submit <= now && GraphicsSubmitDone(doomed.front().due_submit - 2)))) {
            due.push_back(doomed.front());
            doomed.pop_front();
        }
    }
    if (due.empty() || instance == nullptr) {
        return;
    }
    const vk::Device device = instance->GetDevice();
    for (const Doomed& d : due) {
        if (d.view) {
            Indexed::ForgetView(d.view); // R1: its record-mode slots are free again
            device.destroyImageView(d.view);
            c_views_destroyed.fetch_add(1, std::memory_order_relaxed);
        }
        if (d.image) {
            vmaDestroyImage(instance->GetAllocator(), static_cast<VkImage>(d.image), d.allocation);
            c_images_destroyed.fetch_add(1, std::memory_order_relaxed);
        }
        if (d.buffer) {
            Indexed::ForgetBuffer(vk::Buffer{d.buffer}); // R1
            vmaDestroyBuffer(instance->GetAllocator(), d.buffer, d.buffer_allocation);
            c_dev_destroyed.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

/// Everything before in the stream is done and visible before what follows (copies, fills)
void FullBarrier(vk::CommandBuffer cmd) {
    const vk::MemoryBarrier2 barrier{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                     .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands, .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

/// The device buffer holding [address, address + size) (mutex held); null when none; `straddle` when the range starts in one and leaves it
const DeviceBuffer* DeviceBufferOf(u64 address, u64 size, u64& start, bool& straddle) {
    straddle = false;
    auto it = device_buffers.upper_bound(address);
    if (it == device_buffers.begin()) {
        return nullptr;
    }
    --it;
    if (address >= it->second.end) {
        return nullptr;
    }
    if (address + size > it->second.end) {
        straddle = true;
        return nullptr;
    }
    start = it->first;
    return &it->second;
}

/// Kill-experiment TLG_VKB_HOST_CACHE=1: a range read in place gets a copy in the device arena at its first use (in the upload batch); uses from two graphics submits on
/// read the copy. Under the module lock.
struct CachedRange {
    u64 offset;
    u64 queued_at;
};
std::map<std::pair<u64, u64>, CachedRange> host_cache;
u64 host_cache_head = 0;
std::atomic<u64> c_cache_hits{0}, c_cache_copies{0}, c_cache_full{0}, c_cache_bytes{0};

/// V4e check (TLG_VKB_DEVICE_BUFFERS=1, changes nothing on the GPU): every engine buffer read in place, by class: 0 the CPU cannot write it by its description (usage default or
/// immutable, no CPU write access), 1 dynamic or CPU-writable, 2 staging. The class-0 buffers' memory is hashed at creation and watched round robin: a change the hooked writers
/// (UpdateSubresource, our copies and fills) do not explain is counted. Under the module lock.
u32 DeviceBuffersMode() {
    // default 2 since 2026-11-14 (gameplay 25.5 -> 30.7 fps, 0 copies sent back); 1 the check only; 0 off (LEGACY(vkb_device_buffers): every non-UAV buffer read in place)
    static const u32 mode = [] { const char* e = std::getenv("TLG_VKB_DEVICE_BUFFERS"); return e != nullptr && e[0] != 0 ? static_cast<u32>(std::atoi(e)) : 2u; }();
    return mode;
}
struct Watched {
    u64 end, id;
    u32 cls, usage, bind, cpu_access;
    u64 hash;
    bool gpu_written; ///< our copy or fill wrote it (in place: its memory changes legitimately)
    u64 first_bind_submit{}; ///< the graphics submit of its first bind (0: never bound)
    u32 changes_after_bind{};
    u32 settle_changes{}; ///< V4j: changes seen while settling (more than SettleGiveUp: in place for good)
    bool copy_pending{}; ///< issue 111: its first-bind copy is a job made on the engine's thread that our recording side has not run yet
    // V4e (mode 2): its own device copy once its memory has been unchanged for SettleSubmits graphics submits after its first bind
    enum class State : u8 { InPlace, Device, Unstable } state{State::InPlace};
    u64 last_change_submit{}; ///< the submit its memory last changed (or its first bind)
    u64 ready_submit{};       ///< Device: bound from this submit on (the upload batch has copied it)
    VkBuffer device{};
    VmaAllocation allocation{};
};
constexpr u64 SettleSubmits = 8;
constexpr u32 SettleGiveUp = 4;
std::atomic<u64> c_dev2_gave_up{0};
std::vector<u64> settling; ///< addresses of class-0 buffers bound and still in place (mode 2), hashed every submit
std::atomic<u64> c_dev2_made{0}, c_dev2_mb{0}, c_dev2_unstable{0}, c_dev2_binds{0}, c_dev2_bind_mb{0}, c_dev2_failed{0}, c_dev2_settle_changes{0};
std::atomic<u64> c_watch_changed_before_bind{0}, c_watch_changed_after_bind{0}, c_watch_buffers_changed_after_bind{0}, c_watch_buffers_changed_often{0},
    c_watch_changed_within_3{0};
std::map<u64, Watched> watched; ///< by address
std::unordered_map<u64, u64> watched_by_id;
u64 watch_cursor = 0;
std::atomic<u64> c_watch_binds[4], c_watch_bind_bytes[4], c_watch_buffers[3], c_watch_bytes[3], c_watch_mismatch{0}, c_watch_mismatch_gpu{0}, c_watch_cpu_updates{0},
    c_watch_hashed_mb{0}, c_watch_rounds{0};
u64 HashGuest(u64 address, u64 bytes) {
    return XXH3_64bits(reinterpret_cast<const void*>(address), bytes);
}
Watched* WatchedAt(u64 address, u64 size) {
    auto it = watched.upper_bound(address);
    if (it == watched.begin()) {
        return nullptr;
    }
    --it;
    return address >= it->first && address + size <= it->second.end ? &it->second : nullptr;
}
std::vector<std::pair<u64, u64>> gpu_write_pending; ///< destinations of our in-place copies and fills (stream lock held when pushed), marked at the tick
std::mutex gpu_write_mutex;
void NoteGpuWrite(u64 address, u64 size) {
    if (DeviceBuffersMode() == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (Watched* w = WatchedAt(address, size)) {
        w->gpu_written = true;
        if (GpuReadTrap::On()) { // debug (issue 83): its memory is now written by the graphics card
            if (const auto start = watched_by_id.find(w->id); start != watched_by_id.end()) {
                GpuReadTrap::Watch(w->id, start->second, w->end - start->second, GpuReadTrap::Kind::CopyTarget);
            }
        }
    }
}

// R3 census (TLG_VKB_SOURCE_CENSUS=1): where the buffers bound in place live. 0 a class-0 engine buffer (static by its description) not yet copied, 1 class-1 (dynamic
// or CPU-writable), 2 class-2 (staging), 3 the frame allocator's ring, 4 other engine memory
std::atomic<u64> frame_ring_base{0}, frame_ring_size{0};
std::array<std::atomic<u64>, 5> c_src_binds{}, c_src_bytes{};
std::array<std::atomic<u64>, 4> c_src_ring_sizes{}; ///< ring binds by size: < 1 KiB, < 16 KiB, < 256 KiB, larger
bool SourceCensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_SOURCE_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
// per engine buffer read in place: binds, bound bytes, its size, class, usage, bind and CPU access flags (census only)
struct SourceStat {
    u64 binds{}, bytes{}, size{};
    u32 cls{}, usage{}, bind{}, cpu{};
};
std::unordered_map<u64, SourceStat> source_stats; ///< by buffer start (under mutex)
void NoteSource(u64 address, u64 size) {
    u32 k = 4;
    const u64 rb = frame_ring_base.load(std::memory_order_relaxed), rs = frame_ring_size.load(std::memory_order_relaxed);
    if (rb != 0 && address >= rb && address + size <= rb + rs) {
        k = 3;
        c_src_ring_sizes[size < 1024 ? 0 : size < (16u << 10) ? 1 : size < (256u << 10) ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
    } else {
        std::scoped_lock lock{mutex};
        if (const Watched* w = WatchedAt(address, size)) {
            k = std::min<u32>(w->cls, 2);
            const u64 start = std::prev(watched.upper_bound(address))->first;
            SourceStat& st = source_stats[start];
            ++st.binds;
            st.bytes += size;
            st.size = w->end - start, st.cls = w->cls, st.usage = w->usage, st.bind = w->bind, st.cpu = w->cpu_access;
        }
    }
    c_src_binds[k].fetch_add(1, std::memory_order_relaxed);
    c_src_bytes[k].fetch_add(size, std::memory_order_relaxed);
}

// R3a: the frame ring in our device memory (set once; read without a lock)
std::atomic<u64> vram_ring_base{0}, vram_ring_end{0};
VkBuffer vram_ring_buffer{};
VmaAllocation vram_ring_allocation{};
std::atomic<u64> c_ring_vram_binds{0}, c_ring_vram_straddle{0};
std::atomic<u64> host_side_ring_base{0}, host_side_ring_size{0};
constexpr u64 VramRingSlack = 16ull << 20; ///< bound ranges may reach past the ring's end (the guest memory behind it was readable): our block is that much larger
inline bool InVramRing(u64 address, u64 size, BufferWindow& out) {
    const u64 base = vram_ring_base.load(std::memory_order_acquire);
    if (base == 0 || address < base || address >= vram_ring_end.load(std::memory_order_relaxed)) {
        return false;
    }
    if (address + size > vram_ring_end.load(std::memory_order_relaxed)) {
        c_ring_vram_straddle.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    out = {vk::Buffer{vram_ring_buffer}, address - base, true};
    return true;
}

/// R3d (TLG_VKB_STATIC_AT_BIND, default on; =0: settle by hash, V4e): a static engine buffer (usage default or immutable, no CPU write access) gets its device copy at its
/// first bind, copied in the frame stream in front of that bind. The engine's data is in it by then (the V4e check: 0 changes after a first bind); a later
/// UpdateSubresource sends it back in place (BufferCpuWritten). Deletes the settling by hash and the watch.
bool StaticAtBindOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_STATIC_AT_BIND"); return e == nullptr || e[0] != '0'; }();
    return on;
}
std::atomic<u64> c_static_at_bind{0}, c_static_at_bind_mb{0}, c_static_at_bind_failed{0}, c_static_at_bind_zero{0};
/// debug (TLG_VKB_R3D_CHECK=1): a buffer copied at its first bind is hashed then, and the watch runs (all device copies, 16 MiB a submit): a change of its
/// engine memory after the copy (a CPU write the engine makes without UpdateSubresource) is logged with the buffer and sends it back in place
/// debug: TLG_VKB_R3D_SKIP=<bytes>,<bytes>,... buffers of those sizes never get a first-bind copy (they settle by hash as with =0); TLG_VKB_R3D_LOG=<file>
/// writes one line per first-bind copy: graphics submit, address, bytes, usage, bind, cpu access
/// Issue 111 (TLG_VKB_FIRST_BIND_RECORDER, default on; =0 LEGACY(vkb_first_bind_recorder): such a bind settles by hash, read in place and staged per submission for
/// 10 submits or more): a first bind found on our recording side (a draw's late buffers) makes the device copy as one found on the engine's thread does; the copy is
/// recorded in the stream in front of the draw that binds it (Stream::CopyInStream). A device copy made on the engine's thread is bound on the recording side only
/// once its copy job has run there (draws before it in the stream read in place).
bool FirstBindOnRecorderOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_FIRST_BIND_RECORDER"); return e == nullptr || e[0] != '0'; }();
    return on;
}
std::atomic<u64> c_fbr_copies{0}, c_fbr_mb{0}, c_fbr_pending_in_place{0};
bool R3dSkipped(u64 bytes) {
    static const std::vector<u64> sizes = [] {
        std::vector<u64> out;
        if (const char* e = std::getenv("TLG_VKB_R3D_SKIP")) {
            for (const char* p = e; *p != 0;) {
                char* end = nullptr;
                const u64 v = std::strtoull(p, &end, 10);
                if (end == p) {
                    break;
                }
                out.push_back(v);
                p = *end == ',' ? end + 1 : end;
            }
        }
        return out;
    }();
    return std::find(sizes.begin(), sizes.end(), bytes) != sizes.end();
}
void R3dLog(u64 submit, u64 address, u64 bytes, u32 usage, u32 bind, u32 cpu_access) {
    static FILE* f = [] { const char* e = std::getenv("TLG_VKB_R3D_LOG"); return e != nullptr && e[0] != 0 ? std::fopen(e, "w") : nullptr; }();
    if (f != nullptr) {
        std::fprintf(f, "%llu %#llx %llu usage %u bind %#x cpu %#x\n", static_cast<unsigned long long>(submit), static_cast<unsigned long long>(address),
                     static_cast<unsigned long long>(bytes), usage, bind, cpu_access);
        std::fflush(f);
    }
}
/// test (TLG_VKB_R3D_FIRST_IN_PLACE=1): the bind that makes a first-bind copy reads the engine's memory in place; the copy serves from two submits on
bool R3dFirstInPlace() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_R3D_FIRST_IN_PLACE"); return e != nullptr && e[0] == '1'; }();
    return on;
}
/// debug (TLG_VKB_R3D_TRACE=<bytes>): every write that reaches a watched buffer of that size (our GPU copy into it, an UpdateSubresource) is logged with its
/// state (in place / device copy) and graphics submit
u64 R3dTraceBytes() {
    static const u64 bytes = [] { const char* e = std::getenv("TLG_VKB_R3D_TRACE"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull; }();
    return bytes;
}
bool R3dCheckOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_R3D_CHECK"); return e != nullptr && e[0] == '1'; }();
    return on;
}
/// debug (with TLG_VKB_R3D_TRACE=<bytes>): a lookup of a buffer of that size, by path; logged at its 1st, 2nd, 10th, 100th, 1000th lookup of that buffer and path
extern thread_local u64 t_census_tag;
std::vector<std::pair<u64, u64>> r3d_deleted; ///< debug trace: deleted buffers of the traced size (start, end), under the module lock
void R3dTracePath(u64 start, const char* path, u64 address, u64 size, u64 now) {
    static std::mutex m;
    static std::map<std::pair<u64, const char*>, u64> counts;
    u64 n = 0;
    {
        std::scoped_lock lock{m};
        n = ++counts[{start, path}];
    }
    if (n == 1 || n == 2 || n == 10 || n == 100 || n == 1000 || n == 10000) {
        LOG_WARNING(Render_Vulkan, "vkb R3d trace: lookup #{} of buffer {:#x} ({}) range +{:#x} {} bytes, recorder {}, submit {}, tag {:#x}", n, start, path, address - start, size,
                    Stream::OnRecorder(), now, t_census_tag);
    }
}
VkBuffer MakeDeviceBuffer(u64 bytes, VmaAllocation& allocation) {
    const VkBufferCreateInfo ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = bytes,
                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkBuffer handle{};
    if (vmaCreateBuffer(instance->GetAllocator(), &ci, &alloc, &handle, &allocation, nullptr) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return handle;
}

/// issue 111: the resolver has reached the first-bind copy job of the buffer at `address` (its Prepare, no stream lock; or its Run: the stream lock, then this
/// module's lock): from here on in the stream its device copy holds the data, so binds found on the recording side may take it
void CopyJobReached(u64 address, VkBuffer device) {
    if (!FirstBindOnRecorderOn()) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (const auto it = watched.find(address); it != watched.end() && it->second.device == device && it->second.copy_pending) {
        it->second.copy_pending = false;
        BumpBufferEpoch(); // note 46: A1 memo answers may change (an in-place answer of the recording side)
    }
}

/// A window without the stream lock order problem: called with the stream recording held (stream lock, then this module's lock)
bool FindLocked(u64 address, u64 size, BufferWindow& out) {
    struct InitialCopy {
        u64 address{}, bytes{};
        VkBuffer device{};
        bool on_recorder{}; ///< issue 111: found on our recording side: the copy goes in front of the draw (Stream::CopyInStream)
    } initial;
    if (InVramRing(address, size, out)) {
        c_ring_vram_binds.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    {
        std::scoped_lock lock{mutex};
        u64 start = 0;
        bool straddle = false;
        if (const DeviceBuffer* d = DeviceBufferOf(address, size, start, straddle)) {
            out = {vk::Buffer{d->buffer}, address - start, true};
            return true;
        }
        if (straddle) {
            c_find_straddle.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (DeviceBuffersMode() == 2) {
            // V4j: the device copy (V4e), in the same lock; the in-place lookup is not needed for it
            if (R3dTraceBytes() != 0) {
                for (const auto& [a, e] : r3d_deleted) {
                    if (address < e && a < address + size) {
                        R3dTracePath(a, "deleted", address, size, GraphicsSubmits());
                    }
                }
            }
            auto it = watched.upper_bound(address);
            if (it != watched.begin()) {
                --it;
                Watched& wb = it->second;
                if (address >= it->first && address + size <= wb.end && wb.cls == 0) {
                    const u64 now = GraphicsSubmits();
                    if (R3dTraceBytes() != 0 && wb.end - it->first == R3dTraceBytes()) {
                        R3dTracePath(it->first, wb.first_bind_submit == 0 ? "first" : wb.state == Watched::State::Device ? (now >= wb.ready_submit ? "device" : "device not ready") : wb.state == Watched::State::Unstable ? "unstable" : "settling", address, size, now);
                    }
                    const bool on_recorder = Stream::OnRecorder() && Stream::RecorderOn();
                    const bool fbr = FirstBindOnRecorderOn();
                    if (wb.first_bind_submit == 0 && StaticAtBindOn() && !wb.gpu_written && (!on_recorder || fbr) && !R3dSkipped(wb.end - it->first)) {
                        // R3d: its device copy now; the copy job goes into the stream below, after the lock, in front of the job of this bind
                        wb.first_bind_submit = now + 1;
                        const u64 bytes = wb.end - it->first;
                        R3dLog(now, it->first, bytes, wb.usage, wb.bind, wb.cpu_access);
                        if (bytes == R3dTraceBytes()) {
                            LOG_WARNING(Render_Vulkan, "vkb R3d trace: first-bind copy of {:#x} ({} bytes) at submit {}", it->first, bytes, now);
                        }
                        if (R3dCheckOn()) {
                            wb.hash = HashGuest(it->first, bytes); // debug: the bytes the device copy gets
                        }
                        wb.device = MakeDeviceBuffer(bytes, wb.allocation);
                        BumpBufferEpoch(); // note 46: A1 memo answers may change
                        if (wb.device == VK_NULL_HANDLE) {
                            wb.state = Watched::State::Unstable;
                            BumpBufferEpoch(); // note 46: A1 memo answers may change
                            c_static_at_bind_failed.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            wb.state = Watched::State::Device;
                            BumpBufferEpoch(); // note 46: A1 memo answers may change
                            static const bool via_batch = [] { const char* e = std::getenv("TLG_VKB_R3D_VIA_BATCH"); return e != nullptr && e[0] == '1'; }(); // (=2: bound at once)
                            wb.ready_submit = !on_recorder && (R3dFirstInPlace() || via_batch) ? now + 2 : 0; // (the test modes: the engine's thread only)
                            initial = {it->first, bytes, wb.device, on_recorder};
                            wb.copy_pending = fbr && !on_recorder && !via_batch; // issue 111: until its copy job runs on the recording side (test mode via the batch: none)
                            out = {vk::Buffer{wb.device}, address - it->first, true};
                            c_static_at_bind.fetch_add(1, std::memory_order_relaxed);
                            c_static_at_bind_mb.fetch_add(bytes, std::memory_order_relaxed);
                            c_dev2_mb.fetch_add(bytes, std::memory_order_relaxed);
                        }
                    } else if (wb.first_bind_submit == 0) {
                        wb.first_bind_submit = now + 1;
                        wb.last_change_submit = now;
                        settling.push_back(it->first);
                    } else if (wb.state == Watched::State::Device && wb.copy_pending && on_recorder) {
                        c_fbr_pending_in_place.fetch_add(1, std::memory_order_relaxed); // issue 111: a draw before its copy job in the stream: in place (below)
                    } else if (wb.state == Watched::State::Device && now >= wb.ready_submit) {
                        out = {vk::Buffer{wb.device}, address - it->first, true};
                        c_dev2_binds.fetch_add(1, std::memory_order_relaxed);
                        return true;
                    }
                }
            }
        }
    }
    if (initial.device != VK_NULL_HANDLE) {
        const BufferWindow src = SourceWindow(initial.address, initial.bytes);
        if (!src.buffer) {
            c_static_at_bind_zero.fetch_add(1, std::memory_order_relaxed); // no source window: the device copy is filled with zeros
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_WARNING(Render_Vulkan, "vkb R3d: first-bind copy of {:#x} ({} bytes) found no source window: filled with zeros", initial.address, initial.bytes);
            }
        }
        const VkBuffer device = initial.device;
        const u64 bytes = initial.bytes;
        if (initial.on_recorder) {
            // issue 111: in the stream in front of the draw whose lookup this is (its Prepare on the resolver, or its Run on the single recorder)
            Stream::NoteBytes(Stream::Work::Upload, bytes);
            Stream::CopyInStream(src.buffer, src.offset, vk::Buffer{device}, bytes);
            c_fbr_copies.fetch_add(1, std::memory_order_relaxed);
            c_fbr_mb.fetch_add(bytes, std::memory_order_relaxed);
            return true;
        }
        // test: TLG_VKB_R3D_VIA_BATCH=1: the copy in the upload batch (as settling), this bind and the next ones until ready_submit read in place; =2: the copy in the
        // upload batch (in front of this submission), the device copy bound at once
        static const char via_batch = [] { const char* e = std::getenv("TLG_VKB_R3D_VIA_BATCH"); return e != nullptr ? e[0] : '0'; }();
        if (via_batch != '0' && src.buffer) {
            Upload::CopyBuffer(src.buffer, src.offset, vk::Buffer{device}, 0, bytes);
            if (via_batch == '2') {
                return true;
            }
            out = {};
            goto in_place;
        }
        Stream::NoteBytes(Stream::Work::Upload, bytes);
        static const bool dump_on = [] { const char* e = std::getenv("TLG_VKB_R3D_DUMP"); return e != nullptr && e[0] == '1'; }();
        const bool dump = dump_on && R3dTraceBytes() == bytes; // debug R3d trace (with TLG_VKB_R3D_DUMP=1): the host bytes now, and the copy and its source right after the copy on the GPU
        const u64 dump_address = initial.address, dump_submit = GraphicsSubmits();
        if (dump) {
            const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_image_dump" / "r3d";
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (FILE* f = std::fopen((dir / fmt::format("{:x}_s{}_cpu_at_bind.bin", dump_address, dump_submit)).string().c_str(), "wb")) {
                std::fwrite(reinterpret_cast<const void*>(dump_address), 1, bytes, f);
                std::fclose(f);
            }
        }
        // issue 111: the resolver reaches it in stream order (Prepare; with the single recorder its Run): binds found on the recording side may take the copy from there on
        Stream::DeferPrepared(Stream::Work::Upload, [dump_address, device] { CopyJobReached(dump_address, device); },
                              [src, device, bytes, dump, dump_address, dump_submit](vk::CommandBuffer cmd) {
            CopyJobReached(dump_address, device);
            if (src.buffer) {
                Stream::Hazard(cmd, {Stream::BufferUse(src.buffer, src.offset, bytes, false), Stream::BufferUse(vk::Buffer{device}, 0, bytes, true)});
                cmd.copyBuffer(src.buffer, vk::Buffer{device}, vk::BufferCopy{src.offset, 0, bytes});
                static const bool barrier_after = [] { const char* e = std::getenv("TLG_VKB_R3D_BARRIER_AFTER"); return e != nullptr && e[0] == '1'; }(); // test
                if (barrier_after) {
                    const vk::MemoryBarrier2 full{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                                  .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                                  .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
                    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &full});
                }
                if (dump) {
                    DumpBuffer(cmd, vk::Buffer{device}, 0, bytes, fmt::format("r3d/{:x}_s{}_dev_after_copy", dump_address, dump_submit));
                    DumpBuffer(cmd, src.buffer, src.offset, bytes, fmt::format("r3d/{:x}_s{}_src_after_copy", dump_address, dump_submit));
                }
            } else {
                Stream::Hazard(cmd, {Stream::BufferUse(vk::Buffer{device}, 0, bytes, true)});
                cmd.fillBuffer(vk::Buffer{device}, 0, VK_WHOLE_SIZE, 0);
            }
        });
        if (!R3dFirstInPlace()) {
            return true;
        }
        out = {}; // this bind reads in place (below)
    }
in_place:
    if (host_import == nullptr) {
        return false;
    }
    const VkbVulkan::HostImport::Window w = host_import->Find(address, size);
    if (!w.buffer) {
        c_find_none.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    out = {w.buffer, w.offset, false};
    if (SourceCensusOn()) {
        NoteSource(address, size);
    }
    if (DeviceBuffersMode() == 2) {
        return true;
    }
    if (DeviceBuffersMode() == 1) {
        std::scoped_lock lock{mutex};
        Watched* wb = WatchedAt(address, size);
        const u32 cls = wb != nullptr ? wb->cls : 3;
        if (wb != nullptr && wb->first_bind_submit == 0) {
            wb->first_bind_submit = GraphicsSubmits() + 1;
        }
        c_watch_binds[cls].fetch_add(1, std::memory_order_relaxed);
        c_watch_bind_bytes[cls].fetch_add(size, std::memory_order_relaxed);
    }
    if (const vk::Buffer arena = Stream::HostCacheArena(); arena) {
        std::scoped_lock lock{mutex};
        const u64 now = GraphicsSubmits();
        const auto key = std::make_pair(address, size);
        if (const auto it = host_cache.find(key); it != host_cache.end()) {
            if (now >= it->second.queued_at + 2) {
                out = {arena, it->second.offset, false};
                c_cache_hits.fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        }
        const u64 low = w.offset & 0xff;
        const u64 at = ((host_cache_head + 255) & ~255ull) + low;
        if (at + size > Stream::HostCacheBytes) {
            c_cache_full.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        host_cache_head = at + size;
        host_cache.emplace(key, CachedRange{at, now});
        Upload::CopyBuffer(w.buffer, w.offset, arena, at, size);
        c_cache_copies.fetch_add(1, std::memory_order_relaxed);
        c_cache_bytes.fetch_add(size, std::memory_order_relaxed);
        return true;
    }
    if (const vk::Buffer scratch = Stream::HostScratch(); scratch && size + 256 <= Stream::HostScratchBytes) {
        out = {scratch, w.offset & 0xff, false}; // kill-experiment TLG_VKB_HOST_SCRATCH=1: the same alignment, zeroed device memory
    }
    return true;
}

} // namespace

/// V2b: a texture that holds data (not a render or depth target, not dynamic or staging) gets the engine's initial contents, uploaded once
constexpr u32 UsageDynamic = 2, UsageStaging = 3;
std::atomic<u64> c_upload_wanted{0}, c_upload_done{0};
void FillReadersCheck(u64 address, u64 size, int what); // note 49/51 census, below
bool UploadContents(u64 id, bool fresh) {
    vk::Image image{};
    vk::Format format{};
    u32 mips = 0, layers = 0;
    vk::ImageLayout layout{};
    {
        std::scoped_lock lock{mutex};
        const auto it = images.find(id);
        if (it == images.end() || !fresh) {
            return false;
        }
        image = it->second.image, format = it->second.format, mips = it->second.mips, layers = it->second.layers;
        layout = it->second.layout;
        if (FillReadersOn() && it->second.address != 0) { // note 51 census: a texture whose contents come from memory a fill wrote
            CensusTag(id);
            FillReadersCheck(it->second.address, std::max<u64>(it->second.bytes, 1), 2);
        }
    }
    c_upload_wanted.fetch_add(1, std::memory_order_relaxed);
    if (Upload::TextureContents(id, image, format, mips, layers, layout)) {
        c_upload_done.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

void TextureCreated(const Object& texture) {
    if (!On() || !Ready()) {
        return;
    }
    if (GpuReadTrap::On()) { // debug (issue 83, mode 2): a texture the graphics card draws or writes into is watched from its creation on
        if ((texture.bind & (BindRenderTarget | BindDepthStencil | BindUnorderedAccess)) != 0) {
            GpuReadTrap::Watch(texture.id, texture.address, texture.bytes, GpuReadTrap::Kind::Texture);
        } else {
            GpuReadTrap::CpuObjectMade(texture.address, texture.bytes);
        }
    }
    bool fresh = false;
    NewImage fresh_image{};
    {
        std::scoped_lock lock{mutex};
        const bool known = images.contains(texture.id);
        Image* img = MakeImage(texture);
        if (img != nullptr && (texture.bind & BindRenderTarget) != 0 && texture.address != 0) {
            image_at_address[texture.address] = texture.id;
        }
        fresh = img != nullptr && !known && !img->is_depth && (texture.bind & (BindRenderTarget | BindDepthStencil)) == 0 && texture.usage != UsageDynamic &&
                texture.usage != UsageStaging;
        if (fresh && !t_new_images.empty() && t_new_images.back().image == img->image) {
            fresh_image = t_new_images.back();
            t_new_images.pop_back(); // note 66: its upload leaves it in GENERAL
        }
    }
    if (!UploadContents(texture.id, fresh) && fresh_image.image) {
        t_new_images.push_back(fresh_image); // no contents after all: the layout job
    } else if (fresh_image.image) {
        c_layout_by_upload.fetch_add(1, std::memory_order_relaxed);
    }
    InitNewImages();
}

/// Tint hunt census (2026-11-16): the engine's own T# of each texture view (its PS4 data format, number format and channel selects) next to the
/// Vulkan format our view reads, grouped. Our views use the identity channel mapping; a T# whose channel selects differ from what that format gives
/// is read with the wrong channels. Under the view lock.
struct ChannelGroup {
    u64 views = 0;
    u64 example_view = 0, example_texture = 0;
    u32 width = 0, height = 0;
};
std::map<std::string, ChannelGroup> channel_groups;
void NoteViewChannels(const Object& view, const Object& resource, vk::Format format) {
    static constexpr char Sel[8] = {'0', '1', '?', '?', 'R', 'G', 'B', 'A'};
    const u32 w1 = view.tsharp[1], w3 = view.tsharp[3];
    const u32 dfmt = (w1 >> 20) & 0x3f, nfmt = (w1 >> 26) & 0xf;
    const std::string sel{Sel[w3 & 7], Sel[(w3 >> 3) & 7], Sel[(w3 >> 6) & 7], Sel[(w3 >> 9) & 7]};
    const std::string key = fmt::format("{} dxgi {} -> {} | T# dfmt {} nfmt {} sel {}", view.kind == Kind::UnorderedAccessView ? "uav" : "srv",
                                        view.has_desc ? view.desc[0] : resource.format, vk::to_string(format), dfmt, nfmt, sel);
    ChannelGroup& g = channel_groups[key];
    if (g.views++ == 0) {
        g.example_view = view.id, g.example_texture = resource.id, g.width = resource.width, g.height = resource.height;
    }
}

/// api-contract.md R2 / R3 / R5 / R8 (2026-11-18): the channel selects of the engine's own T# of a sampled view (word 3: x y z w, 3 bits each; 0 zero, 1 one,
/// 4..7 memory channels X Y Z W, from the engine format table `Js_InitFormatTable` 0248f390) as a Vulkan component mapping on the view's format. X is the
/// channel in the lowest bits / first byte: Vulkan's R, except in the BGR-ordered formats where it is B. Live case: R32_UINT (39) selects R000, so alpha
/// reads 0 where Vulkan gives 1; latent ones: A8 (65, 000R), the X8 formats (88, 92, 93, BGR1). Identity when the selects give what the format gives anyway.
vk::ComponentMapping ViewComponents(vk::Format format, u32 tsharp3, bool& changed) {
    using S = vk::ComponentSwizzle;
    const bool bgr = format == vk::Format::eB8G8R8A8Unorm || format == vk::Format::eB8G8R8A8Srgb || format == vk::Format::eR5G6B5UnormPack16 ||
                     format == vk::Format::eA1R5G5B5UnormPack16;
    const S channel[4] = {bgr ? S::eB : S::eR, S::eG, bgr ? S::eR : S::eB, S::eA};
    // what Vulkan returns for each component without a swizzle: a channel the format has, else 0 (x, y, z) or 1 (w)
    const u32 present = vk::componentCount(format);
    const auto natural = [&](u32 c) { return c < present ? (c == 0 ? S::eR : c == 1 ? S::eG : c == 2 ? S::eB : S::eA) : (c == 3 ? S::eOne : S::eZero); };
    S out[4];
    changed = false;
    for (u32 c = 0; c < 4; ++c) {
        const u32 sel = (tsharp3 >> (c * 3)) & 7;
        out[c] = sel == 0 ? S::eZero : sel == 1 ? S::eOne : sel >= 4 ? channel[sel - 4] : natural(c);
        if (out[c] != natural(c)) {
            changed = true;
        }
    }
    return changed ? vk::ComponentMapping{out[0], out[1], out[2], out[3]} : vk::ComponentMapping{};
}
std::atomic<u64> c_views_swizzled{0};

void ViewCreated(const Object& view, const Object& resource) {
    if (!On() || !Ready()) {
        return;
    }
    struct AfterLock {
        ~AfterLock() { InitNewImages(); } // note 66: (declared before the lock: runs after it is released)
    } after_lock;
    if (resource.kind == Kind::Buffer) {
        c_views_buffer.fetch_add(1, std::memory_order_relaxed); // a buffer view: (chunk buffer, offset) at the bind, no object
        return;
    }
    std::scoped_lock lock{mutex};
    if (views.contains(view.id)) {
        return;
    }
    Image* img = MakeImage(resource);
    if (img == nullptr) {
        c_views_no_image.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if ((resource.bind & BindRenderTarget) != 0 && resource.address != 0 && !image_at_address.contains(resource.address)) {
        image_at_address[resource.address] = resource.id;
    }
    Range r;
    if (!ViewRange(view.kind, view, *img, r)) {
        c_views_buffer.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (r.type == vk::ImageViewType::e1D || r.type == vk::ImageViewType::e1DArray) { // note 41: 1D is 2D of height 1 here (above)
        r.type = r.type == vk::ImageViewType::e1D ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
    }
    vk::Format format = img->format;
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
    if (img->is_depth) {
        const u32 view_dxgi = view.has_desc ? view.desc[0] : 0;
        if (view.kind == Kind::DepthStencilView) {
            aspect = HasStencil(img->format) ? (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil) : vk::ImageAspectFlagBits::eDepth;
        } else {
            aspect = IsStencilView(view_dxgi) && HasStencil(img->format) ? vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlagBits::eDepth;
        }
    } else if (view.has_desc && view.desc[0] != 0) {
        if (const vk::Format f = ColorFormat(view.desc[0]); f != vk::Format::eUndefined) {
            format = f;
        }
    }
    if (!img->mutable_format && format != img->format) {
        format = img->format;
        c_fixed_view_format_forced.fetch_add(1, std::memory_order_relaxed);
    }
    if (ContractCensusOn()) {
        ContractNote(fmt::format("V view kind {} dxgi {} of texture dxgi {} -> {}", static_cast<u32>(view.kind), view.has_desc ? static_cast<s64>(view.desc[0]) : -1,
                                 resource.format, vk::to_string(format)));
    }
    // R8: a sampled colour view reads the channels the engine's T# selects (storage views and attachments must keep the identity; they are other view kinds).
    // LEGACY(TLG_LEGACY_VIEW_IDENTITY): =1 every view with the identity mapping.
    static const bool legacy_identity = [] { const char* e = std::getenv("TLG_LEGACY_VIEW_IDENTITY"); return e != nullptr && e[0] == '1'; }();
    vk::ComponentMapping components{};
    if (!legacy_identity && view.kind == Kind::ShaderResourceView && view.has_tsharp && !img->is_depth) {
        bool changed = false;
        components = ViewComponents(format, view.tsharp[3], changed);
        if (changed) {
            c_views_swizzled.fetch_add(1, std::memory_order_relaxed);
        }
    }
    const vk::ImageViewCreateInfo ci{
        .image = img->image,
        .viewType = r.type,
        .format = format,
        .components = components,
        .subresourceRange = {aspect, r.base_mip, r.mips, r.base_layer, r.layers},
    };
    const auto [result, handle] = instance->GetDevice().createImageView(ci);
    if (result != vk::Result::eSuccess) {
        c_views_failed.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 16) {
            LOG_ERROR(Render_Vulkan, "vkb: view id {} of texture id {} failed: {}", view.id, resource.id, vk::to_string(result));
        }
        return;
    }
    views.emplace(view.id, View{handle, resource.id, r.type, format, ci.subresourceRange, {}, {}, components});
    c_views_created.fetch_add(1, std::memory_order_relaxed);
    if (view.has_tsharp && !img->is_depth) {
        NoteViewChannels(view, resource, format);
    }
}

bool DynamicDeviceOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_DYNAMIC_DEVICE"); return e == nullptr || e[0] != '0'; }(); // default on since 2026-11-14 (note 29); =0: in place
    return on && On();
}

thread_local u32 t_cpu_write_kind = 2; ///< upload census: 0 Unmap (dynamic), 1 UpdateSubresource, 2 other (moves, counters, ...)
std::atomic<u64> c_dyn_device{0}, c_dyn_unmaps{0}, c_dyn_unmap_mb{0}, c_dyn_checks{0}, c_dyn_changed_outside{0};
bool UploadCensusOn();
std::mutex page_census_mutex;
std::unordered_map<u64, std::vector<u64>> page_hashes; ///< census: per big dynamic buffer, its 4 KiB pages' hashes at the last Unmap
std::atomic<u64> c_pc_unmaps{0}, c_pc_pages{0}, c_pc_changed{0}, c_pc_first_changed_sum{0}, c_pc_last_changed_sum{0};
void PageCensus(u64 memory, u64 bytes) {
    if (!UploadCensusOn() || bytes < (1u << 20)) {
        return;
    }
    const u64 pages = bytes >> 12;
    std::scoped_lock lock{page_census_mutex};
    auto& h = page_hashes[memory];
    const bool first = h.size() != pages;
    h.resize(pages);
    u64 changed = 0, first_changed = pages, last_changed = 0;
    for (u64 i = 0; i < pages; ++i) {
        const u64* q = reinterpret_cast<const u64*>(memory + (i << 12));
        u64 x = 0xcbf29ce484222325ull;
        for (u32 k = 0; k < 512; ++k) { // every qword of the page
            x = (x ^ q[k]) * 0x100000001b3ull;
        }
        if (first || x != h[i]) {
            ++changed;
            first_changed = std::min(first_changed, i);
            last_changed = i;
        }
        h[i] = x;
    }
    if (!first) {
        c_pc_unmaps.fetch_add(1, std::memory_order_relaxed);
        c_pc_pages.fetch_add(pages, std::memory_order_relaxed);
        c_pc_changed.fetch_add(changed, std::memory_order_relaxed);
        c_pc_first_changed_sum.fetch_add(first_changed, std::memory_order_relaxed);
        c_pc_last_changed_sum.fetch_add(last_changed, std::memory_order_relaxed);
    }
}
/// U2 (TLG_VKB_DIRTY_PAGES=1, off while tested): a big dynamic buffer the engine maps every frame and barely changes (heaviest save: two of ~7 MB, 0.05 of 1,750 pages
/// changed per Unmap) is copied to our device copy page by page, only the pages written since its last Unmap. After an Unmap its pages are read only; the first write to
/// a page (any thread, any code) faults once: the handler marks the page and makes it writable. A region not unmapped for a while is given back (writable, forgotten).
#ifdef _WIN32
namespace Dirty {
constexpr u32 MaxRegions = 16;
constexpr u64 MinBytes = 1ull << 20;
constexpr u64 Page = 4096;
struct Region {
    std::atomic<u64> lo{0}, hi{0}; ///< the page span (page aligned) the handler answers for; 0: free
    u64 memory{}, bytes{};
    std::unique_ptr<std::atomic<u8>[]> dirty;
    u64 pages{};
    u64 last_unmap{};
};
std::array<Region, MaxRegions> regions;
std::mutex m; // Unmaps (the handler is lock free)
std::atomic<bool> handler_added{false};
std::atomic<u64> c_faults{0}, c_unmaps{0}, c_full{0}, c_pages_copied{0}, c_bytes_skipped{0}, c_released{0}, c_refused{0};
u64 unmap_clock = 0;
bool On() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_DIRTY_PAGES"); return e != nullptr && e[0] == '1'; }();
    return on;
}
LONG CALLBACK Handler(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD* r = info->ExceptionRecord;
    if (r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r->NumberParameters < 2 || r->ExceptionInformation[0] != 1) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const u64 address = r->ExceptionInformation[1];
    for (Region& g : regions) {
        const u64 lo = g.lo.load(std::memory_order_acquire);
        if (lo == 0 || address < lo || address >= g.hi.load(std::memory_order_relaxed)) {
            continue;
        }
        const u64 page = (address - lo) / Page;
        g.dirty[page].store(1, std::memory_order_release);
        DWORD old = 0;
        VirtualProtect(reinterpret_cast<void*>(lo + page * Page), Page, PAGE_READWRITE, &old);
        c_faults.fetch_add(1, std::memory_order_relaxed);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
void Protect(u64 lo, u64 pages) {
    DWORD old = 0;
    VirtualProtect(reinterpret_cast<void*>(lo), pages * Page, PAGE_READONLY, &old);
}
void Release(Region& g) {
    const u64 lo = g.lo.load(std::memory_order_relaxed);
    g.lo.store(0, std::memory_order_release);
    if (lo != 0) {
        DWORD old = 0;
        VirtualProtect(reinterpret_cast<void*>(lo), g.pages * Page, PAGE_READWRITE, &old);
    }
    c_released.fetch_add(1, std::memory_order_relaxed);
}
/// True: handled (the dirty pages were sent); false: the caller copies the whole range
bool Unmapped(u64 memory, u64 bytes) {
    if (!On() || bytes < MinBytes) {
        return false;
    }
    if (!handler_added.exchange(true)) {
        AddVectoredExceptionHandler(1, Handler);
    }
    std::scoped_lock lock{m};
    ++unmap_clock;
    const u64 lo = memory & ~(Page - 1), hi = (memory + bytes + Page - 1) & ~(Page - 1);
    Region* found = nullptr;
    Region* free_slot = nullptr;
    for (Region& g : regions) {
        const u64 glo = g.lo.load(std::memory_order_relaxed);
        if (glo == 0) {
            free_slot = free_slot ? free_slot : &g;
            continue;
        }
        if (g.memory == memory && g.bytes == bytes) {
            found = &g;
        } else if (lo < g.hi.load(std::memory_order_relaxed) && glo < hi) {
            Release(g); // another range over its pages: forget it (whole copies until it settles again)
            free_slot = free_slot ? free_slot : &g;
        } else if (unmap_clock - g.last_unmap > 4096) {
            Release(g); // not unmapped for a long time: give its pages back
            free_slot = free_slot ? free_slot : &g;
        }
    }
    c_unmaps.fetch_add(1, std::memory_order_relaxed);
    if (found == nullptr) {
        if (free_slot == nullptr) {
            c_refused.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        // first time: the whole range goes now, then the pages are watched
        Region& g = *free_slot;
        g.memory = memory, g.bytes = bytes, g.pages = (hi - lo) / Page;
        g.dirty = std::make_unique<std::atomic<u8>[]>(g.pages);
        for (u64 i = 0; i < g.pages; ++i) {
            g.dirty[i].store(0, std::memory_order_relaxed);
        }
        g.hi.store(hi, std::memory_order_relaxed);
        g.lo.store(lo, std::memory_order_release);
        g.last_unmap = unmap_clock;
        Protect(lo, g.pages);
        c_full.fetch_add(1, std::memory_order_relaxed);
        BufferCpuWritten(memory, bytes);
        return true;
    }
    Region& g = *found;
    g.last_unmap = unmap_clock;
    u64 copied = 0;
    for (u64 i = 0; i < g.pages;) {
        if (g.dirty[i].load(std::memory_order_acquire) == 0) {
            ++i;
            continue;
        }
        u64 j = i;
        while (j < g.pages && g.dirty[j].load(std::memory_order_acquire) != 0) {
            g.dirty[j].store(0, std::memory_order_release); // cleared before the protection: a write after it faults and marks the page again
            ++j;
        }
        Protect(lo + i * Page, j - i);
        const u64 a = std::max(memory, lo + i * Page), b = std::min(memory + bytes, lo + j * Page);
        if (b > a) {
            BufferCpuWritten(a, b - a); // (the copy reads the memory when it runs: a write before then is in it)
            copied += b - a;
        }
        c_pages_copied.fetch_add(j - i, std::memory_order_relaxed);
        i = j;
    }
    c_bytes_skipped.fetch_add(bytes - copied, std::memory_order_relaxed);
    return true;
}
nlohmann::json Report() {
    return {{"on", On()}, {"faults", c_faults.load()}, {"unmaps", c_unmaps.load()}, {"first_full", c_full.load()}, {"pages_copied", c_pages_copied.load()},
            {"mb_skipped", c_bytes_skipped.load() >> 20}, {"released", c_released.load()}, {"refused", c_refused.load()}};
}
} // namespace Dirty
#endif
/// U3 (default on since 2026-11-21, owner play; TLG_VKB_CHANGED_PAGES=0: whole copies): a big dynamic buffer (1 MiB or more) the engine maps every frame is copied to our device copy only where its bytes
/// changed. The engine rewrites ~212 pages of a 7 MB buffer each frame with the same values (U2 census), so neither page protection nor the whole copy pays: the job
/// compares the memory with a copy of what was last sent, page by page, on the resolver (in stream order, before the job is recorded), and copies the changed runs.
namespace Changed {
bool On() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_CHANGED_PAGES"); return e == nullptr || e[0] != '0'; }();
    return on;
}
u64 MinBytesNow() { // TLG_VKB_CHANGED_MIN=<bytes> (default 1 MiB)
    static const u64 n = [] { const char* e = std::getenv("TLG_VKB_CHANGED_MIN"); return e != nullptr ? std::strtoull(e, nullptr, 10) : (1ull << 20); }();
    return n;
}
constexpr u64 Page = 4096;
std::mutex shadows_mutex;
std::unordered_map<u64, std::vector<u8>> shadows; ///< by memory start: what was last sent
std::atomic<u64> c_jobs{0}, c_pages{0}, c_changed_pages{0}, c_runs{0}, c_compare_ns{0};
struct CompareJob final : Stream::Job {
    u64 memory{}, bytes{};
    BufferWindow src{};
    VkBuffer device{};
    u64 offset{};
    bool prepared{false};
    std::vector<std::pair<u64, u64>> runs; ///< (byte offset, bytes) within the range
    void Prepare() override {
        if (prepared) {
            return;
        }
        prepared = true;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<u8>* shadow = nullptr;
        {
            std::scoped_lock lock{shadows_mutex};
            shadow = &shadows[memory];
        }
        const u8* now = reinterpret_cast<const u8*>(memory);
        if (shadow->size() != bytes) {
            shadow->assign(now, now + bytes);
            runs.push_back({0, bytes});
        } else {
            const u64 pages = (bytes + Page - 1) / Page;
            u64 run_start = ~0ull;
            for (u64 i = 0; i < pages; ++i) {
                const u64 a = i * Page, n = std::min(Page, bytes - a);
                const bool diff = std::memcmp(now + a, shadow->data() + a, n) != 0;
                if (diff) {
                    std::memcpy(shadow->data() + a, now + a, n);
                    c_changed_pages.fetch_add(1, std::memory_order_relaxed);
                    if (run_start == ~0ull) {
                        run_start = a;
                    }
                } else if (run_start != ~0ull) {
                    runs.push_back({run_start, a - run_start});
                    run_start = ~0ull;
                }
            }
            if (run_start != ~0ull) {
                runs.push_back({run_start, bytes - run_start});
            }
            c_pages.fetch_add(pages, std::memory_order_relaxed);
        }
        c_runs.fetch_add(runs.size(), std::memory_order_relaxed);
        c_compare_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
    }
    void Run(vk::CommandBuffer cmd) override {
        Prepare(); // (with one recording thread the resolver does not call it)
        for (const auto& [at, n] : runs) {
            Stream::Hazard(cmd, {Stream::BufferUse(src.buffer, src.offset + at, n, false), Stream::BufferUse(vk::Buffer{device}, offset + at, n, true)});
            cmd.copyBuffer(src.buffer, vk::Buffer{device}, vk::BufferCopy{src.offset + at, offset + at, n});
        }
    }
};
/// True: handled (a compare-and-copy job is in the stream)
bool Unmapped(u64 memory, u64 bytes) {
    if (!On() || bytes < MinBytesNow() || !device_ready || (DeviceBuffersMode() == 2 && [&] {
            std::scoped_lock lock{mutex};
            const Watched* w = WatchedAt(memory, bytes);
            return w != nullptr && w->cls == 0 && w->state == Watched::State::Device; // (a watched static buffer: the old path)
        }())) {
        return false;
    }
    VkBuffer device{};
    u64 offset = 0;
    {
        std::scoped_lock lock{mutex};
        u64 start = 0;
        bool straddle = false;
        const DeviceBuffer* d = DeviceBufferOf(memory, bytes, start, straddle);
        if (d == nullptr || straddle) {
            return false;
        }
        device = d->buffer, offset = memory - start;
    }
    const BufferWindow src = SourceWindow(memory, bytes);
    if (!src.buffer) {
        return false;
    }
    auto job = std::make_unique<CompareJob>();
    job->work = Stream::Work::UploadCpuWritten;
    job->memory = memory, job->bytes = bytes, job->src = src, job->device = device, job->offset = offset;
    Stream::NoteBytes(Stream::Work::UploadCpuWritten, 0);
    Stream::Enqueue(std::move(job));
    c_jobs.fetch_add(1, std::memory_order_relaxed);
    return true;
}
nlohmann::json Report() {
    const double j = static_cast<double>(std::max<u64>(c_jobs.load(), 1));
    return {{"on", On()}, {"jobs", c_jobs.load()}, {"pages_per_job", c_pages.load() / j}, {"changed_pages_per_job", c_changed_pages.load() / j},
            {"runs_per_job", c_runs.load() / j}, {"compare_us_per_job", c_compare_ns.load() / j / 1000.0}};
}
} // namespace Changed
void DynamicUnmapped(u64 memory, u64 bytes) {
    c_dyn_unmaps.fetch_add(1, std::memory_order_relaxed);
    c_dyn_unmap_mb.fetch_add(bytes, std::memory_order_relaxed);
    PageCensus(memory, bytes);
    if (Changed::Unmapped(memory, bytes)) {
        return;
    }
#ifdef _WIN32
    if (Dirty::Unmapped(memory, bytes)) {
        return;
    }
#endif
    t_cpu_write_kind = 0;
    BufferCpuWritten(memory, bytes);
    t_cpu_write_kind = 2;
}
void SetCpuWriteKind(u32 kind) {
    t_cpu_write_kind = kind;
}

std::mutex dyn_check_mutex;
std::unordered_map<u64, u64> dyn_check_hash; ///< memory -> hash at its last Unmap
void DynamicCheck(u64 memory, u64 bytes, bool at_unmap) {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_DYN_CHECK"); return e != nullptr && e[0] == '1'; }();
    if (!on || memory == 0 || bytes == 0) {
        return;
    }
    const u64 h = HashGuest(memory, bytes);
    std::scoped_lock lock{dyn_check_mutex};
    if (at_unmap) {
        dyn_check_hash[memory] = h;
        return;
    }
    if (const auto it = dyn_check_hash.find(memory); it != dyn_check_hash.end()) {
        c_dyn_checks.fetch_add(1, std::memory_order_relaxed);
        if (it->second != h) {
            c_dyn_changed_outside.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_WARNING(Render_Vulkan, "vkb R3b check: dynamic buffer memory {:#x} ({} bytes) changed between its Unmap and its next Map", memory, bytes);
            }
        }
        dyn_check_hash.erase(it);
    }
}

void BufferCreated(const Object& buffer) {
    if (!On() || !Ready()) {
        return;
    }
    if (GpuReadTrap::On()) { // debug (issue 83): a buffer the graphics card writes is watched from its creation on
        if ((buffer.bind & BindUnorderedAccess) != 0) {
            GpuReadTrap::Watch(buffer.id, buffer.address, buffer.bytes, GpuReadTrap::Kind::UavBuffer);
        } else {
            GpuReadTrap::CpuObjectMade(buffer.address, buffer.bytes);
        }
    }
    c_buffers.fetch_add(1, std::memory_order_relaxed);
    const bool dynamic_device = DynamicDeviceOn() && buffer.usage == UsageDynamic && (buffer.cpu_access & 0x20000) == 0; // (never one the CPU reads)
    if (((buffer.bind & BindUnorderedAccess) != 0 || dynamic_device) && buffer.address != 0 && buffer.bytes != 0) {
        if (dynamic_device && (buffer.bind & BindUnorderedAccess) == 0) {
            c_dyn_device.fetch_add(1, std::memory_order_relaxed);
        }
        // V2f: a buffer the GPU writes lives in device memory
        const VkBufferCreateInfo ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                    .size = buffer.bytes,
                                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VkBuffer handle{};
        VmaAllocation allocation{};
        if (vmaCreateBuffer(instance->GetAllocator(), &ci, &alloc, &handle, &allocation, nullptr) != VK_SUCCESS) {
            c_dev_failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        {
            std::scoped_lock lock{mutex};
            const u32 due = static_cast<u32>(GraphicsSubmits()) + DestroyDelay();
            // an older buffer at the same memory (its delete not seen) goes
            if (const auto old = device_buffers.find(buffer.address); old != device_buffers.end()) {
                doomed.push_back({due, {}, {}, {}, old->second.buffer, old->second.allocation});
                device_buffer_start.erase(old->second.id);
                device_buffers.erase(old);
                BumpBufferEpoch(); // note 46: A1 memo answers may change
                c_dev_replaced.fetch_add(1, std::memory_order_relaxed);
            }
            device_buffers[buffer.address] = {buffer.address + buffer.bytes, handle, allocation, buffer.id, (buffer.bind & BindUnorderedAccess) != 0};
            device_buffer_start[buffer.id] = buffer.address;
            devbuf_hash.erase(buffer.address); // Y7 census: a new buffer at this memory starts a new baseline
            if (ContractCensusOn()) {
                devbuf_desc[buffer.address] = fmt::format("usage {} bind {:#x} cpu {:#x} misc {:#x} stride {}", buffer.usage, buffer.bind, buffer.cpu_access, buffer.misc, buffer.stride);
            }
        }
        c_dev_buffers.fetch_add(1, std::memory_order_relaxed);
        c_dev_bytes.fetch_add(buffer.bytes, std::memory_order_relaxed);
        // its initial data, copied in the frame stream (so after anything recorded before it, before anything after it)
        const BufferWindow w = SourceWindow(buffer.address, buffer.bytes);
        if (!w.buffer) {
            c_dev_no_initial.fetch_add(1, std::memory_order_relaxed);
        }
        const u64 bytes = buffer.bytes;
        Stream::NoteBytes(Stream::Work::UploadInitial, bytes);
        Stream::Defer(Stream::Work::UploadInitial, [w, handle, bytes](vk::CommandBuffer cmd) {
            if (w.buffer) {
                Stream::Hazard(cmd, {Stream::BufferUse(w.buffer, w.offset, bytes, false), Stream::BufferUse(vk::Buffer{handle}, 0, bytes, true)});
                cmd.copyBuffer(w.buffer, vk::Buffer{handle}, vk::BufferCopy{w.offset, 0, bytes});
            } else {
                Stream::Hazard(cmd, {Stream::BufferUse(vk::Buffer{handle}, 0, bytes, true)});
                cmd.fillBuffer(vk::Buffer{handle}, 0, VK_WHOLE_SIZE, 0);
            }
        });
        return;
    }
    if (buffer.address == 0 || buffer.bytes == 0 || !host_import->Available()) {
        c_buffers_not_in_place.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (host_import->Find(buffer.address, buffer.bytes).buffer) {
        c_buffers_in_place.fetch_add(1, std::memory_order_relaxed);
        c_buffer_bytes.fetch_add(buffer.bytes, std::memory_order_relaxed);
        if (DeviceBuffersMode() != 0) {
            const u32 cls = buffer.usage == UsageStaging ? 2 : (buffer.usage == UsageDynamic || (buffer.cpu_access & 0x10000) != 0) ? 1 : 0;
            std::scoped_lock lock{mutex};
            if (const auto old = watched.find(buffer.address); old != watched.end()) {
                if (old->second.device != VK_NULL_HANDLE) {
                    doomed.push_back({static_cast<u32>(GraphicsSubmits()) + DestroyDelay(), {}, {}, {}, old->second.device, old->second.allocation});
                }
                watched_by_id.erase(old->second.id);
                watched.erase(old);
                BumpBufferEpoch(); // note 46: A1 memo answers may change
            }
            watched[buffer.address] = {buffer.address + buffer.bytes, buffer.id, cls, buffer.usage, buffer.bind, buffer.cpu_access,
                                       cls == 0 ? HashGuest(buffer.address, buffer.bytes) : 0, false};
            watched_by_id[buffer.id] = buffer.address;
            if (R3dTraceBytes() != 0 && buffer.bytes == R3dTraceBytes()) {
                LOG_WARNING(Render_Vulkan, "vkb R3d trace: created buffer {:#x} id {} object usage {} bind {:#x} cpu {:#x} cls {} submit {} recorder {}", buffer.address, buffer.id, buffer.usage, buffer.bind,
                            buffer.cpu_access, cls, GraphicsSubmits(), Stream::OnRecorder());
            }
            c_watch_buffers[cls].fetch_add(1, std::memory_order_relaxed);
            c_watch_bytes[cls].fetch_add(buffer.bytes, std::memory_order_relaxed);
        }
    } else {
        c_buffers_not_in_place.fetch_add(1, std::memory_order_relaxed);
    }
}

void RetireView(vk::ImageView view) {
    if (!view) {
        return;
    }
    std::scoped_lock lock{mutex};
    doomed.push_back({static_cast<u32>(GraphicsSubmits()) + DestroyDelay(), {}, {}, view});
}

namespace {
void ObjectDeletedNow(u64 id);
std::atomic<u64> c_deletes_in_order{0};
} // namespace
/// tier-1-readback-findings.md D4-05: a deleted texture's staging rings go with it (called with `mutex` held; takes `staging_mutex`)
void RetireStaging(u64 texture_id, u32 due);
/// R2a (note 63): an engine object's deletion reaches our tables in stream order: an unlocked job (it ends no pass) erases our entry after every draw and dispatch
/// captured before the delete, on the recording thread. Today the lookups still run at the draw, so nothing reads a deleted id later; R2b moves them to the
/// recording thread, which needs exactly this order. TLG_VKB_DELETE_NOW=1: erased at the call, as before.
void ObjectDeleted(u64 id) {
    if (!On() || !device_ready) {
        return;
    }
    GpuReadTrap::Unwatch(id); // debug (issue 83): at the call, before the engine can reuse the memory
    static const bool now = [] { const char* e = std::getenv("TLG_VKB_DELETE_NOW"); return e != nullptr && e[0] == '1'; }();
    if (now || !Stream::RecorderOn()) {
        ObjectDeletedNow(id);
        return;
    }
    c_deletes_in_order.fetch_add(1, std::memory_order_relaxed);
    Stream::DeferUnlocked([id](vk::CommandBuffer) { ObjectDeletedNow(id); });
}
namespace {
void ObjectDeletedNow(u64 id) {
    const u32 due = static_cast<u32>(GraphicsSubmits()) + DestroyDelay();
    std::scoped_lock lock{mutex};
    if (const auto wid = watched_by_id.find(id); wid != watched_by_id.end()) {
        if (const auto w = watched.find(wid->second); w != watched.end() && w->second.id == id) {
            if (R3dTraceBytes() != 0 && w->second.end - w->first == R3dTraceBytes()) {
                LOG_WARNING(Render_Vulkan, "vkb R3d trace: deleted buffer {:#x} id {} state {} submit {}", w->first, id, static_cast<u32>(w->second.state), GraphicsSubmits());
                r3d_deleted.push_back({w->first, w->second.end});
            }
            if (w->second.device != VK_NULL_HANDLE) {
                doomed.push_back({due, {}, {}, {}, w->second.device, w->second.allocation});
                c_dev2_mb.fetch_sub(w->second.end - w->first, std::memory_order_relaxed);
                BumpBufferEpoch(); // note 46: only a device answer can be in the A1 memo
            }
            watched.erase(w);
        }
        watched_by_id.erase(wid);
    }
    if (const auto v = views.find(id); v != views.end()) {
        doomed.push_back({due, {}, {}, v->second.view});
        for (const vk::ImageView other : v->second.other) {
            if (other) {
                doomed.push_back({due, {}, {}, other});
            }
        }
        for (const auto& [format, formatted] : v->second.formatted) {
            doomed.push_back({due, {}, {}, formatted});
        }
        views.erase(v);
        BumpViewEpoch(); // note 46: A1 memo answers may change
        return;
    }
    if (const auto b = device_buffer_start.find(id); b != device_buffer_start.end()) {
        if (const auto d = device_buffers.find(b->second); d != device_buffers.end() && d->second.id == id) {
            doomed.push_back({due, {}, {}, {}, d->second.buffer, d->second.allocation});
            c_dev_bytes.fetch_sub(d->second.end - d->first, std::memory_order_relaxed);
            devbuf_hash.erase(d->first); // Y7 census
            device_buffers.erase(d);
            BumpBufferEpoch(); // note 46: A1 memo answers may change
        }
        device_buffer_start.erase(b);
        return;
    }
    if (const auto i = images.find(id); i != images.end()) {
        // the image's views the engine has not deleted yet go with it
        for (auto v = views.begin(); v != views.end();) {
            if (v->second.texture_id == id) {
                doomed.push_back({due, {}, {}, v->second.view});
                for (const vk::ImageView other : v->second.other) {
                    if (other) {
                        doomed.push_back({due, {}, {}, other});
                    }
                }
                for (const auto& [format, formatted] : v->second.formatted) {
                    doomed.push_back({due, {}, {}, formatted});
                }
                v = views.erase(v);
                BumpViewEpoch(); // note 46: A1 memo answers may change
            } else {
                ++v;
            }
        }
        doomed.push_back({due, i->second.image, i->second.allocation, {}});
        RetireStaging(id, due); // D4-05: its read-back buffers end with the texture (the engine's next staging texture has a new id)
        c_image_bytes.fetch_sub(i->second.bytes, std::memory_order_relaxed);
        if (ContractCensusOn()) { // review A2: copy-destination history ends with the texture
            std::scoped_lock census_lock{contract_mutex};
            contract_copy_destinations.erase(id);
        }
        images.erase(i);
        BumpViewEpoch(); // note 46: A1 memo answers may change
        return;
    }
    c_deleted_unknown.fetch_add(1, std::memory_order_relaxed);
}
} // namespace

namespace {
void CensusFrame();
}

std::atomic<u64> frame_ring_handle{0}, frame_ring_in_place_begin{0}, frame_ring_in_place_end{0};
void NoteFrameRing(u64 base, u64 size) {
    if (frame_ring_base.load(std::memory_order_relaxed) != base || frame_ring_handle.load(std::memory_order_relaxed) == 0) {
        frame_ring_base.store(base, std::memory_order_relaxed);
        frame_ring_size.store(size, std::memory_order_relaxed);
        if (device_ready && host_import && host_import->Available()) {
            if (const VkbVulkan::HostImport::Window w = host_import->Find(base, size); w.buffer) {
                frame_ring_in_place_begin.store(w.offset, std::memory_order_relaxed);
                frame_ring_in_place_end.store(w.offset + size, std::memory_order_relaxed);
                frame_ring_handle.store(reinterpret_cast<u64>(static_cast<VkBuffer>(w.buffer)), std::memory_order_release);
            }
        }
    }
}

namespace {
/// R3a placed ring (TLG_VKB_RING_PLACED, default on; =0: only the driver's own address, the way before): with VK_EXT_map_memory_placed the ring's memory is mapped at an
/// address the runtime reserves for us below 40 bits (`runtime_low_map`, where the runtime keeps its own memory the game sees), so the engine's 40 bit buffer
/// descriptions can hold it on every driver that has the feature. Without the feature (Windows, older drivers) the driver's own address is used, as before.
bool RingVramOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_RING_VRAM"); return e == nullptr || e[0] != '0'; }(); // default on since 2026-11-14 (note 29); =0: in place
    return on;
}
bool RingPlacedOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_RING_PLACED"); return e == nullptr || e[0] != '0'; }();
    return on;
}
const char* ring_route = "not placed yet"; ///< the report's word for how the ring got (or did not get) its device memory; set once inside PlaceFrameRing's call_once
std::string ring_placed_failure;           ///< why the placed route was not taken, when it was asked for and the driver has it (empty otherwise)
std::atomic<bool> ring_decided{false};     ///< the two above are final (the report reads them only then)
void RingDecided(const char* route) {
    ring_route = route;
    ring_decided.store(true, std::memory_order_release);
}

#ifndef _WIN32
/// The ring's memory mapped at an address below 40 bits: null when the placed route cannot be taken (the reason in ring_placed_failure).
void* MapRingPlaced(VmaAllocation allocation) {
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(instance->GetAllocator(), allocation, &info);
    const u64 align = instance->PlacedMapAlignment();
    if (info.offset != 0 || info.size % align != 0) {
        ring_placed_failure = fmt::format("the memory block (offset {:#x}, {} bytes) is not a whole, {}-aligned allocation", info.offset, info.size, align);
        return nullptr;
    }
    // the runtime hands out its low memory page by page; a stricter alignment is reached by asking for more and using an aligned part of it
    const u64 bytes = info.size + (align > 4096 ? align : 0);
    void* const low = runtime_low_map(bytes, PROT_NONE);
    if (low == nullptr) {
        ring_placed_failure = fmt::format("the runtime has no free address range of {} MiB below 40 bits", bytes >> 20);
        return nullptr;
    }
    const u64 at = (reinterpret_cast<u64>(low) + align - 1) & ~(align - 1);
    const vk::MemoryMapPlacedInfoEXT placed{.pPlacedAddress = reinterpret_cast<void*>(at)};
    const vk::MemoryMapInfo map{.pNext = &placed, .flags = vk::MemoryMapFlagBits::ePlacedEXT, .memory = info.deviceMemory, .offset = 0, .size = VK_WHOLE_SIZE};
    void* mapped = nullptr;
    const vk::Result result = instance->GetDevice().mapMemory2KHR(&map, &mapped);
    if (result != vk::Result::eSuccess || reinterpret_cast<u64>(mapped) != at) {
        ring_placed_failure = fmt::format("the driver did not map it at {:#x} ({}, got {})", at, vk::to_string(result), mapped);
        if (result == vk::Result::eSuccess) {
            (void)instance->GetDevice().unmapMemory2KHR(vk::MemoryUnmapInfo{.memory = info.deviceMemory});
        }
        munmap(low, bytes);
        return nullptr;
    }
    return mapped;
}
#endif
} // namespace

bool PlaceFrameRing(u64 allocator) {
    const bool on = RingVramOn();
    static std::once_flag once;
    static std::atomic<bool> placed{false};
    if (!on || allocator == 0 || !device_ready) {
        return false;
    }
    std::call_once(once, [allocator] {
        const u64 size = *reinterpret_cast<const u32*>(allocator);
#ifndef _WIN32
        const bool want_placed = RingPlacedOn() && instance->IsMapMemoryPlacedSupported();
#else
        const bool want_placed = false;
#endif
        const u64 align = want_placed ? instance->PlacedMapAlignment() : 1;
        const VkBufferCreateInfo ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                    .size = (size + VramRingSlack + align - 1) / align * align,
                                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_UNKNOWN;
        alloc.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        alloc.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT | (want_placed ? 0 : VMA_ALLOCATION_CREATE_MAPPED_BIT); // placed: we map it ourselves, where we choose
        VmaAllocationInfo info{};
        if (size == 0 || vmaCreateBuffer(instance->GetAllocator(), &ci, &alloc, &vram_ring_buffer, &vram_ring_allocation, &info) != VK_SUCCESS) {
            RingDecided("in place: no device-local CPU-visible memory");
            LOG_ERROR(Render_Vulkan, "vkb R3a: no device-local CPU-visible memory for the frame ring ({} bytes): it stays in place", size);
            return;
        }
        void* host = info.pMappedData;
        bool vma_mapped = false; // mapped by vmaMapMemory (unmapped again before the buffer is destroyed)
#ifndef _WIN32
        if (want_placed && (host = MapRingPlaced(vram_ring_allocation)) == nullptr) {
            // the placed route failed: the driver's own address, as on a driver without the feature (named here and in the report: principle 5)
            LOG_WARNING(Render_Vulkan, "vkb R3a: the frame ring cannot be mapped below 40 bits: {}; the driver's own address is tried", ring_placed_failure);
            vma_mapped = vmaMapMemory(instance->GetAllocator(), vram_ring_allocation, &host) == VK_SUCCESS;
            host = vma_mapped ? host : nullptr;
        }
#endif
        if (host == nullptr) {
            RingDecided("in place: the memory could not be mapped");
            LOG_ERROR(Render_Vulkan, "vkb R3a: the frame ring's device memory ({} bytes) could not be mapped: it stays in place", size);
            vmaDestroyBuffer(instance->GetAllocator(), vram_ring_buffer, vram_ring_allocation);
            vram_ring_buffer = VK_NULL_HANDLE;
            return;
        }
        const u64 mapped = reinterpret_cast<u64>(host);
        if (mapped + size + VramRingSlack > (1ull << 40)) {
            // the engine's buffer descriptions hold 40 bit addresses (only the driver's own address can be up there; a placed one is below by construction, as
            // runtime_low_map stays below 0x1000000000)
            RingDecided("in place: the driver's address is above 40 bits");
            LOG_ERROR(Render_Vulkan, "vkb R3a: the frame ring's mapping {:#x} is above 40 bits: it stays in place ({})", mapped,
                      !RingPlacedOn()                                  ? "TLG_VKB_RING_PLACED=0"
                      : instance->IsMapMemoryPlacedSupported()          ? "a mapping below 40 bits failed, see above"
                                                                       : "the driver has no placed mapping, VK_EXT_map_memory_placed; this is the way for such drivers");
            if (vma_mapped) {
                vmaUnmapMemory(instance->GetAllocator(), vram_ring_allocation);
            }
            vmaDestroyBuffer(instance->GetAllocator(), vram_ring_buffer, vram_ring_allocation); // (a persistent VMA mapping is unmapped by VMA)
            vram_ring_buffer = VK_NULL_HANDLE;
            return;
        }
        std::memset(host, 0, size + VramRingSlack); // (what the guest memory behind a block held is not known; zero, not garbage)
        host_side_ring_size.store(size, std::memory_order_relaxed);
        host_side_ring_base.store(*reinterpret_cast<const u64*>(allocator + 0x18), std::memory_order_release);
        vram_ring_end.store(mapped + size + VramRingSlack, std::memory_order_relaxed);
        vram_ring_base.store(mapped, std::memory_order_release);
        *reinterpret_cast<u64*>(allocator + 0x18) = mapped; // from the next allocation on, the blocks are ours (base + offset)
        placed.store(true, std::memory_order_release);
        RingDecided(want_placed && ring_placed_failure.empty() ? "device memory, placed below 40 bits" : "device memory, at the driver's address");
        LOG_INFO(Render_Vulkan, "vkb R3a: the frame ring ({} MiB) is in device memory at {:#x} ({})", size >> 20, mapped, ring_route);
    });
    return placed.load(std::memory_order_acquire);
}

std::array<std::array<std::array<std::atomic<u64>, 2>, 8>, 4> c_map_count{}, c_map_bytes{}; ///< [usage][type][in place]
void NoteBufferMap(u32 usage, u32 type, bool in_place, u64 bytes) {
    usage = std::min<u32>(usage, 3), type = std::min<u32>(type, 7);
    c_map_count[usage][type][in_place ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
    c_map_bytes[usage][type][in_place ? 1 : 0].fetch_add(bytes, std::memory_order_relaxed);
}

bool HostSideRing(u64& base, u64& size) {
    base = host_side_ring_base.load(std::memory_order_acquire);
    size = host_side_ring_size.load(std::memory_order_relaxed);
    return base != 0 && size != 0;
}

/// note 49 census, debug only: TLG_VKB_FILL_READERS=1 writes log/vkb_fill_readers.txt: every in-place fill target (with the engine's caller), then each distinct
/// buffer bind (shader tag), copy or upload source, and texture whose memory overlaps one of them. Answers who reads the host-memory fills.
bool FillReadersOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_FILL_READERS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
namespace {
std::mutex fr_mutex;
std::vector<std::pair<u64, u64>> fr_ranges;
std::set<std::tuple<u64, u64, u64, int>> fr_seen;
std::set<u64> fr_images_seen;
thread_local u64 t_census_tag = 0;
void FillReadersLine(const std::string& text) {
    if (FILE* f = std::fopen((Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_fill_readers.txt").string().c_str(), "a")) {
        std::fputs(text.c_str(), f);
        std::fputc(10, f);
        std::fclose(f);
    }
}
} // namespace
void CensusTag(u64 tag) {
    t_census_tag = tag;
}
/// debug R3d trace: every device buffer of the traced size (the fur outputs) dumped now into vkb_image_dump/r3d/<label>_<address>
void R3dDumpTraced(const std::string& label) {
    if (R3dTraceBytes() == 0) {
        return;
    }
    std::vector<std::pair<u64, VkBuffer>> list;
    {
        std::scoped_lock lock{mutex};
        for (const auto& [start, d] : device_buffers) {
            if (d.end - start == R3dTraceBytes()) {
                list.emplace_back(start, d.buffer);
            }
        }
    }
    const u64 bytes = R3dTraceBytes();
    LOG_WARNING(Render_Vulkan, "vkb R3d trace: dumping {} device buffers of {} bytes ({})", list.size(), bytes, label);
    for (const auto& [start, buffer] : list) {
        Stream::Defer(Stream::Work::Other, [start, buffer, bytes, label](vk::CommandBuffer cmd) {
            DumpBuffer(cmd, vk::Buffer{buffer}, 0, bytes, fmt::format("r3d/{}_{:x}", label, start));
        });
    }
}
u64 R3dTraceSize() {
    return R3dTraceBytes();
}
extern std::atomic<u64> g_buffer_epoch; // note 46, defined below
namespace {
struct ObjectBufferMemo {
    u64 object{}, id{}, epoch{~0ull}, memory{}, bytes{}; ///< memory and bytes: what the window was resolved for (issue 12)
    BufferWindow window{};
    bool ok{};
};
thread_local std::unique_ptr<std::array<ObjectBufferMemo, 4096>> t_object_buffers;
std::atomic<u64> c_r7_ring_buffers{0}, c_r7_window_checks{0}, c_r7_window_differs{0}, c_r7_hits{0}, c_r7_resolves{0}, c_r7_whole_failed{0}, c_r7_range_fallback{0}, c_r7_past_end{0}, c_r7_ring_cb{0}, c_r7_cb_copied{0};
std::atomic<u64> c_memo_memory_moved{0}; ///< entries that matched object, id and epoch but not memory or bytes (switch on: resolved again; off: the old window served)
} // namespace
void NoteConstantCopied() {
    c_r7_cb_copied.fetch_add(1, std::memory_order_relaxed);
}
namespace {
constexpr u64 PoolTableBits = 17, PoolTableSize = 1ull << PoolTableBits;
std::array<std::atomic<u64>, PoolTableSize> pool_cbs{}; ///< engine objects of the per-object constant buffer pool (open addressing, never removed: the pool only grows)
std::atomic<u64> c_pool_cbs{0}, c_pool_full{0}, c_pool_in_place{0}, c_pool_submits{0}, c_pool_submits_late{0};
u64 PoolSlot(u64 object) {
    return (object * 0x9e3779b97f4a7c15ull) >> (64 - PoolTableBits);
}
} // namespace
bool CbInPlaceOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_CB_IN_PLACE"); return e != nullptr && e[0] == '1'; }();
    return on && On();
}
void NotePoolConstantBuffer(u64 object) {
    if (!CbInPlaceOn() || object == 0) {
        return;
    }
    for (u64 n = 0, i = PoolSlot(object); n < PoolTableSize; ++n, i = (i + 1) & (PoolTableSize - 1)) {
        u64 expected = 0;
        if (pool_cbs[i].compare_exchange_strong(expected, object, std::memory_order_acq_rel)) {
            c_pool_cbs.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (expected == object) {
            return;
        }
    }
    c_pool_full.fetch_add(1, std::memory_order_relaxed);
}
bool PoolConstantBuffer(u64 object) {
    for (u64 n = 0, i = PoolSlot(object); n < PoolTableSize; ++n, i = (i + 1) & (PoolTableSize - 1)) {
        const u64 v = pool_cbs[i].load(std::memory_order_acquire);
        if (v == object) {
            return true;
        }
        if (v == 0) {
            return false;
        }
    }
    return false;
}
void NotePoolInPlace() {
    c_pool_in_place.fetch_add(1, std::memory_order_relaxed);
}
void PoolSubmitCheck(bool gpu_two_back_done) {
    c_pool_submits.fetch_add(1, std::memory_order_relaxed);
    if (!gpu_two_back_done) {
        c_pool_submits_late.fetch_add(1, std::memory_order_relaxed);
    }
}
nlohmann::json PoolReport() {
    return {{"on", CbInPlaceOn()}, {"pool_buffers", c_pool_cbs.load()}, {"table_full", c_pool_full.load()}, {"bound_in_place", c_pool_in_place.load()},
            {"submits_checked", c_pool_submits.load()}, {"submits_gpu_more_than_two_behind", c_pool_submits_late.load()}};
}
bool ConstantInRing(u64 address, u64 size, BufferWindow& out) {
    if (InVramRing(address, size, out)) {
        c_r7_ring_cb.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}
bool BufferInRing(u64 address, u64 size, BufferWindow& out) {
    if (InVramRing(address, size, out)) {
        c_r7_ring_buffers.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}
bool BufferOfObject(u64 buffer, u64 offset, u64 size, BufferWindow& out) {
    if (!device_ready || buffer == 0 || size == 0) {
        return false;
    }
    return BufferOfObjectFields(buffer, Util::Rd<u64>(buffer, 0x10), Util::Rd<u64>(buffer, 0x30), Util::Rd<u32>(buffer, 0x180), offset, size, out);
}
bool BufferOfObjectFields(u64 buffer, u64 id, u64 memory, u64 bytes, u64 offset, u64 size, BufferWindow& out) {
    if (!device_ready || buffer == 0 || size == 0) {
        return false;
    }
    if (!t_object_buffers) {
        t_object_buffers = std::make_unique<std::array<ObjectBufferMemo, 4096>>();
    }
    // a buffer whose memory is in the frame ring (dynamic data the engine placed there) moves with the ring every submission: by arithmetic each time, never cached
    if (InVramRing(memory + offset, size, out)) {
        c_r7_ring_buffers.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    ObjectBufferMemo& m = (*t_object_buffers)[((buffer >> 4) ^ (buffer >> 16) ^ id) & 4095];
    const u64 epoch = g_buffer_epoch.load(std::memory_order_acquire);
    // issue 12: a buffer keeps object and id while its memory moves (skin ring rename, recording ring, the engine's buffer move); the entry answers only for the memory it was resolved for. TLG_VKB_MEMO_BY_MEMORY=0: the old key
    static const bool by_memory = [] { const char* e = std::getenv("TLG_VKB_MEMO_BY_MEMORY"); return e == nullptr || e[0] != '0'; }();
    const bool same_key = m.object == buffer && m.id == id && m.epoch == epoch;
    const bool same_memory = m.memory == memory && m.bytes == bytes;
    if (same_key && !same_memory) {
        c_memo_memory_moved.fetch_add(1, std::memory_order_relaxed);
    }
    if (!same_key || (by_memory && !same_memory)) {
        c_r7_resolves.fetch_add(1, std::memory_order_relaxed);
        m = {buffer, id, epoch, memory, bytes, {}, false};
        if (memory != 0 && bytes != 0) {
            m.ok = FindLocked(memory, bytes, m.window);
            // its device copy may have been made just now (R3d): the answer stands only for this epoch
            m.epoch = g_buffer_epoch.load(std::memory_order_acquire) == epoch ? epoch : ~0ull;
        }
        if (!m.ok) {
            c_r7_whole_failed.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        c_r7_hits.fetch_add(1, std::memory_order_relaxed);
    }
    static const bool check = [] { const char* e = std::getenv("TLG_VKB_R7_CHECK"); return e != nullptr && e[0] == '1'; }();
    if (check && m.ok) { // note 56 check: the cached window against a fresh find of the same range
        BufferWindow fresh;
        const bool fresh_ok = FindBuffer(memory + offset, size, fresh);
        const bool same = fresh_ok && fresh.buffer == m.window.buffer && fresh.offset == m.window.offset + offset && fresh.device == m.window.device;
        c_r7_window_checks.fetch_add(1, std::memory_order_relaxed);
        if (!same) {
            c_r7_window_differs.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 24) {
                LOG_WARNING(Render_Vulkan, "vkb R7 window check: object {:#x} id {} memory {:#x} bytes {} usage {} bind {:#x} misc {:#x}: +{} {} B cached {:x}+{} dev {} / fresh ok {} {:x}+{} dev {}", buffer, id, memory, m.bytes,
                            Util::Rd<u32>(buffer, 0x184), Util::Rd<u32>(buffer, 0x188), Util::Rd<u32>(buffer, 0x190), offset, size, reinterpret_cast<u64>(static_cast<VkBuffer>(m.window.buffer)),
                            m.window.offset + offset, m.window.device, fresh_ok, reinterpret_cast<u64>(static_cast<VkBuffer>(fresh.buffer)), fresh.offset, fresh.device);
            }
        }
    }
    if (m.ok) {
        if (offset + size > m.bytes) {
            c_r7_past_end.fetch_add(1, std::memory_order_relaxed); // an index range estimate past the buffer's end (the old path bound it the same way)
        }
        out = m.window;
        out.offset += offset;
        return true;
    }
    // the whole buffer is not in one window of ours (an imported memory chunk boundary): the range alone, by address, as before
    c_r7_range_fallback.fetch_add(1, std::memory_order_relaxed);
    return memory != 0 && FindBuffer(memory + offset, size, out);
}
nlohmann::json BindByObjectReport() {
    return {{"object_hits", c_r7_hits.load()}, {"object_resolves", c_r7_resolves.load()}, {"whole_buffer_not_one_window", c_r7_whole_failed.load()},
            {"range_by_address_fallback", c_r7_range_fallback.load()}, {"index_past_the_end", c_r7_past_end.load()}, {"constants_in_the_frame_ring", c_r7_ring_cb.load()},
            {"constants_copied_at_the_draw", c_r7_cb_copied.load()}, {"window_checks_differs", {c_r7_window_checks.load(), c_r7_window_differs.load()}}, {"buffers_in_the_frame_ring", c_r7_ring_buffers.load()}, {"memo_memory_moved", c_memo_memory_moved.load()}};
}
bool R7CensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_R7_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
namespace {
// [bind][where]: 0 the buffer's own memory at offset 0, 1 its own memory at an offset, 2 the frame ring, 3 elsewhere, 4 the range runs past the buffer's end; [bind][5 + usage]: binds by usage
std::atomic<u64> c_r7[5][10]{};
std::atomic<u64> c_r7_device_home[5]{}; ///< binds of a buffer that has a device buffer of ours from its creation (UAV, dynamic)
std::atomic<u64> c_r7_size_differs[5]{}; ///< the descriptor's size is not the buffer's ByteWidth minus the offset
} // namespace
void R7Census(u32 bind, u64 object, u64 address, u64 size) {
    if (!R7CensusOn() || bind >= 5 || object == 0) {
        return;
    }
    const u64 buffer = (bind == 1 || bind == 2) ? Util::Rd<u64>(object, 0x30) : object;
    if (buffer == 0) {
        c_r7[bind][3].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 memory = Util::Rd<u64>(buffer, 0x30), bytes = Util::Rd<u32>(buffer, 0x180), usage = Util::Rd<u32>(buffer, 0x184);
    const u64 rb = frame_ring_base.load(std::memory_order_relaxed), rs = frame_ring_size.load(std::memory_order_relaxed);
    u32 where = 3;
    if (address >= memory && address < memory + bytes) {
        where = address + size > memory + bytes ? 4 : address == memory ? 0 : 1;
        if (address + size <= memory + bytes && size != memory + bytes - address) {
            c_r7_size_differs[bind].fetch_add(1, std::memory_order_relaxed);
        }
    } else if (rb != 0 && address >= rb && address < rb + rs) {
        where = 2;
    }
    c_r7[bind][where].fetch_add(1, std::memory_order_relaxed);
    if (where == 3 && bind == 0) { // note 56: what a constant buffer's descriptor points at when it is neither its own memory nor the frame ring (first 24 regions)
        static std::mutex m;
        static std::vector<u64> regions;
        std::scoped_lock lock{m};
        if (regions.size() < 24 && std::find(regions.begin(), regions.end(), address >> 20) == regions.end()) {
            regions.push_back(address >> 20);
            std::string owner = "no engine buffer";
            {
                std::scoped_lock objects_lock{mutex};
                u64 start = 0;
                bool straddle = false;
                if (const DeviceBuffer* d = DeviceBufferOf(address, size, start, straddle)) {
                    owner = fmt::format("device buffer of engine id {} at {:#x}", d->id, start);
                } else if (const Watched* w = WatchedAt(address, size)) {
                    owner = fmt::format("engine buffer id {} usage {} bind {:#x} cpu {:#x} (in place)", w->id, w->usage, w->bind, w->cpu_access);
                }
            }
            FillReadersLine(fmt::format("r7 cb elsewhere: cb id {} memory {:#x} bytes {} -> descriptor {:#x}+{}: {}", Util::Rd<u64>(buffer, 0x10), memory, bytes, address, size, owner));
        }
    }
    c_r7[bind][5 + std::min<u64>(usage, 4)].fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lock{mutex};
        if (device_buffer_start.contains(Util::Rd<u64>(buffer, 0x10))) {
            c_r7_device_home[bind].fetch_add(1, std::memory_order_relaxed);
        }
    }
}
nlohmann::json R7CensusReport() {
    nlohmann::json j = nlohmann::json::object();
    static const char* names[] = {"constant_buffer", "buffer_view", "uav_view", "vertex", "index"};
    for (u32 b = 0; b < 5; ++b) {
        j[names[b]] = {{"own_memory_at_0", c_r7[b][0].load()}, {"own_memory_at_offset", c_r7[b][1].load()}, {"frame_ring", c_r7[b][2].load()}, {"elsewhere", c_r7[b][3].load()},
                       {"past_the_end", c_r7[b][4].load()}, {"by_usage_default_immutable_dynamic_staging", {c_r7[b][5].load(), c_r7[b][6].load(), c_r7[b][7].load(), c_r7[b][8].load()}},
                       {"device_home_from_creation", c_r7_device_home[b].load()}, {"size_not_bytewidth_minus_offset", c_r7_size_differs[b].load()}};
    }
    return j;
}
void FillReadersTarget(u64 destination, u64 bytes, u64 caller, const char* what) {
    if (!FillReadersOn()) {
        return;
    }
    std::scoped_lock lock{fr_mutex};
    for (const auto& r : fr_ranges) {
        if (r.first == destination && r.second == bytes) {
            return;
        }
    }
    fr_ranges.push_back({destination, bytes});
    FillReadersLine(fmt::format("{} {:#x}..{:#x} ({} bytes) caller {:#x} submit {}", what, destination, destination + bytes, bytes, caller, GraphicsSubmits()));
}
void FillReadersCheck(u64 address, u64 size, int what) {
    if (!FillReadersOn()) {
        return;
    }
    std::scoped_lock lock{fr_mutex};
    for (const auto& r : fr_ranges) {
        if (address < r.first + r.second && r.first < address + size) {
            if (fr_seen.size() < 4096 && fr_seen.insert({address, size, t_census_tag, what}).second) {
                static const char* names[] = {"bind", "copy/upload source", "texture contents upload"};
                FillReadersLine(fmt::format("{} {:#x}+{} (fill {:#x}+{:#x}) tag {:#x} submit {}", names[what], address, size, r.first, address - r.first, t_census_tag,
                                            GraphicsSubmits()));
            }
            return;
        }
    }
}

BufferWindow SourceWindow(u64 address, u64 size) {
    FillReadersCheck(address, size, 1);
    BufferWindow out;
    if (InVramRing(address, size, out)) {
        return out;
    }
    if (device_ready && host_import && host_import->Available()) {
        if (const VkbVulkan::HostImport::Window w = host_import->Find(address, size); w.buffer) {
            out = {w.buffer, w.offset, false};
        }
    }
    return out;
}

bool FrameRingInPlace(u64& buffer_handle, u64& begin, u64& end) {
    buffer_handle = frame_ring_handle.load(std::memory_order_acquire);
    begin = frame_ring_in_place_begin.load(std::memory_order_relaxed);
    end = frame_ring_in_place_end.load(std::memory_order_relaxed);
    return buffer_handle != 0;
}

namespace {
/// api-contract.md Y7 census (debug, with TLG_VKB_CONTRACT_CENSUS=1): the guest memory under each GPU-written device buffer, hashed every 60 graphics submits; a change
/// no BufferCpuWritten announced is a CPU write (or a file read) our device copy never gets
void DeviceBufferWriteCheck() {
    // (with the read trap on, debug issue 83, this memory is protected and every processor write to it is caught by the trap itself; the hash here
    // would only trap on its own reads)
    if (!ContractCensusOn() || GpuReadTrap::On() || (GraphicsSubmits() % 60) != 0) {
        return;
    }
    std::vector<std::tuple<u64, u64, u64>> ranges; // start, end, id
    {
        std::scoped_lock lock{mutex};
        for (const auto& [start, d] : device_buffers) {
            ranges.emplace_back(start, d.end, d.id);
        }
    }
    for (const auto& [start, end, id] : ranges) {
        const u64 h = HashGuest(start, end - start);
        bool changed = false;
        {
            std::scoped_lock lock{mutex};
            auto [it, fresh] = devbuf_hash.try_emplace(start, h);
            changed = !fresh && it->second != h;
            it->second = h;
        }
        if (changed) {
            std::string desc;
            {
                std::scoped_lock lock{mutex};
                if (const auto d = devbuf_desc.find(start); d != devbuf_desc.end()) {
                    desc = d->second;
                }
            }
            ContractNote(fmt::format("Y7 guest memory of a GPU-written device buffer changed unannounced ({} B, {})", end - start, desc));
        }
    }
}
} // namespace

void ObjectsTick() {
    if (!On() || !device_ready) {
        return;
    }
    DestroyDue(static_cast<u32>(GraphicsSubmits()), false);
    CensusFrame();
    DeviceBufferWriteCheck();
    GpuReadTrap::Tick();
    if (DeviceBuffersMode() != 0) {
        std::vector<std::pair<u64, u64>> marks;
        {
            std::scoped_lock marks_lock{gpu_write_mutex};
            marks.swap(gpu_write_pending);
        }
        for (const auto& [a, n] : marks) {
            NoteGpuWrite(a, n);
        }
    }
    if (DeviceBuffersMode() == 2) {
        // V4j: the settling buffers are hashed outside the lock (the render thread finds buffers under it)
        std::vector<std::pair<u64, u64>> to_hash; // address, bytes
        {
            std::scoped_lock lock{mutex};
            to_hash.reserve(settling.size());
            for (const u64 address : settling) {
                if (const auto it = watched.find(address); it != watched.end() && it->second.state == Watched::State::InPlace) {
                    to_hash.emplace_back(address, it->second.end - address);
                }
            }
        }
        std::unordered_map<u64, u64> hashes;
        hashes.reserve(to_hash.size());
        for (const auto& [address, bytes] : to_hash) {
            hashes[address] = HashGuest(address, bytes);
        }
        std::scoped_lock lock{mutex};
        const u64 now = GraphicsSubmits();
        // bound buffers still in place: hashed every submit; unchanged for SettleSubmits -> their own device copy (copied in this submit's upload batch, bound from two submits on)
        std::erase_if(settling, [&](u64 address) {
            const auto it = watched.find(address);
            if (it == watched.end() || it->second.state != Watched::State::InPlace) {
                return true;
            }
            Watched& w = it->second;
            const u64 bytes = w.end - address;
            const auto hashed = hashes.find(address);
            if (hashed == hashes.end()) {
                return false; // added after the snapshot: next submit
            }
            if (hashed->second != w.hash) {
                w.hash = hashed->second;
                w.last_change_submit = now;
                c_dev2_settle_changes.fetch_add(1, std::memory_order_relaxed);
                if (++w.settle_changes > SettleGiveUp) {
                    w.state = Watched::State::Unstable; // the CPU keeps writing it: in place for good
                    BumpBufferEpoch(); // note 46: A1 memo answers may change
                    c_dev2_gave_up.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                return false;
            }
            if (w.gpu_written) {
                w.state = Watched::State::Unstable; // our copies or fills write it in place: it stays there
                BumpBufferEpoch(); // note 46: A1 memo answers may change
                return true;
            }
            if (now < w.last_change_submit + SettleSubmits) {
                return false;
            }
            const VkbVulkan::HostImport::Window src = host_import->Find(address, bytes);
            const VkBufferCreateInfo ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                        .size = bytes,
                                        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
            VmaAllocationCreateInfo alloc{};
            alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (!src.buffer || vmaCreateBuffer(instance->GetAllocator(), &ci, &alloc, &w.device, &w.allocation, nullptr) != VK_SUCCESS) {
                w.device = VK_NULL_HANDLE;
                BumpBufferEpoch(); // note 46: A1 memo answers may change
                w.state = Watched::State::Unstable;
                BumpBufferEpoch(); // note 46: A1 memo answers may change
                c_dev2_failed.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            Upload::CopyBuffer(src.buffer, src.offset, vk::Buffer{w.device}, 0, bytes);
            w.state = Watched::State::Device;
            BumpBufferEpoch(); // note 46: A1 memo answers may change
            w.ready_submit = now + 2;
            c_dev2_made.fetch_add(1, std::memory_order_relaxed);
            c_dev2_mb.fetch_add(bytes, std::memory_order_relaxed);
            return true;
        });
        if (R3dTraceBytes() != 0) { // debug R3d trace: the traced buffers' memory hashed every submit; a change is logged with their state
            static std::unordered_map<u64, u64> traced_hash;
            for (auto& [start, tw] : watched) {
                if (tw.end - start != R3dTraceBytes()) {
                    continue;
                }
                const u64 h = HashGuest(start, tw.end - start);
                auto [th, fresh] = traced_hash.try_emplace(start, h);
                if (!fresh && th->second != h) {
                    LOG_WARNING(Render_Vulkan, "vkb R3d trace: memory of buffer {:#x} changed (no announcement seen by the hash) state {} first bind {} submit {}", start,
                                static_cast<u32>(tw.state), tw.first_bind_submit, now);
                }
                th->second = h;
            }
        }
        // buffers with a device copy: watched round robin (16 MiB a submit); a change sends one back to in place for good
        // kill-experiment; R3d: off by default with the copies at first bind (TLG_VKB_NO_WATCH=0 turns it on again)
        static const bool no_watch = [] { const char* e = std::getenv("TLG_VKB_NO_WATCH"); return e != nullptr ? e[0] == '1' : StaticAtBindOn() && !R3dCheckOn(); }();
        u64 budget = no_watch ? 0 : R3dCheckOn() ? 16ull << 20 : 4ull << 20; // V4i: 4 MiB a submit (0 copies ever changed after their copy; the hashing was 9 % of the recording thread)
        auto it = watched.lower_bound(watch_cursor);
        for (size_t visited = 0; budget != 0 && !watched.empty() && visited < watched.size(); ++visited) {
            if (it == watched.end()) {
                it = watched.begin();
                c_watch_rounds.fetch_add(1, std::memory_order_relaxed);
            }
            Watched& w = it->second;
            if (w.state == Watched::State::Device) {
                const u64 bytes = w.end - it->first;
                if (const u64 h = HashGuest(it->first, bytes); h != w.hash) {
                    w.hash = h;
                    w.state = Watched::State::Unstable;
                    BumpBufferEpoch(); // note 46: A1 memo answers may change
                    doomed.push_back({static_cast<u32>(now) + DestroyDelay(), {}, {}, {}, w.device, w.allocation});
                    c_dev2_mb.fetch_sub(bytes, std::memory_order_relaxed);
                    w.device = VK_NULL_HANDLE;
                    BumpBufferEpoch(); // note 46: A1 memo answers may change
                    c_dev2_unstable.fetch_add(1, std::memory_order_relaxed);
                    static std::atomic<u32> logged{0};
                    if (logged.fetch_add(1, std::memory_order_relaxed) < (R3dCheckOn() ? 200u : 12u)) {
                        LOG_WARNING(Render_Vulkan, "vkb device buffers: buffer id {} at {:#x} ({} bytes, usage {}, bind {:#x}) changed after its device copy: back in place",
                                    w.id, it->first, bytes, w.usage, w.bind);
                    }
                }
                c_watch_hashed_mb.fetch_add(bytes, std::memory_order_relaxed);
                budget = bytes >= budget ? 0 : budget - bytes;
            }
            ++it;
            watch_cursor = it == watched.end() ? 0 : it->first;
        }
    }
    if (DeviceBuffersMode() == 1) {
        // the watch: up to 16 MiB of class-0 buffers hashed again per graphics submit, round robin
        std::scoped_lock lock{mutex};
        u64 budget = 16ull << 20;
        auto it = watched.lower_bound(watch_cursor);
        while (budget != 0 && !watched.empty()) {
            if (it == watched.end()) {
                it = watched.begin();
                c_watch_rounds.fetch_add(1, std::memory_order_relaxed);
            }
            Watched& w = it->second;
            const u64 bytes = w.end - it->first;
            if (w.cls == 0) {
                const u64 h = HashGuest(it->first, bytes);
                if (h != w.hash) {
                    if (w.gpu_written) {
                        c_watch_mismatch_gpu.fetch_add(1, std::memory_order_relaxed);
                    } else if (w.first_bind_submit == 0) {
                        c_watch_changed_before_bind.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        c_watch_changed_after_bind.fetch_add(1, std::memory_order_relaxed);
                        if (GraphicsSubmits() + 1 <= w.first_bind_submit + 3) {
                            c_watch_changed_within_3.fetch_add(1, std::memory_order_relaxed);
                        }
                        if (++w.changes_after_bind == 1) {
                            c_watch_buffers_changed_after_bind.fetch_add(1, std::memory_order_relaxed);
                        } else if (w.changes_after_bind == 3) {
                            c_watch_buffers_changed_often.fetch_add(1, std::memory_order_relaxed);
                        }
                        static std::atomic<u32> logged{0};
                        if (logged.fetch_add(1, std::memory_order_relaxed) < 12) {
                            LOG_WARNING(Render_Vulkan, "vkb device buffer check: buffer id {} at {:#x} ({} bytes, usage {}, bind {:#x}, cpu access {:#x}) changed without a hooked write",
                                        w.id, it->first, bytes, w.usage, w.bind, w.cpu_access);
                        }
                        c_watch_mismatch.fetch_add(1, std::memory_order_relaxed);
                    }
                    w.hash = h;
                }
                c_watch_hashed_mb.fetch_add(bytes, std::memory_order_relaxed);
                budget = bytes >= budget ? 0 : budget - bytes;
            }
            ++it;
            watch_cursor = it == watched.end() ? 0 : it->first;
        }
    }
    Upload::Submit();
    PublishObjects();
}

VkbVulkan::HostImport* InPlace() {
    return device_ready ? host_import.get() : nullptr;
}

namespace {
/// note 46, A1 (capture by change; TLG_VKB_MEMO, default on, =0 off): each thread keeps the answers of its last lookups (a buffer range's device window, a view as
/// a shader type, a sampler) and reuses them while nothing that could change an answer has happened. Buffers: only device answers are kept (ring, dynamic and static
/// device copies); the buffer epoch moves when a device window is made in place of another, dropped or replaced. Views: the view epoch moves when a view or an image goes,
/// or an image's layout is set back. Samplers never go (one per engine sampler id), so a sampler answer stays.
bool MemoOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_MEMO"); return e == nullptr || e[0] != '0'; }();
    return on;
}
std::atomic<u64> c_memo_hits[3]{}, c_memo_misses[3]{}, c_memo_bumps[2]{};
constexpr u32 MemoSize = 4096; // per thread, direct mapped
struct BufferMemo {
    u64 address{}, size{}, epoch{~0ull};
    BufferWindow window{};
};
struct ViewMemo {
    u64 id{}, epoch{~0ull};
    u32 want{};
    vk::ImageView view{};
    vk::Image image{};
};
struct SamplerMemo {
    u64 key{~0ull};
    vk::Sampler sampler{};
};
thread_local std::unique_ptr<std::array<BufferMemo, MemoSize>> t_buffer_memo;
thread_local std::unique_ptr<std::array<ViewMemo, MemoSize>> t_view_memo;
thread_local std::unique_ptr<std::array<SamplerMemo, MemoSize>> t_sampler_memo;
inline u32 MemoSlot(u64 a, u64 b) {
    const u64 h = (a ^ (b * 0x9e3779b97f4a7c15ull)) * 0xff51afd7ed558ccdull;
    return static_cast<u32>(h >> 52) & (MemoSize - 1);
}
} // namespace
std::atomic<u64> g_buffer_epoch{0}, g_view_epoch{0};
void BumpBufferEpoch() {
    g_buffer_epoch.fetch_add(1, std::memory_order_release);
    c_memo_bumps[0].fetch_add(1, std::memory_order_relaxed);
}
void BumpViewEpoch() {
    g_view_epoch.fetch_add(1, std::memory_order_release);
    c_memo_bumps[1].fetch_add(1, std::memory_order_relaxed);
}
u64 ViewEpoch() {
    return g_view_epoch.load(std::memory_order_acquire);
}
/// note 45: time in our object lookups (views, buffers, samplers, attachments), by lookup kind, on any thread
std::atomic<u64> g_lookup_ns[4]{}, g_lookup_n[4]{};
bool LookupTimesOn() { static const bool on = [] { const char* e = std::getenv("TLG_VKB_TIMES"); return e != nullptr && e[0] == '1'; }(); return on; } ///< TLG_VKB_TIMES=1 only
struct LookupTimer {
    u32 kind;
    u64 t0;
    explicit LookupTimer(u32 k) : kind{k}, t0{LookupTimesOn() ? static_cast<u64>(std::chrono::steady_clock::now().time_since_epoch().count()) : 0} {}
    ~LookupTimer() {
        if (t0 == 0) {
            return;
        }
        g_lookup_ns[kind].fetch_add(static_cast<u64>(std::chrono::steady_clock::now().time_since_epoch().count()) - t0, std::memory_order_relaxed);
        g_lookup_n[kind].fetch_add(1, std::memory_order_relaxed);
    }
};
nlohmann::json LookupReport() {
    static constexpr const char* names[] = {"view", "buffer", "sampler", "attachment"};
    nlohmann::json j = nlohmann::json::object();
    j["memo"] = {{"on", MemoOn()}, {"buffer_hits_misses", {c_memo_hits[0].load(), c_memo_misses[0].load()}}, {"view_hits_misses", {c_memo_hits[1].load(), c_memo_misses[1].load()}},
                 {"sampler_hits_misses", {c_memo_hits[2].load(), c_memo_misses[2].load()}}, {"epoch_bumps_buffer_view", {c_memo_bumps[0].load(), c_memo_bumps[1].load()}}};
    for (u32 i = 0; i < 4; ++i) {
        j[names[i]] = {{"calls", g_lookup_n[i].load()}, {"ms_total", g_lookup_ns[i].load() / 1000000}, {"us_mean", g_lookup_n[i].load() ? static_cast<double>(g_lookup_ns[i].load()) / 1e3 / static_cast<double>(g_lookup_n[i].load()) : 0.0}};
    }
    return j;
}

bool ViewForComputeLookup(u64 view_id, vk::ImageViewType want, vk::CommandBuffer cmd, vk::ImageView& out, vk::Image* image);
bool ViewForCompute(u64 view_id, vk::ImageViewType want, vk::CommandBuffer cmd, vk::ImageView& out, vk::Image* image) {
    const LookupTimer lookup_timer{0};
    if (ContractCensusOn()) { // api-contract.md C10 (debug): a shader reads a texture the target-7 mode drew into
        u64 texture = 0;
        {
            std::scoped_lock lock{mutex};
            if (const auto v = views.find(view_id); v != views.end()) {
                texture = v->second.texture_id;
            }
        }
        ContractSampled(texture);
    }
    if (!MemoOn()) {
        return ViewForComputeLookup(view_id, want, cmd, out, image);
    }
    if (!t_view_memo) {
        t_view_memo = std::make_unique<std::array<ViewMemo, MemoSize>>();
    }
    ViewMemo& m = (*t_view_memo)[MemoSlot(view_id, static_cast<u64>(want) + 1)];
    const u64 epoch = g_view_epoch.load(std::memory_order_acquire);
    if (m.epoch == epoch && m.id == view_id && m.want == static_cast<u32>(want)) {
        out = m.view;
        if (image != nullptr) {
            *image = m.image;
        }
        c_memo_hits[1].fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    c_memo_misses[1].fetch_add(1, std::memory_order_relaxed);
    vk::Image found{};
    if (!ViewForComputeLookup(view_id, want, cmd, out, &found)) {
        return false;
    }
    if (image != nullptr) {
        *image = found;
    }
    m = {view_id, epoch, static_cast<u32>(want), out, found}; // the epoch read before the lookup: a change during it leaves the entry stale-marked
    return true;
}
std::atomic<u64> c_cube_as_array{0};

vk::ImageViewType ModuleViewType(VkbAmdGpu::ImageType type) {
    static const bool cube_as_array = [] { const char* e = std::getenv("TLG_VKB_CUBE_AS_ARRAY"); return e == nullptr || e[0] != '0'; }();
    switch (type) {
    case VkbAmdGpu::ImageType::Color1D: return vk::ImageViewType::e2D; // note 41: the module declares 1D images as 2D
    case VkbAmdGpu::ImageType::Color1DArray: return vk::ImageViewType::e2DArray;
    case VkbAmdGpu::ImageType::Color2DArray:
    case VkbAmdGpu::ImageType::Color2DMsaaArray: return vk::ImageViewType::e2DArray;
    case VkbAmdGpu::ImageType::Color3D: return vk::ImageViewType::e3D;
    case VkbAmdGpu::ImageType::Cube:
        if (cube_as_array) {
            c_cube_as_array.fetch_add(1, std::memory_order_relaxed); // (per module image binding, at the variant's build)
            return vk::ImageViewType::e2DArray;
        }
        return vk::ImageViewType::eCube;
    default: return vk::ImageViewType::e2D;
    }
}

bool ViewForComputeLookup(u64 view_id, vk::ImageViewType want, vk::CommandBuffer cmd, vk::ImageView& out, vk::Image* image) {
    std::scoped_lock lock{mutex};
    const auto v = views.find(view_id);
    if (v == views.end()) {
        return false;
    }
    const auto i = images.find(v->second.texture_id);
    if (i == images.end()) {
        return false;
    }
    if (image != nullptr) {
        *image = i->second.image;
    }
    View& view = v->second;
    if (view.type != want) {
        using T = vk::ImageViewType;
        const u32 layers = view.range.layerCount;
        const bool allowed = (view.type == T::eCube && want == T::e2DArray) || (view.type == T::eCubeArray && want == T::e2DArray) ||
                             (view.type == T::e2D && want == T::e2DArray) || (view.type == T::e2DArray && want == T::e2D && layers == 1) ||
                             (view.type == T::e2DArray && (want == T::eCube || want == T::eCubeArray) && i->second.cube_compatible && layers % 6 == 0) ||
                             (view.type == T::e1D && want == T::e1DArray) || (view.type == T::e1DArray && want == T::e1D && layers == 1);
        const u32 slot = static_cast<u32>(want);
        if (!allowed || slot >= view.other.size()) {
            ++refused_pairs[vk::to_string(view.type) + " as " + vk::to_string(want) + " layers " + std::to_string(layers) + (i->second.cube ? " cube" : "")];
            c_other_view_refused.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!view.other[slot]) {
            vk::ImageSubresourceRange range = view.range;
            if (want == T::eCube) {
                range.layerCount = 6; // a cube read that does not index slices reads the first cube of the range
            } else if (want == T::e2D || want == T::e1D) {
                range.layerCount = 1;
            }
            const auto [result, handle] = instance->GetDevice().createImageView(
                {.image = i->second.image, .viewType = want, .format = view.format, .components = view.components, .subresourceRange = range}); // D4-07: the engine's channels
            if (result != vk::Result::eSuccess) {
                c_other_view_refused.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            view.other[slot] = handle;
            c_other_views.fetch_add(1, std::memory_order_relaxed);
        }
        out = view.other[slot];
        return true;
    }
    out = view.view;
    return true;
}

void CopyTextureAll(u64 destination_id, u64 source_id) {
    u32 mips = 0, layers = 0;
    {
        std::scoped_lock lock{mutex};
        const auto d = images.find(destination_id);
        const auto s = images.find(source_id);
        if (d == images.end() || s == images.end()) {
            c_tex_copy_no_image.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        mips = std::min(d->second.mips, s->second.mips);
        layers = d->second.type == vk::ImageType::e3D ? 1 : std::min(d->second.layers, s->second.layers);
        if (d->second.mips != s->second.mips) {
            c_tex_copy_mismatch.fetch_add(1, std::memory_order_relaxed);
            return; // the engine numbers subresources by its own mip count: both must agree
        }
    }
    for (u32 layer = 0; layer < layers; ++layer) {
        for (u32 mip = 0; mip < mips; ++mip) {
            (void)CopyTexture(destination_id, layer * mips + mip, source_id, layer * mips + mip);
        }
    }
}

namespace {
u32 TexelBytes(vk::Format f); // defined further down
}
bool CopyTexture(u64 destination_id, u32 destination_sub, u64 source_id, u32 source_sub) {
    if (!On() || !device_ready) {
        return false;
    }
    std::unique_lock lock{mutex};
    const auto d = images.find(destination_id);
    const auto s = images.find(source_id);
    if (d == images.end() || s == images.end()) {
        c_tex_copy_no_image.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    Image& to = d->second;
    Image& from = s->second;
    const u32 to_mip = destination_sub % std::max(to.mips, 1u), to_layer = destination_sub / std::max(to.mips, 1u);
    const u32 from_mip = source_sub % std::max(from.mips, 1u), from_layer = source_sub / std::max(from.mips, 1u);
    const auto extent = [](const Image& img, u32 mip) {
        return vk::Extent3D{std::max(img.width >> mip, 1u), std::max(img.height >> mip, 1u), img.type == vk::ImageType::e3D ? std::max(img.depth >> mip, 1u) : 1u};
    };
    const vk::Extent3D a = extent(to, to_mip), b = extent(from, from_mip);
    const bool layers_ok = (to.type == vk::ImageType::e3D || to_layer < to.layers) && (from.type == vk::ImageType::e3D || from_layer < from.layers);
    if (to.is_depth != from.is_depth || to.type != from.type || to_mip >= to.mips || from_mip >= from.mips || !layers_ok || a != b || to.image == from.image) {
        c_tex_copy_mismatch.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // a depth copy carries the stencil plane too where both images have one (api-contract review A4: ExCsDepthCopy, 0248e800, dispatches one raw copy over the
    // depth surface and one over the stencil surface)
    const vk::ImageAspectFlags aspect = !to.is_depth                                         ? vk::ImageAspectFlags{vk::ImageAspectFlagBits::eColor}
                                        : HasStencil(to.format) && HasStencil(from.format) ? (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)
                                                                                           : vk::ImageAspectFlags{vk::ImageAspectFlagBits::eDepth};
    const bool volume = to.type == vk::ImageType::e3D;
    const vk::ImageCopy region{.srcSubresource = {aspect, from_mip, volume ? 0u : from_layer, 1},
                               .dstSubresource = {aspect, to_mip, volume ? 0u : to_layer, 1},
                               .extent = a};
    lock.unlock(); // (no job is made under this module's lock: the recording thread's tick takes it)
    if (ContractCensusOn()) {
        std::scoped_lock census_lock{contract_mutex};
        contract_copy_destinations.insert(destination_id);
        ++contract_counts[fmt::format("W texture copy {} -> {}", vk::to_string(from.format), vk::to_string(to.format))];
    }
    Stream::NoteBytes(Stream::Work::Copy, static_cast<u64>(region.extent.width) * region.extent.height * region.extent.depth * TexelBytes(to.format));
    Stream::Defer(Stream::Work::Copy, [src = from.image, dst = to.image, region](vk::CommandBuffer cmd) {
        Stream::Hazard(cmd, {Stream::ImageUse(src, false), Stream::ImageUse(dst, true)});
        cmd.copyImage(src, vk::ImageLayout::eGeneral, dst, vk::ImageLayout::eGeneral, region);
    });
    c_tex_copies.fetch_add(1, std::memory_order_relaxed);
    return true;
}

namespace {
std::mutex census_mutex;
std::unordered_map<u64, u64> census_frame; ///< (in-place buffer, offset) -> the largest size bound this frame
std::atomic<u64> c_pattern_fills{0};
std::atomic<u64> c_census_binds[4][2]{}, c_census_bytes[4]{}, c_census_written[4]{}, c_census_unique_bytes{0}, c_census_unique_ranges{0}, c_census_frames{0},
    c_census_bucket[4]{};
void CensusFrame() {
    std::scoped_lock lock{census_mutex};
    if (census_frame.empty()) {
        return;
    }
    u64 bytes = 0;
    for (const auto& [key, size] : census_frame) {
        bytes += size;
        c_census_bucket[size < (4u << 10) ? 0 : size < (64u << 10) ? 1 : size < (1u << 20) ? 2 : 3].fetch_add(size, std::memory_order_relaxed);
    }
    c_census_unique_bytes.fetch_add(bytes, std::memory_order_relaxed);
    c_census_unique_ranges.fetch_add(census_frame.size(), std::memory_order_relaxed);
    c_census_frames.fetch_add(1, std::memory_order_relaxed);
    census_frame.clear();
}
} // namespace

void NoteBind(u32 kind, const BufferWindow& w, u64 size, bool written) {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_BIND_CENSUS"); return e != nullptr && e[0] == '1'; }();
    if (!on) {
        return;
    }
    kind = std::min<u32>(kind, 3);
    c_census_binds[kind][w.device ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
    if (w.device) {
        return;
    }
    c_census_bytes[kind].fetch_add(size, std::memory_order_relaxed);
    if (written) {
        c_census_written[kind].fetch_add(1, std::memory_order_relaxed);
    }
    const u64 key = (reinterpret_cast<u64>(static_cast<VkBuffer>(w.buffer)) << 20) ^ w.offset;
    std::scoped_lock lock{census_mutex};
    u64& s = census_frame[key];
    s = std::max(s, size);
}

/// tier-2 gap pass (debug, contract census): does the engine buffer at [address, address + size) live in our device memory (a UAV or dynamic buffer from its
/// creation)? Then the engine's memory under it is not updated by the GPU and a CPU read of that memory sees old data. No side effects (no lookup memo, no copy).
bool BufferOnDevice(u64 address, u64 size) {
    std::scoped_lock lock{mutex};
    u64 start = 0;
    bool straddle = false;
    return DeviceBufferOf(address, std::max<u64>(size, 1), start, straddle) != nullptr;
}

bool BufferGpuWrites(u64 address, u64 size) {
    std::scoped_lock lock{mutex};
    u64 start = 0;
    bool straddle = false;
    const DeviceBuffer* d = DeviceBufferOf(address, std::max<u64>(size, 1), start, straddle);
    return d != nullptr && d->gpu_writes;
}

bool FindBuffer(u64 address, u64 size, BufferWindow& out) {
    const LookupTimer lookup_timer{1};
    FillReadersCheck(address, size, 0);
    if (!device_ready || size == 0) {
        return false;
    }
    // note 53 bound, test only: TLG_VKB_TEST_DOUBLE_LOOKUP=1 runs every buffer lookup once more first, against the same memo state and without storing anything, so
    // the run pays exactly one extra lookup per call; the fps it loses is what the address lookups cost (the answer used is the normal one, the picture is right)
    static const bool double_lookup = [] { const char* e = std::getenv("TLG_VKB_TEST_DOUBLE_LOOKUP"); return e != nullptr && e[0] == '1'; }();
    if (double_lookup) {
        BufferWindow scratch;
        bool hit = false;
        if (MemoOn() && t_buffer_memo) {
            const BufferMemo& m = (*t_buffer_memo)[MemoSlot(address, size)];
            hit = m.epoch == g_buffer_epoch.load(std::memory_order_acquire) && m.address == address && m.size == size;
            if (hit) {
                scratch = m.window;
                c_memo_hits[0].fetch_add(0, std::memory_order_relaxed);
            }
        }
        if (!hit) {
            (void)FindLocked(address, size, scratch);
        }
        asm volatile("" : : "r"(scratch.offset) : "memory");
    }
    if (!MemoOn()) {
        return FindLocked(address, size, out);
    }
    if (!t_buffer_memo) {
        t_buffer_memo = std::make_unique<std::array<BufferMemo, MemoSize>>();
    }
    BufferMemo& m = (*t_buffer_memo)[MemoSlot(address, size)];
    const u64 epoch = g_buffer_epoch.load(std::memory_order_acquire);
    if (m.epoch == epoch && m.address == address && m.size == size) {
        out = m.window;
        c_memo_hits[0].fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    c_memo_misses[0].fetch_add(1, std::memory_order_relaxed);
    if (!FindLocked(address, size, out)) {
        return false;
    }
    if (out.device) { // in-place answers can still move to a device copy later: asked again each time
        m = {address, size, epoch, out};
    }
    return true;
}

void R3dTrace(const char* what, u64 address, u64 bytes) {
    if (R3dTraceBytes() == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    auto it = watched.upper_bound(address);
    if (it == watched.begin()) {
        return;
    }
    --it;
    const Watched& w = it->second;
    if (address >= w.end || w.end - it->first != R3dTraceBytes()) {
        return;
    }
    LOG_WARNING(Render_Vulkan, "vkb R3d trace: {} at {:#x} (+{:#x}, {} bytes) into buffer {:#x}: state {} first bind {} now {}", what, address, address - it->first, bytes,
                it->first, static_cast<u32>(w.state), w.first_bind_submit, GraphicsSubmits());
}
/// census (TLG_VKB_UPLOAD_CENSUS=1): CPU-written bytes by the engine buffer they land in, and by caller (Unmap / UpdateSubresource / other); /stats vkb_objects.upload_census
struct UploadCensusRow {
    u64 calls{}, bytes{}, copied_calls{}, copied_bytes{}, size{};
    u32 usage{}, bind{}, cpu{}, cls{};
    std::array<u64, 3> by_kind{};
    bool device{}, gpu_written{};
};
std::mutex upload_census_mutex;
std::unordered_map<u64, UploadCensusRow> upload_census;
bool UploadCensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_UPLOAD_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
void UploadCensusNote(u64 address, u64 size, bool copied) {
    u64 start = address;
    UploadCensusRow info;
    {
        std::scoped_lock lock{mutex};
        if (Watched* w = WatchedAt(address, size)) {
            start = std::prev(watched.upper_bound(address))->first;
            info.size = w->end - start, info.usage = w->usage, info.bind = w->bind, info.cpu = w->cpu_access, info.cls = w->cls;
            info.device = w->state == Watched::State::Device, info.gpu_written = w->gpu_written;
        }
    }
    std::scoped_lock lock{upload_census_mutex};
    UploadCensusRow& r = upload_census[start];
    r.size = info.size, r.usage = info.usage, r.bind = info.bind, r.cpu = info.cpu, r.cls = info.cls, r.device = info.device, r.gpu_written = info.gpu_written;
    ++r.calls;
    r.bytes += size;
    r.by_kind[std::min<u32>(t_cpu_write_kind, 2)] += size;
    if (copied) {
        ++r.copied_calls;
        r.copied_bytes += size;
    }
}
nlohmann::json UploadCensusReport() {
    std::scoped_lock lock{upload_census_mutex};
    std::vector<std::pair<u64, UploadCensusRow>> v(upload_census.begin(), upload_census.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.copied_bytes > b.second.copied_bytes; });
    u64 total = 0, copied = 0;
    std::array<u64, 3> kinds{};
    for (const auto& [a, r] : v) {
        total += r.bytes, copied += r.copied_bytes;
        for (u32 k = 0; k < 3; ++k) {
            kinds[k] += r.by_kind[k];
        }
    }
    nlohmann::json top = nlohmann::json::array();
    for (size_t i = 0; i < v.size() && i < 24; ++i) {
        const auto& [a, r] = v[i];
        top.push_back({{"buffer", fmt::format("{:#x}", a)}, {"size", r.size}, {"calls", r.calls}, {"mb", r.bytes >> 20}, {"copied_calls", r.copied_calls}, {"copied_mb", r.copied_bytes >> 20},
                       {"unmap_mb", r.by_kind[0] >> 20}, {"update_mb", r.by_kind[1] >> 20}, {"other_mb", r.by_kind[2] >> 20}, {"usage", r.usage}, {"bind", r.bind}, {"cpu", r.cpu},
                       {"cls", r.cls}, {"device_copy", r.device}, {"gpu_written", r.gpu_written}});
    }
    const double un = static_cast<double>(std::max<u64>(c_pc_unmaps.load(), 1));
    nlohmann::json pages = {{"big_unmaps", c_pc_unmaps.load()}, {"pages_per_unmap", c_pc_pages.load() / un}, {"changed_pages_per_unmap", c_pc_changed.load() / un},
                            {"first_changed_page_mean", c_pc_first_changed_sum.load() / un}, {"last_changed_page_mean", c_pc_last_changed_sum.load() / un}};
    return {{"big_buffer_pages", pages}, {"buffers", v.size()}, {"mb", total >> 20}, {"copied_mb", copied >> 20}, {"unmap_mb", kinds[0] >> 20}, {"update_mb", kinds[1] >> 20}, {"other_mb", kinds[2] >> 20}, {"top", top}};
}
void BufferCpuWritten(u64 address, u64 size) {
    R3dTrace("UpdateSubresource", address, size);
    if (!On() || !device_ready || size == 0) {
        return;
    }
    struct CensusAtEnd {
        u64 address, size;
        bool copied{false};
        ~CensusAtEnd() {
            if (UploadCensusOn()) {
                UploadCensusNote(address, size, copied);
            }
        }
    } census_at_end{address, size};
    if (DeviceBuffersMode() == 2) {
        // the GPU wrote our device copy (a copy or fill into it): it holds the buffer, the engine memory under it is older; the CPU's bytes go into it
        VkBuffer written_device{};
        u64 written_offset = 0;
        {
            std::scoped_lock lock{mutex};
            if (Watched* w = WatchedAt(address, size); w != nullptr && w->cls == 0 && w->state == Watched::State::Device && w->gpu_written) {
                written_device = w->device, written_offset = address - std::prev(watched.upper_bound(address))->first;
            }
        }
        static const bool skip_uploads_gw = [] { const char* e = std::getenv("TLG_VKB_TEST_SKIP_CPU_UPLOADS"); return e != nullptr && e[0] == '1'; }();
        if (written_device != VK_NULL_HANDLE) { // (no job is made under this module's lock)
            if (const BufferWindow src = SourceWindow(address, size); src.buffer && !skip_uploads_gw) {
                census_at_end.copied = true;
                Stream::NoteBytes(Stream::Work::UploadCpuWritten, size);
                Stream::Defer(Stream::Work::UploadCpuWritten, [src, device = written_device, offset = written_offset, size](vk::CommandBuffer cmd) {
                    if (Stream::HoistCopy(src.buffer, src.offset, vk::Buffer{device}, offset, size)) {
                        return;
                    }
                    Stream::Hazard(cmd, {Stream::BufferUse(src.buffer, src.offset, size, false), Stream::BufferUse(vk::Buffer{device}, offset, size, true)});
                    cmd.copyBuffer(src.buffer, vk::Buffer{device}, vk::BufferCopy{src.offset, offset, size});
                });
            }
            return;
        }
        std::scoped_lock lock{mutex};
        if (Watched* w = WatchedAt(address, size); w != nullptr && w->cls == 0 && w->state == Watched::State::Device) {
            w->state = Watched::State::Unstable;
            BumpBufferEpoch(); // note 46: A1 memo answers may change
            doomed.push_back({static_cast<u32>(GraphicsSubmits()) + DestroyDelay(), {}, {}, {}, w->device, w->allocation});
            w->device = VK_NULL_HANDLE;
            BumpBufferEpoch(); // note 46: A1 memo answers may change
            c_dev2_unstable.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (DeviceBuffersMode() == 1) {
        std::scoped_lock lock{mutex};
        if (Watched* w = WatchedAt(address, size); w != nullptr && w->cls == 0) {
            const u64 start = std::prev(watched.upper_bound(address))->first;
            w->hash = HashGuest(start, w->end - start); // the engine's own update: expected
            c_watch_cpu_updates.fetch_add(1, std::memory_order_relaxed);
        }
    }
    VkBuffer device_buffer{};
    u64 offset = 0;
    {
        std::scoped_lock lock{mutex};
        u64 start = 0;
        bool straddle = false;
        const DeviceBuffer* d = DeviceBufferOf(address, size, start, straddle);
        if (d == nullptr) {
            return;
        }
        device_buffer = d->buffer, offset = address - start;
        devbuf_hash.erase(start); // Y7 census: an announced write
    }
    const BufferWindow w = SourceWindow(address, size);
    if (!w.buffer) {
        c_copy_unresolved.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // GPU-side bound, kill experiment only (TLG_VKB_TEST_SKIP_CPU_UPLOADS=1, the picture is wrong): the GPU copies of CPU-written buffer data are dropped
    static const bool skip_uploads = [] { const char* e = std::getenv("TLG_VKB_TEST_SKIP_CPU_UPLOADS"); return e != nullptr && e[0] == '1'; }();
    if (skip_uploads) {
        return;
    }
    census_at_end.copied = true;
    Stream::NoteBytes(Stream::Work::UploadCpuWritten, size);
    Stream::Defer(Stream::Work::UploadCpuWritten, [w, device_buffer, offset, size](vk::CommandBuffer cmd) {
        if (Stream::HoistCopy(w.buffer, w.offset, vk::Buffer{device_buffer}, offset, size)) {
            return;
        }
        Stream::Hazard(cmd, {Stream::BufferUse(w.buffer, w.offset, size, false), Stream::BufferUse(vk::Buffer{device_buffer}, offset, size, true)});
        cmd.copyBuffer(w.buffer, vk::Buffer{device_buffer}, vk::BufferCopy{w.offset, offset, size});
    });
    c_cpu_writes.fetch_add(1, std::memory_order_relaxed);
    c_cpu_write_bytes.fetch_add(size, std::memory_order_relaxed);
}

namespace {
/// The size class of a format for a view of an image of another format (a view must have the same texel block size)
u32 TexelBytes(vk::Format f) {
    using F = vk::Format;
    switch (f) {
    case F::eR8Unorm: case F::eR8Snorm: case F::eR8Uint: case F::eR8Sint: case F::eR8Srgb: return 1;
    case F::eR8G8Unorm: case F::eR8G8Snorm: case F::eR8G8Uint: case F::eR8G8Sint: case F::eR16Unorm: case F::eR16Snorm: case F::eR16Uint: case F::eR16Sint:
    case F::eR16Sfloat: case F::eB5G6R5UnormPack16: case F::eR5G6B5UnormPack16: case F::eA1R5G5B5UnormPack16: case F::eB5G5R5A1UnormPack16: case F::eR4G4B4A4UnormPack16:
    case F::eB4G4R4A4UnormPack16: return 2;
    case F::eR8G8B8A8Unorm: case F::eR8G8B8A8Snorm: case F::eR8G8B8A8Uint: case F::eR8G8B8A8Sint: case F::eR8G8B8A8Srgb: case F::eB8G8R8A8Unorm: case F::eB8G8R8A8Srgb:
    case F::eA2B10G10R10UnormPack32: case F::eA2R10G10B10UnormPack32: case F::eA2B10G10R10UintPack32: case F::eB10G11R11UfloatPack32: case F::eE5B9G9R9UfloatPack32:
    case F::eR16G16Unorm: case F::eR16G16Snorm: case F::eR16G16Uint: case F::eR16G16Sint: case F::eR16G16Sfloat: case F::eR32Uint: case F::eR32Sint: case F::eR32Sfloat:
    case F::eA8B8G8R8UnormPack32: case F::eA8B8G8R8SrgbPack32: return 4;
    case F::eR16G16B16A16Unorm: case F::eR16G16B16A16Snorm: case F::eR16G16B16A16Uint: case F::eR16G16B16A16Sint: case F::eR16G16B16A16Sfloat: case F::eR32G32Uint:
    case F::eR32G32Sint: case F::eR32G32Sfloat: return 8;
    case F::eR32G32B32A32Uint: case F::eR32G32B32A32Sint: case F::eR32G32B32A32Sfloat: return 16;
    default: return 0;
    }
}

vk::SamplerAddressMode AddressOf(u32 d3d) {
    switch (d3d) {
    case 2: return vk::SamplerAddressMode::eMirroredRepeat;
    case 3: return vk::SamplerAddressMode::eClampToEdge;
    case 4: return vk::SamplerAddressMode::eClampToBorder;
    case 5: c_sampler_mirror_once.fetch_add(1, std::memory_order_relaxed); return vk::SamplerAddressMode::eMirroredRepeat;
    default: return vk::SamplerAddressMode::eRepeat;
    }
}

template <typename T>
T RdObj(u64 base, u64 offset) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(base + offset), sizeof(T));
    return value;
}
} // namespace

std::string DescribeView(u64 view_id) {
    std::scoped_lock lock{mutex};
    const auto v = views.find(view_id);
    if (v == views.end()) {
        return "unknown view";
    }
    const auto i = images.find(v->second.texture_id);
    if (i == images.end()) {
        return "view without image";
    }
    const Image& m = i->second;
    const View& w = v->second;
    return fmt::format("tex {} {}x{}x{} mips {} layers {} image {} | view {} {} mips {}+{} layers {}+{}{} | {} MB", w.texture_id, m.width, m.height, m.depth, m.mips, m.layers, vk::to_string(m.format),
                       vk::to_string(w.type), vk::to_string(w.format), w.range.baseMipLevel, w.range.levelCount, w.range.baseArrayLayer, w.range.layerCount,
                       m.is_depth ? " depth" : "", m.bytes >> 20);
}

/// note 41, debug only: copies the subresource of our view of engine view `view_id` (its first mip and layer) into a host buffer and writes it, once the GPU has
/// done this command buffer, to log/vkb_image_dump/<name>_<w>x<h>_<format>.bin (raw texels; depth images: the depth aspect). Recording thread, lock held, outside a pass.
void DumpImageCore(vk::CommandBuffer cmd, vk::Image image, vk::Format format, u32 width, u32 height, u32 slices, u32 mip, u32 layer, bool depth, const std::string& name);
/// note 91: the dump files are written by one thread in order (a thread per file was fine for one draw; a frame capture writes thousands)
namespace {
struct DumpTask {
    u64 after{};
    std::function<void()> work;
};
std::mutex dump_mutex;
std::condition_variable dump_cv;
std::deque<DumpTask> dump_tasks;
std::atomic<u64> dump_pending{0};
bool dump_thread_started = false;
void QueueDump(u64 after, std::function<void()> work) {
    dump_pending.fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lock{dump_mutex};
        dump_tasks.push_back({after, std::move(work)});
        if (!dump_thread_started) {
            dump_thread_started = true;
            std::thread([] {
                Common::SetCurrentThreadName("vkb:dumps");
                for (;;) {
                    DumpTask t;
                    {
                        std::unique_lock l{dump_mutex};
                        dump_cv.wait(l, [] { return !dump_tasks.empty(); });
                        t = std::move(dump_tasks.front());
                        dump_tasks.pop_front();
                    }
                    Stream::Wait(t.after);
                    t.work();
                    dump_pending.fetch_sub(1, std::memory_order_relaxed);
                }
            }).detach();
        }
    }
    dump_cv.notify_one();
}
} // namespace

u64 DumpsPending() {
    return dump_pending.load(std::memory_order_relaxed);
}

bool WaitDumps(u64 timeout_ms) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (dump_pending.load() != 0) {
        if (std::chrono::steady_clock::now() > end) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return true;
}

/// debug: the image behind an image view handle as bound (found by handle among our views), dumped as DumpView does; its view and texture ids go into the log
void DumpBoundView(vk::CommandBuffer cmd, vk::ImageView handle, const std::string& name) {
    u64 view_id = 0, texture_id = 0;
    {
        std::scoped_lock lock{mutex};
        for (const auto& [id, v] : views) {
            bool match = v.view == handle;
            for (const vk::ImageView o : v.other) {
                match = match || (o && o == handle);
            }
            for (const auto& [f, o] : v.formatted) {
                match = match || o == handle;
            }
            if (match) {
                view_id = id, texture_id = v.texture_id;
                break;
            }
        }
    }
    LOG_WARNING(Render_Vulkan, "vkb bound view dump {}: handle {:#x} -> view id {} texture id {}", name, reinterpret_cast<u64>(static_cast<VkImageView>(handle)), view_id,
                texture_id);
    if (view_id != 0) {
        DumpView(cmd, view_id, name);
    }
}
void DumpView(vk::CommandBuffer cmd, u64 view_id, const std::string& name) {
    vk::Image image{};
    vk::Format format{};
    u32 width = 0, height = 0, slices = 1, mip = 0, layer = 0;
    bool depth = false;
    {
        std::scoped_lock lock{mutex};
        const auto v = views.find(view_id);
        if (v == views.end()) {
            LOG_WARNING(Render_Vulkan, "vkb image dump {}: unknown view", name);
            return;
        }
        const auto i = images.find(v->second.texture_id);
        if (i == images.end()) {
            return;
        }
        image = i->second.image, format = i->second.format, depth = i->second.is_depth;
        mip = v->second.range.baseMipLevel, layer = v->second.range.baseArrayLayer;
        width = std::max(1u, i->second.width >> mip), height = std::max(1u, i->second.height >> mip);
        if (i->second.type == vk::ImageType::e3D) {
            slices = std::max(1u, i->second.depth >> mip); // the whole volume (slices one after another)
        }
    }
    DumpImageCore(cmd, image, format, width, height, slices, mip, layer, depth, name);
    // debug (TLG_VKB_DUMP_MIPS=1): the view's further mips too, as <name>_mip<n>
    static const bool all_mips = [] { const char* e = std::getenv("TLG_VKB_DUMP_MIPS"); return e != nullptr && e[0] == '1'; }();
    if (all_mips && !depth && slices == 1) {
        u32 levels = 1, full_w = 0, full_h = 0;
        {
            std::scoped_lock lock{mutex};
            if (const auto v = views.find(view_id); v != views.end()) {
                if (const auto i = images.find(v->second.texture_id); i != images.end()) {
                    levels = v->second.range.levelCount == VK_REMAINING_MIP_LEVELS ? i->second.mips - mip : v->second.range.levelCount;
                    full_w = i->second.width, full_h = i->second.height;
                }
            }
        }
        for (u32 m = 1; m < levels; ++m) {
            DumpImageCore(cmd, image, format, std::max(1u, full_w >> (mip + m)), std::max(1u, full_h >> (mip + m)), 1, mip + m, layer, false, fmt::format("{}_mip{}", name, mip + m));
        }
    }
}

void DumpImageCore(vk::CommandBuffer cmd, vk::Image image, vk::Format format, u32 width, u32 height, u32 slices, u32 mip, u32 layer, bool depth, const std::string& name) {
    u32 texel = 0;
    switch (format) {
    case vk::Format::eR8Unorm: texel = 1; break;
    case vk::Format::eR8G8Unorm: case vk::Format::eR16Unorm: case vk::Format::eR16Sfloat: case vk::Format::eD16Unorm: texel = 2; break;
    case vk::Format::eR8G8B8A8Unorm: case vk::Format::eR8G8B8A8Srgb: case vk::Format::eB8G8R8A8Unorm: case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eA2B10G10R10UnormPack32: case vk::Format::eB10G11R11UfloatPack32: case vk::Format::eR32Sfloat: case vk::Format::eR32Uint:
    case vk::Format::eR16G16Sfloat: case vk::Format::eR16G16Unorm: case vk::Format::eD32Sfloat: case vk::Format::eD32SfloatS8Uint: case vk::Format::eE5B9G9R9UfloatPack32:
        texel = 4; break;
    case vk::Format::eR16G16B16A16Sfloat: case vk::Format::eR32G32Sfloat: case vk::Format::eR16G16B16A16Unorm: texel = 8; break;
    case vk::Format::eR32G32B32A32Sfloat: texel = 16; break;
    default: break;
    }
    u32 block = 0; // compressed formats: bytes per 4x4 block (raw blocks are written)
    switch (format) {
    case vk::Format::eBc1RgbaUnormBlock: case vk::Format::eBc1RgbaSrgbBlock: case vk::Format::eBc1RgbUnormBlock: case vk::Format::eBc1RgbSrgbBlock:
    case vk::Format::eBc4UnormBlock: case vk::Format::eBc4SnormBlock: block = 8; break;
    case vk::Format::eBc2UnormBlock: case vk::Format::eBc2SrgbBlock: case vk::Format::eBc3UnormBlock: case vk::Format::eBc3SrgbBlock: case vk::Format::eBc5UnormBlock:
    case vk::Format::eBc5SnormBlock: case vk::Format::eBc6HUfloatBlock: case vk::Format::eBc6HSfloatBlock: case vk::Format::eBc7UnormBlock: case vk::Format::eBc7SrgbBlock:
        block = 16; break;
    default: break;
    }
    if (texel == 0 && block == 0) {
        LOG_WARNING(Render_Vulkan, "vkb image dump {}: format {} not handled", name, vk::to_string(format));
        return;
    }
    const u64 bytes = (block != 0 ? u64{(width + 3) / 4} * ((height + 3) / 4) * block : u64{width} * height * texel) * slices;
    const VkBufferCreateInfo ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance->GetAllocator(), &ci, &alloc, &buffer, &allocation, &info) != VK_SUCCESS || info.pMappedData == nullptr) {
        return;
    }
    const vk::MemoryBarrier2 before{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                    .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands, .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &before});
    const vk::BufferImageCopy region{.bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
                                     .imageSubresource = {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor, mip, layer, 1},
                                     .imageOffset = {0, 0, 0}, .imageExtent = {width, height, slices}};
    cmd.copyImageToBuffer(image, vk::ImageLayout::eGeneral, vk::Buffer{buffer}, region);
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &before});
    const u64 after = Stream::RecordingValue(); // this command buffer goes out with that submission
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_image_dump";
    const std::string file = (dir / (slices > 1 ? fmt::format("{}_{}x{}x{}_{}.bin", name, width, height, slices, vk::to_string(format))
                                                 : fmt::format("{}_{}x{}_{}.bin", name, width, height, vk::to_string(format)))).string();
    QueueDump(after, [buffer, allocation, mapped = info.pMappedData, bytes, after, dir, file] {
        Stream::Wait(after);
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(file).parent_path(), ec);
        if (FILE* f = std::fopen(file.c_str(), "wb")) {
            std::fwrite(mapped, 1, bytes, f);
            std::fclose(f);
        }
        vmaDestroyBuffer(instance->GetAllocator(), buffer, allocation);
        LOG_INFO(Render_Vulkan, "vkb image dump written: {}", file);
    });
}

bool ImageFacts(vk::Image handle, vk::Format& format, u32& width, u32& height, bool& depth) {
    std::scoped_lock lock{mutex};
    for (const auto& [id, img] : images) {
        if (img.image == handle) {
            format = img.format, width = img.width, height = img.height, depth = img.is_depth;
            return true;
        }
    }
    return false;
}

void DumpImageHandle(vk::CommandBuffer cmd, vk::Image handle, const std::string& name) {
    vk::Format format{};
    u32 width = 0, height = 0;
    bool depth = false, found = false;
    {
        std::scoped_lock lock{mutex};
        for (const auto& [id, img] : images) {
            if (img.image == handle) {
                format = img.format, width = img.width, height = img.height, depth = img.is_depth, found = true;
                break;
            }
        }
    }
    if (!found) {
        LOG_WARNING(Render_Vulkan, "vkb image dump {}: unknown image", name);
        return;
    }
    DumpImageCore(cmd, handle, format, width, height, 1, 0, 0, depth, name);
}

void DumpBuffer(vk::CommandBuffer cmd, vk::Buffer source, u64 offset, u64 size, const std::string& name) {
    if (!source || size == 0 || size > (256ull << 20)) {
        return;
    }
    const VkBufferCreateInfo ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance->GetAllocator(), &ci, &alloc, &buffer, &allocation, &info) != VK_SUCCESS || info.pMappedData == nullptr) {
        return;
    }
    const vk::MemoryBarrier2 barrier{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                     .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands, .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    cmd.copyBuffer(source, vk::Buffer{buffer}, vk::BufferCopy{offset, 0, size});
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    const u64 after = Stream::RecordingValue();
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_image_dump";
    const std::string file = (dir / fmt::format("{}_{}B.gpu", name, size)).string();
    QueueDump(after, [buffer, allocation, mapped = info.pMappedData, size, after, dir, file] {
        Stream::Wait(after);
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(file).parent_path(), ec);
        if (FILE* f = std::fopen(file.c_str(), "wb")) {
            std::fwrite(mapped, 1, size, f);
            std::fclose(f);
        }
        vmaDestroyBuffer(instance->GetAllocator(), buffer, allocation);
    });
}

/// debug (TLG_VKB_R3D_STALE_CHECK=<graphics submit>): at the census trigger, every static buffer copied to device memory at or after that submit (R3d / settling) is read
/// back and compared with the engine memory under it; a mismatch (a write our copy never got) is logged with the buffer's description
void R3dStaleCheck(const std::string& label) {
    static const u64 from = [] { const char* e = std::getenv("TLG_VKB_R3D_STALE_CHECK"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull; }();
    if (from == 0 || !device_ready) {
        return;
    }
    struct Item {
        u64 address, bytes, id, first_bind;
        u32 usage, bind, cpu;
        VkBuffer device;
        bool gpu_written;
    };
    std::vector<Item> items;
    u64 total = 0;
    {
        std::scoped_lock lock{mutex};
        for (const auto& [start, w] : watched) {
            if (w.state == Watched::State::Device && w.device != VK_NULL_HANDLE && w.first_bind_submit >= from && total + (w.end - start) <= (1024ull << 20)) {
                items.push_back({start, w.end - start, w.id, w.first_bind_submit, w.usage, w.bind, w.cpu_access, w.device, w.gpu_written});
                total += w.end - start;
            }
        }
    }
    LOG_WARNING(Render_Vulkan, "vkb R3d stale check {}: {} device copies made since submit {} ({} MiB)", label, items.size(), from, total >> 20);
    if (items.empty()) {
        return;
    }
    const VkBufferCreateInfo ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer host{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance->GetAllocator(), &ci, &alloc, &host, &allocation, &info) != VK_SUCCESS || info.pMappedData == nullptr) {
        return;
    }
    Stream::Defer(Stream::Work::Other, [items, host, allocation, mapped = info.pMappedData, label](vk::CommandBuffer cmd) {
        const vk::MemoryBarrier2 barrier{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                         .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands, .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        u64 at = 0;
        for (const Item& it : items) {
            cmd.copyBuffer(vk::Buffer{it.device}, vk::Buffer{host}, vk::BufferCopy{0, at, it.bytes});
            at += it.bytes;
        }
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        const u64 after = Stream::RecordingValue();
        QueueDump(after, [items, host, allocation, mapped, label, after] {
            Stream::Wait(after);
            u64 at = 0, stale = 0, cpu_side = 0;
            for (const Item& it : items) {
                const u8* ours = static_cast<const u8*>(mapped) + at;
                const u8* engine = reinterpret_cast<const u8*>(it.address);
                if (std::memcmp(ours, engine, it.bytes) != 0) {
                    u64 diff = 0, first = ~0ull, last = 0;
                    for (u64 k = 0; k < it.bytes; ++k) {
                        if (ours[k] != engine[k]) {
                            ++diff, first = std::min(first, k), last = k;
                        }
                    }
                    ++stale;
                    // GPU-written: our copy is newer by design (note 94). Not GPU-written: the CPU changed the engine memory after our first-bind copy without an API call (a PS4 habit)
                    cpu_side += it.gpu_written ? 0 : 1;
                    LOG_WARNING(Render_Vulkan, "vkb R3d stale check {}: STALE {:#x} {} B id {} usage {} bind {:#x} cpu {:#x} first bind {} {}: {} bytes differ (+{:#x}..+{:#x})", label,
                                it.address, it.bytes, it.id, it.usage, it.bind, it.cpu, it.first_bind, it.gpu_written ? "gpu-written" : "CPU-SIDE", diff, first, last);
                }
                at += it.bytes;
            }
            LOG_WARNING(Render_Vulkan, "vkb R3d stale check {}: {} of {} copies differ from engine memory, {} of them not GPU-written (CPU writes outside the API)", label, stale, items.size(), cpu_side);
            vmaDestroyBuffer(instance->GetAllocator(), host, allocation);
        });
    });
}

// note 42: a staging texture's data (CopyResource into a usage-3 texture, then Map READ) comes from host buffers our copies fill. Each staging subresource has a ring
// of three: the engine maps right after the copy, before the submission that carries it, so a Map reads the newest copy that is submitted (one frame older, as the
// console's unified memory shows whatever the GPU has written by then)
struct StagingCopy {
    VkBuffer buffer{};
    VmaAllocation allocation{};
    void* mapped{};
    u64 value{~0ull}; ///< the timeline value at which the copy is done; ~0 until recorded
    u64 serial{};     ///< the copy's order among this subresource's copies (0 = never filled)
};
struct StagingRing {
    std::array<StagingCopy, 3> copies{};
    u32 width{}, height{}, depth{1}, texel{};
    vk::Format format{}; ///< the image format of the newest copy (for the developer dump)
    u64 serial{};
};
std::mutex staging_mutex;
std::map<std::pair<u64, u32>, StagingRing> staging_copies;
std::atomic<u64> c_staging_copies{0}, c_staging_reads{0}, c_staging_unhandled{0}, c_staging_unready{0}, c_staging_older{0}, c_staging_retired{0};

void RetireStaging(u64 texture_id, u32 due) {
    std::scoped_lock lock{staging_mutex};
    for (auto it = staging_copies.lower_bound({texture_id, 0u}); it != staging_copies.end() && it->first.first == texture_id;) {
        for (auto& c : it->second.copies) {
            if (c.buffer != VK_NULL_HANDLE) {
                doomed.push_back({due, {}, {}, {}, c.buffer, c.allocation}); // freed once the GPU is past `due`, like the image
            }
        }
        it = staging_copies.erase(it);
        c_staging_retired.fetch_add(1, std::memory_order_relaxed);
    }
}

void StagingCopied(u64 texture_id, u32 sub) {
    vk::Image image{};
    vk::Format format{};
    u32 width = 0, height = 0, depth = 1, mip = 0, layer = 0;
    {
        std::scoped_lock lock{mutex};
        const auto i = images.find(texture_id);
        if (i == images.end() || i->second.is_depth) {
            c_staging_unhandled.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const Image& img = i->second;
        mip = sub % std::max(img.mips, 1u), layer = sub / std::max(img.mips, 1u);
        image = img.image, format = img.format;
        width = std::max(img.width >> mip, 1u), height = std::max(img.height >> mip, 1u);
        if (img.type == vk::ImageType::e3D) {
            // tier-1-readback-findings.md D4-03: a volume's subresource is one mip with all its depth slices (the engine's CopySubresource copies them all)
            depth = std::max(img.depth >> mip, 1u), layer = 0;
        }
    }
    const u32 texel = TexelBytes(format);
    if (texel == 0) {
        c_staging_unhandled.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    VkBuffer buffer{};
    u32 index = 0;
    u64 serial = 0;
    {
        std::scoped_lock lock{staging_mutex};
        auto& ring = staging_copies[{texture_id, sub}];
        if (ring.width != width || ring.height != height || ring.depth != depth || ring.texel != texel) {
            for (auto& c : ring.copies) {
                if (c.buffer != VK_NULL_HANDLE) {
                    if (c.value != ~0ull) {
                        Stream::Wait(c.value);
                    }
                    vmaDestroyBuffer(instance->GetAllocator(), c.buffer, c.allocation);
                }
                c = {};
            }
            ring.width = width, ring.height = height, ring.depth = depth, ring.texel = texel;
        }
        ring.format = format;
        serial = ++ring.serial;
        index = u32(serial % ring.copies.size());
        auto& slot = ring.copies[index];
        if (slot.buffer == VK_NULL_HANDLE) {
            const VkBufferCreateInfo ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = u64{width} * height * depth * texel, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
            VmaAllocationCreateInfo alloc{};
            alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo info{};
            if (vmaCreateBuffer(instance->GetAllocator(), &ci, &alloc, &slot.buffer, &slot.allocation, &info) != VK_SUCCESS) {
                slot = {};
                c_staging_unhandled.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            slot.mapped = info.pMappedData;
        } else if (slot.value != ~0ull) {
            Stream::Wait(slot.value); // three copies back: long done in practice
        }
        slot.value = ~0ull, slot.serial = serial;
        buffer = slot.buffer;
    }
    Stream::Defer(Stream::Work::Copy, [image, buffer, width, height, depth, mip, layer, index, serial, key = std::pair<u64, u32>{texture_id, sub}](vk::CommandBuffer cmd) {
        Stream::Hazard(cmd, {Stream::ImageUse(image, false)});
        const vk::BufferImageCopy region{.imageSubresource = {vk::ImageAspectFlagBits::eColor, mip, layer, 1}, .imageExtent = {width, height, depth}};
        cmd.copyImageToBuffer(image, vk::ImageLayout::eGeneral, vk::Buffer{buffer}, region);
        const vk::MemoryBarrier2 to_host{.srcStageMask = vk::PipelineStageFlagBits2::eCopy, .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                         .dstStageMask = vk::PipelineStageFlagBits2::eHost, .dstAccessMask = vk::AccessFlagBits2::eHostRead};
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_host});
        std::scoped_lock lock{staging_mutex};
        if (const auto it = staging_copies.find(key); it != staging_copies.end() && it->second.copies[index].serial == serial) {
            it->second.copies[index].value = Stream::RecordingValue(); // this command buffer goes out with that submission
        }
    });
    c_staging_copies.fetch_add(1, std::memory_order_relaxed);
}

bool StagingRead(u64 texture_id, u32 sub, u8* out, u32 row_pitch, u32 depth_pitch) {
    StagingCopy copy;
    u32 width = 0, height = 0, depth = 1, texel = 0;
    vk::Format format{};
    bool newest = true;
    {
        std::scoped_lock lock{staging_mutex};
        const auto it = staging_copies.find({texture_id, sub});
        if (it == staging_copies.end()) {
            return false;
        }
        const StagingRing& ring = it->second;
        const u64 submitted = Stream::LastValue();
        const StagingCopy* best = nullptr;
        for (const auto& c : ring.copies) {
            if (c.buffer != VK_NULL_HANDLE && c.serial != 0 && c.value != ~0ull && c.value <= submitted && (best == nullptr || c.serial > best->serial)) {
                best = &c;
            }
        }
        if (best == nullptr) {
            c_staging_unready.fetch_add(1, std::memory_order_relaxed); // nothing submitted yet (the first frames)
            return false;
        }
        newest = best->serial == ring.serial;
        copy = *best, width = ring.width, height = ring.height, depth = ring.depth, texel = ring.texel, format = ring.format;
    }
    if (!newest) {
        c_staging_older.fetch_add(1, std::memory_order_relaxed);
    }
    Stream::Wait(copy.value);
    // tier-1-readback-findings.md D4-04: the allocation may be host-visible without being host-coherent; the GPU's writes are visible to this read only after an invalidate
    // (a no-op on coherent memory)
    vmaInvalidateAllocation(instance->GetAllocator(), copy.allocation, 0, VK_WHOLE_SIZE);
    const u32 row = width * texel;
    const u64 out_row = std::max(row_pitch, row);
    const u64 out_slice = std::max<u64>(depth_pitch, out_row * height); // D4-03: a volume mip's slices, at the depth pitch the engine's Map returned
    for (u32 z = 0; z < depth; ++z) {
        for (u32 y = 0; y < height; ++y) {
            std::memcpy(out + z * out_slice + u64{y} * out_row, static_cast<const u8*>(copy.mapped) + (u64{z} * height + y) * row, row);
        }
    }
    DevLog::Count(DevLog::Kind::StagingRead);
    // a developer dump (Vkb::DevLog): every 50th read's values, each 4 byte texel read as a float whatever the format (a two-channel 16 bit texture shows huge, -nan and inf values)
    static std::mutex log_mutex;
    static std::map<u32, u32> logged_by_size;
    bool log_it = false;
    if (DevLog::On()) {
        std::scoped_lock lock{log_mutex};
        u32& k = logged_by_size[static_cast<u32>(texture_id)];
        log_it = (k++ % 50) == 0 && k < 1000; // every 50th read of each staging texture
    }
    if (const u64 n = c_staging_reads.fetch_add(1, std::memory_order_relaxed); log_it && texel == 4) {
        const float* f = static_cast<const float*>(copy.mapped);
        float lo = f[0], hi = f[0], sum = 0;
        for (u32 i = 0; i < width * height; ++i) {
            lo = std::min(lo, f[i]), hi = std::max(hi, f[i]), sum += f[i];
        }
        u32 zeros = 0;
        for (u32 i = 0; i < width * height; ++i) {
            zeros += f[i] == 0.0f ? 1 : 0;
        }
        LOG_INFO(Render_Vulkan, "staging read {}: tex {} {}x{} {} {} min {} max {} mean {} zeros {} (as floats)", n, texture_id, width, height, vk::to_string(format), newest ? "newest" : "older", lo, hi,
                    sum / float(width * height), zeros);
    }
    return true;
}

std::string DescribeSampler(u64 object) {
    if (object == 0) {
        return "none";
    }
    const u64 d = object + 0x30;
    return fmt::format("filter {:#x} address {},{},{} bias {} aniso {} compare {} lod {}..{}", RdObj<u32>(d, 0), RdObj<u32>(d, 4), RdObj<u32>(d, 8), RdObj<u32>(d, 12),
                       RdObj<float>(d, 16), RdObj<u32>(d, 20), RdObj<u32>(d, 24), RdObj<float>(d, 44), RdObj<float>(d, 48));
}

vk::Sampler SamplerForLookup(u64 object, u64 id, u64 key, bool compare, s32 bias_q, u32 aniso);
vk::Sampler SamplerFor(u64 object, bool compare, s32 bias_q, u32 aniso) {
    const LookupTimer lookup_timer{2};
    if (!device_ready || object == 0) {
        return {};
    }
    const u64 id = RdObj<u64>(object, 0x10);
    // note 40: a compare sampler only for the shaders that sample with a compare; issue 42: one per bias and anisotropy setting (bias -32..31, aniso 0..31)
    const u64 key = ((id * 2 + (compare ? 1 : 0)) << 11) | (static_cast<u64>(std::clamp(bias_q, -32, 31) + 32) << 5) | std::min(aniso, 31u);
    SamplerMemo* memo = nullptr;
    if (MemoOn()) {
        if (!t_sampler_memo) {
            t_sampler_memo = std::make_unique<std::array<SamplerMemo, MemoSize>>();
        }
        memo = &(*t_sampler_memo)[MemoSlot(key, 7)];
        if (memo->key == key) {
            c_memo_hits[2].fetch_add(1, std::memory_order_relaxed);
            return memo->sampler;
        }
        c_memo_misses[2].fetch_add(1, std::memory_order_relaxed);
    }
    const vk::Sampler found = SamplerForLookup(object, id, key, compare, bias_q, aniso);
    if (memo != nullptr && found) {
        *memo = {key, found}; // samplers are never destroyed: one per engine sampler id
    }
    return found;
}
vk::Sampler SamplerForLookup(u64 object, u64 id, u64 key, bool compare, s32 bias_q, u32 aniso_setting) {
    std::scoped_lock lock{mutex};
    if (const auto it = samplers.find(key); it != samplers.end()) {
        return it->second;
    }
    // note 40: the engine sampler's fields as its PS4 code turns them into the S# (native_image_desc.cpp DeriveSampler, proved equal to the descriptor words):
    // the filter is the engine's own value (0 point, 0x7 and 0x10 trilinear, 0x8 anisotropic), the address modes are the console's numbering
    // (0 wrap, 1 mirror, 2 clamp to the last texel, ...), the comparison function is D3D's (1 never .. 8 always), the border is the S#'s type 0 (transparent black)
    const u64 d = object + 0x30;
    const u32 filter = RdObj<u32>(d, 0);
    const std::array<u32, 3> address{RdObj<u32>(d, 4), RdObj<u32>(d, 8), RdObj<u32>(d, 12)};
    const float bias = RdObj<float>(d, 16);
    const u32 aniso = RdObj<u32>(d, 20);
    const u32 compare_func = RdObj<u32>(d, 24);
    float min_lod = RdObj<float>(d, 44), max_lod = RdObj<float>(d, 48);
    // D4-14 (02492e40 setLodRange): the engine's S# holds each limit as unsigned 4.8 fixed point: negative (or NaN) -> 0, truncated to 1/256, at most 0xfff
    const auto lod_4_8 = [](float v) { return v >= 0.0f ? static_cast<float>(std::min(static_cast<u32>(std::min(v, 16.0f) * 256.0f), 0xfffu)) / 256.0f : 0.0f; };
    min_lod = lod_4_8(min_lod), max_lod = lod_4_8(max_lod);
    if (ContractCensusOn()) {
        ContractNote(fmt::format("S sampler filter {} address {} {} {} bias {} aniso {} compare {} cmp_used {} lod {} {}", filter, address[0], address[1], address[2], bias, aniso,
                                 compare_func, compare ? 1 : 0, min_lod, max_lod));
    }
    // api-contract.md S1 / S2 (JsDevice_CreateSamplerState 02492e40): the filter is `Filter % 9` into the engine's three tables (mag 025e0840, min 025e0870,
    // mip 025e08a0: point / linear / anisotropic per index; 9..17 are the same nine with a comparison); the address mode is the engine's value 0 wrap, 1 mirror,
    // 2 clamp, 3 border, 4 mirror once (table 025e08d0 -> GNM 0, 1, 2, 6, 3); the maximum anisotropy is rounded up to 1 / 2 / 4 / 8 / 16
    static constexpr u8 MagTable[9] = {0, 0, 1, 1, 0, 0, 1, 1, 3}; // 0 point, 1 linear, 3 anisotropic
    static constexpr u8 MinTable[9] = {0, 0, 0, 0, 1, 1, 1, 1, 3};
    static constexpr u8 MipTable[9] = {1, 2, 1, 2, 1, 2, 1, 2, 2}; // 1 point, 2 linear
    const u32 f = filter % 9;
    const bool mag_linear = MagTable[f] != 0, min_linear = MinTable[f] != 0, mip_linear = MipTable[f] == 2;
    // issue 42 (part a): the anisotropic filtering setting, with the normal renderer's rule (Core::NativeDesc::ForcedAnisotropy); the engine's mip table
    // (api-contract.md S1) holds only point and linear, never "no mip filter"
    const u32 forced = Core::NativeDesc::ForcedAnisotropy(aniso_setting, min_linear, MipTable[f] != 0, max_lod, compare);
    const bool anisotropic = (MagTable[f] == 3 || forced != 0) && instance->IsAnisotropicFilteringSupported();
    const float game_level = MagTable[f] != 3 ? 1.0f : aniso <= 1 ? 1.0f : aniso == 2 ? 2.0f : aniso <= 4 ? 4.0f : aniso <= 8 ? 8.0f : 16.0f;
    const float aniso_level = forced != 0 ? std::max(game_level, static_cast<float>(forced)) : game_level;
    const auto address_mode = [](u32 mode) {
        switch (mode) {
        case 0: return vk::SamplerAddressMode::eRepeat;
        case 1: return vk::SamplerAddressMode::eMirroredRepeat;
        case 2: return vk::SamplerAddressMode::eClampToEdge;
        case 3: return vk::SamplerAddressMode::eClampToBorder; // the border colour: type 0, transparent black
        case 4: // D4-15: mirror once needs the device feature; without it the nearest valid mode (one mirror lost past the first repeat)
            c_sampler_mirror_once.fetch_add(1, std::memory_order_relaxed);
            return instance->IsMirrorClampToEdgeSupported() ? vk::SamplerAddressMode::eMirrorClampToEdge : vk::SamplerAddressMode::eMirroredRepeat;
        default: c_samplers_invalid.fetch_add(1, std::memory_order_relaxed); return vk::SamplerAddressMode::eRepeat; // the engine asserts above 4
        }
    };
    if (compare_func < 1 || compare_func > 8) {
        c_samplers_invalid.fetch_add(1, std::memory_order_relaxed);
    }
    const vk::SamplerCreateInfo ci{
        .magFilter = mag_linear ? vk::Filter::eLinear : vk::Filter::eNearest,
        .minFilter = min_linear ? vk::Filter::eLinear : vk::Filter::eNearest,
        .mipmapMode = mip_linear ? vk::SamplerMipmapMode::eLinear : vk::SamplerMipmapMode::eNearest,
        .addressModeU = address_mode(address[0]),
        .addressModeV = address_mode(address[1]),
        .addressModeW = address_mode(address[2]),
        // api-contract.md S3: the engine never writes the S#'s bias field (MipLODBias is ignored); issue 42 (part b): the scene texture bias while the AA runs
        .mipLodBias = std::clamp(static_cast<float>(bias_q) * 0.25f, -instance->MaxSamplerLodBias(), instance->MaxSamplerLodBias()),
        .anisotropyEnable = anisotropic,
        .maxAnisotropy = anisotropic ? std::min(aniso_level, instance->MaxSamplerAnisotropy()) : 1.0f,
        .compareEnable = compare,
        .compareOp = static_cast<vk::CompareOp>(compare_func >= 1 && compare_func <= 8 ? compare_func - 1 : 7),
        .minLod = min_lod,
        .maxLod = std::max(min_lod, max_lod),
        .borderColor = vk::BorderColor::eFloatTransparentBlack,
    };
    const auto [result, sampler] = instance->GetDevice().createSampler(ci);
    if (result != vk::Result::eSuccess) {
        return {};
    }
    c_samplers_made.fetch_add(1, std::memory_order_relaxed);
    samplers.emplace(key, sampler);
    return sampler;
}

std::atomic<u64> g_display_texture{0}; ///< note 44: the engine texture id of the last image handed to the presenter
bool DisplayImageAt(u64 address, DisplayImage& out) {
    if (!On() || !device_ready) {
        return false;
    }
    std::scoped_lock lock{mutex};
    const auto at = image_at_address.find(address);
    if (at == image_at_address.end()) {
        c_display_missed.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const auto i = images.find(at->second);
    if (i == images.end() || i->second.is_depth) {
        c_display_missed.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    Image& img = i->second;
    if (img.layout != vk::ImageLayout::eGeneral) {
        c_display_not_general.fetch_add(1, std::memory_order_relaxed); // never drawn into: the presenter's barrier starts from undefined, our tracking says general from here on
        img.layout = vk::ImageLayout::eGeneral;
    }
    out = {img.image, img.format, img.width, img.height, img.mutable_format};
    g_display_texture.store(at->second, std::memory_order_relaxed);
    c_display_found.fetch_add(1, std::memory_order_relaxed);
    return true;
}

vk::Format ViewFormat(u64 view_id) {
    std::scoped_lock lock{mutex};
    const auto v = views.find(view_id);
    return v != views.end() ? v->second.format : vk::Format::eUndefined;
}

bool ViewExtent(u64 view_id, u32& width, u32& height, u32* depth, u64* texture_id) {
    std::scoped_lock lock{mutex};
    const auto v = views.find(view_id);
    if (v == views.end()) {
        return false;
    }
    const auto i = images.find(v->second.texture_id);
    if (i == images.end()) {
        return false;
    }
    width = std::max(1u, i->second.width >> v->second.range.baseMipLevel), height = std::max(1u, i->second.height >> v->second.range.baseMipLevel);
    if (depth != nullptr) {
        *depth = i->second.type == vk::ImageType::e3D ? std::max(1u, i->second.depth >> v->second.range.baseMipLevel) : 1u;
    }
    if (texture_id != nullptr) {
        *texture_id = v->second.texture_id;
    }
    return true;
}

bool AttachmentFor(u64 view_id, vk::Format format, vk::CommandBuffer cmd, Attachment& out) {
    const LookupTimer lookup_timer{3};
    std::scoped_lock lock{mutex};
    const auto v = views.find(view_id);
    if (v == views.end()) {
        return false;
    }
    const auto i = images.find(v->second.texture_id);
    if (i == images.end()) {
        return false;
    }
    View& view = v->second;
    Image& img = i->second;
    vk::ImageView handle = view.view;
    if (format != vk::Format::eUndefined && format != view.format) {
        handle = vk::ImageView{};
        for (const auto& [f, h] : view.formatted) {
            if (f == format) {
                handle = h;
            }
        }
        if (!handle && !img.mutable_format) {
            c_fixed_attachment_refused.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!handle) {
            const u32 a = TexelBytes(format), b = TexelBytes(view.format);
            if (img.is_depth || a == 0 || a != b) {
                c_attach_refused.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            const auto [result, made] = instance->GetDevice().createImageView({.image = img.image, .viewType = view.type, .format = format, .subresourceRange = view.range});
            if (result != vk::Result::eSuccess) {
                c_attach_refused.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            view.formatted.emplace_back(format, made);
            handle = made;
            c_attach_views.fetch_add(1, std::memory_order_relaxed);
        }
    }
    const u32 mip = view.range.baseMipLevel;
    out = {handle, {std::max(img.width >> mip, 1u), std::max(img.height >> mip, 1u)}, view.range.layerCount == VK_REMAINING_ARRAY_LAYERS ? img.layers : view.range.layerCount,
           img.is_depth && HasStencil(img.format), img.image};
    return true;
}

bool ClearColor(u64 view_id, const float* color) {
    if (!On() || !device_ready) {
        return false;
    }
    std::unique_lock lock{mutex};
    const auto v = views.find(view_id);
    const auto i = v != views.end() ? images.find(v->second.texture_id) : images.end();
    if (i == images.end() || i->second.is_depth) {
        c_clear_no_view.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    vk::ClearColorValue value{};
    std::memcpy(value.float32.data(), color, 4 * sizeof(float));
    vk::ImageSubresourceRange range = v->second.range;
    range.aspectMask = vk::ImageAspectFlagBits::eColor;
    const vk::Image clear_image = i->second.image;
    if (ContractCensusOn()) {
        ContractNote(fmt::format("W1 clear view {} image {} colour {:.3g} {:.3g} {:.3g} {:.3g}", vk::to_string(v->second.format), vk::to_string(i->second.format), color[0], color[1],
                                 color[2], color[3]));
    }
    lock.unlock();
    Stream::Defer(Stream::Work::Clear, [image = clear_image, value, range](vk::CommandBuffer cmd) {
        Stream::Hazard(cmd, {Stream::ImageUse(image, true)});
        cmd.clearColorImage(image, vk::ImageLayout::eGeneral, value, range);
    });
    c_clears_color.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool ClearDepthStencil(u64 view_id, u32 flags, float depth, u32 stencil) {
    if (!On() || !device_ready || (flags & 3) == 0) {
        return false;
    }
    // api-contract.md W12: the engine clamps the clear depth to [0, 1] (below 0 -> 0, above 1 -> 1; 024860c0) and writes the stencil value with a full mask
    if (const float clamped = depth < 0.0f ? 0.0f : std::min(depth, 1.0f); clamped != depth || depth != depth) {
        if (ContractCensusOn()) {
            ContractNote("W12 clear depth outside [0, 1] (clamped)");
        }
        depth = depth != depth ? 0.0f : clamped;
    }
    stencil &= 0xffu;
    std::unique_lock lock{mutex};
    const auto v = views.find(view_id);
    const auto i = v != views.end() ? images.find(v->second.texture_id) : images.end();
    if (i == images.end() || !i->second.is_depth) {
        c_clear_no_view.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    vk::ImageSubresourceRange range = v->second.range;
    range.aspectMask = {};
    if ((flags & 1) != 0) {
        range.aspectMask |= vk::ImageAspectFlagBits::eDepth;
    }
    if ((flags & 2) != 0 && HasStencil(i->second.format)) {
        range.aspectMask |= vk::ImageAspectFlagBits::eStencil;
    }
    if (!range.aspectMask) {
        return false;
    }
    const vk::Image clear_image = i->second.image;
    lock.unlock();
    Stream::Defer(Stream::Work::Clear, [image = clear_image, depth, stencil, range](vk::CommandBuffer cmd) {
        Stream::Hazard(cmd, {Stream::ImageUse(image, true)});
        cmd.clearDepthStencilImage(image, vk::ImageLayout::eGeneral, vk::ClearDepthStencilValue{depth, stencil}, range);
    });
    c_clears_depth.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void TextureCpuWritten(u64 texture_id, u32 subresource) {
    if (!On() || !device_ready) {
        return;
    }
    vk::Image image{};
    vk::Format format{};
    u32 mips = 0, layers = 0;
    vk::ImageLayout layout{};
    {
        std::scoped_lock lock{mutex};
        const auto it = images.find(texture_id);
        if (it == images.end() || it->second.is_depth) {
            c_reupload_no_image.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Image& img = it->second;
        if (img.layout == vk::ImageLayout::eUndefined) {
            img.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
            BumpViewEpoch(); // note 46: A1 memo answers may change
        }
        image = img.image, format = img.format, mips = img.mips, layers = img.layers, layout = img.layout;
    }
    c_reupload_wanted.fetch_add(1, std::memory_order_relaxed);
    if (ContractCensusOn()) { // api-contract.md R9 (debug): our image's size against the engine's live T# (a moved texture's T# is rewritten to its logical size)
        u32 w = 0, h = 0, m = 0, iw = 0, ih = 0;
        {
            std::scoped_lock lock{mutex};
            if (const auto it = images.find(texture_id); it != images.end()) {
                iw = it->second.width, ih = it->second.height;
            }
        }
        if (Upload::LiveExtent(texture_id, w, h, m) && (w != iw || h != ih)) {
            ContractNote(fmt::format("R9 re-upload: image {}x{}, engine T# {}x{} ({} mips)", iw, ih, w, h, m));
        }
    }
    if (ContractCensusOn()) { // L5 / L7 (debug), before the stream / batch split (api-contract review A2)
        std::scoped_lock census_lock{contract_mutex};
        ++contract_counts[contract_copy_destinations.contains(texture_id) ? "L7 cpu rewrite of a texture a GPU copy wrote" : "L5 cpu rewrite of a texture"];
    }
    // api-contract.md L5 / L6 / L8 / W5: the new contents are a job of the frame stream, made here, where the engine changed the memory (its hazard barriers order the
    // copy after the earlier reads of the image and before the later ones). LEGACY(upload_batch): TLG_LEGACY_UPLOAD_BATCH=1 uploads in the batch that runs before the frame
    static const bool upload_batch = [] { const char* e = std::getenv("TLG_LEGACY_UPLOAD_BATCH"); return e != nullptr && e[0] == '1'; }();
    if (!upload_batch && layout == vk::ImageLayout::eGeneral && Stream::Ready()) {
        if (ContractCensusOn()) {
            ContractNote("L5 cpu rewrite of a texture, in the stream");
        }
        if (Upload::TextureContentsInStream(texture_id, image, format, mips, layers, subresource)) {
            c_reupload_done.fetch_add(1, std::memory_order_relaxed);
            c_reupload_in_stream.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    if (Upload::TextureContents(texture_id, image, format, mips, layers, layout, subresource)) {
        c_reupload_done.fetch_add(1, std::memory_order_relaxed);
    }
}

std::atomic<u64> c_loaded_reads{0}, c_loaded_textures{0}, c_loaded_buffers{0};
u32 MemoryLoaded(u64 address, u64 bytes) {
    // the engine read a file straight into this memory (Fios, on its I/O thread): a texture there was copied before its data arrived (the probe volumes of the
    // lighting pass 0x775d0654 are streamed in after they are first bound: magenta / green / red ambient light, note 81); its image gets the contents again
    if (!On() || !device_ready || bytes == 0) {
        return 0;
    }
    c_loaded_reads.fetch_add(1, std::memory_order_relaxed);
    // note 84: a texture whose engine object gives no memory size (the probe volumes: +0x38 is 0) gets it from its description, once (outside the lock)
    std::vector<u64> unsized;
    {
        std::scoped_lock lock{mutex};
        for (const auto& [id, img] : images) {
            if (img.guest_bytes == 0 && !img.gpu_written && !img.is_depth && img.address != 0 && !img.size_tried) {
                unsized.push_back(id);
            }
        }
    }
    for (const u64 id : unsized) {
        const u64 size = Upload::GuestSize(id);
        std::scoped_lock lock{mutex};
        if (const auto it = images.find(id); it != images.end()) {
            it->second.guest_bytes = size;
            it->second.size_tried = true;
        }
    }
    boost::container::small_vector<u64, 4> ids;
    {
        std::scoped_lock lock{mutex};
        for (const auto& [id, img] : images) {
            // note 84: only textures the engine fills (a render target's or UAV's contents are ours alone: the stale memory under it overwrote them, the false
            // colour of 2026-11-16), and only over the engine's own memory size (our image can be larger)
            if (img.address != 0 && !img.is_depth && !img.gpu_written && img.guest_bytes != 0 && img.address < address + bytes && address < img.address + img.guest_bytes) {
                ids.push_back(id);
            }
        }
    }
    for (const u64 id : ids) {
        TextureCpuWritten(id);
    }
    c_loaded_textures.fetch_add(ids.size(), std::memory_order_relaxed);
    // buffers (plan D4): a static buffer's device copy made at its first bind (R3d) over the range gets the read's bytes,
    // copied in the stream at this point (the read is complete: its completion called us)
    struct Copy {
        u64 guest, bytes;
        VkBuffer device;
        u64 offset;
    };
    std::vector<Copy> copies;
    {
        std::scoped_lock lock{mutex};
        const auto add = [&](u64 start, u64 end, VkBuffer device) {
            const u64 lo = std::max(address, start), hi = std::min(address + bytes, end);
            if (device != VK_NULL_HANDLE && lo < hi) {
                copies.push_back({lo, hi - lo, device, lo - start});
            }
        };
        // every tracked buffer the read overlaps: the one holding its start (if any), then all that begin inside it (api-contract review B1: a read that begins
        // before the first tracked buffer used to skip them all)
        auto it = watched.upper_bound(address);
        if (it != watched.begin() && std::prev(it)->second.end > address) {
            --it;
        }
        for (; it != watched.end() && it->first < address + bytes; ++it) {
            if (it->second.state == Watched::State::Device) {
                add(it->first, it->second.end, it->second.device);
            }
        }
        // (not the GPU-written device buffers of V2f: their contents are ours alone, note 84)
        if (ContractCensusOn()) { // api-contract.md Y7 (debug): a file read into memory under a GPU-written device buffer
            auto d = device_buffers.upper_bound(address);
            if (d != device_buffers.begin() && std::prev(d)->second.end > address) {
                --d;
            }
            for (; d != device_buffers.end() && d->first < address + bytes; ++d) {
                ContractNote(fmt::format("Y7 file read into a GPU-written device buffer ({} B)", d->second.end - d->first));
            }
        }
    }
    for (const Copy& c : copies) {
        const BufferWindow src = SourceWindow(c.guest, c.bytes);
        if (!src.buffer) {
            c_copy_unresolved.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        Stream::NoteBytes(Stream::Work::UploadCpuWritten, c.bytes);
        Stream::Defer(Stream::Work::UploadCpuWritten, [src, c](vk::CommandBuffer cmd) {
            if (Stream::HoistCopy(src.buffer, src.offset, vk::Buffer{c.device}, c.offset, c.bytes)) {
                return;
            }
            Stream::Hazard(cmd, {Stream::BufferUse(src.buffer, src.offset, c.bytes, false), Stream::BufferUse(vk::Buffer{c.device}, c.offset, c.bytes, true)});
            cmd.copyBuffer(src.buffer, vk::Buffer{c.device}, vk::BufferCopy{src.offset, c.offset, c.bytes});
        });
    }
    c_loaded_buffers.fetch_add(copies.size(), std::memory_order_relaxed);
    return static_cast<u32>(ids.size() + copies.size());
}

u32 ReuploadVolumes() {
    // debug (tint hunt, 2026-11-16): every 3D texture gets its contents again from the engine's memory (the probe volumes are filled by a CPU path no hook sees)
    std::vector<u64> ids;
    {
        std::scoped_lock lock{mutex};
        for (const auto& [id, img] : images) {
            if (img.type == vk::ImageType::e3D && !img.is_depth) {
                ids.push_back(id);
            }
        }
    }
    for (const u64 id : ids) {
        TextureCpuWritten(id);
    }
    return static_cast<u32>(ids.size());
}

/// api-contract.md Y6 (debug census): where the bytes of a range the CPU reads in place (AsyncMap) live on our side: "device" = our separate device copy (the guest
/// memory the CPU reads is stale), "in place" = imported guest memory or the frame ring
std::string ContractWhereLives(u64 address, u64 size) {
    std::scoped_lock lock{mutex};
    u64 start = 0;
    bool straddle = false;
    if (DeviceBufferOf(address, size, start, straddle) != nullptr) {
        return "a GPU-written device buffer";
    }
    if (straddle) {
        return "straddles a device buffer";
    }
    if (auto it = watched.upper_bound(address); it != watched.begin()) {
        --it;
        if (address >= it->first && address < it->second.end && it->second.state == Watched::State::Device) {
            return "a static device copy";
        }
    }
    return "in place";
}

/// api-contract.md Y6: AsyncMap hands the CPU a plain pointer into the buffer's own memory (no Map, no Unmap, no copy): the engine's contract is that the buffer's bytes
/// are where that pointer writes. A device copy of ours (a dynamic buffer's, refreshed only at Unmap) breaks it, so it is retired here and the buffer is read in place
/// from now on (FindLocked falls back to the imported memory). A buffer the GPU writes keeps its device copy (its bytes are ours alone): counted, open to test.
/// LEGACY(asyncmap_device): TLG_LEGACY_ASYNCMAP_DEVICE=1 keeps every device copy.
std::atomic<u64> c_placed_by_asyncmap{0}, c_asyncmap_gpu_written{0};
void BufferCpuAddressable(u64 buffer_id) {
    static const bool legacy = [] { const char* e = std::getenv("TLG_LEGACY_ASYNCMAP_DEVICE"); return e != nullptr && e[0] == '1'; }();
    if (!On() || !device_ready || legacy || buffer_id == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    const auto by_id = device_buffer_start.find(buffer_id); // by the engine object (no address lookup)
    if (by_id == device_buffer_start.end()) {
        return;
    }
    const auto it = device_buffers.find(by_id->second);
    if (it == device_buffers.end() || it->second.id != buffer_id) {
        return;
    }
    if (it->second.gpu_writes) {
        c_asyncmap_gpu_written.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    doomed.push_back({static_cast<u32>(GraphicsSubmits()) + DestroyDelay(), {}, {}, {}, it->second.buffer, it->second.allocation});
    c_dev_bytes.fetch_sub(it->second.end - it->first, std::memory_order_relaxed);
    device_buffer_start.erase(it->second.id);
    devbuf_hash.erase(it->first);
    device_buffers.erase(it);
    BumpBufferEpoch(); // note 46: A1 memo answers may change
    c_placed_by_asyncmap.fetch_add(1, std::memory_order_relaxed);
}

/// api-contract.md V4 / review A6: an engine buffer view's bytes inside its buffer, from the view's own V# (the engine keeps it on the buffer's current memory at every
/// rename): offset = view base - buffer base (the buffer's V# at + 0x60), size = records x stride (bytes if no stride), clamped to the buffer. False: the whole buffer.
std::atomic<u64> c_view_whole{0}, c_view_part{0}, c_view_outside{0};
bool ViewRange(u64 view, bool srv, u64 buffer, u64 bytes, u64& offset, u64& size) {
    offset = 0, size = bytes;
    if (view == 0 || buffer == 0 || bytes == 0) {
        return false;
    }
    const u32* v = reinterpret_cast<const u32*>(view + (srv ? 0x78 : 0x74));
    const u64 base = static_cast<u64>(v[0]) | static_cast<u64>(v[1] & 0xfffu) << 32; // base: 44 bits
    const u32 stride = (v[1] >> 16) & 0x3fffu;
    const u64 records = stride != 0 ? u64{v[2]} * stride : u64{v[2]};
    const u32* b = reinterpret_cast<const u32*>(buffer + 0x60);
    const u64 buffer_base = static_cast<u64>(b[0]) | static_cast<u64>(b[1] & 0xfffu) << 32;
    if (base < buffer_base || base >= buffer_base + bytes || records == 0) {
        c_view_outside.fetch_add(1, std::memory_order_relaxed);
        if (ContractCensusOn()) {
            ContractNote("A6 buffer view outside its buffer (whole buffer bound)");
        }
        return false;
    }
    offset = base - buffer_base;
    size = std::min(records, bytes - offset);
    if (offset == 0 && size == bytes) {
        c_view_whole.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    c_view_part.fetch_add(1, std::memory_order_relaxed);
    if (ContractCensusOn()) {
        ContractNote(offset != 0 ? "A6 buffer view starts inside its buffer (range bound)" : "A6 buffer view shorter than its buffer (range bound)");
    }
    return true;
}

/// api-contract.md W5: the engine moved a texture to new memory (RenderTexture_CopyToNewMemory, 01c97a00: a raw copy of the old texture's memory into a new texture's,
/// from mip `first_mip` on; first_mip > 0 drops the top mips into a half-size texture). Our new image gets the old image's contents by an image copy in the frame
/// stream, by the two engine objects (no address lookup, no upload from guest memory). LEGACY(upload_batch): TLG_LEGACY_UPLOAD_BATCH=1 does not follow moves.
void TextureMoved(u64 new_id, u64 old_id, u32 first_mip) {
    static const bool upload_batch = [] { const char* e = std::getenv("TLG_LEGACY_UPLOAD_BATCH"); return e != nullptr && e[0] == '1'; }();
    if (!On() || !device_ready || upload_batch || new_id == 0 || old_id == 0 || new_id == old_id) {
        return;
    }
    std::unique_lock lock{mutex};
    const auto d = images.find(new_id);
    const auto s = images.find(old_id);
    if (d == images.end() || s == images.end()) {
        c_moved_no_image.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const Image& to = d->second;
    const Image& from = s->second;
    if (to.is_depth != from.is_depth || to.type != from.type || TexelBytes(to.format) != TexelBytes(from.format) || to.image == from.image) {
        c_moved_mismatch.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const vk::ImageAspectFlags aspect = !to.is_depth                                         ? vk::ImageAspectFlags{vk::ImageAspectFlagBits::eColor}
                                        : HasStencil(to.format) && HasStencil(from.format) ? (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)
                                                                                           : vk::ImageAspectFlags{vk::ImageAspectFlagBits::eDepth};
    const bool volume = to.type == vk::ImageType::e3D;
    const u32 layers = volume ? 1u : std::min(std::max(to.layers, 1u), std::max(from.layers, 1u));
    const auto extent = [](const Image& img, u32 mip) {
        return vk::Extent3D{std::max(img.width >> mip, 1u), std::max(img.height >> mip, 1u), img.type == vk::ImageType::e3D ? std::max(img.depth >> mip, 1u) : 1u};
    };
    std::vector<vk::ImageCopy> regions;
    u64 bytes = 0;
    for (u32 m = 0; m < to.mips && m + first_mip < from.mips; ++m) {
        const vk::Extent3D a = extent(to, m), b = extent(from, m + first_mip);
        // the engine copies raw memory: a padded new texture (R9) holds the old one's texels at the same place, so the common part is copied
        vk::Extent3D e{std::min(a.width, b.width), std::min(a.height, b.height), std::min(a.depth, b.depth)};
        if (a != b) {
            c_moved_mismatch.fetch_add(1, std::memory_order_relaxed);
            // D4-16: for a block-compressed format a copy side must be whole blocks unless it reaches that image's mip edge (VUID-vkCmdCopyImage-srcImage-01728..);
            // a side that ends inside a block of the other image is cut to whole blocks
            const auto block = vk::blockExtent(to.format);
            const auto fit = [](u32 n, u32 dst_edge, u32 src_edge, u32 bw) { return bw <= 1 || n % bw == 0 || (n == dst_edge && n == src_edge) ? n : n / bw * bw; };
            const vk::Extent3D cut{fit(e.width, a.width, b.width, block[0]), fit(e.height, a.height, b.height, block[1]), fit(e.depth, a.depth, b.depth, block[2])};
            if (cut != e) {
                c_moved_cut_to_blocks.fetch_add(1, std::memory_order_relaxed);
                e = cut;
            }
            if (e.width == 0 || e.height == 0 || e.depth == 0) {
                continue; // a mip smaller than one block of the other image: nothing whole to copy
            }
        }
        regions.push_back({.srcSubresource = {aspect, m + first_mip, 0, layers}, .dstSubresource = {aspect, m, 0, layers}, .extent = e});
        bytes += static_cast<u64>(e.width) * e.height * e.depth * layers * TexelBytes(to.format);
    }
    const vk::Image src = from.image, dst = to.image;
    lock.unlock(); // (no job is made under this module's lock: the recording thread's tick takes it)
    if (regions.empty()) {
        c_moved_mismatch.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Stream::NoteBytes(Stream::Work::Copy, bytes);
    Stream::Defer(Stream::Work::Copy, [src, dst, regions = std::move(regions)](vk::CommandBuffer cmd) {
        Stream::Hazard(cmd, {Stream::ImageUse(src, false), Stream::ImageUse(dst, true)});
        cmd.copyImage(src, vk::ImageLayout::eGeneral, dst, vk::ImageLayout::eGeneral, regions);
    });
    c_moved_textures.fetch_add(1, std::memory_order_relaxed);
    if (ContractCensusOn()) {
        std::scoped_lock census_lock{contract_mutex};
        contract_copy_destinations.insert(new_id);
        ++contract_counts[first_mip == 0 ? "W5 texture moved (image copy)" : "W5 texture top mips dropped (image copy)"];
    }
}

/// api-contract.md W5b: the engine's buffer move (GpuHeap_MoveOneResource 01f7e850, its buffer branch: a new buffer, the async copy old -> new at 01f7eec3, then the
/// holder swapped to the new buffer at once, so CPU readers (e.g. FUN_01d7a890, the effects' vertex clusters, which turn floats read from it into array indexes)
/// may read the new memory right away). On PS4 the async queue runs that copy within moments; our frame stream would run it a frame or more later. So the copy is
/// made now, on the CPU, when the old memory holds the contents (nothing of ours writes it on the GPU: no GPU-written device copy, no pending in-place GPU write,
/// not marked GPU-written); else it stays our copy in the frame stream (false). LEGACY(move_in_stream): TLG_LEGACY_MOVE_IN_STREAM=1 always in the stream.
std::atomic<u64> c_moves_now{0}, c_moves_in_stream{0};
bool MoveBufferNow(u64 destination, u64 source, u64 bytes) {
    static const bool legacy = [] { const char* e = std::getenv("TLG_LEGACY_MOVE_IN_STREAM"); return e != nullptr && e[0] == '1'; }();
    if (!On() || !device_ready || legacy || bytes == 0 || destination == source || (destination < source + bytes && source < destination + bytes)) {
        return false;
    }
    bool gpu_side = false;
    {
        std::scoped_lock lock{mutex};
        for (auto d = device_buffers.upper_bound(source); !gpu_side && d != device_buffers.begin();) { // the device copy holding the start, and those inside
            --d;
            if (d->second.end > source) {
                gpu_side = d->second.gpu_writes;
            }
            break;
        }
        for (auto d = device_buffers.lower_bound(source); !gpu_side && d != device_buffers.end() && d->first < source + bytes; ++d) {
            gpu_side = d->second.gpu_writes;
        }
        if (!gpu_side) {
            if (const Watched* w = WatchedAt(source, bytes)) {
                gpu_side = w->gpu_written;
            }
        }
    }
    if (!gpu_side) {
        std::scoped_lock marks_lock{gpu_write_mutex};
        for (const auto& [a, n] : gpu_write_pending) {
            if (a < source + bytes && source < a + n) {
                gpu_side = true;
                break;
            }
        }
    }
    if (R3dTraceBytes() != 0 && bytes == R3dTraceBytes()) {
        LOG_WARNING(Render_Vulkan, "vkb R3d trace: engine buffer move {:#x} -> {:#x} ({} bytes), {} submit {}", source, destination, bytes, gpu_side ? "in the stream" : "now", GraphicsSubmits());
    }
    if (gpu_side) {
        c_moves_in_stream.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::memcpy(reinterpret_cast<void*>(destination), reinterpret_cast<const void*>(source), bytes);
    BufferCpuWritten(destination, bytes); // a copy of ours of the new memory (if any) takes the bytes
    c_moves_now.fetch_add(1, std::memory_order_relaxed);
    return true;
}

/// api-contract.md W16: the probe cubemap copy (01c804a0): the engine copies each mip / face of a probe texture as raw memory into its slot of a cube-array texture
/// (layer = slot x 6 + face, CsCopyBufferDwords at 01c806a0). Our images get the same copy, image to image by the two engine objects, in the frame stream.
std::atomic<u64> c_layer_copies{0}, c_layer_copy_refused{0};
void TextureLayerCopied(u64 dst_id, u32 dst_layer, u64 src_id, u32 src_layer, u32 mip) {
    if (!On() || !device_ready || dst_id == 0 || src_id == 0 || dst_id == src_id) {
        return;
    }
    std::unique_lock lock{mutex};
    const auto d = images.find(dst_id);
    const auto s = images.find(src_id);
    if (d == images.end() || s == images.end() || d->second.is_depth || s->second.is_depth || d->second.type == vk::ImageType::e3D ||
        s->second.type == vk::ImageType::e3D || mip >= d->second.mips || mip >= s->second.mips || dst_layer >= d->second.layers || src_layer >= s->second.layers ||
        d->second.format != s->second.format) {
        c_layer_copy_refused.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const auto extent = [mip](const Image& img) { return vk::Extent3D{std::max(img.width >> mip, 1u), std::max(img.height >> mip, 1u), 1u}; };
    const vk::Extent3D a = extent(d->second), b = extent(s->second);
    const vk::ImageCopy region{.srcSubresource = {vk::ImageAspectFlagBits::eColor, mip, src_layer, 1},
                               .dstSubresource = {vk::ImageAspectFlagBits::eColor, mip, dst_layer, 1},
                               .extent = {std::min(a.width, b.width), std::min(a.height, b.height), 1}};
    const vk::Image src = s->second.image, dst = d->second.image;
    lock.unlock(); // (no job is made under this module's lock: the recording thread's tick takes it)
    Stream::NoteBytes(Stream::Work::Copy, u64{region.extent.width} * region.extent.height * TexelBytes(d->second.format));
    Stream::Defer(Stream::Work::Copy, [src, dst, region](vk::CommandBuffer cmd) {
        Stream::Hazard(cmd, {Stream::ImageUse(src, false), Stream::ImageUse(dst, true)});
        cmd.copyImage(src, vk::ImageLayout::eGeneral, dst, vk::ImageLayout::eGeneral, region);
    });
    c_layer_copies.fetch_add(1, std::memory_order_relaxed);
}

namespace {
/// debug R3d trace: GPU copies out of a device buffer of the traced size (the fur results into a staging buffer): destination -> (submit at the copy, stream value it was recorded in)
std::mutex fur_copy_mutex;
std::map<u64, std::pair<u64, u64>> fur_copy;
} // namespace
std::map<u64, u64> fur_staging; ///< memory -> bytes of the buffers the fur readback maps (debug R3d trace)
void R3dTraceMapRead(u64 memory, u64 bytes, u64 caller) {
    if (R3dTraceBytes() == 0) {
        return;
    }
    std::scoped_lock lock{fur_copy_mutex};
    if (caller == 0x1e64d2b) {
        fur_staging[memory] = bytes;
        u64 sum = 0;
        for (u64 i = 0; i + 8 <= std::min<u64>(bytes, 4096); i += 8) {
            sum = sum * 31 + *reinterpret_cast<const u64*>(memory + i);
        }
        LOG_WARNING(Render_Vulkan, "vkb R3d trace: fur readback Map READ of {:#x} ({} bytes) at submit {}, first 4 KiB sum {:#x}, GPU done up to {}", memory, bytes, GraphicsSubmits(), sum,
                    Stream::CompletedValue());
    }
    for (auto it = fur_copy.lower_bound(memory); it != fur_copy.end() && it->first < memory + std::max<u64>(bytes, 1); ++it) {
        LOG_WARNING(Render_Vulkan, "vkb R3d trace: Map READ of {:#x} ({} bytes, caller {:#x}) at submit {}: last fur copy into it at submit {}, recorded in stream value {}, GPU done up to {} -> {}", memory, bytes,
                    caller, GraphicsSubmits(), it->second.first, it->second.second, Stream::CompletedValue(),
                    it->second.second == 0 ? "NOT RECORDED YET" : Stream::CompletedValue() >= it->second.second ? "done" : "NOT DONE");
    }
}
bool CopyBuffer(u64 destination, u64 source, u64 bytes) {
    if (!device_ready || bytes == 0) {
        return false;
    }
    bool fur_traced = false;
    if (R3dTraceBytes() != 0) {
        std::scoped_lock lock{mutex};
        u64 st = 0;
        bool sd = false;
        if (const DeviceBuffer* d = DeviceBufferOf(source, bytes, st, sd); d != nullptr && d->end - st == R3dTraceBytes()) {
            fur_traced = true;
        }
    }
    if (R3dTraceBytes() != 0 && !fur_traced) {
        std::scoped_lock lock{fur_copy_mutex};
        if (auto it = fur_staging.upper_bound(destination); it != fur_staging.begin() && destination < std::prev(it)->first + std::prev(it)->second) {
            fur_traced = true;
        }
    }
    if (fur_traced) {
        LOG_WARNING(Render_Vulkan, "vkb R3d trace: GPU copy out of a fur output buffer {:#x} -> {:#x} ({} bytes) at submit {}", source, destination, bytes, GraphicsSubmits());
        std::scoped_lock lock{fur_copy_mutex};
        fur_copy[destination] = {GraphicsSubmits(), 0};
    }
    R3dTrace("GPU copy", destination, bytes);
    if (ContractCensusOn()) { // api-contract.md W5 (debug): a buffer copy whose destination is a texture's memory (the engine's memory move, FUN_01c97a00)
        std::string key = "W5 buffer copy into no texture";
        {
            std::scoped_lock lock{mutex};
            for (const auto& [id, img] : images) {
                if (img.address != 0 && img.address < destination + bytes && destination < img.address + std::max<u64>(img.guest_bytes, 1)) {
                    key = fmt::format("W5 buffer copy into a texture ({}, {} written by the GPU, source {} texture)", vk::to_string(img.format), img.gpu_written ? "" : "not",
                                      [&] {
                                          for (const auto& [sid, simg] : images) {
                                              if (simg.address != 0 && simg.address < source + bytes && source < simg.address + std::max<u64>(simg.guest_bytes, 1)) {
                                                  return "a";
                                              }
                                          }
                                          return "no";
                                      }());
                    break;
                }
            }
        }
        ContractNote(key);
    }
    BufferWindow to, from;
    if (!FindLocked(destination, bytes, to) || !FindLocked(source, bytes, from)) {
        c_copy_unresolved.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (to.buffer == from.buffer && to.offset < from.offset + bytes && from.offset < to.offset + bytes) {
        c_copy_overlap.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    Stream::NoteBytes(Stream::Work::CopyBuffer, bytes);
    Stream::Defer(Stream::Work::CopyBuffer, [from, to, bytes, fur_traced, destination](vk::CommandBuffer cmd) {
        Stream::Hazard(cmd, {Stream::BufferUse(from.buffer, from.offset, bytes, false), Stream::BufferUse(to.buffer, to.offset, bytes, true)});
        cmd.copyBuffer(from.buffer, to.buffer, vk::BufferCopy{from.offset, to.offset, bytes});
        if (fur_traced) {
            std::scoped_lock lock{fur_copy_mutex};
            fur_copy[destination].second = Stream::RecordingValue();
        }
        if (!to.device) {
            Stream::NoteInPlaceWrite(to.buffer, to.offset, bytes);
        }
    });
    { // a GPU write, in place or into our device copy: marked GPU-written either way (a later engine move copies it on the GPU, not from stale engine memory)
        std::scoped_lock marks_lock{gpu_write_mutex};
        gpu_write_pending.push_back({destination, bytes});
    }
    c_copies.fetch_add(1, std::memory_order_relaxed);
    c_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (to.device || from.device) {
        c_copies_device.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

bool CopyFromGds(u64 gds_offset, u64 destination, u64 bytes) {
    if (!device_ready || bytes == 0 || gds_offset + bytes > (64u << 10)) {
        return false;
    }
    BufferWindow to;
    if (!FindLocked(destination, bytes, to)) {
        c_copy_unresolved.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    Stream::NoteBytes(Stream::Work::CopyGds, bytes);
    Stream::Defer(Stream::Work::CopyGds, [gds_offset, to, bytes](vk::CommandBuffer cmd) {
        Stream::Hazard(cmd, {Stream::BufferUse(Stream::GdsBuffer(), gds_offset, bytes, false), Stream::BufferUse(to.buffer, to.offset, bytes, true)});
        cmd.copyBuffer(Stream::GdsBuffer(), to.buffer, vk::BufferCopy{gds_offset, to.offset, bytes});
        if (!to.device) {
            Stream::NoteInPlaceWrite(to.buffer, to.offset, bytes);
        }
    });
    { // a GPU write, in place or into our device copy: marked GPU-written either way (a later engine move copies it on the GPU, not from stale engine memory)
        std::scoped_lock marks_lock{gpu_write_mutex};
        gpu_write_pending.push_back({destination, bytes});
    }
    c_copies.fetch_add(1, std::memory_order_relaxed);
    c_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

bool FillBuffer(u64 destination, u64 bytes, u32 value) {
    if (!device_ready || bytes == 0 || (destination & 3) != 0 || (bytes & 3) != 0) {
        return false;
    }
    R3dTrace("GPU fill", destination, bytes);
    BufferWindow to;
    if (!FindLocked(destination, bytes, to) || (to.offset & 3) != 0) {
        c_fill_unresolved.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    Stream::Defer(Stream::Work::Clear, [to, bytes, value](vk::CommandBuffer cmd) {
        Stream::Hazard(cmd, {Stream::BufferUse(to.buffer, to.offset, bytes, true)});
        cmd.fillBuffer(to.buffer, to.offset, bytes, value);
        if (!to.device) {
            Stream::NoteInPlaceWrite(to.buffer, to.offset, bytes);
        }
    });
    { // a GPU write, in place or into our device copy: marked GPU-written either way (a later engine move copies it on the GPU, not from stale engine memory)
        std::scoped_lock marks_lock{gpu_write_mutex};
        gpu_write_pending.push_back({destination, bytes});
    }
    c_fills.fetch_add(1, std::memory_order_relaxed);
    return true;
}

namespace {
/// R5: the pipeline of the pattern fill (host shader fill_pattern.comp: destination[i] = source[i mod pattern], as the engine's compute copy with wrap)
struct FillPipeline {
    vk::DescriptorSetLayout set_layout{};
    vk::PipelineLayout layout{};
    vk::Pipeline pipeline{};
};
const FillPipeline& FillPatternPipeline() {
    static FillPipeline p;
    static std::once_flag once;
    std::call_once(once, [] {
        const vk::Device device = instance->GetDevice();
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {.binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        p.set_layout = device.createDescriptorSetLayout({.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR, .bindingCount = 2, .pBindings = bindings.data()}).value;
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = 5 * sizeof(u32)};
        p.layout = device.createPipelineLayout({.setLayoutCount = 1, .pSetLayouts = &p.set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &range}).value;
        const vk::ShaderModule module = VkbVulkan::CompileSPV(std::span<const u32>{FILL_PATTERN_COMP}, device);
        const auto [result, pipeline] = device.createComputePipeline(
            VK_NULL_HANDLE, {.stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"}, .layout = p.layout});
        device.destroyShaderModule(module);
        if (result == vk::Result::eSuccess) {
            p.pipeline = pipeline;
        }
    });
    return p;
}
/// LEGACY(fill_doubling): TLG_LEGACY_FILL_DOUBLING=1 keeps the V5c fill (the pattern copied once, then doubled with about 20 barriers: 2 ms a frame, note 30)
bool FillDoubling() {
    static const bool on = [] { const char* e = std::getenv("TLG_LEGACY_FILL_DOUBLING"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::atomic<u64> c_pattern_fills_shader{0}, c_pattern_fills_in_place{0}, c_pattern_fill_mb_in_place{0}, c_pattern_fill_mb_device{0}, c_fill_skipped_test{0};
} // namespace

bool FillPattern(u64 destination, u64 bytes, u64 source, u32 dwords) {
    if (!device_ready || bytes == 0 || dwords == 0 || dwords > 64 || (destination & 3) != 0 || (bytes & 3) != 0) {
        return false;
    }
    const u64 first = std::min<u64>(dwords * 4ull, bytes);
    BufferWindow to, from;
    if (!FindLocked(destination, bytes, to) || !FindLocked(source, first, from)) {
        c_fill_unresolved.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (to.buffer == from.buffer && to.offset < from.offset + first && from.offset < to.offset + bytes) {
        c_copy_overlap.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // note 49, kill experiment only: TLG_VKB_TEST_SKIP_HOST_FILLS=1 drops the pattern fills into host memory (the picture may be wrong); bounds what they cost
    static const bool skip_host_fills = [] { const char* e = std::getenv("TLG_VKB_TEST_SKIP_HOST_FILLS"); return e != nullptr && e[0] == '1'; }();
    if (skip_host_fills && !to.device) {
        c_fill_skipped_test.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    Stream::NoteBytes(Stream::Work::Fill, bytes);
    if (!to.device) {
        FillReadersTarget(destination, bytes, 0, "pattern fill");
    }
    if (!to.device) {
        // R5: what the in-place fill targets are (the first 8 distinct ones), log/vkb_fill_targets.txt
        static std::mutex m;
        static std::vector<u64> seen;
        std::scoped_lock lock{m};
        if (seen.size() < 8 && std::find(seen.begin(), seen.end(), destination) == seen.end()) {
            seen.push_back(destination);
            std::string text = fmt::format("fill {:#x} {} bytes, pattern {} dwords, source {:#x}: ", destination, bytes, dwords, source);
            {
                std::scoped_lock objects_lock{mutex};
                auto it = watched.upper_bound(destination);
                if (it != watched.begin() && destination >= std::prev(it)->first && destination < std::prev(it)->second.end) {
                    const Watched& wb = std::prev(it)->second;
                    text += fmt::format("engine buffer {:#x}..{:#x} id {} class {} usage {} bind {:#x} cpu_access {:#x} gpu_written {} state {} first_bind_submit {}", std::prev(it)->first,
                                        wb.end, wb.id, wb.cls, wb.usage, wb.bind, wb.cpu_access, wb.gpu_written, static_cast<u32>(wb.state), wb.first_bind_submit);
                } else {
                    text += "no engine buffer known at this address";
                }
            }
            if (FILE* f = std::fopen((Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_fill_targets.txt").string().c_str(), "a")) {
                std::fputs(text.c_str(), f);
                std::fputc('\n', f);
                std::fclose(f);
            }
        }
    }
    const u64 align = std::max<u64>(instance->StorageMinAlignment(), 4);
    const u64 source_base = from.offset & ~(align - 1), destination_base = to.offset & ~(align - 1);
    if (!FillDoubling() && FillPatternPipeline().pipeline && ((from.offset - source_base) & 3) == 0 && ((to.offset - destination_base) & 3) == 0 && bytes / 4 <= 0xffffffffull) {
        // one dispatch, as the engine's own shader: destination[i] = source[i mod pattern]
        Stream::Defer(Stream::Work::Fill, [to, from, bytes, first, source_base, destination_base](vk::CommandBuffer cmd) {
            Stream::Hazard(cmd, {Stream::BufferUse(from.buffer, from.offset, first, false), Stream::BufferUse(to.buffer, to.offset, bytes, true)}, Stream::Agent::Compute);
            const FillPipeline& p = FillPatternPipeline();
            const vk::DescriptorBufferInfo source_info{.buffer = from.buffer, .offset = source_base, .range = from.offset - source_base + first};
            const vk::DescriptorBufferInfo destination_info{.buffer = to.buffer, .offset = destination_base, .range = to.offset - destination_base + bytes};
            const std::array<vk::WriteDescriptorSet, 2> writes{{
                {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &source_info},
                {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &destination_info},
            }};
            const u32 count = static_cast<u32>(bytes / 4);
            const u32 groups = (count + 255) / 256;
            const u32 groups_x = std::min<u32>(groups, 65535), groups_y = (groups + groups_x - 1) / groups_x;
            const std::array<u32, 5> push{static_cast<u32>((from.offset - source_base) / 4), static_cast<u32>((to.offset - destination_base) / 4),
                                          static_cast<u32>(first / 4), count, groups_x};
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, p.pipeline);
            cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, p.layout, 0, writes);
            cmd.pushConstants(p.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push.data());
            cmd.dispatch(groups_x, groups_y, 1);
            if (!to.device) {
                Stream::NoteInPlaceWrite(to.buffer, to.offset, bytes);
            }
        });
        { // a GPU write, in place or into our device copy: marked GPU-written either way (a later engine move copies it on the GPU, not from stale engine memory)
            std::scoped_lock marks_lock{gpu_write_mutex};
            gpu_write_pending.push_back({destination, bytes});
        }
        c_pattern_fills.fetch_add(1, std::memory_order_relaxed);
        c_pattern_fills_shader.fetch_add(1, std::memory_order_relaxed);
        if (!to.device) {
            c_pattern_fills_in_place.fetch_add(1, std::memory_order_relaxed);
            c_pattern_fill_mb_in_place.fetch_add(bytes >> 20, std::memory_order_relaxed);
        } else {
            c_pattern_fill_mb_device.fetch_add(bytes >> 20, std::memory_order_relaxed);
        }
        return true;
    }
    Stream::Defer(Stream::Work::Fill, [to, from, bytes, first](vk::CommandBuffer cmd) {
        Stream::Hazard(cmd, {Stream::BufferUse(from.buffer, from.offset, first, false), Stream::BufferUse(to.buffer, to.offset, bytes, true)}, Stream::Agent::Transfer);
        cmd.copyBuffer(from.buffer, to.buffer, vk::BufferCopy{from.offset, to.offset, first});
        const vk::MemoryBarrier2 step{.srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer, .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                      .dstStageMask = vk::PipelineStageFlagBits2::eAllTransfer, .dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
        // the pattern doubles: [0, done) is copied after itself (the size stays a multiple of the pattern's)
        for (u64 done = first; done < bytes;) {
            const u64 n = std::min(done, bytes - done);
            cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &step});
            cmd.copyBuffer(to.buffer, to.buffer, vk::BufferCopy{to.offset, to.offset + done, n});
            done += n;
        }
        if (!to.device) {
            Stream::NoteInPlaceWrite(to.buffer, to.offset, bytes);
        }
    });
    { // a GPU write, in place or into our device copy: marked GPU-written either way (a later engine move copies it on the GPU, not from stale engine memory)
        std::scoped_lock marks_lock{gpu_write_mutex};
        gpu_write_pending.push_back({destination, bytes});
    }
    c_pattern_fills.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void PublishObjects() {
    static std::atomic<u64> calls{0};
    if ((calls.fetch_add(1, std::memory_order_relaxed) & 63) != 0) {
        return;
    }
    DebugServer::Publish("vkb_objects", ObjectsReport());
}

nlohmann::json ObjectsReport() {
    u64 live_images, live_views, pending;
    nlohmann::json refused = nlohmann::json::object();
    nlohmann::json channel_census = nlohmann::json::object();
    {
        std::scoped_lock lock{mutex};
        live_images = images.size(), live_views = views.size(), pending = doomed.size();
        if (FillReadersOn()) { // note 49 census: textures whose memory overlaps a fill target
            std::scoped_lock fr_lock{fr_mutex};
            for (const auto& [id, im] : images) {
                for (const auto& r : fr_ranges) {
                    if (im.address != 0 && im.address < r.first + r.second && r.first < im.address + std::max<u64>(im.bytes, 1) && fr_images_seen.insert(id).second) {
                        FillReadersLine(fmt::format("texture {} at {:#x}+{} {}x{}x{} {} mips {} layers {} (fill {:#x})", id, im.address, im.bytes, im.width, im.height, im.depth,
                                                    vk::to_string(im.format), im.mips, im.layers, r.first));
                    }
                }
            }
        }
        for (const auto& [pair, n] : refused_pairs) {
            refused[pair] = n;
        }
        for (const auto& [key, g] : channel_groups) {
            channel_census[key] = fmt::format("{} views, e.g. view {} of texture {} ({}x{})", g.views, g.example_view, g.example_texture, g.width, g.height);
        }
    }
    const u64 made = c_images_created.load();
    return nlohmann::json(
                         {{"view_channel_census", channel_census},
                          {"upload_census", UploadCensusOn() ? UploadCensusReport() : nlohmann::json{}},
#ifdef _WIN32
                          {"dirty_pages", Dirty::Report()},
#endif
                          {"changed_pages", Changed::Report()},
                          {"contract_census", ContractReport()},
                          {"gpu_read_trap", GpuReadTrap::Report()},
                          {"display_texture", g_display_texture.load(std::memory_order_relaxed)},
                          {"images_live", live_images},
                          {"images_created", made},
                          {"images_failed", c_images_failed.load()},
                          {"images_with_no_vulkan_format", c_images_unmapped_format.load()},
                          {"images_destroyed", c_images_destroyed.load()},
                          {"image_mb_live", c_image_bytes.load() >> 20},
                          {"image_create_us_mean", made ? static_cast<double>(c_image_ns.load()) / 1e3 / static_cast<double>(made) : 0.0},
                          {"views_live", live_views},
                          {"views_created", c_views_created.load()},
                          {"views_failed", c_views_failed.load()},
                          {"views_swizzled_by_tsharp", c_views_swizzled.load()},
                          {"views_range_clamped", c_views_clamped.load()},
                          {"views_of_a_texture_with_no_image", c_views_no_image.load()},
                          {"views_of_buffers_no_object", c_views_buffer.load()},
                          {"views_destroyed", c_views_destroyed.load()},
                          {"destructions_pending", pending},
                          {"device_buffer_check",
                           {{"mode", DeviceBuffersMode()},
                            {"classes", "0 not CPU-writable by description, 1 dynamic or CPU-writable, 2 staging, 3 no engine buffer at the address"},
                            {"buffers", {c_watch_buffers[0].load(), c_watch_buffers[1].load(), c_watch_buffers[2].load()}},
                            {"buffer_mb", {c_watch_bytes[0].load() >> 20, c_watch_bytes[1].load() >> 20, c_watch_bytes[2].load() >> 20}},
                            {"in_place_binds", {c_watch_binds[0].load(), c_watch_binds[1].load(), c_watch_binds[2].load(), c_watch_binds[3].load()}},
                            {"in_place_bind_mb", {c_watch_bind_bytes[0].load() >> 20, c_watch_bind_bytes[1].load() >> 20, c_watch_bind_bytes[2].load() >> 20,
                                                  c_watch_bind_bytes[3].load() >> 20}},
                            {"class0_changed_without_a_hooked_write", c_watch_mismatch.load()},
                            {"class0_changed_after_our_gpu_write", c_watch_mismatch_gpu.load()},
                            {"class0_changes_before_first_bind", c_watch_changed_before_bind.load()},
                            {"class0_changes_after_first_bind", c_watch_changed_after_bind.load()},
                            {"class0_changes_within_3_submits_of_first_bind", c_watch_changed_within_3.load()},
                            {"class0_buffers_changed_after_bind", c_watch_buffers_changed_after_bind.load()},
                            {"class0_buffers_changed_3_times_after_bind", c_watch_buffers_changed_often.load()},
                            {"class0_updates_by_update_subresource", c_watch_cpu_updates.load()},
                            {"hashed_mb", c_watch_hashed_mb.load() >> 20},
                            {"watch_rounds", c_watch_rounds.load()},
                            {"mode2", {{"device_copies_made", c_dev2_made.load()}, {"device_copy_mb_live", c_dev2_mb.load() >> 20}, {"binds_to_a_device_copy", c_dev2_binds.load()},
                                       {"bound_mb_from_device_copies", c_dev2_bind_mb.load() >> 20}, {"changed_after_the_copy_back_in_place", c_dev2_unstable.load()},
                                       {"changes_while_settling", c_dev2_settle_changes.load()}, {"gave_up_still_changing", c_dev2_gave_up.load()}, {"failed", c_dev2_failed.load()}}}}},
                          {"pattern_fills", c_pattern_fills.load()}, {"pattern_fills_by_shader", c_pattern_fills_shader.load()}, {"pattern_fills_into_host_memory", c_pattern_fills_in_place.load()}, {"pattern_fill_mb_host_memory", c_pattern_fill_mb_in_place.load()}, {"pattern_fill_mb_device", c_pattern_fill_mb_device.load()}, {"test_host_fills_skipped", c_fill_skipped_test.load()},
                          {"bind_census",
                           {{"kinds", "compute descriptor, graphics descriptor, vertex buffer, index buffer"},
                            {"binds_in_place", {c_census_binds[0][0].load(), c_census_binds[1][0].load(), c_census_binds[2][0].load(), c_census_binds[3][0].load()}},
                            {"binds_device", {c_census_binds[0][1].load(), c_census_binds[1][1].load(), c_census_binds[2][1].load(), c_census_binds[3][1].load()}},
                            {"in_place_bound_mb", {c_census_bytes[0].load() >> 20, c_census_bytes[1].load() >> 20, c_census_bytes[2].load() >> 20, c_census_bytes[3].load() >> 20}},
                            {"in_place_binds_written", {c_census_written[0].load(), c_census_written[1].load(), c_census_written[2].load(), c_census_written[3].load()}},
                            {"frames", c_census_frames.load()},
                            {"unique_in_place_mb_per_frame", c_census_frames.load() ? static_cast<double>(c_census_unique_bytes.load()) / 1048576.0 / static_cast<double>(c_census_frames.load()) : 0.0},
                            {"unique_in_place_ranges_per_frame", c_census_frames.load() ? static_cast<double>(c_census_unique_ranges.load()) / static_cast<double>(c_census_frames.load()) : 0.0},
                            {"unique_mb_by_size_lt4k_lt64k_lt1m_ge1m", {c_census_bucket[0].load() >> 20, c_census_bucket[1].load() >> 20, c_census_bucket[2].load() >> 20,
                                                                        c_census_bucket[3].load() >> 20}}}},
                          {"host_cache_experiment", {{"hits", c_cache_hits.load()}, {"copies", c_cache_copies.load()}, {"arena_full", c_cache_full.load()},
                                                     {"mb_copied", c_cache_bytes.load() >> 20}}},
                          {"buffers_seen", c_buffers.load()},
                          {"buffers_in_place", c_buffers_in_place.load()},
                          {"buffers_not_in_place", c_buffers_not_in_place.load()},
                          {"buffer_mb_in_place_seen", c_buffer_bytes.load() >> 20},
                          {"in_place_imported_mb", host_import ? host_import->ImportedBytes() >> 20 : 0},
                          {"in_place_failed_chunks", host_import ? host_import->FailedChunks() : 0},
                          {"gpu_pool", {{"on", host_import ? host_import->OnGpuPool() : false}, {"mb", VkbVulkan::GpuPool::Bytes() >> 20}, {"reason", VkbVulkan::GpuPool::Reason()},
                                        {"exec_dropped", VkbVulkan::GpuPool::ExecDropped()}}},
                          {"objects_before_the_device", c_too_early.load()},
                          {"deletes_of_objects_with_no_vulkan_object", c_deleted_unknown.load()},
                          {"contents_wanted", c_upload_wanted.load()},
                          {"contents_uploaded", c_upload_done.load()},
                          {"upload", Upload::Report()},
                          {"device_buffers", {{"made", c_dev_buffers.load()}, {"failed", c_dev_failed.load()}, {"mb_live", c_dev_bytes.load() >> 20},
                                              {"destroyed", c_dev_destroyed.load()}, {"replaced_at_same_memory", c_dev_replaced.load()},
                                              {"no_initial_data", c_dev_no_initial.load()}, {"cpu_writes_copied", c_cpu_writes.load()},
                                              {"cpu_write_kb", c_cpu_write_bytes.load() >> 10}}},
                          {"other_view_types", {{"made", c_other_views.load()}, {"refused", c_other_view_refused.load()}, {"refused_pairs", refused}}},
                          {"cube_bindings_as_array", c_cube_as_array.load()},
                          {"display", {{"presented_from_our_image", c_display_found.load()}, {"no_image_at_the_display_buffer", c_display_missed.load()},
                                       {"never_drawn_into", c_display_not_general.load()}}},
                          {"attachments", {{"format_views_made", c_attach_views.load()}, {"refused", c_attach_refused.load()}}},
                          {"clears", {{"color", c_clears_color.load()}, {"depth_stencil", c_clears_depth.load()}, {"no_view", c_clear_no_view.load()}}},
                          {"our_samplers", {{"made", c_samplers_made.load()}, {"invalid_description", c_samplers_invalid.load()}, {"min_max_reduction", c_sampler_minmax.load()},
                                            {"border_rounded", c_sampler_border_rounded.load()}, {"mirror_once", c_sampler_mirror_once.load()}}},
                          {"texture_updates", {{"wanted", c_reupload_wanted.load()}, {"uploaded", c_reupload_done.load()}, {"no_image", c_reupload_no_image.load()},
                                               {"in_stream", c_reupload_in_stream.load()}, {"moved_textures", c_moved_textures.load()}, {"moved_no_image", c_moved_no_image.load()},
                                               {"moved_mismatch", c_moved_mismatch.load()}, {"moved_cut_to_blocks", c_moved_cut_to_blocks.load()}}},
                          {"buffer_moves", {{"copied_at_the_move", c_moves_now.load()}, {"in_the_stream", c_moves_in_stream.load()}}},
                          {"probe_layer_copies", {{"copied", c_layer_copies.load()}, {"refused", c_layer_copy_refused.load()}}},
                          {"buffer_views", {{"whole", c_view_whole.load()}, {"range", c_view_part.load()}, {"outside", c_view_outside.load()}}},
                          {"asyncmap_buffers", {{"placed_in_place", c_placed_by_asyncmap.load()}, {"gpu_written_kept_device_copy", c_asyncmap_gpu_written.load()}}},
                          {"file_reads_into_textures", {{"reads", c_loaded_reads.load()}, {"textures", c_loaded_textures.load()}, {"buffers", c_loaded_buffers.load()}}},
                          {"texture_copies", {{"copies", c_tex_copies.load()}, {"no_image", c_tex_copy_no_image.load()}, {"mismatch", c_tex_copy_mismatch.load()}}},
                          {"buffer_lookups", {{"range_leaves_a_device_buffer", c_find_straddle.load()}, {"not_in_place", c_find_none.load()}}},
                          {"r7_census", R7CensusReport()},
                          {"bind_by_object", BindByObjectReport()},
                          {"cb_in_place", PoolReport()}, // note 98
                          {"targets_fixed_test", {{"on", TargetsFixed()}, {"targets", c_fixed_targets.load()}, {"views_forced_to_the_image_format", c_fixed_view_format_forced.load()},
                                                  {"attachments_refused", c_fixed_attachment_refused.load()}}},
                          {"staging_textures", {{"note", "note 42: copies into staging textures written back for the CPU's Map(READ)"}, {"copies", c_staging_copies.load()},
                                                {"reads", c_staging_reads.load()}, {"not_ready", c_staging_unready.load()}, {"older", c_staging_older.load()}, {"unhandled", c_staging_unhandled.load()}, {"retired_with_texture", c_staging_retired.load()}}},
                          {"static_at_first_bind", {{"on", StaticAtBindOn()}, {"made", c_static_at_bind.load()}, {"zero_filled_no_source", c_static_at_bind_zero.load()}, {"mb", c_static_at_bind_mb.load() >> 20},
                                                    {"failed", c_static_at_bind_failed.load()},
                                                    {"found_on_recording_side", {{"on", FirstBindOnRecorderOn()}, {"made", c_fbr_copies.load()}, {"mb", c_fbr_mb.load() >> 20},
                                                                                 {"read_in_place_before_copy_job", c_fbr_pending_in_place.load()}}}}},
                          {"dynamic_device_buffers", {{"on", DynamicDeviceOn()}, {"made", c_dyn_device.load()}, {"unmap_uploads", c_dyn_unmaps.load()},
                                                      {"unmap_upload_mb", c_dyn_unmap_mb.load() >> 20}, {"checks", c_dyn_checks.load()},
                                                      {"changed_between_unmap_and_next_map", c_dyn_changed_outside.load()}}},
                          {"buffer_maps", [] {
                               nlohmann::json out = nlohmann::json::array();
                               for (u32 u = 0; u < 4; ++u) {
                                   for (u32 t = 0; t < 8; ++t) {
                                       for (u32 p = 0; p < 2; ++p) {
                                           if (const u64 n = c_map_count[u][t][p].load(); n != 0) {
                                               out.push_back({{"usage", u}, {"type", t}, {"in_place", p == 1}, {"maps", n}, {"mb", c_map_bytes[u][t][p].load() >> 20}});
                                           }
                                       }
                                   }
                               }
                               return out;
                           }()},
                          {"layout_at_creation_jobs_images_by_upload_in_batch", {true, c_layout_jobs.load(), c_layout_images.load(), c_layout_by_upload.load(), c_layout_in_batch.load()}},
                          {"frame_ring_in_device_memory", {{"on", vram_ring_base.load() != 0}, {"route", ring_decided.load(std::memory_order_acquire) ? ring_route : RingVramOn() ? "not placed yet" : "off (TLG_VKB_RING_VRAM=0)"},
                                                           {"placed_failure", ring_decided.load(std::memory_order_acquire) ? ring_placed_failure : std::string{}}, {"base", vram_ring_base.load()}, {"binds", c_ring_vram_binds.load()},
                                                           {"ranges_leaving_it", c_ring_vram_straddle.load()}}},
                          {"in_place_sources", {{"note", "binds read in place (TLG_VKB_SOURCE_CENSUS=1): engine buffer static / dynamic / staging, frame ring, other"},
                                                {"binds", {c_src_binds[0].load(), c_src_binds[1].load(), c_src_binds[2].load(), c_src_binds[3].load(), c_src_binds[4].load()}},
                                                {"mb", {c_src_bytes[0].load() >> 20, c_src_bytes[1].load() >> 20, c_src_bytes[2].load() >> 20, c_src_bytes[3].load() >> 20,
                                                        c_src_bytes[4].load() >> 20}},
                                                {"ring_binds_by_size_1k_16k_256k_more", {c_src_ring_sizes[0].load(), c_src_ring_sizes[1].load(), c_src_ring_sizes[2].load(),
                                                                                         c_src_ring_sizes[3].load()}},
                                                {"ring", {{"base", frame_ring_base.load()}, {"mb", frame_ring_size.load() >> 20}}},
                                                {"top_buffers", [] {
                                                     std::vector<std::pair<u64, SourceStat>> v;
                                                     {
                                                         std::scoped_lock lock{mutex};
                                                         v.assign(source_stats.begin(), source_stats.end());
                                                     }
                                                     std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
                                                     nlohmann::json out = nlohmann::json::array();
                                                     for (size_t i = 0; i < v.size() && i < 16; ++i) {
                                                         const SourceStat& s = v[i].second;
                                                         out.push_back({{"address", v[i].first}, {"size", s.size}, {"binds", s.binds}, {"bound_mb", s.bytes >> 20}, {"class", s.cls},
                                                                        {"usage", s.usage}, {"bind_flags", s.bind}, {"cpu_access", s.cpu}});
                                                     }
                                                     return out;
                                                 }()}}},
                          {"buffer_copies", {{"copies", c_copies.load()}, {"of_device_buffers", c_copies_device.load()}, {"kb", c_copy_bytes.load() >> 10},
                                             {"unresolved", c_copy_unresolved.load()}, {"overlapping", c_copy_overlap.load()}, {"fills", c_fills.load()},
                                             {"fills_unresolved", c_fill_unresolved.load()}}}});
}

} // namespace Core::Vkb
