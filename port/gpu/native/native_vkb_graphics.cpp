// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_vkb_aa.h"
#include "native/native_vkb_shader_dump.h"
#include "native/native_framewalk.h"
#include "native/native_debug_server.h"
#include "native/native_marker.h"
#include "native/native_vkb_graphics.h"
#include "native/native_vkb_indexed.h"
#include "native/native_vkb_compute.h"
#include "native/native_zpass_gen.h"
#include "native/native_vkb_shader_recipe.h"
#include "native/native_vkb_shader_store.h"

#include "native/shadps4/video_core/renderer_vulkan/vk_common.h"

#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <array>
#include <atomic>
#include <bit>
#include <type_traits>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "tlg_settings.h"
#include <deque>
#include <filesystem>
#include "common/path_util.h"
#include "native/native_vkb.h"
#include "native/shadps4/core/guest_heap.h"
#include "native/native_desc.h"
#include "native/native_device.h"
#include "native/native_draw.h"
#include "native/native_shader_env.h"
#include "native/native_target_inputs.h"
#include "native/native_vkb_objects.h"
#include "native/native_vkb_stream.h"
#include "native/native_vkb_util.h"
#include "native/shadps4/shader_recompiler/ir/passes/ir_passes.h"
#include "native/shadps4/shader_recompiler/backend/bindings.h"
#include "native/shadps4/shader_recompiler/backend/spirv/emit_spirv.h"
#include "native/shadps4/shader_recompiler/frontend/fetch_shader.h"
#include "native/shadps4/shader_recompiler/info.h"
#include "native/shadps4/shader_recompiler/profile.h"
#include "native/shadps4/shader_recompiler/recompiler.h"
#include "native/shadps4/shader_recompiler/runtime_info.h"
#include "native/shadps4/shader_recompiler/specialization.h"
#include "native/shadps4/video_core/amdgpu/regs_shader.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_fixed_state.h"
#include "native/native_vkb_pipeline.h"
#include "native/native_vkb_pipeline_warm.h"
#include "native/vkb_core/vkb_device.h"
#include "native/vkb_core/liverpool_to_vk.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_instance.h"
#include "native/native_vkb_side_census.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_shader_boundary.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_shader_util.h"
#include "native/vkb_core/sampler.h"


extern "C" int runtime_memory_read_backing(uintptr_t address, void* data, uint64_t size);

namespace Core::Vkb::Graphics {

namespace {

using Util::Mix;
using Util::ObjectClass;
using Util::Rd;
using Util::Source;
using Util::SourceOf;
using VkbShader::HwStage;
using VkbShader::SwStage;

// the device context (Ctx in native_draw.cpp)
constexpr u64 CtxInputLayout = 0x68, CtxRenderTargetCount = 0x4484, CtxRenderTargets = 0x4488, CtxDepthStencilView = 0x44d0, CtxPixelShader = 0x4690,
              CtxVertexShader = 0x4698, CtxTarget7 = 0x130e0;
constexpr u64 SrvTable(u32 stage) {
    return 0x580 + stage * 0x400;
}

/// One stage of a variant: its program, the user data it was specialized with (kept: the info's spans point into it), its module and its slot map
struct Stage {
    u32 index{};                          ///< 0 vertex, 1 pixel (the engine's stage numbers)
    u64 shader{};                         ///< the shader object it was compiled from
    Core::NativeDraw::UserLayout layout;
    Core::NativeDraw::UserLayout live;    ///< the user data of the current draw (the flat user data buffer only)
    std::vector<vk::ImageViewType> image_types;
    std::vector<std::unique_ptr<VideoCore::Vkb::Sampler>> constant_samplers; ///< per sampler: ours from the S# the shader holds as a constant (null: from a slot)
    std::vector<Core::NativeDesc::SamplerDesc> constant_descs; ///< issue 42: per sampler, the constant S#'s description (for a draw's bias / anisotropy)
    std::vector<u32> constant_ids;                             ///< issue 42: per sampler, a session-unique number of the constant S# (0: from a slot), for the per-thread memo
    bool has_flat{};
    std::unique_ptr<VkbShader::Info> info;
    VkbShader::RuntimeInfo runtime{};
    VkbShader::RuntimeInfo runtime_in{};   ///< the runtime info and bindings as handed to the translator (the store's inputs: the pipeline
    VkbShader::Backend::Bindings start_in{}; ///< warm-up names the list entry of this stage by them)
    vk::ShaderModule module{};
    std::vector<Source> buffers, images, samplers;
    u32 row_start{}; ///< R2 (record mode): the stage's first entry in the draw's row
};

struct Variant {
    Stage vs, ps;
    bool has_ps{};
    VkbShader::Gcn::FetchShaderData fetch{};
    bool has_fetch{};
    std::vector<Source> attributes; ///< per fetch attribute: the vertex buffer table element it reads (tag kind 5)
    VkbVulkan::Vkb::GraphicsPipelineKey key{};
    std::array<vk::Format, 8> color_formats{}; ///< the pipeline's attachment formats (our views are made in them)
    vk::Format depth_format{};                 ///< our depth image's format (the pipeline's depth format)
    std::unique_ptr<VkbVulkan::Vkb::GraphicsPipeline> pipeline;
    bool ok{};
    bool record{};     ///< R2: compiled in record mode (resources through the global set and the draw's row)
    u32 row_dwords{};  ///< R2: the entries of a draw's row
};

std::mutex mutex;
std::unordered_map<u64, std::unique_ptr<Variant>> variants;
std::unordered_map<u64, std::unique_ptr<Variant>> record_variants; ///< R2: the record-mode twin of a variant (same key)
std::unique_ptr<VkbShader::Pools> pools;
VkbShader::Profile profile{};
VkbShader::Profile record_profile{}; ///< R2: the profile with the constant offset layout (a read-only guest buffer's offset and size in the row)
std::atomic<u64> c_rec_compiled{0}, c_rec_refused{0}, c_rec_draws{0}, c_rec_fallback{0}, c_rec_row_fail{0};
std::array<std::atomic<u64>, 8> c_rec_refused_by{};
constexpr std::array<const char*, 8> RecRefusedNames{"flat_user_data", "written_buffer", "not_indexed_buffer", "bda_or_fault", "image_without_key", "compile", "pipeline", "other"};
/// R2 / R3 switch: TLG_VKB_RECORD=1 (the global set is made) and the scope; this step: the shadow cascades (Z pass views 1, 2)
thread_local bool t_row_failed = false;
thread_local std::array<u64, 24> t_pipe_words16{}; ///< R7 step D: the key of a cache miss, stored when the slow path found the pipeline
thread_local u64 t_pipe_hash = 0;
thread_local bool t_pipe_pending = false; ///< R3: the record-mode draw could not build its row (it is drawn the plain way)
bool RecordOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_RECORD"); return e == nullptr || e[0] != '0'; }();
    return on;
}
const VkbVulkan::Instance* instance = nullptr;

enum Fail : u32 { FailLayout, FailNoCode, FailTranslation, FailFetch, FailNoDynamicVertexInput, FailPipeline, FailCount };
constexpr std::array<const char*, FailCount> FailNames{"user_data_layout", "no_program", "translation", "fetch_shader", "no_dynamic_vertex_input", "pipeline"};
std::array<std::atomic<u64>, FailCount> c_fail{};
std::atomic<u64> c_draws{0}, c_by_kind[3]{}, c_no_vs{0}, c_special{0}, c_no_fixed{0}, c_found{0}, c_compiled{0}, c_failed{0}, c_compile_ns{0}, c_key_ns{0};
std::atomic<u64> c_depth_only{0}, c_targets_seen{0}, c_unsupported_layout_bits{0};
/// D3 census (note 77, TLG_VKB_TARGET_CENSUS=1, debug only): at each variant compile, every bound target's engine description (the view's format and dimension at
/// + 0x40 / + 0x44, its texture's sample count at + 0x194) against what the PS4 register block in the view gives (format, number type, swap, blend bypass, samples)
bool TargetCensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_TARGET_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::mutex target_census_mutex;
std::map<std::string, u64> target_census;
void TargetCensusNote(std::string line) {
    std::scoped_lock lock{target_census_mutex};
    ++target_census[std::move(line)];
}
std::atomic<u64> c_src[2][3][2]{}; ///< [stage vs/ps][buffers/images/samplers][mapped/unmapped]
std::atomic<u64> c_attr_mapped{0}, c_attr_unmapped{0};
std::atomic<u64> c_recorded{0}, c_record_ns{0}, c_skip_not_compiled{0}, c_skip_buffer{0}, c_skip_image{0}, c_skip_sampler{0}, c_skip_uav{0}, c_skip_special{0}, c_skip_target{0},
    c_skip_depth{0}, c_late_views{0}, c_skip_vertex{0}, c_skip_index{0}, c_skip_indirect{0}, c_skip_no_attachment{0}, c_indirect_recorded{0}, c_new_passes{0}, c_flat_rebuilt{0};
std::atomic<u64> c_base_vertex_vulkan{0}, c_base_vertex_shader{0}, c_base_vertex_other_register{0}; ///< issue 36: direct draws with a first vertex: added by Vulkan /
                                                                                                    ///< left to the shader's own code / fetch shader reads another register (Vulkan, as before)

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool Ready() {
    if (instance != nullptr) {
        return true;
    }
    if (!Core::Vkb::Device::Ready()) { // merge note 5: the backend's own device, made by its presenter
        return false;
    }
    instance = &Core::Vkb::Device::Instance();
    profile = Core::Vkb::Device::Profile();
    record_profile = profile;
    record_profile.constant_offsets = true;
    Core::Vkb::ShaderStore::Open(profile); // TLG_VKB_SHADER_PREBUILD (no-op when off)
    pools = std::make_unique<VkbShader::Pools>();
    return true;
}

/// The colour target words of a bound render target view (the 14 words of its block at + 0x54; word 0 zero: not a target)
bool ColorWords(u64 view, std::array<u32, Core::NativeTarget::RawTargetWords::ColorWords>& out) {
    out = {};
    if (view == 0 || Rd<u32>(view + 0x54, 0) == 0) {
        return false;
    }
    for (u32 k = 0; k < 14; ++k) {
        out[k] = Rd<u32>(view + 0x54, k * 4);
    }
    return true;
}

/// The render target views of the draw, slots 0 to 7 (slot 7 is the object at ctx + 0x130e0)
std::array<u64, 8> TargetViews(u64 ctx) {
    std::array<u64, 8> views{};
    const u32 count = Rd<u32>(ctx, CtxRenderTargetCount);
    for (u32 slot = 0; slot < 7; ++slot) {
        views[slot] = slot < count ? Rd<u64>(ctx, CtxRenderTargets + slot * 8) : 0;
    }
    views[7] = Rd<u64>(ctx, CtxTarget7);
    return views;
}

Core::NativeTarget::DepthTarget DepthOf(u64 ctx) {
    std::array<u32, Core::NativeTarget::RawTargetWords::DepthWords> words{};
    if (const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView); dsv != 0) {
        for (u32 k = 0; k < words.size(); ++k) {
            words[k] = Rd<u32>(dsv + 0x58, k * 4);
        }
    }
    return Core::NativeTarget::DepthFromWords(words.data());
}

/// D3 (note 77): the targets described from the engine's own objects (the view's DXGI format at + 0x40, its texture's sample count at + 0x194) instead of the PS4
/// register block the engine keeps in each view (+ 0x54 colour, + 0x58 depth). The values stay in the recompiler's terms (its export conversion needs them).
/// A format missing from the table below falls back to the words (counted).
/// TLG_VKB_TARGET_CHECK=1 (debug): both descriptions at every variant compile, differences counted and the first ones listed.
bool TargetCheckOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_TARGET_CHECK"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::atomic<u64> c_targets_engine{0}, c_targets_unmapped{0}, c_target_checks{0}, c_target_check_differences{0};
std::mutex target_check_mutex;
std::vector<std::string> target_check_lines;
void TargetCheckNote(std::string line) {
    c_target_check_differences.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{target_check_mutex};
    if (target_check_lines.size() < 16) {
        target_check_lines.push_back(std::move(line));
    }
}

/// A DXGI render target format in colour buffer terms: format, number type, component swap (note 77 census, heaviest save; the other rows follow the same rules)
struct CbFormat {
    u32 format{}, number_type{};
    Core::NativeTarget::ColorTarget::SwapMode swap{};
};
std::optional<CbFormat> CbFormatOf(u32 dxgi) {
    using Swap = Core::NativeTarget::ColorTarget::SwapMode;
    constexpr u32 Unorm = 0, Snorm = 1, Uint = 4, Sint = 5, Srgb = 6, Float = 7;
    switch (dxgi) {
    case 2: return CbFormat{14, Float};                 // R32G32B32A32_FLOAT
    case 1: case 3: return CbFormat{14, Uint};          // R32G32B32A32_TYPELESS / _UINT
    case 4: return CbFormat{14, Sint};
    case 9: case 10: return CbFormat{12, Float};        // R16G16B16A16_TYPELESS / _FLOAT (seen)
    case 11: return CbFormat{12, Unorm};
    case 12: return CbFormat{12, Uint};
    case 13: return CbFormat{12, Snorm};
    case 14: return CbFormat{12, Sint};
    case 16: return CbFormat{11, Float};                // R32G32_FLOAT
    case 15: case 17: return CbFormat{11, Uint};
    case 18: return CbFormat{11, Sint};
    case 23: case 24: return CbFormat{9, Unorm};        // R10G10B10A2_TYPELESS / _UNORM (seen)
    case 25: return CbFormat{9, Uint};
    case 26: return CbFormat{6, Float};                 // R11G11B10_FLOAT (seen)
    case 27: case 28: return CbFormat{10, Unorm};       // R8G8B8A8_TYPELESS / _UNORM (seen)
    case 29: return CbFormat{10, Srgb};                 // R8G8B8A8_UNORM_SRGB (seen)
    case 30: return CbFormat{10, Uint};
    case 31: return CbFormat{10, Snorm};
    case 32: return CbFormat{10, Sint};
    case 33: case 34: return CbFormat{5, Float};        // R16G16_TYPELESS / _FLOAT (seen)
    case 35: return CbFormat{5, Unorm};
    case 36: return CbFormat{5, Uint};
    case 37: return CbFormat{5, Snorm};
    case 38: return CbFormat{5, Sint};
    case 41: return CbFormat{4, Float};                 // R32_FLOAT
    case 39: case 42: return CbFormat{4, Uint};
    case 43: return CbFormat{4, Sint};
    case 48: case 49: return CbFormat{3, Unorm};        // R8G8_TYPELESS / _UNORM
    case 50: return CbFormat{3, Uint};
    case 51: return CbFormat{3, Snorm};
    case 52: return CbFormat{3, Sint};
    case 53: case 54: return CbFormat{2, Float};        // R16_TYPELESS / _FLOAT
    case 56: return CbFormat{2, Unorm};
    case 57: return CbFormat{2, Uint};
    case 58: return CbFormat{2, Snorm};
    case 59: return CbFormat{2, Sint};
    case 60: case 61: return CbFormat{1, Unorm};        // R8_TYPELESS / _UNORM (seen)
    case 62: return CbFormat{1, Uint};
    case 63: return CbFormat{1, Snorm};
    case 64: return CbFormat{1, Sint};
    case 87: case 88: case 90: return CbFormat{10, Unorm, Swap::Alternate}; // B8G8R8A8 / B8G8R8X8 / _TYPELESS (seen)
    case 91: return CbFormat{10, Srgb, Swap::Alternate};
    default: return std::nullopt;
    }
}
u32 TextureSamples(u64 view) {
    const u64 resource = Rd<u64>(view, 0x30);
    return std::max(resource != 0 && IsHeapPointer(resource) ? Rd<u32>(resource, 0x194) : 1u, 1u);
}

/// A bound colour target in the recompiler's terms (false: none bound)
bool ColorOf(u64 view, Core::NativeTarget::ColorTarget& out) {
    out = {};
    if (view == 0) {
        return false;
    }
    if (const auto f = CbFormatOf(Rd<u32>(view, 0x40))) {
        out.base_address = 1; // ColorTarget's "bound" test is base_address && format; the backend never uses the address
        out.info.format = f->format;
        out.info.number_type = f->number_type;
        out.info.comp_swap = f->swap;
        out.attrib.num_samples_log2 = out.attrib.num_fragments_log2 = static_cast<u32>(std::bit_width(TextureSamples(view)) - 1);
        c_targets_engine.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    c_targets_unmapped.fetch_add(1, std::memory_order_relaxed);
    std::array<u32, Core::NativeTarget::RawTargetWords::ColorWords> words;
    if (!ColorWords(view, words)) {
        return false;
    }
    out = Core::NativeTarget::ColorFromWords(words.data());
    return true;
}

/// The depth target in the recompiler's terms
struct DepthDesc {
    Core::NativeTarget::DepthTarget::ZFormat z{Core::NativeTarget::DepthTarget::ZFormat::Invalid};
    Core::NativeTarget::DepthTarget::StencilFormat stencil{Core::NativeTarget::DepthTarget::StencilFormat::Invalid};
    u32 samples{1};
};
DepthDesc DepthWordsDesc(u64 ctx) {
    const Core::NativeTarget::DepthTarget depth = DepthOf(ctx);
    DepthDesc d;
    d.z = depth.DepthValid() ? depth.z_info.format : Core::NativeTarget::DepthTarget::ZFormat::Invalid;
    d.stencil = depth.StencilValid() ? depth.stencil_info.format : Core::NativeTarget::DepthTarget::StencilFormat::Invalid;
    d.samples = depth.NumSamples();
    return d;
}
DepthDesc DepthDescOf(u64 ctx) {
    using Z = Core::NativeTarget::DepthTarget::ZFormat;
    using S = Core::NativeTarget::DepthTarget::StencilFormat;
    const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView);
    if (dsv == 0) {
        return {};
    }
    DepthDesc d;
    switch (Rd<u32>(dsv, 0x40)) {
    case 39: case 40: case 41: d.z = Z::Z32Float; break;                                                          // D32_FLOAT (seen) and its typeless forms
    case 19: case 20: case 21: case 22: case 44: case 45: case 46: case 47: d.z = Z::Z32Float, d.stencil = S::Stencil8; break; // D32S8 / D24S8 (seen; kept as 32 bit float)
    case 53: case 55: case 56: d.z = Z::Z16; break;                                                               // D16_UNORM
    default: c_targets_unmapped.fetch_add(1, std::memory_order_relaxed); return DepthWordsDesc(ctx);
    }
    d.samples = TextureSamples(dsv);
    c_targets_engine.fetch_add(1, std::memory_order_relaxed);
    return d;
}

/// What the variant key takes from a bound target (false: none): the engine's format and sample count, remembered per thread by the view and its unique id (+ 0x10,
/// never reused); with the legacy words, CB_COLOR_INFO and CB_COLOR_ATTRIB (or DB_Z_INFO and DB_STENCIL_INFO for the depth view)
bool TargetKey(u64 view, bool depth, u64& a, u64& b) {
    if (view == 0) {
        return false;
    }
    const u64 block = view + (depth ? 0x58 : 0x54);
    const auto words = [&] {
        if (!depth && Rd<u32>(block, 0) == 0) {
            return false;
        }
        a = Rd<u32>(block, depth ? 0 : 16), b = Rd<u32>(block, depth ? 4 : 20);
        return true;
    };
    struct Entry {
        u64 view{}, id{~0ull}, a{}, b{};
        bool mapped{};
    };
    thread_local std::array<Entry, 256> memo{};
    const u64 id = Rd<u64>(view, 0x10);
    Entry& e = memo[((view >> 4) ^ (view >> 12) ^ id) & 255];
    if (e.view != view || e.id != id) {
        const u32 dxgi = Rd<u32>(view, 0x40);
        e = {view, id, 0x100000000ull | dxgi, TextureSamples(view), depth || CbFormatOf(dxgi).has_value()};
    }
    if (!e.mapped) {
        return words();
    }
    a = e.a, b = e.b;
    return true;
}

/// The variant of the draw: the shader ids, the fetch object, the fixed state (cached per state, so its address names it), the target words that describe formats and samples,
/// and the class of every object in the shader resource slots the two shaders use
/// The user data offsets (SH register offsets, 0: none) where the console's indirect draw puts the base vertex and the base instance for this vertex shader
u32 IndirectSgprs(u64 vs) {
    const u8 locations = Rd<u8>(Rd<u64>(vs, 0x120), 0x27);
    constexpr u32 VertexStageSgprBase = 0x4c;
    const u32 vertex = (locations & 0xf) == 0 ? 0 : (locations & 0xf) + VertexStageSgprBase;
    const u32 instance = (locations >> 4) == 0 ? 0 : (locations >> 4) + VertexStageSgprBase;
    return vertex | instance << 16;
}

/// Issue 36: the first vertex Vulkan adds to a direct draw (vertexOffset of an indexed draw, firstVertex of the others). The engine hands a draw's first vertex (the
/// base vertex of an indexed draw) to the vertex shader in the user data register the shader names (low nibble of byte 0x27 of its block, 0 = none), and the
/// shader adds it itself: in its fetch shader, or in its own code when it reads its vertices from buffers. A fetch shader becomes Vulkan's vertex input, so its add
/// has to come from Vulkan; a shader that adds it in its own code gets 0 here, or every vertex index is moved twice (the cave walls of s02_05 drew as huge
/// spikes, 650 ms of GPU time a frame). A shader that names no register cannot add it: Vulkan does. A fetch shader that reads another register than the one named
/// keeps the old way (Vulkan adds it) and is counted. Where the hardware index offset is 0 this is the normal renderer's rule (GetDrawOffsets), read from the
/// engine's side. TLG_VKB_BASE_VERTEX=0: the old way, Vulkan always adds it.
u32 VulkanFirstVertex(u64 ctx, const Variant& v, u32 first) {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_BASE_VERTEX"); return e == nullptr || e[0] != '0'; }();
    if (!on || first == 0) {
        return first;
    }
    const u64 vs = Rd<u64>(ctx, CtxVertexShader);
    const u32 location = vs != 0 ? (Rd<u8>(Rd<u64>(vs, 0x120), 0x27) & 0xfu) : 0u;
    const bool fetch_reads = v.has_fetch && v.fetch.vertex_offset_sgpr != -1;
    if (location == 0 || fetch_reads) {
        (fetch_reads && location != 0 && v.fetch.vertex_offset_sgpr != static_cast<s8>(location) ? c_base_vertex_other_register : c_base_vertex_vulkan)
            .fetch_add(1, std::memory_order_relaxed);
        return first;
    }
    c_base_vertex_shader.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

/// A3 (note 47; TLG_VKB_FAST_KEY, default on, =0 the full key): the same key value as VariantKeyFull, made cheaper. Only the set bits of the
/// shader-resource masks are visited; of a target view's words only the three the key uses are read; a bound object's class is remembered per thread by the object
/// and its unique id (+0x10, never reused), since a view's format, kind and resource and the resource's description do not change for its life.
/// TLG_VKB_KEY_CHECK=1 computes both keys at every draw and counts the differences (must be 0).
u64 ObjectClassMemo(u64 object) {
    if (object == 0 || !IsHeapPointer(object)) { // note 75: an unused slot in the shader's mask can hold anything (0x100000001 crashed here); never dereferenced
        return ObjectClass(object);
    }
    struct Entry {
        u64 object{}, id{~0ull}, cls{};
    };
    thread_local std::unique_ptr<std::array<Entry, 4096>> table;
    if (!table) {
        table = std::make_unique<std::array<Entry, 4096>>();
    }
    const u64 id = Rd<u64>(object, 0x10);
    Entry& e = (*table)[((object >> 3) ^ (object >> 15) ^ id) & 4095];
    if (e.object == object && e.id == id) {
        return e.cls;
    }
    e = {object, id, ObjectClass(object)};
    return e.cls;
}
u64 VariantKeyFull(u64 ctx, u64 vs, u64 ps, u64 fetch, const VkbVulkan::FixedState* fixed, bool indirect);
u64 VariantKeyFast(u64 ctx, u64 vs, u64 ps, u64 fetch, const VkbVulkan::FixedState* fixed, bool indirect) {
    u64 h = Mix(0x6a, Rd<u64>(vs, 0x10));
    if (indirect) {
        h = Mix(h, 0x1d00000000ull | IndirectSgprs(vs));
    }
    h = Mix(h, ps != 0 ? Rd<u64>(ps, 0x10) : 0);
    h = Mix(h, fetch);
    h = Mix(h, Rd<u64>(ctx, CtxInputLayout));
    h = Mix(h, reinterpret_cast<u64>(fixed));
    const auto views = TargetViews(ctx);
    for (u32 slot = 0; slot < 8; ++slot) {
        if (u64 a, b; TargetKey(views[slot], false, a, b)) { // D3: the engine's format and samples
            h = Mix(Mix(Mix(h, slot), a), b);
        }
    }
    if (const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView); dsv != 0) {
        if (u64 a, b; TargetKey(dsv, true, a, b)) {
            h = Mix(Mix(h, a), b);
        }
        h = Mix(h, static_cast<u64>(Core::Vkb::ViewFormat(Rd<u64>(dsv, 0x10))));
    }
    for (u32 stage = 0; stage < 2; ++stage) {
        const u64 shader = stage == 0 ? vs : ps;
        if (shader == 0) {
            continue;
        }
        const u64 masks[2] = {Rd<u64>(shader, 0x30 + 0xc0), Rd<u64>(shader, 0x30 + 0xc8)};
        for (u32 half = 0; half < 2; ++half) {
            for (u64 bits = masks[half]; bits != 0; bits &= bits - 1) {
                const u32 s = half * 64 + static_cast<u32>(std::countr_zero(bits));
                h = Mix(Mix(h, stage << 8 | s), ObjectClassMemo(Rd<u64>(ctx, SrvTable(stage) + s * 8)));
            }
        }
    }
    return h;
}
std::atomic<u64> c_key_checks{0}, c_key_differences{0};
u64 VariantKey(u64 ctx, u64 vs, u64 ps, u64 fetch, const VkbVulkan::FixedState* fixed, bool indirect) {
    static const bool fast = [] { const char* e = std::getenv("TLG_VKB_FAST_KEY"); return e == nullptr || e[0] != '0'; }();
    static const bool check = [] { const char* e = std::getenv("TLG_VKB_KEY_CHECK"); return e != nullptr && e[0] == '1'; }();
    if (check) {
        const u64 a = VariantKeyFast(ctx, vs, ps, fetch, fixed, indirect), b = VariantKeyFull(ctx, vs, ps, fetch, fixed, indirect);
        c_key_checks.fetch_add(1, std::memory_order_relaxed);
        if (a != b) {
            c_key_differences.fetch_add(1, std::memory_order_relaxed);
        }
        return b;
    }
    return fast ? VariantKeyFast(ctx, vs, ps, fetch, fixed, indirect) : VariantKeyFull(ctx, vs, ps, fetch, fixed, indirect);
}
u64 VariantKeyFull(u64 ctx, u64 vs, u64 ps, u64 fetch, const VkbVulkan::FixedState* fixed, bool indirect) {
    u64 h = Mix(0x6a, Rd<u64>(vs, 0x10));
    if (indirect) {
        h = Mix(h, 0x1d00000000ull | IndirectSgprs(vs));
    }
    h = Mix(h, ps != 0 ? Rd<u64>(ps, 0x10) : 0);
    h = Mix(h, fetch);
    h = Mix(h, Rd<u64>(ctx, CtxInputLayout));
    h = Mix(h, reinterpret_cast<u64>(fixed));
    const auto views = TargetViews(ctx);
    for (u32 slot = 0; slot < 8; ++slot) {
        if (u64 a, b; TargetKey(views[slot], false, a, b)) { // D3: the engine's format and samples (legacy: CB_COLOR_INFO, CB_COLOR_ATTRIB)
            h = Mix(Mix(Mix(h, slot), a), b);
        }
    }
    if (const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView); dsv != 0) {
        if (u64 a, b; TargetKey(dsv, true, a, b)) { // legacy: DB_Z_INFO, DB_STENCIL_INFO
            h = Mix(Mix(h, a), b);
        }
        h = Mix(h, static_cast<u64>(Core::Vkb::ViewFormat(Rd<u64>(dsv, 0x10))));
    }
    for (u32 stage = 0; stage < 2; ++stage) {
        const u64 shader = stage == 0 ? vs : ps;
        if (shader == 0) {
            continue;
        }
        const u64 lo = Rd<u64>(shader, 0x30 + 0xc0), hi = Rd<u64>(shader, 0x30 + 0xc8);
        for (u32 s = 0; s < 128; ++s) {
            if ((((s < 64 ? lo : hi) >> (s & 63)) & 1) != 0) {
                h = Mix(Mix(h, stage << 8 | s), ObjectClass(Rd<u64>(ctx, SrvTable(stage) + s * 8)));
            }
        }
    }
    return h;
}

/// issue 135: the reuse list of translated stages. A stage of the same code (hash, length, address), stage and mode is taken again when the shadPS4 stage
/// specialization (runtime info, fetch shader, binding start, the kinds of the bound buffers, images and samplers) matches, plus the formats and number
/// conversions of the buffers and vertex attributes. TLG_VKB_TRANSLATE_REUSE=0 turns it off.
bool TranslationReuseOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_TRANSLATE_REUSE"); return e == nullptr || e[0] != '0'; }();
    return on;
}
std::vector<u32> TranslationExtra(const VkbShader::Info& info) {
    std::vector<u32> extra;
    if (info.sw_stage == VkbShader::SwStage::Vertex) {
        if (VkbShader::Gcn::FetchShaderData fetch{}; VkbShader::Gcn::ParseFetchShader(info, fetch)) {
            for (const auto& attribute : fetch.attributes) {
                const auto sharp = attribute.GetSharp(info);
                extra.push_back(static_cast<u32>(sharp.GetDataFmt()) << 16 | static_cast<u32>(sharp.GetNumberFmt()) << 8 | static_cast<u32>(sharp.GetNumberConversion()));
            }
        }
    }
    for (const auto& b : info.buffers) {
        const auto sharp = b.GetSharp(info);
        extra.push_back(static_cast<u32>(sharp.GetDataFmt()) << 24 | static_cast<u32>(sharp.GetNumberFmt()) << 16 | sharp.GetStride() << 1 | (sharp ? 1u : 0u));
    }
    return extra;
}
struct TranslationEntry {
    VkbShader::Info info;
    VkbShader::StageSpecialization spec;
    std::vector<u32> extra;
    VkbShader::RuntimeInfo runtime_out;
    VkbShader::Backend::Bindings binding_out;
    std::vector<u32> spv;
    explicit TranslationEntry(const VkbShader::RuntimeInfo& runtime) : runtime_out{runtime} {}
};
std::mutex translation_mutex;
std::unordered_map<u64, std::vector<std::unique_ptr<TranslationEntry>>> translations;
constexpr size_t MaxTranslationsPerKey = 16;
bool TranslationReuseFind(u64 key, VkbShader::Info& info, VkbShader::RuntimeInfo& runtime, VkbShader::Backend::Bindings& binding, std::vector<u32>& spv) {
    std::scoped_lock lock{translation_mutex};
    const auto it = translations.find(key);
    if (it == translations.end()) {
        return false;
    }
    for (const auto& e : it->second) {
        VkbShader::Info candidate = e->info; // the translated layout, with this draw's user data
        candidate.user_data = info.user_data;
        candidate.pgm_base = info.pgm_base;
        candidate.indexed = info.indexed;
        candidate.constant_offsets = info.constant_offsets;
        candidate.RefreshFlatBuf();
        if (!e->spec.Matches(candidate, runtime, binding) || TranslationExtra(candidate) != e->extra) {
            continue;
        }
        info = std::move(candidate);
        runtime = e->runtime_out;
        binding = e->binding_out;
        spv = e->spv;
        return true;
    }
    return false;
}
void TranslationReuseAdd(u64 key, VkbShader::Info& info, const VkbShader::RuntimeInfo& runtime_in, const VkbShader::RuntimeInfo& runtime_out,
                         const VkbShader::Backend::Bindings& binding_in, const VkbShader::Backend::Bindings& binding_out, const std::vector<u32>& spv) {
    if (info.translation_failed || info.sw_stage == VkbShader::SwStage::TessellationControl || info.sw_stage == VkbShader::SwStage::TessellationEval) {
        return;
    }
    info.RefreshFlatBuf();
    auto e = std::make_unique<TranslationEntry>(runtime_out);
    e->info = info;
    e->spec = VkbShader::StageSpecialization(info, runtime_in, profile, binding_in);
    e->spec.info = &e->info;
    e->extra = TranslationExtra(info);
    e->binding_out = binding_out;
    e->spv = spv;
    std::scoped_lock lock{translation_mutex};
    auto& list = translations[key];
    if (list.size() < MaxTranslationsPerKey) {
        list.push_back(std::move(e));
    }
}

/// Compiles one stage (mutex held): its user data, program, translation, module and slot map
std::atomic<u32> next_constant_id{1}; ///< issue 42: the session-unique numbers of the shaders' constant S#s (Stage::constant_ids)
bool CompileStage(u64 ctx, u64 shader, u32 stage, u64 code_address, Stage& out, VkbShader::Backend::Bindings& binding, bool record = false) {
    const VkbShader::Profile& stage_profile = record ? record_profile : profile; // R2
    const HwStage hw = stage == 0 ? HwStage::Vertex : HwStage::Fragment;
    const SwStage sw = stage == 0 ? SwStage::Vertex : SwStage::Fragment;
    const u32 inputs_offset = stage == 0 ? 0x28 : 0x3c;
    Core::NativeDraw::BuildUserLayout(ctx, stage, shader, inputs_offset, false, out.layout);
    if (out.layout.unsupported_types != 0) {
        c_unsupported_layout_bits.fetch_or(out.layout.unsupported_types, std::memory_order_relaxed);
        c_fail[FailLayout].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (code_address == 0 || !IsGuestAddress(code_address)) {
        c_fail[FailNoCode].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const u32* code = reinterpret_cast<const u32*>(code_address);
    const auto& binary = VkbAmdGpu::SearchBinaryInfo(code);
    const VkbShader::ShaderParams params{
        .user_data = std::span<const u32, VkbShader::ShaderParams::NumShaderUserData>{out.layout.sgpr.data(), VkbShader::ShaderParams::NumShaderUserData},
        .code = std::span<const u32>{code, binary.length / sizeof(u32)},
        .hash = binary.shader_hash,
    };
    out.info = std::make_unique<VkbShader::Info>(hw, sw, params);
    out.info->pgm_base = params.Base();
    out.info->indexed = record;          // R2: resources through the global set and the draw's row
    out.info->constant_offsets = record; // (the row layout's two entries behind a read-only guest buffer's slot)
    out.row_start = binding.indexed;
    std::vector<u32> spv;
    // note 41, test only: TLG_VKB_TEST_PS_PROBE=<pixel shader hash>: render target 3 gets the coordinates of the shader's first 2D texture sample (x, y) and the first
    // barycentric weight they use (z), so a G-buffer dump shows what the shader receives
    static const u64 probe = [] { const char* e = std::getenv("TLG_VKB_TEST_PS_PROBE"); return e != nullptr ? std::strtoull(e, nullptr, 16) : 0ull; }();
    // TLG_VKB_SHADER_MEMO / TLG_VKB_SHADER_PREBUILD (shader store): a stage translated before with the same inputs, in this session (or an earlier one, on disk), is taken from the store;
    // issue 135: a stage the store lacks may still be in the reuse list (same code, settings and resource kinds, this session). The debug census and the probe work on the
    // translated program, so with them on every stage is translated
    namespace Store = Core::Vkb::ShaderStore;
    const bool debug_tool = Core::Vkb::FlatCensus::On() || probe != 0;
    const Store::Mode store_mode = debug_tool ? Store::Mode::Off : Store::GetMode();
    if (debug_tool && Store::GetMode() != Store::Mode::Off) {
        Store::NoteBypass();
    }
    const bool reuse = TranslationReuseOn() && !debug_tool;
    const u64 reuse_key = Mix(Mix(Mix(binary.shader_hash, binary.length), params.Base()), static_cast<u64>(stage) << 1 | (record ? 1u : 0u));
    const VkbShader::RuntimeInfo runtime_in = out.runtime;
    const VkbShader::Backend::Bindings binding_in = binding;
    const Store::Inputs store_in{.params = params, .runtime = runtime_in, .start = binding, .salt = record ? 1u : 0u}; // R2: a record-mode stage is another translation
    out.runtime_in = runtime_in;
    out.start_in = binding;
    Store::Translation stored;
    const bool found = store_mode != Store::Mode::Off && Store::Find(store_in, stored);
    const bool reused = !found && reuse && TranslationReuseFind(reuse_key, *out.info, out.runtime, binding, spv);
    if (!reused && (!found || store_mode == Store::Mode::Check)) {
        // TLG_VKB_SHADER_RECIPES: a stage the store lacks also keeps its recipe (the guest memory the translation reads, recorded on this thread)
        Core::Vkb::ShaderRecipe::Recording recording{!found && store_mode != Store::Mode::Off && Store::OnDisk() && Core::Vkb::ShaderRecipe::On()};
        const u64 translate_t0 = NowNs();
        const auto program = VkbShader::TranslateProgram(params.code, *pools, *out.info, out.runtime, stage_profile);
        if (out.info->translation_failed) {
            c_fail[FailTranslation].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (Core::Vkb::FlatCensus::On()) { // note 77 census (debug)
            Core::Vkb::FlatCensus::Note(program, shader ? Rd<u64>(shader, 0x120) : 0, inputs_offset, stage == 0 ? "vs" : "ps", out.info->pgm_hash);
        }
        if (probe != 0 && sw == VkbShader::SwStage::Fragment && out.info->pgm_hash == probe) {
            VkbShader::IR::Inst* coords = nullptr;
            for (VkbShader::IR::Block* block : program.blocks) {
                for (VkbShader::IR::Inst& inst : block->Instructions()) {
                    if (coords == nullptr && inst.GetOpcode() == VkbShader::IR::Opcode::ImageSampleImplicitLod && inst.Arg(1).TryInst() != nullptr &&
                        inst.Arg(1).TryInst()->GetOpcode() == VkbShader::IR::Opcode::CompositeConstructF32x2) {
                        coords = inst.Arg(1).TryInst();
                    }
                }
            }
            u32 patched = 0;
            if (coords != nullptr) {
                const VkbShader::IR::Value u = coords->Arg(0), v = coords->Arg(1);
                VkbShader::IR::Value w = u;
                if (const VkbShader::IR::Inst* fma = u.TryInst(); fma != nullptr && fma->GetOpcode() == VkbShader::IR::Opcode::FPFma32) {
                    w = fma->Arg(1);
                }
                for (VkbShader::IR::Block* block : program.blocks) {
                    for (VkbShader::IR::Inst& inst : block->Instructions()) {
                        if (inst.GetOpcode() == VkbShader::IR::Opcode::SetAttribute && inst.Arg(0).Attribute() == VkbShader::IR::Attribute::RenderTarget3) {
                            const u32 comp = inst.Arg(2).U32();
                            inst.SetArg(1, comp == 0 ? u : comp == 1 ? v : comp == 2 ? w : VkbShader::IR::Value{1.0f});
                            ++patched;
                        }
                    }
                }
            }
            LOG_WARNING(Render_Vulkan, "vkb ps probe {:#x}: coordinates {} found, {} exports patched", probe, coords != nullptr, patched);
        }
        spv = VkbShader::Backend::SPIRV::EmitSPIRV(stage_profile, out.runtime, program, binding);
        if (reuse) {
            TranslationReuseAdd(reuse_key, *out.info, runtime_in, out.runtime, binding_in, binding, spv);
        }
        if (!found) {
            Store::NoteBuild(params.hash, static_cast<u32>(hw));
            if (store_mode != Store::Mode::Off && Store::Keep(store_in, stage_profile, *out.info, out.runtime, binding, spv, {}, NowNs() - translate_t0)) {
                recording.Take(store_in, *out.info, 0);
            }
        }
    }
    if (found && (store_mode != Store::Mode::Check || Store::Same(store_in, stored, *out.info, out.runtime, binding, spv, {}))) {
        out.info = std::move(stored.info);
        out.runtime = stored.runtime;
        binding = stored.end;
        spv = std::move(stored.spv);
    }
    { // debug: TLG_VKB_DUMP_SPV=<hashes, hex, comma separated>: each compiled stage of those shaders -> log/vkb_spv/<vs|ps>_<hash>_<n>.spv, its GCN code (.gcn) and user data (.ud)
        static const std::vector<u64> dump_spv = [] {
            std::vector<u64> list;
            if (const char* e = std::getenv("TLG_VKB_DUMP_SPV")) {
                for (const char* p = e; *p != 0;) {
                    char* end = nullptr;
                    const u64 h = std::strtoull(p, &end, 16);
                    if (end == p) {
                        break;
                    }
                    list.push_back(h);
                    p = (*end == ',' || *end == '+') ? end + 1 : end;
                }
            }
            return list;
        }();
        if (std::ranges::find(dump_spv, static_cast<u64>(out.info->pgm_hash)) != dump_spv.end()) {
            static std::atomic<u32> dumped{0};
            const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_spv";
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            const std::string base = (dir / fmt::format("{}_{:08x}_{}_s{}", stage == 0 ? "vs" : "ps", out.info->pgm_hash, dumped.fetch_add(1), Core::Vkb::GraphicsSubmits())).string();
            if (FILE* f = std::fopen((base + ".spv").c_str(), "wb")) {
                std::fwrite(spv.data(), sizeof(u32), spv.size(), f);
                std::fclose(f);
            }
            if (FILE* f = std::fopen((base + ".gcn").c_str(), "wb")) {
                std::fwrite(params.code.data(), sizeof(u32), params.code.size(), f);
                std::fclose(f);
            }
            if (FILE* f = std::fopen((base + ".ud").c_str(), "wb")) {
                std::fwrite(out.layout.sgpr.data(), sizeof(u32), out.layout.sgpr.size(), f);
                std::fclose(f);
            }
        }
    }
    out.module = VkbVulkan::CompileSPV(spv, instance->GetDevice());
    Core::Vkb::ShaderDump::Keep(stage == 0 ? Core::Vkb::ShaderDump::Stage::Vertex : Core::Vkb::ShaderDump::Stage::Pixel, out.info->pgm_hash, spv, params.code); // note 89
    // the slot map: the same layout with a tag per slot, flattened by the shader's own table walker
    Core::NativeDraw::UserLayout tagged;
    Core::NativeDraw::BuildUserLayout(ctx, stage, shader, inputs_offset, true, tagged);
    const std::span<const u32> real = out.info->user_data;
    out.info->user_data = std::span<const u32>{tagged.sgpr.data(), tagged.sgpr.size()};
    out.info->RefreshFlatBuf();
    const std::vector<u32> flat = out.info->flattened_ud_buf;
    out.info->user_data = real;
    out.info->RefreshFlatBuf();
    const auto count = [&](u32 kind, const Source& s) {
        c_src[stage][kind][s.cls == Source::Class::Unmapped ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
    };
    for (const auto& b : out.info->buffers) {
        Source s = SourceOf(b.sharp_fetch, flat, b.IsSpecial(), b.post_op == VkbShader::SharpFetchPostOp::OffsetByProgramBase);
        if (s.cls == Source::Class::ProgramConstant) {
            const Core::NativeDesc::BufferDesc d = Core::NativeDesc::BufferDescOf(b.GetSharp(*out.info));
            s.address = d.address;
            s.size = d.size;
        }
        out.buffers.push_back(s);
        count(0, s);
    }
    for (const auto& image : out.info->images) {
        out.images.push_back(SourceOf(image.sharp_fetch, flat, false, false));
        count(1, out.images.back());
        out.image_types.push_back(Core::Vkb::ModuleViewType(image.GetSharp(*out.info).GetHostViewType(image.is_array)));
    }
    for (const auto& b : out.info->buffers) {
        out.has_flat |= b.buffer_type == VkbShader::BufferType::Flatbuf;
    }
    out.index = stage;
    out.shader = shader;
    for (const auto& sampler : out.info->samplers) {
        Source s = SourceOf(sampler.sharp_fetch, flat, false, false);
        std::unique_ptr<VideoCore::Vkb::Sampler> constant;
        Core::NativeDesc::SamplerDesc constant_desc{};
        u32 constant_id = 0;
        using Summary = std::remove_cvref_t<decltype(sampler.sharp_fetch)>::Summary;
        if (s.cls == Source::Class::Unmapped && sampler.sharp_fetch.summary == Summary::MultiLoad && sampler.sharp_fetch.load_mask == 0) {
            // the S# is a constant of the shader's code (no table load): our sampler made from it, once
            const auto sharp = sampler.GetSharp(*out.info);
            constant_desc = VideoCore::Vkb::SamplerDescFromSharp(sharp);
            constant_id = next_constant_id.fetch_add(1, std::memory_order_relaxed);
            constant = std::make_unique<VideoCore::Vkb::Sampler>(*instance, constant_desc, VkbAmdGpu::BorderColorBuffer{}, sampler.is_depth);
            s.cls = Source::Class::ProgramConstant;
        }
        out.samplers.push_back(s);
        out.constant_samplers.push_back(std::move(constant));
        out.constant_descs.push_back(constant_desc);
        out.constant_ids.push_back(constant_id);
        count(2, out.samplers.back());
    }
    return true;
}

/// Compiles the variant of the draw (mutex held)
/// R2: can a record-mode stage's resources all be found through the global set and the row (counted by the reason it cannot)
bool RecordEligible(const Stage& st) {
    using BT = VkbShader::BufferType;
    const auto refuse = [](u32 why) { c_rec_refused_by[why].fetch_add(1, std::memory_order_relaxed); return false; };
    for (const auto& b : st.info->buffers) {
        if (b.buffer_type == BT::Flatbuf) {
            return refuse(0);
        }
        if (b.buffer_type == BT::BdaPagetable || b.buffer_type == BT::FaultBuffer) {
            return refuse(3);
        }
        if (!VkbShader::Indexed::IsIndexedBufferType(b.buffer_type)) {
            return refuse(2);
        }
        if (b.buffer_type == BT::Guest && b.is_written) {
            return refuse(1);
        }
        if (b.buffer_type == BT::SharedMemory) {
            return refuse(7);
        }
    }
    for (const auto& image : st.info->images) {
        if (!VkbShader::Indexed::IsIndexedImage(image) || image.mip_fallback_mode == VkbShader::MipStorageFallbackMode::DynamicIndex) {
            return refuse(4);
        }
    }
    return true;
}

void Compile(u64 ctx, u64 vs, u64 ps, u64 fetch_object, const VkbVulkan::FixedState& fixed, Variant& v, bool indirect, bool record = false) {
    v.record = record;
    const u64 t0 = NowNs();
    const u64 vs_gnm = Rd<u64>(vs, 0x120);
    const u64 ps_gnm = ps != 0 ? Rd<u64>(ps, 0x120) : 0;
    Core::NativeShaderEnv::DrawEnv env;
    {
        const Core::NativeShaderEnv::Scope scope{vs_gnm, ps_gnm, fetch_object};
        (void)Core::NativeShaderEnv::Current(env);
    }
    auto& key = v.key;
    // the targets: depth from the depth view's words, colour from each render target view's words (the PS4 description the engine keeps in its views; stated in the notes)
    const DepthDesc depth_desc = DepthDescOf(ctx); // D3 (note 77): from the engine's depth view
    const bool db_enabled = fixed.depth_target || fixed.stencil_target;
    if (TargetCheckOn() && Rd<u64>(ctx, CtxDepthStencilView) != 0) {
        const DepthDesc w = DepthWordsDesc(ctx);
        c_target_checks.fetch_add(1, std::memory_order_relaxed);
        if (w.z != depth_desc.z || w.stencil != depth_desc.stencil || w.samples != depth_desc.samples) {
            TargetCheckNote(fmt::format("depth dxgi {}: words z {} stencil {} samples {}, engine z {} stencil {} samples {}", Rd<u32>(Rd<u64>(ctx, CtxDepthStencilView), 0x40),
                                        static_cast<u32>(w.z), static_cast<u32>(w.stencil), w.samples, static_cast<u32>(depth_desc.z), static_cast<u32>(depth_desc.stencil),
                                        depth_desc.samples));
        }
    }
    const Core::NativeTarget::DepthTarget depth = TargetCensusOn() ? DepthOf(ctx) : Core::NativeTarget::DepthTarget{}; // the census prints the words
    if (const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView); dsv != 0 && TargetCensusOn()) {
        const u64 resource = Rd<u64>(dsv, 0x30);
        TargetCensusNote(fmt::format("depth dxgi {} dim {} flags {} tex_samples {} tex_format {} -> z {} stencil {} samples {} | zvalid {} svalid {} db_enabled {}", Rd<u32>(dsv, 0x40),
                                     Rd<u32>(dsv, 0x44), Rd<u32>(dsv, 0x48), resource != 0 ? Rd<u32>(resource, 0x194) : 0, resource != 0 ? Rd<u32>(resource, 0x190) : 0,
                                     static_cast<u32>(depth.z_info.format), static_cast<u32>(depth.stencil_info.format), depth.NumSamples(), depth.DepthValid() ? 1 : 0,
                                     depth.StencilValid() ? 1 : 0, db_enabled ? 1 : 0));
    }
    key.z_format = depth_desc.z;
    key.stencil_format = depth_desc.stencil;
    if (const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView); dsv != 0) {
        v.depth_format = Core::Vkb::ViewFormat(Rd<u64>(dsv, 0x10));
    }
    if (v.depth_format != vk::Format::eD32SfloatS8Uint) {
        key.stencil_format = Core::NativeTarget::DepthTarget::StencilFormat::Invalid; // our depth image has no stencil
    }
    if (v.depth_format == vk::Format::eUndefined) {
        key.z_format = Core::NativeTarget::DepthTarget::ZFormat::Invalid;
    }
    key.depth_clamp_enable = fixed.depth_clamp;
    key.depth_clip_enable = fixed.depth_clip;
    key.negative_one_to_one = fixed.negative_one_to_one;
    key.provoking_first = fixed.provoking_first;
    key.prim_type = fixed.primitive_type;
    key.polygon_mode = fixed.polygon_mode;
    key.logic_op = fixed.logic_op;
    key.logic_op_enable = fixed.logic_op_enabled;
    key.depth_samples = db_enabled && (depth_desc.z != Core::NativeTarget::DepthTarget::ZFormat::Invalid ||
                                       depth_desc.stencil != Core::NativeTarget::DepthTarget::StencilFormat::Invalid) ? depth_desc.samples : 1;
    key.num_samples = key.depth_samples;
    key.alpha_not_exported = fixed.alpha_not_exported;
    const auto views = TargetViews(ctx);
    std::array<Core::NativeTarget::ColorTarget, 8> colors{};
    std::array<u32, Core::NativeTarget::RawTargetWords::ColorWords> words;
    for (u32 cb = 0; cb < 8 && fixed.color_enabled; ++cb) {
        if (!ColorOf(views[cb], colors[cb])) { // D3 (note 77): from the engine's view
            continue;
        }
        c_targets_seen.fetch_add(1, std::memory_order_relaxed);
        if (TargetCheckOn()) {
            c_target_checks.fetch_add(1, std::memory_order_relaxed);
            const auto& e = colors[cb];
            const auto w = ColorWords(views[cb], words) ? Core::NativeTarget::ColorFromWords(words.data()) : Core::NativeTarget::ColorTarget{};
            if (static_cast<bool>(w) != static_cast<bool>(e) || w.GetDataFmt() != e.GetDataFmt() || w.GetNumberFmt() != e.GetNumberFmt() ||
                w.GetNumberConversion() != e.GetNumberConversion() || w.info.comp_swap != e.info.comp_swap || w.info.blend_bypass != e.info.blend_bypass ||
                w.NumSamples() != e.NumSamples()) {
                TargetCheckNote(fmt::format("color dxgi {}: words fmt {} num {} swap {} bypass {} samples {} valid {}, engine fmt {} num {} swap {} samples {}",
                                            Rd<u32>(views[cb], 0x40), w.info.format, w.info.number_type, static_cast<u32>(w.info.comp_swap), w.info.blend_bypass, w.NumSamples(),
                                            static_cast<bool>(w) ? 1 : 0, e.info.format, e.info.number_type, static_cast<u32>(e.info.comp_swap), e.NumSamples()));
            }
        }
        if (TargetCensusOn() && ColorWords(views[cb], words)) {
            const u64 resource = Rd<u64>(views[cb], 0x30);
            const auto c = Core::NativeTarget::ColorFromWords(words.data());
            TargetCensusNote(fmt::format("color dxgi {} dim {} tex_samples {} -> fmt {} num {} swap {} bypass {} frag_log2 {} samp_log2 {} | data {} numfmt {} conv {} valid {}",
                                         Rd<u32>(views[cb], 0x40), Rd<u32>(views[cb], 0x44), resource != 0 ? Rd<u32>(resource, 0x194) : 0, c.info.format, c.info.number_type,
                                         static_cast<u32>(c.info.comp_swap), c.info.blend_bypass, c.attrib.num_fragments_log2, c.attrib.num_samples_log2,
                                         static_cast<u32>(c.GetDataFmt()), static_cast<u32>(c.GetNumberFmt()), static_cast<u32>(c.GetNumberConversion()), c ? 1 : 0));
        }
        if (!colors[cb] || fixed.target_write_mask[cb] == 0) {
            continue;
        }
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = colors[cb].GetDataFmt();
        color_buffer.num_format = colors[cb].GetNumberFmt();
        color_buffer.num_conversion = colors[cb].GetNumberConversion();
        if (env.has_pixel) {
            VkbVulkan::ShaderBoundary::SetExportFormat(color_buffer, env.pixel.fragment.color_export_format[cb]);
        } else {
            VkbVulkan::ShaderBoundary::SetNoExport(color_buffer);
        }
        color_buffer.swizzle = colors[cb].Swizzle();
        const auto& bc = fixed.blend[cb];
        color_buffer.blend_self_scale = bc.enable && !colors[cb].info.blend_bypass && (bc.color_op == vk::BlendOp::eMin || bc.color_op == vk::BlendOp::eMax) &&
                                        bc.src_color == vk::BlendFactor::eSrcColor && bc.dst_color == vk::BlendFactor::eDstColor;
    }
    u32 samples = key.num_samples;
    for (u32 cb = 0; cb < 8; ++cb) {
        if (colors[cb]) {
            samples = std::max(samples, colors[cb].NumSamples());
        }
    }
    // the pixel stage first, then the vertex stage, with one binding counter (the pipeline layout walks the stages in that order)
    VkbShader::Backend::Bindings binding{};
    v.has_ps = env.has_pixel && ps != 0;
    if (v.has_ps) {
        auto& rt = v.ps.runtime;
        rt.Initialize(HwStage::Fragment, SwStage::Fragment);
        rt.hw.fs.dual_source_blending = fixed.DualSourceBlending();
        for (u32 i = 0; i < VkbShader::MaxColorBuffers; ++i) {
            rt.hw.fs.color_buffers[i] = key.color_buffers[i];
        }
        rt.hw.fs.clip_distance_emulation =
            ((env.has_vertex && VkbVulkan::ShaderBoundary::GuestClipDistances(env.vertex)) || fixed.user_clip_plane_mask != 0) && profile.needs_clip_distance_emulation;
        VkbVulkan::ShaderBoundary::ApplyShaderEnv(rt, env, HwStage::Fragment, samples);
        if (!CompileStage(ctx, ps, 1, env.pixel.program.address, v.ps, binding, record)) {
            c_failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    } else {
        c_depth_only.fetch_add(1, std::memory_order_relaxed);
    }
    key.mrt_mask = v.has_ps ? v.ps.info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);
    {
        auto& rt = v.vs.runtime;
        rt.Initialize(HwStage::Vertex, SwStage::Vertex);
        rt.hw.vs.user_clip_plane_mask = fixed.user_clip_plane_mask;
        rt.hw.vs.emulate_depth_negative_one_to_one = !instance->IsDepthClipControlSupported() && fixed.negative_one_to_one;
        rt.hw.vs.clip_disable = fixed.clip_disabled;
        rt.sw.vs.tess_emulated_primitive =
            fixed.primitive_type == Core::NativePrimitive::Kind::RectList || fixed.primitive_type == Core::NativePrimitive::Kind::QuadList;
        VkbVulkan::ShaderBoundary::ApplyShaderEnv(rt, env, HwStage::Vertex, samples);
        if (indirect) {
            // V5a: the indirect draw's base vertex and instance come from its arguments (gl_BaseVertex / gl_BaseInstance where the console's user data had them)
            const u32 sgprs = IndirectSgprs(vs);
            rt.sw.vs.vertex_sgpr_offset = static_cast<u16>(sgprs & 0xffff);
            rt.sw.vs.instance_sgpr_offset = static_cast<u16>(sgprs >> 16);
            rt.sw.vs.is_indirect = true;
        }
        if (!CompileStage(ctx, vs, 0, env.vertex.program.address, v.vs, binding, record)) {
            c_failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    // the vertex attributes: which element of the vertex buffer table each reads (the tagged walk of the fetch shader's table)
    if (v.vs.info->has_fetch_shader) {
        if (!VkbShader::Gcn::ParseFetchShader(*v.vs.info, v.fetch)) {
            c_fail[FailFetch].fetch_add(1, std::memory_order_relaxed);
            c_failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        v.has_fetch = true;
        Core::NativeDraw::UserLayout tagged;
        Core::NativeDraw::BuildUserLayout(ctx, 0, vs, 0x28, true, tagged);
        for (const auto& attribute : v.fetch.attributes) {
            Source s;
            const u32 pointer_lo = attribute.sgpr_base < 16 ? tagged.sgpr[attribute.sgpr_base] : 0;
            const u32 pointer_hi = attribute.sgpr_base + 1 < 16 ? tagged.sgpr[attribute.sgpr_base + 1] : 0;
            const u64 table = static_cast<u64>(pointer_hi) << 32 | pointer_lo;
            u32 kind = 0, slot = 0;
            if (table != 0 && Core::NativeDraw::UserLayout::TagOf(*reinterpret_cast<const u32*>(table + attribute.dword_offset * 4ull), kind, slot) && kind == 5) {
                s.cls = Source::Class::Slot;
                s.kind = 5;
                s.slot = static_cast<u16>(slot);
                c_attr_mapped.fetch_add(1, std::memory_order_relaxed);
            } else {
                c_attr_unmapped.fetch_add(1, std::memory_order_relaxed);
            }
            v.attributes.push_back(s);
        }
    }
    if (!instance->IsVertexInputDynamicState()) {
        c_fail[FailNoDynamicVertexInput].fetch_add(1, std::memory_order_relaxed);
        c_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // the second pass of the key: targets the pixel shader does not write are dropped; blend, write masks, samples
    u8 color_samples = 0;
    for (u32 cb = 0; cb < key.num_color_attachments && fixed.color_enabled; ++cb) {
        const u32 target_mask = fixed.target_write_mask[cb];
        if (!colors[cb] || target_mask == 0) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(VkbShader::PsColorBuffer));
            continue;
        }
        if (fixed.blend[cb].enable && !colors[cb].info.blend_bypass) {
            key.blend[cb] = fixed.blend[cb];
        }
        key.write_masks[cb] = vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};
        color_samples = static_cast<u8>(colors[cb].NumSamples());
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max<u8>(key.num_samples, color_samples);
    }
    key.stage_hashes[static_cast<u32>(SwStage::Vertex)] = v.vs.info->pgm_hash;
    key.stage_hashes[static_cast<u32>(SwStage::Fragment)] = v.has_ps ? v.ps.info->pgm_hash : 0;
    // our pipeline
    std::array<const VkbShader::Info*, VkbVulkan::Vkb::MaxShaderStages> infos{};
    std::array<VkbShader::RuntimeInfo, VkbVulkan::Vkb::MaxShaderStages> runtimes{};
    std::array<vk::ShaderModule, VkbVulkan::Vkb::MaxShaderStages> modules{};
    infos[static_cast<u32>(SwStage::Vertex)] = v.vs.info.get();
    runtimes[static_cast<u32>(SwStage::Vertex)] = v.vs.runtime;
    modules[static_cast<u32>(SwStage::Vertex)] = v.vs.module;
    if (v.has_ps) {
        infos[static_cast<u32>(SwStage::Fragment)] = v.ps.info.get();
        runtimes[static_cast<u32>(SwStage::Fragment)] = v.ps.runtime;
        modules[static_cast<u32>(SwStage::Fragment)] = v.ps.module;
    }
    VkbVulkan::Vkb::GraphicsPipeline::SerializationSupport sdata{};
    for (u32 i = 0; i < key.num_color_attachments; ++i) {
        v.color_formats[i] = instance->GetSupportedFormat(VkbVulkan::Vkb::LiverpoolToVK::SurfaceFormat(key.color_buffers[i].data_format, key.color_buffers[i].num_format),
                                                          vk::FormatFeatureFlagBits2::eColorAttachment);
    }
    if (record) { // R2: every resource of both stages through the global set, or no record twin
        if (!v.attributes.empty()) {
            c_rec_refused_by[7].fetch_add(1, std::memory_order_relaxed);
        }
        if (!v.attributes.empty() || !RecordEligible(v.vs) || (v.has_ps && !RecordEligible(v.ps))) {
            c_rec_refused.fetch_add(1, std::memory_order_relaxed);
            c_failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        v.row_dwords = binding.indexed;
    }
    if (!record) { // the warm-up builds the list's pipelines with the plain profile
        Core::Vkb::PipelineWarm::Check(key, static_cast<u32>(v.depth_format), {&v.vs.runtime_in, &v.vs.start_in},
                                       v.has_ps ? Core::Vkb::PipelineWarm::StageInputs{&v.ps.runtime_in, &v.ps.start_in} : Core::Vkb::PipelineWarm::StageInputs{}); // is this pipeline among the list's converted keys
    }
    v.pipeline = std::make_unique<VkbVulkan::Vkb::GraphicsPipeline>(*instance, Core::Vkb::Device::Scheduler(), Core::Vkb::Device::DescHeap(), record ? record_profile : profile, key, vk::PipelineCache{Core::Vkb::PipelineWarm::Cache()},
                                                             std::span<const VkbShader::Info*, VkbVulkan::Vkb::MaxShaderStages>{infos},
                                                             std::span<const VkbShader::RuntimeInfo, VkbVulkan::Vkb::MaxShaderStages>{runtimes}, v.has_fetch ? &v.fetch : nullptr,
                                                             std::span<const vk::ShaderModule>{modules}, sdata, false, v.depth_format,
                                                             record ? Core::Vkb::Indexed::Layout() : vk::DescriptorSetLayout{});
    if (!v.pipeline->Handle()) {
        c_fail[FailPipeline].fetch_add(1, std::memory_order_relaxed);
        c_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    v.ok = true;
    c_compiled.fetch_add(1, std::memory_order_relaxed);
    if (record) {
        c_rec_compiled.fetch_add(1, std::memory_order_relaxed);
    }
    Core::Vkb::FrameNote(fmt::format("new graphics variant: vs {:#x} ps {:#x} colors {} depth {}", v.vs.info ? v.vs.info->pgm_hash : 0ull,
                                     v.has_ps && v.ps.info ? v.ps.info->pgm_hash : 0ull, key.num_color_attachments, v.depth_format != vk::Format::eUndefined ? 1 : 0)); // note 44
    c_compile_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
}

/// The engine object in slot `slot` of graphics stage `stage`'s table of `kind` (1 resource, 2 sampler, 3 constant buffer)
u64 SlotObject(u64 ctx, u32 stage, u32 kind, u32 slot) {
    u64 table = 0, limit = 0;
    switch (kind) {
    case 1: table = 0x580 + stage * 0x400, limit = 128; break;
    case 2: table = 0x1d80 + stage * 0x80, limit = 16; break;
    case 3: table = 0x280 + stage * 0x80, limit = 16; break;
    default: return 0;
    }
    if (slot >= limit) {
        return 0;
    }
    const u64 object = Rd<u64>(ctx, table + slot * 8);
    return object != 0 && IsHeapPointer(object) && (object & 7) == 0 ? object : 0;
}

/// What one stage binds: descriptor writes (in the pipeline layout's order) are appended; buffer offsets go into the push data
struct Bound {
    boost::container::small_vector<vk::DescriptorBufferInfo, 32> buffers;
    boost::container::small_vector<vk::DescriptorImageInfo, 32> images;
    boost::container::small_vector<vk::DescriptorImageInfo, 16> samplers;
    boost::container::small_vector<u8, 32> stageable; ///< V4h: per buffer, read only in place
    boost::container::small_vector<std::pair<u64, u64>, 32> guest; ///< note 41, debug dumps: per buffer, the guest range it stands for (0: none)
};

/// A4 (note 48): a draw's record for the recording thread, taken from the job pool and filled in place by Record (before: locals moved into a lambda, the lambda
/// moved into a heap job, and a copy of the whole fixed state in it)
using Ranges = boost::container::small_vector<std::pair<u32, u32>, 6>;
/// R2b (note 64): a buffer the capture leaves to the recording thread: the engine object's fields copied at the draw (the object is never read again: it may be gone
/// when the job runs), or (object 0) a range by address in `offset`. Frame ring windows are never late (they are arithmetic on the capture's placement).
struct LateBuffer {
    enum What : u8 { Stage, Vertex, Index, Args };
    u64 object{}, id{}, memory{}, bytes{}, offset{}, size{};
    u16 slot{};      ///< Stage: index into Bound::buffers; Vertex: the binding
    u8 push_index{}; ///< Stage: its push data buffer offset
    What what{};
    bool written{};
};
using LateList = boost::container::small_vector<LateBuffer, 16>;
/// R2b part 2b (note 74, TLG_VKB_LATE_IMAGES=1, off by default): the capture keeps only the engine view ids of the draw's shader images and targets; our views,
/// attachments and the pass size are found by the resolver (stream order: deletions reach our tables in order, R2a; images keep one layout, part 2a)
struct LateImage {
    u64 view_id{};
    vk::ImageViewType type{};
    u16 slot{}; ///< index into Bound::images
    bool written{};
};
struct LateTarget {
    u64 view_id{};
    vk::Format format{};
    u8 cb{}; ///< colour target; 0xff: the depth target
};
using LateImages = boost::container::small_vector<LateImage, 16>;
using LateTargets = boost::container::small_vector<LateTarget, 9>;
bool ResolveOnRecorder();
bool LateImagesOn();
struct DrawJob final : Stream::Job {
    LateList late; ///< R2b: resolved first in Run, on the recording thread (R4a: in Prepare, on the resolver)
    LateImages late_images;   ///< R2b part 2b
    LateTargets late_targets; ///< R2b part 2b
    bool want_stencil{};      ///< R2b part 2b: the variant has a stencil format (the depth attachment's stencil is used)
    bool AnyLate() const {
        return !late.empty() || !late_images.empty() || !late_targets.empty();
    }
    bool ResolveLate();
    bool prepared{}, prepared_ok{};
    void Prepare() override {
        prepared = true;
        prepared_ok = !AnyLate() || ResolveLate();
    }
    std::vector<std::pair<u64, std::string>> dump_views;
    bool dump_buffers{};
    std::string dump_prefix; ///< note 90: debugger mode's slow-pass dump goes to vkb_image_dump/<prefix>
    u64 debug_id{};          ///< note 91: debugger mode: the draw's id (per-draw GPU time, capture row); 0 otherwise
    u64 vs_hash{};
    bool capture{};          ///< note 91: the draw belongs to the frame capture
    bool jittered{};         ///< F1c: the scene geometry's sub-pixel offset applies (viewport)
    std::array<float, 2> jitter{};
    boost::container::small_vector<Stream::Use, 64> uses;
    boost::container::small_vector<Stream::Use, 9> attachment_uses;
    Stream::Pass pass{};
    Bound b;
    Ranges ranges;
    boost::container::small_vector<vk::DescriptorType, 32> image_types;
    vk::Pipeline pipeline{};
    vk::PipelineLayout layout{};
    bool push_descriptors{};
    vk::DescriptorSetLayout set_layout{};
    VkbShader::PushData push{};
    const VkbVulkan::FixedState* fixed_state{};
    std::array<vk::ColorComponentFlags, VkbAmdGpu::NUM_COLOR_BUFFERS> masks{};
    boost::container::small_vector<vk::VertexInputBindingDescription2EXT, 16> input_bindings;
    boost::container::small_vector<vk::VertexInputAttributeDescription2EXT, 16> input_attributes;
    boost::container::small_vector<vk::Buffer, 16> vertex_buffers;
    boost::container::small_vector<vk::DeviceSize, 16> vertex_offsets;
    boost::container::small_vector<u64, 16> vertex_sizes;
    vk::Buffer index_buffer{};
    u64 index_offset{};
    vk::IndexType index_type{vk::IndexType::eUint16};
    u64 index_size{};
    vk::Buffer args_buffer{};
    u64 args_offset{};
    u32 kind{};
    u64 a1{}, a2{}, a3{};
    u32 first_vertex{}; ///< issue 36: what Vulkan adds to the vertex index (VulkanFirstVertex)
    u32 instances{};
    u64 ps_hash{};
    u32 color_format0{};
    bool record{};    ///< R3: a record-mode draw: its resources are in its row (push row base), the global set is bound
    void Run(vk::CommandBuffer cmd) override;
    static void* operator new(size_t) {
        return Stream::JobBlocks<sizeof(DrawJob)>::Get();
    }
    static void operator delete(void* p) {
        Stream::JobBlocks<sizeof(DrawJob)>::Put(p);
    }
};

/// Binds one stage (stream recording held, outside a pass: images may be moved to the general layout). False: the draw cannot run (counted).
/// R5 TLG_VKB_DRAW_DUMP: one line into log/vkb_draw_dump.txt (its own file: the game log drops lines when it is busy)
std::mutex g_live_dump_mutex;
std::string g_live_dump_label; ///< debug: the label of the armed live image dump (TLG_VKB_DRAW_CENSUS_TRIGGER "... ps=<hash>")
std::atomic<u64> g_live_dump_ps{0};
std::atomic<u64> g_live_dump_vs{0}; ///< debug: "<label> vs=<hash>": the first draw of that vertex shader is dumped (as ps= does)
void DrawDumpLine(const std::string& line) {
    static std::mutex m;
    std::scoped_lock lock{m};
    if (FILE* f = std::fopen((Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_draw_dump.txt").string().c_str(), "a")) {
        std::fprintf(f, "%s\n", line.c_str());
        std::fclose(f);
    }
}

std::atomic<u64> c_cb_at_draw{0};
std::atomic<u64> c_dc_flat_n{0}, c_dc_flat_same{0}, c_dc_flat_dwords{0}; ///< X0 census
thread_local bool t_depth_pass = false;
bool DepthCensusOn();
thread_local u64 t_dc_copies = 0; ///< note 109 census: constant buffer copies at the draw on this thread
/// O1 census (debug, TLG_VKB_CB_CENSUS=1): a constant buffer copied at the draw (R7) whose bytes equal what was copied for the same address earlier in the same
/// graphics submission (a copy a once-per-Map scheme would save)
bool CbCensusOn() { static const bool on = [] { const char* e = std::getenv("TLG_VKB_CB_CENSUS"); return e != nullptr && e[0] == '1'; }(); return on; }
std::atomic<u64> c_cbc_copies{0}, c_cbc_bytes{0}, c_cbc_same{0}, c_cbc_same_bytes{0}, c_cbc_same_address_changed{0};
thread_local std::unordered_map<u64, u64> t_cbc_seen; ///< address | size << 48 -> hash of the bytes copied last
thread_local u64 t_cbc_epoch = 0;

/// note 41: descriptor sets for the pipelines whose bindings exceed the push descriptor limit (32 on this GPU; the boy's G-buffer pipeline has 35). Pools are reused once
/// the stream's timeline passes the last submission that used them. Recording thread only.
std::atomic<u64> c_big_sets{0}, c_big_pools{0};
vk::DescriptorSet BigDescriptorSet(vk::DescriptorSetLayout set_layout) {
    struct Pool {
        vk::DescriptorPool pool;
        u64 last_value;
    };
    thread_local std::deque<Pool> used; // R4a: per recording thread
    thread_local vk::DescriptorPool current{};
    const auto device = instance->GetDevice();
    const auto fresh = [&]() -> vk::DescriptorPool {
        if (!used.empty() && Stream::CompletedValue() >= used.front().last_value) {
            const vk::DescriptorPool pool = used.front().pool;
            used.pop_front();
            (void)device.resetDescriptorPool(pool);
            return pool;
        }
        static constexpr std::array<vk::DescriptorPoolSize, 5> sizes{{{vk::DescriptorType::eStorageBuffer, 8192}, {vk::DescriptorType::eSampledImage, 8192},
                                                                       {vk::DescriptorType::eStorageImage, 2048}, {vk::DescriptorType::eSampler, 2048},
                                                                       {vk::DescriptorType::eUniformBuffer, 1024}}};
        const auto [result, pool] = device.createDescriptorPool({.maxSets = 256, .poolSizeCount = static_cast<u32>(sizes.size()), .pPoolSizes = sizes.data()});
        c_big_pools.fetch_add(1, std::memory_order_relaxed);
        return result == vk::Result::eSuccess ? pool : vk::DescriptorPool{};
    };
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!current) {
            current = fresh();
            if (!current) {
                return {};
            }
        }
        vk::DescriptorSet set{};
        const vk::DescriptorSetAllocateInfo info{.descriptorPool = current, .descriptorSetCount = 1, .pSetLayouts = &set_layout};
        if (device.allocateDescriptorSets(&info, &set) == vk::Result::eSuccess) {
            c_big_sets.fetch_add(1, std::memory_order_relaxed);
            return set;
        }
        used.push_back({current, Stream::RecordingValue()}); // full: its sets go out with this submission at the latest
        current = fresh();
    }
    return {};
}
std::atomic<u64> c_state_sent{0}, c_state_skipped{0}, c_state_pipelines{0}, c_state_vertex_inputs{0}; ///< note 60: the redundant-state filter
/// note 52 census (TLG_VKB_STAGE_CENSUS=1): does a stage resolve to exactly what it resolved at the previous draw (same shader, same buffer windows, views, samplers)?
bool StageCensusOn() { static const bool on = [] { const char* e = std::getenv("TLG_VKB_STAGE_CENSUS"); return e != nullptr && e[0] == '1'; }(); return on; }
std::atomic<u64> c_stage_same[2]{}, c_stage_changed[2]{}, c_stage_buffers_same[2]{}, c_stage_images_same[2]{}, c_stage_samplers_same[2]{};
thread_local u64 t_stage_hash[3]{}, t_stage_prev[2][3]{};
inline void StageMix(u32 part, u64 v) {
    t_stage_hash[part] = (t_stage_hash[part] ^ v) * 0x100000001b3ull;
}
/// Issue 42: a shader's constant S# with a draw's bias / anisotropy (`look` not plain): made once per description and look, kept for the session as the
/// slot samplers (Core::Vkb::SamplerFor); the plain one stays the stage's own. `look` is already the sampler's own (Aa::ForSampler: no bias on a depth read).
/// A per-thread memo keyed by the constant's session-unique number answers the repeat draws; the shared table (one mutex) only a thread's first use.
std::mutex constant_look_mutex;
std::unordered_map<std::string, std::unique_ptr<VideoCore::Vkb::Sampler>> constant_look_samplers; ///< key: the description's bytes, bias, anisotropy, depth
thread_local std::unordered_map<u64, vk::Sampler> t_constant_look; ///< key: constant number, bias, forced level, depth; samplers are never destroyed, numbers never reused
vk::Sampler ConstantSamplerFor(const Stage& st, u32 i, bool is_depth, Core::Vkb::Aa::SamplerLook look) {
    const Core::NativeDesc::SamplerDesc& desc = st.constant_descs[i];
    const u32 aniso = VideoCore::Vkb::ForcedAnisotropy(desc, look.aniso, is_depth); // the one rule (Core::NativeDesc::ForcedAnisotropy)
    if (look.bias_q == 0 && aniso == 0) {
        return st.constant_samplers[i]->Handle();
    }
    const u64 memo_key = static_cast<u64>(st.constant_ids[i]) << 32 | static_cast<u64>(static_cast<u16>(look.bias_q)) << 16 | static_cast<u64>(aniso & 0x7fffu) << 1 |
                         (is_depth ? 1u : 0u);
    if (const auto it = t_constant_look.find(memo_key); it != t_constant_look.end()) {
        return it->second;
    }
    static_assert(sizeof(Core::NativeDesc::SamplerDesc) == 16 * sizeof(u32), "its bytes are the key below: no padding bytes (native_desc.h)");
    std::string key(reinterpret_cast<const char*>(&desc), sizeof(desc));
    const std::array<s32, 3> extra{look.bias_q, static_cast<s32>(aniso), is_depth ? 1 : 0};
    key.append(reinterpret_cast<const char*>(extra.data()), sizeof(extra));
    vk::Sampler handle;
    {
        std::scoped_lock lock{constant_look_mutex};
        auto& made = constant_look_samplers[key];
        if (!made) {
            made = std::make_unique<VideoCore::Vkb::Sampler>(*instance, desc, VkbAmdGpu::BorderColorBuffer{}, is_depth, static_cast<float>(look.bias_q) * 0.25f, aniso);
        }
        handle = made->Handle();
    }
    t_constant_look.emplace(memo_key, handle);
    return handle;
}

bool BindStage(u64 ctx, Stage& st, vk::CommandBuffer cmd, VkbShader::PushData& push, u32& buffer_index, Bound& b, Ranges& ranges,
               boost::container::small_vector<Stream::Use, 64>& uses, LateList* late_list, LateImages* late_images = nullptr, Core::Vkb::Aa::SamplerLook look = {}) {
    if (StageCensusOn()) {
        for (u32 k = 0; k < 3; ++k) {
            t_stage_hash[k] = 0xcbf29ce484222325ull ^ reinterpret_cast<u64>(st.info.get());
        }
    }
    const u64 alignment = instance->StorageMinAlignment();
    const auto& info = *st.info;
    const u32 first_buffer = static_cast<u32>(b.buffers.size());
    // R5 TLG_VKB_DRAW_DUMP=<pixel or vertex shader hash, hex>: its 100th, 400th and 1,000th bind write what it samples to log/vkb_draw_dump.txt
    static const u64 dump_hash = [] { const char* e = std::getenv("TLG_VKB_DRAW_DUMP"); return e != nullptr ? std::strtoull(e, nullptr, 16) : 0ull; }();
    static std::atomic<u32> dump_uses{0};
    const bool dumping = dump_hash != 0 && st.info != nullptr && (st.info->pgm_hash == dump_hash || (st.info->pgm_hash & 0xffffffffull) == dump_hash) &&
                         [] {
                             static const u32 every = [] { const char* e = std::getenv("TLG_VKB_DRAW_DUMP_EVERY"); return e != nullptr ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u; }();
                             const u32 n = dump_uses.fetch_add(1, std::memory_order_relaxed);
                             return n == 100 || n == 400 || n == 1000 || (every != 0 && n % every == 0);
                         }();
    std::string dump_text;
    LateList* const late = dumping ? nullptr : late_list; // R2b: a dumped draw resolves here (its text needs the windows)
    LateImages* const late_img = dumping ? nullptr : late_images; // R2b part 2b
    for (u32 i = 0; i < st.buffers.size(); ++i) {
        const Source& src = st.buffers[i];
        const auto& desc = info.buffers[i];
        // R7 (note 56): a constant buffer the engine renamed outside the frame ring gets its bytes copied now, at the draw (a D3D11 driver's per-draw version)
        const auto copy_at_draw = [&](u64 address, u64 size) {
            const Stream::Slice slice = Stream::HostRing(size, alignment);
            std::memcpy(slice.data, reinterpret_cast<const void*>(address), size);
            push.buf_offsets[buffer_index] = 0;
            b.buffers.push_back({slice.buffer, slice.offset, size});
            b.guest.resize(b.buffers.size());
            b.guest.back() = {address, size};
            b.stageable.resize(b.buffers.size());
            b.stageable.back() = 0;
            ++t_dc_copies;
            if (CbCensusOn()) {
                if (const u64 epoch = Core::Vkb::GraphicsSubmits(); epoch != t_cbc_epoch) {
                    t_cbc_epoch = epoch, t_cbc_seen.clear();
                }
                u64 h = 0xcbf29ce484222325ull;
                for (u64 k = 0; k + 8 <= size; k += 8) {
                    h = (h ^ Rd<u64>(address, k)) * 0x100000001b3ull;
                }
                c_cbc_copies.fetch_add(1, std::memory_order_relaxed), c_cbc_bytes.fetch_add(size, std::memory_order_relaxed);
                const auto [it, fresh] = t_cbc_seen.try_emplace(address | size << 48, h);
                if (!fresh) {
                    if (it->second == h) {
                        c_cbc_same.fetch_add(1, std::memory_order_relaxed), c_cbc_same_bytes.fetch_add(size, std::memory_order_relaxed);
                    } else {
                        c_cbc_same_address_changed.fetch_add(1, std::memory_order_relaxed), it->second = h;
                    }
                }
            }
            if (StageCensusOn()) {
                StageMix(0, address), StageMix(0, size);
            }
            return true;
        };
        const auto bind_window = [&](u64 address, u64 size, const Core::Vkb::BufferWindow& w) {
            if (dumping) { // note 41
                dump_text += fmt::format("\n  buffer {} class {} kind {} slot {} type {}{}: guest {:#x} {} B, ours offset {} {}", i, static_cast<u32>(src.cls), src.kind, src.slot,
                                         static_cast<u32>(desc.buffer_type), desc.is_written ? " written" : "", address, size, w.offset, w.device ? "device" : "host");
                if (src.kind == 3 && size <= 1024) { // its bytes now (floats)
                    dump_text += "\n   ";
                    for (u64 k = 0; k + 4 <= size; k += 4) {
                        dump_text += fmt::format(" {:.4g}", *reinterpret_cast<const float*>(address + k));
                    }
                }
            }
            // note 41 (test, TLG_VKB_CB_AT_DRAW=1): a constant buffer read in place gets its bytes now, at the engine's draw call (D3D's meaning); the game writes some
            // of them (the scene slot matrices) straight into ring memory it reuses, so a copy made later, at the submission, can hold another draw's data
            static const bool cb_at_draw = [] { const char* e = std::getenv("TLG_VKB_CB_AT_DRAW"); return e != nullptr && e[0] == '1'; }();
            if (cb_at_draw && src.cls == Source::Class::Slot && src.kind == 3 && !w.device && !desc.is_written && size <= 65536) {
                const Stream::Slice slice = Stream::HostRing(size, alignment);
                std::memcpy(slice.data, reinterpret_cast<const void*>(address), size);
                push.buf_offsets[buffer_index] = 0;
                b.buffers.push_back({slice.buffer, slice.offset, size});
                b.guest.resize(b.buffers.size());
                b.guest.back() = {address, size};
                b.stageable.resize(b.buffers.size());
                b.stageable.back() = 0;
                c_cb_at_draw.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            (void)copy_at_draw;
            if (StageCensusOn()) {
                StageMix(0, address), StageMix(0, size), StageMix(0, reinterpret_cast<u64>(static_cast<VkBuffer>(w.buffer))), StageMix(0, w.offset);
            }
            const u64 aligned = w.offset & ~(alignment - 1);
            push.buf_offsets[buffer_index] = static_cast<u8>(w.offset - aligned);
            b.buffers.push_back({w.buffer, aligned, size + (w.offset - aligned)});
            b.guest.resize(b.buffers.size());
            b.guest.back() = {address, size};
            b.stageable.resize(b.buffers.size());
            b.stageable.back() = !w.device && !desc.is_written ? 1 : 0;
            uses.push_back(Stream::BufferUse(w.buffer, w.offset, size, desc.is_written));
            Core::Vkb::NoteBind(1, w, size, desc.is_written);
            return true;
        };
        // R2b (note 64): the window found later, by the recording thread; a placeholder binding now (filled in DrawJob::ResolveLate)
        const auto defer = [&](u64 object, u64 id, u64 memory, u64 bytes, u64 offset, u64 size) {
            late->push_back({.object = object, .id = id, .memory = memory, .bytes = bytes, .offset = offset, .size = size, .slot = static_cast<u16>(b.buffers.size()),
                             .push_index = static_cast<u8>(buffer_index), .what = LateBuffer::Stage, .written = desc.is_written});
            push.buf_offsets[buffer_index] = 0;
            b.buffers.push_back({});
            b.guest.resize(b.buffers.size());
            b.guest.back() = {object != 0 ? memory + offset : offset, size};
            b.stageable.resize(b.buffers.size());
            b.stageable.back() = 0;
            return true;
        };
        const auto bind_range = [&](u64 address, u64 size) {
            if (late != nullptr) {
                return defer(0, 0, 0, 0, address, size);
            }
            Core::Vkb::BufferWindow w;
            return Core::Vkb::FindBuffer(address, size, w) && bind_window(address, size, w);
        };
        bool ok = false;
        if (src.cls == Source::Class::Slot && src.kind != 4) {
            // R7 (note 56): by the engine object in the slot. A constant buffer's current data is where the engine renamed it (its buffer address word, the engine's own
            // pointer to the data; every constant buffer is renamed); a buffer view is its buffer from offset 0, the whole buffer (note 56 census: always so)
            const u64 object = SlotObject(ctx, st.index, src.kind, src.slot);
            if (object != 0 && src.kind == 3) {
                const u64 address = static_cast<u64>(Rd<u32>(object, 0x60)) | static_cast<u64>(Rd<u32>(object, 0x64) & 0xffu) << 32;
                const u64 size = Rd<u32>(object, 0x180);
                if (size > 65536 && Core::Vkb::ContractCensusOn()) { // api-contract.md C17 (debug): a constant buffer above the usual 64 KB uniform range
                    Core::Vkb::ContractNote("C17 constant buffer above 64 KB");
                }
                Core::Vkb::BufferWindow w;
                if (address != 0 && size != 0) {
                    if (Core::Vkb::ConstantInRing(address, size, w)) {
                        ok = bind_window(address, size, w);
                    } else if (Core::Vkb::CbInPlaceOn() && !desc.is_written && Core::Vkb::PoolConstantBuffer(object) &&
                               (w = Core::Vkb::SourceWindow(address, size)).buffer) { // note 98: a pool constant buffer, read in place (no copy)
                        Core::Vkb::NotePoolInPlace();
                        ok = bind_window(address, size, w);
                    } else if (!desc.is_written && size <= 65536 && !Core::Vkb::R7NoCopy()) {
                        Core::Vkb::NoteConstantCopied();
                        ok = copy_at_draw(address, size);
                    } else {
                        ok = bind_range(address, size);
                    }
                }
            } else if (object != 0 && src.kind == 1 && (Rd<u32>(object, 0x44) == 1 || Rd<u32>(object, 0x44) == 0xb)) {
                const u64 buffer = Rd<u64>(object, 0x30);
                const u64 bytes = buffer != 0 ? Rd<u32>(buffer, 0x180) : 0;
                u64 offset = 0, size = bytes; // api-contract.md V4 / review A6: the view's own range of its buffer
                Core::Vkb::ViewRange(object, true, buffer, bytes, offset, size);
                Core::Vkb::BufferWindow w;
                if (late != nullptr && bytes != 0) { // R2b: the frame ring now (arithmetic), any other buffer by the recording thread
                    const u64 memory = Rd<u64>(buffer, 0x30);
                    ok = Core::Vkb::BufferInRing(memory + offset, size, w) ? bind_window(memory + offset, size, w)
                                                                          : defer(buffer, Rd<u64>(buffer, 0x10), memory, bytes, offset, size);
                } else {
                    ok = bytes != 0 && Core::Vkb::BufferOfObject(buffer, offset, size, w) && bind_window(Rd<u64>(buffer, 0x30) + offset, size, w);
                }
            }
        } else if (src.cls == Source::Class::Slot) {
            c_skip_uav.fetch_add(1, std::memory_order_relaxed); // an unordered access view in a graphics stage: not bound yet
            return false;
        } else if (src.cls == Source::Class::ProgramConstant) {
            ok = src.size != 0 && bind_range(src.address, src.size);
        } else if (src.cls == Source::Class::Special) {
            if (desc.buffer_type == VkbShader::BufferType::Flatbuf) {
                // RESIDUAL (as for compute): the flat user data is rebuilt from the engine objects per draw
                Core::NativeDraw::BuildUserLayout(ctx, st.index, st.shader, st.index == 0 ? 0x28 : 0x3c, false, st.live);
                st.info->user_data = std::span<const u32>{st.live.sgpr.data(), st.live.sgpr.size()};
                st.info->RefreshFlatBuf();
                if (t_depth_pass && DepthCensusOn()) { // X0 census: the flat user data of the cascades' stages (size, same as the stage's previous draw)
                    u64 h = 0xcbf29ce484222325ull;
                    for (const u32 w : st.info->flattened_ud_buf) {
                        h = (h ^ w) * 0x100000001b3ull;
                    }
                    thread_local u64 prev[2]{};
                    const u32 si = st.index & 1;
                    c_dc_flat_n.fetch_add(1, std::memory_order_relaxed);
                    c_dc_flat_same.fetch_add(prev[si] == h ? 1 : 0, std::memory_order_relaxed);
                    c_dc_flat_dwords.fetch_add(st.info->flattened_ud_buf.size(), std::memory_order_relaxed);
                    prev[si] = h;
                    static std::atomic<u32> shown{0};
                    if (shown.fetch_add(1, std::memory_order_relaxed) < 6) {
                        std::string t = fmt::format("X0 flat user data: stage {} shader {:#x} {} dwords:", st.index, st.info->pgm_hash, st.info->flattened_ud_buf.size());
                        for (const u32 w : st.info->flattened_ud_buf) {
                            t += fmt::format(" {:08x}", w);
                        }
                        LOG_WARNING(Render_Vulkan, "{}", t);
                    }
                }
                const u64 bytes = std::max<u64>(st.info->flattened_ud_buf.size() * sizeof(u32), 16);
                const Stream::Slice slice = Stream::HostRing(bytes, alignment);
                std::memset(slice.data, 0, bytes);
                std::memcpy(slice.data, st.info->flattened_ud_buf.data(), st.info->flattened_ud_buf.size() * sizeof(u32));
                b.buffers.push_back({slice.buffer, slice.offset, bytes});
                c_flat_rebuilt.fetch_add(1, std::memory_order_relaxed);
                ok = true;
            } else if (desc.buffer_type == VkbShader::BufferType::ClipPlanes || desc.buffer_type == VkbShader::BufferType::GdsBuffer) {
                b.buffers.push_back({desc.buffer_type == VkbShader::BufferType::GdsBuffer ? Stream::GdsBuffer() : Stream::NullBuffer(), 0, VK_WHOLE_SIZE});
                ok = true;
            } else {
                c_skip_special.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
        }
        if (!ok) {
            c_skip_buffer.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        ++buffer_index;
    }
    const u32 first_image = static_cast<u32>(b.images.size());
    for (u32 i = 0; i < st.images.size(); ++i) {
        const Source& src = st.images[i];
        const u64 object = src.cls == Source::Class::Slot && src.kind == 1 ? SlotObject(ctx, st.index, 1, src.slot) : 0;
        if (late_img != nullptr) { // R2b part 2b: the view id now, our view by the resolver
            if (object == 0) {
                c_skip_image.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            late_img->push_back({.view_id = Rd<u64>(object, 0x10), .type = st.image_types[i], .slot = static_cast<u16>(b.images.size()), .written = info.images[i].is_written});
            b.images.push_back({{}, {}, vk::ImageLayout::eGeneral});
            continue;
        }
        vk::ImageView view{};
        vk::Image image{};
        if (object == 0 || !Core::Vkb::ViewForCompute(Rd<u64>(object, 0x10), st.image_types[i], cmd, view, &image)) {
            c_skip_image.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (StageCensusOn()) {
            StageMix(1, object), StageMix(1, reinterpret_cast<u64>(static_cast<VkImageView>(view)));
        }
        b.images.push_back({{}, view, vk::ImageLayout::eGeneral});
        uses.push_back(Stream::ImageUse(image, info.images[i].is_written));
        if (dumping) {
            dump_text += fmt::format("\n  image {} slot {} as {}{}: {}", i, src.slot, vk::to_string(st.image_types[i]), info.images[i].is_written ? " written" : "",
                                     Core::Vkb::DescribeView(Rd<u64>(object, 0x10)));
        }
    }
    const u32 first_sampler = static_cast<u32>(b.samplers.size());
    for (u32 i = 0; i < st.samplers.size(); ++i) {
        const Source& src = st.samplers[i];
        const u64 object = src.cls == Source::Class::Slot && src.kind == 2 ? SlotObject(ctx, st.index, 2, src.slot) : 0;
        const bool is_depth = i < info.samplers.size() && info.samplers[i].is_depth;
        const Core::Vkb::Aa::SamplerLook own = Core::Vkb::Aa::ForSampler(look, is_depth);
        const vk::Sampler sampler = st.constant_samplers[i] ? ConstantSamplerFor(st, i, is_depth, own)
                                    : object != 0              ? Core::Vkb::SamplerFor(object, is_depth, own.bias_q, own.aniso)
                                                               : vk::Sampler{};
        if (!sampler) {
            c_skip_sampler.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (StageCensusOn()) {
            StageMix(2, reinterpret_cast<u64>(static_cast<VkSampler>(sampler)));
        }
        b.samplers.push_back({sampler, {}, vk::ImageLayout::eUndefined});
        if (dumping) {
            dump_text += fmt::format("\n  sampler {} {}", i, st.constant_samplers[i] ? std::string("constant (from the shader)") : Core::Vkb::DescribeSampler(object));
        }
    }
    if (dumping) {
        DrawDumpLine(fmt::format("bind: stage {} hash {:#x}, {} images, {} samplers:{}", st.index, st.info->pgm_hash, st.images.size(), st.samplers.size(), dump_text));
    }
    ranges.push_back({first_buffer, static_cast<u32>(b.buffers.size()) - first_buffer});
    ranges.push_back({first_image, static_cast<u32>(b.images.size()) - first_image});
    ranges.push_back({first_sampler, static_cast<u32>(b.samplers.size()) - first_sampler});
    if (StageCensusOn() && st.index < 2) {
        const u32 s = st.index;
        bool all = true;
        std::atomic<u64>* same[3] = {c_stage_buffers_same, c_stage_images_same, c_stage_samplers_same};
        for (u32 k = 0; k < 3; ++k) {
            const bool eq = t_stage_hash[k] == t_stage_prev[s][k];
            same[k][s].fetch_add(eq ? 1 : 0, std::memory_order_relaxed);
            all = all && eq;
            t_stage_prev[s][k] = t_stage_hash[k];
        }
        (all ? c_stage_same : c_stage_changed)[s].fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

/// R3c: per-thread chunks of the host ring for rows and copied constants (one ring lock per chunk, not two per draw). A chunk is taken in stream order like any slice;
/// what it holds is used by work enqueued after the chunk was taken.
Stream::Slice RowSlice(u64 size) {
    constexpr u64 Chunk = 256 * 1024;
    thread_local Stream::Slice chunk{};
    thread_local u64 used = Chunk;
    thread_local u64 generation = ~0ull;
    size = (size + 255) & ~u64{255}; // (256: the storage buffer offset alignment of every device)
    if (size > Chunk) {
        return Stream::HostRing(size, 256);
    }
    if (used + size > Chunk || generation != Stream::HostRingGeneration()) {
        chunk = Stream::HostRing(Chunk, 256);
        used = 0;
        generation = Stream::HostRingGeneration();
    }
    Stream::Slice out{chunk.buffer, chunk.offset + used, static_cast<u8*>(chunk.data) + used};
    used += size;
    return out;
}

/// R3: a record-mode stage's entries of the draw's row (slots, a read-only guest buffer's byte offset and size, image and sampler slots), in the order the recompiler's
/// indexed mode assigned them (buffers, images, samplers); the hazards into `uses`. False: the draw cannot go this way (it then takes the plain path).
/// R7 step C: the row entries of one batch's resources, made once per batch (its first draw pushed their hazards into the batch); empty when the batch changes
struct RowMemoEntry {
    u64 key{}, serial{};
    u32 a{}, b{}, c{};
};
/// R7 step D: the lean path's per-thread state, one plain block (fixed-size open-addressing tables)
struct PipeSlotLean {
    u64 hash{};
    std::array<u64, 24> words{};
    void* v{};
    const VkbVulkan::FixedState* fixed{};
};
struct LeanBlock {
    std::array<RowMemoEntry, 4096> memo{};
    std::array<PipeSlotLean, 4096> pipes{};
};
thread_local LeanBlock* t_lean = nullptr;
inline LeanBlock& Lean() {
    if (t_lean == nullptr) {
        t_lean = new LeanBlock;
    }
    return *t_lean;
}
thread_local u64 t_row_memo_serial = ~0ull;
inline RowMemoEntry* MemoFind(u64 key, u64 serial) {
    auto& memo = Lean().memo;
    const u32 at = static_cast<u32>((key * 0x9e3779b97f4a7c15ull) >> 52) & 4095;
    for (u32 i = 0; i < 4; ++i) {
        RowMemoEntry& e = memo[(at + i) & 4095];
        if (e.key == key && e.serial == serial) {
            return &e;
        }
    }
    return nullptr;
}
inline void MemoPut(u64 key, u64 serial, u32 a, u32 b, u32 c) {
    auto& memo = Lean().memo;
    const u32 at = static_cast<u32>((key * 0x9e3779b97f4a7c15ull) >> 52) & 4095;
    for (u32 i = 0; i < 4; ++i) {
        RowMemoEntry& e = memo[(at + i) & 4095];
        if (e.serial != serial || e.key == key) {
            e = {key, serial, a, b, c};
            return;
        }
    }
    memo[at] = {key, serial, a, b, c};
}
std::atomic<u64> c_row_memo_hits{0}, c_row_memo_misses{0};
thread_local Core::Vkb::Aa::SamplerLook t_row_look{}; ///< issue 42 (merge): the draw's sampler look, set by the caller of BuildRow / FastRow
bool BuildRow(u64 ctx, const Stage& st, u32* row, boost::container::small_vector<Stream::Use, 64>& uses, u64 memo_serial = 0) {
    using BT = VkbShader::BufferType;
    t_row_memo_serial = memo_serial;
    const auto& info = *st.info;
    const u64 alignment = instance->StorageMinAlignment();
    u32 at = st.row_start;
    for (u32 i = 0; i < st.buffers.size(); ++i) {
        const Source& src = st.buffers[i];
        const auto& desc = info.buffers[i];
        if (desc.buffer_type == BT::GdsBuffer) {
            row[at++] = Core::Vkb::Indexed::BufferSlot(Stream::GdsBuffer());
            continue;
        }
        if (desc.buffer_type == BT::ClipPlanes) {
            row[at++] = Core::Vkb::Indexed::BufferSlot(Stream::NullBuffer());
            continue;
        }
        if (desc.buffer_type != BT::Guest) {
            return false;
        }
        Core::Vkb::BufferWindow w;
        u64 address = 0, size = 0;
        bool ok = false;
        u64 memo_key = 0;
        if (memo_serial != 0) { // the identity of what this entry resolves: the slot object and the words the resolution reads
            if (src.cls == Source::Class::Slot) {
                const u64 object = SlotObject(ctx, st.index, src.kind, src.slot);
                const u64 word = object == 0 ? 0 : src.kind == 3 ? Rd<u64>(object, 0x60) ^ (u64{Rd<u32>(object, 0x180)} << 40)
                                                                   : (Rd<u64>(object, 0x30) != 0 ? Rd<u64>(Rd<u64>(object, 0x30), 0x30) : 0) ^ (u64{Rd<u32>(object, 0x44)} << 56);
                memo_key = (object * 0x9e3779b97f4a7c15ull) ^ word ^ (u64{src.kind} << 60) ^ 1;
            } else {
                memo_key = (src.address * 0x9e3779b97f4a7c15ull) ^ src.size ^ (u64{7} << 60);
            }
            if (const RowMemoEntry* e = MemoFind(memo_key, memo_serial)) {
                c_row_memo_hits.fetch_add(1, std::memory_order_relaxed);
                row[at++] = e->a;
                row[at++] = e->b;
                row[at++] = e->c;
                continue;
            }
            c_row_memo_misses.fetch_add(1, std::memory_order_relaxed);
        }
        if (src.cls == Source::Class::Slot && src.kind == 3) {
            const u64 object = SlotObject(ctx, st.index, 3, src.slot);
            if (object != 0) {
                address = static_cast<u64>(Rd<u32>(object, 0x60)) | static_cast<u64>(Rd<u32>(object, 0x64) & 0xffu) << 32;
                size = Rd<u32>(object, 0x180);
                if (address != 0 && size != 0) {
                    if (Core::Vkb::ConstantInRing(address, size, w)) {
                        ok = true;
                    } else if (Core::Vkb::R7NoCopy()) { // bound / bisect (TLG_VKB_R7_NO_CB_COPY=1): in place, not copied
                        ok = Core::Vkb::FindBuffer(address, size, w);
                    } else if (size <= 65536) { // R7: copied at the draw (a D3D11 driver's per-draw version)
                        const Stream::Slice slice = RowSlice(size);
                        std::memcpy(slice.data, reinterpret_cast<const void*>(address), size);
                        w.buffer = slice.buffer, w.offset = slice.offset, w.device = false;
                        ok = true;
                        Core::Vkb::NoteConstantCopied();
                    }
                }
            }
        } else if (src.cls == Source::Class::Slot && src.kind == 1) {
            const u64 object = SlotObject(ctx, st.index, 1, src.slot);
            if (object != 0 && (Rd<u32>(object, 0x44) == 1 || Rd<u32>(object, 0x44) == 0xb)) {
                const u64 buffer = Rd<u64>(object, 0x30);
                const u64 bytes = buffer != 0 ? Rd<u32>(buffer, 0x180) : 0;
                u64 offset = 0;
                size = bytes;
                Core::Vkb::ViewRange(object, true, buffer, bytes, offset, size);
                const u64 memory = buffer != 0 ? Rd<u64>(buffer, 0x30) : 0;
                address = memory + offset;
                ok = bytes != 0 && (Core::Vkb::BufferInRing(address, size, w) || (Core::Vkb::BufferOfObject(buffer, offset, size, w) && w.device));
            }
        } else if (src.cls == Source::Class::ProgramConstant && src.size != 0) {
            address = src.address, size = src.size;
            ok = Core::Vkb::FindBuffer(address, size, w) && w.device;
        }
        if (!ok || !w.buffer || (w.offset & 3) != 0) {
            return false;
        }
        row[at++] = Core::Vkb::Indexed::BufferSlot(w.buffer);
        row[at++] = static_cast<u32>(w.offset);
        row[at++] = static_cast<u32>(size);
        if (memo_key != 0) {
            MemoPut(memo_key, memo_serial, row[at - 3], row[at - 2], row[at - 1]);
        }
        uses.push_back(Stream::BufferUse(w.buffer, w.offset, size, false));
        Core::Vkb::NoteBind(1, w, size, false);
    }
    for (u32 i = 0; i < st.images.size(); ++i) {
        const Source& src = st.images[i];
        const u64 object = src.cls == Source::Class::Slot && src.kind == 1 ? SlotObject(ctx, st.index, 1, src.slot) : 0;
        const u64 image_key = memo_serial != 0 && object != 0 ? (object * 0x9e3779b97f4a7c15ull) ^ (u64{static_cast<u32>(st.image_types[i])} << 48) ^ (u64{info.images[i].array_key} << 56) ^ 2 : 0;
        if (image_key != 0) {
            if (const RowMemoEntry* e = MemoFind(image_key, memo_serial)) {
                c_row_memo_hits.fetch_add(1, std::memory_order_relaxed);
                row[at++] = e->a;
                continue;
            }
        }
        vk::ImageView view{};
        vk::Image image{};
        if (object == 0 || !Core::Vkb::ViewForCompute(Rd<u64>(object, 0x10), st.image_types[i], vk::CommandBuffer{}, view, &image)) {
            return false;
        }
        row[at++] = Core::Vkb::Indexed::ImageSlot(info.images[i].array_key, view);
        if (image_key != 0) {
            MemoPut(image_key, memo_serial, row[at - 1], 0, 0);
        }
        uses.push_back(Stream::ImageUse(image, info.images[i].is_written));
    }
    for (u32 i = 0; i < st.samplers.size(); ++i) {
        const Source& src = st.samplers[i];
        const u64 object = src.cls == Source::Class::Slot && src.kind == 2 ? SlotObject(ctx, st.index, 2, src.slot) : 0;
        const bool is_depth = i < info.samplers.size() && info.samplers[i].is_depth;
        const Core::Vkb::Aa::SamplerLook own = Core::Vkb::Aa::ForSampler(t_row_look, is_depth);
        const vk::Sampler sampler = st.constant_samplers[i] ? ConstantSamplerFor(st, i, is_depth, own) : object != 0 ? Core::Vkb::SamplerFor(object, is_depth, own.bias_q, own.aniso) : vk::Sampler{};
        if (!sampler) {
            return false;
        }
        row[at++] = Core::Vkb::Indexed::SamplerSlot(sampler);
    }
    return true;
}

/// R7 step C: a cached record-mode draw goes straight into the open batch of its pass (the same depth and colour targets): its row (memo per batch), its index buffer,
/// its push constants; no draw job. False: the generic path (first draw of a batch, other targets, AA on, a buffer the lean path does not take).
std::atomic<u64> c_lean{0}, c_lean_refused{0};
std::atomic<u64> c_lean_ns{0}, c_lean_n{0}, c_lean_words_ns{0}, c_lean_row_ns{0}; ///< TLG_VKB_LEAN_TIMES=1
bool LeanAppend(u64 ctx, Variant& v, const VkbVulkan::FixedState& fixed, u32 kind, u64 a1, u64 a2, u64 a3);

/// note 109 census (debug, TLG_VKB_DEPTH_CENSUS=1, inside the shadow cascade passes the Z pass hook marks): does each stage read the same engine objects as at the
/// previous draw (the same shader, the same slot objects, a constant buffer's same rename address and size, a buffer view's same buffer memory, the same image and sampler
/// objects); does the whole draw (variant, fixed state, both stages, vertex elements, index view, targets, instances); its constant buffer copies; and the time of Draw and of
/// its two BindStage calls. What a capture that keeps an unchanged stage's bindings would leave out.
bool DepthCensusOn() { static const bool on = [] { const char* e = std::getenv("TLG_VKB_DEPTH_CENSUS"); return e != nullptr && e[0] == '1'; }(); return on; }
std::atomic<u64> c_dc_draws{0}, c_dc_stage_n[2]{}, c_dc_stage_same[2]{}, c_dc_draw_same{0}, c_dc_copies{0}, c_dc_draw_ns{0}, c_dc_bind_ns{0}, c_dc_variant_same{0};
std::atomic<u64> c_dc_same_but_cb[2]{}; ///< stages equal except a constant buffer's rename address
thread_local u64 t_dc_prev_stage[2]{}, t_dc_prev_stage_nocb[2]{}, t_dc_prev_draw{0};
thread_local const void* t_dc_prev_variant = nullptr;
inline u64 DcMix(u64 h, u64 v) { return (h ^ v) * 0x100000001b3ull + 0x9E3779B97F4A7C15ull; }
std::pair<u64, u64> DcStageKey(u64 ctx, const Stage& st) { // (with the constant buffers' addresses, without)
    u64 h = DcMix(0xcbf29ce484222325ull, reinterpret_cast<u64>(st.info.get())), n = h;
    for (const Source& src : st.buffers) {
        if (src.cls == Source::Class::Slot && src.kind != 4) {
            const u64 object = SlotObject(ctx, st.index, src.kind, src.slot);
            h = DcMix(h, object), n = DcMix(n, object);
            if (object != 0 && src.kind == 3) {
                h = DcMix(DcMix(h, Rd<u64>(object, 0x60)), Rd<u32>(object, 0x180));
            } else if (object != 0 && src.kind == 1) {
                const u64 buffer = Rd<u64>(object, 0x30);
                const u64 memory = buffer != 0 ? Rd<u64>(buffer, 0x30) : 0;
                h = DcMix(DcMix(h, buffer), memory), n = DcMix(DcMix(n, buffer), memory);
            }
        } else {
            h = DcMix(DcMix(h, src.address), src.size), n = DcMix(DcMix(n, src.address), src.size);
        }
    }
    for (const Source& src : st.images) {
        const u64 object = src.cls == Source::Class::Slot && src.kind == 1 ? SlotObject(ctx, st.index, 1, src.slot) : 0;
        h = DcMix(h, object), n = DcMix(n, object);
    }
    for (const Source& src : st.samplers) {
        const u64 object = src.cls == Source::Class::Slot && src.kind == 2 ? SlotObject(ctx, st.index, 2, src.slot) : 0;
        h = DcMix(h, object), n = DcMix(n, object);
    }
    return {h, n};
}
/// X0 shape census (debug, with TLG_VKB_DEPTH_CENSUS): per pipeline variant of the cascades, what its stages read (buffer sources by class / kind, images, samplers,
/// vertex attributes) and how many draws use it: what a record-mode shader must fetch from a draw record.
std::mutex dc_shape_mutex;
std::unordered_map<const void*, std::pair<std::string, u64>> dc_shapes;
std::string DcShape(const Variant& v, const VkbVulkan::FixedState* fixed, u32 kind) {
    const auto stage = [](const Stage& st) {
        std::string t = fmt::format("{:#x} bufs[", st.info ? st.info->pgm_hash : 0ull);
        for (const Source& src : st.buffers) {
            t += src.cls == Source::Class::Slot ? fmt::format(" slot k{} s{}", src.kind, src.slot)
                 : src.cls == Source::Class::ProgramConstant ? fmt::format(" progconst {}B", src.size)
                                                             : std::string(" special");
        }
        t += fmt::format(" ] images {} samplers {}", st.images.size(), st.samplers.size());
        for (u32 i = 0; i < st.samplers.size(); ++i) {
            if (st.constant_samplers[i]) {
                t += " (const sampler)";
            }
        }
        return t;
    };
    std::string text = fmt::format("kind {} vs {} | attrs {}", kind, stage(v.vs), v.attributes.size());
    for (const Source& a : v.attributes) {
        text += fmt::format(" a{}", a.cls == Source::Class::Slot ? static_cast<int>(a.slot) : -1);
    }
    text += v.has_ps ? " | ps " + stage(v.ps) : std::string(" | no ps");
    text += fmt::format(" | colors {} depth {}", v.key.num_color_attachments, fixed != nullptr && fixed->depth_write ? "write" : "test-only");
    return text;
}
void DcCensus(u64 ctx, const Variant& v, const VkbVulkan::FixedState* fixed, u32 kind) {
    c_dc_draws.fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lock{dc_shape_mutex};
        auto [it, fresh] = dc_shapes.try_emplace(&v);
        if (fresh) {
            it->second.first = DcShape(v, fixed, kind);
        }
        ++it->second.second;
    }
    u64 draw = DcMix(DcMix(DcMix(0xcbf29ce484222325ull, reinterpret_cast<u64>(&v)), reinterpret_cast<u64>(fixed)), kind);
    c_dc_variant_same.fetch_add(t_dc_prev_variant == &v ? 1 : 0, std::memory_order_relaxed);
    t_dc_prev_variant = &v;
    for (u32 k = 0; k < 2; ++k) {
        if (k == 1 && !v.has_ps) {
            continue;
        }
        const Stage& st = k == 0 ? v.vs : v.ps;
        const auto [key, nocb] = DcStageKey(ctx, st);
        c_dc_stage_n[k].fetch_add(1, std::memory_order_relaxed);
        c_dc_stage_same[k].fetch_add(key == t_dc_prev_stage[k] ? 1 : 0, std::memory_order_relaxed);
        c_dc_same_but_cb[k].fetch_add(key != t_dc_prev_stage[k] && nocb == t_dc_prev_stage_nocb[k] ? 1 : 0, std::memory_order_relaxed);
        t_dc_prev_stage[k] = key, t_dc_prev_stage_nocb[k] = nocb;
        draw = DcMix(draw, key);
    }
    for (u32 i = 0; i < v.attributes.size(); ++i) {
        Core::NativeDraw::VertexElement e;
        if (v.attributes[i].cls == Source::Class::Slot && Core::NativeDraw::VertexElementOf(ctx, v.attributes[i].slot, e)) {
            draw = DcMix(DcMix(DcMix(DcMix(draw, e.buffer), e.start), e.stride), e.bytes);
        }
    }
    const u64 index_view = Rd<u64>(ctx, 0x58);
    draw = DcMix(DcMix(DcMix(draw, index_view), index_view != 0 ? Rd<u64>(index_view, 0x30) : 0), Rd<u32>(ctx, 0x64));
    const auto targets = TargetViews(ctx);
    for (u32 cb = 0; cb < 8; ++cb) {
        draw = DcMix(draw, targets[cb]);
    }
    draw = DcMix(DcMix(draw, Rd<u64>(ctx, CtxDepthStencilView)), Rd<u32>(ctx, 0x4724));
    c_dc_draw_same.fetch_add(draw == t_dc_prev_draw ? 1 : 0, std::memory_order_relaxed);
    t_dc_prev_draw = draw;
}

/// Binds and records one draw of a compiled variant into the frame stream (mutex held)
/// note 45 (R2 design): Record's phases on the engine thread (0 views + attachments, 1 stage bindings, 2 vertex + index, 3 the rest + the job hand-over) and the job on the recorder
std::atomic<u64> c_phase_ns[4]{}, c_job_ns{0}, c_job_n{0}, c_fixed_ns{0}, c_variant_find_ns{0};
/// issue 82 point 2 (TLG_VKB_TARGET_MEMO, default on, =0 off): most draws use the same targets as the draw before them on the same thread. Each thread keeps the
/// attachments it found for its last draw and reuses them while the draw's target set (view ids, formats, slots in use) is the same and the view epoch has not
/// moved (a view or an image went, note 46): then neither the first-use view check nor the attachment lookups run again, and the shared objects lock is not taken
/// for them. Not used with the late images (R2b part 2b, resolved on the recorder) or the contract census (it notes every draw's targets).
bool TargetMemoOn() { static const bool on = [] { const char* e = std::getenv("TLG_VKB_TARGET_MEMO"); return e == nullptr || e[0] != '0'; }(); return on; }
std::atomic<u64> c_target_memo_hits{0}, c_target_memo_misses{0};
struct TargetSet {
    u64 epoch{~0ull};
    std::array<u64, 8> ids{};            ///< engine view id per colour slot (0: no view)
    std::array<vk::Format, 8> formats{}; ///< the pipeline's format per colour slot
    u64 depth_id{};                      ///< engine view id of the depth target (0: none)
    u32 color_count{}, mrt_mask{};
    bool depth{}, stencil{};
    bool operator==(const TargetSet&) const = default;
};
struct TargetMemo {
    TargetSet key{}; ///< epoch ~0: nothing kept
    std::array<vk::ImageView, 8> colors{};
    vk::ImageView depth{};
    bool depth_has_stencil{};
    u32 width{}, height{}, layers{};
    boost::container::small_vector<Stream::Use, 9> uses;
};
bool TimesOn() { static const bool on = [] { const char* e = std::getenv("TLG_VKB_TIMES"); return e != nullptr && e[0] == '1'; }(); return on || (t_depth_pass && DepthCensusOn()); } // note 109 census: the cascade draws timed alone ///< note 45: these timers only with TLG_VKB_TIMES=1
bool BatchOn();
bool JoinBatch(DrawJob& job);
thread_local u64 t_join_dsv = 0, t_join_rtv0 = 0; ///< R7 step C: the engine views of the draw joining (set by Record)
thread_local u64 t_join_rtvs = 0; ///< G2: all colour targets of the draw joining
/// G2 (TLG_VKB_RECORD_ALL, with TLG_VKB_RECORD): record mode, batches and the lean path for the indexed draws of every pass, not only the shadow cascades.
/// OFF by default since 2026-11-23 (tabnul issue 130: the coloured glass of the eye sign in s07 AL_a01_SkyAthletics_01 is lost when its draw joins a batch of colour
/// draws; right with TLG_VKB_RECORD_BATCH=0). TLG_VKB_RECORD_ALL=1 turns it on until the batch keeps what that draw needs. TLG_VKB_RECORD (the cascades) stays on.
bool RecordAll() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_RECORD_ALL"); return e != nullptr && e[0] == '1'; }();
    return on;
}
inline u64 RtvHash(u64 ctx) {
    u64 h = 0xcbf29ce484222325ull ^ Rd<u32>(ctx, CtxRenderTargetCount);
    for (u32 i = 0; i < 8; ++i) {
        h = (h ^ Rd<u64>(ctx, CtxRenderTargets + i * 8)) * 0x100000001b3ull;
    }
    return h;
}
/// F1c: the draw's AA sub-pixel offset (the scene geometry is jittered), decided in game order; the job path and the lean batch path (issue 134) both use it
bool AaDrawJitter(u64 ctx, const Variant& v, const VkbVulkan::FixedState& fixed, u32 kind, u64 a1, u64 a2, u64 a3, u32 instances, std::array<float, 2>& jitter) {
    const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView);
    // the depth texture's id (+0x30 resource, +0x10 id): the motion pass and the G-buffer bind the same depth through different view objects
    const u64 dsv_resource = dsv != 0 && IsHeapPointer(dsv) ? Rd<u64>(dsv, 0x30) : 0;
    const u64 dsv_id = dsv_resource != 0 && IsHeapPointer(dsv_resource) ? Rd<u64>(dsv_resource, 0x10) : 0;
    const u64 ps = v.has_ps && v.ps.info ? v.ps.info->pgm_hash : 0ull;
    if (ps == Core::Vkb::Aa::MotionPassPs) {
        Core::Vkb::Aa::NoteSceneDepth(dsv_id);
    }
    const bool geometry = (kind == 2) || a1 > 6 || instances > 1;
    // issue 54: a mesh is its index data, as the normal renderer's rule (vk_rasterizer.cpp: index address and count): the address of the first index and
    // the index count (an indirect draw: the index data and its arguments), whatever shader draws it, so a second pass of a surface with another vertex shader
    // matches its shifted pre-pass. Not indexed: no mesh. TLG_VKB_AA_MESH_KEY=0: the old key (counts and the vertex shader; it never matched, issue 54)
    u64 mesh = 0;
    if (!Core::Vkb::Aa::MeshKeySwitch()) {
        mesh = Mix(Mix(Mix(a1, a2), a3), v.vs.info ? v.vs.info->pgm_hash : 0ull);
    } else if (const u64 index_view = Rd<u64>(ctx, 0x58); (kind == 0 || kind == 2) && index_view != 0) {
        const u64 index_address = static_cast<u64>(Rd<u32>(index_view, 0x64) & 0xff) << 32 | (Rd<u32>(index_view, 0x60) & 0xfffffffeu);
        const u64 index_bytes = Rd<u32>(ctx, 0x64) == 1 ? 4 : 2;
        mesh = index_address == 0 ? 0 : kind == 0 ? Mix(index_address + a2 * index_bytes, a1) : Mix(Mix(index_address, a1), a2);
    }
    // does the pixel shader read a scene depth (decals, lights): its used texture slots' resources against the scene depth ids
    bool samples_depth = false;
    const u64 ps_object = Rd<u64>(ctx, CtxPixelShader);
    if (Core::Vkb::Aa::IsSceneDepth(dsv_id) && ps_object != 0 && IsHeapPointer(ps_object)) {
        const u64 masks[2] = {Rd<u64>(ps_object, 0x30 + 0xc0), Rd<u64>(ps_object, 0x30 + 0xc8)};
        for (u32 half = 0; half < 2 && !samples_depth; ++half) {
            for (u64 bits = masks[half]; bits != 0 && !samples_depth; bits &= bits - 1) {
                const u32 s = half * 64 + static_cast<u32>(std::countr_zero(bits));
                const u64 view = Rd<u64>(ctx, SrvTable(1) + s * 8);
                const u64 resource = view != 0 && IsHeapPointer(view) ? Rd<u64>(view, 0x30) : 0;
                samples_depth = resource != 0 && IsHeapPointer(resource) && Core::Vkb::Aa::IsSceneDepth(Rd<u64>(resource, 0x10));
            }
        }
    }
    return Core::Vkb::Aa::DrawJitter(dsv_id, fixed.depth_write, geometry, samples_depth, mesh, jitter);
}

void Record(u64 ctx, Variant& v, const VkbVulkan::FixedState& fixed, u32 kind, u64 a1, u64 a2, u64 a3) {
    if (Core::Vkb::FillReadersOn() || Core::Vkb::R3dTraceSize() != 0) {
        Core::Vkb::CensusTag((v.has_ps && v.ps.info ? v.ps.info->pgm_hash : 0x10000000000ull) | (v.vs.info ? u64{v.vs.info->pgm_hash} << 32 : 0)); // note 49 census: a draw, its pixel shader (debug R3d trace: vertex shader above)
    }
    // note 41, test only: TLG_VKB_TEST_SKIP_DRAW_PS=hex,hex,... skips the draws of those pixel shaders (visual bisect; the picture is wrong)
    static const std::vector<u64> skip_ps = [] {
        std::vector<u64> out;
        if (const char* e = std::getenv("TLG_VKB_TEST_SKIP_DRAW_PS")) {
            for (const char* p = e; *p != 0;) {
                char* end = nullptr;
                const u64 h = std::strtoull(p, &end, 16);
                if (end == p) {
                    break;
                }
                out.push_back(h);
                p = *end == ',' ? end + 1 : end;
            }
        }
        return out;
    }();
    if (!skip_ps.empty() && v.has_ps && v.ps.info && std::find(skip_ps.begin(), skip_ps.end(), v.ps.info->pgm_hash) != skip_ps.end()) {
        return;
    }
    const u64 t0 = TimesOn() ? NowNs() : 0; // note 60: no clock reads per draw with the timers off
    // views the engine made through a path its view-creation hooks do not see are noted at their first use (their description is kept at + 0x40)
    // R2b part 2b: a view id seen with a format is never checked again on this thread (ids are never reused); the check itself stays here (it reads the engine view)
    thread_local std::array<u64, 1024> known_views{};
    // issue 82 point 2: this draw's target set against the one of the last draw on this thread (the epoch is read before any lookup: a change during them marks it stale)
    thread_local TargetMemo t_targets;
    const bool target_memo = TargetMemoOn() && !LateImagesOn() && !Core::Vkb::ContractCensusOn();
    TargetSet target_key{};
    bool targets_reused = false, targets_keep = target_memo;
    if (target_memo) {
        target_key.epoch = Core::Vkb::ViewEpoch();
        const auto targets = TargetViews(ctx);
        target_key.color_count = std::min<u32>(v.key.num_color_attachments, 8);
        target_key.mrt_mask = v.key.mrt_mask;
        for (u32 cb = 0; cb < target_key.color_count; ++cb) {
            target_key.ids[cb] = targets[cb] != 0 ? Rd<u64>(targets[cb], 0x10) : 0;
            target_key.formats[cb] = v.color_formats[cb];
        }
        const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView);
        target_key.depth_id = dsv != 0 ? Rd<u64>(dsv, 0x10) : 0;
        target_key.depth = v.key.z_format != Core::NativeTarget::DepthTarget::ZFormat::Invalid;
        target_key.stencil = v.key.stencil_format != Core::NativeTarget::DepthTarget::StencilFormat::Invalid;
        targets_reused = target_key == t_targets.key;
        (targets_reused ? c_target_memo_hits : c_target_memo_misses).fetch_add(1, std::memory_order_relaxed);
    }
    const auto view_known = [](u64 id) {
        if (!LateImagesOn()) {
            return false;
        }
        u64& slot = known_views[(id * 0x9e3779b97f4a7c15ull) >> 54];
        if (slot == id) {
            return true;
        }
        if (Core::Vkb::ViewFormat(id) == vk::Format::eUndefined) {
            return false;
        }
        slot = id;
        return true;
    };
    if (!targets_reused) { // issue 82: a reused target set had every view known at the last draw, and none has gone since (the epoch)
        const auto targets = TargetViews(ctx);
        for (u32 cb = 0; cb < v.key.num_color_attachments; ++cb) {
            if (targets[cb] != 0 && !view_known(Rd<u64>(targets[cb], 0x10)) && Core::Vkb::ViewFormat(Rd<u64>(targets[cb], 0x10)) == vk::Format::eUndefined) {
                Core::NativeDevice::NoteView(Core::NativeDevice::Kind::RenderTargetView, targets[cb], targets[cb] + 0x40);
                c_late_views.fetch_add(1, std::memory_order_relaxed);
                targets_keep = false; // noted just now: the next draw checks again
            }
        }
        if (const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView); dsv != 0 && !view_known(Rd<u64>(dsv, 0x10)) && Core::Vkb::ViewFormat(Rd<u64>(dsv, 0x10)) == vk::Format::eUndefined) {
            Core::NativeDevice::NoteView(Core::NativeDevice::Kind::DepthStencilView, dsv, dsv + 0x40);
            c_late_views.fetch_add(1, std::memory_order_relaxed);
            targets_keep = false;
        }
    }
    u64 ph = TimesOn() ? NowNs() : 0;
    const auto phase = [&](u32 i) { if (!TimesOn()) { return; } const u64 n = NowNs(); c_phase_ns[i].fetch_add(n - ph, std::memory_order_relaxed); ph = n; };
    const vk::CommandBuffer cmd{}; // V4g: resolved here; its commands are recorded by the recording thread (the job below)
    // the attachments, in the pipeline's formats (their images move to the general layout here, before the pass)
    std::unique_ptr<DrawJob> job{new DrawJob}; // A4 (note 48): the draw's record, filled in place
    Stream::Pass& pass = job->pass;
    auto& attachment_uses = job->attachment_uses; // V4d: written by the pass
    auto& uses = job->uses;                       // V4d: read by this draw
    pass.color_count = v.key.num_color_attachments;
    u32 width = ~0u, height = ~0u, layers = ~0u;
    const auto views = TargetViews(ctx);
    const bool late_images = LateImagesOn(); // R2b part 2b: the targets by view id, resolved later
    if (late_images) {
        for (u32 cb = 0; cb < pass.color_count; ++cb) {
            if ((v.key.mrt_mask & (1u << cb)) == 0 || v.color_formats[cb] == vk::Format::eUndefined || views[cb] == 0) {
                continue;
            }
            job->late_targets.push_back({.view_id = Rd<u64>(views[cb], 0x10), .format = v.color_formats[cb], .cb = static_cast<u8>(cb)});
        }
        if (v.key.z_format != Core::NativeTarget::DepthTarget::ZFormat::Invalid) {
            const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView);
            if (dsv == 0) {
                c_skip_depth.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            job->late_targets.push_back({.view_id = Rd<u64>(dsv, 0x10), .format = vk::Format::eUndefined, .cb = 0xff});
            job->want_stencil = v.key.stencil_format != Core::NativeTarget::DepthTarget::StencilFormat::Invalid;
        }
        if (job->late_targets.empty()) {
            c_skip_no_attachment.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        width = height = layers = 0; // set by the resolver
    }
    if (Core::Vkb::ContractCensusOn()) { // api-contract.md C17 (debug): domain / hull / geometry shader slots (ctx + 0x4678 / 0x4680 / 0x4688), topology >= 10 (patches)
        const u64 ds = Rd<u64>(ctx, 0x4678), hs = Rd<u64>(ctx, 0x4680), gs = Rd<u64>(ctx, 0x4688);
        const u32 topology = Rd<u32>(ctx, 0x4480);
        if (ds != 0 || hs != 0 || gs != 0 || topology >= 10) {
            Core::Vkb::ContractNote(fmt::format("C17 draw with stages ds {} hs {} gs {}, topology {}", ds != 0, hs != 0, gs != 0, topology));
        }
    }
    if (Core::Vkb::ContractCensusOn() && Rd<u64>(ctx, CtxTarget7) != 0) { // api-contract.md C10 (debug): the target-7 mode (the Pro's ID buffer, note 10)
        const auto describe = [](u64 view) -> std::string {
            const u64 resource = view != 0 ? Rd<u64>(view, 0x30) : 0;
            return resource == 0 ? std::string("none") : fmt::format("{}x{} dxgi {}", Rd<u32>(resource, 0x180), Rd<u32>(resource, 0x184), Rd<u32>(resource, 0x190));
        };
        const u64 t7 = Rd<u64>(ctx, CtxTarget7);
        const u64 t7_resource = Rd<u64>(t7, 0x30);
        Core::Vkb::ContractTarget7(t7_resource != 0 ? Rd<u64>(t7_resource, 0x10) : 0,
                                   fmt::format("C10 draw in target-7 mode: target 7 {}, target 0 {}, pixel shader writes target 7 {}, uconfig gradient mode {}", describe(t7),
                                               describe(Rd<u64>(ctx, 0x4488)), (v.key.mrt_mask >> 7) & 1, Rd<u8>(ctx, 0x44c8)));
    }
    if (targets_reused) { // issue 82 point 2: the last draw's attachments (same views, formats and slots; nothing gone since)
        pass.colors = t_targets.colors;
        pass.depth = t_targets.depth;
        pass.depth_has_stencil = t_targets.depth_has_stencil;
        attachment_uses = t_targets.uses;
        width = t_targets.width, height = t_targets.height, layers = t_targets.layers;
    }
    for (u32 cb = 0; !late_images && !targets_reused && cb < pass.color_count; ++cb) {
        if ((v.key.mrt_mask & (1u << cb)) == 0 || v.color_formats[cb] == vk::Format::eUndefined || views[cb] == 0) {
            continue;
        }
        Core::Vkb::Attachment a;
        if (!Core::Vkb::AttachmentFor(Rd<u64>(views[cb], 0x10), v.color_formats[cb], cmd, a)) {
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 12) {
                const u64 view = views[cb];
                const u64 resource = Rd<u64>(view, 0x30);
                LOG_WARNING(Render_Vulkan, "vkb draw: no attachment for target slot {}: view id {} tag {} kind {:#x} desc {:#x} {:#x}; resource {:#x} id {} kind {} {}x{} format {} bind {:#x} memory {:#x}",
                            cb, Rd<u64>(view, 0x10), Rd<u32>(view, 0x18), Rd<u32>(view, 0x44), Rd<u32>(view, 0x40), Rd<u32>(view, 0x48), resource,
                            resource ? Rd<u64>(resource, 0x10) : 0, resource ? Rd<u32>(resource, 0x124) : 0, resource ? Rd<u32>(resource, 0x180) : 0,
                            resource ? Rd<u32>(resource, 0x184) : 0, resource ? Rd<u32>(resource, 0x190) : 0, resource ? Rd<u32>(resource, 0x1a0) : 0,
                            resource ? Rd<u64>(resource, 0x30) : 0);
            }
            c_skip_target.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (Core::Vkb::ContractCensusOn()) { // api-contract.md V7 (debug): the RTV's DXGI sRGB-ness (29, 91, 93; 90 / 92 typeless map to sRGB in the engine table) against our attachment format
            const u32 dxgi = Rd<u32>(views[cb], 0x40);
            const bool engine_srgb = dxgi == 29 || dxgi == 91 || dxgi == 93 || dxgi == 90 || dxgi == 92;
            const bool ours_srgb = vk::to_string(v.color_formats[cb]).find("Srgb") != std::string::npos;
            Core::Vkb::ContractNote(fmt::format("V7 render target dxgi {} -> {} ({})", dxgi, vk::to_string(v.color_formats[cb]), engine_srgb == ours_srgb ? "same sRGB-ness" : "sRGB DIFFERS"));
        }
        pass.colors[cb] = a.view;
        attachment_uses.push_back(Stream::ImageUse(a.image, true));
        width = std::min(width, a.extent.width), height = std::min(height, a.extent.height), layers = std::min(layers, a.layers);
    }
    if (!late_images && !targets_reused && v.key.z_format != Core::NativeTarget::DepthTarget::ZFormat::Invalid) {
        const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView);
        Core::Vkb::Attachment a;
        if (dsv == 0 || !Core::Vkb::AttachmentFor(Rd<u64>(dsv, 0x10), vk::Format::eUndefined, cmd, a)) {
            c_skip_depth.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        pass.depth = a.view;
        attachment_uses.push_back(Stream::ImageUse(a.image, true));
        pass.depth_has_stencil = a.has_stencil && v.key.stencil_format != Core::NativeTarget::DepthTarget::StencilFormat::Invalid;
        width = std::min(width, a.extent.width), height = std::min(height, a.extent.height), layers = std::min(layers, a.layers);
    }
    if (width == ~0u) {
        c_skip_no_attachment.fetch_add(1, std::memory_order_relaxed);
        return;
    }
#if defined(__clang__)
    pass.extent = {width, height};
#else // GCC: {..} also matches operator=(VkExtent2D const&) and is ambiguous
    pass.extent = vk::Extent2D{width, height};
#endif
    pass.layers = layers;
    if (targets_keep && !targets_reused) { // issue 82: found in full; kept for the next draw on this thread
        t_targets.key = target_key;
        t_targets.colors = pass.colors;
        t_targets.depth = pass.depth;
        t_targets.depth_has_stencil = pass.depth_has_stencil;
        t_targets.uses = attachment_uses;
        t_targets.width = width, t_targets.height = height, t_targets.layers = layers;
    }
    phase(0);
    // the resources of both stages, pixel first (the pipeline layout's order)
    VkbShader::PushData& push = job->push;
    push.xoffset = fixed.transform.x_offset;
    push.xscale = fixed.transform.x_scale;
    push.yoffset = fixed.transform.y_offset;
    push.yscale = fixed.transform.y_scale;
    Bound& b = job->b;
    Ranges& ranges = job->ranges;
    u32 buffer_index = 0;
    LateList* const late = ResolveOnRecorder() && !(v.record && BatchOn()) ? &job->late : nullptr; // R2b (note 64); R3b: a batched draw's index buffer is found now
    LateImages* const late_imgs = late_images ? &job->late_images : nullptr; // R2b part 2b
    // issue 42: the draw's sampler look (the anisotropy setting; the scene texture bias for a scene material pass while the AA runs)
    const Core::Vkb::Aa::SamplerLook look = Core::Vkb::Aa::LookFor(v.key.mrt_mask);
    const bool dc = t_depth_pass && DepthCensusOn();
    const u64 dc_b0 = dc ? NowNs() : 0;
    const u64 dc_c0 = t_dc_copies;
    if (v.record) {
        t_row_look = look;
        // R3: the draw's row in the host ring (the row table); the shaders read every resource through it
        const Stream::Slice slice = RowSlice(std::max<u32>(v.row_dwords, 1) * 4);
        u32* row = reinterpret_cast<u32*>(slice.data);
        if ((v.has_ps && !BuildRow(ctx, v.ps, row, uses)) || !BuildRow(ctx, v.vs, row, uses)) {
            c_rec_row_fail.fetch_add(1, std::memory_order_relaxed);
            t_row_failed = true; // the caller draws it the plain way
            return;
        }
        push.buf_offsets.fill(0);
        push.row_base = static_cast<u32>(slice.offset / 4);
        job->record = true;
        c_rec_draws.fetch_add(1, std::memory_order_relaxed);
    } else {
    if (v.has_ps && !BindStage(ctx, v.ps, cmd, push, buffer_index, b, ranges, uses, late, late_imgs, look)) {
        return;
    }
    if (!BindStage(ctx, v.vs, cmd, push, buffer_index, b, ranges, uses, late, late_imgs, look)) {
        return;
    }
    }
    if (dc) {
        c_dc_bind_ns.fetch_add(NowNs() - dc_b0, std::memory_order_relaxed);
        c_dc_copies.fetch_add(t_dc_copies - dc_c0, std::memory_order_relaxed);
    }
    phase(1);
    // the vertex inputs: each fetch attribute from its element of the input layout
    auto& input_bindings = job->input_bindings;
    auto& input_attributes = job->input_attributes;
    auto& vertex_buffers = job->vertex_buffers;
    auto& vertex_offsets = job->vertex_offsets;
    auto& vertex_sizes = job->vertex_sizes; // V4h: in place (staged), 0 on a device copy
    using InstanceIdType = VkbShader::Gcn::VertexAttribute::InstanceIdType;
    for (u32 i = 0; i < v.attributes.size(); ++i) {
        const auto& attribute = v.fetch.attributes[i];
        Core::NativeDraw::VertexElement e;
        Core::Vkb::BufferWindow w;
        bool vertex_late = false;
        if (late != nullptr && v.attributes[i].cls == Source::Class::Slot && Core::NativeDraw::VertexElementOf(ctx, v.attributes[i].slot, e) && e.bytes > attribute.inst_offset) {
            // R2b: the frame ring now, any other buffer by the recording thread
            const u64 size = e.bytes - attribute.inst_offset;
            if (e.buffer != 0) {
                const u64 memory = Rd<u64>(e.buffer, 0x30), offset = e.start + attribute.inst_offset;
                if (!Core::Vkb::BufferInRing(memory + offset, size, w)) {
                    late->push_back({.object = e.buffer, .id = Rd<u64>(e.buffer, 0x10), .memory = memory, .bytes = Rd<u32>(e.buffer, 0x180), .offset = offset, .size = size,
                                     .slot = static_cast<u16>(i), .what = LateBuffer::Vertex});
                    vertex_late = true;
                }
            } else {
                late->push_back({.offset = e.address + attribute.inst_offset, .size = size, .slot = static_cast<u16>(i), .what = LateBuffer::Vertex});
                vertex_late = true;
            }
        } else if (v.attributes[i].cls != Source::Class::Slot || !Core::NativeDraw::VertexElementOf(ctx, v.attributes[i].slot, e) || e.bytes <= attribute.inst_offset ||
            !Core::Vkb::BufferOfObject(e.buffer, e.start + attribute.inst_offset, e.bytes - attribute.inst_offset, w)) { // R7: by the slot's buffer object
            c_skip_vertex.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const u32 data_format = attribute.data_format != 0 ? attribute.data_format : e.data_format;
        const u32 num_format = attribute.data_format != 0 ? attribute.num_format : e.num_format;
        const auto step = attribute.GetStepRate();
        const u32 divisor = step == InstanceIdType::OverStepRate0 ? v.vs.runtime.sw.vs.step_rate_0 : step == InstanceIdType::OverStepRate1 ? v.vs.runtime.sw.vs.step_rate_1 : 1;
        input_bindings.push_back({.binding = i, .stride = e.stride, .inputRate = step == InstanceIdType::None ? vk::VertexInputRate::eVertex : vk::VertexInputRate::eInstance,
                                  .divisor = divisor});
        input_attributes.push_back({.location = i, .binding = i,
                                    .format = VkbVulkan::Vkb::LiverpoolToVK::SurfaceFormat(static_cast<VkbAmdGpu::DataFormat>(data_format), static_cast<VkbAmdGpu::NumberFormat>(num_format)),
                                    .offset = 0});
        if (vertex_late) { // R2b: filled by DrawJob::ResolveLate
            vertex_buffers.push_back({});
            vertex_sizes.push_back(0);
            vertex_offsets.push_back(0);
            continue;
        }
        vertex_buffers.push_back(w.buffer);
        vertex_sizes.push_back(w.device ? 0 : e.bytes - attribute.inst_offset);
        Core::Vkb::NoteBind(2, w, e.bytes - attribute.inst_offset, false);
        vertex_offsets.push_back(w.offset);
        uses.push_back(Stream::BufferUse(w.buffer, w.offset, e.bytes - attribute.inst_offset, false));
    }
    // the index buffer
    vk::Buffer& index_buffer = job->index_buffer;
    u64& index_offset = job->index_offset;
    vk::IndexType& index_type = job->index_type;
    u64& index_size = job->index_size; // V4h: in place (staged), 0 on a device copy
    vk::Buffer& args_buffer = job->args_buffer;
    u64& args_offset = job->args_offset;
    if (kind == 0 || kind == 2) {
        const u64 view = Rd<u64>(ctx, 0x58);
        const u32 type = Rd<u32>(ctx, 0x64);
        const u64 address = view != 0 ? (static_cast<u64>(Rd<u32>(view, 0x64) & 0xff) << 32 | (Rd<u32>(view, 0x60) & 0xfffffffeu)) : 0;
        index_type = type == 1 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
        const u64 index_bytes = type == 1 ? 4 : 2;
        // an indirect draw's index count is in its arguments: the whole index range of the view (its max index size, as the console's VGT_INDX_MAX_SIZE)
        const u64 index_range = kind == 0 ? (a2 + a1) * index_bytes : static_cast<u64>(view != 0 ? Rd<u32>(view, 0x68) : 0) * index_bytes;
        Core::Vkb::BufferWindow w;
        bool index_late = false;
        if (late != nullptr && view != 0 && address != 0 && index_range != 0 && Rd<u32>(ctx, 0x60) == 0x39 && type <= 1) {
            // R2b: the frame ring now, any other buffer by the recording thread
            const u64 memory = Rd<u64>(view, 0x30);
            if (memory == 0 || !Core::Vkb::BufferInRing(memory, index_range, w)) {
                if (memory != 0) {
                    late->push_back({.object = view, .id = Rd<u64>(view, 0x10), .memory = memory, .bytes = Rd<u32>(view, 0x180), .offset = 0, .size = index_range, .what = LateBuffer::Index});
                } else {
                    late->push_back({.offset = address, .size = index_range, .what = LateBuffer::Index});
                }
                index_late = true;
            } else if ((w.offset % index_bytes) != 0) {
                c_skip_index.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        if (!index_late) {
        // R7 (note 56): the index buffer object's own memory from its start (census: always so)
        const bool found = late != nullptr ? static_cast<bool>(w.buffer) : view != 0 && Core::Vkb::BufferOfObject(view, 0, index_range, w);
        if (address == 0 || index_range == 0 || Rd<u32>(ctx, 0x60) != 0x39 || type > 1 || !found || (w.offset % index_bytes) != 0) {
            c_skip_index.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (Core::Vkb::R7CensusOn()) {
            Core::Vkb::R7Census(4, view, address, index_range); // note 56
        }
        index_buffer = w.buffer, index_offset = w.offset;
        index_size = w.device ? 0 : index_range;
        Core::Vkb::NoteBind(3, w, index_range, false);
        uses.push_back(Stream::BufferUse(w.buffer, w.offset, index_range, false));
        }
    }
    if (kind == 2) {
        // V5a: DrawIndexedInstancedIndirect(args buffer a1, offset a2): the arguments (index count, instance count, first index, base vertex, first instance: the layout of
        // VkDrawIndexedIndirectCommand) at the buffer's memory plus the offset (the engine's SET_BASE + data offset)
        const u64 packed = a1 != 0 && IsHeapPointer(a1) ? Rd<u64>(a1, 0x30) : 0;
        const u64 args = ((packed & 0xfffffff8ull) | ((packed >> 32) & 0xff) << 32) + static_cast<u32>(a2);
        Core::Vkb::BufferWindow w;
        if (late != nullptr && packed != 0) { // R2b: by the recording thread
            late->push_back({.object = a1, .id = Rd<u64>(a1, 0x10), .memory = Rd<u64>(a1, 0x30), .bytes = Rd<u32>(a1, 0x180), .offset = static_cast<u32>(a2), .size = 20,
                             .what = LateBuffer::Args});
        } else if (packed == 0 || !Core::Vkb::BufferOfObject(a1, static_cast<u32>(a2), 20, w) || (w.offset & 3) != 0) {
            c_skip_indirect.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (late == nullptr) { // (R2b: else filled by DrawJob::ResolveLate)
            args_buffer = w.buffer, args_offset = w.offset;
            uses.push_back(Stream::BufferUse(w.buffer, w.offset, 20, false));
        }
        c_indirect_recorded.fetch_add(1, std::memory_order_relaxed);
    }
    phase(2);
    // the pass, then the pipeline and its state
    // V4g: the image types of the descriptor writes (in the order the job writes them), then the commands, on the recording thread
    auto& image_types = job->image_types;
    image_types.resize(b.images.size());
    for (u32 r = 0; r < ranges.size(); r += 3) {
        const Stage& st = (v.has_ps && r == 0) ? v.ps : v.vs;
        for (u32 k = 0; k < ranges[r + 1].second; ++k) {
            image_types[ranges[r + 1].first + k] = st.info->images[k].is_written ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        }
    }
    const u32 instances = std::max(Rd<u32>(ctx, 0x4724), 1u);
    job->instances = instances;
    {
        static const u64 dump_hash = [] { const char* e = std::getenv("TLG_VKB_DRAW_DUMP"); return e != nullptr ? std::strtoull(e, nullptr, 16) : 0ull; }();
        static std::atomic<u32> dump_draws{0};
        const u64 ps = v.has_ps && v.ps.info ? v.ps.info->pgm_hash : 0ull;
        if (dump_hash != 0 && (ps == dump_hash || (ps & 0xffffffffull) == dump_hash) && [] { const u32 n = dump_draws.fetch_add(1, std::memory_order_relaxed); return n == 100 || n == 400 || n == 1000; }()) {
            DrawDumpLine(fmt::format("draw: kind {} a1 {} a2 {} a3 {} instances {} (ctx+0x4724 {}); pass {}x{} layers {} colors {}", kind, a1, a2, a3, instances, Rd<u32>(ctx, 0x4724), pass.extent.width, pass.extent.height, pass.layers, pass.color_count));
        }
    }
    // note 41, debug only: TLG_VKB_DRAW_CENSUS=1: every draw of the graphics submissions from TLG_VKB_IMAGE_DUMP_AT (default 2600) on for about one frame (8 submissions),
    // one line each into log/vkb_draw_dump.txt: shaders, counts, pass, and what each pixel shader image slot holds
    {
        static const bool census = [] { const char* e = std::getenv("TLG_VKB_DRAW_CENSUS"); return e != nullptr && e[0] == '1'; }();
        static std::atomic<u64> census_at = [] { const char* e = std::getenv("TLG_VKB_IMAGE_DUMP_AT"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 2600ull; }();
        // TLG_VKB_DRAW_CENSUS_TRIGGER=<file> (debug): when the file appears, the census runs again from the next submission (its first line is the label), and the file is removed
        if (static const char* trigger = std::getenv("TLG_VKB_DRAW_CENSUS_TRIGGER"); census && trigger != nullptr) {
            static std::atomic<u64> checked{0};
            const u64 now = Core::Vkb::GraphicsSubmits();
            if (checked.exchange(now, std::memory_order_relaxed) != now && std::filesystem::exists(trigger)) {
                std::string label;
                {
                    std::ifstream f(trigger);
                    std::getline(f, label);
                }
                std::error_code ec;
                std::filesystem::remove(trigger, ec);
                census_at.store(now + 1, std::memory_order_relaxed);
                DrawDumpLine(fmt::format("=== census {} from submit {}", label, now + 1));
                Core::Vkb::R3dDumpTraced(label.substr(0, label.find(' '))); // debug R3d trace (TLG_VKB_R3D_TRACE)
                Core::Vkb::R3dStaleCheck(label.substr(0, label.find(' '))); // debug (TLG_VKB_R3D_STALE_CHECK)
                // "<label> ps=<hash>": the first draw of that pixel shader from now on dumps its images into vkb_image_dump/<label>/
                if (const auto at = label.find("ps="); at != std::string::npos) {
                    std::scoped_lock lock{g_live_dump_mutex};
                    g_live_dump_label = label.substr(0, at);
                    while (!g_live_dump_label.empty() && g_live_dump_label.back() == ' ') {
                        g_live_dump_label.pop_back();
                    }
                    g_live_dump_ps.store(std::strtoull(label.c_str() + at + 3, nullptr, 16), std::memory_order_relaxed);
                } else if (const auto vat = label.find("vs="); vat != std::string::npos) {
                    std::scoped_lock lock{g_live_dump_mutex};
                    g_live_dump_label = label.substr(0, vat);
                    while (!g_live_dump_label.empty() && g_live_dump_label.back() == ' ') {
                        g_live_dump_label.pop_back();
                    }
                    g_live_dump_vs.store(std::strtoull(label.c_str() + vat + 3, nullptr, 16), std::memory_order_relaxed);
                }
            }
        }
        if (census && Core::Vkb::GraphicsSubmits() >= census_at.load(std::memory_order_relaxed) && Core::Vkb::GraphicsSubmits() < census_at.load(std::memory_order_relaxed) + 8) {
            std::string text = fmt::format("submit {} vs {:#x} ps {:#x} kind {} a1 {} a2 {} instances {} pass {}x{} colors {} depth {}", Core::Vkb::GraphicsSubmits(),
                                           v.vs.info ? v.vs.info->pgm_hash : 0ull, v.has_ps && v.ps.info ? v.ps.info->pgm_hash : 0ull, kind, a1, a2, instances, pass.extent.width, pass.extent.height, pass.color_count,
                                           pass.depth ? 1 : 0);
            text += fmt::format(" dtest {} dwrite {} dcmp {} cull {}", fixed.depth_test ? 1 : 0, fixed.depth_write ? 1 : 0, vk::to_string(fixed.depth_compare),
                                static_cast<u32>(VkCullModeFlags(fixed.cull_mode)));
            text += fmt::format(" stest {} sref {} cmpmask {:x} wmask {:x} sfunc {} pass {} fail {} zfail {} stencil_attached {}", fixed.stencil_test ? 1 : 0, fixed.stencil_reference[0],
                                fixed.stencil_compare_mask[0], fixed.stencil_write_mask[0], vk::to_string(fixed.stencil_front_ops.compare_op),
                                vk::to_string(fixed.stencil_front_ops.pass_op), vk::to_string(fixed.stencil_front_ops.fail_op), vk::to_string(fixed.stencil_front_ops.depth_fail_op),
                                pass.depth_has_stencil ? 1 : 0);
            text += " bufs";
            for (u32 i = 0; i < b.guest.size(); ++i) {
                text += fmt::format(" {:#x}/{}", b.guest[i].first, b.guest[i].second);
            }
            if (v.has_ps) {
                text += " expfmt";
                for (u32 cb = 0; cb < 8; ++cb) {
                    text += fmt::format(" {}:{}/{}", cb, static_cast<u32>(v.ps.runtime.hw.fs.color_buffers[cb].export_format),
                                        static_cast<u32>(v.ps.runtime.hw.fs.color_buffers[cb].data_format));
                }
            }
            text += fmt::format(" vp {},{} {}x{} z {}..{} sc {},{} {}x{} clipdis {} fs {}", fixed.viewports[0].x, fixed.viewports[0].y, fixed.viewports[0].width,
                                fixed.viewports[0].height, fixed.viewports[0].minDepth, fixed.viewports[0].maxDepth, fixed.scissors[0].offset.x, fixed.scissors[0].offset.y,
                                fixed.scissors[0].extent.width, fixed.scissors[0].extent.height, fixed.clip_disabled ? 1 : 0, static_cast<u32>(fixed.front_face));
            if (const u64 raster = Rd<u64>(ctx, 0x46c8); raster != 0) {
                text += fmt::format(" bias {} clamp {} slope {}", Rd<s32>(raster, 0x3c), Rd<float>(raster, 0x40), Rd<float>(raster, 0x44));
            }
            if (const u64 blend = Rd<u64>(ctx, 0x46f0); blend != 0) { // api map: D3D11_BLEND_DESC AlphaToCoverageEnable (not implemented), blend enable of target 0
                text += fmt::format(" a2c {} blend0 {}", Rd<u8>(blend, 0x30), Rd<u8>(blend, 0x34));
            }
            {
                const auto targets = TargetViews(ctx);
                for (u32 cb = 0; cb < v.key.num_color_attachments; ++cb) {
                    text += fmt::format(" m{}={:x}/{:x}{}", cb, static_cast<u32>(VkColorComponentFlags(v.key.write_masks[cb])), fixed.target_write_mask[cb],
                                        v.key.blend[cb].enable ? "b" : "");
                    text += fmt::format(" o{}={} ({})", cb, targets[cb] != 0 ? Rd<u64>(targets[cb], 0x10) : 0ull, targets[cb] != 0 ? Core::Vkb::DescribeView(Rd<u64>(targets[cb], 0x10)).substr(0, 12) : std::string());
                }
            }
            if (v.has_ps) {
                for (u32 i = 0; i < v.ps.images.size(); ++i) {
                    const Source& src = v.ps.images[i];
                    const u64 object = src.cls == Source::Class::Slot && src.kind == 1 ? SlotObject(ctx, v.ps.index, 1, src.slot) : 0;
                    text += fmt::format(" | t{} {}", src.slot, object != 0 ? fmt::format("view {} {}", Rd<u64>(object, 0x10), Core::Vkb::DescribeView(Rd<u64>(object, 0x10))) : std::string("none"));
                }
            }
            DrawDumpLine(text);
        }
    }
    // note 41, debug only: TLG_VKB_IMAGE_DUMP_PS=<pixel shader hash> (TLG_VKB_IMAGE_DUMP_AT=<graphics submit>, default 2600): the first such draw from then on dumps its
    // images (inputs in0.., the color targets out0.., the depth target depth) right after the draw, once
    auto& dump_views = job->dump_views;
    bool& dump_buffers = job->dump_buffers;
    job->dump_prefix.clear();
    {
        static const u64 dump_ps = [] { const char* e = std::getenv("TLG_VKB_IMAGE_DUMP_PS"); return e != nullptr ? std::strtoull(e, nullptr, 16) : 0ull; }();
        static const u64 dump_at = [] { const char* e = std::getenv("TLG_VKB_IMAGE_DUMP_AT"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 2600ull; }();
        static const u64 dump_nth = [] { const char* e = std::getenv("TLG_VKB_IMAGE_DUMP_NTH"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull; }();
        static std::atomic<u64> seen{0};
        // note 90: in debugger mode a slow pass's pixel shader is armed (Stream::TakeSlowPsArm); its next draw is dumped the same way into vkb_image_dump/debugger/,
        // which the freeze moves into its folder (slow_draws/)
        const bool debugger_dump = Core::FrameWalk::DebuggerMode() && v.has_ps && v.ps.info && Stream::TakeSlowPsArm(v.ps.info->pgm_hash);
        std::string live_prefix;
        if (u64 live = g_live_dump_ps.load(std::memory_order_relaxed); live != 0 && v.has_ps && v.ps.info &&
            (v.ps.info->pgm_hash == live || (v.ps.info->pgm_hash & 0xffffffffull) == live) && g_live_dump_ps.compare_exchange_strong(live, 0)) {
            std::scoped_lock lock{g_live_dump_mutex};
            live_prefix = g_live_dump_label + "/";
        } else if (u64 live_vs = g_live_dump_vs.load(std::memory_order_relaxed); live_vs != 0 && v.vs.info &&
                   (v.vs.info->pgm_hash == live_vs || (v.vs.info->pgm_hash & 0xffffffffull) == live_vs) && g_live_dump_vs.compare_exchange_strong(live_vs, 0)) {
            std::scoped_lock lock{g_live_dump_mutex};
            live_prefix = g_live_dump_label + "/";
        }
        const std::string prefix = debugger_dump ? fmt::format("debugger/ps_{:016x}/", v.ps.info->pgm_hash) : live_prefix;
        job->dump_prefix = prefix;
        if (debugger_dump || !live_prefix.empty() || (dump_ps != 0 && v.has_ps && v.ps.info && v.ps.info->pgm_hash == dump_ps && Core::Vkb::GraphicsSubmits() >= dump_at && seen.fetch_add(1) == dump_nth)) {
            for (const Stage* st : {&v.vs, &v.ps}) {
                for (u32 i = 0; i < st->images.size(); ++i) {
                    const Source& src = st->images[i];
                    const u64 object = src.cls == Source::Class::Slot && src.kind == 1 ? SlotObject(ctx, st->index, 1, src.slot) : 0;
                    if (object != 0) {
                        dump_views.push_back({Rd<u64>(object, 0x10), fmt::format("{}{}_in{}_slot{}", prefix, st == &v.ps ? "ps" : "vs", i, src.slot)});
                    }
                }
            }
            const auto targets = TargetViews(ctx);
            for (u32 cb = 0; cb < v.key.num_color_attachments; ++cb) {
                if (targets[cb] != 0) {
                    dump_views.push_back({Rd<u64>(targets[cb], 0x10), fmt::format("{}out{}", prefix, cb)});
                }
            }
            if (const u64 dsv = Rd<u64>(ctx, CtxDepthStencilView); dsv != 0) {
                dump_views.push_back({Rd<u64>(dsv, 0x10), prefix + "depth"});
            }
            // and the buffers: the guest bytes now (at the draw call), the GPU's copy after the draw
            const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_image_dump" / prefix;
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            b.guest.resize(b.buffers.size());
            for (u32 i = 0; i < b.guest.size(); ++i) {
                const auto [address, size] = b.guest[i];
                if (address != 0 && size != 0 && size <= (64u << 20)) {
                    if (FILE* f = std::fopen((dir / fmt::format("g{}_{:#x}_{}B.raw", i, address, size)).string().c_str(), "wb")) {
                        std::fwrite(reinterpret_cast<const void*>(address), 1, size, f);
                        std::fclose(f);
                    }
                }
            }
            dump_buffers = true;
            LOG_INFO(Render_Vulkan, "vkb image dump: draw of ps {:#x} at graphics submit {}: {} images", dump_ps, Core::Vkb::GraphicsSubmits(), dump_views.size());
        }
    }
    Core::Vkb::Census::Work(0);
    // the glitch marker's and the frame walk's draw table (draws.json): one row per draw, as the PS4 path wrote (F6 / F5; TLG_MARKER=0 off)
    if (Core::Marker::Enabled()) {
        Core::Marker::NoteRow({.vs_hash = v.vs.info ? v.vs.info->pgm_hash : 0ull,
                               .ps_hash = v.has_ps && v.ps.info ? v.ps.info->pgm_hash : 0ull,
                               .color0 = Rd<u64>(ctx, 0x4488), // engine render target view of slot 0 (Ctx::RenderTargets)
                               .depth = Rd<u64>(ctx, 0x44d0),  // engine depth stencil view (Ctx::DepthStencilView)
                               .frame = static_cast<u32>(Core::DebugServer::PresentCount()),
                               .count = static_cast<u32>(a1),
                               .instances = instances,
                               .kind = static_cast<u8>(kind == 2 ? 1 : 0),
                               .draw_class = 15,
                               .flags = 2,
                               .colors = static_cast<u8>(pass.color_count),
                               .width = static_cast<u16>(pass.extent.width),
                               .height = static_cast<u16>(pass.extent.height)});
    }
    job->pipeline = v.pipeline->Handle();
    job->layout = v.pipeline->GetLayout();
    job->push_descriptors = v.pipeline->UsesPushDescriptors();
    job->set_layout = v.pipeline->GetDescriptorSetLayout();
    job->fixed_state = &fixed; // interned by FixedStateOf, never freed
    job->masks = v.key.write_masks;
    job->kind = kind, job->a1 = a1, job->a2 = a2, job->a3 = a3;
    job->first_vertex = kind == 2 ? 0u : VulkanFirstVertex(ctx, v, static_cast<u32>(kind == 0 ? a3 : a2));
    job->jittered = false;
    if (Core::Vkb::Aa::On()) [[unlikely]] { // F1c: decided here, in game order (two recording threads must not see different offsets within a frame)
        job->jittered = AaDrawJitter(ctx, v, fixed, kind, a1, a2, a3, instances, job->jitter);
    }
    job->ps_hash = v.has_ps && v.ps.info ? v.ps.info->pgm_hash : 0ull;
    job->color_format0 = static_cast<u32>(v.color_formats[0]);
    job->debug_id = 0, job->capture = false;
    job->vs_hash = v.vs.info ? v.vs.info->pgm_hash : 0ull;
    if (Core::FrameWalk::DebuggerMode()) [[unlikely]] { // note 91
        static std::atomic<u64> ids{0};
        job->debug_id = ids.fetch_add(1, std::memory_order_relaxed) + 1;
        if (Stream::FrameCaptureOn()) {
            job->capture = true;
            Stream::CaptureDrawQueued();
        }
    }
    job->work = Stream::Work::Draw;
    t_join_dsv = Rd<u64>(ctx, CtxDepthStencilView), t_join_rtv0 = Rd<u64>(ctx, CtxRenderTargets), t_join_rtvs = RtvHash(ctx);
    if (job->record && BatchOn() && JoinBatch(*job)) { // R3b
        phase(3);
        c_recorded.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Stream::Enqueue(std::move(job));
    phase(3);
    c_recorded.fetch_add(1, std::memory_order_relaxed);
    if (TimesOn()) {
        c_record_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    }
}

/// R2b (note 64): the buffers the capture left are found here, on the recording thread (stream recording held: the objects lock nests inside it, as FindLocked expects).
/// The census tools and TLG_VKB_CB_AT_DRAW resolve on the engine's render thread at the draw (they read engine objects or thread-local tags while they resolve).
std::atomic<u64> c_late_buffers{0}, c_late_failed{0}, c_late_targets{0}, c_late_images{0};
bool LateImagesOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_VKB_LATE_IMAGES");
        const auto set = [](const char* name) { const char* v = std::getenv(name); return v != nullptr && v[0] != 0 && v[0] != '0'; };
        // the draw dump and census print the pass at the capture
        return e != nullptr && e[0] == '1' && ResolveOnRecorder() && !set("TLG_VKB_DRAW_DUMP") && !set("TLG_VKB_DRAW_CENSUS") &&
               !set("TLG_VKB_IMAGE_DUMP_PS");
    }();
    return on;
}
bool ResolveOnRecorder() {
    static const bool on = [] {
        const char* cb = std::getenv("TLG_VKB_CB_AT_DRAW");
        return (cb == nullptr || cb[0] != '1') && Stream::RecorderOn() && !StageCensusOn() && !Core::Vkb::R7CensusOn() &&
               !Core::Vkb::FillReadersOn();
    }();
    return on;
}
bool DrawJob::ResolveLate() {
    const u64 alignment = instance->StorageMinAlignment();
    c_late_buffers.fetch_add(late.size(), std::memory_order_relaxed);
    for (const LateBuffer& l : late) {
        Core::Vkb::BufferWindow w;
        const bool found = l.object != 0 ? Core::Vkb::BufferOfObjectFields(l.object, l.id, l.memory, l.bytes, l.offset, l.size, w) : Core::Vkb::FindBuffer(l.offset, l.size, w);
        switch (l.what) {
        case LateBuffer::Stage: {
            if (!found) {
                c_skip_buffer.fetch_add(1, std::memory_order_relaxed);
                c_late_failed.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            const u64 aligned = w.offset & ~(alignment - 1);
            push.buf_offsets[l.push_index] = static_cast<u8>(w.offset - aligned);
#if defined(__clang__)
            b.buffers[l.slot] = {w.buffer, aligned, l.size + (w.offset - aligned)};
#else // GCC: {..} also matches operator=(VkDescriptorBufferInfo const&) and is ambiguous
            b.buffers[l.slot] = vk::DescriptorBufferInfo{w.buffer, aligned, l.size + (w.offset - aligned)};
#endif
            b.stageable[l.slot] = !w.device && !l.written ? 1 : 0;
            uses.push_back(Stream::BufferUse(w.buffer, w.offset, l.size, l.written));
            Core::Vkb::NoteBind(1, w, l.size, l.written);
            break;
        }
        case LateBuffer::Vertex:
            if (!found) {
                c_skip_vertex.fetch_add(1, std::memory_order_relaxed);
                c_late_failed.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            vertex_buffers[l.slot] = w.buffer;
            vertex_offsets[l.slot] = w.offset;
            vertex_sizes[l.slot] = w.device ? 0 : l.size;
            Core::Vkb::NoteBind(2, w, l.size, false);
            uses.push_back(Stream::BufferUse(w.buffer, w.offset, l.size, false));
            break;
        case LateBuffer::Index:
            if (!found || (w.offset % (index_type == vk::IndexType::eUint32 ? 4 : 2)) != 0) {
                c_skip_index.fetch_add(1, std::memory_order_relaxed);
                c_late_failed.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            index_buffer = w.buffer, index_offset = w.offset;
            index_size = w.device ? 0 : l.size;
            Core::Vkb::NoteBind(3, w, l.size, false);
            uses.push_back(Stream::BufferUse(w.buffer, w.offset, l.size, false));
            break;
        case LateBuffer::Args:
            if (!found || (w.offset & 3) != 0) {
                c_skip_indirect.fetch_add(1, std::memory_order_relaxed);
                c_late_failed.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            args_buffer = w.buffer, args_offset = w.offset;
            uses.push_back(Stream::BufferUse(w.buffer, w.offset, 20, false));
            break;
        }
    }
    // R2b part 2b: the targets (the pass size comes from them) and the shader images
    if (!late_targets.empty()) {
        c_late_targets.fetch_add(late_targets.size(), std::memory_order_relaxed);
        u32 width = ~0u, height = ~0u, layers = ~0u;
        for (const LateTarget& t : late_targets) {
            Core::Vkb::Attachment a;
            if (!Core::Vkb::AttachmentFor(t.view_id, t.format, vk::CommandBuffer{}, a)) {
                (t.cb == 0xff ? c_skip_depth : c_skip_target).fetch_add(1, std::memory_order_relaxed);
                c_late_failed.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            if (t.cb == 0xff) {
                pass.depth = a.view;
                pass.depth_has_stencil = a.has_stencil && want_stencil;
            } else {
                pass.colors[t.cb] = a.view;
            }
            attachment_uses.push_back(Stream::ImageUse(a.image, true));
            width = std::min(width, a.extent.width), height = std::min(height, a.extent.height), layers = std::min(layers, a.layers);
        }
#if defined(__clang__)
        pass.extent = {width, height};
#else // GCC: {..} also matches operator=(VkExtent2D const&) and is ambiguous
        pass.extent = vk::Extent2D{width, height};
#endif
        pass.layers = layers;
    }
    c_late_images.fetch_add(late_images.size(), std::memory_order_relaxed);
    for (const LateImage& li : late_images) {
        vk::ImageView view{};
        vk::Image image{};
        if (!Core::Vkb::ViewForCompute(li.view_id, li.type, vk::CommandBuffer{}, view, &image)) {
            c_skip_image.fetch_add(1, std::memory_order_relaxed);
            c_late_failed.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        b.images[li.slot].imageView = view;
        uses.push_back(Stream::ImageUse(image, li.written));
    }
    return true;
}

struct DrawStateCache {
    u64 generation{~0ull};
    VkCommandBuffer cmd{};
    VkPipeline pipeline{};
    bool jittered{};
    std::array<float, 2> jitter{};
    const VkbVulkan::FixedState* fixed{};
    u64 pass_key{~0ull};
    std::array<vk::ColorComponentFlags, VkbAmdGpu::NUM_COLOR_BUFFERS> masks{};
    boost::container::small_vector<vk::VertexInputBindingDescription2EXT, 16> bindings;
    boost::container::small_vector<vk::VertexInputAttributeDescription2EXT, 16> attributes;
    bool vertex_input{};
    boost::container::small_vector<vk::Buffer, 16> vertex_buffers;
    boost::container::small_vector<vk::DeviceSize, 16> vertex_offsets;
    VkBuffer index_buffer{};
    u64 index_offset{~0ull};
    vk::IndexType index_type{};
};
thread_local DrawStateCache t_draw_state; ///< R3b: namespace scope (a batch job resets it)
thread_local VkCommandBuffer t_global_cmd{};
thread_local u64 t_global_generation{~0ull};
/// R3b: the dynamic state a draw's pipeline declares (from the interned fixed state, the pass, the write masks); shared by the draw job and the batch job
void SendDynamicState(vk::CommandBuffer cmd, const VkbVulkan::FixedState& fixed, const Stream::Pass& pass, const std::array<vk::ColorComponentFlags, VkbAmdGpu::NUM_COLOR_BUFFERS>& masks,
                      bool jittered, const std::array<float, 2>& jitter, u64 ps_hash, u32 color_format0) {
    // the dynamic state the pipeline declares, all of it, every draw
    boost::container::small_vector<vk::Viewport, 16> viewports;
    boost::container::small_vector<vk::Rect2D, 16> scissors;
    for (u32 i = 0; i < std::max(fixed.viewport_count, 1u); ++i) {
        vk::Viewport viewport = fixed.viewports[i];
        if (!instance->IsDepthRangeUnrestrictedSupported()) {
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }
        if (fixed.clip_disabled && !fixed.viewport_empty) {
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance->GetMaxViewportWidth(), 16384));
            viewport.height = float(std::min<u32>(instance->GetMaxViewportHeight(), 16384));
        } else if (jittered) { // F1c: the frame's sub-pixel offset, in pixels (their viewport jitter)
            viewport.x += jitter[0];
            viewport.y += jitter[1];
        }
        viewports.push_back(viewport);
        scissors.push_back(fixed.scissors[i]);
    }
    cmd.setViewportWithCount(viewports);
    cmd.setScissorWithCount(scissors);
    {
        const bool dt = fixed.depth_test && fixed.depth_target && pass.depth;
        const bool st = fixed.stencil_test && fixed.stencil_target && pass.depth_has_stencil;
        Stream::NotePassDraw((pass.depth ? 1u : 0u) | (pass.depth_has_stencil ? 2u : 0u) | (dt ? 4u : 0u) | (st ? 8u : 0u) |
                                 (fixed.depth_bounds_test && pass.depth ? 16u : 0u) | (dt && fixed.depth_write && !fixed.depth_clear ? 32u : 0u),
                             ps_hash, color_format0, viewports[0], scissors[0]);
    }
    static const u64 test_nodepth_ps = [] { const char* e = std::getenv("TLG_VKB_TEST_NODEPTH_PS"); return e != nullptr ? std::strtoull(e, nullptr, 16) : 0ull; }(); // note 41, test
    const bool depth_test = fixed.depth_test && fixed.depth_target && pass.depth && !(test_nodepth_ps != 0 && ps_hash == test_nodepth_ps);
    cmd.setDepthTestEnable(depth_test);
    cmd.setDepthWriteEnable(depth_test && fixed.depth_write && !fixed.depth_clear);
    cmd.setDepthCompareOp(depth_test ? fixed.depth_compare : vk::CompareOp::eAlways);
    if (instance->IsDepthBoundsSupported()) {
        cmd.setDepthBoundsTestEnable(fixed.depth_bounds_test && pass.depth);
        cmd.setDepthBounds(fixed.depth_bounds_min, fixed.depth_bounds_max);
    }
    // note 41, test only: TLG_VKB_TEST_SHADOW_BIAS=<constant>,<slope>: that bias on every draw into a pass without color targets
    static const std::array<float, 2> test_bias = [] {
        std::array<float, 2> r{0.0f, 0.0f};
        if (const char* e = std::getenv("TLG_VKB_TEST_SHADOW_BIAS")) {
            std::sscanf(e, "%f,%f", &r[0], &r[1]);
        }
        return r;
    }();
    static const float test_eq_bias = [] { const char* e = std::getenv("TLG_VKB_TEST_EQ_BIAS"); return e != nullptr ? static_cast<float>(std::atof(e)) : 0.0f; }();
    if (test_eq_bias != 0.0f && pass.color_count > 0 && fixed.depth_test && !fixed.depth_write) {
        cmd.setDepthBiasEnable(true);
        cmd.setDepthBias(test_eq_bias, 0.0f, 0.0f);
    } else if ((test_bias[0] != 0.0f || test_bias[1] != 0.0f) && pass.color_count == 0) {
        cmd.setDepthBiasEnable(true);
        cmd.setDepthBias(test_bias[0], 0.0f, test_bias[1]);
    } else {
        cmd.setDepthBiasEnable(fixed.depth_bias);
        cmd.setDepthBias(fixed.depth_bias_constant, fixed.depth_bias_clamp, fixed.depth_bias_slope);
    }
    const bool stencil_test = fixed.stencil_test && fixed.stencil_target && pass.depth_has_stencil;
    cmd.setStencilTestEnable(stencil_test);
    cmd.setStencilOp(vk::StencilFaceFlagBits::eFront, fixed.stencil_front_ops.fail_op, fixed.stencil_front_ops.pass_op, fixed.stencil_front_ops.depth_fail_op,
                     fixed.stencil_front_ops.compare_op);
    cmd.setStencilOp(vk::StencilFaceFlagBits::eBack, fixed.stencil_back_ops.fail_op, fixed.stencil_back_ops.pass_op, fixed.stencil_back_ops.depth_fail_op,
                     fixed.stencil_back_ops.compare_op);
    cmd.setStencilReference(vk::StencilFaceFlagBits::eFront, fixed.stencil_reference[0]);
    cmd.setStencilReference(vk::StencilFaceFlagBits::eBack, fixed.stencil_reference[1]);
    cmd.setStencilWriteMask(vk::StencilFaceFlagBits::eFront, fixed.stencil_clear ? 0u : fixed.stencil_write_mask[0]);
    cmd.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, fixed.stencil_clear ? 0u : fixed.stencil_write_mask[1]);
    cmd.setStencilCompareMask(vk::StencilFaceFlagBits::eFront, fixed.stencil_compare_mask[0]);
    cmd.setStencilCompareMask(vk::StencilFaceFlagBits::eBack, fixed.stencil_compare_mask[1]);
    cmd.setCullMode(fixed.cull_mode);
    cmd.setFrontFace(fixed.front_face);
    const bool restart = fixed.primitive_restart && (instance->IsListRestartSupported() || !fixed.list_topology) &&
                         (instance->IsPatchListRestartSupported() || !fixed.patch_list_topology);
    cmd.setPrimitiveRestartEnable(restart);
    cmd.setRasterizerDiscardEnable(fixed.rasterizer_discard);
    cmd.setLineWidth(1.0f);
    cmd.setBlendConstants(fixed.blend_constants.data());
    if (instance->IsDynamicColorWriteMaskSupported() && pass.color_count != 0) {
        cmd.setColorWriteMaskEXT(0, std::span<const vk::ColorComponentFlags>{masks.data(), pass.color_count});
    }
    }

/// R3b: the record-mode draws of one pass, recorded together: the hazards of all and the pass once, then per draw only what changed (pipeline, dynamic state,
/// index buffer), the push constants (the row base) and the draw.
struct BatchDraw {
    vk::Pipeline pipeline{};
    vk::PipelineLayout layout{};
    const VkbVulkan::FixedState* fixed{};
    std::array<vk::ColorComponentFlags, VkbAmdGpu::NUM_COLOR_BUFFERS> masks{};
    VkbShader::PushData push{};
    vk::Buffer index_buffer{};
    u64 index_offset{};
    vk::IndexType index_type{};
    u64 index_size{}; ///< in place (staged by the recorder), 0 on a device copy
    vk::Buffer args_buffer{}; ///< an indirect draw's arguments (else null)
    u64 args_offset{};
    u32 count{}, first{}, instances{1};
    s32 vertex_offset{};
    u64 ps_hash{};
    u32 color_format0{};
    bool jittered{};
    std::array<float, 2> jitter{};
};
std::atomic<u64> c_batches{0}, c_batch_draws{0};
std::atomic<u64> c_flush_by[3]{}; ///< R3b: batches cut by another job / submit (0), by a pass change (1), at the pass's end (2)
thread_local u32 t_flush_reason = 0;
struct BatchJob final : Stream::Job {
    Stream::Pass pass{};
    u64 engine_dsv{}, engine_rtv0{}; ///< R7 step C: the engine views the pass was made from
    u64 engine_rtvs{}; ///< G2: all its colour targets
    u64 serial{};
    boost::container::small_vector<Stream::Use, 9> attachment_uses;
    std::vector<Stream::Use> uses;
    std::vector<BatchDraw> draws;
    void Run(vk::CommandBuffer cmd) override {
        for (BatchDraw& d : draws) { // V4h: an index buffer read in place gets its copy in front of the submission
            if (d.index_size != 0) {
                (void)Stream::StageRead(d.index_buffer, d.index_offset, d.index_size);
            }
        }
        if (!Stream::LegacyBarriersOn()) {
            (void)Stream::Hazard(cmd, std::span<const Stream::Use>{uses.data(), uses.size()});
        }
        if (Stream::UsePass(cmd, pass, std::span<const Stream::Use>{attachment_uses.data(), attachment_uses.size()})) {
            c_new_passes.fetch_add(1, std::memory_order_relaxed);
        }
        t_draw_state = {}; // the plain draws after the batch send their state again
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, draws.front().layout, 0, Core::Vkb::Indexed::Set(), {});
        t_global_cmd = static_cast<VkCommandBuffer>(cmd), t_global_generation = Stream::RecordingGeneration();
        cmd.setVertexInputEXT({}, {}); // the cascades' vertex shaders fetch their vertices themselves (no attributes)
        vk::Pipeline pipeline{};
        const VkbVulkan::FixedState* fixed{};
        const std::array<vk::ColorComponentFlags, VkbAmdGpu::NUM_COLOR_BUFFERS>* masks{};
        bool jittered{};
        std::array<float, 2> jitter{};
        vk::Buffer index_buffer{};
        u64 index_offset{~0ull};
        vk::IndexType index_type{};
        for (const BatchDraw& d : draws) {
            if (d.pipeline != pipeline) {
                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, d.pipeline);
                pipeline = d.pipeline;
            }
            if (d.fixed != fixed || masks == nullptr || *masks != d.masks || d.jittered != jittered || d.jitter != jitter) {
                SendDynamicState(cmd, *d.fixed, pass, d.masks, d.jittered, d.jitter, d.ps_hash, d.color_format0);
                fixed = d.fixed, masks = &d.masks, jittered = d.jittered, jitter = d.jitter;
            }
            cmd.pushConstants(d.layout, VkbVulkan::Vkb::AllGraphicsStageBits, 0, sizeof(d.push), &d.push);
            if (d.index_buffer != index_buffer || d.index_offset != index_offset || d.index_type != index_type) {
                cmd.bindIndexBuffer(d.index_buffer, d.index_offset, d.index_type);
                index_buffer = d.index_buffer, index_offset = d.index_offset, index_type = d.index_type;
            }
            if (d.args_buffer) {
                cmd.drawIndexedIndirect(d.args_buffer, d.args_offset, 1, 20);
            } else {
                cmd.drawIndexed(d.count, d.instances, d.first, d.vertex_offset, 0);
            }
        }
        t_draw_state = {};
        c_batches.fetch_add(1, std::memory_order_relaxed);
        c_batch_draws.fetch_add(draws.size(), std::memory_order_relaxed);
    }
};
thread_local BatchJob* t_batch = nullptr; ///< R7 step D: a plain pointer (owned; enqueued at the flush)
bool BatchMatches(u64 ctx) {
    return t_batch && !t_batch->draws.empty() && t_batch->engine_dsv == Rd<u64>(ctx, CtxDepthStencilView) && t_batch->engine_rtv0 == Rd<u64>(ctx, CtxRenderTargets) &&
           t_batch->engine_dsv != 0 && (!RecordAll() || t_batch->engine_rtvs == RtvHash(ctx));
}
u64 CurrentBatchSerial() {
    return t_batch ? t_batch->serial : 0;
}

/// G1 (scene on the GPU, the CPU part of "raw capture"): a cascade row from per-thread tables by engine object id. An engine id (+0x10) is never used again after its
/// object is deleted, and our answer for an id (its device buffer and window, its view, its sampler, their global-set slots) changes only when that object is deleted;
/// so an entry checked by (id, object pointer, the memory word it was made from) holds for the object's life, without the global memo epochs (which bump ~1.5 times a
/// frame). Only device answers are kept (ring windows move every submission: by arithmetic per draw). Hazards once per object per batch; no shared counters per draw.
/// Switches: TLG_VKB_FAST_ROW (default on since 2026-11-21, =0 off), TLG_VKB_FAST_ROW_VERIFY=1 (both rows made, the old one drawn, differences counted).
struct FastBuf {
    u64 id{}, object{}, buffer{}, memory{};
    u32 gen{};
    u64 epoch{};
    vk::Buffer vk_buffer{};
    u32 slot{}, offset{}, size{};
    u64 use_offset{}, use_size{};
    u64 serial{};
    bool device{};
};
struct FastImg {
    u64 id{}, object{};
    u32 gen{};
    u64 epoch{};
    u32 want{}, key{}, slot{};
    vk::Image image{};
    u64 serial{};
};
struct FastSmp {
    u64 id{}, object{};
    u32 depth{}, slot{};
};
struct FastTables {
    std::vector<FastBuf> bufs = std::vector<FastBuf>(1u << 16);
    std::vector<FastBuf> idx = std::vector<FastBuf>(1u << 15);
    std::vector<FastImg> imgs = std::vector<FastImg>(1u << 14);
    std::vector<FastSmp> smps = std::vector<FastSmp>(1u << 12);
    u64 hits{}, misses{}, rows{}, ring{};
};
thread_local FastTables* t_fast = nullptr;
thread_local bool t_fast_verifying = false;
std::atomic<u64> c_fast_hits{0}, c_fast_misses{0}, c_fast_rows{0}, c_fast_ring{0}, c_fast_verify{0}, c_fast_verify_diff{0}, c_fast_refused{0};
inline FastTables& Fast() {
    if (t_fast == nullptr) {
        t_fast = new FastTables();
    }
    return *t_fast;
}
inline u32 FastHash(u64 id, u64 extra = 0) {
    return static_cast<u32>(((id ^ (extra * 0x9e3779b97f4a7c15ull)) * 0xff51afd7ed558ccdull) >> 40);
}
bool FastRowOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_FAST_ROW"); return e == nullptr || e[0] != '0'; }();
    return on;
}
/// G3 (TLG_VKB_ROW_BY_POINTER=1, off: verified 0 differences in 6.58 M rows, but no fps (106.9 off / 106.2 on, 2026-11-21)): buffer views, images and index buffers found by their pointer in the context, checked by the pointer's
/// generation (native_vkb_objects.h) and the global set's free epoch, without reading the engine objects (each a likely cache miss)
bool RowByPointer() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_ROW_BY_POINTER"); return e != nullptr && e[0] == '1'; }();
    return on;
}
bool FastRowVerify() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_FAST_ROW_VERIFY"); return e != nullptr && e[0] == '1'; }();
    return on;
}
void FastPublish(FastTables& f) {
    if ((++f.rows & 1023) == 0) {
        c_fast_rows.fetch_add(1024, std::memory_order_relaxed);
        c_fast_hits.fetch_add(std::exchange(f.hits, 0), std::memory_order_relaxed);
        c_fast_misses.fetch_add(std::exchange(f.misses, 0), std::memory_order_relaxed);
        c_fast_ring.fetch_add(std::exchange(f.ring, 0), std::memory_order_relaxed);
    }
}
bool FastRow(u64 ctx, const Stage& st, u32* row, boost::container::small_vector<Stream::Use, 64>& uses, u64 serial) {
    using BT = VkbShader::BufferType;
    FastTables& f = Fast();
    const auto& info = *st.info;
    u32 at = st.row_start;
    for (u32 i = 0; i < st.buffers.size(); ++i) {
        const Source& src = st.buffers[i];
        const auto& desc = info.buffers[i];
        if (desc.buffer_type == BT::GdsBuffer) {
            row[at++] = Core::Vkb::Indexed::BufferSlot(Stream::GdsBuffer());
            continue;
        }
        if (desc.buffer_type == BT::ClipPlanes) {
            row[at++] = Core::Vkb::Indexed::BufferSlot(Stream::NullBuffer());
            continue;
        }
        if (desc.buffer_type != BT::Guest || src.cls != Source::Class::Slot || (src.kind != 1 && src.kind != 3)) {
            return false;
        }
        const u64 object = SlotObject(ctx, st.index, src.kind, src.slot);
        if (object == 0) {
            return false;
        }
        if (src.kind == 3) { // a constant buffer: renamed by the engine at every update, so its current data per draw (as the old row)
            const u64 address = static_cast<u64>(Rd<u32>(object, 0x60)) | static_cast<u64>(Rd<u32>(object, 0x64) & 0xffu) << 32;
            const u64 size = Rd<u32>(object, 0x180);
            if (address == 0 || size == 0) {
                return false;
            }
            Core::Vkb::BufferWindow w;
            if (!Core::Vkb::ConstantInRing(address, size, w)) {
                if (size > 65536) {
                    return false;
                }
                if (Core::Vkb::R7NoCopy()) { // bound / bisect (TLG_VKB_R7_NO_CB_COPY=1): in place, not copied
                    if (!Core::Vkb::FindBuffer(address, size, w) || !w.buffer || (w.offset & 3) != 0) {
                        return false;
                    }
                    row[at++] = Core::Vkb::Indexed::BufferSlot(w.buffer);
                    row[at++] = static_cast<u32>(w.offset);
                    row[at++] = static_cast<u32>(size);
                    uses.push_back(Stream::BufferUse(w.buffer, w.offset, size, false));
                    continue;
                }
                if (t_fast_verifying) { // (verify: the old row made its own copy; this entry is not compared)
                    row[at++] = ~0u, row[at++] = ~0u, row[at++] = ~0u;
                    continue;
                }
                const Stream::Slice slice = RowSlice(size);
                std::memcpy(slice.data, reinterpret_cast<const void*>(address), size);
                w.buffer = slice.buffer, w.offset = slice.offset, w.device = false;
            }
            if (!w.buffer || (w.offset & 3) != 0) {
                return false;
            }
            row[at++] = Core::Vkb::Indexed::BufferSlot(w.buffer);
            row[at++] = static_cast<u32>(w.offset);
            row[at++] = static_cast<u32>(size);
            uses.push_back(Stream::BufferUse(w.buffer, w.offset, size, false));
            continue;
        }
        // a buffer view
        const bool by_pointer = RowByPointer();
        const u32 gen = by_pointer ? Core::Vkb::ObjectPointerGen(object) : 0;
        const u64 epoch = by_pointer ? Core::Vkb::Indexed::Epoch() : 0;
        FastBuf* hit = nullptr;
        if (by_pointer) {
            FastBuf& p = f.bufs[FastHash(object) & (f.bufs.size() - 1)];
            if (p.object == object && p.gen == gen && p.epoch == epoch && p.vk_buffer) {
                hit = &p;
            }
        }
        const u64 id = hit != nullptr ? hit->id : Rd<u64>(object, 0x10);
        const u64 buffer = hit != nullptr ? hit->buffer : Rd<u64>(object, 0x30);
        const u64 memory = hit != nullptr ? hit->memory : (buffer != 0 ? Rd<u64>(buffer, 0x30) : 0);
        FastBuf& e = hit != nullptr ? *hit : f.bufs[FastHash(by_pointer ? object : id) & (f.bufs.size() - 1)];
        if (hit != nullptr || (!by_pointer && e.id == id && e.object == object && e.buffer == buffer && e.memory == memory && e.vk_buffer)) {
            ++f.hits;
        } else {
            ++f.misses;
            if (Rd<u32>(object, 0x44) != 1 && Rd<u32>(object, 0x44) != 0xb) {
                return false;
            }
            const u64 bytes = buffer != 0 ? Rd<u32>(buffer, 0x180) : 0;
            u64 offset = 0, size = bytes;
            Core::Vkb::ViewRange(object, true, buffer, bytes, offset, size);
            Core::Vkb::BufferWindow w;
            if (bytes == 0) {
                return false;
            }
            if (Core::Vkb::BufferInRing(memory + offset, size, w)) { // ring data moves every submission: by arithmetic, not kept
                ++f.ring;
                if (!w.buffer || (w.offset & 3) != 0) {
                    return false;
                }
                row[at++] = Core::Vkb::Indexed::BufferSlot(w.buffer);
                row[at++] = static_cast<u32>(w.offset);
                row[at++] = static_cast<u32>(size);
                uses.push_back(Stream::BufferUse(w.buffer, w.offset, size, false));
                continue;
            }
            if (!Core::Vkb::BufferOfObject(buffer, offset, size, w) || !w.device || !w.buffer || (w.offset & 3) != 0) {
                return false;
            }
            const u32 slot = Core::Vkb::Indexed::BufferSlot(w.buffer);
            if (slot == 0) {
                return false;
            }
            e = {id, object, buffer, memory, gen, epoch, w.buffer, slot, static_cast<u32>(w.offset), static_cast<u32>(size), w.offset, size, 0, true};
        }
        row[at++] = e.slot;
        row[at++] = e.offset;
        row[at++] = e.size;
        if (e.serial != serial) { // its hazard once per batch
            uses.push_back(Stream::BufferUse(e.vk_buffer, e.use_offset, e.use_size, false));
            e.serial = serial;
        }
    }
    for (u32 i = 0; i < st.images.size(); ++i) {
        const Source& src = st.images[i];
        const u64 object = src.cls == Source::Class::Slot && src.kind == 1 ? SlotObject(ctx, st.index, 1, src.slot) : 0;
        if (object == 0) {
            return false;
        }
        const u32 want = static_cast<u32>(st.image_types[i]);
        const u32 key = info.images[i].array_key;
        const bool written = info.images[i].is_written;
        const bool by_pointer = RowByPointer();
        const u32 gen = by_pointer ? Core::Vkb::ObjectPointerGen(object) : 0;
        const u64 epoch = by_pointer ? Core::Vkb::Indexed::Epoch() : 0;
        FastImg& e = f.imgs[FastHash(object, want | key << 8) & (f.imgs.size() - 1)];
        if (e.object == object && e.want == want && e.key == key && e.slot != 0 && (by_pointer ? (e.gen == gen && e.epoch == epoch) : e.id == Rd<u64>(object, 0x10))) {
            ++f.hits;
        } else {
            ++f.misses;
            const u64 id = Rd<u64>(object, 0x10);
            vk::ImageView view{};
            vk::Image image{};
            if (!Core::Vkb::ViewForCompute(id, st.image_types[i], vk::CommandBuffer{}, view, &image)) {
                return false;
            }
            const u32 slot = Core::Vkb::Indexed::ImageSlot(key, view);
            if (slot == 0) {
                return false;
            }
            e = {id, object, gen, epoch, want, key, slot, image, 0};
        }
        row[at++] = e.slot;
        if (written || e.serial != serial) {
            uses.push_back(Stream::ImageUse(e.image, written));
            e.serial = serial;
        }
    }
    for (u32 i = 0; i < st.samplers.size(); ++i) {
        const bool is_depth_s = i < info.samplers.size() && info.samplers[i].is_depth;
        const Core::Vkb::Aa::SamplerLook own = Core::Vkb::Aa::ForSampler(t_row_look, is_depth_s);
        if (st.constant_samplers[i]) {
            row[at++] = Core::Vkb::Indexed::SamplerSlot(ConstantSamplerFor(st, i, is_depth_s, own));
            continue;
        }
        const Source& src = st.samplers[i];
        const u64 object = src.cls == Source::Class::Slot && src.kind == 2 ? SlotObject(ctx, st.index, 2, src.slot) : 0;
        if (object == 0) {
            return false;
        }
        const u64 id = Rd<u64>(object, 0x10);
        const u32 depth = i < info.samplers.size() && info.samplers[i].is_depth ? 1 : 0;
        const u32 look_key = static_cast<u32>(static_cast<u16>(own.bias_q)) << 16 | (own.aniso & 0x7fffu) << 1 | depth;
        FastSmp& e = f.smps[FastHash(id, look_key + 1) & (f.smps.size() - 1)];
        if (e.id == id && e.object == object && e.depth == look_key && e.slot != 0) {
            ++f.hits;
        } else {
            ++f.misses;
            const vk::Sampler sampler = Core::Vkb::SamplerFor(object, depth != 0, own.bias_q, own.aniso);
            const u32 slot = sampler ? Core::Vkb::Indexed::SamplerSlot(sampler) : 0;
            if (slot == 0) {
                return false;
            }
            e = {id, object, look_key, slot};
        }
        row[at++] = e.slot;
    }
    return true;
}
/// G1: the index buffer of a cascade draw from the table by the index buffer's id (its device or in-place window from its start; the range of this draw on top)
bool FastIndex(u64 view, u64 index_range, Core::Vkb::BufferWindow& w, boost::container::small_vector<Stream::Use, 64>& uses, u64 serial) {
    FastTables& f = Fast();
    const bool by_pointer = RowByPointer();
    const u32 gen = by_pointer ? Core::Vkb::ObjectPointerGen(view) : 0;
    const u64 epoch = by_pointer ? Core::Vkb::Indexed::Epoch() : 0;
    FastBuf& e = f.idx[FastHash(view, 3) & (f.idx.size() - 1)];
    if (by_pointer && e.object == view && e.gen == gen && e.epoch == epoch && e.vk_buffer && index_range <= e.use_size) {
        ++f.hits;
        w.buffer = e.vk_buffer, w.offset = e.use_offset, w.device = e.device;
        if (e.serial != serial) {
            uses.push_back(Stream::BufferUse(e.vk_buffer, e.use_offset, e.use_size, false));
            e.serial = serial;
        }
        return true;
    }
    const u64 id = Rd<u64>(view, 0x10);
    const u64 memory = Rd<u64>(view, 0x30);
    const u64 bytes = Rd<u32>(view, 0x180);
    if (!by_pointer && e.id == id && e.object == view && e.memory == memory && e.vk_buffer && index_range <= e.use_size) {
        ++f.hits;
    } else {
        ++f.misses;
        if (Core::Vkb::BufferInRing(memory, index_range, w)) { // ring data moves every submission: not kept
            uses.push_back(Stream::BufferUse(w.buffer, w.offset, index_range, false));
            return true;
        }
        Core::Vkb::BufferWindow whole;
        if (!Core::Vkb::BufferOfObject(view, 0, index_range, whole) || !whole.buffer) {
            return false;
        }
        if (!whole.device) { // in place: its device copy may come at a later bind (R3d): not kept
            w = whole;
            uses.push_back(Stream::BufferUse(w.buffer, w.offset, index_range, false));
            return true;
        }
        e = {id, view, view, memory, gen, epoch, whole.buffer, 0, 0, 0, whole.offset, std::max<u64>(bytes, index_range), 0, true};
    }
    w.buffer = e.vk_buffer, w.offset = e.use_offset, w.device = e.device;
    if (e.serial != serial) {
        uses.push_back(Stream::BufferUse(e.vk_buffer, e.use_offset, e.use_size, false));
        e.serial = serial;
    }
    return true;
}
void FlushBatchHere();
/// G2: a batch is cut after this many draws (TLG_VKB_BATCH_MAX, default 128; 0 = only at the pass's end) so the recording thread starts on a pass while the engine still
/// draws it: a whole pass in one batch reaches the GPU late, and the engine waits for the GPU after every dispatch (note 33)
inline void BatchCutIfFull() {
    static const u32 max = [] { const char* e = std::getenv("TLG_VKB_BATCH_MAX"); return e != nullptr ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 128u; }();
    if (max != 0 && t_batch != nullptr && t_batch->draws.size() >= max) {
        t_flush_reason = 1;
        FlushBatchHere();
    }
}
bool LeanAppend(u64 ctx, Variant& v, const VkbVulkan::FixedState& fixed, u32 kind, u64 a1, u64 a2, u64 a3) {
    // issue 134: with AA on, the lean path takes the job path's jitter decision (AaDrawJitter, same game order); before, AA sent every draw down the job path (DLAA
    // and FSR 3.1 condor 112 fps against 120 with AA off, the GPU half idle). The motion pass and SMAA stay on the job path (their recording-time notes).
    // TLG_VKB_AA_LEAN=0: the old rule (no lean draws with AA on)
    static const bool aa_lean = [] { const char* e = std::getenv("TLG_VKB_AA_LEAN"); return e == nullptr || e[0] != '0'; }();
    const bool aa = Core::Vkb::Aa::On();
    if (!BatchMatches(ctx) || (v.key.num_color_attachments != 0 && !RecordAll()) || (kind != 0 && kind != 2)) {
        return false;
    }
    if (aa && (!aa_lean || (v.has_ps && v.ps.info && v.ps.info->pgm_hash == Core::Vkb::Aa::MotionPassPs) || (v.vs.info && v.vs.info->pgm_hash == Core::Vkb::Aa::SmaaVs))) {
        return false;
    }
    const u64 serial = t_batch->serial;
    // step E bound, kill experiment only (TLG_VKB_TEST_LEAN_FREE=1, the picture is wrong): the draw reuses the previous draw's row and index buffer (no row, no lookups)
    static const bool lean_free = [] { const char* e = std::getenv("TLG_VKB_TEST_LEAN_FREE"); return e != nullptr && e[0] == '1'; }();
    if (lean_free && !t_batch->draws.empty()) {
        BatchDraw d = t_batch->draws.back();
        d.pipeline = v.pipeline->Handle();
        d.layout = v.pipeline->GetLayout();
        d.fixed = &fixed;
        if (kind == 0) {
            d.count = static_cast<u32>(a1), d.first = static_cast<u32>(a2), d.vertex_offset = static_cast<s32>(static_cast<u32>(a3)), d.args_buffer = vk::Buffer{};
        }
        t_batch->draws.push_back(d);
        return true;
    }
    BatchDraw d{};
    d.pipeline = v.pipeline->Handle();
    d.layout = v.pipeline->GetLayout();
    d.fixed = &fixed;
    d.masks = v.key.write_masks;
    d.push.xoffset = fixed.transform.x_offset;
    d.push.xscale = fixed.transform.x_scale;
    d.push.yoffset = fixed.transform.y_offset;
    d.push.yscale = fixed.transform.y_scale;
    d.ps_hash = v.has_ps && v.ps.info ? v.ps.info->pgm_hash : 0ull;
    d.color_format0 = static_cast<u32>(v.color_formats[0]);
    d.instances = std::max(Rd<u32>(ctx, 0x4724), 1u);
    boost::container::small_vector<Stream::Use, 64> uses;
    const Stream::Slice slice = RowSlice(std::max<u32>(v.row_dwords, 1) * 4);
    u32* row = reinterpret_cast<u32*>(slice.data);
    t_row_look = Core::Vkb::Aa::LookFor(v.key.mrt_mask); // issue 42 (merge): as the plain path's samplers
    static const bool lean_times = [] { const char* e = std::getenv("TLG_VKB_LEAN_TIMES"); return e != nullptr && e[0] == '1'; }();
    const u64 r0 = lean_times ? NowNs() : 0;
    if (FastRowOn() && FastRowVerify()) { // G1 verify: both rows; the old one is drawn
        std::array<u32, 1024> fast_row{};
        boost::container::small_vector<Stream::Use, 64> fast_uses;
        t_fast_verifying = true;
        const bool fast_ok = v.row_dwords <= fast_row.size() && (!v.has_ps || FastRow(ctx, v.ps, fast_row.data(), fast_uses, ~serial)) && FastRow(ctx, v.vs, fast_row.data(), fast_uses, ~serial);
        t_fast_verifying = false;
        if ((v.has_ps && !BuildRow(ctx, v.ps, row, uses, serial)) || !BuildRow(ctx, v.vs, row, uses, serial)) {
            return false;
        }
        c_fast_verify.fetch_add(1, std::memory_order_relaxed);
        bool same = fast_ok;
        for (u32 i = 0; same && i < v.row_dwords; ++i) {
            same = fast_row[i] == ~0u || fast_row[i] == row[i];
        }
        if (!same) {
            static std::atomic<u32> shown{0};
            if (shown.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::string a, b;
                for (u32 i = 0; i < std::min<u32>(v.row_dwords, 24); ++i) {
                    a += fmt::format(" {:x}", fast_row[i]), b += fmt::format(" {:x}", row[i]);
                }
                LOG_WARNING(Render_Vulkan, "G1 verify: fast row {} differs ({} dwords): fast{} / old{}", fast_ok ? "made" : "refused", v.row_dwords, a, b);
            }
            c_fast_verify_diff.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (FastRowOn()) { // G1: rows from the tables by engine id
        if ((v.has_ps && !FastRow(ctx, v.ps, row, uses, serial)) || !FastRow(ctx, v.vs, row, uses, serial)) {
            c_fast_refused.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        FastPublish(Fast());
    } else if ((v.has_ps && !BuildRow(ctx, v.ps, row, uses, serial)) || !BuildRow(ctx, v.vs, row, uses, serial)) {
        return false;
    }
    if (lean_times) {
        c_lean_row_ns.fetch_add(NowNs() - r0, std::memory_order_relaxed);
    }
    d.push.row_base = static_cast<u32>(slice.offset / 4);
    // the index buffer (the engine's index view, its own memory from its start, R7 / note 56)
    const u64 view = Rd<u64>(ctx, 0x58);
    { // research (TLG_VKB_ID_CENSUS=1): are the engine's object ids dense (a flat table by id instead of hash lookups)?
        static const bool id_census = [] { const char* e = std::getenv("TLG_VKB_ID_CENSUS"); return e != nullptr && e[0] == '1'; }();
        if (id_census) {
            static std::atomic<u64> lo{~0ull}, hi{0}, n{0};
            const u64 vb = SlotObject(ctx, 0, 1, 0);
            const u64 vbuf = vb != 0 ? Rd<u64>(vb, 0x30) : 0;
            for (const u64 id : {view != 0 ? Rd<u64>(view, 0x10) : 0, vb != 0 ? Rd<u64>(vb, 0x10) : 0, vbuf != 0 ? Rd<u64>(vbuf, 0x10) : 0}) {
                if (id == 0) {
                    continue;
                }
                u64 cur = lo.load();
                while (id < cur && !lo.compare_exchange_weak(cur, id)) {
                }
                cur = hi.load();
                while (id > cur && !hi.compare_exchange_weak(cur, id)) {
                }
            }
            if ((n.fetch_add(1) & 0xfffff) == 0) {
                LOG_WARNING(Render_Vulkan, "id census: index view id {} (object {:#x}), vertex view id {}, its buffer id {}; ids seen {}..{}", view != 0 ? Rd<u64>(view, 0x10) : 0, view,
                            vb != 0 ? Rd<u64>(vb, 0x10) : 0, vbuf != 0 ? Rd<u64>(vbuf, 0x10) : 0, lo.load(), hi.load());
            }
        }
    }
    const u32 type = Rd<u32>(ctx, 0x64);
    if (view == 0 || Rd<u32>(ctx, 0x60) != 0x39 || type > 1) {
        return false;
    }
    const u64 index_bytes = type == 1 ? 4 : 2;
    const u64 index_range = kind == 0 ? (a2 + a1) * index_bytes : static_cast<u64>(Rd<u32>(view, 0x68)) * index_bytes;
    Core::Vkb::BufferWindow w;
    if (FastRowOn() && !FastRowVerify()) { // G1: the index buffer from its table by id
        if (index_range == 0 || !FastIndex(view, index_range, w, uses, serial) || (w.offset % index_bytes) != 0) {
            return false;
        }
    } else {
        if (index_range == 0 || !Core::Vkb::BufferOfObject(view, 0, index_range, w) || (w.offset % index_bytes) != 0) {
            return false;
        }
        if (FastRowOn()) { // G1 verify: the index buffer too
            Core::Vkb::BufferWindow fw;
            boost::container::small_vector<Stream::Use, 64> fast_uses;
            if (!FastIndex(view, index_range, fw, fast_uses, ~serial) || fw.buffer != w.buffer || fw.offset != w.offset || fw.device != w.device) {
                c_fast_verify_diff.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<u32> shown{0};
                if (shown.fetch_add(1, std::memory_order_relaxed) < 8) {
                    LOG_WARNING(Render_Vulkan, "G1 verify: index buffer differs: fast {:x}+{} dev {} / old {:x}+{} dev {}", reinterpret_cast<u64>(static_cast<VkBuffer>(fw.buffer)), fw.offset, fw.device,
                                reinterpret_cast<u64>(static_cast<VkBuffer>(w.buffer)), w.offset, w.device);
                }
            }
        }
        uses.push_back(Stream::BufferUse(w.buffer, w.offset, index_range, false));
    }
    d.index_buffer = w.buffer, d.index_offset = w.offset, d.index_size = w.device ? 0 : index_range;
    d.index_type = type == 1 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
    if (kind == 2) {
        Core::Vkb::BufferWindow aw;
        if (a1 == 0 || !IsHeapPointer(a1) || !Core::Vkb::BufferOfObject(a1, static_cast<u32>(a2), 20, aw) || (aw.offset & 3) != 0) {
            return false;
        }
        d.args_buffer = aw.buffer, d.args_offset = aw.offset;
        uses.push_back(Stream::BufferUse(aw.buffer, aw.offset, 20, false));
    } else {
        d.count = static_cast<u32>(a1), d.first = static_cast<u32>(a2), d.vertex_offset = static_cast<s32>(static_cast<u32>(a3));
    }
    if (!t_batch || t_batch->serial != serial) { // a job went into the stream while the row was made (it flushed the batch): the generic path
        return false;
    }
    if (aa) [[unlikely]] { // issue 134: decided last, so a refused draw is decided once (by the job path)
        d.jittered = AaDrawJitter(ctx, v, fixed, kind, a1, a2, a3, d.instances, d.jitter);
    }
    t_batch->uses.insert(t_batch->uses.end(), uses.begin(), uses.end());
    t_batch->draws.push_back(d);
    c_lean.fetch_add(1, std::memory_order_relaxed);
    BatchCutIfFull();
    return true;
}
thread_local bool t_batch_flushing = false;
std::atomic<u64> c_cut_by_work[16]{};
void FlushBatchHere();
void FlushBeforeJob(u32 work) {
    if (t_batch && !t_batch->draws.empty() && !t_batch_flushing) {
        c_cut_by_work[std::min<u32>(work, 15)].fetch_add(1, std::memory_order_relaxed);
    }
    FlushBatchHere();
}
void FlushBatchHere() {
    if (t_batch_flushing || !t_batch || t_batch->draws.empty()) {
        return;
    }
    t_batch_flushing = true;
    c_flush_by[std::min<u32>(t_flush_reason, 2)].fetch_add(1, std::memory_order_relaxed);
    t_flush_reason = 0;
    t_batch->work = Stream::Work::Draw;
    t_batch->weight = static_cast<u32>(std::max<size_t>(t_batch->draws.size(), 1)); // G2: a batch counts as its draws in the submission split
    std::unique_ptr<Stream::Job> job{t_batch};
    t_batch = nullptr;
    Stream::EnqueueNoHook(std::move(job));
    t_batch_flushing = false;
}
bool BatchOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_VKB_RECORD_BATCH");
        const bool want = e == nullptr || e[0] != '0';
        if (want) {
            Stream::SetBeforeEnqueue(&FlushBeforeJob);
        }
        return want;
    }();
    return on;
}
/// The draw job of a record-mode indexed draw joins the open batch (false: it goes alone)
bool JoinBatch(DrawJob& job) {
    if ((job.kind != 0 && (job.kind != 2 || !job.args_buffer)) || !job.dump_views.empty() || job.debug_id != 0 || job.vs_hash == Core::Vkb::Aa::SmaaVs || !job.index_buffer) {
        return false;
    }
    if (t_batch && !(t_batch->pass == job.pass)) {
        t_flush_reason = 1;
        FlushBatchHere();
    }
    if (!t_batch) {
        static std::atomic<u64> serials{1};
        t_batch = new BatchJob;
        t_batch->serial = serials.fetch_add(1, std::memory_order_relaxed);
        t_batch->engine_dsv = t_join_dsv, t_batch->engine_rtv0 = t_join_rtv0, t_batch->engine_rtvs = t_join_rtvs;
        t_batch->pass = job.pass;
        t_batch->attachment_uses = job.attachment_uses;
        t_batch->draws.reserve(1024);
        t_batch->uses.reserve(2048);
    }
    t_batch->uses.insert(t_batch->uses.end(), job.uses.begin(), job.uses.end());
    t_batch->draws.push_back({.pipeline = job.pipeline, .layout = job.layout, .fixed = job.fixed_state, .masks = job.masks, .push = job.push,
                              .index_buffer = job.index_buffer, .index_offset = job.index_offset, .index_type = job.index_type, .index_size = job.index_size, .args_buffer = job.kind == 2 ? job.args_buffer : vk::Buffer{}, .args_offset = job.args_offset, .count = static_cast<u32>(job.a1),
                              .first = static_cast<u32>(job.a2), .instances = job.instances, .vertex_offset = static_cast<s32>(static_cast<u32>(job.a3)), .ps_hash = job.ps_hash,
                              .color_format0 = job.color_format0, .jittered = job.jittered, .jitter = job.jitter});
    return true;
}

void DrawJob::Run(vk::CommandBuffer cmd) {
    const VkbVulkan::FixedState& fixed = *fixed_state;
    struct JobTime {
        u64 t0;
        ~JobTime() {
            if (t0 != 0) {
                c_job_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
                c_job_n.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } job_time{TimesOn() ? NowNs() : 0};
    if (prepared ? !prepared_ok : (AnyLate() && !ResolveLate())) { // R2b (note 64): a buffer not found: the draw is skipped, as at the capture before
        return;
    }
    // note 65 bound, kill experiment only (TLG_VKB_TEST_NO_DRAW_COMMANDS=1, nothing is drawn): the draw job resolves, then records nothing (no barrier, no pass,
    // no state, no draw): what the recording thread's own work for draws costs the frame
    static const bool no_commands = [] { const char* e = std::getenv("TLG_VKB_TEST_NO_DRAW_COMMANDS"); return e != nullptr && e[0] == '1'; }();
    if (no_commands) {
        return;
    }
    // V4h: engine memory read in place gets a copy in our device ring in front of the submission
    for (u32 i = 0; i < b.buffers.size() && i < b.stageable.size(); ++i) {
        if (b.stageable[i] != 0) {
            (void)Stream::StageRead(b.buffers[i].buffer, b.buffers[i].offset, b.buffers[i].range);
        }
    }
    for (u32 i = 0; i < vertex_buffers.size() && i < vertex_sizes.size(); ++i) {
        if (vertex_sizes[i] != 0) {
            (void)Stream::StageRead(vertex_buffers[i], vertex_offsets[i], vertex_sizes[i]);
        }
    }
    if (index_size != 0 && index_buffer) {
        (void)Stream::StageRead(index_buffer, index_offset, index_size);
    }
    // V4d: a barrier only when this draw reads what earlier work since the last barrier wrote (that ends the open pass; it begins again below), or its pass's attachments need one
    static const bool no_draw_barrier = [] { const char* e = std::getenv("TLG_VKB_NO_DRAW_BARRIER"); return e != nullptr && e[0] == '1'; }(); // R5 kill-experiment (note 32)
    if (!Stream::LegacyBarriersOn() && !no_draw_barrier) {
        Stream::SetHazardTag(ps_hash);
        (void)Stream::Hazard(cmd, std::span<const Stream::Use>{uses.data(), uses.size()});
    }
    if (Core::Vkb::Aa::On() && ps_hash == Core::Vkb::Aa::MotionPassPs) [[unlikely]] { // F1: the scene colour, the game's motion and the depth of this frame
        boost::container::small_vector<vk::Image, 9> images;
        for (const Stream::Use& u : attachment_uses) {
            images.push_back(vk::Image{reinterpret_cast<VkImage>(u.handle)});
        }
        Core::Vkb::Aa::NoteMotionPass(std::span<const vk::Image>{images.data(), images.size()}, pass.color_count, pass.depth ? true : false);
    }
    if (Stream::UsePass(cmd, pass, std::span<const Stream::Use>{attachment_uses.data(), attachment_uses.size()})) {
        c_new_passes.fetch_add(1, std::memory_order_relaxed);
    }
    // note 60: what this recording already holds (a D3D11 driver's redundant-state filter): the pipeline, the dynamic state, the vertex input, the vertex and index
    // buffers are sent only when they differ from the previous draw's in the same command buffer. TLG_VKB_ALL_STATE=1 sends everything every draw, as before.
    DrawStateCache& state = t_draw_state;
    static const bool all_state = [] { const char* e = std::getenv("TLG_VKB_ALL_STATE"); return e != nullptr && e[0] == '1'; }();
    if (all_state || state.generation != Stream::RecordingGeneration() || state.cmd != static_cast<VkCommandBuffer>(cmd)) {
        state = {};
        state.generation = Stream::RecordingGeneration();
        state.cmd = static_cast<VkCommandBuffer>(cmd);
    }
    if (state.pipeline != static_cast<VkPipeline>(pipeline)) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        state.pipeline = static_cast<VkPipeline>(pipeline);
        c_state_pipelines.fetch_add(1, std::memory_order_relaxed);
    }
    // R3: a record-mode draw binds the global set (once while the command buffer's set 0 is it) and only pushes its row base
    if (record) {
        if (t_global_cmd != static_cast<VkCommandBuffer>(cmd) || t_global_generation != Stream::RecordingGeneration()) {
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, Core::Vkb::Indexed::Set(), {});
            t_global_cmd = static_cast<VkCommandBuffer>(cmd), t_global_generation = Stream::RecordingGeneration();
        }
    }
    // A2 (note 46): the descriptor writes are made here, on the recording thread, in its own reused list (they were 4 KB of every draw's job)
    thread_local boost::container::small_vector<vk::WriteDescriptorSet, 64> writes;
    writes.clear();
    u32 binding = 0;
    for (u32 r = 0; r < ranges.size(); r += 3) {
        for (u32 k = 0; k < ranges[r].second; ++k) {
            writes.push_back({.dstBinding = binding++, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &b.buffers[ranges[r].first + k]});
        }
        for (u32 k = 0; k < ranges[r + 1].second; ++k) {
            writes.push_back({.dstBinding = binding++, .descriptorCount = 1,
                                .descriptorType = image_types[ranges[r + 1].first + k],
                                .pImageInfo = &b.images[ranges[r + 1].first + k]});
        }
        for (u32 k = 0; k < ranges[r + 2].second; ++k) {
            writes.push_back({.dstBinding = binding++, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler, .pImageInfo = &b.samplers[ranges[r + 2].first + k]});
        }
    }
    if (!writes.empty() && push_descriptors) {
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, layout, 0, writes);
        t_global_cmd = {}; // (set 0 is a push set now: the next record-mode draw binds the global set again)
    } else if (!writes.empty()) {
        // note 41: more bindings than push descriptors allow (the layout is a plain one, as the old path's): a descriptor set of ours, written now
        const vk::DescriptorSet set = BigDescriptorSet(set_layout);
        if (set) {
            for (auto& write : writes) {
                write.dstSet = set;
            }
            instance->GetDevice().updateDescriptorSets(writes, {});
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, set, {});
            t_global_cmd = {};
        }
    }
    cmd.pushConstants(layout, VkbVulkan::Vkb::AllGraphicsStageBits, 0, sizeof(push), &push);
    // note 59 bound, kill experiment only (TLG_VKB_TEST_STATE_ONCE=1, the picture is wrong): the dynamic state sent once per command buffer, not every draw
    static const bool state_once = [] { const char* e = std::getenv("TLG_VKB_TEST_STATE_ONCE"); return e != nullptr && e[0] == '1'; }();
    thread_local VkCommandBuffer t_state_cmd{};
    const bool first_in_cmd = t_state_cmd != static_cast<VkCommandBuffer>(cmd);
    t_state_cmd = static_cast<VkCommandBuffer>(cmd);
    // note 60: the dynamic state comes from the interned fixed state, the pass's depth / stencil / colour count and the write masks (the two note 41 test switches
    // aside): when all of them equal the previous draw's in this recording, nothing is sent
    const u64 pass_key = (pass.depth ? 1ull : 0ull) | (pass.depth_has_stencil ? 2ull : 0ull) | static_cast<u64>(pass.color_count) << 2; // (the note 41 test switches need TLG_VKB_ALL_STATE=1)
    const bool same_state = state.fixed == fixed_state && state.pass_key == pass_key && state.masks == masks && state.jittered == jittered && state.jitter == jitter;
    const bool send_state = state_once ? first_in_cmd : (!same_state || !Stream::StampsLean());
    state.fixed = fixed_state, state.pass_key = pass_key, state.masks = masks, state.jittered = jittered, state.jitter = jitter;
    (same_state ? c_state_skipped : c_state_sent).fetch_add(1, std::memory_order_relaxed);
    if (send_state) {
        SendDynamicState(cmd, fixed, pass, masks, jittered, jitter, ps_hash, color_format0);
    }
    const auto same_span = [](const auto& a, const auto& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0; };
    if (!state.vertex_input || !same_span(state.bindings, input_bindings) || !same_span(state.attributes, input_attributes)) {
        cmd.setVertexInputEXT(input_bindings, input_attributes);
        state.bindings.assign(input_bindings.begin(), input_bindings.end());
        state.attributes.assign(input_attributes.begin(), input_attributes.end());
        state.vertex_input = true;
        c_state_vertex_inputs.fetch_add(1, std::memory_order_relaxed);
    }
    if (!vertex_buffers.empty() && (!same_span(state.vertex_buffers, vertex_buffers) || !same_span(state.vertex_offsets, vertex_offsets))) {
        cmd.bindVertexBuffers(0, vertex_buffers, vertex_offsets);
        state.vertex_buffers.assign(vertex_buffers.begin(), vertex_buffers.end());
        state.vertex_offsets.assign(vertex_offsets.begin(), vertex_offsets.end());
    }
    static const bool no_draw = [] { const char* e = std::getenv("TLG_VKB_NO_DRAW"); return e != nullptr && *e == '1'; }(); // kill experiment: everything but the draw
    if (no_draw) {
    } else if (vs_hash == Core::Vkb::Aa::SmaaVs && Core::Vkb::Aa::Running()) [[unlikely]] {
        // F1: the native AA replaces the game's SMAA: its edge weights are cleared instead of drawn (their SkipGameSmaa)
        const vk::ClearAttachment clear{vk::ImageAspectFlagBits::eColor, 0, vk::ClearColorValue{std::array{0.0f, 0.0f, 0.0f, 0.0f}}};
        cmd.clearAttachments(clear, vk::ClearRect{vk::Rect2D{{0, 0}, pass.extent}, 0, 1});
    } else if (kind == 0) {
        if (state.index_buffer != static_cast<VkBuffer>(index_buffer) || state.index_offset != index_offset || state.index_type != index_type) {
            cmd.bindIndexBuffer(index_buffer, index_offset, index_type);
            state.index_buffer = static_cast<VkBuffer>(index_buffer), state.index_offset = index_offset, state.index_type = index_type;
        }
        cmd.drawIndexed(static_cast<u32>(a1), instances, static_cast<u32>(a2), static_cast<s32>(first_vertex), 0);
    } else if (kind == 2) {
        if (state.index_buffer != static_cast<VkBuffer>(index_buffer) || state.index_offset != index_offset || state.index_type != index_type) {
            cmd.bindIndexBuffer(index_buffer, index_offset, index_type);
            state.index_buffer = static_cast<VkBuffer>(index_buffer), state.index_offset = index_offset, state.index_type = index_type;
        }
        cmd.drawIndexedIndirect(args_buffer, args_offset, 1, 20);
    } else {
        cmd.draw(static_cast<u32>(a1), instances, first_vertex, 0);
    }
    if (debug_id != 0) [[unlikely]] { // note 91: the draw's GPU time, and its row in the frame capture
        Stream::StampDraw(debug_id, vs_hash, ps_hash, kind == 2 ? 0u : static_cast<u32>(a1), instances);
        if (capture) {
            nlohmann::json images = nlohmann::json::array();
            for (const auto& im : b.images) {
                images.push_back(fmt::format("{:#x}", reinterpret_cast<u64>(static_cast<VkImageView>(im.imageView))));
            }
            std::string push_hex;
            const auto* pb = reinterpret_cast<const u8*>(&push);
            for (size_t i = 0; i < sizeof(push); ++i) {
                push_hex += fmt::format("{:02x}", pb[i]);
            }
            nlohmann::json row{{"id", debug_id}, {"vs", fmt::format("{:016x}", vs_hash)}, {"ps", fmt::format("{:016x}", ps_hash)},
                               {"kind", kind == 0 ? "indexed" : kind == 1 ? "auto" : "indirect"}, {"count", a1}, {"first", a2}, {"instances", instances},
                               {"targets", pass.color_count}, {"size", fmt::format("{}x{}", pass.extent.width, pass.extent.height)}, {"depth", pass.depth ? true : false},
                               {"images", images}, {"push_data_hex", push_hex}};
            (void)Stream::NoteCapturedDraw(std::move(row), std::span<const vk::DescriptorBufferInfo>{b.buffers.data(), b.buffers.size()});
        }
    }
    if (!dump_views.empty() || dump_buffers) {
        Stream::BreakPass(cmd);
        for (const auto& [id, name] : dump_views) {
            Core::Vkb::DumpView(cmd, id, name);
        }
        for (u32 i = 0; dump_buffers && i < b.buffers.size(); ++i) {
            if (b.buffers[i].range != VK_WHOLE_SIZE) {
                Core::Vkb::DumpBuffer(cmd, b.buffers[i].buffer, b.buffers[i].offset, b.buffers[i].range, fmt::format("{}g{}", dump_prefix, i));
            }
        }
        if (dump_buffers && !dump_prefix.empty() && kind == 0 && index_buffer) { // debug: the index buffer as bound (first index + count indices)
            const u64 index_bytes = index_type == vk::IndexType::eUint32 ? 4 : 2;
            Core::Vkb::DumpBuffer(cmd, index_buffer, index_offset, (static_cast<u64>(a2) + a1) * index_bytes, fmt::format("{}index_{}", dump_prefix, index_bytes == 4 ? "u32" : "u16"));
        }
        for (u32 i = 0; dump_buffers && !dump_prefix.empty() && i < b.images.size(); ++i) { // debug: the images as bound (by their view handles)
            if (b.images[i].imageView) {
                Core::Vkb::DumpBoundView(cmd, b.images[i].imageView, fmt::format("{}bound{}", dump_prefix, i));
            }
        }
        if (dump_buffers && kind == 2 && args_buffer) { // an indirect draw's arguments (index count, instances, first index, base vertex, first instance)
            Core::Vkb::DumpBuffer(cmd, args_buffer, args_offset, 20, dump_prefix + "args");
        }
        if (!dump_prefix.empty()) { // note 90: what the draw was and how it bound its buffers (binding order = the shader's buffer bindings, vertex stage first)
            nlohmann::json bufs = nlohmann::json::array();
            for (u32 i = 0; i < b.buffers.size(); ++i) {
                const auto guest = i < b.guest.size() ? b.guest[i] : std::pair<u64, u64>{};
                bufs.push_back({{"index", i}, {"offset", b.buffers[i].offset}, {"range", b.buffers[i].range == VK_WHOLE_SIZE ? -1.0 : static_cast<double>(b.buffers[i].range)},
                                {"guest_address", fmt::format("{:#x}", guest.first)}, {"guest_size", guest.second}});
            }
            std::string push_hex;
            const auto* pb = reinterpret_cast<const u8*>(&push);
            for (size_t i = 0; i < sizeof(push); ++i) {
                push_hex += fmt::format("{:02x}", pb[i]);
            }
            const nlohmann::json meta{{"ps", fmt::format("{:016x}", ps_hash)}, {"kind", kind}, {"a1", a1}, {"a2", a2}, {"instances", instances}, {"buffers", bufs},
                                      {"push_data_hex", push_hex},
                                      {"note", "debugger mode slow-pass draw (note 90): ps_/vs_inN = input images, outN = colour targets, depth; gN_*.gpu = buffer N as the GPU "
                                               "had it right after the draw, gN_<address>_*.raw = the guest bytes at the draw call; images as raw texels (vkb_dump_png.py)"}};
            const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_image_dump" / dump_prefix / "draw.json";
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            if (FILE* f = std::fopen(path.string().c_str(), "wb")) {
                const std::string text = meta.dump(1);
                std::fwrite(text.data(), 1, text.size(), f);
                std::fclose(f);
            }
        }
    }
}

} // namespace


/// GPU-driven research census (debug, TLG_VKB_GD_CENSUS=1, every draw): (1) could the draw's shaders take resources from a GPU-written row (the record-mode rules),
/// by the first reason they cannot; (2) what its resources are (constant buffers in the frame ring / copied at the draw, views, samplers, fixed-function vertex
/// input); (3) do the engine object fields a GPU translation would read still hold at the next graphics submit (when the GPU starts), and do the bytes of constant
/// buffers that are copied at the draw.
bool GdCensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_GD_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
u32 GdWhy(const Stage& st) {
    using BT = VkbShader::BufferType;
    for (const auto& b : st.info->buffers) {
        if (b.buffer_type == BT::Flatbuf) return 0;
        if (b.buffer_type == BT::BdaPagetable || b.buffer_type == BT::FaultBuffer) return 3;
        if (!VkbShader::Indexed::IsIndexedBufferType(b.buffer_type)) return 2;
        if (b.buffer_type == BT::Guest && b.is_written) return 1;
        if (b.buffer_type == BT::SharedMemory) return 7;
    }
    for (const auto& image : st.info->images) {
        if (!VkbShader::Indexed::IsIndexedImage(image) || image.mip_fallback_mode == VkbShader::MipStorageFallbackMode::DynamicIndex) return 4;
    }
    return 8;
}
std::atomic<u64> gd_draws{0}, gd_why[9]{}, gd_depth_n{0}, gd_depth_ok{0}, gd_kind_n[3]{}, gd_kind_ok[3]{}, gd_attr{0}, gd_attr_ok{0}, gd_cb_ring{0}, gd_cb_copy{0},
    gd_cb_other{0}, gd_res{0}, gd_samplers{0}, gd_special{0}, gd_progconst{0}, gd_unmapped{0}, gd_snap_n{0}, gd_snap_changed{0}, gd_cb_hash_n{0}, gd_cb_hash_changed{0},
    gd_checks{0}, gd_variants{0};
std::mutex gd_mutex;
std::unordered_map<const void*, u32> gd_variant_why;
struct GdSnap {
    u64 object, word, address;
    u32 size;
    u64 hash;
};
thread_local std::vector<GdSnap> t_gd_snaps;
thread_local u64 t_gd_submit = 0;
u64 GdWord(u32 kind, u64 object) {
    if (kind == 3) {
        return Rd<u64>(object, 0x60) ^ (u64{Rd<u32>(object, 0x180)} << 40);
    }
    const u64 r = Rd<u64>(object, 0x30);
    return r ^ (u64{Rd<u32>(object, 0x44)} << 56) ^ (r != 0 && IsHeapPointer(r) ? Rd<u64>(r, 0x30) * 0x9e3779b97f4a7c15ull : 0);
}
u64 GdHash(u64 address, u32 size) {
    std::array<u8, 256> bytes{};
    const u32 n = std::min<u32>(size, 256);
    if (!runtime_memory_read_backing(address, bytes.data(), n)) {
        return 0;
    }
    u64 h = 0xcbf29ce484222325ull;
    for (u32 i = 0; i < n; ++i) {
        h = (h ^ bytes[i]) * 0x100000001b3ull;
    }
    return h;
}
void GdSources(u64 ctx, const Stage& st) {
    for (const auto* list : {&st.buffers, &st.images, &st.samplers}) {
        for (const Source& src : *list) {
            if (src.cls == Source::Class::Special) {
                gd_special.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (src.cls == Source::Class::ProgramConstant) {
                gd_progconst.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (src.cls == Source::Class::Unmapped) {
                gd_unmapped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (src.kind == 2) {
                gd_samplers.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            const u64 object = SlotObject(ctx, st.index, src.kind, src.slot);
            if (object == 0) {
                continue;
            }
            GdSnap s{object, GdWord(src.kind == 3 ? 3 : 1, object), 0, 0, 0};
            if (src.kind == 3) {
                const u64 address = static_cast<u64>(Rd<u32>(object, 0x60)) | static_cast<u64>(Rd<u32>(object, 0x64) & 0xffu) << 32;
                const u32 size = Rd<u32>(object, 0x180);
                Core::Vkb::BufferWindow w;
                if (address == 0 || size == 0) {
                    gd_cb_other.fetch_add(1, std::memory_order_relaxed);
                } else if (Core::Vkb::ConstantInRing(address, size, w)) {
                    gd_cb_ring.fetch_add(1, std::memory_order_relaxed);
                } else {
                    gd_cb_copy.fetch_add(1, std::memory_order_relaxed);
                    s.address = address, s.size = size, s.hash = GdHash(address, size);
                }
            } else {
                gd_res.fetch_add(1, std::memory_order_relaxed);
            }
            if (t_gd_snaps.size() < 65536) {
                t_gd_snaps.push_back(s);
            }
        }
    }
}
void GdCensus(u64 ctx, const Variant& v, u32 kind) {
    const u64 submit = Core::Vkb::GraphicsSubmits();
    if (submit != t_gd_submit) { // the GPU has started the work these draws went with: do the objects still say the same?
        for (const GdSnap& s : t_gd_snaps) {
            gd_snap_n.fetch_add(1, std::memory_order_relaxed);
            if (GdWord(s.address != 0 ? 3 : 1, s.object) != s.word) {
                gd_snap_changed.fetch_add(1, std::memory_order_relaxed);
            }
            if (s.address != 0) {
                gd_cb_hash_n.fetch_add(1, std::memory_order_relaxed);
                if (GdHash(s.address, s.size) != s.hash) {
                    gd_cb_hash_changed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        gd_checks.fetch_add(1, std::memory_order_relaxed);
        t_gd_snaps.clear();
        t_gd_submit = submit;
    }
    u32 why = 8;
    {
        std::scoped_lock lock{gd_mutex};
        auto [it, fresh] = gd_variant_why.try_emplace(&v, 8u);
        if (fresh) {
            gd_variants.fetch_add(1, std::memory_order_relaxed);
            it->second = GdWhy(v.vs);
            if (it->second == 8 && v.has_ps) {
                it->second = GdWhy(v.ps);
            }
        }
        why = it->second;
    }
    const bool ok = why == 8;
    gd_draws.fetch_add(1, std::memory_order_relaxed);
    gd_why[why].fetch_add(1, std::memory_order_relaxed);
    gd_kind_n[std::min<u32>(kind, 2)].fetch_add(1, std::memory_order_relaxed);
    gd_kind_ok[std::min<u32>(kind, 2)].fetch_add(ok ? 1 : 0, std::memory_order_relaxed);
    if (t_depth_pass) {
        gd_depth_n.fetch_add(1, std::memory_order_relaxed);
        gd_depth_ok.fetch_add(ok ? 1 : 0, std::memory_order_relaxed);
    }
    if (!v.attributes.empty()) {
        gd_attr.fetch_add(1, std::memory_order_relaxed);
        gd_attr_ok.fetch_add(ok ? 1 : 0, std::memory_order_relaxed);
    }
    GdSources(ctx, v.vs);
    if (v.has_ps) {
        GdSources(ctx, v.ps);
    }
}
nlohmann::json GdReport() {
    nlohmann::json by = nlohmann::json::object();
    for (u32 i = 0; i < 8; ++i) {
        by[RecRefusedNames[i]] = gd_why[i].load();
    }
    by["eligible"] = gd_why[8].load();
    return {{"draws", gd_draws.load()}, {"variants", gd_variants.load()}, {"by_why", by},
            {"depth_pass_n_ok", {gd_depth_n.load(), gd_depth_ok.load()}},
            {"kind_n_ok", {gd_kind_n[0].load(), gd_kind_ok[0].load(), gd_kind_n[1].load(), gd_kind_ok[1].load(), gd_kind_n[2].load(), gd_kind_ok[2].load()}},
            {"vertex_attributes_n_ok", {gd_attr.load(), gd_attr_ok.load()}}, {"cb_ring_copy_other", {gd_cb_ring.load(), gd_cb_copy.load(), gd_cb_other.load()}},
            {"resources_samplers_special_progconst_unmapped", {gd_res.load(), gd_samplers.load(), gd_special.load(), gd_progconst.load(), gd_unmapped.load()}},
            {"object_words_checked_changed", {gd_snap_n.load(), gd_snap_changed.load()}},
            {"copied_cb_bytes_checked_changed", {gd_cb_hash_n.load(), gd_cb_hash_changed.load()}}, {"submit_checks", gd_checks.load()}};
}
void DrawImpl(u64 ctx, u32 kind, u64 a1, u64 a2, u64 a3);
/// R7 step B: the generator's hint and the per-thread pipeline cache of record-mode draws. Its key holds every context word the fixed state and the variant key read
/// in a cascade pass (shaders, input layout, state objects, depth target, viewport, the batch's views, the material and its render object), so a hit is exactly the slow
/// path's answer; TLG_VKB_R7_VERIFY=1 computes the slow path too and counts disagreements.
struct StepHint {
    u64 batch{}, material{};
    bool valid{};
};
thread_local StepHint t_hint;
std::atomic<u64> c_pipe_hits{0}, c_pipe_misses{0}, c_pipe_verify{0}, c_pipe_verify_diff{0};
void SetStepHint(u64 batch, u64 material) {
    t_hint = {batch, material, true};
}
void ClearStepHint() {
    t_hint.valid = false;
}
/// R7 step D: the context words the fixed state (Fixed::Build's key) and the variant key (VariantKeyFast) read
inline u64 IdOf(u64 object) {
    return object != 0 ? Rd<u64>(object, 0x10) : 0;
}
void LeanWords(u64 ctx, u32 kind, u64 vs, u64 ps, std::array<u64, 24>& w) {
    const u64 blend = Rd<u64>(ctx, 0x46f0), raster = Rd<u64>(ctx, 0x46c8), ds = Rd<u64>(ctx, 0x46e0), dsv = Rd<u64>(ctx, CtxDepthStencilView), il = Rd<u64>(ctx, CtxInputLayout);
    w[0] = vs, w[1] = IdOf(vs), w[2] = ps, w[3] = ps != 0 ? IdOf(ps) : 0;
    w[4] = il, w[5] = il != 0 ? Rd<u64>(vs + 0x30, 0xe0) : 0; // the input layout and the fetch object
    w[6] = blend, w[7] = IdOf(blend), w[8] = raster, w[9] = IdOf(raster), w[10] = ds, w[11] = IdOf(ds);
    w[12] = dsv, w[13] = IdOf(dsv);
    w[14] = Rd<u64>(ctx, 0x46e8) | u64{Rd<u32>(ctx, 0x4480)} << 8 | u64{kind} << 40 | u64{Rd<u64>(ctx, 0x130e0) != 0} << 44;
    w[15] = Rd<u64>(ctx, 0x46f8), w[16] = Rd<u64>(ctx, 0x4700); // blend factor
    w[17] = Rd<u64>(ctx, 0x44ec), w[18] = Rd<u64>(ctx, 0x44f4), w[19] = Rd<u64>(ctx, 0x44fc); // viewport
    u64 t = 0;
    for (u32 cb = 0; cb < 8; ++cb) {
        t = (t ^ Rd<u64>(ctx, CtxRenderTargets + cb * 8)) * 0x100000001b3ull;
    }
    w[20] = t;
    for (u32 stage = 0; stage < 2; ++stage) { // the objects in the shader slots each stage uses (the variant key's object classes)
        const u64 shader = stage == 0 ? vs : ps;
        u64 h = 0x84222325ull + stage;
        if (shader != 0) {
            const u64 masks[2] = {Rd<u64>(shader, 0x30 + 0xc0), Rd<u64>(shader, 0x30 + 0xc8)};
            for (u32 half = 0; half < 2; ++half) {
                for (u64 bits = masks[half]; bits != 0; bits &= bits - 1) {
                    const u32 slot = half * 64 + static_cast<u32>(std::countr_zero(bits));
                    const u64 object = Rd<u64>(ctx, SrvTable(stage) + slot * 8);
                    h = (h ^ object ^ (u64{slot} << 56)) * 0x100000001b3ull;
                    h = (h ^ (object == 0 ? 0 : RowByPointer() ? u64{Core::Vkb::ObjectPointerGen(object)} | (1ull << 40) : Rd<u64>(object, 0x10))) * 0x100000001b3ull; // G3: no read of the object
                }
            }
        }
        w[21 + stage] = h;
    }
    w[23] = ps != 0 ? Rd<u64>(ps, 0x120) : 0; // (the pixel shader's Gnm block: the export mask the fixed state reads)
}
/// GPU-driven bound, kill experiment only (TLG_VKB_TEST_RAW_ONLY=1, nothing is drawn): every draw does only what stays on the CPU in "raw capture, GPU
/// translation": the pipeline found by a per-thread cache keyed by the context words (LeanWords: shaders, state objects, targets, viewport, the slot objects' ids),
/// then a raw record written (per source of both stages: the slot's object, for a constant buffer its current address and size; the draw's arguments and index view),
/// then the draw is dropped. Against TLG_VKB_TEST_NO_CAPTURE=1 (nothing at all): the cost of the CPU part of the design.
struct RawPipe {
    u64 hash{};
    std::array<u64, 24> words{};
    const Variant* v{};
};
struct RawState {
    std::vector<RawPipe> pipes = std::vector<RawPipe>(65536);
    u64 draws = 0;
    std::vector<u64> ring = std::vector<u64>(1u << 17);
    u32 at = 0;
};
thread_local RawState* t_raw = nullptr;
std::atomic<u64> c_raw_draws{0}, c_raw_hits{0}, c_raw_misses{0}, c_raw_words{0};
void RawOnly(u64 ctx, u32 kind, u64 a1, u64 a2, u64 a3) {
    if (t_raw == nullptr) {
        t_raw = new RawState();
    }
    RawState& r = *t_raw;
    const u64 vs = Rd<u64>(ctx, CtxVertexShader);
    const u64 ps = Rd<u64>(ctx, CtxPixelShader);
    if (vs == 0) {
        return;
    }
    std::array<u64, 24> words;
    LeanWords(ctx, kind, vs, ps, words);
    u64 h = 0xcbf29ce484222325ull;
    for (const u64 w : words) {
        h = (h ^ w) * 0x100000001b3ull;
    }
    RawPipe& slot = r.pipes[static_cast<u32>(h >> 48) & 65535];
    if ((++r.draws & 1023) == 0) {
        c_raw_draws.fetch_add(1024, std::memory_order_relaxed);
    }
    const Variant* v = nullptr;
    if (slot.hash == h && slot.words == words) {
        v = slot.v;
        ++r.at; // (a local counter: no shared atomic on the hot path)
    } else {
        c_raw_misses.fetch_add(1, std::memory_order_relaxed);
        const VkbVulkan::FixedState* fixed = Core::NativeDraw::FixedStateOf(ctx);
        if (fixed == nullptr) {
            return;
        }
        const u64 fetch = Core::NativeDraw::FetchObjectOf(ctx, vs);
        const u64 key = VariantKey(ctx, vs, ps, fetch, fixed, kind == 2);
        std::scoped_lock lock{mutex};
        if (!Ready()) {
            return;
        }
        auto [it, fresh] = variants.try_emplace(key);
        if (fresh) {
            it->second = std::make_unique<Variant>();
            Compile(ctx, vs, ps, fetch, *fixed, *it->second, kind == 2);
        }
        if (!it->second->ok) {
            return;
        }
        v = it->second.get();
        slot.hash = h, slot.words = words, slot.v = v;
    }
    // the raw record
    u64* out = r.ring.data();
    u32 at = r.at & ((1u << 17) - 1);
    if (at + 512 > r.ring.size()) {
        at = 0;
    }
    u32 n = 0;
    out[at + n++] = reinterpret_cast<u64>(v);
    out[at + n++] = a1, out[at + n++] = a2, out[at + n++] = a3;
    out[at + n++] = Rd<u64>(ctx, 0x58), out[at + n++] = Rd<u64>(ctx, 0x60), out[at + n++] = Rd<u32>(ctx, 0x4724);
    const auto stage = [&](const Stage& st) {
        for (const auto* list : {&st.buffers, &st.images, &st.samplers}) {
            for (const Source& src : *list) {
                if (src.cls != Source::Class::Slot || n > 480) {
                    continue;
                }
                const u64 object = SlotObject(ctx, st.index, src.kind, src.slot);
                out[at + n++] = object;
                if (src.kind == 3 && object != 0) {
                    out[at + n++] = static_cast<u64>(Rd<u32>(object, 0x60)) | static_cast<u64>(Rd<u32>(object, 0x64) & 0xffu) << 32 | u64{Rd<u32>(object, 0x180)} << 40;
                }
            }
        }
    };
    stage(v->vs);
    if (v->has_ps) {
        stage(v->ps);
    }
    for (const Source& src : v->attributes) { // the vertex buffers of a fixed-function vertex input
        if (src.cls == Source::Class::Slot && n < 500) {
            out[at + n++] = SlotObject(ctx, 0, src.kind, src.slot);
        }
    }
    r.at = at + n;
}
void SetDepthPass(bool on) {
    t_depth_pass = on;
}
void FlushBatch() {
    t_flush_reason = 2;
    FlushBatchHere();
}
void Draw(u64 ctx, u32 kind, u64 a1, u64 a2, u64 a3) {
    // GPU-driven bound, kill experiment only (TLG_VKB_TEST_NO_CAPTURE=1, nothing is drawn): every draw returns before our capture, so the render thread runs the
    // engine alone: the fps ceiling of moving all per-draw work off the CPU
    static const bool no_capture = [] { const char* e = std::getenv("TLG_VKB_TEST_NO_CAPTURE"); return e != nullptr && e[0] == '1'; }();
    if (no_capture) {
        c_draws.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    static const bool raw_only = [] { const char* e = std::getenv("TLG_VKB_TEST_RAW_ONLY"); return e != nullptr && e[0] == '1'; }();
    if (raw_only) {
        RawOnly(ctx, kind, a1, a2, a3);
        return;
    }
    if (t_depth_pass && DepthCensusOn()) {
        const u64 t0 = NowNs();
        DrawImpl(ctx, kind, a1, a2, a3);
        c_dc_draw_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
        return;
    }
    DrawImpl(ctx, kind, a1, a2, a3);
}
void DrawImpl(u64 ctx, u32 kind, u64 a1, u64 a2, u64 a3) {
    c_draws.fetch_add(1, std::memory_order_relaxed);
    // sensitivity experiment: TLG_VKB_SPIN_NS=N spends N ns more per draw on the render thread (how much of its time is on the frame's critical path)
    static const u64 spin_ns = [] { const char* e = std::getenv("TLG_VKB_SPIN_NS"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull; }();
    if (spin_ns != 0) {
        const u64 until = NowNs() + spin_ns;
        while (NowNs() < until) {
        }
    }
    c_by_kind[std::min<u32>(kind, 2)].fetch_add(1, std::memory_order_relaxed);
    const u64 vs = Rd<u64>(ctx, CtxVertexShader);
    const u64 ps = Rd<u64>(ctx, CtxPixelShader);
    if (vs == 0 || !IsHeapPointer(vs) || Rd<u64>(vs, 0x120) == 0 || (ps != 0 && (!IsHeapPointer(ps) || Rd<u64>(ps, 0x120) == 0))) {
        c_no_vs.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // a vertex shader variant, a hull / domain or geometry stage, a dispatch draw: the pipeline is vertex plus pixel only (counted)
    if (Rd<u64>(vs, 0x420) != 0 || Rd<u64>(ctx, 0x4680) != 0 || Rd<u64>(ctx, 0x4688) != 0 || Rd<u32>(ctx, 0x130c8) == 0x200) {
        c_special.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // R7 step D: a record-mode draw of a cascade finds its pipeline and fixed state in the per-thread cache, keyed by every context word the fixed state and
    // the variant key read (state objects and their ids, shaders, input layout, targets, viewport, blend factor, stencil reference, topology, the objects in the
    // shader slots the stages use); a hit is the slow path's answer by construction (TLG_VKB_R7_VERIFY=1 checks it)
    t_pipe_pending = false;
    if ((t_depth_pass || RecordAll()) && RecordOn() && (kind == 0 || kind == 2) && BatchMatches(ctx) && Ready()) {
        static const bool lean_times = [] { const char* e = std::getenv("TLG_VKB_LEAN_TIMES"); return e != nullptr && e[0] == '1'; }();
        const u64 lt0 = lean_times ? NowNs() : 0;
        struct LeanTimer {
            u64 t0;
            ~LeanTimer() {
                if (t0 != 0) {
                    c_lean_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
                    c_lean_n.fetch_add(1, std::memory_order_relaxed);
                }
            }
        } lean_timer{lt0};
        std::array<u64, 24> words{};
        LeanWords(ctx, kind, vs, ps, words);
        if (lean_times) {
            c_lean_words_ns.fetch_add(NowNs() - lt0, std::memory_order_relaxed);
        }
        u64 h = 0xcbf29ce484222325ull;
        for (const u64 w : words) {
            h = (h ^ w) * 0x100000001b3ull;
        }
        PipeSlotLean& slot = Lean().pipes[static_cast<u32>(h >> 52) & 4095];
        if (slot.hash == h && slot.words == words && slot.v != nullptr) {
            c_pipe_hits.fetch_add(1, std::memory_order_relaxed);
            Variant& cv = *static_cast<Variant*>(slot.v);
            static const bool verify = [] { const char* e = std::getenv("TLG_VKB_R7_VERIFY"); return e != nullptr && e[0] == '1'; }();
            if (verify) {
                const VkbVulkan::FixedState* f = Core::NativeDraw::FixedStateOf(ctx);
                const u64 k = f != nullptr ? VariantKey(ctx, vs, ps, Core::NativeDraw::FetchObjectOf(ctx, vs), f, kind == 2) : 0;
                std::scoped_lock lock{mutex};
                const auto r = record_variants.find(k);
                c_pipe_verify.fetch_add(1, std::memory_order_relaxed);
                if (f != slot.fixed || r == record_variants.end() || r->second.get() != &cv) {
                    c_pipe_verify_diff.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (LeanAppend(ctx, cv, *slot.fixed, kind, a1, a2, a3)) {
                return;
            }
            c_lean_refused.fetch_add(1, std::memory_order_relaxed);
        } else {
            c_pipe_misses.fetch_add(1, std::memory_order_relaxed);
            t_pipe_words16 = words, t_pipe_hash = h, t_pipe_pending = true; // filled below, when the slow path has the answer
        }
    }
    const u64 fs0 = TimesOn() ? NowNs() : 0;
    const VkbVulkan::FixedState* fixed = Core::NativeDraw::FixedStateOf(ctx);
    if (TimesOn()) {
        c_fixed_ns.fetch_add(NowNs() - fs0, std::memory_order_relaxed);
    }
    if (fixed == nullptr) {
        c_no_fixed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 fetch = Core::NativeDraw::FetchObjectOf(ctx, vs);
    const u64 k0 = TimesOn() ? NowNs() : 0;
    const u64 key = VariantKey(ctx, vs, ps, fetch, fixed, kind == 2);
    // O2 bound (debug, TLG_VKB_O2_TWICE=1): the fixed state and the variant key computed a second time and thrown away: the fps drop is their cost (L-KEY's ceiling,
    // note 96), measured without timers and without a wrong pipeline
    static const bool o2_twice = [] { const char* e = std::getenv("TLG_VKB_O2_TWICE"); return e != nullptr && e[0] == '1'; }();
    if (o2_twice) {
        const VkbVulkan::FixedState* again = Core::NativeDraw::FixedStateOf(ctx);
        volatile u64 sink = again != nullptr ? VariantKey(ctx, vs, ps, Core::NativeDraw::FetchObjectOf(ctx, vs), again, kind == 2) : 0;
        (void)sink;
    }
    if (TimesOn()) {
        c_key_ns.fetch_add(NowNs() - k0, std::memory_order_relaxed);
    }
    const u64 vl0 = TimesOn() ? NowNs() : 0;
    std::scoped_lock lock{mutex};
    if (!Ready()) {
        return;
    }
    auto [it, fresh] = variants.try_emplace(key);
    if (TimesOn()) {
        c_variant_find_ns.fetch_add(NowNs() - vl0, std::memory_order_relaxed);
    }
    if (!fresh) {
        c_found.fetch_add(1, std::memory_order_relaxed);
    } else {
        it->second = std::make_unique<Variant>();
        Compile(ctx, vs, ps, fetch, *fixed, *it->second, kind == 2);
    }
    if (!it->second->ok) {
        c_skip_not_compiled.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (GdCensusOn()) {
        GdCensus(ctx, *it->second, kind);
    }
    if (Core::Vkb::Compute::LanternCensusActive()) { // lantern census: does this draw read the lantern particle state?
        const Variant& lv = *it->second;
        for (const Stage* st : {&lv.vs, &lv.ps}) {
            if (st == &lv.ps && !lv.has_ps) {
                continue;
            }
            for (const Source& src : st->buffers) {
                const u64 object = src.cls == Source::Class::Slot && src.kind == 1 ? SlotObject(ctx, st->index, 1, src.slot) : 0;
                if (object == 0 || (Rd<u32>(object, 0x44) != 1 && Rd<u32>(object, 0x44) != 0xb)) {
                    continue;
                }
                const u64 buffer = Rd<u64>(object, 0x30);
                const u64 bytes = buffer != 0 ? Rd<u32>(buffer, 0x180) : 0;
                u64 offset = 0, size = bytes;
                Core::Vkb::ViewRange(object, true, buffer, bytes, offset, size);
                if (bytes != 0) {
                    Core::Vkb::Compute::LanternDrawRead(Rd<u64>(buffer, 0x30) + offset, size, lv.vs.info ? lv.vs.info->pgm_hash : 0, lv.has_ps && lv.ps.info ? lv.ps.info->pgm_hash : 0, st->index, src.slot);
                }
            }
        }
    }
    if (t_depth_pass && Core::NativeZPassGen::R7CensusOn()) { // R7 research census: the prediction of this draw, and our pipeline for it
        static const bool sized = [] { Core::NativeZPassGen::R7SetFixedBytes(sizeof(VkbVulkan::FixedState)); return true; }();
        (void)sized;
        Core::NativeZPassGen::R7Check(ctx, kind, a1, a2, it->second.get(), fixed);
    }
    if ((t_depth_pass || RecordAll()) && RecordOn() && (kind == 0 || kind == 2) && Core::Vkb::Indexed::On()) { // R2 / R3: the cascades' indexed draws through their record-mode twin when it exists
        auto [rit, rfresh] = record_variants.try_emplace(key);
        if (rfresh) {
            rit->second = std::make_unique<Variant>();
            Compile(ctx, vs, ps, fetch, *fixed, *rit->second, kind == 2, true);
        }
        if (rit->second->ok) {
            if (t_pipe_pending) { // R7 step D: the cache learns the slow path's answer
                PipeSlotLean& slot = Lean().pipes[static_cast<u32>(t_pipe_hash >> 52) & 4095];
                slot.hash = t_pipe_hash, slot.words = t_pipe_words16, slot.v = rit->second.get(), slot.fixed = fixed;
                t_pipe_pending = false;
            }
            t_row_failed = false;
            Record(ctx, *rit->second, *fixed, kind, a1, a2, a3);
            if (!t_row_failed) {
                return;
            }
        }
        c_rec_fallback.fetch_add(1, std::memory_order_relaxed);
    }
    if (Core::FrameWalk::DebuggerMode()) [[unlikely]] { // note 89: the shaders of the scene for the freeze
        Core::Vkb::ShaderDump::NoteDraw(it->second->vs.info ? it->second->vs.info->pgm_hash : 0, it->second->has_ps && it->second->ps.info ? it->second->ps.info->pgm_hash : 0);
    }
    static const bool no_draw_record = [] { const char* e = std::getenv("TLG_VKB_NO_DRAW_RECORD"); return e != nullptr && *e == '1'; }(); // kill experiment: variant found, nothing recorded
    // note 62 bound: TLG_VKB_NO_DRAW_RECORD_AFTER=N: the same kill experiment from graphics submit N on (after the load, which the plain switch does not survive)
    static const u64 no_record_after = [] { const char* e = std::getenv("TLG_VKB_NO_DRAW_RECORD_AFTER"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull; }();
    if (t_depth_pass && DepthCensusOn()) {
        DcCensus(ctx, *it->second, fixed, kind);
    }
    if (!no_draw_record && !(no_record_after != 0 && Core::Vkb::GraphicsSubmits() >= no_record_after)) {
        Record(ctx, *it->second, *fixed, kind, a1, a2, a3);
    }
}

nlohmann::json TargetCensusReport() {
    std::scoped_lock lock{target_census_mutex};
    nlohmann::json j = nlohmann::json::object();
    for (const auto& [line, n] : target_census) {
        j[line] = n;
    }
    return j;
}
/// issue 75: the translator's reads with automatic level choice in per-pixel branches (pixel shaders): moved before the branch, or kept with a reason
/// (reads = moved + in_uniform_branch_only + outside_any_branch + the kept ones)
nlohmann::json ReadsOutOfBranchReport() {
    namespace O = VkbShader::Optimization;
    const auto& c = O::ReadsOutOfBranchCounters();
    nlohmann::json kept = nlohmann::json::object();
    for (u32 i = 0; i < O::ReadsOutOfBranchReasonCount; ++i) {
        kept[O::ReadsOutOfBranchReasonNames[i]] = c.not_moved[i].load();
    }
    nlohmann::json shaders = nlohmann::json::array();
    for (const auto& s : O::ReadsOutOfBranchShaders()) {
        nlohmann::json k = nlohmann::json::object();
        for (u32 i = 0; i < O::ReadsOutOfBranchReasonCount; ++i) {
            if (s.not_moved[i] != 0) {
                k[O::ReadsOutOfBranchReasonNames[i]] = s.not_moved[i];
            }
        }
        shaders.push_back({{"ps", fmt::format("{:016x}", s.hash)}, {"applied", s.applied}, {"reads", s.reads}, {"moved", s.moved},
                           {"ops_moved", s.ops_moved}, {"in_uniform_branch", s.in_uniform_branch}, {"outside_any_branch", s.outside_branches}, {"kept", k}});
    }
    return {{"on", profile.reads_out_of_branch},
            {"pixel_shader_translations", c.pixel_shaders.load()},
            {"translations_with_reads_in_a_branch", c.shaders_with_reads_in_branch.load()},
            {"reads", c.reads.load()},
            {"moved", c.moved.load()},
            {"ops_moved", c.ops_moved.load()},
            {"in_uniform_branch_only", c.in_uniform_branch.load()},
            {"outside_any_branch", c.outside_branches.load()},
            {"kept_in_per_pixel_branch", kept},
            {"shaders", shaders}};
}
/// look option Realistic tree shade (Extras, TLG_LEAF_AO; off by default): ambient light inside tree crowns; translations of the listed ambient shaders by result (a mismatch
/// leaves the shader unchanged)
nlohmann::json LeafAoReport() {
    namespace O = VkbShader::Optimization;
    const auto& c = O::LeafAoCounters();
    nlohmann::json results = nlohmann::json::object();
    for (u32 i = 0; i < O::LeafAoResultCount; ++i) {
        results[O::LeafAoResultNames[i]] = c.results[i].load();
    }
    nlohmann::json shaders = nlohmann::json::array();
    for (const auto& s : O::LeafAoShaders()) {
        shaders.push_back({{"cs", fmt::format("{:016x}", s.hash)}, {"result", O::LeafAoResultNames[static_cast<u32>(s.result)]}});
    }
    float strength = 0.0f, radius = 0.0f;
    TlgSettings::LeafAoLive(strength, radius);
    return {{"on", strength > 0.0f}, {"strength", strength}, {"radius_m", radius},
            {"materials_mask", profile.leaf_ao_materials}, {"translations", results}, {"shaders", shaders}};
}

nlohmann::json Report() {
    const u64 compiled = c_compiled.load(), draws = c_draws.load();
    nlohmann::json fails = nlohmann::json::object();
    for (u32 i = 0; i < FailCount; ++i) {
        fails[FailNames[i]] = c_fail[i].load();
    }
    const auto src = [&](u32 stage) {
        return nlohmann::json{{"buffers", {c_src[stage][0][0].load(), c_src[stage][0][1].load()}},
                              {"images", {c_src[stage][1][0].load(), c_src[stage][1][1].load()}},
                              {"samplers", {c_src[stage][2][0].load(), c_src[stage][2][1].load()}}};
    };
    return {{"draws", draws},
            {"gd_census", GdReport()},
            {"raw_only_draws_misses", {c_raw_draws.load(), c_raw_misses.load()}},
            {"fast_row", {{"rows_hits_misses_ring", {c_fast_rows.load(), c_fast_hits.load(), c_fast_misses.load(), c_fast_ring.load()}}, {"refused", c_fast_refused.load()}, {"verify_diff", {c_fast_verify.load(), c_fast_verify_diff.load()}}}},
            {"record", {{"compiled", c_rec_compiled.load()}, {"refused", c_rec_refused.load()}, {"refused_by", [] {
                             nlohmann::json j = nlohmann::json::object();
                             for (u32 i = 0; i < RecRefusedNames.size(); ++i) {
                                 j[RecRefusedNames[i]] = c_rec_refused_by[i].load();
                             }
                             return j;
                         }()},
                         {"draws", c_rec_draws.load()}, {"batches_draws", {c_batches.load(), c_batch_draws.load()}}, {"lean_appended_refused", {c_lean.load(), c_lean_refused.load()}}, {"lean_ns_n_words_row", {c_lean_ns.load(), c_lean_n.load(), c_lean_words_ns.load(), c_lean_row_ns.load()}}, {"row_memo_hits_misses", {c_row_memo_hits.load(), c_row_memo_misses.load()}}, {"pipe_cache_hits_misses_verify_diff", {c_pipe_hits.load(), c_pipe_misses.load(), c_pipe_verify.load(), c_pipe_verify_diff.load()}}, {"flush_by_job_pass_end", {c_flush_by[0].load(), c_flush_by[1].load(), c_flush_by[2].load()}}, {"cut_by_work_kind", [] { nlohmann::json j = nlohmann::json::object(); for (u32 i = 0; i < 16; ++i) { if (c_cut_by_work[i].load() != 0) { j[std::to_string(i)] = c_cut_by_work[i].load(); } } return j; }()}, {"fallback_draws", c_rec_fallback.load()}, {"row_failures", c_rec_row_fail.load()}, {"global_set", Core::Vkb::Indexed::Report()}}},
            {"depth_census", {{"draws", c_dc_draws.load()}, {"variant_same", c_dc_variant_same.load()}, {"vs_n_same_samebutcb", {c_dc_stage_n[0].load(), c_dc_stage_same[0].load(), c_dc_same_but_cb[0].load()}},
                              {"ps_n_same_samebutcb", {c_dc_stage_n[1].load(), c_dc_stage_same[1].load(), c_dc_same_but_cb[1].load()}}, {"draw_same", c_dc_draw_same.load()},
                              {"shapes", [] {
                                   std::scoped_lock lock{dc_shape_mutex};
                                   std::vector<std::pair<u64, std::string>> rows;
                                   for (const auto& [k, e] : dc_shapes) {
                                       rows.push_back({e.second, e.first});
                                   }
                                   std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                                   nlohmann::json j = nlohmann::json::array();
                                   for (const auto& [n, t] : rows) {
                                       j.push_back(fmt::format("{} draws: {}", n, t));
                                   }
                                   return j;
                               }()},
                              {"flat_n_same_dwords", {c_dc_flat_n.load(), c_dc_flat_same.load(), c_dc_flat_dwords.load()}}, {"cb_copies", c_dc_copies.load()}, {"draw_ms", static_cast<double>(c_dc_draw_ns.load()) / 1e6}, {"bind_ms", static_cast<double>(c_dc_bind_ns.load()) / 1e6}}},
            {"by_kind_indexed_auto_indirect", {c_by_kind[0].load(), c_by_kind[1].load(), c_by_kind[2].load()}},
            {"first_vertex_by_vulkan_by_shader", {c_base_vertex_vulkan.load(), c_base_vertex_shader.load()}},
            {"first_vertex_fetch_reads_another_register", c_base_vertex_other_register.load()},
            {"no_vertex_shader", c_no_vs.load()},
            {"special_pipeline", c_special.load()},
            {"no_fixed_state", c_no_fixed.load()},
            {"variant_found", c_found.load()},
            {"variants", compiled + c_failed.load()},
            {"compiled", compiled},
            {"failed", c_failed.load()},
            {"failed_by", fails},
            {"unsupported_layout_type_bits", fmt::format("{:#x}", c_unsupported_layout_bits.load())},
            {"depth_only_variants", c_depth_only.load()},
            {"compile_ms_mean", compiled ? static_cast<double>(c_compile_ns.load()) / 1e6 / static_cast<double>(compiled) : 0.0},
            {"shader_store", Core::Vkb::ShaderStore::Report()}, // TLG_VKB_SHADER_PREBUILD: graphics and compute stages
            {"shader_recipes", Core::Vkb::ShaderRecipe::Report()}, // TLG_VKB_SHADER_RECIPES
            {"key_check_draws_differences", {c_key_checks.load(), c_key_differences.load()}},
            {"variant_key_us_mean", draws ? static_cast<double>(c_key_ns.load()) / 1e3 / static_cast<double>(draws) : 0.0},
            {"slot_map_mapped_unmapped", {{"vertex", src(0)}, {"pixel", src(1)}}},
            {"vertex_attributes_mapped_unmapped", {c_attr_mapped.load(), c_attr_unmapped.load()}},
            {"recorded", c_recorded.load()},
            {"record_us_mean", c_recorded.load() ? static_cast<double>(c_record_ns.load()) / 1e3 / static_cast<double>(c_recorded.load()) : 0.0},
            {"record_phase_us_mean_targets_stages_vertex_handover", [&] {
                 nlohmann::json a = nlohmann::json::array();
                 for (u32 i = 0; i < 4; ++i) {
                     a.push_back(c_recorded.load() ? static_cast<double>(c_phase_ns[i].load()) / 1e3 / static_cast<double>(c_recorded.load()) : 0.0);
                 }
                 return a;
             }()},
            {"fixed_state_us_mean", draws ? static_cast<double>(c_fixed_ns.load()) / 1e3 / static_cast<double>(draws) : 0.0},
            {"r2b_on_late_buffers_failed", {ResolveOnRecorder(), c_late_buffers.load(), c_late_failed.load()}},
            {"r2b2b_on_late_targets_images", {LateImagesOn(), c_late_targets.load(), c_late_images.load()}},
            {"target_memo_on_hits_misses", {TargetMemoOn(), c_target_memo_hits.load(), c_target_memo_misses.load()}}, // issue 82 point 2
            {"target_census", TargetCensusReport()},
            {"flat_census", Core::Vkb::FlatCensus::Report()},
            {"d3_targets_engine_unmapped", {c_targets_engine.load(), c_targets_unmapped.load()}},
            {"d3_target_check_compares_differences", {c_target_checks.load(), c_target_check_differences.load()}},
            {"d3_target_check_first", [] { std::scoped_lock lock{target_check_mutex}; return nlohmann::json(target_check_lines); }()},
            {"state_filter_sent_skipped_pipelines_vertex_inputs", {c_state_sent.load(), c_state_skipped.load(), c_state_pipelines.load(), c_state_vertex_inputs.load()}},
            {"cb_census_copies_bytes_same_samebytes_addresschanged", {c_cbc_copies.load(), c_cbc_bytes.load(), c_cbc_same.load(), c_cbc_same_bytes.load(), c_cbc_same_address_changed.load()}},
            {"stage_census_vs_ps_same_changed_buffers_images_samplers_same", {{c_stage_same[0].load(), c_stage_changed[0].load(), c_stage_buffers_same[0].load(), c_stage_images_same[0].load(), c_stage_samplers_same[0].load()},
                                                                               {c_stage_same[1].load(), c_stage_changed[1].load(), c_stage_buffers_same[1].load(), c_stage_images_same[1].load(), c_stage_samplers_same[1].load()}}},
            {"variant_find_us_mean", draws ? static_cast<double>(c_variant_find_ns.load()) / 1e3 / static_cast<double>(draws) : 0.0},
            {"lookups", Core::Vkb::LookupReport()},
            {"aa", Core::Vkb::Aa::Report()},
            {"reads_out_of_branch", ReadsOutOfBranchReport()},
            {"leaf_ao", LeafAoReport()},
            {"recorder_job_us_mean", c_job_n.load() ? static_cast<double>(c_job_ns.load()) / 1e3 / static_cast<double>(c_job_n.load()) : 0.0},
            {"new_passes", c_new_passes.load()},
            {"indirect_recorded", c_indirect_recorded.load()},
            {"views_noted_at_first_use", c_late_views.load()},
            {"flat_user_data_rebuilt", c_flat_rebuilt.load()},
            {"skipped", {{"not_compiled", c_skip_not_compiled.load()}, {"buffer", c_skip_buffer.load()}, {"image", c_skip_image.load()}, {"sampler", c_skip_sampler.load()},
                         {"unordered_access", c_skip_uav.load()}, {"special_buffer", c_skip_special.load()}, {"color_target", c_skip_target.load()},
                         {"depth_target", c_skip_depth.load()}, {"no_attachment", c_skip_no_attachment.load()}, {"vertex_input", c_skip_vertex.load()},
                         {"index_buffer", c_skip_index.load()}, {"indirect", c_skip_indirect.load()}}}};
}

} // namespace Core::Vkb::Graphics
