// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_vkb_aa.h"
#include "native/native_vkb_shader_dump.h"
#include "native/native_framewalk.h"
#include "native/native_debug_server.h"
#include "native/native_marker.h"
#include "native/native_vkb_compute.h"
#include "native/native_vkb_shader_recipe.h"
#include "native/native_vkb_shader_store.h"
#include "native/native_device.h"

#include "native/shadps4/video_core/renderer_vulkan/vk_common.h"

#include <algorithm>
#include <bit>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <filesystem>
#include <mutex>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "tlg_settings.h"
#include "native/shadps4/core/guest_heap.h"
#include "native/native_dispatch_inputs.h"
#include "native/native_draw.h"
#include "native/native_desc.h"
#include "native/native_vkb.h"
#include "native/vkb_core/vkb_device.h"
#include "native/native_vkb_write_window.h"
#include "native/native_vkb_objects.h"
#include "native/native_vkb_stream.h"
#include "native/native_vkb_util.h"
#include "native/shadps4/shader_recompiler/backend/bindings.h"
#include "native/shadps4/shader_recompiler/backend/spirv/emit_spirv.h"
#include "native/shadps4/shader_recompiler/info.h"
#include "native/shadps4/shader_recompiler/profile.h"
#include "native/shadps4/shader_recompiler/recompiler.h"
#include "native/shadps4/shader_recompiler/runtime_info.h"
#include "native/shadps4/video_core/amdgpu/regs_shader.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_host_import.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_instance.h"
#include "native/native_vkb_side_census.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_shader_boundary.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_shader_util.h"


namespace Core::Vkb::Compute {

namespace {

using Util::Mix;
using Util::ObjectClass;
using Util::Rd;
using Util::Source;
using Util::SourceOf;

struct Variant {
    VkbAmdGpu::ComputeProgram program{};
    Core::NativeDraw::UserLayout layout; ///< the user data the module was specialized with (kept: the info's spans point into it)
    std::unique_ptr<VkbShader::Info> info;
    vk::ShaderModule module{};
    vk::DescriptorSetLayout set_layout{};
    vk::PipelineLayout pipeline_layout{};
    vk::Pipeline pipeline{};
    std::vector<Source> buffers, images, samplers;
    std::vector<vk::ImageViewType> image_types; ///< the view type the module declares for each image
    Core::NativeDraw::UserLayout live;            ///< the user data of the current dispatch (the flat user data buffer only)
    bool has_flat{}, multi_binding{};
    bool async{}; ///< compiled for the async compute context
    bool ok{};
    u32 grid_x{0}; ///< native replacement module (research section 13): the game's 1x1x1 dispatch runs as grid_x groups (0: as the game asks)
    std::vector<Core::Vkb::WriteWindow::Window> windows; ///< R5 (note 38): per buffer binding, the range one dispatch can write
};

/// R5 (note 38): `TLG_VKB_WRITE_WINDOWS=1`: a written buffer binding with a proved write window is tracked by that range, not the whole binding
bool WriteWindowsOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_WRITE_WINDOWS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
std::atomic<u64> c_window_narrowed{0}, c_window_whole{0}, c_window_empty{0};

/// Native parallel shaders (research section 13): removed 2026-11-21 (the module was decompiled from the game's shader); Variant::grid_x stays for a run-time version
bool NativeCullOn() {
    return false;
}
std::atomic<u64> c_native_cull_modules{0}, c_native_cull_dispatches{0};

std::mutex mutex;
std::unordered_map<u64, std::unique_ptr<Variant>> variants; ///< by variant key
std::unordered_map<u64, u32> variants_per_shader;
std::unique_ptr<VkbShader::Pools> pools;
VkbShader::Profile profile{};
const VkbVulkan::Instance* instance = nullptr;

std::atomic<u64> c_dispatches{0}, c_found{0}, c_compiled{0}, c_failed{0}, c_failed_unsupported_input{0}, c_failed_translation{0}, c_failed_pipeline{0},
    c_too_many_bindings{0}, c_compile_ns{0}, c_no_shader{0}, c_key_ns{0};
std::atomic<u64> c_async_dispatches{0}, c_async_recorded{0}, c_async_unsupported{0}, c_gds_inits{0};
std::atomic<u64> c_recorded{0}, c_indirect{0}, c_skip_not_compiled{0}, c_skip_multi_binding{0}, c_skip_buffer{0}, c_skip_image{0}, c_skip_image_type{0}, c_skip_sampler{0},
    c_skip_indirect_args{0}, c_skip_shared_memory{0}, c_skip_special{0}, c_flat_rebuilt{0}, c_record_ns{0};
/// GPU time per shader (timestamps around every dispatch)
struct Timed {
    u64 generation{};
    u64 shader_id{}, pgm_hash{};
    u32 begin{};
    u32 x{}, y{}, z{};
    bool indirect{};
};
std::vector<Timed> frame_timed;
std::mutex timed_mutex; ///< V4g: frame_timed, shader_time and the totals (the recording thread adds, the submit reads)
struct ShaderTime {
    u64 pgm_hash{}, ns{}, count{}, max_ns{}, indirect{};
    u32 max_x{}, max_y{}, max_z{};
};
std::unordered_map<u64, ShaderTime> shader_time;
u64 timed_submits = 0, timed_ns = 0, timed_submit_max_ns = 0;
std::atomic<u64> c_buf_slot{0}, c_buf_special{0}, c_buf_constant{0}, c_buf_unmapped{0}, c_img_slot{0}, c_img_unmapped{0}, c_smp_slot{0}, c_smp_unmapped{0};

/// note 60: the record and key timers only with TLG_VKB_TIMES=1 (a clock read per dispatch costs about 3 % of the render thread)
bool DispatchTimesOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_TIMES"); return e != nullptr && e[0] == '1'; }();
    return on;
}
u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// The variant of the bound compute shader: the shader's id and the class of every object in the slots its usage table names
u64 VariantKey(u64 ctx, u64 shader) {
    u64 h = Mix(0x7c, Rd<u64>(shader, 0x10));
    const u64 gnm = Rd<u64>(shader, 0x120);
    const u32 count = Rd<u8>(gnm, 3);
    constexpr u32 Stage = 5;
    for (u32 i = 0; i < count && i < 40; ++i) {
        const u64 e = gnm + 0x28 + i * 4;
        const u32 type = Rd<u8>(e, 0);
        const u32 slot = Rd<u8>(e, 1);
        switch (type) {
        case 0: h = Mix(h, ObjectClass(Rd<u64>(ctx, 0x580 + Stage * 0x400 + slot * 8))); break;
        case 4: h = Mix(h, ObjectClass(Rd<u64>(ctx, 0x2098 + slot * 8))); break;
        case 19:
            // note 67: only the slots the shader's used-slot mask names (shader + 0xf0 / + 0xf8, as the engine's PrepareDispatch walks them); the other slots can hold
            // stale or non-pointer values (a crash read 0x100000001 from an unused slot)
            for (u32 half = 0; half < 2; ++half) {
                for (u64 bits = Rd<u64>(shader, 0xf0 + half * 8); bits != 0; bits &= bits - 1) {
                    const u32 s = half * 64 + static_cast<u32>(std::countr_zero(bits));
                    if (const u64 o = Rd<u64>(ctx, 0x580 + Stage * 0x400 + s * 8); o != 0) {
                        h = Mix(Mix(h, s), ObjectClass(o));
                    }
                }
            }
            break;
        case 25:
            for (u32 bits = Rd<u32>(shader, 0x100) & 0xff; bits != 0; bits &= bits - 1) { // note 67: the shader's UAV mask (shader + 0x100)
                const u32 s = static_cast<u32>(std::countr_zero(bits));
                if (const u64 o = Rd<u64>(ctx, 0x2098 + s * 8); o != 0) {
                    h = Mix(Mix(h, s), ObjectClass(o));
                }
            }
            break;
        default:
            h = Mix(h, type); // constant buffers and samplers do not change a module
            break;
        }
    }
    // Tint hunt kill experiment (2026-11-16, TLG_VKB_FULL_COMPUTE_KEY=1): the key also takes every descriptor word the compile reads (each used resource slot's
    // T# / V# without its address, every sampler slot's S#), so a module is never reused with descriptors it was not specialized with.
    static const bool full = [] { const char* e = std::getenv("TLG_VKB_FULL_COMPUTE_KEY"); return e != nullptr && e[0] == '1'; }();
    if (full) {
        for (u32 half = 0; half < 2; ++half) {
            for (u64 bits = Rd<u64>(shader, 0xf0 + half * 8); bits != 0; bits &= bits - 1) {
                const u32 s = half * 64 + static_cast<u32>(std::countr_zero(bits));
                const u64 o = Rd<u64>(ctx, 0x580 + Stage * 0x400 + s * 8);
                if (o == 0 || !IsHeapPointer(o)) {
                    continue;
                }
                const u32 kind = Rd<u32>(o, 0x44);
                const bool buffer = kind == 1 || kind == 0xb;
                const u64 src = o + (buffer ? 0x78 : 0x58);
                h = Mix(h, 0x5100 + s);
                h = Mix(h, Rd<u32>(src, 4) & ~0xffu); // word 1 without the address's high bits
                for (u32 w = 2; w < (buffer ? 4u : 8u); ++w) {
                    h = Mix(h, Rd<u32>(src, w * 4));
                }
            }
        }
        for (u32 s = 0; s < 16; ++s) {
            if (const u64 o = Rd<u64>(ctx, 0x2000 + s * 8); o != 0 && IsHeapPointer(o)) {
                h = Mix(h, 0x5200 + s);
                for (u32 w = 0; w < 4; ++w) {
                    h = Mix(h, Rd<u32>(o, 0x64 + w * 4));
                }
            }
        }
    }
    return h;
}

/// The variant of the shader bound at the async context: the shader's id and the class of every view its masks name
u64 AsyncVariantKey(u64 jcc, u64 shader) {
    u64 h = Mix(0xa5c, Rd<u64>(shader, 0x10));
    for (u32 mask = Rd<u32>(shader, 0xf0), index = 0; mask != 0; mask >>= 1, ++index) {
        if ((mask & 1) != 0) {
            h = Mix(Mix(h, index), ObjectClass(Rd<u64>(jcc, 0x1d18 + index * 8)));
        }
    }
    for (u32 mask = Rd<u32>(shader, 0x100), index = 0; mask != 0; mask >>= 1, ++index) {
        if ((mask & 1) != 0) {
            h = Mix(Mix(h, 0x100 + index), ObjectClass(Rd<u64>(jcc, 0x2118 + index * 8)));
        }
    }
    return h;
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
    Core::Vkb::ShaderStore::Open(profile); // TLG_VKB_SHADER_PREBUILD (no-op when off)
    pools = std::make_unique<VkbShader::Pools>();
    Stream::Init(*instance);
    Core::Vkb::Aa::Init(*instance);
    return true;
}

/// Compiles the variant of the shader bound at `ctx` (mutex held)
void BuildLayout(u64 ctx, u64 shader, bool async, bool tags, Core::NativeDraw::UserLayout& out) {
    async ? Core::NativeDraw::BuildAsyncLayout(ctx, tags, out) : Core::NativeDraw::BuildUserLayout(ctx, 5, shader, 0x28, tags, out);
}

void Compile(u64 ctx, u64 shader, Variant& v) {
    const u64 t0 = NowNs();
    const u64 gnm = Rd<u64>(shader, 0x120);
    const u64 regs = gnm + 8;
    BuildLayout(ctx, shader, v.async, false, v.layout);
    if (v.layout.unsupported_types != 0) {
        c_failed_unsupported_input.fetch_add(1, std::memory_order_relaxed);
        c_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // the program registers as the engine's Gnm block holds them (COMPUTE_PGM_LO, RSRC1, RSRC2, NUM_THREAD_X / Y / Z) and the user data
    std::array<u32, 80> words{};
    words[0x07] = Rd<u32>(regs, 16);
    words[0x08] = Rd<u32>(regs, 20);
    words[0x09] = Rd<u32>(regs, 24);
    words[0x0c] = Rd<u32>(regs, 0);
    words[0x12] = Rd<u32>(regs, 8);
    words[0x13] = Rd<u32>(regs, 12);
    std::memcpy(&words[0x40], v.layout.sgpr.data(), 16 * sizeof(u32));
    static_assert(sizeof(VkbAmdGpu::ComputeProgram) == sizeof(words));
    std::memcpy(&v.program, words.data(), sizeof(words));
    Core::NativeDispatch::DispatchInputs in{};
    in.num_thread_x = {v.program.num_thread_x.full, v.program.num_thread_x.partial};
    in.num_thread_y = {v.program.num_thread_y.full, v.program.num_thread_y.partial};
    in.num_thread_z = {v.program.num_thread_z.full, v.program.num_thread_z.partial};
    in.address = v.program.address;
    in.settings.num_vgprs = v.program.settings.num_vgprs;
    in.settings.num_sgprs = v.program.settings.num_sgprs;
    in.settings.fp_round_mode32 = static_cast<u32>(v.program.settings.fp_round_mode32);
    in.settings.fp_round_mode64 = static_cast<u32>(v.program.settings.fp_round_mode64);
    in.settings.fp_denorm_mode32 = static_cast<u32>(v.program.settings.fp_denorm_mode32);
    in.settings.fp_denorm_mode64 = static_cast<u32>(v.program.settings.fp_denorm_mode64);
    in.settings.dx10_clamp = v.program.settings.dx10_clamp;
    in.settings.ieee_mode = v.program.settings.ieee_mode;
    in.settings.scratch_en = v.program.settings.scratch_en;
    in.settings.num_user_regs = v.program.settings.num_user_regs;
    in.settings.tgid_enable = v.program.settings.tgid_enable;
    in.settings.lds_dwords = v.program.settings.lds_dwords;
    VkbShader::RuntimeInfo runtime{};
    runtime.Initialize(VkbShader::HwStage::Compute, VkbShader::SwStage::Compute);
    VkbVulkan::ShaderBoundary::FillComputeInfo(runtime, in);
    const VkbShader::ShaderParams params = VkbAmdGpu::GetParams(v.program);
    v.info = std::make_unique<VkbShader::Info>(VkbShader::HwStage::Compute, VkbShader::SwStage::Compute, params);
    v.info->pgm_base = params.Base();
    VkbShader::Backend::Bindings binding{};
    std::vector<u32> spv;
    // TLG_VKB_SHADER_MEMO / TLG_VKB_SHADER_PREBUILD (shader store): a program translated before with the same inputs, in this session (or an earlier one, on disk), is taken from the store, with
    // its write windows (they also depend on the local size, the salt); the debug census works on the translated program, so with it on every program is translated
    namespace Store = Core::Vkb::ShaderStore;
    const Store::Mode store_mode = Core::Vkb::FlatCensus::On() ? Store::Mode::Off : Store::GetMode();
    if (Core::Vkb::FlatCensus::On() && Store::GetMode() != Store::Mode::Off) {
        Store::NoteBypass();
    }
    const VkbShader::RuntimeInfo runtime_in = runtime;
    const Store::Inputs store_in{.params = params, .runtime = runtime_in, .start = binding, .salt = v.program.num_thread_x.full};
    Store::Translation stored;
    const bool found = store_mode != Store::Mode::Off && Store::Find(store_in, stored);
    if (!found || store_mode == Store::Mode::Check) {
        // TLG_VKB_SHADER_RECIPES: a program the store lacks also keeps its recipe (the guest memory the translation reads, recorded on this thread)
        Core::Vkb::ShaderRecipe::Recording recording{!found && store_mode != Store::Mode::Off && Store::OnDisk() && Core::Vkb::ShaderRecipe::On()};
        const u64 translate_t0 = NowNs();
        const auto program = VkbShader::TranslateProgram(params.code, *pools, *v.info, runtime, profile);
        if (v.info->translation_failed) {
            c_failed_translation.fetch_add(1, std::memory_order_relaxed);
            c_failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        spv = VkbShader::Backend::SPIRV::EmitSPIRV(profile, runtime, program, binding);
        // (2026-11-21: the native parallel culling module for 0xa3c4525b was removed: it was decompiled from the game's shader and must not ship. To come back
        // made at run time from the player's own game files (a SPIR-V patch of the recompiled module: start = global id, stride = the whole grid).)
        // debug (tint hunt, 2026-11-16): TLG_VKB_DUMP_SPV=<shader hashes, hex, comma separated> writes each variant's module to log/vkb_spv/cs_<hash>_<n>.spv (compare with the
        // fork renderer's TLG_DUMP_SHADERS=1 dumps)
        static const std::vector<u64> dump_spv = [] { // comma separated hashes
            std::vector<u64> out;
            if (const char* e = std::getenv("TLG_VKB_DUMP_SPV")) {
                for (const char* p = e; *p != 0;) {
                    char* end = nullptr;
                    const u64 h = std::strtoull(p, &end, 16);
                    if (end == p) {
                        break;
                    }
                    out.push_back(h);
                    p = (*end == ',' || *end == '+') ? end + 1 : end;
                }
            }
            return out;
        }();
        if (std::ranges::find(dump_spv, v.info->pgm_hash) != dump_spv.end()) {
            static std::atomic<u32> dumped{0};
            const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_spv";
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (FILE* f = std::fopen((dir / fmt::format("cs_{:08x}_{}.spv", v.info->pgm_hash, dumped.fetch_add(1))).string().c_str(), "wb")) {
                std::fwrite(spv.data(), sizeof(u32), spv.size(), f);
                std::fclose(f);
            }
        }
        if (Core::Vkb::FlatCensus::On()) { // note 77 census (debug)
            Core::Vkb::FlatCensus::Note(program, gnm, 0x28, v.async ? "cs_async" : "cs", v.info->pgm_hash);
        }
        v.windows = Core::Vkb::WriteWindow::Analyze(program, *v.info, v.program.num_thread_x.full);
        for (size_t i = 0; i < v.windows.size(); ++i) {
            if (const auto& w = v.windows[i]; w.valid) {
                LOG_INFO(Render_Vulkan, "vkb write window: shader {:#x} b{}: {}*wg + {}*lid + {}*cb[{}][{}] + [{}, {}] (local x {}){}", v.info->pgm_hash, i, w.a_wg, w.a_lid, w.k,
                         static_cast<s32>(w.cb_binding), w.cb_index, w.c_min, w.c_max, v.program.num_thread_x.full,
                         w.guarded ? fmt::format(", gid < {}", w.guard_immediate ? fmt::format("{}", w.guard_value) : fmt::format("cb[{}][{}]{}", w.guard_binding, w.guard_index, w.guard_signed ? " signed" : ""))
                                   : std::string{});
            }
        }
        if (!found) {
            Store::NoteBuild(params.hash, static_cast<u32>(VkbShader::HwStage::Compute));
            if (store_mode != Store::Mode::Off && Store::Keep(store_in, profile, *v.info, runtime, binding, spv, v.windows, NowNs() - translate_t0)) {
                recording.Take(store_in, *v.info, v.program.num_thread_x.full);
            }
        }
    }
    if (found && (store_mode != Store::Mode::Check || Store::Same(store_in, stored, *v.info, runtime, binding, spv, v.windows))) {
        v.info = std::move(stored.info);
        runtime = stored.runtime;
        binding = stored.end;
        spv = std::move(stored.spv);
        v.windows = std::move(stored.windows);
    }
    const vk::Device device = instance->GetDevice();
    v.module = VkbVulkan::CompileSPV(spv, device);
    Core::Vkb::ShaderDump::Keep(v.async ? Core::Vkb::ShaderDump::Stage::AsyncCompute : Core::Vkb::ShaderDump::Stage::Compute, v.info->pgm_hash, spv, params.code); // note 89
    // our layout: the module's ABI (storage buffers, then images, then samplers; one push-descriptor set; the push data block)
    u32 index = 0;
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    for (const auto& b : v.info->buffers) {
        (void)b;
        bindings.push_back({.binding = index++, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute});
    }
    for (const auto& image : v.info->images) {
        const u32 n = image.NumBindings(*v.info);
        v.multi_binding |= n != 1;
        bindings.push_back({.binding = index, .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage,
                            .descriptorCount = n, .stageFlags = vk::ShaderStageFlagBits::eCompute});
        index += n;
    }
    for (const auto& s : v.info->samplers) {
        (void)s;
        bindings.push_back({.binding = index++, .descriptorType = vk::DescriptorType::eSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute});
    }
    if (index >= instance->MaxPushDescriptors()) {
        c_too_many_bindings.fetch_add(1, std::memory_order_relaxed);
        c_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    v.set_layout = device.createDescriptorSetLayout({.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                                                     .bindingCount = static_cast<u32>(bindings.size()), .pBindings = bindings.data()}).value;
    const vk::PushConstantRange push{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(VkbShader::PushData)};
    v.pipeline_layout = device.createPipelineLayout({.setLayoutCount = 1, .pSetLayouts = &v.set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &push}).value;
    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{.requiredSubgroupSize = 64};
    const vk::ComputePipelineCreateInfo ci{
        .stage = {.pNext = instance->IsSubgroupSize64Supported() ? &subgroup : nullptr, .stage = vk::ShaderStageFlagBits::eCompute, .module = v.module, .pName = "main"},
        .layout = v.pipeline_layout,
    };
    const auto [result, pipeline] = device.createComputePipeline(VK_NULL_HANDLE, ci);
    if (result != vk::Result::eSuccess) {
        c_failed_pipeline.fetch_add(1, std::memory_order_relaxed);
        c_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    v.pipeline = pipeline;
    // the slot map: the same layout with a tag per slot, flattened by the shader's own table walker
    Core::NativeDraw::UserLayout tagged;
    BuildLayout(ctx, shader, v.async, true, tagged);
    const std::span<const u32> real_ud = v.info->user_data;
    v.info->user_data = std::span<const u32>{tagged.sgpr.data(), tagged.sgpr.size()};
    v.info->RefreshFlatBuf();
    const std::vector<u32> flat = v.info->flattened_ud_buf;
    v.info->user_data = real_ud;
    v.info->RefreshFlatBuf();
    for (const auto& b : v.info->buffers) {
        Source s = SourceOf(b.sharp_fetch, flat, b.IsSpecial(), b.post_op == VkbShader::SharpFetchPostOp::OffsetByProgramBase);
        if (s.cls == Source::Class::ProgramConstant) {
            const Core::NativeDesc::BufferDesc d = Core::NativeDesc::BufferDescOf(b.GetSharp(*v.info)); // the shader's own memory: the same for every dispatch
            s.address = d.address;
            s.size = d.size;
        }
        v.has_flat |= b.buffer_type == VkbShader::BufferType::Flatbuf;
        v.buffers.push_back(s);
        (s.cls == Source::Class::Slot ? c_buf_slot : s.cls == Source::Class::Special ? c_buf_special : s.cls == Source::Class::ProgramConstant ? c_buf_constant : c_buf_unmapped)
            .fetch_add(1, std::memory_order_relaxed);
    }
    for (const auto& image : v.info->images) {
        const Source s = SourceOf(image.sharp_fetch, flat, false, false);
        v.images.push_back(s);
        v.image_types.push_back(Core::Vkb::ModuleViewType(image.GetSharp(*v.info).GetHostViewType(image.is_array)));
        (s.cls == Source::Class::Slot ? c_img_slot : c_img_unmapped).fetch_add(1, std::memory_order_relaxed);
    }
    for (const auto& sampler : v.info->samplers) {
        const Source s = SourceOf(sampler.sharp_fetch, flat, false, false);
        v.samplers.push_back(s);
        (s.cls == Source::Class::Slot ? c_smp_slot : c_smp_unmapped).fetch_add(1, std::memory_order_relaxed);
    }
    v.ok = true;
    c_compiled.fetch_add(1, std::memory_order_relaxed);
    Core::Vkb::FrameNote(fmt::format("new compute variant: cs {:#x}{}", v.info ? v.info->pgm_hash : 0ull, v.async ? " async" : "")); // note 44
    c_compile_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
}

/// The engine object in slot `slot` of the compute stage's table of `kind` (UserLayout tags: 1 resource, 2 sampler, 3 constant buffer, 4 unordered access)
u64 SlotObject(u64 ctx, u32 kind, u32 slot, bool async) {
    u64 table = 0, limit = 0;
    switch (kind) {
    case 1: table = async ? 0x1d18 : 0x1980, limit = async ? 32 : 128; break;
    case 2: table = async ? 0x2178 : 0x2000, limit = 16; break;
    case 3: table = async ? 0x21f8 : 0x500, limit = 16; break;
    case 4: table = async ? 0x2118 : 0x2098, limit = 8; break;
    default: return 0;
    }
    if (slot >= limit) {
        return 0;
    }
    const u64 object = Rd<u64>(ctx, table + slot * 8);
    return object != 0 && IsHeapPointer(object) && (object & 7) == 0 ? object : 0;
}

/// Binds and records one dispatch of a compiled variant into the frame stream (mutex held)
// Debug frame dump (ArmFrameDump, /vkb_dump_frame): the images of the dispatches of a window of graphics submissions, once each
struct FrameDumpState {
    std::mutex mutex;
    std::atomic<bool> armed{false};
    u64 from = 0, to = 0; ///< graphics submissions [from, to)
    u32 min_width = 0, max_images = 0, images = 0, dispatches = 0;
    std::unordered_set<u64> seen; ///< engine view ids already dumped
    std::string log;              ///< one line per dumped image, in order (written as log/vkb_image_dump/frame_dump.txt at the end)
};
FrameDumpState frame_dump;
void FrameDumpDispatch(u64 ctx, const Variant& v, std::vector<std::pair<u64, std::string>>& dump_views) {
    if (!frame_dump.armed.load(std::memory_order_relaxed)) [[likely]] {
        return;
    }
    std::scoped_lock lock{frame_dump.mutex};
    const u64 submit = Core::Vkb::GraphicsSubmits();
    if (submit < frame_dump.from) {
        return;
    }
    if (submit >= frame_dump.to || frame_dump.images >= frame_dump.max_images) {
        frame_dump.armed.store(false, std::memory_order_relaxed);
        const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_image_dump";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (FILE* f = std::fopen((dir / "frame_dump.txt").string().c_str(), "w")) {
            std::fputs(frame_dump.log.c_str(), f);
            std::fclose(f);
        }
        LOG_INFO(Render_Vulkan, "vkb frame dump: {} images from {} dispatches written", frame_dump.images, frame_dump.dispatches);
        return;
    }
    const u32 index = frame_dump.dispatches++;
    for (u32 i = 0; i < v.images.size() && frame_dump.images < frame_dump.max_images; ++i) {
        const Source& src = v.images[i];
        const u64 object = src.cls == Source::Class::Slot ? SlotObject(ctx, src.kind, src.slot, v.async) : 0;
        if (object == 0) {
            continue;
        }
        const u64 view_id = Rd<u64>(object, 0x10);
        u32 w = 0, h = 0, d = 1;
        u64 texture_id = 0;
        const bool written = v.info->images[i].is_written;
        // screen-sized images and every volume; a written one each time it is written (later writes into the lit image), a read one once
        if (!Core::Vkb::ViewExtent(view_id, w, h, &d, &texture_id) || (w < frame_dump.min_width && d <= 1) || (!frame_dump.seen.insert(view_id).second && !written)) {
            continue;
        }
        const std::string name = fmt::format("fd{:03}_d{:04}_cs{:08x}_i{}{}_v{}", frame_dump.images, index, v.info->pgm_hash, i, written ? "W" : "R", view_id);
        dump_views.push_back({view_id, name});
        std::string extra;
        if (d > 1 || h < 2160) { // a volume or a table (not a screen target): the engine's own copy (guest memory, in its PS4 layout) and its T#, to compare with ours
            Core::NativeDevice::Object view_object, texture;
            const bool view_known = Core::NativeDevice::Lookup(view_id, view_object), texture_known = Core::NativeDevice::Lookup(texture_id, texture);
            extra = fmt::format("  lookup view {} texture {} (id {}) address {:#x} bytes {} kind {} T# {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}", view_known,
                                texture_known, texture_id, texture.address, texture.bytes, static_cast<u32>(texture.kind), view_object.tsharp[0], view_object.tsharp[1],
                                view_object.tsharp[2], view_object.tsharp[3], view_object.tsharp[4], view_object.tsharp[5], view_object.tsharp[6], view_object.tsharp[7]);
            if (texture_known && texture.address != 0 && texture.bytes != 0 && texture.bytes <= (256ull << 20)) {
                const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_image_dump";
                std::error_code ec;
                std::filesystem::create_directories(dir, ec);
                if (FILE* f = std::fopen((dir / fmt::format("{}_guest_{:#x}_{}B.raw", name, texture.address, texture.bytes)).string().c_str(), "wb")) {
                    std::fwrite(reinterpret_cast<const void*>(texture.address), 1, texture.bytes, f);
                    std::fclose(f);
                }
                extra += "  guest copy written";
            }
        }
        frame_dump.log += fmt::format("{}  submit {}  dispatch {}  shader {:#010x}  image {} {}  {}{}\n", name, submit, index, v.info->pgm_hash, i, written ? "written" : "read",
                                      Core::Vkb::DescribeView(view_id), extra);
        ++frame_dump.images;
    }
}


/// Lantern census (debug, TLG_VKB_LANTERN_CENSUS=1): the four particle shaders tabnul's lantern change skips the waits between (0x1171c360, 0xa4f2ed2d, 0x82c47ba4,
/// 0x7a83ef95; note 38: data-indexed read-modify-write). For an exact rule (a wait only where two dispatches really touch the same entries) the CPU must be able to read
/// each dispatch's index data before it runs. Per (shader, binding): binds, written by the shader, kept on our device (the CPU's copy is stale), overlapping a range some
/// compute dispatch wrote earlier (GPU-written), CPU-readable (none of these). Plus a binary dump of the first dispatches (log/lantern_census.bin): per dispatch its
/// shader, groups, the first 128 dwords of its constant buffer 0 and every read-only binding that is CPU-readable (its first 64 KB), for an offline overlap check.
namespace {
constexpr std::array<u64, 4> LanternShaders{0x1171c360, 0xa4f2ed2d, 0x82c47ba4, 0x7a83ef95};
bool LanternCensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_LANTERN_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
struct LanternStat {
    u64 binds{}, written{}, on_device{}, gpu_written{}, cpu_readable{}, size_max{};
    u32 kind{}, slot{};
};
std::mutex lantern_mutex;
std::map<std::pair<u64, u32>, LanternStat> lantern_stats;
struct WrittenRange {
    u64 begin, end, submit;
};
std::vector<WrittenRange> lantern_written; ///< ranges compute dispatches wrote (any shader), newest last, kept 8 submits
u64 lantern_dispatches = 0, lantern_dumped = 0, lantern_bytes = 0;
std::set<std::pair<u64, u64>> lantern_snaps; ///< (submit, address) already written
std::vector<std::pair<u64, u64>> lantern_state; ///< the ranges the lantern shaders write (their particle state)
std::map<std::string, u64> lantern_readers; ///< other shaders that read those ranges: 'cs <hash> b<binding>' / 'draw vs <hash> ps <hash> s<stage> slot <n>'
u64 lantern_hit_base = 0; ///< the state range the last hit fell in
bool LanternStateHit(u64 address, u64 size) {
    for (const auto& [b, e] : lantern_state) {
        if (b < address + size && address < e) {
            lantern_hit_base = b;
            return true;
        }
    }
    return false;
}
std::FILE* lantern_file = nullptr;
void LanternNote(u64 hash, const Variant& v, const boost::container::small_vector<std::pair<u64, u64>, 16>& guest_of, u32 x, u32 y, u32 z) {
    const u64 submit = Core::Vkb::GraphicsSubmits();
    const bool lantern = std::find(LanternShaders.begin(), LanternShaders.end(), hash) != LanternShaders.end();
    std::scoped_lock lock{lantern_mutex};
    // every dispatch's written ranges (the GPU-written check)
    while (!lantern_written.empty() && lantern_written.front().submit + 8 < submit) {
        lantern_written.erase(lantern_written.begin());
    }
    if (lantern_written.size() > 200000) {
        lantern_written.erase(lantern_written.begin(), lantern_written.begin() + 100000);
    }
    if (!lantern) {
        for (u32 i = 0; i < guest_of.size() && i < v.info->buffers.size(); ++i) {
            if (v.info->buffers[i].is_written && guest_of[i].first != 0) {
                lantern_written.push_back({guest_of[i].first, guest_of[i].first + guest_of[i].second, submit});
            }
            if (guest_of[i].first != 0 && LanternStateHit(guest_of[i].first, guest_of[i].second)) {
                ++lantern_readers[fmt::format("cs {:#x} b{}{} reads {:#x}+{:#x} (state {:#x})", hash, i, v.info->buffers[i].is_written ? " W" : "", guest_of[i].first, guest_of[i].second, lantern_hit_base)];
            }
        }
        return;
    }
    ++lantern_dispatches;
    const bool dump = lantern_dumped < 40000;
    if (dump && lantern_file == nullptr) {
        const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "lantern_census.bin";
        lantern_file = std::fopen(path.string().c_str(), "wb");
    }
    std::vector<u8> record;
    const auto put = [&](const void* p, size_t n) { record.insert(record.end(), static_cast<const u8*>(p), static_cast<const u8*>(p) + n); };
    if (dump) {
        const u32 magic = 0x4c414e54; // 'LANT'
        put(&magic, 4);
        put(&submit, 8);
        put(&hash, 8);
        put(&x, 4), put(&y, 4), put(&z, 4);
    }
    u32 cb0_binding = ~0u;
    for (u32 i = 0; i < v.buffers.size() && i < guest_of.size() && i < v.info->buffers.size(); ++i) {
        const Source& src = v.buffers[i];
        const bool written = v.info->buffers[i].is_written;
        const u64 address = guest_of[i].first, size = guest_of[i].second;
        LanternStat& s = lantern_stats[{hash, i}];
        s.kind = src.cls == Source::Class::Slot ? src.kind : 100 + static_cast<u32>(src.cls);
        s.slot = src.slot;
        ++s.binds;
        s.size_max = std::max(s.size_max, size);
        if (src.cls == Source::Class::Slot && src.kind == 3 && src.slot == 0) {
            cb0_binding = i;
        }
        if (written) {
            ++s.written;
        }
        bool on_device = false, gpu_written = false;
        if (address != 0 && size != 0) {
            on_device = Core::Vkb::BufferOnDevice(address, size);
            for (const WrittenRange& r : lantern_written) {
                if (r.begin < address + size && address < r.end) {
                    gpu_written = true;
                    break;
                }
            }
        }
        s.on_device += on_device ? 1 : 0;
        s.gpu_written += gpu_written ? 1 : 0;
        const bool readable = !written && !on_device && !gpu_written && address != 0 && size != 0;
        s.cpu_readable += readable ? 1 : 0;
        if (written && address != 0) {
            lantern_written.push_back({address, address + size, submit});
            if (std::find(lantern_state.begin(), lantern_state.end(), std::pair<u64, u64>{address, address + size}) == lantern_state.end()) {
                lantern_state.push_back({address, address + size});
            }
        }
    }
    if (dump) {
        std::array<u32, 128> cb0{};
        if (cb0_binding == ~0u && !guest_of.empty() && guest_of[0].first != 0 && guest_of[0].second <= 4096) {
            cb0_binding = 0;
        }
        if (cb0_binding != ~0u && guest_of[cb0_binding].first != 0) {
            const u64 n = std::min<u64>(guest_of[cb0_binding].second, sizeof(cb0));
            std::memcpy(cb0.data(), reinterpret_cast<const void*>(guest_of[cb0_binding].first), n);
        }
        put(cb0.data(), sizeof(cb0));
        u32 count = 0;
        for (u32 i = 0; i < guest_of.size() && i < v.info->buffers.size(); ++i) {
            count += i != cb0_binding && guest_of[i].first != 0 ? 1 : 0;
        }
        put(&count, 4);
        for (u32 i = 0; i < guest_of.size() && i < v.info->buffers.size(); ++i) {
            if (i == cb0_binding || guest_of[i].first == 0) {
                continue;
            }
            const u64 address = guest_of[i].first, size = guest_of[i].second;
            const u32 written = v.info->buffers[i].is_written ? 1 : 0;
            const u32 device = Core::Vkb::BufferOnDevice(address, size) ? (Core::Vkb::BufferGpuWrites(address, size) ? 1u : 2u) : 0u;
            put(&i, 4), put(&address, 8), put(&size, 8), put(&written, 4), put(&device, 4);
            // a CPU-readable buffer's bytes once per submission (up to 8 MB; the file stops at 768 MB)
            if (!written && device != 1 && lantern_snaps.insert({submit, address}).second && lantern_bytes < (768ull << 20)) {
                const u32 bytes = static_cast<u32>(std::min<u64>(size, 8ull << 20));
                std::vector<u8> snap;
                const u32 smagic = 0x534e4150; // 'SNAP'
                snap.insert(snap.end(), reinterpret_cast<const u8*>(&smagic), reinterpret_cast<const u8*>(&smagic) + 4);
                snap.insert(snap.end(), reinterpret_cast<const u8*>(&submit), reinterpret_cast<const u8*>(&submit) + 8);
                snap.insert(snap.end(), reinterpret_cast<const u8*>(&address), reinterpret_cast<const u8*>(&address) + 8);
                snap.insert(snap.end(), reinterpret_cast<const u8*>(&bytes), reinterpret_cast<const u8*>(&bytes) + 4);
                snap.insert(snap.end(), reinterpret_cast<const u8*>(address), reinterpret_cast<const u8*>(address) + bytes);
                if (lantern_file != nullptr) {
                    std::fwrite(snap.data(), 1, snap.size(), lantern_file);
                }
                lantern_bytes += snap.size();
            }
        }
        if (lantern_file != nullptr) {
            std::fwrite(record.data(), 1, record.size(), lantern_file);
            if (++lantern_dumped % 500 == 0 || lantern_dumped == 40000) {
                std::fflush(lantern_file);
            }
        }
    }
    if (lantern_dispatches % 3000 == 1) {
        std::string text;
        for (const auto& [key, s] : lantern_stats) {
            text += fmt::format(" | {:#x} b{} k{} s{}: binds {} written {} device {} gpu_written {} cpu_readable {} max {}B", key.first, key.second, s.kind, s.slot, s.binds,
                                s.written, s.on_device, s.gpu_written, s.cpu_readable, s.size_max);
        }
        for (const auto& [name, n] : lantern_readers) {
            text += fmt::format(" || reader {}: {}", name, n);
        }
        LOG_WARNING(Render_Vulkan, "lantern census: {} dispatches, {} dumped{}", lantern_dispatches, lantern_dumped, text);
    }
}
} // namespace
/// Lantern census: a draw's buffer view reads the lantern shaders' particle state (called by the graphics side)
void LanternDrawReadImpl(u64 address, u64 size, u64 vs_hash, u64 ps_hash, u32 stage, u32 slot) {
    std::scoped_lock lock{lantern_mutex};
    if (LanternStateHit(address, size)) {
        ++lantern_readers[fmt::format("draw vs {:#x} ps {:#x} s{} slot {} reads {:#x}+{:#x} (state {:#x})", vs_hash, ps_hash, stage, slot, address, size, lantern_hit_base)];
    }
}
void Record(u64 ctx, u64 shader, Variant& v, u32 x, u32 y, u32 z, u64 indirect_address) {
    if (v.grid_x != 0 && indirect_address == 0 && x == 1 && y == 1 && z == 1) {
        x = v.grid_x;
        c_native_cull_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
    if (Core::Vkb::FillReadersOn() || Core::Vkb::R3dTraceSize() != 0) {
        Core::Vkb::CensusTag((1ull << 63) | shader); // note 49 census: compute, the engine's shader id
    }
    if (v.multi_binding) {
        c_skip_multi_binding.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 t0 = DispatchTimesOn() ? NowNs() : 0; // note 60: no clock reads per dispatch with the timers off
    const u64 alignment = instance->StorageMinAlignment();
    VkbShader::PushData push{};
    TlgSettings::LeafAoLive(push.xoffset, push.yoffset); // the leaf ambient occlusion pass reads them (PushData::LeafAoStrengthIndex/RadiusIndex)
    boost::container::small_vector<vk::DescriptorBufferInfo, 16> buffer_infos;
    boost::container::small_vector<vk::DescriptorImageInfo, 16> image_infos;
    boost::container::small_vector<vk::WriteDescriptorSet, 32> writes;
    const vk::CommandBuffer cmd{}; // V4g: resolved here; its commands are recorded by the recording thread (the job below)
    boost::container::small_vector<Stream::Use, 32> uses; // V4d: what the dispatch reads and writes
    boost::container::small_vector<std::pair<u64, u32>, 4> gds_inits; // GDS offset, start value
    boost::container::small_vector<u8, 16> stageable; // V4h: per buffer binding, read only in place
    // buffers: the engine objects' live ranges, ours in device memory for a buffer the GPU writes (V2f), else where they are in guest memory; the remainder of the offset below
    // the storage alignment goes through the push data
    // R5 (note 38): per binding, its guest range and its entry in `uses` (the written range is narrowed below when the shader proves a write window)
    boost::container::small_vector<std::pair<u64, u64>, 16> guest_of(v.buffers.size(), {0, 0});
    boost::container::small_vector<u32, 16> use_of(v.buffers.size(), ~0u);
    const auto bind_window = [&](u32 index, u64 address, u64 size, const Core::Vkb::BufferWindow& w) {
        if (index < guest_of.size()) {
            guest_of[index] = {address, size};
            use_of[index] = static_cast<u32>(uses.size());
        }
        const u64 aligned = w.offset & ~(alignment - 1);
        push.buf_offsets[index] = static_cast<u8>(w.offset - aligned);
        buffer_infos.push_back({w.buffer, aligned, size + (w.offset - aligned)});
        stageable.resize(buffer_infos.size());
        stageable.back() = !w.device && !v.info->buffers[index].is_written ? 1 : 0;
        uses.push_back(Stream::BufferUse(w.buffer, w.offset, size, v.info->buffers[index].is_written));
        Core::Vkb::NoteBind(0, w, size, v.info->buffers[index].is_written);
        if (Core::Vkb::R3dTraceSize() != 0 && size == Core::Vkb::R3dTraceSize()) { // debug R3d trace: the traced buffer bound to a compute shader
            static std::mutex last_mutex;
            static std::map<std::pair<u64, u32>, std::pair<u64, u64>> last; // (shader, binding) -> (address, binds)
            bool changed = false;
            u64 count = 0;
            {
                std::scoped_lock lock{last_mutex};
                auto& [a, c] = last[{shader, index}];
                changed = a != address;
                a = address;
                count = ++c;
            }
            if (changed || count % 10 == 0) {
                LOG_WARNING(Render_Vulkan, "vkb R3d trace: compute {:#x} binding {} at {:#x} ({} bytes): written {}, device copy {}, bind #{} submit {}", shader, index, address, size,
                            v.info->buffers[index].is_written, w.device, count, Core::Vkb::GraphicsSubmits());
            }
            // the input's device copy as the dispatch sees it, at a few of its binds after a first-bind copy (its 3rd, 30th and 300th bind of that buffer)
            static std::unordered_map<u64, u32> binds_of;
            static std::mutex binds_mutex;
            u32 n = 0;
            {
                std::scoped_lock lock{binds_mutex};
                n = ++binds_of[address];
            }
            static const bool dump_on = [] { const char* e = std::getenv("TLG_VKB_R3D_DUMP"); return e != nullptr && e[0] == '1'; }();
            if (dump_on && !v.info->buffers[index].is_written && w.device && (n == 3 || n == 30 || n == 300)) {
                const Core::Vkb::BufferWindow dw = w;
                const u64 bytes = size, at = address, submit = Core::Vkb::GraphicsSubmits();
                Stream::Defer(Stream::Work::Other, [dw, bytes, at, n, submit](vk::CommandBuffer cmd) {
                    Core::Vkb::DumpBuffer(cmd, dw.buffer, dw.offset, bytes, fmt::format("r3d/{:x}_s{}_dev_bind{}", at, submit, n));
                });
            }
        }
        return true;
    };
    const auto bind_range = [&](u32 index, u64 address, u64 size) {
        Core::Vkb::BufferWindow w;
        return Core::Vkb::FindBuffer(address, size, w) && bind_window(index, address, size, w);
    };
    // R7 (note 56): a constant buffer renamed outside the frame ring: its bytes copied now, at the dispatch
    const auto copy_at_dispatch = [&](u32 index, u64 address, u64 size) {
        const Stream::Slice slice = Stream::HostRing(size, alignment);
        std::memcpy(slice.data, reinterpret_cast<const void*>(address), size);
        if (index < guest_of.size()) {
            guest_of[index] = {address, size};
        }
        push.buf_offsets[index] = 0;
        buffer_infos.push_back({slice.buffer, slice.offset, size});
        stageable.resize(buffer_infos.size());
        stageable.back() = 0;
        return true;
    };
    // R7 (note 56): a slot's buffer by its engine object (constant buffer: where the engine renamed it; buffer view and buffer UAV: their buffer, whole, from offset 0)
    const auto bind_object = [&](u32 index, const Source& src) {
        const u64 object = SlotObject(ctx, src.kind, src.slot, v.async);
        if (object == 0) {
            return false;
        }
        Core::Vkb::BufferWindow w;
        if (src.kind == 3) {
            const u64 address = static_cast<u64>(Rd<u32>(object, 0x60)) | static_cast<u64>(Rd<u32>(object, 0x64) & 0xffu) << 32;
            const u64 size = Rd<u32>(object, 0x180);
            if (address == 0 || size == 0) {
                return false;
            }
            if (Core::Vkb::ConstantInRing(address, size, w)) {
                return bind_window(index, address, size, w);
            }
            if (!v.info->buffers[index].is_written && size <= 65536 && !Core::Vkb::R7NoCopy()) {
                Core::Vkb::NoteConstantCopied();
                return copy_at_dispatch(index, address, size);
            }
            return bind_range(index, address, size);
        }
        const bool view = (src.kind == 1 && (Rd<u32>(object, 0x44) == 1 || Rd<u32>(object, 0x44) == 0xb)) || (src.kind == 4 && Rd<u32>(object, 0x44) == 1);
        const u64 buffer = view ? Rd<u64>(object, 0x30) : 0;
        const u64 bytes = buffer != 0 ? Rd<u32>(buffer, 0x180) : 0;
        u64 offset = 0, size = bytes; // api-contract.md V4 / review A6: the view's own range of its buffer (SRV kind 1, UAV kind 4)
        Core::Vkb::ViewRange(object, src.kind == 1, buffer, bytes, offset, size);
        return bytes != 0 && Core::Vkb::BufferOfObject(buffer, offset, size, w) && bind_window(index, Rd<u64>(buffer, 0x30) + offset, size, w);
    };
    for (u32 i = 0; i < v.buffers.size(); ++i) {
        const Source& src = v.buffers[i];
        const auto& desc = v.info->buffers[i];
        bool ok = false;
        if (src.cls == Source::Class::Slot) {
            ok = bind_object(i, src);
        } else if (src.cls == Source::Class::ProgramConstant) {
            ok = src.size != 0 && bind_range(i, src.address, src.size);
        } else if (src.cls == Source::Class::Special) {
            switch (desc.buffer_type) {
            case VkbShader::BufferType::GdsBuffer:
                buffer_infos.push_back({Stream::GdsBuffer(), 0, VK_WHOLE_SIZE});
                uses.push_back(Stream::BufferUse(Stream::GdsBuffer(), 0, VK_WHOLE_SIZE, desc.is_written));
                ok = true;
                break;
            case VkbShader::BufferType::Flatbuf: {
                // RESIDUAL (stated in the plan's notes): the flat user data is the shader's own table walk over descriptor words, rebuilt from the engine objects per dispatch
                BuildLayout(ctx, shader, v.async, false, v.live);
                v.info->user_data = std::span<const u32>{v.live.sgpr.data(), v.live.sgpr.size()};
                v.info->RefreshFlatBuf();
                const u64 bytes = std::max<u64>(v.info->flattened_ud_buf.size() * sizeof(u32), 16);
                const Stream::Slice slice = Stream::HostRing(bytes, alignment);
                std::memset(slice.data, 0, bytes);
                std::memcpy(slice.data, v.info->flattened_ud_buf.data(), v.info->flattened_ud_buf.size() * sizeof(u32));
                buffer_infos.push_back({slice.buffer, slice.offset, bytes});
                c_flat_rebuilt.fetch_add(1, std::memory_order_relaxed);
                ok = true;
                break;
            }
            case VkbShader::BufferType::SharedMemory: {
                const u64 groups = indirect_address != 0 ? 0 : static_cast<u64>(x) * y * z;
                const u64 bytes = static_cast<u64>(v.program.SharedMemSize()) * groups;
                if (bytes == 0 || bytes > (8ull << 20)) {
                    c_skip_shared_memory.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                const Stream::Slice slice = Stream::HostRing(bytes, alignment);
                std::memset(slice.data, 0, bytes);
                buffer_infos.push_back({slice.buffer, slice.offset, bytes});
                ok = true;
                break;
            }
            default:
                c_skip_special.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        if (!ok) {
            c_skip_buffer.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    if (LanternCensusOn() && v.info) {
        LanternNote(v.info->pgm_hash, v, guest_of, x, y, z);
    }
    // lantern waits, exact rule: the records this dispatch processes (constant buffer binding 0: count at dword 104, first at 106; thread i takes record first + i while
    // i < count) and the particle slot each names (binding 1, the CPU-written emitter records of 60 dwords: slot at dword 6); all four shaders have this layout
    Stream::LanternFootprint lantern{};
    if (v.info && Stream::IsLanternShader(v.info->pgm_hash) && guest_of.size() >= 2 && v.buffers.size() >= 2 && v.buffers[0].cls == Source::Class::Slot && v.buffers[0].kind == 3 &&
        v.buffers[1].cls == Source::Class::Slot && v.buffers[1].kind == 1 && guest_of[0].first != 0 && guest_of[0].second >= 107 * 4 && guest_of[1].first != 0 && indirect_address == 0) {
        const u32 count = *reinterpret_cast<const u32*>(guest_of[0].first + 104 * 4);
        const u32 first = *reinterpret_cast<const u32*>(guest_of[0].first + 106 * 4);
        const u32 n = count; // (every record the guard lets through: a superset of what the threads reach, so the footprint is never too small)
        bool ok = n <= 4096 && (static_cast<u64>(first) + n) * 240 <= guest_of[1].second;
        for (u32 g = 0; ok && g < n; ++g) {
            const u32 slot = *reinterpret_cast<const u32*>(guest_of[1].first + (static_cast<u64>(first + g) * 60 + 6) * 4);
            if (slot == 0) {
                continue;
            }
            if (lantern.slot_count == lantern.slots.size()) {
                ok = false; // more slots than the footprint holds: keep the wait
                break;
            }
            lantern.slots[lantern.slot_count++] = slot;
        }
        lantern.known = ok;
        lantern.record_begin = first, lantern.record_end = first + n;
    }
    if (WriteWindowsOn() && !v.windows.empty()) {
        const auto read_cb = [&](u32 binding, u32 index, u64& out) {
            if (binding >= guest_of.size() || guest_of[binding].first == 0 || (static_cast<u64>(index) + 1) * 4 > guest_of[binding].second) {
                return false;
            }
            out = *reinterpret_cast<const u32*>(guest_of[binding].first + static_cast<u64>(index) * 4);
            return true;
        };
        for (u32 i = 0; i < v.windows.size() && i < use_of.size(); ++i) {
            if (!v.windows[i].valid || use_of[i] == ~0u || use_of[i] >= uses.size()) {
                continue;
            }
            Stream::Use& u = uses[use_of[i]];
            const u64 misalign = u.begin - (u.begin & ~(alignment - 1));
            u64 lo = 0, hi = 0;
            if ((misalign & 3) != 0 || u.end == ~0ull ||
                !Core::Vkb::WriteWindow::Evaluate(v.windows[i], v.program.num_thread_x.full, indirect_address != 0 ? 0u : x, read_cb, lo, hi)) {
                c_window_whole.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            const u64 size = u.end - u.begin;
            const u64 b = std::min(lo * 4, size), e = std::min(hi * 4, size);
            u.end = u.begin + e;
            u.begin = u.begin + std::min(b, e);
            c_window_narrowed.fetch_add(1, std::memory_order_relaxed);
            if (u.begin == u.end) {
                c_window_empty.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    // images: our view of the engine view in the slot (its image in the general layout)
    for (u32 i = 0; i < v.images.size(); ++i) {
        const Source& src = v.images[i];
        const u64 object = src.cls == Source::Class::Slot ? SlotObject(ctx, src.kind, src.slot, v.async) : 0;
        vk::ImageView view{};
        vk::Image image{};
        if (object == 0 || !Core::Vkb::ViewForCompute(Rd<u64>(object, 0x10), v.image_types[i], cmd, view, &image)) {
            c_skip_image.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        image_infos.push_back({{}, view, vk::ImageLayout::eGeneral});
        uses.push_back(Stream::ImageUse(image, v.info->images[i].is_written));
    }
    // samplers: ours from the engine sampler's description
    boost::container::small_vector<vk::DescriptorImageInfo, 8> sampler_infos;
    const Core::Vkb::Aa::SamplerLook look = Core::Vkb::Aa::LookFor(0); // issue 42: compute gets the anisotropy setting, never the scene bias (their rule)
    for (u32 si = 0; si < v.samplers.size(); ++si) {
        const Source& src = v.samplers[si];
        const u64 object = src.cls == Source::Class::Slot ? SlotObject(ctx, src.kind, src.slot, v.async) : 0;
        const bool compare = si < v.info->samplers.size() && v.info->samplers[si].is_depth;
        const Core::Vkb::Aa::SamplerLook own = Core::Vkb::Aa::ForSampler(look, compare);
        const vk::Sampler sampler = object != 0 ? Core::Vkb::SamplerFor(object, compare, own.bias_q, own.aniso) : vk::Sampler{};
        if (!sampler) {
            c_skip_sampler.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        sampler_infos.push_back({sampler, {}, vk::ImageLayout::eUndefined});
    }
    vk::Buffer args_buffer{};
    u64 args_offset = 0;
    if (indirect_address != 0) {
        Core::Vkb::BufferWindow w;
        if (!Core::Vkb::FindBuffer(indirect_address, 12, w) || (w.offset & 3) != 0) {
            c_skip_indirect_args.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        args_buffer = w.buffer, args_offset = w.offset;
        uses.push_back(Stream::BufferUse(w.buffer, w.offset, 12, false));
    }
    boost::container::small_vector<vk::DescriptorType, 16> image_types;
    for (u32 i = 0; i < image_infos.size(); ++i) {
        image_types.push_back(v.info->images[i].is_written ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage);
    }
    // append / consume counters that start at a value: our GDS gets it in front of the dispatch (the engine's DMA packet on the console). Both contexts do it:
    // the async one (PrepareDispatch 0246f8e0: GDS base ctx + 0x22f8, UAVs + 0x2118, initial counts + 0x2158, cleared to -1 once written) and the device context
    // (api-contract.md W7, FUN_0247dad0: GDS base ctx + 0x130b8, UAVs + 0x2098, initial counts + 0x20d8, written before every dispatch until the UAVs are set again)
    {
        const u64 shader_obj = shader;
        const u32 counter_base = Rd<u32>(ctx, v.async ? 0x22f8 : 0x130b8);
        const u64 uavs = v.async ? 0x2118 : 0x2098, inits = v.async ? 0x2158 : 0x20d8;
        for (u32 mask = Rd<u32>(shader_obj, 0x100), index = 0; mask != 0 && index < 8; mask >>= 1, ++index) {
            if ((mask & 1) == 0) {
                continue;
            }
            const u64 view = Rd<u64>(ctx, uavs + index * 8);
            if (view == 0 || !IsHeapPointer(view) || Rd<u32>(view, 0x44) != 1 || (Rd<u8>(view, 0x50) & 2) == 0) {
                continue;
            }
            if (const u32 init = Rd<u32>(ctx, inits + index * 4); init != 0xffffffffu && counter_base + index * 4ull + 4 <= (64u << 10)) {
                gds_inits.push_back({counter_base + index * 4ull, init});
                c_gds_inits.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    // api-contract.md W7 census (debug): device dispatches with a counter UAV
    if (!v.async && Core::Vkb::ContractCensusOn()) {
        for (u32 index = 0; index < 8; ++index) {
            const u64 view = Rd<u64>(ctx, 0x2098 + index * 8);
            if (view != 0 && IsHeapPointer(view) && Rd<u32>(view, 0x44) == 1 && (Rd<u8>(view, 0x50) & 2) != 0) {
                Core::Vkb::ContractNote(Rd<u32>(ctx, 0x20d8 + index * 4) != 0xffffffffu ? "W7 device dispatch with a counter UAV and an initial count"
                                                                                          : "W7 device dispatch with a counter UAV, count kept");
            }
        }
    }
    static std::unordered_map<u64, bool> logged_shaders; // a developer dump (Vkb::DevLog): what each new shader binds; the shaders are counted at their first variant
    if (const u64 sid = Rd<u64>(shader, 0x10); DevLog::On() && !logged_shaders[sid]) {
        logged_shaders[sid] = true;
        std::string text;
        for (u32 i = 0; i < v.buffers.size(); ++i) {
            const auto& d = v.info->buffers[i];
            text += fmt::format(" b{}[{} k{} s{} type{} {} {}B]", i, static_cast<u32>(v.buffers[i].cls), v.buffers[i].kind, v.buffers[i].slot, static_cast<u32>(d.buffer_type),
                                d.is_written ? "W" : "R", i < buffer_infos.size() ? buffer_infos[i].range : 0);
        }
        for (u32 i = 0; i < v.images.size(); ++i) {
            const u64 object = v.images[i].cls == Source::Class::Slot ? SlotObject(ctx, v.images[i].kind, v.images[i].slot, v.async) : 0;
            text += fmt::format(" i{}[k{} s{} {} {}]", i, v.images[i].kind, v.images[i].slot, v.info->images[i].is_written ? "W" : "R",
                                object != 0 ? Core::Vkb::DescribeView(Rd<u64>(object, 0x10)) : std::string("no view")); // note 41: what each image is
        }
        for (u32 i = 0; i < v.samplers.size(); ++i) {
            const u64 object = v.samplers[i].cls == Source::Class::Slot ? SlotObject(ctx, v.samplers[i].kind, v.samplers[i].slot, v.async) : 0;
            text += fmt::format(" s{}[s{} {}{}]", i, v.samplers[i].slot, Core::Vkb::DescribeSampler(object), v.info->samplers[i].is_depth ? " cmp" : ""); // note 41
        }
        LOG_INFO(Render_Vulkan, "vkb bindings: shader {} hash {:#x} groups {}x{}x{} lds {} B:{}", sid, v.info->pgm_hash, x, y, z, v.program.SharedMemSize(), text);
    }
    // V4g: the commands, on the recording thread (in the order of the engine's calls)
    static const bool no_barrier = [] { const char* e = std::getenv("TLG_VKB_NO_DISPATCH_BARRIER"); return e != nullptr && e[0] == '1'; }(); // kill-experiment
    static const bool no_dispatch = [] { const char* e = std::getenv("TLG_VKB_NO_DISPATCH"); return e != nullptr && *e == '1'; }(); // kill experiment: everything but the dispatch
    // note 41, debug only: TLG_VKB_IMAGE_DUMP_CS=<shader hash> (TLG_VKB_IMAGE_DUMP_AT=<graphics submit>, default 2600; TLG_VKB_IMAGE_DUMP_NTH=<n>, default 0: the n-th
    // such dispatch from then on) dumps its images right after the dispatch, once
    std::vector<std::pair<u64, std::string>> dump_views;
    {
        static const u64 dump_cs = [] { const char* e = std::getenv("TLG_VKB_IMAGE_DUMP_CS"); return e != nullptr ? std::strtoull(e, nullptr, 16) : 0ull; }();
        static const u64 dump_at = [] { const char* e = std::getenv("TLG_VKB_IMAGE_DUMP_AT"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 2600ull; }();
        static const u64 dump_nth = [] { const char* e = std::getenv("TLG_VKB_IMAGE_DUMP_NTH"); return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull; }();
        static std::atomic<u64> seen{0};
        if (dump_cs != 0 && v.info && v.info->pgm_hash == dump_cs && Core::Vkb::GraphicsSubmits() >= dump_at && seen.fetch_add(1) == dump_nth) {
            for (u32 i = 0; i < v.images.size(); ++i) {
                const Source& src = v.images[i];
                const u64 object = src.cls == Source::Class::Slot ? SlotObject(ctx, src.kind, src.slot, v.async) : 0;
                if (object != 0) {
                    dump_views.push_back({Rd<u64>(object, 0x10), fmt::format("cs_i{}_k{}_s{}{}", i, src.kind, src.slot, v.info->images[i].is_written ? "_W" : "")});
                }
            }
            LOG_INFO(Render_Vulkan, "vkb image dump: dispatch of {:#x} groups {}x{}x{} async {}: {} images", dump_cs, x, y, z, v.async, dump_views.size());
            // the buffers as the engine's memory holds them now (what the CPU wrote; a buffer the GPU writes is not current here)
            const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_image_dump";
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            for (u32 i = 0; i < guest_of.size(); ++i) {
                if (guest_of[i].first != 0 && guest_of[i].second != 0 && guest_of[i].second <= (64ull << 20)) {
                    if (FILE* f = std::fopen((dir / fmt::format("cs_b{}_{:#x}_{}B.raw", i, guest_of[i].first, guest_of[i].second)).string().c_str(), "wb")) {
                        std::fwrite(reinterpret_cast<const void*>(guest_of[i].first), 1, guest_of[i].second, f);
                        std::fclose(f);
                    }
                }
            }
        }
    }
    const bool dump_buffers = !dump_views.empty(); // the TLG_VKB_IMAGE_DUMP_CS dump only (the frame dump takes images alone)
    FrameDumpDispatch(ctx, v, dump_views);
    if (v.info->pgm_hash == Core::Vkb::Aa::CombineCs) [[unlikely]] {
        Core::Vkb::Aa::FollowMenu(); // live AA switching: the in-game Anti-aliasing choice, read in game order
    }
    const bool aa_combine = v.info->pgm_hash == Core::Vkb::Aa::CombineCs && Core::Vkb::Aa::On();
    const std::array<float, 2> aa_jitter = aa_combine ? Core::Vkb::Aa::OnCombineQueued() : std::array<float, 2>{};
    const bool aa_cut = aa_combine && Core::Vkb::Aa::TakeFrameCut(); // K4: the frame's camera cut, taken in game order with its jitter
    Stream::Defer(Stream::Work::Dispatch, [aa_jitter, aa_cut, dump_views = std::move(dump_views), dump_buffers, uses = std::move(uses), gds_inits = std::move(gds_inits), buffer_infos = std::move(buffer_infos), image_infos = std::move(image_infos),
                                           sampler_infos = std::move(sampler_infos), image_types = std::move(image_types), pipeline = v.pipeline, layout = v.pipeline_layout, push,
                                           args_buffer, args_offset, x, y, z, sid = Rd<u64>(shader, 0x10), hash = v.info->pgm_hash,
                                           stageable = std::move(stageable), lantern, job_async = v.async](vk::CommandBuffer cmd) mutable {
        // the async context's append / consume counters that start at a value: our GDS gets it in front of the dispatch (the engine's DMA packet on the console)
        for (const auto& [offset, init] : gds_inits) {
            Stream::Hazard(cmd, {Stream::BufferUse(Stream::GdsBuffer(), offset, 4, true)}, Stream::Agent::Transfer);
            cmd.fillBuffer(Stream::GdsBuffer(), offset, 4, init);
        }
        if (hash == Core::Vkb::Aa::CombineCs && (Core::Vkb::Aa::On() || Core::Vkb::Aa::SwitchDue())) [[unlikely]] { // F1: native AA writes the scene colour the combine reads
            Core::Vkb::Aa::BeforeCombine(cmd, aa_jitter, aa_cut);
        }
        // V4d: a barrier only when the dispatch touches what earlier work since the last barrier wrote (or writes what it read)
        if (!no_barrier) {
            Stream::SetHazardTag(hash);
            Stream::SetLanternFootprint(lantern.known || Stream::IsLanternShader(hash) ? &lantern : nullptr);
            Stream::SetJobAsync(job_async);
            Stream::Hazard(cmd, std::span<const Stream::Use>{uses.data(), uses.size()});
            Stream::SetJobAsync(false);
            Stream::SetLanternFootprint(nullptr);
        }
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        for (u32 i = 0; i < buffer_infos.size() && i < stageable.size(); ++i) {
            if (stageable[i] != 0) {
                (void)Stream::StageRead(buffer_infos[i].buffer, buffer_infos[i].offset, buffer_infos[i].range);
            }
        }
        boost::container::small_vector<vk::WriteDescriptorSet, 32> writes;
        u32 binding = 0;
        for (u32 i = 0; i < buffer_infos.size(); ++i) {
            writes.push_back({.dstBinding = binding++, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &buffer_infos[i]});
        }
        for (u32 i = 0; i < image_infos.size(); ++i) {
            writes.push_back({.dstBinding = binding++, .descriptorCount = 1, .descriptorType = image_types[i], .pImageInfo = &image_infos[i]});
        }
        for (u32 i = 0; i < sampler_infos.size(); ++i) {
            writes.push_back({.dstBinding = binding++, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler, .pImageInfo = &sampler_infos[i]});
        }
        if (!writes.empty()) {
            cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
        }
        cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        const u32 stamp = Stream::Stamp(cmd);
        if (no_dispatch) {
        } else if (args_buffer) {
            cmd.dispatchIndirect(args_buffer, args_offset);
            c_indirect.fetch_add(1, std::memory_order_relaxed);
        } else {
            cmd.dispatch(x, y, z);
        }
        for (const auto& [id, name] : dump_views) {
            Core::Vkb::DumpView(cmd, id, name);
        }
        if (dump_buffers) {
            for (u32 i = 0; i < buffer_infos.size(); ++i) {
                Core::Vkb::DumpBuffer(cmd, buffer_infos[i].buffer, buffer_infos[i].offset, buffer_infos[i].range == VK_WHOLE_SIZE ? 0 : buffer_infos[i].range, fmt::format("cs_b{}", i));
            }
        }
        if (stamp != ~0u && Stream::Stamp(cmd) == stamp + 1) {
            std::scoped_lock timed_lock{timed_mutex};
            frame_timed.push_back({Stream::Generation(), sid, hash, stamp, x, y, z, args_buffer ? true : false});
        }
    });
    c_recorded.fetch_add(1, std::memory_order_relaxed);
    if (v.async) {
        c_async_recorded.fetch_add(1, std::memory_order_relaxed);
    }
    Core::Vkb::Census::Work(v.async ? 2 : 1, v.info ? v.info->pgm_hash : 0, x, y, z);
    // the glitch marker's and the frame walk's draw table: one row per dispatch (count / instances = groups x / y; draw class 3 compute)
    if (Core::Marker::Enabled()) {
        Core::Marker::NoteRow({.vs_hash = v.info ? v.info->pgm_hash : 0ull,
                               .frame = static_cast<u32>(Core::DebugServer::PresentCount()),
                               .count = x,
                               .instances = y,
                               .kind = static_cast<u8>(indirect_address != 0 ? 3 : 2),
                               .draw_class = 3,
                               .flags = static_cast<u8>(2 | (v.async ? 4 : 0))});
    }
    if (DispatchTimesOn()) {
        c_record_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    }
}

} // namespace

void Dispatch(u64 ctx, u32 x, u32 y, u32 z, u64 indirect_address) {
    c_dispatches.fetch_add(1, std::memory_order_relaxed);
    const u64 shader = Rd<u64>(ctx, 0x4670);
    if (shader == 0 || Rd<u64>(shader, 0x120) == 0) {
        c_no_shader.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (indirect_address == 0 && (x == 0 || y == 0 || z == 0)) {
        return;
    }
    const u64 k0 = DispatchTimesOn() ? NowNs() : 0;
    const u64 key = VariantKey(ctx, shader);
    if (DispatchTimesOn()) {
        c_key_ns.fetch_add(NowNs() - k0, std::memory_order_relaxed);
    }
    std::scoped_lock lock{mutex};
    if (!Ready()) {
        return;
    }
    auto [it, fresh] = variants.try_emplace(key);
    if (!fresh) {
        c_found.fetch_add(1, std::memory_order_relaxed);
    } else {
        it->second = std::make_unique<Variant>();
        if (++variants_per_shader[Rd<u64>(shader, 0x10)] == 1) {
            DevLog::Count(DevLog::Kind::ComputeShaderBindings); // the shaders the developer dump "vkb bindings" describes
        }
        Compile(ctx, shader, *it->second);
    }
    if (!it->second->ok) {
        c_skip_not_compiled.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (Core::FrameWalk::DebuggerMode() && it->second->info) [[unlikely]] { // note 89: the shaders of the scene for the freeze
        Core::Vkb::ShaderDump::NoteDispatch(it->second->info->pgm_hash, it->second->async);
    }
    static const bool no_record = [] { const char* e = std::getenv("TLG_VKB_NO_RECORD"); return e != nullptr && *e == '1'; }(); // kill experiment: compile only (V3a)
    // note 71 bound only: TLG_VKB_TEST_SKIP_HASHES=hex,... skips the device-context dispatches of those shaders (the async ones: TLG_VKB_TEST_SKIP_ASYNC_HASHES); the picture is wrong
    static const std::vector<u64> skip_device = [] {
        std::vector<u64> out;
        if (const char* e = std::getenv("TLG_VKB_TEST_SKIP_HASHES")) {
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
    if (!skip_device.empty() && it->second->info && std::find(skip_device.begin(), skip_device.end(), it->second->info->pgm_hash) != skip_device.end()) {
        return;
    }
    if (!no_record) {
        Record(ctx, shader, *it->second, x, y, z, indirect_address);
    }
}

void DispatchAsync(u64 jcc, u32 x, u32 y, u32 z) {
    c_async_dispatches.fetch_add(1, std::memory_order_relaxed);
    static const bool no_async = [] { const char* e = std::getenv("TLG_VKB_NO_ASYNC_DISPATCH"); return e != nullptr && e[0] == '1'; }(); // note 34 bound only: what overlap could hide
    if (no_async) {
        return;
    }
    const u64 shader = Rd<u64>(jcc, 0x1d10);
    if (shader == 0 || !IsHeapPointer(shader) || Rd<u64>(shader, 0x120) == 0 || x == 0 || y == 0 || z == 0) {
        c_no_shader.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 key = AsyncVariantKey(jcc, shader);
    std::scoped_lock lock{mutex};
    if (!Ready()) {
        return;
    }
    auto [it, fresh] = variants.try_emplace(key);
    if (!fresh) {
        c_found.fetch_add(1, std::memory_order_relaxed);
    } else {
        it->second = std::make_unique<Variant>();
        it->second->async = true;
        if (++variants_per_shader[Rd<u64>(shader, 0x10)] == 1) {
            DevLog::Count(DevLog::Kind::ComputeShaderBindings); // the shaders the developer dump "vkb bindings" describes
        }
        Compile(jcc, shader, *it->second);
        if (!it->second->ok && (it->second->layout.unsupported_types >> 31) != 0) {
            c_async_unsupported.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (!it->second->ok) {
        c_skip_not_compiled.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (Core::FrameWalk::DebuggerMode() && it->second->info) [[unlikely]] { // note 89: the shaders of the scene for the freeze
        Core::Vkb::ShaderDump::NoteDispatch(it->second->info->pgm_hash, it->second->async);
    }
    // note 39 bound only: TLG_VKB_TEST_SKIP_ASYNC_HASHES=hex,hex,... skips the async dispatches of those shaders (the picture is wrong)
    static const std::vector<u64> skip_hashes = [] {
        std::vector<u64> out;
        if (const char* e = std::getenv("TLG_VKB_TEST_SKIP_ASYNC_HASHES")) {
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
    if (!skip_hashes.empty() && it->second->info && std::find(skip_hashes.begin(), skip_hashes.end(), it->second->info->pgm_hash) != skip_hashes.end()) {
        return;
    }
    Record(jcc, shader, *it->second, x, y, z, 0);
}

nlohmann::json GpuTimeReport() {
    std::scoped_lock lock{timed_mutex};
    std::vector<std::pair<u64, ShaderTime>> all(shader_time.begin(), shader_time.end());
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
    nlohmann::json top = nlohmann::json::array();
    const double frames = static_cast<double>(std::max<u64>(Core::Vkb::GraphicsSubmits(), 1));
    for (size_t i = 0; i < all.size() && i < (Core::Vkb::Census::On() ? 200u : 15u); ++i) {
        const ShaderTime& st = all[i].second;
        top.push_back({{"shader_id", all[i].first}, {"pgm_hash", fmt::format("{:#x}", st.pgm_hash)}, {"ms_per_frame", static_cast<double>(st.ns) / 1e6 / frames},
                       {"dispatches", st.count}, {"indirect", st.indirect}, {"us_mean", static_cast<double>(st.ns) / 1e3 / static_cast<double>(st.count)},
                       {"us_max", static_cast<double>(st.max_ns) / 1e3}, {"groups_of_max", fmt::format("{}x{}x{}", st.max_x, st.max_y, st.max_z)}});
    }
    return {{"frames", Core::Vkb::GraphicsSubmits()}, {"timed_submissions", timed_submits}, {"dispatch_ms_per_frame", static_cast<double>(timed_ns) / 1e6 / frames},
            {"worst_submission_ms", static_cast<double>(timed_submit_max_ns) / 1e6}, {"top_shaders", top}};
}

nlohmann::json ArmFrameDump(u32 submits, u32 min_width, u32 max_images) {
    std::scoped_lock lock{frame_dump.mutex};
    frame_dump.from = Core::Vkb::GraphicsSubmits() + 1;
    frame_dump.to = frame_dump.from + std::max(submits, 1u);
    frame_dump.min_width = min_width, frame_dump.max_images = std::max(max_images, 1u);
    frame_dump.images = 0, frame_dump.dispatches = 0;
    frame_dump.seen.clear();
    frame_dump.log.clear();
    frame_dump.armed.store(true, std::memory_order_relaxed);
    return {{"armed_from_submit", frame_dump.from}, {"to_submit", frame_dump.to}, {"min_width", min_width}, {"max_images", frame_dump.max_images},
            {"folder", (Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vkb_image_dump").string()}};
}

void FrameDone() {
    if (!Stream::Ready()) {
        return;
    }
    const std::vector<Stream::Stamps> finished = Stream::FinishedStamps();
    if (finished.empty()) {
        return;
    }
    std::scoped_lock lock{timed_mutex};
    for (const Stream::Stamps& f : finished) {
        u64 ns_total = 0;
        for (const Timed& t : frame_timed) {
            if (t.generation != f.generation || t.begin + 1 >= f.ns.size() || f.ns[t.begin + 1] < f.ns[t.begin]) {
                continue;
            }
            const u64 ns = f.ns[t.begin + 1] - f.ns[t.begin];
            ns_total += ns;
            ShaderTime& st = shader_time[t.shader_id];
            st.pgm_hash = t.pgm_hash;
            st.ns += ns;
            ++st.count;
            st.indirect += t.indirect ? 1 : 0;
            if (ns > st.max_ns) {
                st.max_ns = ns, st.max_x = t.x, st.max_y = t.y, st.max_z = t.z;
            }
        }
        std::erase_if(frame_timed, [&](const Timed& t) { return t.generation <= f.generation; });
        timed_ns += ns_total;
        timed_submit_max_ns = std::max(timed_submit_max_ns, ns_total);
        ++timed_submits;
    }
}

u64 VariantsWithAWindow() {
    std::scoped_lock lock{mutex};
    u64 n = 0;
    for (const auto& [key, var] : variants) {
        if (var) {
            for (const auto& w : var->windows) {
                if (w.valid) {
                    ++n;
                    break;
                }
            }
        }
    }
    return n;
}

nlohmann::json Report() {
    const u64 compiled = c_compiled.load(), dispatches = c_dispatches.load();
    u64 shaders = 0, most = 0;
    {
        std::scoped_lock lock{mutex};
        shaders = variants_per_shader.size();
        for (const auto& [id, n] : variants_per_shader) {
            most = std::max<u64>(most, n);
        }
    }
    return {{"dispatches", dispatches},
            {"variant_found", c_found.load()},
            {"shaders", shaders},
            {"most_variants_of_one_shader", most},
            {"compiled", compiled},
            {"native_cull", {{"on", NativeCullOn()}, {"modules", c_native_cull_modules.load()}, {"dispatches", c_native_cull_dispatches.load()}}},
            {"failed", c_failed.load()},
            {"failed_unsupported_input_type", c_failed_unsupported_input.load()},
            {"failed_translation", c_failed_translation.load()},
            {"failed_pipeline", c_failed_pipeline.load()},
            {"failed_too_many_bindings", c_too_many_bindings.load()},
            {"compile_ms_mean", compiled ? static_cast<double>(c_compile_ns.load()) / 1e6 / static_cast<double>(compiled) : 0.0},
            {"variant_key_us_mean", dispatches ? static_cast<double>(c_key_ns.load()) / 1e3 / static_cast<double>(dispatches) : 0.0},
            {"no_shader", c_no_shader.load()},
            {"buffers", {{"slot", c_buf_slot.load()}, {"special", c_buf_special.load()}, {"program_constant", c_buf_constant.load()}, {"unmapped", c_buf_unmapped.load()}}},
            {"images", {{"slot", c_img_slot.load()}, {"unmapped", c_img_unmapped.load()}}},
            {"samplers", {{"slot", c_smp_slot.load()}, {"unmapped", c_smp_unmapped.load()}}},
            {"recorded", c_recorded.load()},
            {"recorded_indirect", c_indirect.load()},
            {"write_windows", {{"on", WriteWindowsOn()}, {"narrowed", c_window_narrowed.load()}, {"kept_whole", c_window_whole.load()}, {"empty", c_window_empty.load()}, {"variants_with_a_window", VariantsWithAWindow()}}},
            {"async_census", Core::Vkb::Census::Report()},
            {"async", {{"dispatches", c_async_dispatches.load()}, {"recorded", c_async_recorded.load()}, {"layouts_not_known", c_async_unsupported.load()},
                       {"gds_counter_inits", c_gds_inits.load()}}},
            {"record_us_mean", c_recorded.load() ? static_cast<double>(c_record_ns.load()) / 1e3 / static_cast<double>(c_recorded.load()) : 0.0},
            {"flat_user_data_rebuilt", c_flat_rebuilt.load()},
            {"skipped", {{"not_compiled", c_skip_not_compiled.load()}, {"image_per_mip_bindings", c_skip_multi_binding.load()}, {"buffer", c_skip_buffer.load()},
                         {"image", c_skip_image.load()}, {"image_view_type", c_skip_image_type.load()}, {"sampler", c_skip_sampler.load()},
                         {"indirect_arguments", c_skip_indirect_args.load()}, {"shared_memory", c_skip_shared_memory.load()}, {"other_special", c_skip_special.load()}}},
            {"stream", Stream::Report()},
            {"gpu_time", GpuTimeReport()}};
}

bool LanternCensusActive() {
    return LanternCensusOn();
}
void LanternDrawRead(u64 address, u64 size, u64 vs_hash, u64 ps_hash, u32 stage, u32 slot) {
    LanternDrawReadImpl(address, size, vs_hash, ps_hash, stage, slot);
}
} // namespace Core::Vkb::Compute
