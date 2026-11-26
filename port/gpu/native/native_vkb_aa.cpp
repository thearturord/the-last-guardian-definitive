// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/native_vkb_aa.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>

#include <nlohmann/json.hpp>
#include <vk_mem_alloc.h>

#include "native/shadps4/video_core/renderer_vulkan/vk_scheduler.h"
#include "common/logging/log.h"
#include "ffx_vk_portable.h"
#include "native/native_vkb_dlss_glue.h"
#include "native/native_vkb_fsr4.h"
#include "native/native_vkb_stream.h"
#include "native/native_vkb_objects.h"
#include "native/shadps4/video_core/host_shaders/vkb_aa_motion_comp.h"
#include "native/shadps4/video_core/host_shaders/vkb_rcas_comp.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_instance.h"
#include "tlg_overlay.h"
#include "tlg_settings.h"
#include "tlg_toggles.h"
// issue 57: the normal renderer's own TAA shader (compiled from their taa.comp in this build; only read, never changed)
#include "video_core/host_shaders/taa_comp.h"

namespace {
/// a device-wide wait: every queue must be left alone meanwhile (Vulkan), so the backend's recording and upload threads wait at the queue lock, as the
/// presenter's swapchain recreate does (a wait without it lost the device, owner play 2026-11-16)
void IdleDevice(vk::Device device) {
    std::scoped_lock queue_lock{VkbVulkan::Scheduler::submit_mutex};
    (void)device.waitIdle();
}
} // namespace

namespace Core::Vkb::Aa {
namespace {

enum class Kind : u32 { Off, Fsr3, Dlaa, Fsr4, Taa };

} // namespace

bool PicturePart(char part) {
    static const std::string parts = [] {
        const char* e = std::getenv("TLG_VKB_PICTURE_SETTINGS");
        const std::string v = e != nullptr ? e : "";
        return v.empty() || v == "1" ? std::string("fsba") : v == "0" ? std::string() : v;
    }();
    return parts.find(part) != std::string::npos;
}

bool Fsr4Switch() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_VKB_FSR4");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

bool TaaSwitch() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_VKB_TAA");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

namespace {

/// The AA in use (recording thread writes, any thread reads): decided in Init from TLG_VKB_AA; an AA that cannot run hands over to the next
/// (HandOver: DLAA to FSR 4 or FSR 3.1; FSR 4, TAA to FSR 3.1)
std::atomic<Kind> active{Kind::Off};
const char* asked = ""; ///< TLG_VKB_AA as given (report)

Kind Chosen() {
    return active.load(std::memory_order_relaxed);
}

/// The fresh-launch default's "RDNA 4 card" (owner, pull request 51: "FSR 4 on an RDNA 4 card (RX 9000)"): an AMD device in the PCI device range 0x75xx,
/// the range of the RDNA 4 chips (Navi 48: 0x7550, 0x7551; Navi 44: 0x7590); RDNA 3 is 0x74xx, RDNA 2 0x73xx. Asked once, from the device's own properties.
bool IsRdna4(const VkbVulkan::Instance& inst) {
    return inst.GetVendorID() == 0x1002 && (inst.GetDeviceID() & 0xff00u) == 0x7500u;
}

/// TLG_VKB_AA_CHOICE=0: the old way (a choice the engine has no AA for, taa or an unknown value, is AA off without a line).
bool ChoiceSwitch() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_VKB_AA_CHOICE");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

/// TLG_VKB_DLAA_FALLBACK=0: the old way for DLAA (issue 99): DLAA without DLSS is FSR 3.1, an installed DLSS that cannot start or fails is AA off for the session
bool DlaaFallbackSwitch() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_VKB_DLAA_FALLBACK");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

/// The hand-over (issue 99): one route for every anti-aliasing that cannot run, at the start or on any frame. Before it each case had its own (a choice the
/// engine has no AA for, issue 57; a chosen FSR 4, issue 55; a chosen TAA, issue 57) and DLAA had none (AA off for the rest of the session). The owner's
/// rules: "the game is never left without AA" (pull request 51); a DLAA that fails falls back "to FSR 4, or FSR 3.1 where FSR 4 is not available, like the
/// old engine does" (issue 99). So: DLAA -> FSR 4 on a card that runs it (the normal renderer's FSR 4 rule, IsFsr4Int8Supported), else FSR 3.1; FSR 4, TAA
/// and a choice without an AA -> FSR 3.1; FSR 3.1 itself has nothing after it (`failed`: AA off, counted). A fault (something installed or chosen that
/// does not work) is an error line and an on-screen message; the two quiet cases the owner named (DLSS not installed, FSR 4 model files not installed) are
/// an info line. The line is written once the AA that takes over is made (EnsureContext), so a chain (DLAA -> FSR 4 without model files -> FSR 3.1) is one
/// line that names every step and the AA that took over. Counted per kind (the report).
enum class From : u32 { Choice, Dlaa, Fsr4, Taa, Count };
constexpr std::array<const char*, u32(From::Count)> from_names{"the anti-aliasing choice", "DLAA", "FSR 4", "TAA"};
std::array<std::atomic<u64>, u32(From::Count)> c_handover_faults{}, c_handover_quiet{};
std::mutex handover_mutex;
std::array<std::string, u32(From::Count)> handover_fault, handover_quiet; ///< the last reason per kind, fault or quiet (report; under handover_mutex)
std::string handover_line;                 ///< the steps since the last AA was made (under handover_mutex)
bool handover_line_fault = false;          ///< one of them was a fault (under handover_mutex)
std::string took_over;                     ///< the AA that took over last (report; under handover_mutex)
std::atomic<bool> handover_due{false};     ///< a hand-over waits for its line (EnsureContext, or BeforeCombine when nothing could be made)

Kind HandOver(From from, const VkbVulkan::Instance& inst, const std::string& why, bool fault) {
    const u32 f = u32(from);
    (fault ? c_handover_faults : c_handover_quiet)[f].fetch_add(1, std::memory_order_relaxed);
    const Kind next = from == From::Dlaa && Fsr4Switch() && inst.IsFsr4Int8Supported() ? Kind::Fsr4 : Kind::Fsr3;
    {
        std::scoped_lock lock{handover_mutex};
        (fault ? handover_fault : handover_quiet)[f] = why;
        handover_line += handover_line.empty() ? "" : "; then ";
        handover_line += from == From::Choice ? why : fmt::format("{} could not run ({})", from_names[f], why);
        handover_line_fault = handover_line_fault || fault;
    }
    active.store(next, std::memory_order_relaxed);
    handover_due.store(true, std::memory_order_release);
    return next;
}

/// The one line of a hand-over, once the AA that took over is made (`made`: its name) or nothing could be made (null: AA off)
void HandOverLine(const char* made) {
    if (!handover_due.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    std::scoped_lock lock{handover_mutex};
    if (handover_line.empty()) {
        return; // a second hand-over's flag, its steps already in the line written before
    }
    took_over = made != nullptr ? made : "off";
    const std::string text = handover_line + ". " + (made != nullptr ? fmt::format("{} native AA took over", made) : std::string("Nothing could take over: AA off"));
    if (handover_line_fault || made == nullptr) {
        LOG_ERROR(Render_Vulkan, "vkb aa: {}", text);
        std::string notice = text;
        notice[0] = char(std::toupper(static_cast<unsigned char>(notice[0])));
        TlgOverlay::Notice(notice, 15.0f);
    } else {
        LOG_INFO(Render_Vulkan, "vkb aa: {}", text);
    }
    handover_line.clear();
    handover_line_fault = false;
}

/// DLAA as chosen, or (DLSS installed but its bridge or NGX did not come up with our device) the fallback at the start
Kind DlaaOrFallback(const VkbVulkan::Instance& inst) {
    std::string why;
    if (!DlaaFallbackSwitch() || Core::Vkb::DlssGlue::Usable(why)) {
        return Kind::Dlaa;
    }
    return HandOver(From::Dlaa, inst, "DLSS is installed but cannot start: " + why, true);
}

/// TLG_VKB_AA: fsr3 (or 1), fsr4, dlaa, off, taa, auto (the launcher on a fresh launch: the player chose nothing). The launcher passes every choice the game's
/// own settings can hold (tlg_run.backend_aa_env, the rule of TlgSettings::Load). The owner's rules (pull request 51): a choice is used as is (DLAA that
/// cannot run is FSR 4 where the card runs it, else FSR 3.1: HandOver, issue 99); the fresh-launch default is DLSS if present, else FSR 4 on an RDNA 4 card, else FSR 3.1; never without AA, so a
/// choice this engine has no AA for (an unknown value; taa with TLG_VKB_TAA=0) is a fault and FSR 3.1 runs. Not set at all (a tool that starts the engine without the
/// launcher): off, with an info line. TLG_VKB_FSR4=0: the old way for FSR 4 (fsr4 runs as FSR 3.1, auto as dlaa); TLG_VKB_AA_CHOICE=0: the old way for
/// taa and unknown values (off, no line).
Kind Resolve(const VkbVulkan::Instance& inst) {
    const char* e = std::getenv("TLG_VKB_AA");
    asked = e != nullptr ? e : "";
    const std::string v = asked;
    if (v == "fsr3" || v == "1") {
        return Kind::Fsr3;
    }
    if (v == "off" || v == "0" || v.empty()) {
        if (ChoiceSwitch()) {
            LOG_INFO(Render_Vulkan, "vkb aa: anti-aliasing off ({})", v.empty() ? "TLG_VKB_AA not set: the engine was started without the launcher"
                                                                                  : "the player's choice: Off, the game's own");
        }
        return Kind::Off;
    }
    if (v == "fsr4") {
        if (Fsr4Switch()) {
            return Kind::Fsr4;
        }
        LOG_INFO(Render_Vulkan, "vkb aa: FSR 4 asked for, TLG_VKB_FSR4=0: FSR 3.1 native AA (the old way)");
        return Kind::Fsr3;
    }
    if (v == "auto" && Fsr4Switch()) {
        if (Core::Vkb::DlssGlue::Wanted()) {
            LOG_INFO(Render_Vulkan, "vkb aa: no anti-aliasing chosen yet: DLAA (DLSS is installed)");
            return DlaaOrFallback(inst);
        }
        const bool rdna4 = IsRdna4(inst);
        LOG_INFO(Render_Vulkan, "vkb aa: no anti-aliasing chosen yet, DLSS not installed, {} (device {:04x}:{:04x}): {}", rdna4 ? "an RDNA 4 card" : "not an RDNA 4 card",
                 inst.GetVendorID(), inst.GetDeviceID(), rdna4 ? "FSR 4 native AA" : "FSR 3.1 native AA");
        return rdna4 ? Kind::Fsr4 : Kind::Fsr3;
    }
    if (v == "dlaa" || v == "auto") {
        if (Core::Vkb::DlssGlue::Wanted()) {
            return DlaaOrFallback(inst);
        }
        if (PicturePart('f')) {
            if (DlaaFallbackSwitch()) {
                return HandOver(From::Dlaa, inst, "tlg_dlss.dll / nvngx_dlss.dll are not next to the executable", false);
            }
            // the normal renderer's rule (TlgSettings::ConfigureDlssSupport): DLAA without DLSS is FSR 3.1 native AA, on every machine
            LOG_INFO(Render_Vulkan, "vkb aa: DLAA asked for, but tlg_dlss.dll / nvngx_dlss.dll are not next to the executable: FSR 3.1 native AA instead");
            return Kind::Fsr3;
        }
        LOG_WARNING(Render_Vulkan, "vkb aa: DLAA asked for, but tlg_dlss.dll / nvngx_dlss.dll are not next to the executable: AA off");
        return Kind::Off;
    }
    if (!ChoiceSwitch()) {
        return Kind::Off; // the old way
    }
    if (v == "taa") {
        if (TaaSwitch()) {
            return Kind::Taa; // issue 57: the normal renderer's TAA (their taa.comp), native size
        }
        return HandOver(From::Choice, inst, "TAA was chosen, but the new engine has no TAA yet (TLG_VKB_TAA=0)", true); // the old way
    }
    return HandOver(From::Choice, inst, fmt::format("anti-aliasing '{}' is not a choice the new engine knows", v), true);
}

/// Live switching (owner hand test 2026-11-22: "it needs to be swappable"): TLG_VKB_AA is the AA at the start; a change of the in-game Anti-aliasing row
/// (or of tlg.ini with a settings reload) switches the AA on the next frame, with the same rules as the start (Resolve). TLG_VKB_AA_LIVE=0: the old way
/// (the start choice for the whole session). The menu is read on the game's render thread (FollowMenu, at the combine); the switch itself (resources freed
/// and the new AA made) happens on the recording thread in BeforeCombine, where every AA resource is made and used.
int menu_code = -1;                       ///< the menu's AA when last read (game's render thread)
std::atomic<int> requested_code{-1};      ///< the menu's AA to switch to
std::atomic<bool> switch_due{false};      ///< a switch waits for the recording thread
std::atomic<u64> c_switches{0};

/// The menu's AA as one code: its upscaler with "native AA" on (TlgSettings::SetUpscaler keeps dlaa = upscaler != Off), else off
int MenuCode() {
    const auto& v = TlgSettings::Get();
    return v.dlaa.load() ? v.upscaler.load() : int(TlgSettings::UpscalerOff);
}

const char* KindName(Kind k) {
    return k == Kind::Fsr3 ? "FSR 3.1" : k == Kind::Dlaa ? "DLAA" : k == Kind::Fsr4 ? "FSR 4" : k == Kind::Taa ? "TAA" : "off";
}

/// A menu choice to the engine's AA, the rules of Resolve
Kind MenuKind(int code, const VkbVulkan::Instance& inst) {
    switch (code) {
    case TlgSettings::UpscalerDlss:
        if (Core::Vkb::DlssGlue::Wanted()) {
            return DlaaOrFallback(inst);
        }
        return HandOver(From::Dlaa, inst, "tlg_dlss.dll / nvngx_dlss.dll are not next to the executable", false);
    case TlgSettings::UpscalerFsr4:
    case TlgSettings::UpscalerFsr411: // the new engine has one FSR 4
        return Fsr4Switch() ? Kind::Fsr4 : Kind::Fsr3;
    case TlgSettings::UpscalerTaa:
        return TaaSwitch() ? Kind::Taa : HandOver(From::Choice, inst, "TAA was chosen, but the new engine has no TAA yet (TLG_VKB_TAA=0)", true);
    case TlgSettings::UpscalerFsr3:
        return Kind::Fsr3;
    default:
        return Kind::Off;
    }
}

const VkbVulkan::Instance* instance = nullptr;

struct Input {
    vk::Image image{};
    vk::Format format{};
    u32 width{}, height{};
    bool depth{};
};
/// The newest motion pass's attachments (recording threads, under `mutex`)
std::mutex mutex;
Input scene, motion, depth;
u64 inputs_serial = 0, used_serial = 0;

FfxVkPortableUpscaleContext* context = nullptr;
std::atomic<bool> sharpness_env_noted{}; ///< issue 42: TLG_VKB_AA_SHARPNESS was set while the Sharpening setting applies (noted once)
u32 ctx_w = 0, ctx_h = 0;
VkImage output = VK_NULL_HANDLE;
VmaAllocation output_alloc = nullptr;
u64 frame_id = 0;
std::chrono::steady_clock::time_point last_run{};
bool failed = false;
bool fsr4_made = false;                     ///< the FSR 4 provider exists (Fsr4::Create returned Ok)
std::atomic<u64> c_fsr4_runs{0};

std::atomic<u64> c_motion_passes{0}, c_runs{0}, c_no_inputs{0}, c_unusable{0}, c_failed{0}, c_resets{0}, c_cut_resets{0};
std::atomic<u64> c_rcas_passes{0}, c_scene_bias_draws{0};

// Issue 42 (part s): RCAS passes after the AA, as the normal renderer's ExtraSharpen / DlssSharpen; the result ping-pongs between `output` and this image
VkImage sharpen_img = VK_NULL_HANDLE;
VmaAllocation sharpen_alloc = nullptr;
vk::ImageView sharpen_view{};
vk::DescriptorSetLayout rcas_set_layout{};
vk::PipelineLayout rcas_layout{};
vk::Pipeline rcas_pipeline{};
bool rcas_broken = false; ///< the pipeline could not be made (logged once, in the report): the AA's own sharpening only
std::atomic<bool> frame_cut{false}; ///< K4: the engine's camera-cut flag of the frame being recorded (render thread)

// Issue 57: TAA, the normal renderer's (their TemporalUpscaler::RecordTaa with taa.comp): two histories (scene colour and depth, RGBA32F) alternate
// between read and write; its result goes to `output`, its sharpening is the RCAS passes after it (as DLAA: an AA without RCAS of its own)
std::array<VkImage, 2> taa_hist{};
std::array<VmaAllocation, 2> taa_hist_allocs{};
std::array<vk::ImageView, 2> taa_hist_views{};
u32 taa_next = 0; ///< the history written this frame (the other one is read)
vk::DescriptorSetLayout taa_set_layout{};
vk::PipelineLayout taa_layout{};
vk::Pipeline taa_pipeline{};
vk::Sampler taa_sampler{};
std::atomic<u64> c_taa_runs{0};

bool Facts(vk::Image image, Input& out) {
    out.image = image;
    return image && Core::Vkb::ImageFacts(image, out.format, out.width, out.height, out.depth);
}

vk::ImageUsageFlags UsageOf(const Input& in) {
    // the rule our images are made with (native_vkb_objects.cpp): transfer and sampled, plus the attachment use these targets have
    return vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled |
           (in.depth ? vk::ImageUsageFlagBits::eDepthStencilAttachment : vk::ImageUsageFlagBits::eColorAttachment);
}

FfxVkPortableImage Describe(VkImage image, VkFormat format, u32 w, u32 h, VkImageUsageFlags usage, VkImageAspectFlags aspect, FfxVkPortableResourceState state) {
    FfxVkPortableImage out{};
    out.structSize = sizeof(out);
    out.image = image;
    out.format = format;
    out.extent = {w, h};
    out.mipCount = 1;
    out.arrayLayers = 1;
    out.usage = usage;
    out.aspect = aspect;
    out.state = state;
    return out;
}

// F1d (DLAA): the motion in pixels (vkb_aa_motion.comp), views of the inputs (by image), the conversion pipeline
/// F2: the motion and depth copies rotate over this many frames, each published with the timeline value of its command buffer; the presenter takes the newest whose
/// value the GPU has passed (the frame being flipped; the recorder may already record the next ones into other slots). Upstream records frame generation on the frame's
/// own scheduler; ours is the presenter's, so this stands in for that order.
constexpr u32 FgSlots = 4;
std::array<VkImage, FgSlots> motion_pxs{};
std::array<VmaAllocation, FgSlots> motion_px_allocs{};
std::array<vk::ImageView, FgSlots> motion_px_views{};
u32 px_index = 0; ///< the slot of this run
VkImage motion_px = VK_NULL_HANDLE;
vk::ImageView motion_px_view{}, output_view{};
// F2: the newest frame's inputs for frame generation
std::mutex fg_mutex;
struct FgSlot {
    FgInputs inputs{};
    u64 value = 0;  ///< the stream timeline value the slot's commands go out with
    u64 serial = 0; ///< 0: empty
};
std::array<FgSlot, FgSlots> fg_slots{};
u64 fg_serial = 0, fg_taken = 0;
// F2: our copy of the scene depth (double-buffered with the motion): the presenter reads only images the AA owns, the backend frees the game's (loading)
std::array<VkImage, FgSlots> depth_copies{};
std::array<VmaAllocation, FgSlots> depth_copy_allocs{};
std::array<vk::ImageView, FgSlots> depth_copy_views{};
vk::Format depth_copy_format{};
u32 depth_copy_w = 0, depth_copy_h = 0;
vk::DescriptorSetLayout motion_set_layout{};
vk::PipelineLayout motion_layout{};
vk::Pipeline motion_pipeline{};
u32 dl_w = 0, dl_h = 0;

/// A view of one of the game's images for this run only: retired through the backend once the submit is done (a cache by handle went stale when the backend
/// freed an image during loading and a new one got the same handle: device lost, owner play 2026-11-16)
vk::ImageView ViewOf(VkImage image, vk::Format format, vk::ImageAspectFlags aspect) {
    const auto [r, view] = instance->GetDevice().createImageView(
        {.image = vk::Image{image}, .viewType = vk::ImageViewType::e2D, .format = format, .subresourceRange = {aspect, 0, 1, 0, 1}});
    if (r != vk::Result::eSuccess) {
        return {};
    }
    Core::Vkb::RetireView(view);
    return view;
}

// FSR 4: our copies of the scene colour and depth. The provider registers at most 8 images, by view, and never forgets one (MAX_EXTERNAL_IMAGES in
// ffx_vk_fsr4_v07.c); the game's images get a new view every run here (ViewOf), so it is handed views that stay: these two, the motion slots, the output
VkImage f4_color = VK_NULL_HANDLE, f4_depth = VK_NULL_HANDLE;
VmaAllocation f4_color_alloc = nullptr, f4_depth_alloc = nullptr;
vk::ImageView f4_color_view{}, f4_depth_view{};
vk::Format f4_color_format{}, f4_depth_format{};
u32 f4_w = 0, f4_h = 0;

/// The device is idle (DestroyContext, or a format change below)
void DestroyFsr4Inputs() {
    const vk::Device device = instance->GetDevice();
    if (f4_color != VK_NULL_HANDLE) {
        device.destroyImageView(f4_color_view);
        vmaDestroyImage(instance->GetAllocator(), f4_color, f4_color_alloc);
        f4_color = VK_NULL_HANDLE, f4_color_alloc = nullptr, f4_color_view = vk::ImageView{};
    }
    if (f4_depth != VK_NULL_HANDLE) {
        device.destroyImageView(f4_depth_view);
        vmaDestroyImage(instance->GetAllocator(), f4_depth, f4_depth_alloc);
        f4_depth = VK_NULL_HANDLE, f4_depth_alloc = nullptr, f4_depth_view = vk::ImageView{};
    }
}

bool MakeCopyImage(u32 w, u32 h, vk::Format format, vk::ImageAspectFlags aspect, VkImage& image, VmaAllocation& alloc, vk::ImageView& view) {
    const VkImageCreateInfo ci{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                               .imageType = VK_IMAGE_TYPE_2D,
                               .format = static_cast<VkFormat>(format),
                               .extent = {w, h, 1},
                               .mipLevels = 1,
                               .arrayLayers = 1,
                               .samples = VK_SAMPLE_COUNT_1_BIT,
                               .tiling = VK_IMAGE_TILING_OPTIMAL,
                               .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                               .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    const VmaAllocationCreateInfo alloc_info{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(instance->GetAllocator(), &ci, &alloc_info, &image, &alloc, nullptr) != VK_SUCCESS) {
        image = VK_NULL_HANDLE, alloc = nullptr;
        return false;
    }
    const auto [r, v] = instance->GetDevice().createImageView(
        {.image = vk::Image{image}, .viewType = vk::ImageViewType::e2D, .format = format, .subresourceRange = {aspect, 0, 1, 0, 1}});
    if (r != vk::Result::eSuccess) {
        vmaDestroyImage(instance->GetAllocator(), image, alloc);
        image = VK_NULL_HANDLE, alloc = nullptr;
        return false;
    }
    view = v;
    return true;
}

bool EnsureFsr4Inputs(u32 w, u32 h, vk::Format color_format, vk::Format depth_format) {
    if (f4_color != VK_NULL_HANDLE && f4_depth != VK_NULL_HANDLE && f4_w == w && f4_h == h && f4_color_format == color_format && f4_depth_format == depth_format) {
        return true;
    }
    if (f4_color != VK_NULL_HANDLE || f4_depth != VK_NULL_HANDLE) {
        IdleDevice(instance->GetDevice()); // earlier frames may still read them
        DestroyFsr4Inputs();
    }
    if (!MakeCopyImage(w, h, color_format, vk::ImageAspectFlagBits::eColor, f4_color, f4_color_alloc, f4_color_view) ||
        !MakeCopyImage(w, h, depth_format, vk::ImageAspectFlagBits::eDepth, f4_depth, f4_depth_alloc, f4_depth_view)) {
        DestroyFsr4Inputs();
        return false;
    }
    f4_w = w, f4_h = h, f4_color_format = color_format, f4_depth_format = depth_format;
    return true;
}

void DestroyDepthCopies() {
    {
        std::scoped_lock fg_lock{fg_mutex};
        fg_slots = {};
    }
    for (u32 i = 0; i < FgSlots; ++i) {
        if (depth_copies[i] != VK_NULL_HANDLE) {
            instance->GetDevice().destroyImageView(depth_copy_views[i]);
            vmaDestroyImage(instance->GetAllocator(), depth_copies[i], depth_copy_allocs[i]);
            depth_copies[i] = VK_NULL_HANDLE;
        }
    }
}

/// F2: our depth images at the render size and the game's depth format (destroyed only after a device-wide wait)
bool EnsureDepthCopies(u32 w, u32 h, vk::Format format) {
    if (depth_copies[0] != VK_NULL_HANDLE && depth_copy_format == format && depth_copy_w == w && depth_copy_h == h) {
        return true;
    }
    DestroyDepthCopies(); // EnsureDlaa has idled the device on a size change; a format change idles here
    const VkImageCreateInfo ci{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                               .imageType = VK_IMAGE_TYPE_2D,
                               .format = static_cast<VkFormat>(format),
                               .extent = {w, h, 1},
                               .mipLevels = 1,
                               .arrayLayers = 1,
                               .samples = VK_SAMPLE_COUNT_1_BIT,
                               .tiling = VK_IMAGE_TILING_OPTIMAL,
                               .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, // source: FSR's depth copy
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                               .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    const VmaAllocationCreateInfo alloc{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    for (u32 i = 0; i < FgSlots; ++i) {
        if (vmaCreateImage(instance->GetAllocator(), &ci, &alloc, &depth_copies[i], &depth_copy_allocs[i], nullptr) != VK_SUCCESS) {
            return false;
        }
        depth_copy_views[i] = instance->GetDevice().createImageView({.image = vk::Image{depth_copies[i]}, .viewType = vk::ImageViewType::e2D, .format = format,
                                                                     .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}}).value;
    }
    depth_copy_format = format, depth_copy_w = w, depth_copy_h = h;
    return true;
}

bool EnsureDlaa(u32 w, u32 h) {
    const vk::Device device = instance->GetDevice();
    if (!motion_pipeline) {
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {.binding = 0, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        motion_set_layout = device.createDescriptorSetLayout({.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                                                              .bindingCount = static_cast<u32>(bindings.size()), .pBindings = bindings.data()}).value;
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = 8};
        motion_layout = device.createPipelineLayout({.setLayoutCount = 1, .pSetLayouts = &motion_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &range}).value;
        const std::span<const u32> code{VKB_AA_MOTION_COMP};
        const vk::ShaderModule module = device.createShaderModule({.codeSize = code.size_bytes(), .pCode = code.data()}).value;
        const auto [r, pipeline] = device.createComputePipeline({}, {.stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"}, .layout = motion_layout});
        device.destroyShaderModule(module);
        if (r != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "vkb aa: motion pipeline failed");
            return false;
        }
        motion_pipeline = pipeline;
    }
    if (motion_pxs[0] != VK_NULL_HANDLE && dl_w == w && dl_h == h) {
        px_index = (px_index + 1) % FgSlots;
        motion_px = motion_pxs[px_index], motion_px_view = motion_px_views[px_index];
        return true;
    }
    if (motion_pxs[0] != VK_NULL_HANDLE) {
        IdleDevice(device);
        for (u32 i = 0; i < FgSlots; ++i) {
            device.destroyImageView(motion_px_views[i]);
            vmaDestroyImage(instance->GetAllocator(), motion_pxs[i], motion_px_allocs[i]);
            motion_pxs[i] = VK_NULL_HANDLE;
        }
    }
    const VkImageCreateInfo ci{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                               .imageType = VK_IMAGE_TYPE_2D,
                               .format = VK_FORMAT_R16G16_SFLOAT,
                               .extent = {w, h, 1},
                               .mipLevels = 1,
                               .arrayLayers = 1,
                               .samples = VK_SAMPLE_COUNT_1_BIT,
                               .tiling = VK_IMAGE_TILING_OPTIMAL,
                               .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                               .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    const VmaAllocationCreateInfo alloc{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    for (u32 i = 0; i < FgSlots; ++i) {
        if (vmaCreateImage(instance->GetAllocator(), &ci, &alloc, &motion_pxs[i], &motion_px_allocs[i], nullptr) != VK_SUCCESS) {
            return false;
        }
        motion_px_views[i] = device.createImageView({.image = vk::Image{motion_pxs[i]}, .viewType = vk::ImageViewType::e2D, .format = vk::Format::eR16G16Sfloat,
                                                     .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}).value;
    }
    px_index = 0;
    motion_px = motion_pxs[0], motion_px_view = motion_px_views[0];
    {
        std::scoped_lock fg_lock{fg_mutex};
        fg_slots = {};
    }
    dl_w = w, dl_h = h;
    return true;
}

void DestroyContext() {
    // every image below may still be in use by frames already sent (the output and TAA's histories too: second read of issue 57)
    if (context != nullptr || fsr4_made || output != VK_NULL_HANDLE || taa_hist[0] != VK_NULL_HANDLE) {
        IdleDevice(instance->GetDevice());
    }
    if (context != nullptr) {
        ffxVkPortableUpscaleContextDestroy(context);
        context = nullptr;
    }
    if (fsr4_made) {
        Core::Vkb::Fsr4::Destroy();
        fsr4_made = false;
    }
    DestroyFsr4Inputs();
    for (u32 i = 0; i < taa_hist.size(); ++i) {
        if (taa_hist[i] != VK_NULL_HANDLE) {
            instance->GetDevice().destroyImageView(taa_hist_views[i]);
            vmaDestroyImage(instance->GetAllocator(), taa_hist[i], taa_hist_allocs[i]);
            taa_hist[i] = VK_NULL_HANDLE, taa_hist_allocs[i] = nullptr, taa_hist_views[i] = vk::ImageView{};
        }
    }
    if (output != VK_NULL_HANDLE) {
        if (output_view) {
            instance->GetDevice().destroyImageView(output_view);
            output_view = vk::ImageView{};
        }
        vmaDestroyImage(instance->GetAllocator(), output, output_alloc);
        output = VK_NULL_HANDLE, output_alloc = nullptr;
    }
    if (sharpen_img != VK_NULL_HANDLE) {
        instance->GetDevice().destroyImageView(sharpen_view);
        sharpen_view = vk::ImageView{};
        vmaDestroyImage(instance->GetAllocator(), sharpen_img, sharpen_alloc);
        sharpen_img = VK_NULL_HANDLE, sharpen_alloc = nullptr;
    }
}

/// A chosen FSR 4 or TAA that cannot run: a fault, said in the log and on screen (owner, pull request 51: no quiet switch); FSR 3.1 native AA takes over
/// (HandOver) so the game keeps its anti-aliasing
void Fsr4Fault(const std::string& why) {
    HandOver(From::Fsr4, *instance, why, true);
}
void TaaFault(const std::string& why) {
    HandOver(From::Taa, *instance, why, true);
}

/// TAA's pipeline (once; their layout: 4 sampled inputs with a nearest sampler, 3 storage outputs, 64 bytes of push constants) and its two histories at
/// this size (made after DestroyContext, so the first frame is a reset). False: `why` says what failed.
bool EnsureTaa(u32 w, u32 h, std::string& why) {
    const vk::Device device = instance->GetDevice();
    if (!taa_pipeline) {
        std::array<vk::DescriptorSetLayoutBinding, 7> bindings{};
        for (u32 i = 0; i < bindings.size(); ++i) {
            bindings[i] = {.binding = i, .descriptorType = i < 4 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage,
                           .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute};
        }
        // each part made once; one that failed is tried again on the next call (null)
        if (!taa_set_layout) {
            taa_set_layout = device.createDescriptorSetLayout({.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                                                               .bindingCount = static_cast<u32>(bindings.size()), .pBindings = bindings.data()}).value;
        }
        if (taa_set_layout && !taa_layout) {
            const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = 64};
            taa_layout = device.createPipelineLayout({.setLayoutCount = 1, .pSetLayouts = &taa_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &range}).value;
        }
        if (!taa_sampler) {
            taa_sampler = device.createSampler({.magFilter = vk::Filter::eNearest, .minFilter = vk::Filter::eNearest, .mipmapMode = vk::SamplerMipmapMode::eNearest,
                                                .addressModeU = vk::SamplerAddressMode::eClampToEdge, .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                                                .addressModeW = vk::SamplerAddressMode::eClampToEdge}).value;
        }
        if (!taa_set_layout || !taa_layout || !taa_sampler) {
            why = "its pipeline layout could not be made";
            return false;
        }
        const std::span<const u32> code{TAA_COMP};
        const auto [mr, module] = device.createShaderModule({.codeSize = code.size_bytes(), .pCode = code.data()});
        if (mr != vk::Result::eSuccess) {
            why = "its shader could not be loaded";
            return false;
        }
        const auto [r, pipeline] = device.createComputePipeline({}, {.stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"}, .layout = taa_layout});
        device.destroyShaderModule(module);
        if (r != vk::Result::eSuccess) {
            why = "its pipeline could not be made";
            return false;
        }
        taa_pipeline = pipeline;
    }
    const VkImageCreateInfo ci{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                               .imageType = VK_IMAGE_TYPE_2D,
                               .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                               .extent = {w, h, 1},
                               .mipLevels = 1,
                               .arrayLayers = 1,
                               .samples = VK_SAMPLE_COUNT_1_BIT,
                               .tiling = VK_IMAGE_TILING_OPTIMAL,
                               .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                               .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    const VmaAllocationCreateInfo alloc{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    for (u32 i = 0; i < taa_hist.size(); ++i) {
        if (vmaCreateImage(instance->GetAllocator(), &ci, &alloc, &taa_hist[i], &taa_hist_allocs[i], nullptr) != VK_SUCCESS) {
            taa_hist[i] = VK_NULL_HANDLE, taa_hist_allocs[i] = nullptr;
            why = "its history images could not be made";
            return false;
        }
        const auto [vr, view] = device.createImageView({.image = vk::Image{taa_hist[i]}, .viewType = vk::ImageViewType::e2D, .format = vk::Format::eR32G32B32A32Sfloat,
                                                        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
        if (vr != vk::Result::eSuccess) {
            why = "its history views could not be made";
            return false; // the caller's DestroyContext frees the images made so far
        }
        taa_hist_views[i] = view;
    }
    taa_next = 0;
    return true;
}

const char* AaName() {
    return Chosen() == Kind::Fsr4 ? "FSR 4" : Chosen() == Kind::Taa ? "TAA" : Chosen() == Kind::Dlaa ? "DLAA" : "FSR 3.1";
}

bool EnsureContext(u32 w, u32 h) {
    if ((context != nullptr || (Chosen() == Kind::Dlaa && output != VK_NULL_HANDLE) || (Chosen() == Kind::Fsr4 && fsr4_made) ||
         (Chosen() == Kind::Taa && taa_hist[0] != VK_NULL_HANDLE)) && ctx_w == w && ctx_h == h) {
        HandOverLine(AaName()); // a hand-over to an AA that already exists (none today) still gets its line
        return true;
    }
    DestroyContext();
    if (Chosen() == Kind::Fsr4) {
        const auto start = Core::Vkb::Fsr4::Create(*instance, w, h);
        if (start == Core::Vkb::Fsr4::Start::Ok) {
            fsr4_made = true;
        } else if (start == Core::Vkb::Fsr4::Start::NoModel) {
            // the owner's one exception: the FSR 4 files are not installed, FSR 3.1 instead with an info line, nothing on screen
            HandOver(From::Fsr4, *instance, "its model files are not installed: " + Core::Vkb::Fsr4::Problem(), false);
        } else {
            Fsr4Fault(Core::Vkb::Fsr4::Problem());
        }
    }
    if (Chosen() == Kind::Taa) {
        std::string why;
        if (!EnsureTaa(w, h, why)) {
            DestroyContext(); // a history made before the failure
            TaaFault(why);
        }
    }
    if (Chosen() == Kind::Dlaa || Chosen() == Kind::Fsr4 || Chosen() == Kind::Taa) {
        goto make_output; // DLAA (DlssGlue), FSR 4 (native_vkb_fsr4) and TAA (above) have their own; only the output image is shared
    }
    {
    FfxVkPortableDeviceInfo device_info{};
    device_info.structSize = sizeof(device_info);
    device_info.instance = instance->GetInstance();
    device_info.physicalDevice = instance->GetPhysicalDevice();
    device_info.device = instance->GetDevice();
    device_info.getDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
    device_info.queue = instance->GetGraphicsQueue();
    device_info.queueFamilyIndex = instance->GetGraphicsQueueFamilyIndex();
    device_info.shaderFloat16Enabled = instance->IsShaderFloat16Supported() ? VK_TRUE : VK_FALSE;
    device_info.subgroupSizeControlEnabled = VK_TRUE;
    device_info.synchronization2Enabled = VK_TRUE;
    device_info.shaderStorageImageWriteWithoutFormatEnabled = VK_TRUE;
    FfxVkPortableUpscaleCreateInfo create_info{};
    create_info.structSize = sizeof(create_info);
    create_info.flags = FFX_VK_PORTABLE_CONTEXT_HDR_COLOR_INPUT | FFX_VK_PORTABLE_CONTEXT_AUTO_EXPOSURE; // native AA runs on the HDR scene colour
    create_info.maxRenderSize = {w, h};
    create_info.maxOutputSize = {w, h};
    if (const u64 issues = ffxVkPortableValidateUpscaleCreateInfo(&create_info); issues != 0) {
        LOG_ERROR(Render_Vulkan, "vkb aa: FSR 3.1 create info rejected ({:#x})", issues);
        return false;
    }
    if (ffxVkPortableUpscaleContextCreate(&device_info, &create_info, &context) != FFX_VK_PORTABLE_OK) {
        LOG_ERROR(Render_Vulkan, "vkb aa: FSR 3.1 context creation failed");
        context = nullptr;
        return false;
    }
    }
make_output:
    const VkImageCreateInfo ci{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                               .imageType = VK_IMAGE_TYPE_2D,
                               .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                               .extent = {w, h, 1},
                               .mipLevels = 1,
                               .arrayLayers = 1,
                               .samples = VK_SAMPLE_COUNT_1_BIT,
                               .tiling = VK_IMAGE_TILING_OPTIMAL,
                               .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                               .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    const VmaAllocationCreateInfo alloc{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(instance->GetAllocator(), &ci, &alloc, &output, &output_alloc, nullptr) != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "vkb aa: output image failed");
        if (context != nullptr) {
            ffxVkPortableUpscaleContextDestroy(context);
            context = nullptr;
        }
        return false;
    }
    output_view = instance->GetDevice().createImageView({.image = vk::Image{output}, .viewType = vk::ImageViewType::e2D, .format = vk::Format::eR16G16B16A16Sfloat,
                                                         .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}).value;
    if (PicturePart('s')) { // issue 42: the second image of the RCAS passes (same format and use as the output)
        if (vmaCreateImage(instance->GetAllocator(), &ci, &alloc, &sharpen_img, &sharpen_alloc, nullptr) != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "vkb aa: sharpening image failed");
            sharpen_img = VK_NULL_HANDLE, sharpen_alloc = nullptr;
            DestroyContext();
            return false;
        }
        sharpen_view = instance->GetDevice().createImageView({.image = vk::Image{sharpen_img}, .viewType = vk::ImageViewType::e2D,
                                                              .format = vk::Format::eR16G16B16A16Sfloat,
                                                              .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}).value;
    }
    ctx_w = w, ctx_h = h;
    frame_id = 0;
    if (Chosen() == Kind::Fsr3) {
        LOG_INFO(Render_Vulkan, "vkb aa: FSR 3.1 native AA {}x{} on the game's motion vectors", w, h);
    } else if (Chosen() == Kind::Fsr4) {
        LOG_INFO(Render_Vulkan, "vkb aa: FSR 4 native AA {}x{} on the game's motion vectors (in render pixels)", w, h);
    } else if (Chosen() == Kind::Taa) {
        LOG_INFO(Render_Vulkan, "vkb aa: TAA {}x{} on the game's motion vectors (in render pixels)", w, h);
    }
    HandOverLine(AaName());
    return true;
}

void Barrier(vk::CommandBuffer cmd) {
    const vk::MemoryBarrier2 b{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                               .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands, .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
}

bool EnsureRcas() {
    if (rcas_pipeline || rcas_broken) {
        return !rcas_broken;
    }
    rcas_broken = true; // until made
    const vk::Device device = instance->GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
        {.binding = 0, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
    }};
    rcas_set_layout = device.createDescriptorSetLayout({.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                                                        .bindingCount = static_cast<u32>(bindings.size()), .pBindings = bindings.data()}).value;
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = 8};
    rcas_layout = device.createPipelineLayout({.setLayoutCount = 1, .pSetLayouts = &rcas_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &range}).value;
    const std::span<const u32> code{VKB_RCAS_COMP};
    const vk::ShaderModule module = device.createShaderModule({.codeSize = code.size_bytes(), .pCode = code.data()}).value;
    const auto [r, pipeline] = device.createComputePipeline({}, {.stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"}, .layout = rcas_layout});
    device.destroyShaderModule(module);
    if (r != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "vkb aa: sharpening pipeline failed");
        return false;
    }
    rcas_pipeline = pipeline;
    rcas_broken = false;
    return true;
}

/// One RCAS pass (vkb_rcas.comp) from `src` into `dst`, both RGBA16F in GENERAL
void RcasPass(vk::CommandBuffer cmd, vk::ImageView src, vk::ImageView dst, u32 w, u32 h, float strength) {
    const vk::DescriptorImageInfo src_info{.imageView = src, .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo dst_info{.imageView = dst, .imageLayout = vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 2> writes{{
        {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &src_info},
        {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &dst_info},
    }};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, rcas_pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, rcas_layout, 0, writes);
    const std::array<float, 2> params{strength, 1.0f};
    cmd.pushConstants(rcas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params), params.data());
    cmd.dispatch((w + 7) / 8, (h + 7) / 8, 1);
    Barrier(cmd);
    c_rcas_passes.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

void Init(const VkbVulkan::Instance& instance_) {
    instance = &instance_;
    active.store(Resolve(instance_), std::memory_order_relaxed);
    if (Core::Vkb::DlssGlue::LiveAaSwitch()) {
        // the in-game menu greys out what the card cannot run; the normal renderer sets these (their vk_temporal_upscaler.cpp), the new engine did not, so
        // DLAA and FSR 4 were greyed out in its menu. One FSR 4 here: both FSR 4 rows run it.
        const bool fsr4 = Fsr4Switch() && instance_.IsFsr4Int8Supported();
        TlgSettings::ConfigureUpscalerSupport(fsr4, fsr4);
        std::string why;
        const bool dlss = Core::Vkb::DlssGlue::Usable(why);
        TlgSettings::ConfigureDlssSupport(dlss, dlss ? nullptr : why.c_str());
        menu_code = MenuCode(); // the start value is not a switch
        if (asked[0] == '\0') {
            // TLG_VKB_AA not set (started without the launcher): the menu shows the tlg.ini choice, so run that choice, else the menu says
            // DLAA while the engine runs off until the row is changed by hand
            active.store(MenuKind(menu_code, instance_), std::memory_order_relaxed);
            LOG_INFO(Render_Vulkan, "vkb aa: TLG_VKB_AA not set: the in-game menu's choice ({}) runs", KindName(active.load()));
        }
    }
}

void FollowMenu() {
    if (!Core::Vkb::DlssGlue::LiveAaSwitch() || instance == nullptr) {
        return;
    }
    const int code = MenuCode();
    if (code == menu_code) {
        return;
    }
    menu_code = code;
    requested_code.store(code, std::memory_order_relaxed);
    switch_due.store(true, std::memory_order_release);
}

bool SwitchDue() {
    return switch_due.load(std::memory_order_acquire);
}

bool On() {
    return Chosen() != Kind::Off;
}

bool Running();

bool JitterWanted() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_AA_JITTER"); return e == nullptr || e[0] != '0'; }();
    return on;
}

bool MeshKeySwitch() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_AA_MESH_KEY"); return e == nullptr || e[0] != '0'; }();
    return on;
}

// F1c state: the game's render thread only
u64 scene_depth_view = 0;
std::array<float, 2> frame_jitter{};
u32 jitter_index = 0;
std::unordered_set<u64> jittered_meshes;
std::atomic<u64> c_jittered{0}, c_jittered_meshes{0};

float Halton(u32 index, u32 base) {
    float f = 1.0f, r = 0.0f;
    for (u32 i = index; i > 0; i /= base) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(i % base);
    }
    return r;
}

u64 scene_depth_other = 0; ///< the scene depth is double-buffered (frames alternate two textures, census 2026-11-16): both recent ones count
void NoteSceneDepth(u64 depth_view_id) {
    if (depth_view_id != scene_depth_view) {
        scene_depth_other = scene_depth_view;
        scene_depth_view = depth_view_id;
    }
}

std::unordered_map<u64, std::array<u64, 4>> depth_census; ///< TLG_VKB_AA_CENSUS=1: depth id -> draws [write+geometry, write, test only, total] (render thread)
bool CensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_AA_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}

bool IsSceneDepth(u64 depth_view_id) {
    return depth_view_id != 0 && (depth_view_id == scene_depth_view || depth_view_id == scene_depth_other);
}

bool DrawJitter(u64 depth_view_id, bool depth_write, bool geometry, bool samples_depth, u64 mesh_key, std::array<float, 2>& jitter) {
    if (CensusOn()) {
        auto& c = depth_census[depth_view_id];
        c[0] += depth_write && geometry, c[1] += depth_write, c[2] += !depth_write, c[3] += 1;
    }
    if (!JitterWanted() || !Running() || depth_view_id == 0 || (depth_view_id != scene_depth_view && depth_view_id != scene_depth_other)) {
        return false;
    }
    bool on = false;
    if (geometry && !samples_depth) {
        on = true;
        if (depth_write) {
            jittered_meshes.insert(mesh_key);
        }
    } else if (mesh_key != 0 && jittered_meshes.contains(mesh_key)) {
        on = true; // a redraw of a mesh drawn jittered into this depth takes the same shift (water over its pre-pass)
        c_jittered_meshes.fetch_add(1, std::memory_order_relaxed);
    }
    if (on) {
        jitter = frame_jitter;
        c_jittered.fetch_add(1, std::memory_order_relaxed);
    }
    return on;
}

void NoteFrameCut(bool cut) {
    frame_cut.store(cut, std::memory_order_relaxed);
}

bool TakeFrameCut() {
    return frame_cut.exchange(false, std::memory_order_relaxed);
}

std::array<float, 2> OnCombineQueued() {
    const std::array<float, 2> used = frame_jitter;
    if (CensusOn()) {
        static u32 frames = 0;
        if (++frames % 300 == 0) {
            std::string line;
            for (const auto& [id, c] : depth_census) {
                line += fmt::format(" [{:#x}: wg {} w {} t {} all {}]", id, c[0], c[1], c[2], c[3]);
            }
            LOG_WARNING(Render_Vulkan, "vkb aa census (300 frames): scene depth {:#x}, running {}:{}", scene_depth_view, Running(), line);
        }
        if (frames % 300 == 0) {
            depth_census.clear();
        }
    }
    jittered_meshes.clear();
    if (JitterWanted() && Running()) {
        jitter_index = jitter_index % 8 + 1; // native size: 8 phases (their JitterPhases at a ratio of 1)
        frame_jitter = {Halton(jitter_index, 2) - 0.5f, Halton(jitter_index, 3) - 0.5f};
    } else {
        frame_jitter = {};
    }
    return used;
}

bool Running() {
    return On() && !failed && c_runs.load(std::memory_order_relaxed) != 0;
}

void NoteMotionPass(std::span<const vk::Image> attachments, u32 color_count, bool has_depth) {
    c_motion_passes.fetch_add(1, std::memory_order_relaxed);
    if (color_count < 3 || !has_depth || attachments.size() < color_count + 1) {
        return;
    }
    Input s, m, d;
    if (!Facts(attachments[0], s) || !Facts(attachments[2], m) || !Facts(attachments[color_count], d)) {
        return;
    }
    std::scoped_lock lock{mutex};
    scene = s, motion = m, depth = d;
    ++inputs_serial;
}

void BeforeCombine(vk::CommandBuffer cmd, std::array<float, 2> jitter, bool camera_cut) {
    if (instance != nullptr && switch_due.exchange(false, std::memory_order_acq_rel)) [[unlikely]] {
        const Kind before = Chosen();
        DestroyContext(); // idles the device first: frames already sent may still use the old AA's images
        if (before == Kind::Dlaa) {
            Core::Vkb::DlssGlue::Release(instance->GetDevice());
        }
        failed = false; // a new choice gets a new try
        frame_id = 0;
        ctx_w = ctx_h = 0;
        const Kind next = MenuKind(requested_code.load(std::memory_order_relaxed), *instance);
        active.store(next, std::memory_order_relaxed);
        c_switches.fetch_add(1, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "vkb aa: the menu's anti-aliasing changed ({}): {} -> {}", TlgSettings::UpscalerName(requested_code.load(std::memory_order_relaxed)),
                 KindName(before), KindName(next));
    }
    if (!On() || instance == nullptr || failed) {
        return;
    }
    Input s, m, d;
    {
        std::scoped_lock lock{mutex};
        if (inputs_serial == used_serial) {
            c_no_inputs.fetch_add(1, std::memory_order_relaxed);
            return; // no motion pass since the last run: a menu, a loading screen
        }
        used_serial = inputs_serial;
        s = scene, m = motion, d = depth;
    }
    const u32 w = s.width, h = s.height;
    if (s.format != vk::Format::eB10G11R11UfloatPack32 || m.format != vk::Format::eR16G16Sfloat || m.width != w || m.height != h || d.width != w || d.height != h ||
        !d.depth) {
        if (c_unusable.fetch_add(1, std::memory_order_relaxed) == 0) {
            LOG_WARNING(Render_Vulkan, "vkb aa: inputs unusable (scene {} {}x{}, motion {} {}x{}, depth {} {}x{})", vk::to_string(s.format), s.width, s.height,
                        vk::to_string(m.format), m.width, m.height, vk::to_string(d.format), d.width, d.height);
        }
        return;
    }
    if (depth_copies[0] != VK_NULL_HANDLE && depth_copy_format != d.format) {
        IdleDevice(instance->GetDevice()); // the presenter may read the old copies
        DestroyDepthCopies();
    }
    const bool context_ok = EnsureContext(w, h); // may turn a chosen FSR 4 into FSR 3.1 (missing model files, a fault)
    // the motion in render pixels, previous - current: what DLSS, FSR 4 and TAA read (the normal renderer gives FSR 4 and TAA the same: their
    // RecordGameMotion, mode 2 without a camera), and frame generation
    const bool want_px = Chosen() == Kind::Dlaa || Chosen() == Kind::Fsr4 || Chosen() == Kind::Taa || Core::Vkb::DlssGlue::AnyFgWanted();
    if (!context_ok || (want_px && !EnsureDlaa(w, h)) || (Core::Vkb::DlssGlue::AnyFgWanted() && !EnsureDepthCopies(w, h, d.format))) {
        HandOverLine(nullptr); // a hand-over waiting for an AA that could not be made
        failed = true;
        c_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    float frame_ms = std::chrono::duration<float, std::milli>(now - last_run).count();
    // fresh history after a gap (menu, load) or on the engine's camera cut (api-contract.md K4: the frame the engine clears its own TAA history in)
    const bool reset = frame_id == 0 || frame_ms > 100.0f || camera_cut;
    if (camera_cut) {
        const u64 cuts = c_cut_resets.fetch_add(1, std::memory_order_relaxed) + 1;
        static u64 last_cut_run = 0;
        const u64 run = c_runs.load(std::memory_order_relaxed);
        if (run != last_cut_run + 1) { // the first frame of a run of cut frames (a paused or zero-step stretch is one run)
            LOG_WARNING(Render_Vulkan, "vkb aa: camera cut at AA run {} (cut frames so far {})", run, cuts);
        }
        last_cut_run = run;
    }
    if (frame_ms <= 0.0f || frame_ms > 200.0f) {
        frame_ms = 16.6f;
    }
    last_run = now;
    if (reset) {
        c_resets.fetch_add(1, std::memory_order_relaxed);
    }

    Barrier(cmd);
    // the output starts each frame in GENERAL (FSR writes all of it)
    const vk::ImageMemoryBarrier2 out_barrier{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eNone,
                                              .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                              .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                                              .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral, .image = vk::Image{output},
                                              .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &out_barrier});

    static const float sharpness = [] { const char* e = std::getenv("TLG_VKB_AA_SHARPNESS"); return e != nullptr ? static_cast<float>(std::atof(e)) : 0.5f; }();
    // issue 42 (part s): the player's Sharpening setting, read each frame (the menu changes it live), as the normal renderer: 0..100% is the AA's own
    // RCAS (FSR 3.1's; DLAA has none, so one RCAS pass stands in), every 100% above adds one RCAS pass, the last one partial (their ExtraSharpen)
    const bool setting_sharpen = PicturePart('s') && sharpen_img != VK_NULL_HANDLE && EnsureRcas();
    const auto& settings = TlgSettings::Get();
    const float wanted = !setting_sharpen ? std::min(std::max(sharpness, 0.0f), 1.0f) : settings.sharpen ? std::clamp(settings.sharpness.load(), 0.0f, 4.0f) : 0.0f;
    const float own = std::min(wanted, 1.0f); ///< the AA's own share
    static const bool sharpness_env_set = std::getenv("TLG_VKB_AA_SHARPNESS") != nullptr;
    if (setting_sharpen && sharpness_env_set && !sharpness_env_noted.exchange(true, std::memory_order_relaxed)) {
        // rule 5: the old value is not swapped silently (also in the report: picture_settings.sharpness_env_ignored)
        LOG_INFO(Render_Vulkan, "vkb aa: TLG_VKB_AA_SHARPNESS is not used: the player's Sharpening setting applies (TLG_VKB_PICTURE_SETTINGS part s)");
    }
    if (want_px) {
        // the game's motion as DLSS reads it: render pixels, previous - current (vkb_aa_motion.comp)
        const vk::DescriptorImageInfo game_info{.imageView = ViewOf(m.image, m.format, vk::ImageAspectFlagBits::eColor), .imageLayout = vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo px_info{.imageView = motion_px_view, .imageLayout = vk::ImageLayout::eGeneral};
        const std::array<vk::WriteDescriptorSet, 2> writes{{
            {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &game_info},
            {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &px_info},
        }};
        const vk::ImageMemoryBarrier2 px_barrier{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eNone,
                                                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader, .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                                 .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral, .image = vk::Image{motion_px},
                                                 .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &px_barrier});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, motion_pipeline);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, motion_layout, 0, writes);
        const std::array<float, 2> size{static_cast<float>(w), static_cast<float>(h)};
        cmd.pushConstants(motion_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(size), size.data());
        cmd.dispatch((w + 7) / 8, (h + 7) / 8, 1);
        Barrier(cmd);
        if (Core::Vkb::DlssGlue::AnyFgWanted()) { // F2: this frame's depth (our copy) and motion for the presenter's frame generation
            const vk::Image copy{depth_copies[px_index]};
            const vk::ImageMemoryBarrier2 to_dst{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eNone,
                                                 .dstStageMask = vk::PipelineStageFlagBits2::eTransfer, .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                                 .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral, .image = copy,
                                                 .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}};
            cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_dst});
            const vk::ImageCopy region{.srcSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1}, .dstSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
                                       .extent = {w, h, 1}};
            cmd.copyImage(d.image, vk::ImageLayout::eGeneral, copy, vk::ImageLayout::eGeneral, region);
            Barrier(cmd);
            std::scoped_lock fg_lock{fg_mutex};
            fg_slots[px_index] = {.inputs = {.depth = copy, .motion = vk::Image{motion_px}, .depth_view = depth_copy_views[px_index], .motion_view = motion_px_view,
                                             .depth_format = d.format, .width = w, .height = h, .jitter_x = jitter[0], .jitter_y = jitter[1], .reset = reset},
                                  .value = Core::Vkb::Stream::RecordingValue(),
                                  .serial = ++fg_serial};
        }
    }
    if (Chosen() == Kind::Fsr4) {
        namespace F4 = Core::Vkb::Fsr4;
        if (!EnsureFsr4Inputs(w, h, s.format, d.format)) {
            Fsr4Fault("its input images could not be made");
            Barrier(cmd);
            return;
        }
        // this frame's scene colour and depth into our copies (the depth aspect only), both left in GENERAL
        const std::array<vk::ImageMemoryBarrier2, 2> to_dst{{
            {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eNone,
             .dstStageMask = vk::PipelineStageFlagBits2::eTransfer, .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
             .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral, .image = vk::Image{f4_color},
             .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}},
            {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eNone,
             .dstStageMask = vk::PipelineStageFlagBits2::eTransfer, .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
             .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral, .image = vk::Image{f4_depth},
             .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}},
        }};
        cmd.pipelineBarrier2({.imageMemoryBarrierCount = static_cast<u32>(to_dst.size()), .pImageMemoryBarriers = to_dst.data()});
        const vk::ImageCopy color_region{.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                         .extent = {w, h, 1}};
        const vk::ImageCopy depth_region{.srcSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1}, .dstSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
                                         .extent = {w, h, 1}};
        cmd.copyImage(s.image, vk::ImageLayout::eGeneral, vk::Image{f4_color}, vk::ImageLayout::eGeneral, color_region);
        cmd.copyImage(d.image, vk::ImageLayout::eGeneral, vk::Image{f4_depth}, vk::ImageLayout::eGeneral, depth_region);
        Barrier(cmd);
        F4::Frame f{.cmd = cmd,
                    .color = {vk::Image{f4_color}, f4_color_view, w, h},
                    .depth = {vk::Image{f4_depth}, f4_depth_view, w, h},
                    .motion = {vk::Image{motion_px}, motion_px_view, w, h},
                    .output = {vk::Image{output}, output_view, w, h}};
        // the normal renderer's jitter sign rule for FSR 4 (their RecordFsr4: the same convention as FSR 3, flipped by the menu's invert setting)
        const float sign = settings.fsr4_invert_jitter.load() ? -1.0f : 1.0f;
        f.jitter = {sign * jitter[0], sign * jitter[1]};
        f.frame_ms = frame_ms;
        f.near_plane = 0.1f, f.far_plane = 3000.0f, f.vertical_fov = 1.0f; // as the FSR 3.1 path below: the game's camera is not read yet
        f.sharpness = own;
        f.reset = reset;
        f.auto_exposure = settings.fsr4_auto_exposure.load();
        ++frame_id;
        if (!F4::Record(f)) {
            Fsr4Fault(F4::Problem()); // this frame keeps the scene colour as drawn; FSR 3.1 from the next one
            Barrier(cmd);
            return;
        }
        c_fsr4_runs.fetch_add(1, std::memory_order_relaxed);
        goto copy_back;
    }
    if (Chosen() == Kind::Taa) {
        // their RecordTaa: the histories alternate between read and write (a reset starts both from nothing; the shader then reads no history)
        std::array<vk::ImageMemoryBarrier2, 2> hist_barriers{};
        for (u32 i = 0; i < hist_barriers.size(); ++i) {
            hist_barriers[i] = {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eShaderStorageWrite,
                                .oldLayout = reset ? vk::ImageLayout::eUndefined : vk::ImageLayout::eGeneral, .newLayout = vk::ImageLayout::eGeneral,
                                .image = vk::Image{taa_hist[i]}, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        }
        cmd.pipelineBarrier2({.imageMemoryBarrierCount = static_cast<u32>(hist_barriers.size()), .pImageMemoryBarriers = hist_barriers.data()});
        // their bindings: colour, depth, motion (pixels), history read; result, history written, diagnostic (their TLG_TAA_DIAGNOSTICS only, never written
        // here: diagnostics 0, so the output stands in)
        const std::array<vk::ImageView, 7> views{ViewOf(s.image, s.format, vk::ImageAspectFlagBits::eColor), ViewOf(d.image, d.format, vk::ImageAspectFlagBits::eDepth),
                                                 motion_px_view, taa_hist_views[1 - taa_next], output_view, taa_hist_views[taa_next], output_view};
        std::array<vk::DescriptorImageInfo, 7> infos{};
        std::array<vk::WriteDescriptorSet, 7> writes{};
        for (u32 i = 0; i < infos.size(); ++i) {
            infos[i] = {.sampler = i < 4 ? taa_sampler : vk::Sampler{}, .imageView = views[i], .imageLayout = vk::ImageLayout::eGeneral};
            writes[i] = {.dstBinding = i, .descriptorCount = 1,
                         .descriptorType = i < 4 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage, .pImageInfo = &infos[i]};
        }
        // their push constants: the jitter, the reset, the A/B toggles (TLG_TOGGLE_FILE bits 51-55, default all off) and the camera for the depth test
        // (current projection, previous projection, row Z of previous view * inverse view). The new engine reads no camera, and the normal renderer finds
        // none in this game either (CameraMotion::OnConstants; its logs say "no game camera"): both give zeros, the same depth test
        struct Params {
            std::array<float, 2> jitter;
            u32 reset, pad;
            std::array<std::array<float, 4>, 3> depth;
        } params{jitter, reset ? 1u : 0u, 0u, {}};
        static_assert(sizeof(Params) == 64);
        params.pad |= (TlgToggle::Disabled(TlgToggle::TaaTonemapBlend) ? 1u << 8 : 0u) | (TlgToggle::Disabled(TlgToggle::TaaClip) ? 2u << 8 : 0u) |
                      (TlgToggle::Disabled(TlgToggle::TaaVariance) ? 4u << 8 : 0u) | (TlgToggle::Disabled(TlgToggle::TaaFilter) ? 8u << 8 : 0u) |
                      (TlgToggle::Disabled(TlgToggle::TaaKeepNearerHistory) ? 16u << 8 : 0u);
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, taa_pipeline);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, taa_layout, 0, writes);
        cmd.pushConstants(taa_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params), &params);
        cmd.dispatch((w + 7) / 8, (h + 7) / 8, 1);
        taa_next = 1 - taa_next;
        ++frame_id;
        c_taa_runs.fetch_add(1, std::memory_order_relaxed);
        goto copy_back;
    }
    if (Chosen() == Kind::Dlaa) {
        using Core::Vkb::DlssGlue::Image;
        const Image color{s.image, ViewOf(s.image, s.format, vk::ImageAspectFlagBits::eColor), s.format, vk::ImageAspectFlagBits::eColor, w, h};
        const Image dimg{d.image, ViewOf(d.image, d.format, vk::ImageAspectFlagBits::eDepth), d.format, vk::ImageAspectFlagBits::eDepth, w, h};
        const Image mimg{vk::Image{motion_px}, motion_px_view, vk::Format::eR16G16Sfloat, vk::ImageAspectFlagBits::eColor, w, h};
        const Image oimg{vk::Image{output}, output_view, vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor, w, h};
        ++frame_id; // the history counter both paths share (a reset only after a gap)
        if (!Core::Vkb::DlssGlue::Run(instance->GetDevice(), cmd, color, dimg, mimg, oimg, jitter[0], jitter[1], reset, frame_ms, setting_sharpen ? 0.0f : own)) {
            if (DlaaFallbackSwitch()) {
                // issue 99: this frame keeps the scene colour as drawn; the AA that takes over runs from the next one (EnsureContext makes it)
                std::string why;
                if (Core::Vkb::DlssGlue::Usable(why)) {
                    why = "DLSS did not run on a frame";
                }
                HandOver(From::Dlaa, *instance, why, true);
                Core::Vkb::DlssGlue::Release(instance->GetDevice()); // DLAA is not used again this session
                Barrier(cmd);
                return;
            }
            LOG_ERROR(Render_Vulkan, "vkb aa: DLAA failed; AA off"); // TLG_VKB_DLAA_FALLBACK=0: the old way
            failed = true;
            c_failed.fetch_add(1, std::memory_order_relaxed);
            Barrier(cmd);
            return;
        }
        goto copy_back;
    }
    {
    FfxVkPortableUpscaleDispatchInfo info{};
    info.structSize = sizeof(info);
    info.commandBuffer = cmd;
    info.color = Describe(s.image, static_cast<VkFormat>(s.format), w, h, static_cast<VkImageUsageFlags>(UsageOf(s)), VK_IMAGE_ASPECT_COLOR_BIT,
                          FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    info.depth = Describe(d.image, static_cast<VkFormat>(d.format), w, h, static_cast<VkImageUsageFlags>(UsageOf(d)), VK_IMAGE_ASPECT_DEPTH_BIT,
                          FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    // the game's motion target: UV units, current - previous, unjittered (their camera_motion.comp mode 2); FSR wants pixels toward the previous frame
    info.motionVectors = Describe(m.image, static_cast<VkFormat>(m.format), w, h, static_cast<VkImageUsageFlags>(UsageOf(m)), VK_IMAGE_ASPECT_COLOR_BIT,
                                  FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    info.output = Describe(output, VK_FORMAT_R16G16B16A16_SFLOAT, w, h, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT, FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS);
    info.exposure.structSize = sizeof(info.exposure);
    info.reactiveMask.structSize = sizeof(info.reactiveMask);
    info.transparencyAndCompositionMask.structSize = sizeof(info.transparencyAndCompositionMask);
    info.jitterOffset = {jitter[0], jitter[1]}; // F1c: the offset this frame's geometry had (pixels; their sign convention)
    info.motionVectorScale = {-static_cast<float>(w), -static_cast<float>(h)};
    info.renderSize = {w, h};
    info.outputSize = {w, h};
    info.frameTimeMilliseconds = frame_ms;
    info.preExposure = 1.0f;
    info.cameraNear = 0.1f; // their fallback: the game's camera is not read yet
    info.cameraFar = 3000.0f;
    info.cameraVerticalFovRadians = 1.0f;
    info.viewSpaceToMeters = 1.0f;
    info.sharpness = own;
    info.enableSharpening = own > 0.0f ? VK_TRUE : VK_FALSE;
    info.reset = reset ? VK_TRUE : VK_FALSE;
    info.frameId = frame_id++;
    FfxVkPortableUpscaleCreateInfo create_info{};
    create_info.structSize = sizeof(create_info);
    create_info.flags = FFX_VK_PORTABLE_CONTEXT_HDR_COLOR_INPUT | FFX_VK_PORTABLE_CONTEXT_AUTO_EXPOSURE;
    create_info.maxRenderSize = {w, h};
    create_info.maxOutputSize = {w, h};
    if (const u64 issues = ffxVkPortableValidateUpscaleDispatchInfo(&create_info, &info); issues != 0) {
        LOG_ERROR(Render_Vulkan, "vkb aa: FSR 3.1 dispatch rejected ({:#x}); AA off", issues);
        failed = true;
        c_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (ffxVkPortableUpscaleContextRecordDispatch(context, &info) != FFX_VK_PORTABLE_OK) {
        LOG_ERROR(Render_Vulkan, "vkb aa: FSR 3.1 dispatch failed; AA off");
        failed = true;
        c_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    }
copy_back:
    Barrier(cmd);
    VkImage result = output;
    if (setting_sharpen) {
        // the passes: DLAA's and TAA's own share first (they have no RCAS of their own; their RecordTaa runs one RCAS pass at 0..100%), then one per
        // 100% above
        std::array<float, 5> passes{};
        u32 count = 0;
        if ((Chosen() == Kind::Dlaa || Chosen() == Kind::Taa) && own > 0.0f) {
            passes[count++] = own;
        }
        for (float extra = wanted - 1.0f; extra > 0.0f && count < passes.size(); extra -= 1.0f) {
            passes[count++] = std::min(extra, 1.0f);
        }
        if (count != 0) {
            const vk::ImageMemoryBarrier2 to_general{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands, .srcAccessMask = vk::AccessFlagBits2::eNone,
                                                     .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                                     .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                                                     .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral, .image = vk::Image{sharpen_img},
                                                     .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
            cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_general});
        }
        vk::ImageView from = output_view, to = sharpen_view;
        for (u32 i = 0; i < count; ++i) {
            RcasPass(cmd, from, to, w, h, passes[i]);
            std::swap(from, to);
            result = result == output ? sharpen_img : output;
        }
    }
    // the result replaces the scene colour (B10G11R11: no alpha to keep); both images stay in GENERAL (note 66)
    const vk::ImageBlit region{.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                               .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<s32>(w), static_cast<s32>(h), 1}},
                               .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                               .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<s32>(w), static_cast<s32>(h), 1}}};
    cmd.blitImage(vk::Image{result}, vk::ImageLayout::eGeneral, s.image, vk::ImageLayout::eGeneral, region, vk::Filter::eNearest);
    Barrier(cmd);
    c_runs.fetch_add(1, std::memory_order_relaxed);
}

std::atomic<u64> c_fg_made{0}, c_fg_presented{0}, c_fg_failed{0};
void NoteFg(bool made, bool presented, bool failed) {
    (made ? c_fg_made : presented ? c_fg_presented : c_fg_failed).fetch_add(1, std::memory_order_relaxed);
    (void)failed;
}

bool TakeFgInputs(FgInputs& out) {
    const u64 done = Core::Vkb::Stream::CompletedValue();
    std::scoped_lock lock{fg_mutex};
    const FgSlot* best = nullptr;
    for (const FgSlot& slot : fg_slots) {
        if (slot.serial > fg_taken && slot.value != 0 && slot.value <= done && (best == nullptr || slot.serial > best->serial)) {
            best = &slot;
        }
    }
    if (best == nullptr) {
        return false;
    }
    fg_taken = best->serial;
    out = best->inputs;
    return true;
}

SamplerLook LookFor(u32 mrt_mask) {
    SamplerLook look{};
    const auto& settings = TlgSettings::Get();
    if (PicturePart('a')) {
        look.aniso = static_cast<u32>(std::max(0, settings.aniso.load())); // 0, 2, 4, 8 or 16 (tlg_settings.cpp)
    }
    // their Rasterizer::BindResources + TemporalUpscaler::SceneMipBias: a graphics pipeline with 5 or more colour outputs (the scene's material pass)
    // samples with the Lab tab's bias in quarter steps while DLAA or FSR native AA runs. Their bias also has a log2(render / output) term, which is 0
    // here by construction: the new engine's AA is native AA only (its context is made at one size for input and output, EnsureContext), it has no
    // upscaling mode (their TLG_RENDER_RESOLUTION engine upscale and scaled sessions are not part of the new engine)
    // their SceneMipBias is 0 under TAA: no bias then
    if (PicturePart('b') && std::popcount(mrt_mask & 0xffu) >= 5 && Running() && Chosen() != Kind::Taa) {
        look.bias_q = settings.lod_bias.load();
        if (look.bias_q != 0) {
            c_scene_bias_draws.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return look;
}

static std::string Why(const std::array<std::string, u32(From::Count)>& list, From from) {
    std::scoped_lock lock{handover_mutex};
    return list[u32(from)];
}

nlohmann::json Report() {
    return {{"on", On()},
            {"kind", Chosen() == Kind::Fsr3 ? "fsr3" : Chosen() == Kind::Dlaa ? "dlaa" : Chosen() == Kind::Fsr4 ? "fsr4" : Chosen() == Kind::Taa ? "taa" : "off"},
            {"asked", asked},
            {"live", {{"switch", Core::Vkb::DlssGlue::LiveAaSwitch()}, {"switches", c_switches.load()},
                      {"menu", menu_code < 0 ? "" : TlgSettings::UpscalerName(menu_code)}}},
            {"choice", {{"switch", ChoiceSwitch()}, {"faults", c_handover_faults[u32(From::Choice)].load()}, {"fault", Why(handover_fault, From::Choice)}}},
            {"fsr4", {{"switch", Fsr4Switch()}, {"device_supported", instance != nullptr && instance->IsFsr4Int8Supported()},
                      {"rdna4", instance != nullptr && IsRdna4(*instance)}, {"model_missing", c_handover_quiet[u32(From::Fsr4)].load() != 0}, {"runs", c_fsr4_runs.load()},
                      {"faults", c_handover_faults[u32(From::Fsr4)].load()}, {"fault", Why(handover_fault, From::Fsr4)}}},
            {"dlaa", {{"fallback_switch", DlaaFallbackSwitch()}, {"faults", c_handover_faults[u32(From::Dlaa)].load()},
                      {"not_installed", c_handover_quiet[u32(From::Dlaa)].load()}, {"fault", Why(handover_fault, From::Dlaa)}}},
            {"took_over", [] { std::scoped_lock lock{handover_mutex}; return took_over; }()}, // issue 99: the AA of the last hand-over ("" none, "off")
            {"taa", {{"switch", TaaSwitch()}, {"runs", c_taa_runs.load()}, {"faults", c_handover_faults[u32(From::Taa)].load()},
                     {"fault", Why(handover_fault, From::Taa)},
                     {"camera", false}}}, // no camera for its depth test (the normal renderer finds none in this game either)
            {"size", fmt::format("{}x{}", ctx_w, ctx_h)},
            {"motion_passes", c_motion_passes.load()},
            {"runs", c_runs.load()},
            {"combines_without_inputs", c_no_inputs.load()},
            {"unusable", c_unusable.load()},
            {"failed", c_failed.load()},
            {"history_resets", c_resets.load()},
            {"history_resets_camera_cut", c_cut_resets.load()},
            {"jitter", JitterWanted()},
            {"jittered_draws", c_jittered.load()},
            {"jittered_mesh_redraws", c_jittered_meshes.load()},
            {"mesh_key_index_data", MeshKeySwitch()}, // issue 54
            {"picture_settings", {{"fallback_fsr3", PicturePart('f')}, {"sharpening", PicturePart('s')}, {"scene_bias", PicturePart('b')}, {"aniso", PicturePart('a')},
                                  {"rcas_passes", c_rcas_passes.load()}, {"rcas_broken", rcas_broken}, {"sharpness_env_ignored", sharpness_env_noted.load()},
                                  {"scene_bias_draws", c_scene_bias_draws.load()}}},
            {"frame_generation", {{"made", c_fg_made.load()}, {"presented", c_fg_presented.load()}, {"failed", c_fg_failed.load()}}}};
}

} // namespace Core::Vkb::Aa
