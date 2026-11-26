// SPDX-FileCopyrightText: Copyright 2026 tlg contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// tlg: Intel XeSS Super Resolution (Vulkan backend of the XeSS SDK) as an alternative to DLSS and
// FSR. Intel's libxess.dll is loaded at run time from next to tlg-probe.exe (tools/fetch_xess.sh
// puts it there, unchanged, with Intel's license); the port contains no Intel code. Without the
// DLL, on Linux (the SDK has no Linux build) or with TLG_XESS=0 nothing changes.
//
// It takes the same inputs as DLSS: HDR scene color, depth, motion vectors in render pixels
// (current to previous) and the jitter offset, and writes the output-size image.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Xess {
public:
    /// The process-wide instance, or null when libxess.dll is absent or TLG_XESS=0.
    static Xess* Get();
    ~Xess();

    /// Instance creation: adds the instance extensions XeSS needs.
    void AppendInstanceExtensions(std::vector<const char*>& enabled);
    /// Device creation: adds its device extensions. Disables XeSS when the device lacks them.
    void AppendDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                                std::vector<const char*>& enabled);
    /// Right before vkCreateDevice: XeSS fills in / appends its features on the chain that
    /// starts at `features2` (the VkPhysicalDeviceFeatures2). Returns the chain head to use as
    /// VkDeviceCreateInfo::pNext.
    void* AppendDeviceFeatures(vk::Instance instance, vk::PhysicalDevice physical, void* features2);
    /// After the device exists.
    void Initialize(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device);
    void Shutdown();

    [[nodiscard]] bool Available() const;
    /// Why XeSS cannot run on this system (empty when it can, or before device creation).
    [[nodiscard]] std::string Problem() const;

    struct Resource {
        vk::Image image;
        vk::ImageView view;
        vk::Format format;
        vk::ImageAspectFlags aspect;
        u32 width, height;
    };
    struct FeatureDesc {
        u32 input_width, input_height, output_width, output_height;
        int quality; ///< xess_quality_settings_t
        bool hdr = true;           ///< false: the color input is already tone mapped (LDR)
        bool auto_exposure = true; ///< HDR only: let XeSS measure the exposure
        bool operator==(const FeatureDesc&) const = default;
    };
    struct Frame {
        Resource color, depth, motion, output;
        u32 input_width, input_height; ///< render size (may be smaller than the color image)
        float jitter_x, jitter_y;
        bool reset;
    };

    /// XeSS quality setting for an output/render size ratio (native AA .. Ultra Performance).
    static int QualityForScale(float scale);
    [[nodiscard]] bool HasFeature(const FeatureDesc& desc) const;
    /// (Re)initializes the context for `desc`. The GPU must be done with the previous one.
    bool CreateFeature(const FeatureDesc& desc);
    /// Records XeSS into `command`. Inputs must be in ShaderReadOnlyOptimal, the output in General.
    bool Evaluate(vk::CommandBuffer command, const Frame& frame);
    void ReleaseFeature();

private:
    Xess();
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
