// SPDX-FileCopyrightText: Copyright 2026 tlg contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// The XeSS SDK is used through its public C interface only (as documented in Intel's XeSS-SR
// developer guide): the few types and entry points below are declared here and resolved from
// libxess.dll at run time, so building the port needs no Intel files.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string_view>

#include "video_core/renderer_vulkan/vk_xess.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Vulkan {

namespace {
// xess_quality_settings_t
constexpr int QualityUltraPerformance = 100;
constexpr int QualityPerformance = 101;
constexpr int QualityBalanced = 102;
constexpr int QualityQuality = 103;
constexpr int QualityNativeAa = 106;
} // namespace

int Xess::QualityForScale(float scale) {
    // Native AA only for equal sizes; XeSS 2+ ratios: Q 1.5, B 1.7, P 2.0, UP 3.0.
    return scale >= 2.9f    ? QualityUltraPerformance
           : scale >= 1.95f ? QualityPerformance
           : scale >= 1.65f ? QualityBalanced
           : scale > 1.01f  ? QualityQuality
                            : QualityNativeAa;
}

#ifdef _WIN32

namespace {
constexpr wchar_t LibraryName[] = L"libxess.dll";

// The SDK's public ABI (xess.h / xess_vk.h, structures packed to 8 bytes).
using XessContext = struct XessContextOpaque*;
using XessResult = int; // 0 success, > 0 warning, < 0 error

#pragma pack(push, 8)
struct Xess2d {
    u32 x, y;
};
struct XessVersion {
    u16 major, minor, patch, reserved;
};
struct XessImageView {
    VkImageView image_view;
    VkImage image;
    VkImageSubresourceRange subresource_range;
    VkFormat format;
    unsigned int width, height;
};
struct XessExecuteParams {
    XessImageView color, velocity, depth, exposure_scale, responsive_mask, output;
    float jitter_x, jitter_y, exposure_scale_value;
    u32 reset_history, input_width, input_height;
    Xess2d color_base, velocity_base, depth_base, responsive_mask_base, reserved0, output_base;
};
struct XessInitParams {
    Xess2d output_resolution;
    int quality;
    u32 flags, creation_node_mask, visible_node_mask;
    VkDeviceMemory temp_buffer_heap;
    u64 buffer_heap_offset;
    VkDeviceMemory temp_texture_heap;
    u64 texture_heap_offset;
    VkPipelineCache pipeline_cache;
};
#pragma pack(pop)

// xess_init_flags_t
constexpr u32 FlagLdrInputColor = 1u << 6;
constexpr u32 FlagAutoExposure = 1u << 8;

using LogCallback = void (*)(const char* message, int level);
using GetVersionFn = XessResult (*)(XessVersion*);
using InstanceExtensionsFn = XessResult (*)(u32*, const char* const**, u32*);
using DeviceExtensionsFn = XessResult (*)(VkInstance, VkPhysicalDevice, u32*, const char* const**);
using DeviceFeaturesFn = XessResult (*)(VkInstance, VkPhysicalDevice, void**);
using CreateContextFn = XessResult (*)(VkInstance, VkPhysicalDevice, VkDevice, XessContext*);
using InitFn = XessResult (*)(XessContext, const XessInitParams*);
using ExecuteFn = XessResult (*)(XessContext, VkCommandBuffer, const XessExecuteParams*);
using DestroyContextFn = XessResult (*)(XessContext);
using SetLoggingFn = XessResult (*)(XessContext, int, LogCallback);
using IsOptimalDriverFn = XessResult (*)(XessContext);
using SetScaleFn = XessResult (*)(XessContext, float, float);

std::filesystem::path ExecutableDirectory() {
    std::wstring path(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    path.resize(length);
    return std::filesystem::path{path}.parent_path();
}

void Log(const char* message, int level) {
    // Debug/info (0/1) are noisy: keep warnings and errors.
    if (level >= 2) {
        std::printf("XeSS: %s%s\n", level == 2 ? "warning: " : "", message);
    }
}

bool AddExtensions(const char* const* names, u32 count,
                   const std::vector<vk::ExtensionProperties>& available,
                   std::vector<std::string>& storage, std::vector<const char*>& enabled) {
    storage.clear();
    for (u32 i = 0; i < count; ++i) {
        const auto match = std::ranges::find_if(available, [&](const auto& extension) {
            return std::strcmp(extension.extensionName.data(), names[i]) == 0;
        });
        if (match == available.end()) {
            std::printf("XeSS: missing Vulkan extension %s\n", names[i]);
            return false;
        }
        storage.emplace_back(names[i]);
    }
    for (const auto& name : storage) {
        if (std::ranges::none_of(enabled, [&](const char* current) { return name == current; })) {
            enabled.push_back(name.c_str());
        }
    }
    return true;
}
} // namespace

struct Xess::Impl {
    HMODULE module{};
    GetVersionFn get_version{};
    InstanceExtensionsFn instance_extensions_fn{};
    DeviceExtensionsFn device_extensions_fn{};
    DeviceFeaturesFn device_features_fn{};
    CreateContextFn create_context{};
    InitFn init{};
    ExecuteFn execute{};
    DestroyContextFn destroy_context{};
    SetLoggingFn set_logging{};
    IsOptimalDriverFn is_optimal_driver{};
    SetScaleFn set_jitter_scale{};

    vk::Instance instance;
    vk::PhysicalDevice physical;
    vk::Device device;
    XessContext context{};
    std::vector<std::string> instance_extensions, device_extensions;
    std::optional<FeatureDesc> feature;
    std::string problem;
    bool eligible{true}, instance_ready{}, device_ready{}, available{};

    void Disable(std::string_view reason) {
        eligible = available = false;
        problem = reason;
        std::printf("XeSS: unavailable (%.*s)\n", int(reason.size()), reason.data());
    }

    template <typename T>
    bool Resolve(T& target, const char* name) {
        target = reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(module, name)));
        return target != nullptr;
    }
};

Xess* Xess::Get() {
    static Xess* const xess = []() -> Xess* {
        const char* setting = std::getenv("TLG_XESS");
        if (setting && setting[0] == '0') {
            return nullptr;
        }
        if (!std::filesystem::is_regular_file(ExecutableDirectory() / LibraryName)) {
            return nullptr;
        }
        auto* created = new Xess;
        std::atexit([] {
            if (Xess* live = Get()) {
                live->Shutdown();
            }
        });
        return created;
    }();
    return xess;
}

Xess::Xess() : impl{std::make_unique<Impl>()} {
    impl->module = LoadLibraryW((ExecutableDirectory() / LibraryName).c_str());
    if (!impl->module) {
        impl->Disable("libxess.dll could not be loaded");
        return;
    }
    const bool resolved =
        impl->Resolve(impl->get_version, "xessGetVersion") &&
        impl->Resolve(impl->instance_extensions_fn, "xessVKGetRequiredInstanceExtensions") &&
        impl->Resolve(impl->device_extensions_fn, "xessVKGetRequiredDeviceExtensions") &&
        impl->Resolve(impl->device_features_fn, "xessVKGetRequiredDeviceFeatures") &&
        impl->Resolve(impl->create_context, "xessVKCreateContext") &&
        impl->Resolve(impl->init, "xessVKInit") && impl->Resolve(impl->execute, "xessVKExecute") &&
        impl->Resolve(impl->destroy_context, "xessDestroyContext") &&
        impl->Resolve(impl->set_logging, "xessSetLoggingCallback") &&
        impl->Resolve(impl->is_optimal_driver, "xessIsOptimalDriver") &&
        impl->Resolve(impl->set_jitter_scale, "xessSetJitterScale");
    if (!resolved) {
        impl->Disable("libxess.dll is from an unsupported version (no Vulkan interface)");
        return;
    }
    XessVersion version{};
    if (impl->get_version(&version) == 0) {
        std::printf("XeSS: library %u.%u.%u\n", version.major, version.minor, version.patch);
        if (version.major < 2) {
            impl->Disable("libxess.dll is older than XeSS 2");
        }
    }
}

Xess::~Xess() {
    Shutdown();
}

void Xess::AppendInstanceExtensions(std::vector<const char*>& enabled) {
    if (!impl->eligible) {
        return;
    }
    u32 count{}, min_api{};
    const char* const* names{};
    if (impl->instance_extensions_fn(&count, &names, &min_api) < 0) {
        return impl->Disable("instance extension query failed");
    }
    if (min_api > VK_API_VERSION_1_3) {
        return impl->Disable("needs a newer Vulkan version than the port uses");
    }
    const auto [result, extensions] = vk::enumerateInstanceExtensionProperties();
    if (result != vk::Result::eSuccess ||
        !AddExtensions(names, count, extensions, impl->instance_extensions, enabled)) {
        return impl->Disable("required Vulkan instance extensions are missing");
    }
    impl->instance_ready = true;
}

void Xess::AppendDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                                  std::vector<const char*>& enabled) {
    if (!impl->eligible || !impl->instance_ready) {
        return;
    }
    // XeSS's Vulkan path needs shader stores without a format and mutable descriptors.
    vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT mutable_features{};
    vk::PhysicalDeviceFeatures2 features{};
    features.pNext = &mutable_features;
    physical.getFeatures2(&features);
    if (!features.features.shaderStorageImageWriteWithoutFormat ||
        !mutable_features.mutableDescriptorType) {
        return impl->Disable("this GPU or driver lacks Vulkan features XeSS needs");
    }
    u32 count{};
    const char* const* names{};
    if (impl->device_extensions_fn(instance, physical, &count, &names) < 0) {
        return impl->Disable("device extension query failed");
    }
    const auto [result, extensions] = physical.enumerateDeviceExtensionProperties();
    if (result != vk::Result::eSuccess ||
        !AddExtensions(names, count, extensions, impl->device_extensions, enabled)) {
        return impl->Disable("required Vulkan device extensions are missing");
    }
    impl->device_ready = true;
}

void* Xess::AppendDeviceFeatures(vk::Instance instance, vk::PhysicalDevice physical,
                                 void* features2) {
    if (!impl->eligible || !impl->device_ready) {
        return features2;
    }
    void* chain = features2;
    if (impl->device_features_fn(instance, physical, &chain) < 0 || !chain) {
        impl->device_ready = false;
        impl->Disable("device feature query failed");
        return features2;
    }
    return chain;
}

void Xess::Initialize(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device) {
    if (!impl->eligible || !impl->device_ready || impl->available) {
        return;
    }
    impl->instance = instance;
    impl->physical = physical;
    impl->device = device;
    // One context is made now to prove XeSS runs here; CreateFeature initializes it per size.
    if (impl->create_context(instance, physical, device, &impl->context) < 0 || !impl->context) {
        impl->context = {};
        return impl->Disable("XeSS context creation failed (update the GPU driver)");
    }
    impl->set_logging(impl->context, 2, Log);
    if (impl->is_optimal_driver(impl->context) != 0) {
        std::printf("XeSS: warning: the GPU driver is older than XeSS recommends\n");
    }
    impl->available = true;
    std::printf("XeSS: ready\n");
}

bool Xess::Available() const {
    return impl->available;
}

std::string Xess::Problem() const {
    return impl->problem;
}

bool Xess::HasFeature(const FeatureDesc& desc) const {
    return impl->feature && *impl->feature == desc;
}

bool Xess::CreateFeature(const FeatureDesc& desc) {
    if (!impl->available) {
        return false;
    }
    impl->feature.reset();
    // A context is initialized once per setup: start from a fresh one on every change.
    if (impl->context) {
        impl->destroy_context(impl->context);
        impl->context = {};
    }
    if (impl->create_context(impl->instance, impl->physical, impl->device, &impl->context) < 0 ||
        !impl->context) {
        impl->context = {};
        return false;
    }
    impl->set_logging(impl->context, 2, Log);
    XessInitParams init{};
    init.output_resolution = {desc.output_width, desc.output_height};
    init.quality = desc.quality;
    // Depth: the port's depth is not inverted (near = 0), XeSS's default.
    init.flags = desc.hdr ? (desc.auto_exposure ? FlagAutoExposure : 0) : FlagLdrInputColor;
    const XessResult result = impl->init(impl->context, &init);
    if (result < 0) {
        std::printf("XeSS: init failed (%d) for %ux%u -> %ux%u, quality %d\n", result,
                    desc.input_width, desc.input_height, desc.output_width, desc.output_height,
                    desc.quality);
        return false;
    }
    impl->feature = desc;
    std::printf("XeSS: %ux%u -> %ux%u, quality %d\n", desc.input_width, desc.input_height,
                desc.output_width, desc.output_height, desc.quality);
    return true;
}

bool Xess::Evaluate(vk::CommandBuffer command, const Frame& frame) {
    if (!impl->available || !impl->feature || !impl->context) {
        return false;
    }
    const auto image = [](const Resource& r) {
        return XessImageView{static_cast<VkImageView>(r.view),
                             static_cast<VkImage>(r.image),
                             {static_cast<VkImageAspectFlags>(r.aspect), 0, 1, 0, 1},
                             static_cast<VkFormat>(r.format),
                             r.width,
                             r.height};
    };
    XessExecuteParams parameters{};
    parameters.color = image(frame.color);
    parameters.velocity = image(frame.motion);
    parameters.depth = image(frame.depth);
    parameters.output = image(frame.output);
    parameters.jitter_x = frame.jitter_x;
    parameters.jitter_y = frame.jitter_y;
    parameters.exposure_scale_value = 1.0f;
    parameters.reset_history = frame.reset ? 1 : 0;
    parameters.input_width = frame.input_width;
    parameters.input_height = frame.input_height;
    const XessResult result =
        impl->execute(impl->context, static_cast<VkCommandBuffer>(command), &parameters);
    if (result < 0) {
        std::printf("XeSS: execute failed (%d)\n", result);
        return false;
    }
    return true;
}

void Xess::ReleaseFeature() {
    impl->feature.reset();
}

void Xess::Shutdown() {
    impl->available = false;
    impl->feature.reset();
    if (impl->context && impl->destroy_context) {
        impl->destroy_context(impl->context);
        impl->context = {};
    }
}

#else

// The XeSS SDK has no Linux library: XeSS stays unavailable there.
struct Xess::Impl {};
Xess::Xess() = default;
Xess::~Xess() = default;
Xess* Xess::Get() {
    return nullptr;
}
void Xess::AppendInstanceExtensions(std::vector<const char*>&) {}
void Xess::AppendDeviceExtensions(vk::Instance, vk::PhysicalDevice, std::vector<const char*>&) {}
void* Xess::AppendDeviceFeatures(vk::Instance, vk::PhysicalDevice, void* features2) {
    return features2;
}
void Xess::Initialize(vk::Instance, vk::PhysicalDevice, vk::Device) {}
void Xess::Shutdown() {}
bool Xess::Available() const {
    return false;
}
std::string Xess::Problem() const {
    return "XeSS is only available on Windows";
}
bool Xess::HasFeature(const FeatureDesc&) const {
    return false;
}
bool Xess::CreateFeature(const FeatureDesc&) {
    return false;
}
bool Xess::Evaluate(vk::CommandBuffer, const Frame&) {
    return false;
}
void Xess::ReleaseFeature() {}

#endif

} // namespace Vulkan
