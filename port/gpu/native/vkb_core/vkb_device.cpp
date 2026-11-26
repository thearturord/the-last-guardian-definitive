// SPDX-FileCopyrightText: Copyright 2026 tlg_pc
// SPDX-License-Identifier: GPL-2.0-or-later

#include "native/vkb_core/vkb_device.h"

#include <array>
#include <cstdlib>
#include <memory>
#include "common/assert.h"
#include "common/logging/log.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_host_import.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_instance.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_resource_pool.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_scheduler.h"
#include "native/vkb_core/profile.h"

namespace Core::Vkb::Device {
namespace {

/// The old port's pipeline cache heap sizes
constexpr std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

const VkbVulkan::Instance* g_instance = nullptr;
VkbVulkan::Scheduler* g_scheduler = nullptr;
std::unique_ptr<VkbVulkan::DescriptorHeap> g_desc_heap;
VkbShader::Profile g_profile{};

} // namespace

void Init(const VkbVulkan::Instance& instance, VkbVulkan::Scheduler& scheduler) {
    g_instance = &instance;
    g_scheduler = &scheduler;
    g_desc_heap = std::make_unique<VkbVulkan::DescriptorHeap>(instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes);
    g_profile = VideoCore::Vkb::ProfileFor(instance);
    if (const char* e = std::getenv("TLG_VKB_LEAF_AO"); e != nullptr && *e != 0) { // the old switch of this look option
        LOG_WARNING(Render_Vulkan, "TLG_VKB_LEAF_AO={} is not read any more: use TLG_LEAF_AO (same form) or the Extras setting Realistic tree shade", e);
    }
    VkbVulkan::GpuPool::Setup(instance); // before the game has memory: the guest pool may become our memory (TLG_VKB_GPU_POOL)
}

bool Ready() noexcept {
    return g_instance != nullptr;
}

const VkbVulkan::Instance& Instance() {
    ASSERT_MSG(g_instance != nullptr, "vkb device: used before the backend's presenter made it");
    return *g_instance;
}

VkbVulkan::Scheduler& Scheduler() {
    ASSERT_MSG(g_scheduler != nullptr, "vkb device: used before the backend's presenter made it");
    return *g_scheduler;
}

VkbVulkan::DescriptorHeap& DescHeap() {
    ASSERT_MSG(g_desc_heap != nullptr, "vkb device: used before the backend's presenter made it");
    return *g_desc_heap;
}

const VkbShader::Profile& Profile() {
    return g_profile;
}

} // namespace Core::Vkb::Device
