// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "native/native_prof.h"
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include "native/native_display.h"
#include "native/native_flight.h"
#include "native/native_presenter.h"
#include "native/vkb_core/vkb_presenter.h"
#include "common/assert.h"
#include "tlg_toggles.h"
#include "tlg_freeze.h"
#include "tlg_settings.h"
#include "tlg_graphics.h"
#include "common/debug.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "native/shadps4/core/libraries/videoout/driver.h"
#include "native/shadps4/core/libraries/videoout/videoout_error.h"
#include "native/native_vkb_shader_store.h"

using VkbVulkan::g_presenter;

namespace Libraries::VkbVideoOut {

constexpr static bool Is32BppPixelFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8R8G8B8Srgb:
    case PixelFormat::A8B8G8R8Srgb:
    case PixelFormat::A2R10G10B10:
    case PixelFormat::A2R10G10B10Srgb:
    case PixelFormat::A2R10G10B10Bt2020Pq:
        return true;
    default:
        return false;
    }
}

constexpr u32 PixelFormatBpp(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::A16R16G16B16Float:
        return 8;
    default:
        return 4;
    }
}

VideoOutDriver::VideoOutDriver(u32 width, u32 height) {
    main_port.resolution.full_width = width;
    main_port.resolution.full_height = height;
    main_port.resolution.pane_width = width;
    main_port.resolution.pane_height = height;
    const char* separate = std::getenv("TLG_PRESENT_THREAD");
    separate_swap = !(separate && separate[0] == '0');
    if (separate_swap) {
        swap_thread = std::jthread([&](std::stop_token token) { SwapThread(token); });
    }
    present_thread = std::jthread([&](std::stop_token token) { PresentThread(token); });
}

void VideoOutDriver::RunPresenter(std::function<void()> work, bool if_idle) {
    if (!separate_swap) {
        work();
        return;
    }
    {
        std::scoped_lock lock{swap_mutex};
        if (if_idle && (swap_busy || !swap_queue.empty())) {
            return; // a redraw of the last frame is pointless while frames are queued
        }
        swap_queue.push_back(std::move(work));
    }
    swap_cv.notify_one();
}

void VideoOutDriver::SwapThread(std::stop_token token) {
    Common::SetCurrentThreadName("tlg:Present");
    while (true) {
        std::function<void()> work;
        {
            std::unique_lock lock{swap_mutex};
            swap_busy = false;
            FLIGHT_SCOPE("present: waiting for a request");
            if (!swap_cv.wait(lock, token, [this] { return !swap_queue.empty(); })) {
                return;
            }
            work = std::move(swap_queue.front());
            swap_queue.pop_front();
            swap_busy = true;
        }
        FLIGHT_SCOPE("present: running a request");
        work();
    }
}

VideoOutDriver::~VideoOutDriver() = default;

int VideoOutDriver::Open(const ServiceThreadParams* params) {
    if (main_port.is_open) {
        return ORBIS_VIDEO_OUT_ERROR_RESOURCE_BUSY;
    }
    main_port.is_open = true;
    return 1;
}

void VideoOutDriver::Close(s32 handle) {
    std::scoped_lock lock{mutex};

    // Mark as closed
    main_port.is_open = false;
    main_port.flip_rate = 0;
    Core::NativePresenter::Reset();

    // Clear port information
    std::memset(main_port.buffer_labels.data(), 0, sizeof(main_port.buffer_labels));
    for (u32 i = 0; i < MaxDisplayBuffers; ++i) {
        Core::NativeDisplay::Release(i);
    }
    Core::NativeDisplay::SetLabels(main_port.buffer_labels.data());
    std::memset(main_port.groups.data(), 0, sizeof(main_port.groups));
    std::memset(&main_port.vblank_status, 0, sizeof(main_port.vblank_status));
    main_port.flip_status = FlipStatus{};

    // Re-initialize buffers
    std::memset(main_port.buffer_slots.data(), 0, sizeof(main_port.buffer_slots));
    for (auto& buffer : main_port.buffer_slots) {
        buffer.group_index = -1;
    }

    // Clear events
    for (auto event : main_port.flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.flip_events.clear();
    for (auto event : main_port.vblank_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.vblank_events.clear();
}

VideoOutPort* VideoOutDriver::GetPort(int handle) {
    if (handle != 1) [[unlikely]] {
        return nullptr;
    }
    return &main_port;
}

int VideoOutDriver::RegisterBuffers(VideoOutPort* port, s32 startIndex, void* const* addresses,
                                    s32 bufferNum, const BufferAttribute* attribute) {
    const s32 group_index = port->FindFreeGroup();
    if (group_index >= MaxDisplayBufferGroups) {
        return ORBIS_VIDEO_OUT_ERROR_NO_EMPTY_SLOT;
    }

    if (startIndex + bufferNum > MaxDisplayBuffers || startIndex > MaxDisplayBuffers ||
        bufferNum > MaxDisplayBuffers) {
        LOG_ERROR(Lib_VideoOut,
                  "Attempted to register too many buffers startIndex = {}, bufferNum = {}",
                  startIndex, bufferNum);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    const s32 end_index = startIndex + bufferNum;
    if (bufferNum > 0 &&
        std::any_of(port->buffer_slots.begin() + startIndex, port->buffer_slots.begin() + end_index,
                    [](auto& buffer) { return buffer.group_index != -1; })) {
        return ORBIS_VIDEO_OUT_ERROR_SLOT_OCCUPIED;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "startIndex = {}, bufferNum = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             startIndex, bufferNum, GetPixelFormatString(attribute->pixel_format),
             attribute->aspect_ratio, static_cast<u32>(attribute->tiling_mode), attribute->width,
             attribute->height, attribute->pitch_in_pixel, attribute->option);

    auto& group = port->groups[group_index];
    std::memcpy(&group.attrib, attribute, sizeof(BufferAttribute));
    group.is_occupied = true;

    for (u32 i = 0; i < bufferNum; i++) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        port->buffer_slots[startIndex + i] = VideoOutBuffer{
            .group_index = group_index,
            .address_left = address,
            .address_right = 0,
        };

        // Reset flip label also when registering buffer
        port->buffer_labels[startIndex + i] = 0;
        Core::NativeDisplay::Release(startIndex + i);
        port->SignalVoLabel();

        g_presenter->RegisterVideoOutSurface(group, address);
        LOG_INFO(Lib_VideoOut, "buffers[{}] = {:#x}", i + startIndex, address);
    }

    return group_index;
}

int VideoOutDriver::UnregisterBuffers(VideoOutPort* port, s32 attributeIndex) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    auto& group = port->groups[attributeIndex];
    group.is_occupied = false;

    for (auto& buffer : port->buffer_slots) {
        if (buffer.group_index != attributeIndex) {
            continue;
        }
        buffer.group_index = -1;
    }

    return ORBIS_OK;
}

int VideoOutDriver::ChangeBufferAttribute(VideoOutPort* port, s32 attributeIndex,
                                          const BufferAttribute* attribute) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "attributeIndex = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             attributeIndex, GetPixelFormatString(attribute->pixel_format), attribute->aspect_ratio,
             static_cast<u32>(attribute->tiling_mode), attribute->width, attribute->height,
             attribute->pitch_in_pixel, attribute->option);

    std::unique_lock lock{port->port_mutex};
    std::memcpy(&port->groups[attributeIndex].attrib, attribute, sizeof(BufferAttribute));
    return 0;
}

void VideoOutDriver::Flip(const Request& req) {
    // Update HDR status before presenting, then present the frame (tlg: on the swap thread).
    TlgFreeze::flips_shown.fetch_add(1, std::memory_order_relaxed);
    RunPresenter([this, frame = req.frame, hdr = req.port->is_hdr] {
        g_presenter->SetHDR(hdr);
        g_presenter->Present(frame);
        TlgFreeze::presents.fetch_add(1, std::memory_order_relaxed);
    });
    // Frame rate line, same text start and same window as the normal renderer's ("Frame stats: N FPS, worst
    // frame M ms"), so one reader handles both logs. Only with TLG_FRAME_STATS=1; window TLG_FRAME_STATS_MS.
    Core::Vkb::ShaderStore::NoteFrame(); // TLG_SHADER_BUILD_STATS=1: the frame time and the shaders translated in it (no-op otherwise)
    static const bool frame_stats = EmulatorSettingsImpl::Flag("TLG_FRAME_STATS", false);
    if (frame_stats) {
        static const double window_s = [] {
            const char* env = std::getenv("TLG_FRAME_STATS_MS");
            const int ms = env ? std::atoi(env) : 0;
            return ms > 0 ? ms / 1000.0 : 5.0;
        }();
        using Clock = std::chrono::steady_clock;
        static Clock::time_point window_start = Clock::now(), last = window_start;
        static u32 frames;
        static double worst_ms;
        // Same "Stall:" and "Frame pacing:" lines as the normal renderer (text start and rules), so the bench
        // tool gets p99, slow frames and stalls for both renderers (issue 82). The new engine has no wait counters.
        static std::vector<double> intervals;
        static bool primed = false;
        static u32 stalls_shown = 0;
        const auto now = Clock::now();
        const double frame_ms = std::chrono::duration<double, std::milli>(now - last).count();
        worst_ms = std::max(worst_ms, frame_ms);
        last = now;
        ++frames;
        if (primed) {
            intervals.push_back(frame_ms);
            if (frame_ms > 40.0 && stalls_shown < 10) {
                ++stalls_shown;
                std::printf("Stall: %.1f ms frame; new engine\n", frame_ms);
            }
        }
        primed = true;
        const double window = std::chrono::duration<double>(now - window_start).count();
        if (window >= window_s) {
            std::printf("Frame stats: %.1f FPS, worst frame %.1f ms (vblank %u Hz); new engine\n", frames / window,
                        worst_ms, EmulatorSettings.GetVblankFrequency());
            if (intervals.size() > 2) {
                std::vector<double> sorted = intervals;
                std::sort(sorted.begin(), sorted.end());
                const double median = sorted[sorted.size() / 2];
                double sum = 0, sq = 0;
                u32 spikes = 0;
                for (const double ms : intervals) {
                    sum += ms;
                    sq += ms * ms;
                    spikes += ms > 1.5 * median;
                }
                const double mean = sum / intervals.size();
                std::printf("Frame pacing: median %.2f ms, stddev %.2f ms, p99 %.2f ms, %u frames over 1.5x median\n", median,
                            std::sqrt(std::max(0.0, sq / intervals.size() - mean * mean)),
                            sorted[std::min(sorted.size() - 1, sorted.size() * 99 / 100)], spikes);
            }
            intervals.clear();
            stalls_shown = 0;
            window_start = now;
            frames = 0;
            worst_ms = 0;
        }
    }
    // Update flip status.
    auto* port = req.port;
    {
        std::unique_lock lock{port->port_mutex};
        auto& flip_status = port->flip_status;
        flip_status.count++;
        flip_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
        flip_status.tsc = Libraries::Kernel::sceKernelReadTsc();
        flip_status.flip_arg = req.flip_arg;
        flip_status.current_buffer = req.index;
        if (req.eop) {
            --flip_status.gc_queue_num;
        }
        --flip_status.flip_pending_num;
    }

    // Trigger flip events for the port.
    for (auto event : port->flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->TriggerEvent(
                static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(OrbisVideoOutInternalEventId::Flip) |
                                        (req.flip_arg << 16)));
        }
    }

    // The buffer the screen had before is free again (E6.2d: the presenter object knows which one it was); its label goes back to 0, the console's interface
    if (const s32 prev = Core::NativePresenter::NoteShown(req.index); prev != -1) {
        port->buffer_labels[prev] = 0;
        Core::NativeDisplay::Release(static_cast<u32>(prev));
        port->SignalVoLabel();
    }
}

void VideoOutDriver::DrawBlankFrame() {
    RunPresenter([this] {
        const auto empty_frame = g_presenter->PrepareBlankFrame(true);
        g_presenter->Present(empty_frame, false, false);
    }, true);
}

void VideoOutDriver::DrawLastFrame() {
    RunPresenter([this] {
        const auto frame = g_presenter->PrepareLastFrame();
        if (frame != nullptr) {
            g_presenter->Present(frame, true);
        }
    }, true);
}

bool VideoOutDriver::SubmitFlip(VideoOutPort* port, s32 index, s64 flip_arg,
                                bool is_eop /*= false*/) {
    {
        std::unique_lock lock{port->port_mutex};
        if (index != -1 && port->flip_status.flip_pending_num > 16) {
            LOG_ERROR(Lib_VideoOut, "Flip queue is full");
            return false;
        }

        if (is_eop) {
            ++port->flip_status.gc_queue_num;
        }
        TlgFreeze::flips_submitted.fetch_add(1, std::memory_order_relaxed);
        ++port->flip_status.flip_pending_num; // integral GPU and CPU pending flips counter
        port->flip_status.submit_tsc = Libraries::Kernel::sceKernelReadTsc();
    }

    // No command processor thread: a flip from the CPU is prepared on the caller's thread too (the presenter serializes frame preparation)
    SubmitFlipInternal(port, index, flip_arg, is_eop);

    return true;
}

void VideoOutDriver::SubmitFlipInternal(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop) {
    TLG_PROF_LEVEL("present.submit_flip", 1);
    VkbVulkan::Frame* frame;
    if (index == -1) {
        frame = g_presenter->PrepareBlankFrame(false);
    } else {
        const auto& buffer = port->buffer_slots[index];
        ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
        const auto& group = port->groups[buffer.group_index];
        frame = g_presenter->PrepareFrame(group, buffer.address_left);
    }

    TLG_PROF("present.push_request");
    Core::NativePresenter::Push({.frame = frame, .port = port, .flip_arg = flip_arg, .index = index, .eop = is_eop});
}

void VideoOutDriver::PresentThread(std::stop_token token) {
    const std::chrono::nanoseconds vblank_period(1000000000 /
                                                 EmulatorSettings.GetVblankFrequency());

    Common::SetCurrentThreadName("shadPS4:PresentThread");
    Common::SetCurrentThreadRealtime(vblank_period);

    Common::AccurateTimer timer{vblank_period};

    // tlg: frame limit (see EmulatorSettings::GetFrameLimit). A request waits in the queue
    // until its slot; slots advance by one period (no drift) but never lag behind by more.
    // tlg: the in-game menu changes the limit live (TlgSettings::fps_limit); TLG_FPS_LIMIT
    // and the uncapped presets keep their fixed limit.
    // tlg: with the 480 Hz vblank (TLG_FPS=unlocked) a menu limit still applies; "Unlimited"
    // falls back to the preset's limit (the display refresh, at most 120).
    const bool fixed_limit = std::getenv("TLG_FPS_LIMIT") != nullptr;
    const bool fast_vblank = EmulatorSettings.GetVblankFrequency() == 480;
    const auto current_limit = [&] {
        const auto& s = TlgSettings::Get();
        if (fixed_limit) return EmulatorSettings.GetFrameLimit();
        // merge plan M6: the fork's Lua fixes set this lock (shim/tlg_lua.cpp): 30 while the game's slow motion runs (s06 tail catch, s10 falling
        // bridge; above 30 FPS the catch can be missed and the game waits forever) and the credits' rate. Their VideoOut obeys it first; so does ours.
        if (const u32 credits = s.credits_fps_lock) return credits;
        const u32 limit = u32(TlgGraphics::PresentLimit(s.fps_limit, s.applied_engine_fps));
        if (limit) return limit;
        if (fast_vblank) return EmulatorSettings.GetFrameLimit();
        // the launcher's 30 / 60 FPS modes run a 240 Hz vblank: the limit is the engine rate (as their VideoOut does)
        const u32 engine_fps = u32(s.applied_engine_fps.load());
        return EmulatorSettings.GetVblankFrequency() > engine_fps ? engine_fps : 0u;
    };
    // E6.2d: the queue of flips, the frame limit's slots and what is on screen belong to the presenter object (native/native_presenter.h); this thread is its timer.
    Core::NativePresenter::SetFrameLimit(current_limit(), std::chrono::steady_clock::now());
    std::printf("VideoOut: vblank %u Hz, frame limit %u FPS\n",
                EmulatorSettings.GetVblankFrequency(), Core::NativePresenter::FrameLimit());

    const auto receive_request = [] -> Request {
        Core::NativePresenter::Request taken;
        if (Core::NativePresenter::TryTake(taken)) {
            return {.frame = static_cast<VkbVulkan::Frame*>(taken.frame),
                    .port = static_cast<VideoOutPort*>(taken.port),
                    .flip_arg = taken.flip_arg,
                    .index = taken.index,
                    .eop = taken.eop};
        }
        return {};
    };

    // tlg: with a frame limit (uncapped presets) a queued flip is presented as soon as it
    // arrives and its slot allows, between vblanks, instead of on the next vblank tick
    // (Core::NativePresenter::ShowBetweenVblanks).

    while (!token.stop_requested()) {
        if (const u32 limit = current_limit(); limit != Core::NativePresenter::FrameLimit()) {
            Core::NativePresenter::SetFrameLimit(limit, std::chrono::steady_clock::now());
            std::printf("VideoOut: frame limit %u FPS\n", limit);
        }
        timer.Start();
        const auto tick_deadline = std::chrono::steady_clock::now() + vblank_period;

        if (DebugState.IsGuestThreadsPaused()) {
            DrawLastFrame();
            timer.End();
            continue;
        }

        // Check if it's time to take a request.
        auto& vblank_status = main_port.vblank_status;
        const auto now = std::chrono::steady_clock::now();
        const bool flip_slot = Core::NativePresenter::SlotOpen(now);
        if (flip_slot && vblank_status.count % (main_port.flip_rate + 1) == 0) {
            const auto request = receive_request();
            if (request) {
                Core::NativePresenter::NoteTaken(now);
            } else {
                Core::NativePresenter::NoteStarved();
            }
            if (!request) {
                if (timer.GetTotalWait().count() < 0) { // Dont draw too fast
                    if (!main_port.is_open) {
                        DrawBlankFrame();
                    } else if (VkbVulkan::Presenter::MustKeepDrawing()) {
                        DrawLastFrame();
                    }
                }
            } else {
                Flip(request);
                FRAME_END;
            }
        }

        {
            // Needs lock here as can be concurrently read by `sceVideoOutGetVblankStatus`
            std::scoped_lock lock{main_port.vo_mutex};

            // Trigger flip events for the port
            for (auto event : main_port.vblank_events) {
                auto equeue = Kernel::GetEqueue(event);
                if (equeue != nullptr) {
                    equeue->TriggerEvent(
                        static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                        Kernel::OrbisKernelEvent::Filter::VideoOut,
                        reinterpret_cast<void*>(
                            static_cast<u64>(OrbisVideoOutInternalEventId::Vblank) |
                            (vblank_status.count << 16)));
                }
            }

            // Update vblank status
            vblank_status.count++;
            TlgFreeze::vblanks.fetch_add(1, std::memory_order_relaxed);
            vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            main_port.vblank_cv.notify_all();
        }

        if (!Core::NativePresenter::ShowBetweenVblanks() || main_port.flip_rate != 0) {
            timer.End();
            continue;
        }
        while (!token.stop_requested()) {
            if (!Core::NativePresenter::WaitForRequest(tick_deadline)) {
                break; // next vblank
            }
            if (const auto next_flip = Core::NativePresenter::NextSlot(); std::chrono::steady_clock::now() < next_flip) {
                if (next_flip >= tick_deadline) {
                    break;
                }
                std::this_thread::sleep_until(next_flip);
            }
            const auto now = std::chrono::steady_clock::now();
            const auto request = receive_request();
            if (request) {
                Core::NativePresenter::NoteTaken(now);
                Flip(request);
                FRAME_END;
            }
        }
        std::this_thread::sleep_until(tick_deadline);
    }
}

} // namespace Libraries::VkbVideoOut
