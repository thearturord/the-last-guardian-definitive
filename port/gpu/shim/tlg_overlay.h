// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: in-game settings menu (Dear ImGui), drawn by the presenter into the swapchain image
// after the game frame, at display resolution. Insert (keyboard) or R3+L2 (gamepad) opens it;
// while it is open the game gets no pad/keyboard input. Settings live in tlg_settings.h.

#pragma once

#include <string>
#include <vector>
#include "common/types.h"
#include "tlg_png.h"
#include "video_core/renderer_vulkan/vk_common.h"

union SDL_Event;
struct SDL_Window;

namespace Vulkan {
class Instance;
}

#include <mutex>
#include "tlg_perf_stats.h"

namespace TlgOverlay {

/// Present thread, once: the ImGui context and its Vulkan backend.
void Init(const Vulkan::Instance& instance, vk::Format format, u32 image_count);

/// VKB: the device as plain handles, for a presenter that has no Vulkan::Instance (the engine-to-Vulkan backend, TLG_VKB=1). The performance
/// items that read the device (GPU, VRAM) stay empty under it.
struct Device {
    u32 api_version{};
    vk::Instance instance{};
    vk::PhysicalDevice physical_device{};
    vk::Device device{};
    u32 queue_family{};
    vk::Queue queue{};
};
void Init(const Device& device, vk::Format format, u32 image_count);
/// VKB: the GPU name and video memory figures of the performance overlay under such a presenter
void SetStatsDevice(TlgPerfStats::Device device);
/// VKB: the lock every submission to the graphics queue holds, for the overlay's own font atlas uploads (ImGui submits and waits on the queue
/// itself). Default: their renderer's (Vulkan::Scheduler::submit_mutex); the engine-to-Vulkan backend's presenter sets its own.
void SetQueueMutex(std::mutex* mutex);

/// Window thread, for every SDL event: true when the menu consumed it.
bool HandleEvent(const SDL_Event& event);
/// Turns SDL text input on while the menu edits a value (window thread, once per poll).
void UpdateTextInput(SDL_Window* window);

/// Start-up loading screen (pipeline cache warm-up): `total` 0 shows no count.
void SetLoading(bool active, u32 done, u32 total);
/// Background cache warm-up: a small corner note while the game runs; `total` 0 hides it.
void SetBackgroundLoading(u32 done, u32 total);
/// Background shader build (shader_prebuild): the same corner note; `total` 0 hides it.
void SetShaderBuild(u32 done, u32 total);

/// Whether anything is drawn this frame (menu open or FPS counter on).
bool Visible();

/// Present thread: draws into `view` (layout ColorAttachmentOptimal).
void Render(vk::CommandBuffer cmdbuf, vk::ImageView view, vk::Extent2D extent);
/// Frame generation: the presenter marks the generated images it presents, so the counter shows real frames and frames on screen apart
void SetPresentingGenerated(bool generated);

/// A short message at the top of the screen for `seconds` (any thread), e.g. an input mode change.
void Notice(const std::string& text, float seconds);
/// Photo mode screenshot: Cross (photo mode) or Print Screen asks; the presenter takes the request
/// once and saves the next presented frame (render resolution, no overlay) as a PNG.
bool TakeScreenshotRequest();
/// The game's pause menu opened or closed (shows the quit hint).
void SetPauseMenu(bool open);
bool PauseMenuOpen();

/// The menu is open or a text entry is shown: the game's input is held neutral.
bool CapturesInput();

/// Pad thread: the buttons the game reads now (PS4 pad bits), for the photo mode buttons.
void SetGameButtons(u32 buttons);

/// Window thread: the game's text dialog (sceImeDialog). Shown as a box on screen until
/// `active` is false; typing goes to the window (sdl_window), the box only displays it.
void SetTextEntry(bool active, const std::string& prompt, const std::string& text);

/// The game's save dialogs (sceSaveDataDialog). kind: 1 save, 2 load, 3 delete (0: a save made
/// without a dialog). Rows carry the game's text for each save; the dates, sizes and every other
/// word are the port's, in the menu language (English, or the language pack).
struct SaveRow {
    std::string title, subtitle, detail;
    s64 mtime = 0;  // 0: unknown
    u64 bytes = 0;  // 0: unknown
    bool is_new = false;
};
/// The list: arrows/D-pad or the mouse choose, Enter/Cross or a click confirm, Esc/Circle or a
/// right click cancel. `pictures`: one per row (empty = none). PollSaveList: -1 while open,
/// -2 cancelled, else the chosen index.
void BeginSaveList(int kind, std::vector<SaveRow> rows, std::vector<TlgPng::Image> pictures, int focus);
int PollSaveList();
/// A message the player answers (see tlggpu_save_message_begin). PollSaveMessage: -1 while
/// open, 1 yes/OK, 2 no/cancel.
void BeginSaveMessage(int kind, int code, u64 value, const std::string& text, int buttons,
                      const SaveRow* entry, TlgPng::Image picture);
int PollSaveMessage();
/// The game closed its dialog first: the list or message goes away unanswered.
void CloseSaveDialog();
/// "Saving..." panel: active 1 on (percent < 0: no bar; text null keeps the last), 0 off,
/// 2 on for a moment (dialogs the port answers at once).
void SaveProgress(int kind, int active, int percent, const char* text);
/// "Game saved" in the bottom-right corner.
void SaveToast(int kind);

} // namespace TlgOverlay
