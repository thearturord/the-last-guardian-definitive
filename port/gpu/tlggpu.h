/* C interface of the GPU library (gpu/): shadPS4's Liverpool/Vulkan video core,
 * GnmDriver, VideoOut and kernel event queues, adapted to the native loader. */
#ifndef TLGGPU_H
#define TLGGPU_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    const char *title;          /* window title */
    const char *serial;         /* CUSA id, names the pipeline cache */
    const char *user_dir;       /* pipeline cache/logs directory */
    uint32_t sdk_version;       /* from the eboot's procparam */
    uint32_t psf_attributes;    /* param.sfo ATTRIBUTE */
    int32_t width, height;      /* initial window size */
} TlgGpuConfig;
/* Registers kernel event queues (needed with or without graphics). */
void tlggpu_register_kernel(void);
/* Creates window, Vulkan device, presenter and GPU command processor. */
int tlggpu_init(const TlgGpuConfig *config);
/* Unless TLG_CACHE_FLUSH_ON_EXIT=0: waits at most two seconds for queued shader-cache writes
 * before window close or restart skips destructors (from SpartanPort, @maze1377). */
void tlggpu_flush_cache(void);
/* 1 after the player asked to quit (Ctrl+X, window close, controller combo; tlg_lua.cpp TlgQuit): the
 * game's own exit then ends in sceSystemServiceLoadExec, which closes instead of restarting. */
int tlggpu_quit_requested(void);
/* Function for an imported NID ("NID#lib#mod"), or 0 when the GPU library does not provide it. */
uintptr_t tlggpu_resolve(const char *scoped_nid);
/* Called first by the loader's SIGSEGV handler: 1 when a GPU page-tracking fault was handled. */
int tlggpu_handle_fault(void *ucontext, void *address);
/* Debug switch TLG_VKB_GPU_READ_TRAP (issue 83): asked first by the loader's fault handler. 1 when the fault was an access to
 * memory the graphics card writes (noted, then single-stepped) or the single step after it. Linux: the ucontext, SIGSEGV or
 * SIGTRAP and the fault address; Windows: the EXCEPTION_POINTERS (sig and address unused). 0 at once when the switch is off. */
int tlggpu_read_trap(void *info, int sig, void *address);
/* TLG_WRITE_LOG=1: prints the logged GPU-side writes to guest memory near the fault. */
void tlggpu_dump_guest_writes(void *ucontext);
/* Keyboard text entry through the game window (IME dialog). begin returns 0 when
 * no window exists; poll returns 0 typing, 1 confirmed, 2 cancelled (UTF-8 text). */
int tlggpu_text_input_begin(const char *initial_utf8, const char *prompt_utf8);
int tlggpu_text_input_poll(char *out_utf8, uint64_t size);
/* The save dialogs (sceSaveDataDialog), drawn over the game. kind: 1 save, 2 load, 3 delete
 * (the dialog's dispType); 0 for the toast of a save the game made without a dialog. */
typedef struct TlgSaveEntry {
    const char *title, *subtitle, *detail; /* the game's text for the save (UTF-8) */
    const char *icon;                      /* PNG path drawn beside it, or NULL */
    int64_t mtime;                         /* when it was saved (0: unknown) */
    uint64_t bytes;                        /* size on disk (0: unknown) */
    int is_new;                            /* the "New save" row */
} TlgSaveEntry;
/* The list. begin: 0 when no window exists. poll: -1 while open, -2 cancelled, else the index. */
int tlggpu_save_list_begin(int kind, const TlgSaveEntry *entries, int count, int focus);
int tlggpu_save_list_poll(void);
/* A message the player answers. code: a system message type (1 no data ... 14 too large), 100 the
 * game's own text, 101 an error code (value). buttons: 0 OK, 1 yes/no, 3 OK/cancel, -1 the
 * code's own. entry: the save it is about, or NULL. begin: 0 when no window exists.
 * poll: -1 while open, 1 yes/OK, 2 no/cancel. */
int tlggpu_save_message_begin(int kind, int code, uint64_t value, const char *text, int buttons,
                              const TlgSaveEntry *entry);
int tlggpu_save_message_poll(void);
/* The game closed its dialog: the list or message goes away unanswered. */
void tlggpu_save_dialog_close(void);
/* "Saving..." panel: active 1 shows it (percent < 0: no bar; text NULL keeps the last), 0 hides
 * it; active 2 shows it briefly, for dialogs the port answers at once. */
void tlggpu_save_progress(int kind, int active, int percent, const char *text);
/* "Game saved" in the corner (repeats within a few seconds are dropped). */
void tlggpu_save_toast(int kind);
/* Trophies (issue #60): the game's trophy file and npbind.dat (copied), the unlock file to keep,
 * and the game language (trophy text language). Unlock returns 1 for a new unlock (pop-up shown). */
void tlggpu_trophy_setup(const void *trp, uint64_t trp_size, const void *npbind, uint64_t npbind_size,
                         const char *store, int language);
int tlggpu_trophy_unlock(int id);
void tlggpu_trophy_flags(uint32_t flags[4]);
int tlggpu_trophy_next_locked(void);
int tlggpu_trophy_count(void);
/* 1 while the in-game settings menu is open: the game's pad input is held neutral. */
int tlggpu_overlay_captures_input(void);
/* The buttons the game reads now (PS4 pad bits: controller, keyboard and mouse after the
 * remaps), after every pad read. Photo mode takes its buttons from here. */
void tlggpu_pad_kind(int xbox);
void tlggpu_pad_game_buttons(uint32_t buttons);
/* Photo mode: the sticks as the game would get them (0..255). 1 = the photo camera takes the
 * pad (world running, sticks on the camera): the game gets centered sticks and no buttons. */
int tlggpu_photo_takes_pad(uint8_t lx, uint8_t ly, uint8_t rx, uint8_t ry);
/* A short message at the top of the screen (any thread). */
void tlggpu_overlay_notice(const char *text, float seconds);
/* The mouse, gathered by the window thread while the game has it (SDL relative mode: the window
 * has focus, the port's menu is closed and Esc did not release it; a click takes it again).
 * take: motion in mouse counts and wheel notches since the last call, then cleared. Buttons are
 * bits (1 << (SDL button - 1)): held now, and pressed since the last call (a short click). */
typedef struct {
    float dx, dy;
    uint32_t held, pressed;
    int32_t wheel_up, wheel_down, wheel_left, wheel_right;
    int32_t captured;
} TlgGpuMouse;
void tlggpu_mouse_take(TlgGpuMouse *mouse);
/* Mouse and keyboard settings (tlg.ini; the menu changes the mouse ones live). key_map: the
 * tlg.ini key_map value, the TLG_KEY_MAP syntax (runtime_pad.c), read once at start. */
typedef struct {
    int32_t mouse_enabled, invert_y;
    float sensitivity, smoothing, accel, deadzone;
    char key_map[1024];
} TlgGpuInputSettings;
void tlggpu_input_settings(TlgGpuInputSettings *settings);
/* Sound output (tlg.ini audio_device, audio_surround), read when an audio port opens. device: the
 * playback device name, empty = the system default. surround: always open 7.1, also when the device
 * reports stereo (Windows Sonic / Dolby Atmos for Headphones turn it into headphone surround). */
typedef struct {
    int32_t surround;
    char device[256];
} TlgGpuAudioSettings;
void tlggpu_audio_settings(TlgGpuAudioSettings *settings);
/* Freeze report (src/runtime_freeze.c): the GPU library marks its blocking calls (event queue,
 * GPU idle and vblank waits, GPU thread commands, readbacks, page faults) through these hooks. */
typedef void (*TlgGpuWaitBegin)(const char *kind, const void *object, const char *name, uintptr_t frame);
typedef void (*TlgGpuWaitEnd)(void);
void tlggpu_set_wait_hooks(TlgGpuWaitBegin begin, TlgGpuWaitEnd end);
/* The game's kernel semaphores (src/runtime_sema.c): the GPU library signals one as the game's own signal call does
 * (VKB: a backend wait that holds a game worker keeps the game's job scheduler awake, issue 78). */
typedef int32_t (*TlgGpuGameSemaSignal)(uint32_t id, int32_t count);
void tlggpu_set_game_sema_signal(TlgGpuGameSemaSignal signal);
/* The game executable's address range, so diagnostics (TLG_FAULT_PROFILE) print guest offsets. */
void tlggpu_set_guest_image(uintptr_t base, uint64_t size);
/* VKB: the engine-to-Vulkan backend's guest hooks (TLG_VKB=1; the hooks file of the game), once the eboot is loaded and patched. Nothing without TLG_VKB. */
void tlggpu_install_hooks(uintptr_t image_base, uint64_t image_size);
/* Flip requests from the game, flips handed to the presenter, presents done, vblank ticks,
 * page-protection faults being handled now and handled so far. */
typedef struct { uint64_t submitted, shown, presented, vblanks, faults_in_progress, faults_handled; } TlgGpuFrameCounters;
void tlggpu_frame_counters(TlgGpuFrameCounters *out);
/* Number of symbols registered by the vendored libraries (diagnostics). */
unsigned tlggpu_symbol_count(void);
#ifdef __cplusplus
}
#endif
#endif
