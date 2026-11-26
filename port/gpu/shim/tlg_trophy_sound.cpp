// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: the trophy unlock chime (issue #60). Our own sound, made here in code, not a recording:
// three quick rising bell notes (B5, E6, B6) with a soft echo, about 1.5 s. Played on its own SDL
// stream on the system's default sound device, at a fixed level (the system volume sets it).
#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "tlg_trophy.h"

namespace TlgTrophy {
namespace {

constexpr int Rate = 48000;
std::atomic<bool> playing{false};

// Interleaved stereo float samples, peak 1.0.
std::vector<float> MakeChime() {
    struct Note {
        float start, freq, pan, decay;
    };
    static const Note notes[] = {
        {0.00f, 987.77f, -0.25f, 0.30f},  // B5
        {0.07f, 1318.51f, 0.00f, 0.32f},  // E6
        {0.14f, 1975.53f, 0.25f, 0.55f},  // B6, rings longest
    };
    // Bell partials: the fundamental, then two inharmonic overtones that fade faster.
    static const float partial_ratio[] = {1.0f, 2.76f, 5.40f};
    static const float partial_gain[] = {1.0f, 0.18f, 0.06f};
    static const float partial_fade[] = {1.0f, 0.55f, 0.30f};
    constexpr float length = 1.6f, echo_delay = 0.09f, echo_gain = 0.25f;
    const int frames = int(length * Rate);
    std::vector<float> out(size_t(frames) * 2, 0.0f);
    constexpr float two_pi = 6.28318530718f;
    for (const Note& note : notes) {
        const float left = std::sqrt(0.5f * (1.0f - note.pan)), right = std::sqrt(0.5f * (1.0f + note.pan));
        for (int f = int(note.start * Rate); f < frames; ++f) {
            const float t = float(f) / Rate - note.start;
            const float attack = std::min(t / 0.004f, 1.0f);
            float v = 0.0f;
            for (int p = 0; p < 3; ++p) {
                v += partial_gain[p] * std::exp(-t / (note.decay * partial_fade[p])) *
                     std::sin(two_pi * note.freq * partial_ratio[p] * t);
            }
            v *= attack;
            out[size_t(f) * 2] += v * left;
            out[size_t(f) * 2 + 1] += v * right;
        }
    }
    // One echo, sides swapped, for a little room.
    const int delay = int(echo_delay * Rate);
    for (int f = frames - 1; f >= delay; --f) {
        out[size_t(f) * 2] += echo_gain * out[size_t(f - delay) * 2 + 1];
        out[size_t(f) * 2 + 1] += echo_gain * out[size_t(f - delay) * 2];
    }
    float peak = 0.0f;
    for (float v : out) peak = std::max(peak, std::fabs(v));
    // Fade the tail to zero so the stream ends without a click.
    const int tail = Rate / 5;
    for (int f = frames - tail; f < frames; ++f) {
        const float k = float(frames - f) / tail;
        out[size_t(f) * 2] *= k;
        out[size_t(f) * 2 + 1] *= k;
    }
    if (peak > 0.0f) {
        for (float& v : out) v /= peak;
    }
    return out;
}

} // namespace

void PlayChime() {
    // No sound when the game runs without sound (TLG_AUDIO=none), or while one plays.
    if (!SDL_WasInit(SDL_INIT_AUDIO) || playing.exchange(true)) {
        return;
    }
    std::thread([] {
        static const std::vector<float> chime = MakeChime();
        // Peak at -6 dB, so it sits under the game's own sound; the system volume sets the rest.
        constexpr float gain = 0.5f;
        std::vector<float> data(chime.size());
        for (size_t i = 0; i < chime.size(); ++i) data[i] = chime[i] * gain;
        const SDL_AudioSpec spec = {SDL_AUDIO_F32, 2, Rate};
        SDL_AudioStream* stream =
            SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
        if (stream) {
            SDL_PutAudioStreamData(stream, data.data(), int(data.size() * sizeof(float)));
            SDL_FlushAudioStream(stream);
            SDL_ResumeAudioStreamDevice(stream);
            for (int wait = 0; wait < 60 && SDL_GetAudioStreamQueued(stream) > 0; ++wait) {
                SDL_Delay(50);
            }
            SDL_Delay(150); // the device's own buffer
            SDL_DestroyAudioStream(stream);
        } else {
            SDL_Log("Trophies: no sound stream for the chime (%s)", SDL_GetError());
        }
        playing = false;
    }).detach();
}

} // namespace TlgTrophy
