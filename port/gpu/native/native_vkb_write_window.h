// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <vector>
#include "common/types.h"

namespace VkbShader {
struct Info;
namespace IR {
struct Program;
}
} // namespace VkbShader

/// R5 (note 38): the range of a written buffer binding that one dispatch can write, proved from the shader's final IR, so that the hazard tracker sees that range and
/// not the whole binding (the write-after-write chains of async batch A, notes 32 and 37).
///
/// A binding gets a window when it is write-only (stores only: no load, atomic or constant read of it) and every store's dword index is
///   a_wg * WorkgroupId.x + a_lid * LocalInvocationId.x + k * cb[cb_binding][cb_index] + c   (a_wg, a_lid, k >= 0; one constant-buffer term at most)
/// (GlobalInvocationId.x counts as a_wg = local_x, a_lid = 1). When the index is a multiple of the global id (a_wg == a_lid * local_x) and every store sits under an
/// enclosing `if` whose condition implies `gid < cb[guard_binding][guard_index]` (or an immediate), the guard bounds the global ids too.
/// The backend evaluates the window at record time from the constant buffer the engine bound; anything not proved keeps the whole binding.
namespace Core::Vkb::WriteWindow {

struct Window {
    bool valid{};
    u64 a_wg{}, a_lid{}, k{};    ///< dword coefficients
    u32 cb_binding{~0u}, cb_index{}; ///< the constant-buffer term (cb_binding ~0u: none)
    u64 c_min{}, c_max{};        ///< the constant part over all stores, c_max including the last component of the widest store
    bool guarded{};              ///< every store is under gid < guard
    bool guard_immediate{};
    bool guard_signed{};
    u32 guard_binding{~0u}, guard_index{}, guard_value{};
    bool operator==(const Window&) const = default; ///< tlg: the shader store's check mode compares translations
};

/// One entry per buffer binding of `info` (same order); `local_x` is the work group size in x
std::vector<Window> Analyze(const VkbShader::IR::Program& program, const VkbShader::Info& info, u32 local_x);

/// The dword range [lo, hi) a dispatch of `groups_x` groups (0: unknown, indirect) can write through `w`; `read_cb(binding, index, out)` reads a constant-buffer dword of
/// this dispatch. False: no window (the whole binding). True with lo == hi: the dispatch writes nothing there.
template <typename ReadCb>
bool Evaluate(const Window& w, u32 local_x, u32 groups_x, ReadCb&& read_cb, u64& lo, u64& hi) {
    if (!w.valid) {
        return false;
    }
    u64 cbv = 0;
    if (w.cb_binding != ~0u && !read_cb(w.cb_binding, w.cb_index, cbv)) {
        return false;
    }
    // the largest global id that can store
    u64 max_threads = groups_x != 0 ? static_cast<u64>(groups_x) * local_x : ~0ull;
    if (w.guarded) {
        u64 bound = w.guard_value;
        if (!w.guard_immediate) {
            u32 v = 0;
            u64 v64 = 0;
            if (!read_cb(w.guard_binding, w.guard_index, v64)) {
                return false;
            }
            v = static_cast<u32>(v64);
            bound = v;
            if (w.guard_signed && static_cast<s32>(v) <= 0) {
                bound = 0;
            }
        }
        max_threads = std::min(max_threads, bound);
    }
    if (max_threads == ~0ull) {
        return false; // indirect and unguarded: unknown
    }
    if (max_threads == 0) {
        lo = hi = 0;
        return true;
    }
    u64 span = 0;
    if (w.a_wg == w.a_lid * local_x) {
        span = w.a_lid * (max_threads - 1);
    } else {
        if (groups_x == 0) {
            return false;
        }
        span = w.a_wg * (groups_x - 1) + w.a_lid * (local_x - 1);
    }
    lo = w.k * cbv + w.c_min;
    hi = w.k * cbv + w.c_max + span + 1;
    return true;
}

} // namespace Core::Vkb::WriteWindow
