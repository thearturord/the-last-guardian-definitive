// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <unordered_map>
#include <utility>

#include <boost/container/small_vector.hpp>

#include "native/shadps4/shader_recompiler/ir/basic_block.h"
#include "native/shadps4/shader_recompiler/ir/microinstruction.h"
#include "native/shadps4/shader_recompiler/ir/opcodes.h"

namespace VkbShader::Optimization {

static bool IsAllowedInst(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::Ballot ||
           inst.GetOpcode() == IR::Opcode::UnpackUint2x32 ||
           inst.GetOpcode() == IR::Opcode::BitCastU32F32 ||
           inst.GetOpcode() == IR::Opcode::BitCastF32U32;
}

static bool IsIdenticalInst(const IR::Inst* a, const IR::Inst* b) {
    if (a->GetOpcode() != b->GetOpcode()) {
        return false;
    }
    for (size_t i = 0; i < a->NumArgs(); i++) {
        if (a->Arg(i) != b->Arg(i)) {
            return false;
        }
    }
    return true;
}

struct CseKeyHash {
    size_t operator()(const std::pair<IR::Opcode, IR::Value>& key) const noexcept {
        return std::hash<IR::Value>{}(key.second) * 31 + static_cast<size_t>(key.first);
    }
};

static void RunLocalCSE(IR::Block* block) {
    boost::container::small_vector<IR::Inst*, 8> inst_list;
    // TLG_VKB_TRANSLATE_FAST: the same search through a hash of the opcode and the one argument the allowed opcodes have, when that
    // argument is an instruction (an immediate one goes through the list as before: two identical instructions have arguments of the same
    // kind, so they always meet on the same side). Neither side holds two identical instructions (a second one is replaced, not added), and
    // their arguments do not change while the block is walked (they come before them), so the one match the hash finds is the one the walk
    // through the list finds.
    std::unordered_map<std::pair<IR::Opcode, IR::Value>, IR::Inst*, CseKeyHash> by_key;
    const bool fast = IR::FastTranslate();
    auto it = std::ranges::find_if_not(*block, IR::IsPhi);
    for (; it != block->end(); ++it) {
        IR::Inst& inst{*it};
        if (!IsAllowedInst(inst)) {
            continue;
        }
        if (fast && inst.NumArgs() == 1 && !inst.Arg(0).IsImmediate()) {
            const auto [found, is_new] = by_key.try_emplace({inst.GetOpcode(), inst.Arg(0)}, &inst);
            if (!is_new) {
                inst.ReplaceUsesWithAndRemove(IR::Value{found->second});
            }
            continue;
        }
        const auto it = std::ranges::find_if(
            inst_list, [&](const IR::Inst* b) { return IsIdenticalInst(&inst, b); });
        if (it != inst_list.end()) {
            inst.ReplaceUsesWithAndRemove(IR::Value{*it});
        } else {
            inst_list.emplace_back(&inst);
        }
    }
}

} // namespace VkbShader::Optimization
