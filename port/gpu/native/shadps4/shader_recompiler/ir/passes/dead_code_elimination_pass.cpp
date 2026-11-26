// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdint>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>
#include "native/shadps4/shader_recompiler/ir/program.h"

namespace VkbShader::Optimization {
namespace {

/// TLG_VKB_TRANSLATE_FAST: the live instructions as a flat set (open addressing): no node per instruction. The pass only asks it
/// "insert" and "contains", so the set's kind cannot change what the pass removes.
class FlatInstSet {
public:
    std::pair<IR::Inst*, bool> insert(IR::Inst* inst) { // as std::unordered_set::insert: .second = it was not in the set
        if ((count + 1) * 2 > slots.size()) {
            Grow();
        }
        const bool placed = Place(slots, inst);
        count += placed;
        return {inst, placed};
    }
    bool contains(IR::Inst* inst) const {
        if (slots.empty()) {
            return false;
        }
        const size_t mask = slots.size() - 1;
        for (size_t i = Hash(inst) & mask;; i = (i + 1) & mask) {
            if (slots[i] == inst) {
                return true;
            }
            if (slots[i] == nullptr) {
                return false;
            }
        }
    }

private:
    static size_t Hash(const IR::Inst* inst) {
        return static_cast<size_t>((reinterpret_cast<std::uintptr_t>(inst) >> 3) * 0x9E3779B97F4A7C15ull >> 16);
    }
    static bool Place(std::vector<IR::Inst*>& table, IR::Inst* inst) {
        const size_t mask = table.size() - 1;
        for (size_t i = Hash(inst) & mask;; i = (i + 1) & mask) {
            if (table[i] == inst) {
                return false;
            }
            if (table[i] == nullptr) {
                table[i] = inst;
                return true;
            }
        }
    }
    void Grow() {
        std::vector<IR::Inst*> bigger(std::max<size_t>(256, slots.size() * 2), nullptr);
        for (IR::Inst* inst : slots) {
            if (inst != nullptr) {
                Place(bigger, inst);
            }
        }
        slots.swap(bigger);
    }

    std::vector<IR::Inst*> slots;
    size_t count = 0;
};

template <typename LiveSet>
void Run(IR::Program& program, LiveSet& live) {
    // We iterate over the instructions in reverse order.
    // This is because removing an instruction reduces the number of uses for earlier instructions.
    std::queue<IR::Inst*> worklist;

    for (IR::Block* block : program.post_order_blocks) {
        auto it{block->end()};
        while (it != block->begin()) {
            --it;
            const bool has_side_effects = it->MayHaveSideEffects();
            if (!it->HasUses() && !has_side_effects) {
                it->Invalidate();
                it = block->Instructions().erase(it);
            } else if (has_side_effects) {
                live.insert(&*it);
                worklist.push(&*it);
            }
        }
    }

    while (!worklist.empty()) {
        IR::Inst* inst = worklist.front();
        worklist.pop();

        for (size_t i = 0; i < inst->NumArgs(); i++) {
            if (inst->Arg(i).IsImmediate()) {
                continue;
            }
            IR::Inst* operand = inst->Arg(i).Inst();
            if (live.insert(operand).second) {
                worklist.push(operand);
            }
        }
    }

    for (IR::Block* block : program.post_order_blocks) {
        for (auto it = block->begin(); it != block->end();) {
            if (live.contains(&*it)) {
                ++it;
            } else {
                it->Invalidate();
                it = block->Instructions().erase(it);
            }
        }
    }
}

} // namespace

void DeadCodeEliminationPass(IR::Program& program) {
    if (IR::FastTranslate()) {
        FlatInstSet live;
        Run(program, live);
    } else {
        std::unordered_set<IR::Inst*> live;
        Run(program, live);
    }
}

} // namespace VkbShader::Optimization
