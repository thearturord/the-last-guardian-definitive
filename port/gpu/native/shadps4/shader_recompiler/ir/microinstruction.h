// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>
#include <cstring>
#include <type_traits>

#include <boost/container/list.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/intrusive/list.hpp>

#include "common/assert.h"
#include "native/shadps4/shader_recompiler/ir/opcodes.h"
#include "native/shadps4/shader_recompiler/ir/ssa.h"
#include "native/shadps4/shader_recompiler/ir/value.h"

namespace VkbShader::IR {

struct Use {
    Inst* user;
    u32 operand;

    Use() = default;
    Use(Inst* user_, u32 operand_) : user(user_), operand(operand_) {}
    Use(const Use&) = default;
    bool operator==(const Use&) const noexcept = default;
};

/// TLG_VKB_TRANSLATE_FAST (issue 160 step 2b, default on; 0 = as before): a use is removed from its list without a search, the dead code
/// pass keeps a flat set, the local common-subexpression step a hash. Every list keeps its order, so the translation is the same word for
/// word. Read once, the first time it is asked: a process translates in one way only.
[[nodiscard]] bool FastTranslate() noexcept;
/// Uses that TLG_VKB_TRANSLATE_FAST found not where their operand said, removed by the old search (never seen; an error line at the first):
/// the new engine's shader store report, translator_use_slot_misses.
[[nodiscard]] u64 UseSlotMisses() noexcept;

using UseList = boost::container::list<Use>;

class Inst : public boost::intrusive::list_base_hook<> {
public:
    explicit Inst(IR::Opcode op_, u32 flags_) noexcept;
    explicit Inst(const Inst& base);
    ~Inst();

    Inst& operator=(const Inst&) = delete;

    Inst& operator=(Inst&&) = delete;
    Inst(Inst&&) = delete;

    IR::Block* GetParent() const {
        ASSERT(parent);
        return parent;
    }
    void SetParent(IR::Block* block) {
        parent = block;
    }

    int UseCount() const noexcept {
        return uses.size();
    }

    bool HasUses() const noexcept {
        return uses.size() > 0;
    }

    IR::Opcode GetOpcode() const noexcept {
        return op;
    }

    size_t NumArgs() const {
        return op == IR::Opcode::Phi ? phi_args.size() : NumArgsOf(op);
    }

    Value Arg(size_t index) const noexcept {
        if (op == IR::Opcode::Phi) {
            return phi_args[index].second;
        } else {
            return args[index];
        }
    }

    void ReplaceUsesWithAndRemove(Value replacement) {
        ReplaceUsesWith(replacement, false);
    }

    void ReplaceUsesWith(Value replacement) {
        ReplaceUsesWith(replacement, true);
    }

    template <typename FlagsType>
        requires(sizeof(FlagsType) <= sizeof(u64) && std::is_trivially_copyable_v<FlagsType>)
    [[nodiscard]] FlagsType Flags() const noexcept {
        FlagsType ret;
        std::memcpy(reinterpret_cast<char*>(&ret), &flags, sizeof(ret));
        return ret;
    }

    template <typename FlagsType>
        requires(sizeof(FlagsType) <= sizeof(u64) && std::is_trivially_copyable_v<FlagsType>)
    void SetFlags(FlagsType value) noexcept {
        std::memcpy(&flags, &value, sizeof(value));
    }

    template <typename DefinitionType>
    void SetDefinition(DefinitionType def) {
        definition = std::bit_cast<u32>(def);
    }

    template <typename DefinitionType>
    [[nodiscard]] DefinitionType Definition() const noexcept {
        return std::bit_cast<DefinitionType>(definition);
    }

    const auto& Uses() const {
        return uses;
    }

    void SetRegTag(RegTag new_tag) {
        reg_tag = new_tag;
    }

    RegTag GetRegTag() const {
        return reg_tag;
    }

    bool MayHaveSideEffects() const noexcept;
    bool AreAllArgsImmediates() const;

    IR::Type Type() const;
    void ReplaceOpcode(IR::Opcode opcode);
    void SetArg(size_t index, Value value);

    Block* PhiBlock(size_t index) const;
    void AddPhiOperand(Block* predecessor, const Value& value);

    void Invalidate();
    void ClearArgs();

    u32 scc_index{};

private:
    struct NonTriviallyDummy {
        NonTriviallyDummy() noexcept {}
    };

    void Use(Inst* used, u32 operand);
    void UndoUse(Inst* used, u32 operand);
    /// TLG_VKB_TRANSLATE_FAST: where this instruction's use of its operand sits in the used instruction's list (filled by Use, emptied by
    /// UndoUse; an empty slot's iterator is never read: a list iterator made without a list holds no defined value)
    UseList::iterator& UseSlot(u32 operand) noexcept {
        return op == Opcode::Phi ? phi_args[operand].use : arg_uses[operand];
    }
    bool SlotFilled(u32 operand) const noexcept {
        return op == Opcode::Phi ? phi_args[operand].use_filled : (arg_uses_filled >> operand & 1u) != 0;
    }
    void FillSlot(u32 operand, bool filled) noexcept {
        if (op == Opcode::Phi) {
            phi_args[operand].use_filled = filled;
        } else {
            arg_uses_filled = static_cast<u8>(filled ? arg_uses_filled | 1u << operand : arg_uses_filled & ~(1u << operand));
        }
    }
    void ReplaceUsesWith(Value replacement, bool preserve);

    IR::Opcode op{};
    u32 definition{};
    u64 flags{};
    IR::Block* parent{};
    RegTag reg_tag{};
    u8 arg_uses_filled{}; ///< TLG_VKB_TRANSLATE_FAST: which of arg_uses are filled (SlotFilled)
    /// a phi's operand: its block, its value and where its use sits in the value's list (TLG_VKB_TRANSLATE_FAST)
    struct PhiArg {
        Block* first;
        Value second;
        UseList::iterator use;
        bool use_filled = false;
        PhiArg(Block* block, const Value& value) : first{block}, second{value} {}
    };
    union {
        NonTriviallyDummy dummy{};
        boost::container::small_vector<PhiArg, 2> phi_args;
        std::array<Value, 6> args;
    };
    std::array<UseList::iterator, 6> arg_uses; ///< TLG_VKB_TRANSLATE_FAST: UseSlot of the arguments (not a phi)

    UseList uses;
};
static_assert(sizeof(Inst) <= 240, "Inst size unintentionally increased (issue 160 step 2b: +56 for arg_uses and arg_uses_filled)");

[[nodiscard]] inline bool IsPhi(const Inst& inst) {
    return inst.GetOpcode() == Opcode::Phi;
}

} // namespace VkbShader::IR
