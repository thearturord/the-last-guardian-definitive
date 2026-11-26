// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/classes.h"
#include "native/shadps4/shader_recompiler/frontend/control_flow_graph.h"
#include "native/shadps4/shader_recompiler/frontend/decode.h"
#include "native/shadps4/shader_recompiler/frontend/structured_control_flow.h"
#include "native/shadps4/shader_recompiler/frontend/translate/translate.h"
#include "native/shadps4/shader_recompiler/ir/passes/ir_passes.h"
#include "native/shadps4/shader_recompiler/ir/post_order.h"
#include "native/shadps4/shader_recompiler/ir/program.h"
#include "native/shadps4/shader_recompiler/profile.h"
#include "native/shadps4/shader_recompiler/recompiler.h"

namespace VkbShader {

IR::BlockList GenerateBlocks(const IR::AbstractSyntaxList& syntax_list) {
    size_t num_syntax_blocks{};
    for (const auto& [_, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            ++num_syntax_blocks;
        }
    }
    IR::BlockList blocks{};
    blocks.reserve(num_syntax_blocks);
    for (const auto& [data, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            blocks.push_back(data.block);
        }
    }
    return blocks;
}

void EmitControlFlowGraph(IR::Program& program, Pools& pools, Gcn::CFG& cfg,
                          RuntimeInfo& runtime_info, const Profile& profile) {
    Gcn::Translator translator{program.info, runtime_info, profile};
    for (auto& block : cfg) {
        const u32 start = block.begin_index;
        const u32 size = block.end_index - start + 1;
        auto* ir_block = pools.block_pool.Create(pools.inst_pool);
        ir_block->cfg_block = &block;
        block.ir_block = ir_block;
        translator.Translate(ir_block, block.begin, block.cond,
                             std::span{program.ins_list}.subspan(start, size));
        program.blocks.push_back(ir_block);
    }
    translator.EmitPrologue(program.blocks.front());
    ASSERT_MSG(!program.info.translation_failed, "Shader translation has failed");
    for (auto& block : cfg) {
        auto* ir_block = block.ir_block;
        if (block.branch_true) {
            auto* true_block = block.branch_true->ir_block;
            ir_block->AddBranch(true_block);
        }
        if (block.branch_false) {
            auto* false_block = block.branch_false->ir_block;
            ir_block->AddBranch(false_block);
        }
    }
    program.post_order_blocks = VkbShader::IR::PostOrder(program.blocks.front());
}

IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools, Info& info,
                             RuntimeInfo& runtime_info, const Profile& profile) {
    // Ensure first instruction is expected.
    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    if (code[0] != token_mov_vcchi) {
        LOG_WARNING(Render_Recompiler, "First instruction is not s_mov_b32 vcc_hi, #imm");
    }

    Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Gcn::GcnDecodeContext decoder;

    // Decode and save instructions
    IR::Program program{info};
    program.ins_list.reserve(code.size());
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
    }

    // Clear any previous pooled data.
    pools.ReleaseContents();

    // Create control flow graph
    Common::ObjectPool<Gcn::Block> gcn_block_pool{64};
    Gcn::CFG cfg{gcn_block_pool, program.ins_list};
    EmitControlFlowGraph(program, pools, cfg, runtime_info, profile);

    // On NVIDIA GPUs HW interpolation of clip distance values seems broken, and we need to emulate
    // it with expensive discard in PS.
    VkbShader::InjectClipDistanceAttributes(program, runtime_info);

    // Run optimization passes on unstructured graph
    if (!profile.support_float64) {
        VkbShader::Optimization::LowerFp64ToFp32(program);
    }
    VkbShader::Optimization::SsaRewritePass(program);
    VkbShader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    VkbShader::Optimization::ReadLaneEliminationPass(program);
    if (info.sw_stage == SwStage::TessellationControl) {
        VkbShader::Optimization::TessellationPreprocess(program, runtime_info);
        VkbShader::Optimization::HullShaderTransform(program, runtime_info);
    } else if (info.sw_stage == SwStage::TessellationEval) {
        VkbShader::Optimization::TessellationPreprocess(program, runtime_info);
        VkbShader::Optimization::DomainShaderTransform(program, runtime_info);
    }
    VkbShader::Optimization::RingAccessElimination(program, runtime_info);
    VkbShader::IR::DumpProgram(program, info, "pre-res-discover.");
    auto resources = VkbShader::Optimization::ResourceDiscoverPass(program, profile);
    VkbShader::Optimization::FlattenExtendedUserdataPass(program);
    VkbShader::IR::DumpProgram(program, info, "pre-res-patch.");
    VkbShader::Optimization::ResourcePatchingPass(program.info, resources, profile);
    VkbShader::Optimization::LowerBufferFormatToRaw(program);
    VkbShader::Optimization::SharedMemorySimplifyPass(program, profile);
    VkbShader::Optimization::SharedMemoryToStoragePass(program, runtime_info, profile);
    VkbShader::Optimization::LowerUserClipPlanes(program, runtime_info);
    VkbShader::Optimization::PhiSimplificationPass(program);
    VkbShader::Optimization::InverseBallotEliminationPass(program);
    VkbShader::IR::DumpProgram(program, info, "pre-lower-phi.");

    // Prepare for structurization by clearing flow graph and lowering phis
    for (auto* ir_block : program.blocks) {
        ir_block->imm_predecessors.clear();
        ir_block->imm_successors.clear();
        ir_block->ssa_state.Reset();
    }
    VkbShader::Optimization::LowerPhisToRegsPass(program);

    // Structurize control flow graph and create program.
    program.syntax_list = VkbShader::Gcn::BuildASL(pools, cfg, info);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = VkbShader::IR::PostOrder(program.syntax_list.front().data.block);

    // Run optimization passes on structured graph
    VkbShader::Optimization::SsaRepairPass(program);
    VkbShader::Optimization::SsaRewritePass(program);
    VkbShader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);
    VkbShader::Optimization::DeadCodeEliminationPass(program);
    VkbShader::Optimization::LowerWave64BallotPass(program, runtime_info, profile);
    VkbShader::Optimization::LowerHardwareIntrinsics(program);
    VkbShader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    VkbShader::Optimization::DeadCodeEliminationPass(program);
    if (info.sw_stage == SwStage::Fragment && profile.needs_integer_interpolation_fix) {
        VkbShader::Optimization::InterpolatedIntegerPass(program);
    }
    VkbShader::Optimization::ReadsOutOfBranchPass(program, profile.reads_out_of_branch); // tlg issue 75
    VkbShader::Optimization::LeafAmbientOcclusionPass(program, profile); // tlg look option, off by default
    VkbShader::Optimization::CollectShaderInfoPass(program, profile);
    VkbShader::IR::DumpProgram(program, info);

    return program;
}

} // namespace VkbShader
