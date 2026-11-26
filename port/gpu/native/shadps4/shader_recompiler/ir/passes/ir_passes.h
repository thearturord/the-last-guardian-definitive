// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <vector>
#include "native/shadps4/shader_recompiler/ir/basic_block.h"
#include "native/shadps4/shader_recompiler/ir/passes/resource_pass.h"
#include "native/shadps4/shader_recompiler/ir/program.h"

namespace VkbShader {
struct Profile;
void InjectClipDistanceAttributes(IR::Program& program, RuntimeInfo& runtime_info);
} // namespace VkbShader

namespace VkbShader::Optimization {

void SsaRewritePass(IR::Program& program);
void SsaRepairPass(IR::Program& program);
void PhiSimplificationPass(IR::Program& program);
void InverseBallotEliminationPass(IR::Program& program);
void LowerWave64BallotPass(IR::Program& program, const RuntimeInfo& runtime_info,
                           const Profile& profile);
void LowerHardwareIntrinsics(IR::Program& program);
void LowerPhisToRegsPass(IR::Program& program);
void DeadCodeEliminationPass(IR::Program& program);
/// tlg: exact integers from interpolated float varyings on GPUs without manual interpolation.
void InterpolatedIntegerPass(IR::Program& program);
void ConstantPropagationPass(IR::BlockList& program);
void FlattenExtendedUserdataPass(IR::Program& program);
void ReadLaneEliminationPass(IR::Program& program);
ResourceDiscoveryList ResourceDiscoverPass(IR::Program& program, const Profile& profile);
void ResourcePatchingPass(VkbShader::Info& info, const ResourceDiscoveryList& sharp_usages,
                          const Profile& profile);
void CollectShaderInfoPass(IR::Program& program, const Profile& profile);
void LowerBufferFormatToRaw(IR::Program& program);
void LowerUserClipPlanes(IR::Program& program, const RuntimeInfo& runtime_info);
void LowerFp64ToFp32(IR::Program& program);
void RingAccessElimination(const IR::Program& program, const RuntimeInfo& runtime_info);
void TessellationPreprocess(IR::Program& program, RuntimeInfo& runtime_info);
void HullShaderTransform(IR::Program& program, const RuntimeInfo& runtime_info);
void DomainShaderTransform(const IR::Program& program, const RuntimeInfo& runtime_info);
void SharedMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                             const Profile& profile);
void SharedMemorySimplifyPass(IR::Program& program, const Profile& profile);
void SharedMemoryToStoragePass(IR::Program& program, const RuntimeInfo& runtime_info,
                               const Profile& profile);

/// tlg issue 75: texture reads with automatic level choice out of per-pixel branches (pixel shaders; see the pass file).
/// Why a read stays inside its branch.
enum class ReadsOutOfBranchReason : u32 {
    ConstantRead,     ///< an operand is a constant read inside the branch at a per-pixel address, or the shader writes memory
    MemoryLoad,       ///< an operand comes from a buffer or shared memory load inside the branch
    TextureRead,      ///< an operand comes from another kind of texture read (fixed level, gather, fetch, size) inside the branch
    MergedValue,      ///< an operand is a value merged from two ways through a branch inside
    LaneExchange,     ///< an operand reads another pixel's value
    Loop,             ///< the read sits in a loop (or after a per-pixel loop exit)
    AfterEarlyReturn, ///< some pixels may have left the shader before the read
    WritesMemory,     ///< the shader writes memory: a read is not moved past a write
    Other,
};
constexpr u32 ReadsOutOfBranchReasonCount = 9;
constexpr std::array<const char*, ReadsOutOfBranchReasonCount> ReadsOutOfBranchReasonNames{
    "constant_read", "memory_load", "texture_read", "merged_value", "lane_exchange", "loop", "after_early_return", "writes_memory", "other"};
struct ReadsOutOfBranchStats {
    /// reads = moved + in_uniform_branch + outside_branches + the sum of not_moved
    std::atomic<u64> pixel_shaders{}, shaders_with_reads_in_branch{}, reads{}, moved{}, ops_moved{}, in_uniform_branch{}, outside_branches{};
    std::array<std::atomic<u64>, ReadsOutOfBranchReasonCount> not_moved{};
};
struct ReadsOutOfBranchShader {
    u64 hash{};
    bool applied{}; ///< moved (switch on) or only counted
    u32 reads{}, moved{}, ops_moved{}, in_uniform_branch{}, outside_branches{};
    std::array<u32, ReadsOutOfBranchReasonCount> not_moved{};
};
/// `move` false (switch off): only counts what it would move.
void ReadsOutOfBranchPass(IR::Program& program, bool move);
ReadsOutOfBranchStats& ReadsOutOfBranchCounters();
/// Every pixel shader with a read inside a per-pixel branch, by hash (last translation).
std::vector<ReadsOutOfBranchShader> ReadsOutOfBranchShaders();

/// tlg look option (Extras Realistic tree shade / TLG_LEAF_AO, off by default): ambient light inside tree crowns, see leaf_ambient_occlusion_pass.cpp
enum class LeafAoResult : u32 { Applied, MismatchResources, MismatchWrite };
constexpr u32 LeafAoResultCount = 3;
constexpr std::array<const char*, LeafAoResultCount> LeafAoResultNames{"applied", "mismatch_resources", "mismatch_write"};
struct LeafAoStats {
    std::array<std::atomic<u64>, LeafAoResultCount> results{}; ///< translations of the listed ambient shaders, by result
};
struct LeafAoShader {
    u64 hash{};
    LeafAoResult result{};
};
void LeafAmbientOcclusionPass(IR::Program& program, const Profile& profile);
LeafAoStats& LeafAoCounters();
/// Every listed ambient shader translated with the switch on, by hash (last translation).
std::vector<LeafAoShader> LeafAoShaders();

} // namespace VkbShader::Optimization
