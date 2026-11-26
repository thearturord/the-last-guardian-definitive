// SPDX-License-Identifier: GPL-2.0-or-later
// tlg (look option, off by default): ambient light inside tree crowns (leaf ambient occlusion).
//
// The game lights its leaf cards in three steps: the sun (diffuse, specular and a back-light term, all multiplied by
// the sun shadow), then the ambient pass (light-probe grid irradiance plus the sky reflection). Nothing in the frame
// tells the ambient pass how deep a leaf sits inside its crown: the G-buffer has no occlusion channel and the game's
// screen-space occlusion does not reach this pass. So every leaf, also deep inside the crown, gets the full sky and
// bounce light, and the crowns look flat and over-bright. The PS4 picture is the same (the owner checked it).
//
// This pass changes the game's ambient pass only for pixels of the leaf material entries, and only when switched on:
// around the pixel it looks at 12 depth samples within a world radius; the share of samples that belong to geometry
// not far behind the pixel (the crown around it, as seen by the camera) takes ambient light away:
//   ambient *= 1 - strength * share
// Leaves at the edge of a crown (sky behind the samples) keep their light; leaves in the middle lose it. The sun term
// is untouched, so sunlit leaves stay bright. Grass, the boy and Trico have other material entries and do not change.
//
// Which shader is the ambient pass and which of its resources is the depth, the material and the camera is shader
// knowledge, kept as data in kRecipes below. The pass checks that the shader really reads those resources and that its
// only image write adds a value to what it read from the same image; if not, it changes nothing and counts the
// shader as a mismatch (LeafAoCounters).
//
// Live: strength and radius are read per dispatch from two push-data words compute shaders do not use (the viewport
// offsets, PushData::LeafAoStrengthIndex / LeafAoRadiusIndex, filled by the rasterizer from Extras > Realistic tree shade or
// TLG_LEAF_AO=<strength>[,<radius in metres>]). Strength 0 (off) gives every pixel radius 0 and factor 1: the picture
// is unchanged and the 12 reads hit the pixel's own cached depth. The material entries are part of the Profile
// (TLG_LEAF_AO=<s>,<r>,<entry>...; default 14); TLG_LEAF_AO=off leaves the shader untranslated.

#include <array>
#include <cmath>
#include <mutex>
#include <vector>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/resource.h"

namespace Shader::Optimization {

namespace {

/// The game's ambient pass and the roles of its resources (image numbers in the shader's own resource order, the
/// constant buffer read at camera_dword: the inverse view-projection, rows at +0..+18).
struct LeafAoRecipe {
    u64 hash;
    const char* name;
    u32 depth_image;    ///< depth target, .x
    u32 material_image; ///< G-buffer target 3, .w * 255 = material entry
    u32 camera_dword;   ///< first dword of the inverse view-projection in the shader's constant buffer
};
constexpr std::array<LeafAoRecipe, 1> kRecipes{{
    {0x775d0654ull, "CS_Deferred_Indirect SH_GRID=1 (ambient pass)", 1, 4, 68},
}};

LeafAoStats g_stats;
std::mutex g_mutex;
std::vector<LeafAoShader> g_shaders;

void Note(u64 hash, LeafAoResult result) {
    g_stats.results[static_cast<u32>(result)].fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{g_mutex};
    for (auto& s : g_shaders) {
        if (s.hash == hash) {
            s.result = result;
            return;
        }
    }
    g_shaders.push_back({hash, result});
}

/// The value read from `handle` by an ImageRead, through component picks and vectors rebuilt from them (the ambient
/// pass adds its light to what the target already holds).
bool ReadsFrom(const IR::Value& v, const IR::Value& handle) {
    IR::Value cur = v;
    for (int depth = 0; depth < 6; ++depth) {
        IR::Inst* inst = cur.TryInst();
        if (inst == nullptr) {
            return false;
        }
        const IR::Opcode op = inst->GetOpcode();
        if (op == IR::Opcode::ImageRead) {
            return inst->Arg(0).IsImmediate() && handle.IsImmediate() && (inst->Arg(0).U32() & 0xffff) == (handle.U32() & 0xffff);
        }
        if (op != IR::Opcode::CompositeExtractF32x4 && op != IR::Opcode::CompositeExtractF32x3 && op != IR::Opcode::CompositeExtractF32x2) {
            return false;
        }
        IR::Inst* source = inst->Arg(0).TryInst();
        if (source != nullptr && inst->Arg(1).IsImmediate() && source->GetOpcode() == IR::Opcode::CompositeShuffleF32x4) {
            // shuffle(a, b, i0..i3): component e is a[i] for i < 4, else b[i - 4]
            const u32 element = inst->Arg(1).U32();
            if (element >= 4 || !source->Arg(2 + element).IsImmediate()) {
                return false;
            }
            const u32 pick = source->Arg(2 + element).U32();
            IR::Inst* from = source->Arg(pick < 4 ? 0 : 1).TryInst();
            if (from == nullptr || pick >= 8) {
                return false;
            }
            if (from->GetOpcode() == IR::Opcode::ImageRead) {
                cur = source->Arg(pick < 4 ? 0 : 1);
                continue;
            }
            if (from->GetOpcode() != IR::Opcode::CompositeConstructF32x4 || (pick & 3) >= from->NumArgs()) {
                return false;
            }
            cur = from->Arg(pick & 3);
        } else if (source != nullptr && inst->Arg(1).IsImmediate() &&
            (source->GetOpcode() == IR::Opcode::CompositeConstructF32x4 || source->GetOpcode() == IR::Opcode::CompositeConstructF32x3 ||
             source->GetOpcode() == IR::Opcode::CompositeConstructF32x2)) {
            const u32 element = inst->Arg(1).U32();
            if (element >= source->NumArgs()) {
                return false;
            }
            cur = source->Arg(element); // the rebuilt vector's component
        } else {
            cur = inst->Arg(0);
        }
    }
    return false;
}

} // namespace

LeafAoStats& LeafAoCounters() {
    return g_stats;
}

std::vector<LeafAoShader> LeafAoShaders() {
    std::scoped_lock lock{g_mutex};
    return g_shaders;
}

void LeafAmbientOcclusionPass(IR::Program& program, const Profile& profile) {
    if (profile.leaf_ao_materials == 0 || program.info.sw_stage != SwStage::Compute) {
        return;
    }
    const u64 hash = program.info.pgm_hash;
    const LeafAoRecipe* recipe = nullptr;
    for (const auto& r : kRecipes) {
        if (r.hash == hash) {
            recipe = &r;
        }
    }
    if (recipe == nullptr) {
        return;
    }
    // what the shader itself does: its depth and material reads, its camera constants, its one image write
    IR::Inst* depth_read = nullptr;
    IR::Inst* material_read = nullptr;
    IR::Inst* camera_read = nullptr;
    IR::Inst* write = nullptr;
    IR::Block* write_block = nullptr;
    u32 writes = 0;
    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            const IR::Opcode op = inst.GetOpcode();
            if (op == IR::Opcode::ImageRead && inst.Arg(0).IsImmediate()) {
                const u32 image = inst.Arg(0).U32() & 0xffff;
                if (image == recipe->depth_image && depth_read == nullptr) {
                    depth_read = &inst;
                } else if (image == recipe->material_image && material_read == nullptr) {
                    material_read = &inst;
                }
            } else if (op == IR::Opcode::ReadConstBuffer && inst.Arg(1).IsImmediate() && inst.Arg(1).U32() == recipe->camera_dword && camera_read == nullptr) {
                camera_read = &inst;
            } else if (op == IR::Opcode::ImageWrite) {
                ++writes;
                write = &inst;
                write_block = block;
            }
        }
    }
    // the depth read's pixel is reused for every new read: it must come from the entry block (before any branch)
    if (depth_read == nullptr || material_read == nullptr || camera_read == nullptr || writes != 1 || program.blocks.empty() ||
        depth_read->GetParent() != program.blocks.front() || !material_read->Arg(2).IsEmpty() && !material_read->Arg(2).IsImmediate() ||
        !depth_read->Arg(2).IsEmpty() && !depth_read->Arg(2).IsImmediate()) {
        Note(hash, LeafAoResult::MismatchResources);
        return;
    }
    // the write: (target + light).xyz; the light operand of each of the three adds gets the factor
    IR::Inst* color = write->Arg(4).TryInst();
    // (the translator may put a component shuffle with a constant vector around the construct: look through one)
    if (color != nullptr && color->GetOpcode() == IR::Opcode::CompositeShuffleF32x4) {
        IR::Inst* inner = nullptr;
        u32 constructs = 0;
        for (u32 k = 0; k < 2; ++k) {
            IR::Inst* a = color->Arg(k).TryInst();
            if (a == nullptr || a->GetOpcode() != IR::Opcode::CompositeConstructF32x4) {
                continue;
            }
            bool all_constant = true;
            for (u32 e = 0; e < a->NumArgs(); ++e) {
                all_constant = all_constant && a->Arg(e).IsImmediate();
            }
            if (!all_constant) {
                inner = a;
                ++constructs;
            }
        }
        color = constructs == 1 ? inner : nullptr;
    }
    if (color == nullptr || color->GetOpcode() != IR::Opcode::CompositeConstructF32x4) {
        Note(hash, LeafAoResult::MismatchWrite);
        return;
    }
    std::array<IR::Inst*, 3> adds{};
    std::array<u32, 3> light_arg{};
    u32 found = 0;
    for (u32 c = 0; c < 4; ++c) {
        IR::Inst* add = color->Arg(c).TryInst();
        if (add == nullptr || add->GetOpcode() != IR::Opcode::FPAdd32) {
            continue;
        }
        u32 light = 2;
        if (ReadsFrom(add->Arg(0), write->Arg(0))) {
            light = 1;
        } else if (ReadsFrom(add->Arg(1), write->Arg(0))) {
            light = 0;
        }
        if (light == 2 || add->GetParent() != write_block || found == 3) {
            Note(hash, LeafAoResult::MismatchWrite);
            return;
        }
        adds[found] = add;
        light_arg[found++] = light;
    }
    if (found != 3) {
        Note(hash, LeafAoResult::MismatchWrite);
        return;
    }
    // the factor, computed right before the first add (all its inputs are known there)
    IR::Inst* first = nullptr;
    for (IR::Inst& inst : write_block->Instructions()) {
        if (&inst == adds[0] || &inst == adds[1] || &inst == adds[2]) {
            first = &inst;
            break;
        }
    }
    IR::IREmitter ir{*write_block, IR::Block::InstructionList::s_iterator_to(*first)};
    const IR::Value coords = depth_read->Arg(1); // the pixel of the shader's own depth read (defined at the top; the write builds its own later)
    const IR::U32 px = IR::U32{ir.CompositeExtract(coords, 0)};
    const IR::U32 py = IR::U32{ir.CompositeExtract(coords, 1)};
    const IR::Value size = ir.ImageQueryDimension(depth_read->Arg(0), ir.Imm32(0u), ir.Imm1(false), depth_read->Flags<IR::TextureInstInfo>());
    const IR::F32 width = ir.ConvertUToF(32, 32, ir.CompositeExtract(size, 0));
    const IR::F32 height = ir.ConvertUToF(32, 32, ir.CompositeExtract(size, 1));
    const IR::F32 xf = ir.ConvertUToF(32, 32, px);
    const IR::F32 yf = ir.ConvertUToF(32, 32, py);
    const auto cam = [&](u32 k) {
        return ir.BitCast<IR::F32>(ir.ReadConstBuffer(camera_read->Arg(0), ir.Imm32(recipe->camera_dword + k), camera_read->Flags<IR::BufferInstInfo>()));
    };
    const IR::F32 sx = cam(0), sy = cam(1), sz = cam(2), wx = cam(3), wy = cam(7), wd = cam(11), w0 = cam(15);
    const auto read = [&](IR::Inst* like, const IR::Value& at) {
        // the same image, level and sample choice as the shader's own read (both may be absent)
        const IR::U32 lod = like->Arg(2).IsEmpty() ? IR::U32{} : IR::U32{like->Arg(2)};
        const IR::U32 ms = like->Arg(3).IsEmpty() ? IR::U32{} : IR::U32{like->Arg(3)};
        return ir.ImageRead(like->Arg(0), at, lod, ms, like->Flags<IR::TextureInstInfo>());
    };
    // view depth of a pixel (centre at +0.5) with depth d: 1 / the inverse view-projection's w row
    const auto view_depth = [&](const IR::F32& x, const IR::F32& y, const IR::F32& d) {
        const IR::F32 nx = ir.FPSub(ir.FPMul(ir.FPAdd(x, ir.Imm32(0.5f)), ir.FPDiv(ir.Imm32(2.0f), width)), ir.Imm32(1.0f));
        const IR::F32 ny = ir.FPSub(ir.Imm32(1.0f), ir.FPMul(ir.FPAdd(y, ir.Imm32(0.5f)), ir.FPDiv(ir.Imm32(2.0f), height)));
        const IR::F32 w = ir.FPAdd(ir.FPAdd(ir.FPMul(wx, nx), ir.FPMul(wy, ny)), ir.FPAdd(ir.FPMul(wd, d), w0));
        return IR::F32{ir.FPAbs(ir.FPRecip(w))};
    };
    const IR::F32 depth = IR::F32{ir.CompositeExtract(read(depth_read, coords), 0)};
    const IR::F32 z = view_depth(xf, yf, depth);
    // leaf pixels only: the material entry of the pixel is one of the switch's entries
    const IR::F32 entry_f = IR::F32{ir.CompositeExtract(read(material_read, coords), 3)};
    const IR::U32 entry = IR::U32{ir.ConvertFToU(32, ir.FPAdd(ir.FPMul(entry_f, ir.Imm32(255.0f)), ir.Imm32(0.5f)))};
    IR::U1 leaf = ir.Imm1(false);
    for (u32 k = 0; k < 32; ++k) {
        if ((profile.leaf_ao_materials >> k) & 1u) {
            leaf = ir.LogicalOr(leaf, ir.IEqual(entry, ir.Imm32(k)));
        }
    }
    // live values (push data); strength 0 = off for every pixel
    const IR::F32 strength = ir.TlgPushF32(PushData::LeafAoStrengthIndex);
    const IR::F32 radius_m = ir.TlgPushF32(PushData::LeafAoRadiusIndex);
    leaf = ir.LogicalAnd(leaf, ir.FPGreaterThan(strength, ir.Imm32(0.0f)));
    // the radius in pixels: world units per screen half-width at depth z
    const IR::F32 scale = ir.FPSqrt(ir.FPAdd(ir.FPAdd(ir.FPMul(sx, sx), ir.FPMul(sy, sy)), ir.FPMul(sz, sz)));
    const IR::F32 radius_px = ir.FPMin(ir.FPMax(ir.FPMul(ir.FPDiv(radius_m, ir.FPMax(ir.FPMul(scale, z), ir.Imm32(1e-6f))),
                                                        ir.FPMul(width, ir.Imm32(0.5f))),
                                                 ir.Imm32(2.0f)),
                                        ir.Imm32(96.0f));
    // other pixels: radius 0, so their samples all read the pixel's own (cached) depth (second read: the reads cost little there)
    const IR::F32 radius_used = IR::F32{ir.Select(leaf, radius_px, ir.Imm32(0.0f))};
    // the pixel's own turn of the pattern (fixed per pixel, no flicker)
    const IR::F32 turn = ir.FPFract(ir.FPMul(ir.Imm32(52.9829189f), ir.FPFract(ir.FPAdd(ir.FPMul(xf, ir.Imm32(0.06711056f)), ir.FPMul(yf, ir.Imm32(0.00583715f))))));
    const IR::F32 far_x = ir.FPSub(width, ir.Imm32(1.0f)), far_y = ir.FPSub(height, ir.Imm32(1.0f));
    const IR::F32 reach = ir.FPAdd(z, radius_m);
    IR::F32 hits = ir.Imm32(0.0f);
    for (u32 i = 0; i < 12; ++i) {
        // two rings of six directions (half and full radius), the outer one turned by 30 degrees
        const float a = 6.2831853f * static_cast<float>(i % 6) / 6.0f + (i < 6 ? 0.0f : 0.5235988f);
        const float ring = i < 6 ? 0.5f : 1.0f;
        const IR::F32 r = ir.FPMul(radius_used, ir.FPAdd(ir.Imm32(ring * 0.75f), ir.FPMul(turn, ir.Imm32(ring * 0.5f))));
        const IR::F32 qx = ir.FPFloor(ir.FPMin(ir.FPMax(ir.FPAdd(xf, ir.FPMul(r, ir.Imm32(std::cos(a)))), ir.Imm32(0.0f)), far_x));
        const IR::F32 qy = ir.FPFloor(ir.FPMin(ir.FPMax(ir.FPAdd(yf, ir.FPMul(r, ir.Imm32(std::sin(a)))), ir.Imm32(0.0f)), far_y));
        const IR::Value at = ir.CompositeConstruct(ir.ConvertFToU(32, qx), ir.ConvertFToU(32, qy));
        const IR::F32 dn = IR::F32{ir.CompositeExtract(read(depth_read, at), 0)};
        const IR::F32 zn = view_depth(qx, qy, dn);
        hits = ir.FPAdd(hits, IR::F32{ir.Select(ir.FPLessThan(zn, reach), ir.Imm32(1.0f), ir.Imm32(0.0f))});
    }
    const IR::F32 factor = ir.FPSub(ir.Imm32(1.0f), ir.FPMul(hits, ir.FPMul(strength, ir.Imm32(1.0f / 12.0f))));
    const IR::F32 f = IR::F32{ir.Select(leaf, factor, ir.Imm32(1.0f))};
    for (u32 c = 0; c < 3; ++c) {
        IR::Inst* add = adds[c];
        IR::IREmitter at{*write_block, IR::Block::InstructionList::s_iterator_to(*add)};
        add->SetArg(light_arg[c], at.FPMul(IR::F32{add->Arg(light_arg[c])}, f));
    }
    Note(hash, LeafAoResult::Applied);
}

} // namespace Shader::Optimization
