// SPDX-License-Identifier: GPL-2.0-or-later
// The new engine's pipeline warm-up from the shipped shader list: the conversion and the hit/miss check (native_vkb_pipeline_warm.h).

#include "native/native_vkb_pipeline_warm.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <thread>
#include <type_traits>
#include <xxhash.h>

#include "common/logging/log.h"
#include "native/native_vkb_pipeline.h"
#include "native/native_vkb_shader_list.h"
#include "native/native_vkb_shader_store.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_instance.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_shader_util.h"
#include "native/vkb_core/liverpool_to_vk.h"
#include "native/vkb_core/vkb_device.h"
#include "tlg_overlay.h"
#include "video_core/renderer_vulkan/shader_list.h" // the normal renderer's list reader (read only)
#include "video_core/renderer_vulkan/shader_recipe.h" // LogPortableKey (the log line players' logs add keys from)

namespace Core::Vkb::PipelineWarm {
namespace {

using Key = VkbVulkan::Vkb::GraphicsPipelineKey;
using ZFormat = Core::NativeTarget::DepthTarget::ZFormat;
using StencilFormat = Core::NativeTarget::DepthTarget::StencilFormat;
namespace L2V = VkbVulkan::Vkb::LiverpoolToVK;

struct Converted {
    Key key;
    vk::Format depth_format = vk::Format::eUndefined;
    bool depth_variant = false; ///< made from a key without depth target
    s64 vs_entry = -1, ps_entry = -1;
};

/// The inputs a list entry's stage was stored with (the store's lookup without the program words)
struct WarmStage {
    u64 pgm = 0;
    VkbShader::RuntimeInfo runtime{};
    VkbShader::Backend::Bindings start{};
    u64 salt = 0;
};

/// One list entry's inputs in the new engine's form: a stage the game translates with these is that entry
struct EntryInputs {
    u64 perm = 0;
    VkbShader::RuntimeInfo runtime{};
    VkbShader::Backend::Bindings start{};
    u64 salt = 0;
};

/// A stage of an own key: the list entry (program hash, permutation); pgm 0 is no stage
struct ListStage {
    u64 pgm = 0, perm = 0;
};

struct State {
    std::mutex mutex;
    bool noted = false;
    std::vector<Converted> all;                  ///< every converted key, in the list's order (the warm-up builds them in this order)
    std::unordered_map<s64, WarmStage> stages;   ///< per list entry a key uses
    u64 stages_not_converted = 0;
    u64 list_keys = 0;
    std::map<std::string, u64> left;
    std::unordered_map<u64, std::vector<Converted>> by_stages; ///< vs ^ (ps * odd): the converted keys of one stage pair
    std::unordered_set<u64> vs_known, ps_known;                ///< the stages the list's keys use, each on its own
    u64 hits_depth_variant = 0;                                ///< hits on a key the list had without depth target
    u64 hits = 0, misses = 0, misses_early = 0, misses_no_pair = 0;
    std::map<std::string, u64> miss_fields; ///< per member: how often it differed in the closest key of a miss
    u32 logged = 0;
    std::unordered_map<u64, std::vector<EntryInputs>> inputs_of; ///< per program hash: its graphics list entries' inputs
    std::unordered_set<std::string> own_names;                    ///< own keys of the list or logged in this session
    u64 own_list = 0, own_logged = 0;
    std::map<std::string, u64> own_left; ///< own keys not read from the list or not logged, by reason
};

State& St() {
    static State s;
    return s;
}

void Say(const std::string& line) {
    std::printf("%s\n", line.c_str()); // the run log (out/logs): always on, so players' logs show which pipelines the list misses
    LOG_INFO(Render_Vulkan, "{}", line);
}

/// every 25 pipelines a running total (the run log gets no report at exit)
void Total(State& s) {
    const u64 n = s.hits + s.misses + s.misses_early;
    if (n % 25 != 0) {
        return;
    }
    std::string fields;
    for (const auto& [name, count] : s.miss_fields) {
        fields += fmt::format(" {}={}", name, count);
    }
    Say(fmt::format("vkb pipeline warm-up check: {} pipelines: {} hit ({} with the depth target added), {} miss ({} no key with the two stages, {} before the list was "
                    "read); differing:{}",
                    n, s.hits, s.hits_depth_variant, s.misses + s.misses_early, s.misses_no_pair, s.misses_early, fields.empty() ? " -" : fields));
}

u64 PairOf(u64 vs, u64 ps) {
    return vs ^ (ps * 0x9e3779b97f4a7c15ull);
}

Converted Convert(const ListKey& k) {
    Converted c;
    Key& key = c.key;
    key.stage_hashes[static_cast<u32>(VkbShader::SwStage::Vertex)] = k.vs;
    key.stage_hashes[static_cast<u32>(VkbShader::SwStage::Fragment)] = k.ps;
    key.num_color_attachments = k.num_color_attachments;
    key.mrt_mask = k.mrt_mask;
    key.num_samples = static_cast<u8>(k.num_samples);
    key.depth_samples = static_cast<u8>(k.depth_samples);
    key.z_format = static_cast<ZFormat>(k.z_format);
    key.stencil_format = static_cast<StencilFormat>(k.stencil_format);
    key.depth_clamp_enable = k.depth_clamp;
    key.depth_clip_enable = k.depth_clip;
    key.negative_one_to_one = k.clip_minus_w;
    key.provoking_first = k.provoking_last ? 0 : 1;
    key.prim_type = static_cast<Core::NativePrimitive::Kind>(k.prim_type);
    key.polygon_mode = k.polygon_mode == 0 ? vk::PolygonMode::ePoint : k.polygon_mode == 1 ? vk::PolygonMode::eLine : vk::PolygonMode::eFill;
    key.logic_op_enable = k.rop3 != static_cast<u32>(VkbAmdGpu::ColorControl::LogicOp::Copy) ? 1 : 0;
    key.logic_op = key.logic_op_enable ? L2V::LogicOp(static_cast<VkbAmdGpu::ColorControl::LogicOp>(k.rop3)) : vk::LogicOp::eCopy;
    for (u32 i = 0; i < 8; ++i) {
        if ((k.cb_shader_mask >> (i * 4) & 8) == 0) {
            key.alpha_not_exported |= static_cast<u8>(1u << i);
        }
    }
    for (u32 cb = 0; cb < 8; ++cb) {
        const auto& o = k.color[cb];
        auto& b = key.color_buffers[cb];
        b.data_format = static_cast<VkbAmdGpu::DataFormat>(o.data_format);
        b.num_format = static_cast<VkbAmdGpu::NumberFormat>(o.num_format);
        b.num_conversion = static_cast<VkbAmdGpu::NumberConversion>(o.num_conversion);
        b.export_format = static_cast<VkbAmdGpu::ShaderExportFormat>(o.export_format);
        for (u32 i = 0; i < 4; ++i) {
            b.swizzle.array[i] = static_cast<decltype(b.swizzle.array)::value_type>(o.swizzle[i]);
        }
        VkbAmdGpu::BlendControl bc;
        static_assert(sizeof(bc) == sizeof(o.blend));
        std::memcpy(&bc, &o.blend, sizeof(bc));
        if (bc.enable) {
            auto& t = key.blend[cb];
            t.src_color = L2V::BlendFactor(bc.color_src_factor);
            t.dst_color = L2V::BlendFactor(bc.color_dst_factor);
            t.color_op = L2V::BlendOp(bc.color_func);
            t.src_alpha = L2V::BlendFactor(bc.separate_alpha_blend ? bc.alpha_src_factor : bc.color_src_factor);
            t.dst_alpha = L2V::BlendFactor(bc.separate_alpha_blend ? bc.alpha_dst_factor : bc.color_dst_factor);
            t.alpha_op = L2V::BlendOp(bc.separate_alpha_blend ? bc.alpha_func : bc.color_func);
            t.enable = 1;
            b.blend_self_scale = (t.color_op == vk::BlendOp::eMin || t.color_op == vk::BlendOp::eMax) && t.src_color == vk::BlendFactor::eSrcColor &&
                                 t.dst_color == vk::BlendFactor::eDstColor;
        }
        key.write_masks[cb] = vk::ColorComponentFlags{o.write_mask};
        key.color_samples[cb] = static_cast<u8>(o.samples);
    }
    if (key.z_format != ZFormat::Invalid || key.stencil_format != StencilFormat::Invalid) {
        c.depth_format = L2V::DepthFormat(key.z_format, key.stencil_format);
    }
    if (c.depth_format != vk::Format::eD32SfloatS8Uint) {
        key.stencil_format = StencilFormat::Invalid; // our depth image has no stencil (Compile)
    }
    if (c.depth_format == vk::Format::eUndefined) {
        key.z_format = ZFormat::Invalid;
    }
    return c;
}

/// The members of two keys that differ
std::vector<std::string> Differ(const Key& a, vk::Format fa, const Key& b, vk::Format fb) {
    std::vector<std::string> d;
    const auto add = [&](bool differs, std::string name) {
        if (differs) {
            d.push_back(std::move(name));
        }
    };
    add(a.num_color_attachments != b.num_color_attachments, "num_color_attachments");
    add(a.mrt_mask != b.mrt_mask, "mrt_mask");
    add(a.num_samples != b.num_samples, "num_samples");
    add(a.depth_samples != b.depth_samples, "depth_samples");
    add(a.z_format != b.z_format, "z_format");
    add(a.stencil_format != b.stencil_format, "stencil_format");
    add(a.depth_clamp_enable != b.depth_clamp_enable, "depth_clamp");
    add(a.depth_clip_enable != b.depth_clip_enable, "depth_clip");
    add(a.negative_one_to_one != b.negative_one_to_one, "negative_one_to_one");
    add(a.provoking_first != b.provoking_first, "provoking_first");
    add(a.prim_type != b.prim_type, "prim_type");
    add(a.polygon_mode != b.polygon_mode, "polygon_mode");
    add(a.logic_op != b.logic_op || a.logic_op_enable != b.logic_op_enable, "logic_op");
    add(a.alpha_not_exported != b.alpha_not_exported, "alpha_not_exported");
    add(a.patch_control_points != b.patch_control_points, "patch_control_points");
    add(std::memcmp(a.vertex_buffer_formats.data(), b.vertex_buffer_formats.data(), sizeof(a.vertex_buffer_formats)) != 0, "vertex_buffer_formats");
    add(fa != fb, "depth_format");
    for (u32 cb = 0; cb < 8; ++cb) {
        const auto& x = a.color_buffers[cb];
        const auto& y = b.color_buffers[cb];
        add(x.data_format != y.data_format || x.num_format != y.num_format || x.num_conversion != y.num_conversion, fmt::format("color{}.format", cb));
        add(x.export_format != y.export_format, fmt::format("color{}.export", cb));
        add(!(x.swizzle == y.swizzle), fmt::format("color{}.swizzle", cb));
        add(x.blend_self_scale != y.blend_self_scale, fmt::format("color{}.self_scale", cb));
        add(std::memcmp(&a.blend[cb], &b.blend[cb], sizeof(a.blend[cb])) != 0, fmt::format("blend{}", cb));
        add(a.write_masks[cb] != b.write_masks[cb], fmt::format("write_mask{}", cb));
        add(a.color_samples[cb] != b.color_samples[cb], fmt::format("color_samples{}", cb));
    }
    if (d.empty() && !(a == b)) {
        d.push_back("padding");
    }
    return d;
}

/// The own key form (Vulkan::ShaderList::OtherKeyKind): LEB128 numbers, version and kind first as in the normal renderer's portable keys.
/// Raise the version when the fields below change; a key of another version is not read.
constexpr u64 OwnKeyVersion = 1;

/// Every member of the key but the stage hashes (the stages are list entries), as get/set pairs
template <typename F>
void KeyFields(Key& key, F&& f) {
#define TLG_OWN_FIELD(m) f([&] { return static_cast<u64>(m); }, [&](u64 v) { m = static_cast<std::remove_cvref_t<decltype(m)>>(v); })
    for (auto& format : key.vertex_buffer_formats) {
        TLG_OWN_FIELD(format);
    }
    TLG_OWN_FIELD(key.patch_control_points);
    TLG_OWN_FIELD(key.num_color_attachments);
    for (auto& b : key.color_buffers) {
        TLG_OWN_FIELD(b.data_format);
        TLG_OWN_FIELD(b.num_format);
        TLG_OWN_FIELD(b.num_conversion);
        TLG_OWN_FIELD(b.export_format);
        TLG_OWN_FIELD(b.blend_self_scale);
        for (auto& c : b.swizzle.array) {
            TLG_OWN_FIELD(c);
        }
    }
    for (auto& t : key.blend) {
        TLG_OWN_FIELD(t.src_color);
        TLG_OWN_FIELD(t.dst_color);
        TLG_OWN_FIELD(t.src_alpha);
        TLG_OWN_FIELD(t.dst_alpha);
        TLG_OWN_FIELD(t.color_op);
        TLG_OWN_FIELD(t.alpha_op);
        TLG_OWN_FIELD(t.enable);
    }
    for (auto& mask : key.write_masks) {
        f([&] { return static_cast<u64>(static_cast<VkColorComponentFlags>(mask)); },
          [&](u64 v) { mask = vk::ColorComponentFlags{static_cast<VkColorComponentFlags>(v)}; });
    }
    TLG_OWN_FIELD(key.logic_op);
    TLG_OWN_FIELD(key.polygon_mode);
    TLG_OWN_FIELD(key.alpha_not_exported);
    TLG_OWN_FIELD(key.logic_op_enable);
    TLG_OWN_FIELD(key.num_samples);
    TLG_OWN_FIELD(key.depth_samples);
    for (auto& n : key.color_samples) {
        TLG_OWN_FIELD(n);
    }
    TLG_OWN_FIELD(key.mrt_mask);
    TLG_OWN_FIELD(key.z_format);
    TLG_OWN_FIELD(key.stencil_format);
    TLG_OWN_FIELD(key.depth_clamp_enable);
    TLG_OWN_FIELD(key.prim_type);
    TLG_OWN_FIELD(key.negative_one_to_one);
    TLG_OWN_FIELD(key.provoking_first);
    TLG_OWN_FIELD(key.depth_clip_enable);
#undef TLG_OWN_FIELD
}

void Put(std::vector<u8>& out, u64 v) {
    for (;; v >>= 7) {
        if (v < 0x80) {
            out.push_back(static_cast<u8>(v));
            return;
        }
        out.push_back(static_cast<u8>(v | 0x80));
    }
}

struct In {
    std::span<const u8> bytes;
    size_t pos = 0;
    bool ok = true;
    u64 Get() {
        u64 v = 0;
        for (int shift = 0;; shift += 7) {
            if (!ok || pos >= bytes.size() || shift > 63) {
                ok = false;
                return 0;
            }
            const u8 byte = bytes[pos++];
            v |= static_cast<u64>(byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) {
                return v;
            }
        }
    }
};

std::vector<u8> EncodeOwn(const Key& key_in, vk::Format depth_format, const ListStage& vs, const ListStage& ps) {
    std::vector<u8> out;
    Put(out, OwnKeyVersion);
    Put(out, Vulkan::ShaderList::OtherKeyKind);
    Put(out, static_cast<u64>(depth_format));
    for (const u64 v : {vs.pgm, vs.perm, ps.pgm, ps.perm}) {
        Put(out, v);
    }
    Key key = key_in;
    KeyFields(key, [&](auto get, auto) { Put(out, get()); });
    return out;
}

/// False when the bytes are damaged, of another version or do not end with the last field
bool DecodeOwn(std::span<const u8> bytes, Key& key, vk::Format& depth_format, ListStage& vs, ListStage& ps) {
    In in{bytes};
    if (in.Get() != OwnKeyVersion || in.Get() != Vulkan::ShaderList::OtherKeyKind) {
        return false;
    }
    depth_format = static_cast<vk::Format>(in.Get());
    vs.pgm = in.Get();
    vs.perm = in.Get();
    ps.pgm = in.Get();
    ps.perm = in.Get();
    key = Key{};
    KeyFields(key, [&](auto, auto set) { set(in.Get()); });
    key.stage_hashes[static_cast<u32>(VkbShader::SwStage::Vertex)] = vs.pgm;
    key.stage_hashes[static_cast<u32>(VkbShader::SwStage::Fragment)] = ps.pgm;
    return in.ok && in.pos == bytes.size() && vs.pgm != 0;
}

std::string OwnName(std::span<const u8> bytes) {
    return fmt::format("v_0x{:016x}.vkey", XXH3_64bits(bytes.data(), bytes.size()));
}

/// A miss as an own key in the log (once per key and session), when both stages are list entries and it reads back the same
void LogOwn(State& s, const Key& key, vk::Format depth_format, const StageInputs& vs_in, const StageInputs& ps_in) {
    const auto entry = [&](u64 pgm, const StageInputs& in, ListStage& out) {
        if (pgm == 0) {
            return in.runtime == nullptr;
        }
        if (in.runtime == nullptr || in.start == nullptr) {
            return false;
        }
        const auto it = s.inputs_of.find(pgm);
        if (it == s.inputs_of.end()) {
            return false;
        }
        for (const auto& e : it->second) {
            if (e.salt == 0 && e.start == *in.start && std::memcmp(&e.runtime, in.runtime, sizeof(VkbShader::RuntimeInfo)) == 0) {
                out = ListStage{.pgm = pgm, .perm = e.perm};
                return true;
            }
        }
        return false;
    };
    const u32 v = static_cast<u32>(VkbShader::SwStage::Vertex);
    const u32 f = static_cast<u32>(VkbShader::SwStage::Fragment);
    for (u32 i = 0; i < key.stage_hashes.size(); ++i) {
        if (i != v && i != f && key.stage_hashes[i] != 0) {
            ++s.own_left["another stage kind"];
            return;
        }
    }
    ListStage vs, ps;
    if (key.stage_hashes[v] == 0 || !entry(key.stage_hashes[v], vs_in, vs) || !entry(key.stage_hashes[f], ps_in, ps)) {
        ++s.own_left["a stage that is no list entry"];
        return;
    }
    const auto bytes = EncodeOwn(key, depth_format, vs, ps);
    Key back;
    vk::Format depth_back{};
    ListStage vs_back, ps_back;
    if (!DecodeOwn(bytes, back, depth_back, vs_back, ps_back) || !(back == key) || depth_back != depth_format) {
        ++s.own_left["does not read back the same"];
        return;
    }
    auto name = OwnName(bytes);
    if (!s.own_names.insert(name).second) {
        return;
    }
    ++s.own_logged;
    Vulkan::ShaderRecipe::LogPortableKey(name, bytes);
}

} // namespace

void NoteList(const Vulkan::ShaderList::List& list) {
    std::map<std::string, u64> left;
    const auto keys = ReadListKeys(list, left);
    std::unordered_map<u64, std::vector<Converted>> by_stages;
    std::unordered_set<u64> vs_known, ps_known;
    std::vector<Converted> all;
    std::unordered_map<s64, WarmStage> stages;
    u64 stages_not_converted = 0;
    const auto note_stage = [&](s64 idx) {
        if (idx < 0 || stages.contains(idx)) {
            return;
        }
        std::string why;
        const auto st = Core::Vkb::ShaderList::Convert(list.entries[static_cast<size_t>(idx)], why);
        if (!st) {
            ++stages_not_converted;
            return;
        }
        stages.emplace(idx, WarmStage{.pgm = st->pgm_hash, .runtime = st->runtime, .start = st->start, .salt = st->salt});
    };
    // every graphics entry's inputs: a stage the game translates is named by its entry (own keys, LogOwn)
    std::unordered_map<u64, std::vector<EntryInputs>> inputs_of;
    std::map<std::pair<u64, u64>, s64> entry_of_id;
    for (size_t i = 0; i < list.entries.size(); ++i) {
        const auto& e = list.entries[i];
        entry_of_id.emplace(std::pair{e.pgm_hash, static_cast<u64>(e.perm_idx)}, static_cast<s64>(i));
        std::string why;
        const auto st = Core::Vkb::ShaderList::Convert(e, why);
        if (!st || st->kind != 0) {
            continue;
        }
        inputs_of[st->pgm_hash].push_back(EntryInputs{.perm = static_cast<u64>(e.perm_idx), .runtime = st->runtime, .start = st->start, .salt = st->salt});
    }
    for (const auto& k : keys) {
        auto& list_of_pair = by_stages[PairOf(k.vs, k.ps)];
        Converted c = Convert(k);
        c.vs_entry = k.vs_entry;
        c.ps_entry = k.ps_entry;
        note_stage(k.vs_entry);
        note_stage(k.ps_entry);
        list_of_pair.push_back(c);
        all.push_back(c);
        vs_known.insert(k.vs);
        ps_known.insert(k.ps);
        // the normal renderer leaves the depth target out when the draw does not use depth; the new engine keeps the bound depth view
        // (Compile): the same key with each depth format the engine's views have
        if (c.key.z_format == ZFormat::Invalid && c.key.stencil_format == StencilFormat::Invalid) {
            for (const auto& [z, stencil, format] : {std::tuple{ZFormat::Z32Float, StencilFormat::Invalid, vk::Format::eD32Sfloat},
                                                     std::tuple{ZFormat::Z32Float, StencilFormat::Stencil8, vk::Format::eD32SfloatS8Uint},
                                                     std::tuple{ZFormat::Z16, StencilFormat::Invalid, vk::Format::eD16Unorm}}) {
                Converted v = c;
                v.key.z_format = z;
                v.key.stencil_format = stencil;
                v.depth_format = format;
                v.depth_variant = true;
                list_of_pair.push_back(v);
                all.push_back(v);
            }
        }
    }
    // the list's own keys (this engine's key form, from players' logs): built as they are
    std::unordered_set<std::string> own_names;
    std::map<std::string, u64> own_left;
    u64 own_list = 0;
    for (const auto& bytes : list.other_keys) {
        Converted c;
        ListStage vs, ps;
        if (!DecodeOwn(bytes, c.key, c.depth_format, vs, ps)) {
            ++own_left["damaged or of another version"];
            continue;
        }
        const auto vs_it = entry_of_id.find({vs.pgm, vs.perm});
        const auto ps_it = ps.pgm == 0 ? entry_of_id.end() : entry_of_id.find({ps.pgm, ps.perm});
        if (vs_it == entry_of_id.end() || (ps.pgm != 0 && ps_it == entry_of_id.end())) {
            ++own_left["a stage the list's entries do not have"];
            continue;
        }
        c.vs_entry = vs_it->second;
        c.ps_entry = ps.pgm == 0 ? -1 : ps_it->second;
        note_stage(c.vs_entry);
        note_stage(c.ps_entry);
        by_stages[PairOf(vs.pgm, ps.pgm)].push_back(c);
        all.push_back(c);
        vs_known.insert(vs.pgm);
        ps_known.insert(ps.pgm);
        own_names.insert(OwnName(bytes));
        ++own_list;
    }
    auto& s = St();
    std::scoped_lock lock{s.mutex};
    s.inputs_of = std::move(inputs_of);
    s.own_names = std::move(own_names);
    s.own_left = std::move(own_left);
    s.own_list = own_list;
    s.all = std::move(all);
    s.stages = std::move(stages);
    s.stages_not_converted = stages_not_converted;
    s.list_keys = keys.size();
    s.left = std::move(left);
    s.by_stages = std::move(by_stages);
    s.vs_known = std::move(vs_known);
    s.ps_known = std::move(ps_known);
    s.noted = true;
    std::string left_text;
    for (const auto& [why, n] : s.left) {
        left_text += fmt::format("{}{} {}", left_text.empty() ? "" : ", ", n, why);
    }
    for (const auto& [why, n] : s.own_left) {
        left_text += fmt::format("{}{} own keys: {}", left_text.empty() ? "" : ", ", n, why);
    }
    Say(fmt::format("vkb pipeline warm-up: {} graphics keys of the list converted, {} own keys, {} stage pairs; left out: {}", s.list_keys, s.own_list,
                    s.by_stages.size(), left_text.empty() ? "none" : left_text));
}

void Check(const Key& key, u32 depth_format, const StageInputs& vs_in, const StageInputs& ps_in) {
    const auto fmt_in = static_cast<vk::Format>(depth_format);
    auto& s = St();
    std::scoped_lock lock{s.mutex};
    if (!s.noted) {
        ++s.misses_early;
        Total(s);
        return;
    }
    const u64 vs = key.stage_hashes[static_cast<u32>(VkbShader::SwStage::Vertex)];
    const u64 ps = key.stage_hashes[static_cast<u32>(VkbShader::SwStage::Fragment)];
    const auto it = s.by_stages.find(PairOf(vs, ps));
    if (it == s.by_stages.end()) {
        ++s.misses;
        ++s.misses_no_pair;
        if (s.logged < 200) {
            ++s.logged;
            Say(fmt::format("vkb pipeline warm-up check: miss vs {:016x} ps {:016x}: no key of the list has these two stages (vs in the list: {}, ps in the list: {})",
                            vs, ps, s.vs_known.contains(vs) ? "yes" : "no", s.ps_known.contains(ps) ? "yes" : "no"));
        }
        LogOwn(s, key, fmt_in, vs_in, ps_in);
        Total(s);
        return;
    }
    const std::vector<std::string>* best = nullptr;
    std::vector<std::string> closest;
    for (const auto& c : it->second) {
        if (c.key == key && c.depth_format == fmt_in) {
            ++s.hits;
            s.hits_depth_variant += c.depth_variant ? 1 : 0;
            Total(s);
            return;
        }
        auto d = Differ(key, fmt_in, c.key, c.depth_format);
        if (best == nullptr || d.size() < closest.size()) {
            closest = std::move(d);
            best = &closest;
        }
    }
    ++s.misses;
    for (const auto& name : closest) {
        ++s.miss_fields[name];
    }
    if (s.logged < 200) {
        ++s.logged;
        std::string names;
        for (const auto& name : closest) {
            names += names.empty() ? name : " " + name;
        }
        Say(fmt::format("vkb pipeline warm-up check: miss vs {:016x} ps {:016x} ({} keys with these stages), closest differs in: {}", vs, ps, it->second.size(),
                        names));
    }
    LogOwn(s, key, fmt_in, vs_in, ps_in);
    Total(s);
}

namespace {

std::atomic<bool> warm_started{false};
std::atomic<u64> w_total{0}, w_done{0}, w_built{0}, w_no_stage{0}, w_failed{0}, w_ms{0};

bool WarmOn() {
    const char* e = std::getenv("TLG_VKB_PIPELINE_WARM");
    return e == nullptr || e[0] != '0';
}

struct Prepared {
    ShaderStore::Translation t;
    vk::ShaderModule module{};
};

void Work() {
    using namespace std::chrono_literals;
    const auto t0 = std::chrono::steady_clock::now();
    // the device (the presenter makes it) and the store's folder listing come a little after the pre-build may finish
    for (int i = 0; i < 1200 && (!Core::Vkb::Device::Ready() || !ShaderStore::Ready()); ++i) {
        std::this_thread::sleep_for(100ms);
    }
    if (!Core::Vkb::Device::Ready() || !ShaderStore::Ready()) {
        Say("vkb pipeline warm-up: not done (the device or the shader store was not ready after 120 s)");
        return;
    }
    std::vector<Converted> all;
    std::unordered_map<s64, WarmStage> stages;
    {
        auto& s = St();
        std::scoped_lock lock{s.mutex};
        all = s.all;
        stages = s.stages;
    }
    const auto& instance = Core::Vkb::Device::Instance();
    const vk::Device device = instance.GetDevice();
    // each stage once: its stored translation and its module
    std::unordered_map<s64, Prepared> prepared;
    u64 needs_draw_stages = 0;
    for (const auto& [idx, st] : stages) {
        Prepared p;
        if (!ShaderStore::ForWarm(st.pgm, st.runtime, st.start, st.salt, p.t) || p.t.spv.empty()) {
            continue;
        }
        // the pipeline reads each image's descriptor from the draw's user data (NumBindings): the warm-up has no draw, so a zeroed
        // stand-in (one binding per image, as for every image without a dynamic mip index, which is left to the game below)
        p.t.info->flattened_ud_buf.assign(std::max<size_t>(p.t.info->srt_info.flattened_bufsize_dw, 4096), 0u);
        // an image with a dynamic mip index: its binding count comes from the draw's image descriptor, which the warm-up does not have
        bool needs_draw = false;
        for (const auto& image : p.t.info->images) {
            needs_draw |= image.mip_fallback_mode == VkbShader::MipStorageFallbackMode::DynamicIndex;
        }
        if (needs_draw) {
            ++needs_draw_stages;
            continue;
        }
        p.module = VkbVulkan::CompileSPV(p.t.spv, device);
        if (!p.module) {
            continue;
        }
        prepared.emplace(idx, std::move(p));
    }
    w_total.store(all.size(), std::memory_order_relaxed);
    Say(fmt::format("vkb pipeline warm-up: {} pipelines to make, {} of {} stages ready ({} need a draw's image)", all.size(), prepared.size(), stages.size(),
                    needs_draw_stages));
    TlgOverlay::SetShaderBuild(0, static_cast<u32>(std::max<size_t>(all.size(), 1)));
    const vk::PipelineCache cache{Cache()};
    std::atomic<size_t> next{0};
    const auto worker = [&] {
        for (size_t i = next.fetch_add(1); i < all.size(); i = next.fetch_add(1)) {
            const Converted& c = all[i];
            const auto vs = prepared.find(c.vs_entry);
            const auto ps = c.ps_entry < 0 ? prepared.end() : prepared.find(c.ps_entry);
            if (vs == prepared.end() || (c.ps_entry >= 0 && ps == prepared.end())) {
                w_no_stage.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::array<const VkbShader::Info*, VkbVulkan::Vkb::MaxShaderStages> infos{};
                std::array<VkbShader::RuntimeInfo, VkbVulkan::Vkb::MaxShaderStages> runtimes{};
                std::array<vk::ShaderModule, VkbVulkan::Vkb::MaxShaderStages> modules{};
                const u32 v = static_cast<u32>(VkbShader::SwStage::Vertex);
                const u32 f = static_cast<u32>(VkbShader::SwStage::Fragment);
                infos[v] = vs->second.t.info.get();
                runtimes[v] = vs->second.t.runtime;
                modules[v] = vs->second.module;
                if (ps != prepared.end()) {
                    infos[f] = ps->second.t.info.get();
                    runtimes[f] = ps->second.t.runtime;
                    modules[f] = ps->second.module;
                }
                VkbVulkan::Vkb::GraphicsPipeline::SerializationSupport sdata{};
                const VkbVulkan::Vkb::GraphicsPipeline pipeline{instance, Core::Vkb::Device::Scheduler(), Core::Vkb::Device::DescHeap(), Core::Vkb::Device::Profile(),
                                                               c.key, cache, std::span<const VkbShader::Info*, VkbVulkan::Vkb::MaxShaderStages>{infos},
                                                               std::span<const VkbShader::RuntimeInfo, VkbVulkan::Vkb::MaxShaderStages>{runtimes}, nullptr,
                                                               std::span<const vk::ShaderModule>{modules}, sdata, false, c.depth_format};
                (pipeline.Handle() ? w_built : w_failed).fetch_add(1, std::memory_order_relaxed);
            }
            const u64 done = w_done.fetch_add(1, std::memory_order_relaxed) + 1;
            TlgOverlay::SetShaderBuild(static_cast<u32>(done), static_cast<u32>(all.size()));
        }
    };
    // a few threads: the game runs next to it (the normal renderer's warm-up also leaves cores free)
    const u32 threads = std::clamp<u32>(std::thread::hardware_concurrency() / 4, 1, 6);
    std::vector<std::thread> pool;
    for (u32 i = 0; i < threads; ++i) {
        pool.emplace_back(worker);
    }
    for (auto& t : pool) {
        t.join();
    }
    for (auto& [idx, p] : prepared) {
        device.destroyShaderModule(p.module);
    }
    TlgOverlay::SetShaderBuild(0, 0);
    const auto ms = static_cast<u64>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
    w_ms.store(ms, std::memory_order_relaxed);
    Say(fmt::format("vkb pipeline warm-up: done in {} ms on {} threads: {} made, {} without a stored stage, {} failed", ms, threads, w_built.load(),
                    w_no_stage.load(), w_failed.load()));
}

} // namespace

void Start() {
    if (!WarmOn()) {
        Say("vkb pipeline warm-up: off (TLG_VKB_PIPELINE_WARM=0)");
        return;
    }
    if (!ShaderStore::OnDisk() || warm_started.exchange(true)) {
        return;
    }
    {
        auto& s = St();
        std::scoped_lock lock{s.mutex};
        if (!s.noted || s.all.empty()) {
            return;
        }
    }
    std::thread{Work}.detach();
}

VkPipelineCache Cache() {
    static std::once_flag once;
    static vk::PipelineCache cache{};
    std::call_once(once, [] {
        auto [result, made] = Core::Vkb::Device::Instance().GetDevice().createPipelineCache(vk::PipelineCacheCreateInfo{});
        if (result == vk::Result::eSuccess) {
            cache = made;
        }
    });
    return static_cast<VkPipelineCache>(cache);
}

nlohmann::json Report() {
    auto& s = St();
    std::scoped_lock lock{s.mutex};
    return nlohmann::json{{"list_read", s.noted},   {"list_keys", s.list_keys},         {"left_out", s.left},
                          {"hits", s.hits},         {"misses", s.misses},               {"misses_before_list", s.misses_early},
                          {"misses_no_stage_pair", s.misses_no_pair}, {"miss_fields", s.miss_fields},
                          {"own_keys", {{"in_list", s.own_list}, {"logged", s.own_logged}, {"left_out", s.own_left}}},
                          {"warm", {{"pipelines", w_total.load()}, {"done", w_done.load()}, {"made", w_built.load()}, {"no_stage", w_no_stage.load()},
                                    {"failed", w_failed.load()}, {"ms", w_ms.load()}, {"stages_not_converted", s.stages_not_converted}}}};
}

} // namespace Core::Vkb::PipelineWarm
