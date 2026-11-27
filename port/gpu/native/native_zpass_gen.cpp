// SPDX-License-Identifier: GPL-2.0-or-later
// N6.3i: the cascade generator, see native_zpass_gen.h and documents/tlg/n6-pass-generator-design.md.
//
// N6.4 adds the instance group batches (views 0, 1, 2): FUN_01ee4e50 (the binding of the group and the upload of its data) is the engine's own function, called by the generator and
// announced to the update list by its hook; the draws are DrawIndexedIndirect through the draw hook; the clear after the batch is plain stores.
// What the engine's loop does for a pass without special batch entries (RenderDev_DrawZPass, ghidra 0x01c74a00), as native code:
//   prologue   ctx + 0x4480 = 4 (dirty bit ctx + 0x447c |= 1 when it changed), ctx + 0x68 = 0
//   per batch  ctx + 0x580, the index buffer view (ctx + 0x58, + 0x60 = 0x39), FUN_01c932d0 (ctx + 0x288 / 0x308 and the table at ctx + 0x6c0)
//   per prim   the material render object into the context (FUN_01c92b60, and its early copy of the resource slot), the resource set (FUN_01c92f30, the engine's own function, for the
//              materials that have one), the variant's shader pair with the engine's reference counting, the raster state object, the draw through the draw hook, the counters
//   epilogue   ctx + 0x980 = 0, ctx + 0x46c8 = the default raster state object
// The variant of a primitive (the engine searches its cache with a key built from the material, the mesh and the pass flags and stores the result behind the primitive record) is a
// function of fields of the snapshot (audit N6.3b, f). The generator learns key -> variant object from the passes the engine's loop runs (the first use of every key, the verify passes) and
// declines a pass that has a key it has not learned or that two variants answered for. Nothing the loop does is guessed: a pass the generator does not model is run by the loop.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "native/native_linux.h" // Linux side of the Windows calls (empty on Windows)
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "native/native_draw.h"
#include "native/native_log.h"
#include "native/native_zpass_gen.h"
#include "native/native_job_gen.h"
#include "native/native_site_gen.h"
#include "native/native_debug_server.h"
#include "native/native_vkb_graphics.h"
#include "native/native_vkb_stream.h"
#include "native/shadps4/core/guest_heap.h"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>

namespace Core::NativeZPassGen {
/// note 57: under the engine-to-Vulkan backend (TLG_VKB=1) the generators are off by default: the engine's own pass loops call the API, and our backend takes the
/// calls. Each switch's own =1 / =0 still wins.
inline bool GeneratorsDefaultOn() {
    static const bool on = [] {
        const char* vkb = std::getenv("TLG_VKB");
        return !(vkb != nullptr && vkb[0] == '1');
    }();
    return on;
}

namespace {

constexpr u64 ZPassBegin = 0x01c74a00, ZPassEnd = 0x01c76710; // DrawZPass and the next function (DrawGBufferPrimitives)
constexpr u64 GlobalDefaultKind = 0x069ae968;                 // DAT_069ae968: the material kind of the engine's default material
constexpr u64 GlobalContext = 0x069a1b40;                     // DAT_069a1b40: the render context object (its +0x38 holder has the default objects)
constexpr u64 FunctionBindResources = 0x01c92f30;             // FUN_01c92f30(renderDev + 0x2910, ctx, material, special)
constexpr size_t CtxBytes = 0x14000;
constexpr u32 DecalStageBytes = 0x1f0; // A7: the decal loop's constant block (renderDev + 0x3990 .. 0x3b80)
constexpr u32 OrMask = 0x80000000u; // arena entry size flag: the entry is an OR of bits (dirty bytes), compared as "these bits are set"

template <class T>
T R(u64 address) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(T));
    return value;
}
template <class T>
void W(u64 address, T value) {
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(T));
}
u64 Global(u64 ghidra_address) {
    return NativeDraw::ImageBase() + (ghidra_address - 0x1000000);
}

/// Cost census of what the generators call (`TLG_GEN_CENSUS=1`, /zpassgen key `engine_call_census`): calls and wall time of every engine function the generators still call, and of the
/// parts of the generator itself, per call kind. The time of a call includes everything it calls.
enum class CallId : u32 {
    ResourceSet,       // FUN_01c92f30
    GroupBind,         // FUN_01ee4e50
    ForwardPrologue,   // FUN_01c710f0
    FindVariant,       // FUN_01c822c0
    HandleLookup,      // FUN_01c59060 (the G-buffer prologue's)
    SpecialToken,      // the special entry's virtual +0x48
    HandleFree,        // FUN_01e45a60
    DrawIssue,         // the draw hook (tag, the engine's draw function)
    MaterialBlock,     // FUN_01c92b60 as plain stores (with the reader lock)
    ShaderPair,        // SetShader x2
    PassDepth,         // a whole generated depth pass
    PassGBuffer,       // a whole generated G-buffer pass
    PassForward,       // a whole generated forward pass
    ScanDepth,
    ScanGBuffer,
    ScanForward,
    PassDecal,
    ScanDecal,
    DecalUpdate, // the decal loop's UpdateSubresource (context virtual + 0xd0)
    PassSkin,
    ScanSkin,
    SkinUpdate,
    PassWet,
    ScanWet,
    WetUpdate,
    PassInst,
    ScanInst,
    Count
};
constexpr const char* CallNames[static_cast<u32>(CallId::Count)] = {"FUN_01c92f30 resource set", "FUN_01ee4e50 instance group bind", "FUN_01c710f0 forward prologue", "FUN_01c822c0 variant find", "FUN_01c59060 handle lookup (G-buffer prologue)",
                                                                      "special entry virtual +0x48", "FUN_01e45a60 handle free", "draw issue (hook, tag, engine draw)", "FUN_01c92b60 material block (plain stores)",
                                                                      "shader pair (2 x SetShader)", "generated depth pass (whole)", "generated G-buffer pass (whole)", "generated forward pass (whole)", "scan, depth", "scan, G-buffer", "scan, forward", "generated decal pass (whole)", "scan, decal", "decal constant buffer update (UpdateSubresource)", "generated skinning loop (whole)", "scan, skinning", "skinning constant buffer update (UpdateSubresource)", "generated wet loop (whole)", "scan, wet", "wet constant buffer update (UpdateSubresource)", "generated instancing loop (whole)", "scan, instancing"};
struct CallCensus {
    std::atomic<u64> calls{0}, ticks{0};
};
bool CensusOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_GEN_CENSUS");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return on;
}
std::array<CallCensus, static_cast<u32>(CallId::Count)>& Census() {
    static std::array<CallCensus, static_cast<u32>(CallId::Count)> census;
    return census;
}
u64 Ticks() {
#ifdef _WIN32
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    return static_cast<u64>(value.QuadPart);
#else
    return 0;
#endif
}
template <class F>
auto Timed(CallId id, F&& body) {
    if (!CensusOn()) {
        return body();
    }
    const u64 t0 = Ticks();
    struct Done {
        CallId id;
        u64 t0;
        ~Done() {
            auto& c = Census()[static_cast<u32>(id)];
            c.calls.fetch_add(1, std::memory_order_relaxed);
            c.ticks.fetch_add(Ticks() - t0, std::memory_order_relaxed);
        }
    } done{id, t0};
    return body();
}

struct Config {
    bool on{false};
    std::array<bool, 16> views{};
};
/// Parallel scene passes (2026-11-21, default under TLG_VKB): the shadow cascades (views 1 and 2) and the G-buffer passes are walked and drawn on worker threads
/// (parallel-passes-research.md). The generators run only for the passes the workers take: alone (on the engine's thread) they are slower than the engine's loop
/// (102.7 vs 110.5 fps). TLG_VKB_PAR_ZPASS=0 / TLG_VKB_PAR_GBUFFER=0 give a pass back to the engine's loop; the TLG_*_GEN switches still win when set.
bool VkbBackendOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB"); return e != nullptr && e[0] == '1'; }();
    return on;
}
bool ParZpassSwitch() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_PAR_ZPASS"); return e != nullptr ? e[0] == '1' : VkbBackendOn(); }();
    return on;
}
/// The G-buffer passes on the workers: default again under TLG_VKB since 2026-11-27 (off 2026-11-23..27 for tabnul issue 130, black floors and rooms: pieces the
/// engine's loop creates at first use were left out (PR 14: such a pass stays on the engine's thread, the learn request kept per pass list), and the heap mover's
/// copies landed behind the worker pass (98552398: into the oldest open placeholder)). TLG_VKB_PAR_GBUFFER=0 turns it off.
bool ParGbufferSwitch() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_PAR_GBUFFER"); return e != nullptr ? e[0] == '1' : VkbBackendOn(); }();
    return on;
}
const Config& Cfg() {
    static const Config config = [] {
        Config c;
        const char* e = std::getenv("TLG_ZPASS_GEN");
        // under TLG_VKB: on for the cascade workers (views 1 and 2); without TLG_VKB: default on since 2026-10-27 (the owner's play test); TLG_ZPASS_GEN=0 turns it off
        c.on = e != nullptr ? *e != '0' : (GeneratorsDefaultOn() || ParZpassSwitch());
        const char* v = std::getenv("TLG_ZPASS_GEN_VIEWS");
        std::string list = v != nullptr && *v != 0 ? v : (VkbBackendOn() ? "1,2" : "0,1,2,5,6,7");
        std::replace(list.begin(), list.end(), '+', ','); // (R7: '+' also separates, for env lists that split at commas)
        size_t at = 0;
        while (at < list.size()) {
            const size_t comma = list.find(',', at);
            const std::string item = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            const int view = std::atoi(item.c_str());
            if (view >= 0 && view < 16) {
                c.views[view] = true;
            }
            if (comma == std::string::npos) {
                break;
            }
            at = comma + 1;
        }
        return c;
    }();
    return config;
}

/// Everything about one batch that does not change inside a pass.
struct BatchLevel {
    u64 batch{};
    u64 value_580{}, view_58{}, pass_object_288{};
    bool bvar54{false};
    u32 table_count{0};
    std::array<u64, 16> table_entry{}; // the table entry object each value comes from
    std::array<std::pair<u32, u64>, 16> table{}; // (index j, value for ctx + 0x6c0 + 8 j), ascending j; the engine's loop fills them downwards, the values do not depend on each other
    u32 first_step{0}, step_count{0};
    // instance group batches (N6.4)
    u64 special{0};         // batch + 0x48, the special entry (class 0x26dbfc8: the vertex modifier entry)
    bool special_modifier{false}; // special + 0x40 != 0
    bool visibility{false}; // pass kind 3 with desc + 0x1d clear and batch flag 0x8: the primitive is skipped when the visibility cache says culled
    bool group{false};
    bool only_580{false};   // the loop writes ctx + 0x580 and nothing else for this batch (its group has no instances or no object)
    u64 inner{0};           // the mesh render data's inner object (first argument of FUN_01ee4e50)
    u64 group_object{0};
    u64 indirect_buffer{0}; // the argument buffer object of the group (group object + 0x110)
    // what FUN_01ee4e50 writes into the context (the verify run applies these; the generator calls the function)
    u64 g70{0}, g78{0}, g340{0};
    u32 g170{0}, g174{0};
    bool g340_set{false};
    u32 table_dwords{0};
    std::array<u64, 64> gtable{};
    bool main_any{false}, worker_any{false}; // P1: the batch has steps the engine's thread makes / the worker makes
};

struct Step {
    u32 index{};    // the primitive's index in its batch (the indirect arguments are addressed with it)
    u64 record{};   // the primitive record (0x38 bytes)
    u64 material{};
    u64 key{};
    u64 variant{};  // the variant object learned for the key
    u32 flag{};     // material + 0xd4 & 1
    u8 lane{0};     // P1: 0 the engine's thread draws it, 1 the worker, 2 the worker draws it and the engine's thread sets its state (the pass's last primitive: the context's end state)
};

struct Plan {
    bool scanned{false};
    std::vector<const char*> reasons;
    u32 view_type{0}, kind{0};
    u64 render_device{0}, ctx{0};
    u32 flag14{0}, flag1c{0};
    s32 indirect_mode{0};
    u64 visibility_owner{0}; // desc + 0x28: the object whose +0x790 is the id the cache is keyed with
    std::vector<BatchLevel> batches;
    std::vector<Step> steps;
    u32 group_batches{0}, group_batches_skipped{0};
    std::vector<u8> culled; // P1: per step, the visibility cache's answer read on the engine's thread when the pass was handed over
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* reason) {
        for (const char* r : reasons) {
            if (r == reason) {
                return;
            }
        }
        reasons.push_back(reason);
    }
};

struct TableEntry {
    u64 variant{0};
    bool ambiguous{false};
};

struct ViewStats {
    u64 passes{}, generated{}, generated_draws{}, generated_batches{}, eligible{};
    u64 loop_passes{}, verify_passes{}, verify_clean{}, verify_mismatched{}, verify_draws{};
    u64 not_scanned{};
    bool disabled{false};
    std::map<std::string, u64> declines;
    std::map<u32, u64> kinds; // passes by desc + 0x10
    u64 group_batches{}, group_batches_skipped{}, verify_raced{};
};

struct State {
    std::mutex mutex;
    std::map<u32, ViewStats> views;
    std::unordered_map<u64, TableEntry> table;
    std::vector<u64> cache_seen;
    u64 table_resets{0}, learned{0}, conflicts{0}, learn_not_in_cache{0}, learn_faults{0}, scan_faults{0};
    std::map<std::string, u64> mismatch_kinds;
    std::map<std::string, u64> special_classes;
    std::map<std::string, u64> raced_words; // verify: words where the live value differed from the shadow read but equals what the object holds when compared
    std::vector<std::string> samples;
};
State& S() {
    static State state;
    return state;
}

// ------------------------------------------------------------------------------------------------------------------------------- the target of the generator

/// B4: the instancing scene block (the holder's `+ 0x10`, 0x5b0 bytes copied from the scene) of the pass being verified, and its source: the verify run re-reads the scene when a word differs
thread_local u64 g_inst_scene_block{0}, g_inst_scene_source{0};

struct Mark {
    size_t arena_end;
    u32 kind;
    u64 count, first, a3;
    u64 vertex_shader, pixel_shader;
    // A3 (FUN_01c92f30 natively): what the resource set before this draw left in the device (the matrices block and the constant buffer it fills), for the verify run
    bool rs_valid{false};
    u32 rs_n{0}, rs_size{0};
    std::array<u8, 112> rs_bytes{};
    // A3 (FUN_01ee4e50 natively): the uploader buffer the group bind filled: the node's buffer and the staged data it was filled from
    u64 gb_buffer{0}, gb_src{0};
    u32 gb_n{0};
    // A7: the decal loop's constant block (renderDev + 0x3990) at this draw, index into Target::stages
    s32 stage{-1};
    // B3: the constant blocks (live address, bytes) the skinning loop updated buffers from before this dispatch
    std::vector<std::pair<u64, std::vector<u8>>> blocks;
    // B3: dword 3 of the block at flag_block is a flag computed from the float at flag_source, which the engine's main thread changes: the verify run re-reads it
    u64 flag_source{0}, flag_block{0};
    // B5: the block of a wet force dispatch (system + 0x130): its colour lanes (bytes 0x40 .. 0x4f) come from system + 0x254 .. 0x258, which the game's thread writes
    u64 color_block{0};
    // B5: the block of a wet time step (system + 0x80): its second dword is the frame time (context global + 0x19c), which the main thread updates
    u64 time_block{0}, time_source{0};
    // B5: which wet block this dispatch consumed and what it is made from (the verify run builds it again when it compares): 1 step (system + 0x1c8, the frame time), 2 intersect, 3 force
    u32 wet_kind{0};
    u64 wet_system{0}, wet_object{0}, wet_water{0}, wet_block{0}, wet_context_global{0};
    // the call site (ghidra return address) the generator gives the dispatch; the verify run compares it with the loop's own (0: not set)
    u64 site{0};
};

/// Where the generator writes: the live device context, or a private copy of it that logs every write (the verify run).
struct Target {
    bool live{true};
    bool worker{false}; // P1: the worker's run (a private copy of the context; lanes 1 and 2; no engine-visible side effects but its draws)
    bool split{false};  // P1: the engine's thread's run of a handed-over pass (lanes 0 and 2's state)
    u32 refused{0};     // P1: worker draws not drawn (fetch shader missing, a binding it cannot make)
    bool step_failed{false}; // P1b: the worker could not make this primitive's binding (the draw is left out)
    u8* ring{nullptr};       // P1b: the worker's block of the frame ring (taken by the engine's thread at the hand-over), for the matrices and instance data
    u64 arena_size{0}, arena_used{0}, arena_need{0};
    u8* rs_clone{nullptr};    // P1b: the worker's own copy of the resource set's constant buffer object (renamed per primitive, as the engine renames its own)
    u8* group_clone{nullptr}; // P1b: the same for the instance group's buffer
    u8* rs_stage{nullptr};    // P1b: the worker's copy of the block at renderDev + 0x3790 (the matrices go in, the whole block goes to the ring)
    u32 rs_fail{0}, group_fail{0}, unlearned{0};
    const std::unordered_map<u64, std::pair<u64, u64>>* moved_check{nullptr}; // P2 census: per batch, the constants' address and the vertex view's memory at the hand-over
    u32 moved_cb{0}, moved_vb{0}, moved_checked{0};
    u8* base{nullptr};
    std::vector<u8> arena; // shadow: entries [u32 offset][u32 size][bytes]
    std::vector<Mark> marks;
    u64 batch_owner{0};                          // shadow: the batch whose level words (ctx + 0x580, 0x58, 0x288, 0x308) are being written
    std::unordered_map<size_t, u64> batch_owner_map;
    u64 indexed_site{0x01c76433}, indirect_site{0x01c7640f}, indexed_a4{1}; // the loop's call sites (ghidra return addresses) and the fifth argument of its indexed draw
    u64 table_entry{0};                          // shadow: the table entry object whose value is being written (ctx + 0x6c0 + 8 j)
    std::unordered_map<size_t, u64> table_owner; // shadow: arena position of an entry -> that table entry object
    u64 direct_object{0};                        // shadow: the engine object whose word (at direct_offset) the value being written was read from
    u32 direct_offset{0};
    std::unordered_map<size_t, std::pair<u64, u32>> direct_owner; // shadow: arena position of an entry -> (object, offset) it was read from
    u64 slot_material{0};                       // shadow: the material whose resource slot words are being written
    std::unordered_map<size_t, u64> slot_owner; // shadow: arena position of an entry -> that material
    s32 draws{0};
    s64 triangles{0};
    std::vector<std::array<u8, DecalStageBytes>> stages; // shadow (A7): the decal loop's constant block as the generator built it, one per updated primitive
    s32 stage_pending{-1};
    std::vector<std::pair<u64, std::vector<u8>>> blocks; // shadow (B3): blocks staged since the last dispatch
    bool rs_pending{false}; // shadow: the native resource set ran since the last draw
    u32 rs_n{0}, rs_size{0};
    std::array<u8, 112> rs_bytes{};
    u64 gb_buffer{0}, gb_src{0}; // shadow: the native group bind ran since the last draw
    u32 gb_n{0};

    template <class T>
    T Get(u32 offset) const {
        T value;
        std::memcpy(&value, base + offset, sizeof(T));
        return value;
    }
    void Log(u32 offset, u32 size, const void* bytes) {
        const size_t at = arena.size();
        if (slot_material != 0) {
            slot_owner[at] = slot_material;
        }
        if (table_entry != 0) {
            table_owner[at] = table_entry;
        }
        if (direct_object != 0) {
            direct_owner[at] = {direct_object, direct_offset};
        }
        if (batch_owner != 0) {
            batch_owner_map[at] = batch_owner;
        }
        arena.resize(at + 8 + (size & ~OrMask));
        std::memcpy(arena.data() + at, &offset, 4);
        std::memcpy(arena.data() + at + 4, &size, 4);
        std::memcpy(arena.data() + at + 8, bytes, size & ~OrMask);
    }
    void Put(u32 offset, const void* bytes, u32 size) {
        std::memcpy(base + offset, bytes, size);
        if (!live) {
            Log(offset, size, bytes);
        }
    }
    template <class T>
    void Set(u32 offset, T value) {
        Put(offset, &value, sizeof(T));
    }
    void Or8(u32 offset, u8 bits) {
        base[offset] |= bits;
        if (!live) {
            Log(offset, 1 | OrMask, &bits);
        }
    }
};

void ReaderLock(u64 address) {
    // FUN_0120d030: increment the low half of the dword when the high half (the writer flag) is clear
    auto* word = reinterpret_cast<std::atomic<u32>*>(address);
    for (;;) {
        u32 expected = word->load(std::memory_order_relaxed) & 0xffff;
        if (word->compare_exchange_strong(expected, expected + 1, std::memory_order_acquire)) {
            return;
        }
        std::this_thread::yield();
    }
}
void ReaderUnlock(u64 address) {
    reinterpret_cast<std::atomic<u32>*>(address)->fetch_sub(1, std::memory_order_release);
}

using GuestBind = void(PS4_SYSV_ABI*)(u64, u64, u64, u64);
using GuestRelease = void(PS4_SYSV_ABI*)(u64);
using GuestGroupBind = void(PS4_SYSV_ABI*)(u64, u64, u64);
constexpr u64 FunctionGroupBind = 0x01ee4e50; // FUN_01ee4e50(inner, instance group object, view type)

/// The engine's reference counted shader slot (ctx + 0x4698 vertex, ctx + 0x4690 pixel): release the old object, store the new one, count it, set the dirty bit.
void SetShader(Target& t, u32 offset, u64 object, u8 dirty_bit) {
    const u64 current = t.Get<u64>(offset);
    if (current == object) {
        return;
    }
    if (t.live && !t.worker) { // (the worker's private context holds no references)
        if (current != 0) {
            auto* count = reinterpret_cast<std::atomic<s64>*>(current + 8);
            if (count->fetch_sub(1) == 1) {
                reinterpret_cast<GuestRelease>(R<u64>(R<u64>(current) + 0x10))(current);
            }
        }
        t.Set<u64>(offset, object);
        if (object != 0) {
            reinterpret_cast<std::atomic<s64>*>(object + 8)->fetch_add(1);
        }
    } else {
        t.Set<u64>(offset, object);
    }
    t.Or8(0x46a0, dirty_bit);
}

/// FUN_01c92b60 with the early copy of the resource slot the loop makes before the resource set: the material's render object into the context.
void CopyEarlyBlock(Target& t, u64 material) {
    const u64 object = R<u64>(material + 0x38);
    const u64 index_word = R<u64>(object + 0x440);
    const u64 index = (index_word & 0xffffffffull) * 0xb0;
    const u64 slot = R<u64>(object + 0xc0 + index);
    t.Put(0x980, reinterpret_cast<const void*>(object + 0x450), 8);
    if ((index_word >> 32) == 0) {
        t.Set<u64>(0x290, slot);
        t.Set<u64>(0x310, slot);
    } else {
        const u64 second = R<u64>(object + 0x2d0 + index);
        t.Set<u64>(0x290, slot);
        t.Set<u64>(0x298, second);
        t.Set<u64>(0x310, slot);
        t.Set<u64>(0x318, second);
    }
}

void BindMaterialImpl(Target& t, u64 material);
void BindMaterial(Target& t, u64 material) {
    if (!t.live) {
        BindMaterialImpl(t, material);
        return;
    }
    Timed(CallId::MaterialBlock, [&] { BindMaterialImpl(t, material); });
}
void BindMaterialImpl(Target& t, u64 material) {
    if (t.live) {
        ReaderLock(material + 0x1e0);
    }
    const u64 object = R<u64>(material + 0x38);
    t.Put(0x980, reinterpret_cast<const void*>(object + 0x450), 0x48);
    if (R<u32>(object + 0x444) != 0) {
        t.Put(0x9d0, reinterpret_cast<const void*>(object + 0x498), 0x48);
    }
    if (const u64 views = R<u64>(object + 0x4f0); views != 0) {
        t.Put(0x790, reinterpret_cast<const void*>(views), R<u32>(object + 0x4f8) * 8);
    }
    if (const u64 views = R<u64>(object + 0x4e0); views != 0) {
        t.Put(0xb38, reinterpret_cast<const void*>(views), R<u32>(object + 0x4e8) * 8);
    }
    const u64 index_word = R<u64>(object + 0x440);
    const u64 index = (index_word & 0xffffffffull) * 0xb0;
    const u64 slot = R<u64>(object + 0xc0 + index);
    if ((index_word >> 32) == 0) {
        t.Set<u64>(0x290, slot);
        t.Set<u64>(0x310, slot);
    } else {
        const u64 second = R<u64>(object + 0x2d0 + index);
        t.Set<u64>(0x290, slot);
        t.Set<u64>(0x298, second);
        t.Set<u64>(0x310, slot);
        t.Set<u64>(0x318, second);
    }
    if (t.live) {
        ReaderUnlock(material + 0x1e0);
    }
}

/// The key the engine's variant search is a function of, as far as the snapshot says (audit N6.3b and f), plus what the audit did not need to look at (bVar54).
u64 Mix(u64 h, u64 value) {
    return (h ^ value) * 0x100000001b3ull + 0x9E3779B97F4A7C15ull;
}

struct MatInfo {
    bool ok{false};
    const char* reason{nullptr};
    u32 flag{0};
    u64 hash{0};
};

MatInfo ReadMaterial(u64 material, u64 default_kind) {
    MatInfo info;
    if (R<u64>(material + 0x18) == default_kind) {
        info.reason = "material of the engine's default kind (the loop keeps the previous shaders)";
        return info;
    }
    const u64 object = R<u64>(material + 0x38);
    if (object == 0) {
        info.reason = "material without a render object";
        return info;
    }
    const u32 count_a = R<u32>(object + 0x4f8), count_b = R<u32>(object + 0x4e8);
    if ((R<u64>(object + 0x4f0) != 0 && (count_a == 0 || count_a > 62)) || (R<u64>(object + 0x4e0) != 0 && (count_b == 0 || count_b > 72))) {
        info.reason = "material with a shader resource count the engine asserts on";
        return info;
    }
    if ((R<u64>(object + 0x440) & 0xffffffffull) >= 64) {
        info.reason = "material with a constant buffer slot index out of range";
        return info;
    }
    (void)R<u64>(object + 0xc0 + (R<u64>(object + 0x440) & 0xffffffffull) * 0xb0);
    info.flag = R<u32>(material + 0xd4) & 1;
    u64 h = 0xcbf29ce484222325ull;
    for (const u32 offset : {0x24u, 0x28u, 0xd0u, 0xd4u, 0x158u, 0x15cu, 0x164u, 0x1bcu}) {
        h = Mix(h, R<u32>(material + offset));
    }
    // the variant search also reads two fields of the cached object behind the material (+0x48)
    if (const u64 cached = R<u64>(material + 0x48); cached != 0) {
        h = Mix(h, R<u32>(cached + 0x24));
        h = Mix(h, R<u32>(cached + 0x28));
    } else {
        h = Mix(h, 0xdeadbeefull);
    }
    info.hash = h;
    info.ok = true;
    return info;
}

void FillBatchLevel(BatchLevel& level, u64 batch, u64 default_object) {
    const u64 mesh = R<u64>(batch + 0x20);
    const u64 render_data = R<u64>(mesh + 0x38);
    const u64 inner = R<u64>(render_data + 0x20);
    level.batch = batch;
    level.inner = inner;
    level.value_580 = R<u64>(R<u64>(R<u64>(inner + 0x48) + 0x38) + 0x28);
    level.pass_object_288 = R<u64>(R<u64>(batch + 0x68) + 0x130);
    level.view_58 = 0;
    if (const u64 holder = R<u64>(inner + 0x50); holder != 0) {
        level.view_58 = R<u64>(R<u64>(holder + 0x38) + 0x20);
    }
    const u32 array_count = R<u32>(inner + 0x68);
    const u64 array = R<u64>(inner + 0x60);
    level.bvar54 = false;
    if (array_count != 0) {
        const u64 first = R<u64>(array);
        if (first != default_object && first != 0) {
            level.bvar54 = R<u64>(first + 0x38) != 0;
        }
    }
    const u32 entries = std::min<u32>(R<u32>(mesh + 0x68), 16);
    level.table_count = 0;
    for (u32 j = 0; j < entries; ++j) {
        if (j >= array_count) {
            continue;
        }
        const u64 entry = R<u64>(array + u64{j} * 8);
        if (entry == default_object || entry == 0) {
            continue;
        }
        const u64 holder = R<u64>(entry + 0x38);
        if (holder == 0) {
            continue;
        }
        level.table_entry[level.table_count] = entry;
        level.table[level.table_count++] = {j, R<u64>(holder + 0x28)};
    }
}

u64 StepKey(u64 material_hash, const Plan& plan, u16 batch_flags, bool bvar54, u16 mesh_word, u64 group_bits) {
    u64 h = material_hash;
    h = Mix(h, plan.view_type);
    h = Mix(h, plan.flag14);
    h = Mix(h, plan.flag1c);
    h = Mix(h, batch_flags & 0x10);
    h = Mix(h, bvar54 ? 1 : 0);
    h = Mix(h, mesh_word);
    h = Mix(h, group_bits);
    return h;
}

/// The visibility cache of pass kind 3 (the loop, after the raster state of a primitive of a batch with flag 0x8): the primitive's data object holds 8 hash chains at +0x68 keyed with the id
/// at owner + 0x790 (the node's first qword is the id in both halves). Returns 0 when there is no node, else the node.
u64 VisibilityNode(u64 owner, u64 record) {
    const u32 id = R<u32>(owner + 0x790);
    const u64 data = R<u64>(record + 0x28);
    for (u64 node = R<u64>(data + 0x68 + u64(id & 7) * 8); node != 0; node = R<u64>(node + 0x10)) {
        const u64 key = R<u64>(node);
        if (static_cast<u32>(key) == id && static_cast<u32>(key >> 32) == id) {
            return node;
        }
    }
    return 0;
}

/// True when the loop would not create a cache entry for the primitive: a node exists, or the list of the current slot is full (then it draws without one).
bool VisibilityKnown(const Plan& plan, u64 rd, u64 record) {
    if (VisibilityNode(plan.visibility_owner, record) != 0) {
        return true;
    }
    const u32 list = R<u32>(rd + 0x10978);
    return R<u32>(rd + 0x10988 + u64{list} * 0x18) > 0x7f;
}

// ------------------------------------------------------------------------------------------------------------------------------- scan (read only, guarded)

struct ScanArgs {
    Plan* plan;
    u64 render_device, desc;
    const std::unordered_map<u64, TableEntry>* table;
    bool worker{false}; // P1b: the worker's scan (the live context is the engine thread's: its shader references are not ours to release)
};

void ScanImpl(ScanArgs& args) {
    Plan& plan = *args.plan;
    const u64 rd = args.render_device, desc = args.desc;
    plan.render_device = rd;
    const u64 list = R<u64>(desc);
    const u64 view_word = R<u64>(desc + 8);
    plan.view_type = static_cast<u32>(view_word);
    plan.kind = R<u32>(desc + 0x10);
    const u32 mask = R<u32>(desc + 0x20);
    plan.flag14 = R<u8>(desc + 0x14);
    plan.flag1c = R<u8>(desc + 0x1c);
    const u8 flag_hi = static_cast<u8>(view_word >> 32);
    const u8 flag_cvar = static_cast<u8>(view_word >> 40);
    plan.ctx = R<u64>(rd + 0x2d8);
    plan.indirect_mode = R<s32>(rd + (plan.view_type == 0 ? 0x38e0 : 0x38e4));
    if (list == 0 || plan.ctx == 0) {
        plan.Decline("no list or no device context");
        return;
    }
    const u8 flag_1d = R<u8>(desc + 0x1d);
    if (plan.kind == 1 || plan.kind == 2 || plan.kind == 5) {
        plan.Decline("pass kind with queries or a target switch (desc + 0x10 in 1, 2, 5)");
    }
    if (R<u64>(rd + 0x8930) != 0) {
        plan.Decline("the engine's per draw register packet is on (renderDev + 0x8930)");
    }
    for (const u32 offset : {0x4698u, 0x4690u}) {
        if (args.worker) {
            break;
        }
        if (const u64 shader = R<u64>(plan.ctx + offset); shader != 0 && R<s64>(shader + 8) <= 1) {
            plan.Decline("the context holds a shader with a single reference (releasing it destroys it)");
        }
    }
    const u64 default_kind = R<u64>(Global(GlobalDefaultKind));
    const u64 context_global = R<u64>(Global(GlobalContext));
    const u64 default_object = R<u64>(R<u64>(context_global + 0x38) + 0x20);
    const u64 batch_array = R<u64>(list);
    const u32 batch_count = R<u32>(list + 8);
    if (batch_array == 0 || batch_count > 100000) {
        plan.Decline("batch list unreadable");
        return;
    }
    std::unordered_map<u64, MatInfo> materials;
    for (u32 b = 0; b < batch_count; ++b) {
        const u64 batch = R<u64>(batch_array + u64{b} * 8);
        if (batch == 0) {
            plan.Decline("null batch");
            continue;
        }
        const u16 batch_flags = R<u16>(batch + 0x66);
        if (flag_hi != 0 && (batch_flags & 2) != 0) {
            continue; // the loop skips the batch before it touches anything
        }
        const u64 mesh = R<u64>(batch + 0x20);
        if (mesh == 0) {
            continue;
        }
        const u64 group = R<u64>(batch + 0x38);
        const u64 render_data = R<u64>(mesh + 0x38);
        if (render_data == 0 || R<u64>(render_data + 0x20) == 0) {
            plan.Decline("mesh render data not created yet (the loop creates it at first use)");
            continue;
        }
        BatchLevel level;
        FillBatchLevel(level, batch, default_object);
        level.first_step = static_cast<u32>(plan.steps.size());
        if (group == 0 && flag_cvar != 0) {
            // pass flag byte 5: the loop draws only instance group batches; it has written ctx + 0x580 and leaves
            level.only_580 = true;
            plan.batches.push_back(level);
            continue;
        }
        u64 special_bits = 0;
        if (const u64 special = R<u64>(batch + 0x48); special != 0) {
            // the only class in use (live data): its virtuals are +0x48 (the token the variant key uses: the material's, or the VERTEX_MODIFIER token when special + 0x40 is set; a
            // reference counted handle, net zero over the loop's sequence), +0x50 (special + 0x40 != 0) and +0x58 (the resource table fill inside FUN_01c92f30, the engine's own)
            const u64 vtable = R<u64>(special);
            if (vtable - NativeDraw::ImageBase() + 0x1000000 != 0x26dbfc8) {
                S().special_classes[fmt::format("vtable {:#x}", vtable - NativeDraw::ImageBase() + 0x1000000)]++;
                plan.Decline("batch with a special entry of an unknown class");
                continue;
            }
            level.special = special;
            level.special_modifier = R<u64>(special + 0x40) != 0;
            special_bits = (1ull << 48) | (u64(level.special_modifier) << 49);
        }
        if (plan.kind == 3 && flag_1d == 0 && (batch_flags & 8) != 0) {
            level.visibility = true;
            plan.visibility_owner = R<u64>(desc + 0x28);
            if (plan.visibility_owner == 0) {
                plan.Decline("pass kind 3 without the visibility owner (desc + 0x28)");
                continue;
            }
        }
        u64 group_bits = special_bits;
        if (group != 0) {
            if (plan.indirect_mode == 0) {
                plan.Decline("instance group batch without the indirect mode (DrawIndexedInstanced is not hooked)");
                continue;
            }
            const u64 group_inner = R<u64>(group + 8);
            if (group_inner == 0) {
                if (flag_cvar != 0) {
                    level.only_580 = true;
                    plan.batches.push_back(level);
                } else {
                    plan.Decline("instance group without its inner object");
                }
                continue;
            }
            level.group = true;
            level.bvar54 = false; // the loop's flag for the batch of a group
            ++plan.group_batches;
            const u64 group_object = R<u64>(group_inner + 0x38);
            u32 instances = 0;
            if (group_object != 0) {
                instances = R<u32>(R<u64>(group_object + 0x28) + 0x48);
            }
            if (group_object == 0 || instances == 0) {
                level.only_580 = true; // the loop leaves such a batch after writing ctx + 0x580
                ++plan.group_batches_skipped;
                level.step_count = 0;
                plan.batches.push_back(level);
                continue;
            }
            level.group_object = group_object;
            level.indirect_buffer = R<u64>(group_object + 0x110);
            const u32 views_count = R<u32>(group_object + 0xe0);
            if (views_count > 45) {
                plan.Decline("instance group with more shader resource views than the engine accepts");
                continue;
            }
            // what FUN_01ee4e50 writes (read now for the verify run, and to find stale pointers under the guard)
            const s32 ring_index = R<s32>(group_object + 0x20);
            if (ring_index != 0 && ring_index != 1) {
                plan.Decline("instance group with a ring index outside 0 and 1");
                continue;
            }
            const u64 ring_entry = R<u64>(R<u64>(group_object + 0xb8) + u64(ring_index) * 8);
            const u64 view_object = R<u64>(R<u64>(level.inner + 0x48) + 0x38);
            level.g70 = R<u64>(view_object + 0x20);
            level.g78 = R<u64>(ring_entry + 0x20);
            level.g170 = R<u32>(view_object + 0x40);
            level.g174 = R<u32>(ring_entry + 0x40);
            if (const u64 uploader = R<u64>(group_object + 0xf0); uploader != 0) {
                level.g340 = R<u64>(R<u64>(uploader) + 0x10);
                level.g340_set = true;
            }
            level.gtable[0] = R<u64>(group_object + 0x1b0 + u64(plan.view_type) * 8);
            level.gtable[1] = R<u64>(group_object + 0x1f8 + u64(ring_index) * 8);
            level.gtable[2] = R<u64>(group_object + 0x1f8 + u64(1 - ring_index) * 8);
            const u64 views_array = R<u64>(group_object + 0xd8);
            for (u32 i = 0; i < views_count; ++i) {
                const u64 holder = R<u64>(views_array + u64{i} * 8);
                level.gtable[3 + i] = holder != 0 ? R<u64>(holder + 0x28) : 0;
            }
            level.table_dwords = 3 + views_count;
            group_bits = (1ull << 40) | (u64(plan.indirect_mode != 0) << 32) | R<u32>(group_inner + 0x40);
        }
        const u16 mesh_word = R<u16>(mesh + 0x78);
        const u16 prim_count = R<u16>(batch + 100);
        const u64 prim_array = R<u64>(batch + 0x58);
        for (u32 p = 0; p < prim_count; ++p) {
            const u64 record = prim_array + u64{p} * 0x38;
            const u64 material = R<u64>(record + 0x18);
            auto found = materials.find(material);
            if (found == materials.end()) {
                found = materials.emplace(material, ReadMaterial(material, default_kind)).first;
            }
            const MatInfo& info = found->second;
            if (!info.ok) {
                // the flag of the material decides whether the loop looks at the primitive at all: read it for the mask test even for materials the generator cannot take
                if ((mask & (2 - (R<u32>(material + 0xd4) & 1))) != 0) {
                    plan.Decline(info.reason);
                }
                continue;
            }
            if ((mask & (2 - info.flag)) == 0) {
                continue;
            }
            if (R<u64>(record + 0x28) == 0) {
                plan.Decline("primitive record without its per primitive data");
                continue;
            }
            Step step;
            step.record = record;
            step.material = material;
            step.flag = info.flag;
            step.key = StepKey(info.hash, plan, batch_flags, level.bvar54, mesh_word, group_bits);
            step.index = p;
            if (level.visibility && !VisibilityKnown(plan, rd, record)) {
                plan.Decline("visibility cache entry missing (the loop creates it, with a query around the draw)");
            }
            const auto entry = args.table->find(step.key);
            if (entry == args.table->end()) {
                plan.Decline("variant key not learned yet (the loop picks the variant at first use)");
            } else if (entry->second.ambiguous) {
                plan.Decline("variant key with two different variants");
            } else {
                step.variant = entry->second.variant;
                if (R<u64>(step.variant + 8) == 0 || R<u64>(step.variant + 0x18) == 0) {
                    plan.Decline("variant without a shader pair");
                }
            }
            plan.steps.push_back(step);
        }
        level.step_count = static_cast<u32>(plan.steps.size()) - level.first_step;
        plan.batches.push_back(level);
    }
    plan.scanned = true;
}

// ------------------------------------------------------------------------------------------------------------------------------- the generator

/// Reuse census (`TLG_GEN_CENSUS=1`): how many draws find the device context, as far as the generators and the record builder look at it, exactly as the previous draw of the pass
/// left it (ctx bytes 0 to 0xd80, the shader pair, the raster state object); only the draw arguments differ then.
struct ReuseCensus {
    std::atomic<u64> draws{0}, same_as_previous{0}, same_as_one_of_last_8{0};
};
ReuseCensus& Reuse() {
    static ReuseCensus census;
    return census;
}
std::array<std::atomic<u64>, 0xd80 / 8> reuse_changed_words;
void CountReuse(const Target& t) {
    thread_local std::array<u64, 0xd80 / 8> previous{};
    for (u32 i = 0; i < 0xd80 / 8; ++i) {
        u64 word;
        std::memcpy(&word, t.base + i * 8, 8);
        if (word != previous[i]) {
            reuse_changed_words[i].fetch_add(1, std::memory_order_relaxed);
            previous[i] = word;
        }
    }
    thread_local std::array<u64, 8> recent{};
    thread_local u32 next = 0;
    thread_local u64 last = 0;
    u64 h = 0xcbf29ce484222325ull;
    for (u32 offset = 0; offset < 0xd80; offset += 8) {
        u64 word;
        std::memcpy(&word, t.base + offset, 8);
        h = (h ^ word) * 0x100000001b3ull;
    }
    for (const u32 offset : {0x4690u, 0x4698u, 0x46c8u}) {
        u64 word;
        std::memcpy(&word, t.base + offset, 8);
        h = (h ^ word) * 0x100000001b3ull;
    }
    auto& r = Reuse();
    r.draws.fetch_add(1, std::memory_order_relaxed);
    if (h == last) {
        r.same_as_previous.fetch_add(1, std::memory_order_relaxed);
    }
    bool seen = false;
    for (const u64 x : recent) {
        seen |= x == h;
    }
    if (seen) {
        r.same_as_one_of_last_8.fetch_add(1, std::memory_order_relaxed);
    }
    last = h;
    recent[next++ & 7] = h;
}

void IssueDraw(Target& t, u32 kind, u64 count, u64 first, u64 a3 = 0) {
    if (t.worker) {
        if (!NativeDraw::IssueWorkerDraw(reinterpret_cast<u64>(t.base), kind == 0 ? 0u : 2u, count, first, a3)) {
            ++t.refused;
        }
        return;
    }
    if (t.live && CensusOn()) {
        CountReuse(t);
    }
    if (t.live) {
        Timed(CallId::DrawIssue, [&] {
        if (kind == 0) {
            NativeDraw::IssueIndexedDraw(reinterpret_cast<u64>(t.base), count, first, t.indexed_site, t.indexed_a4, a3);
        } else {
            NativeDraw::IssueIndirectDraw(reinterpret_cast<u64>(t.base), count, first, t.indirect_site);
        }
        });
    } else {
        t.marks.push_back({t.arena.size(), kind, count, first, a3, t.Get<u64>(0x4698), t.Get<u64>(0x4690)});
        if (t.rs_pending) {
            Mark& m = t.marks.back();
            m.rs_valid = true;
            m.rs_n = t.rs_n;
            m.rs_size = t.rs_size;
            m.rs_bytes = t.rs_bytes;
            t.rs_pending = false;
        }
        if (t.gb_buffer != 0) {
            Mark& m = t.marks.back();
            m.gb_buffer = t.gb_buffer;
            m.gb_src = t.gb_src;
            m.gb_n = t.gb_n;
            t.gb_buffer = 0;
        }
    }
}

u64 HandleLookup(u64 render_device, u64 object);
extern thread_local bool t_par_worker, t_par_lookup_failed;
void ResourceSetStep(Target& t, u64 rd, u64 material, u64 special, bool shadow_ok);
bool ParRealBuffers();
bool GroupBindNative(Target& t, const BatchLevel& level);
bool GroupBindWorker(Target& t, const BatchLevel& level);

void Generate(Target& t, const Plan& plan) {
    const u64 rd = plan.render_device;
    const bool pure_pass = std::none_of(plan.batches.begin(), plan.batches.end(), [](const BatchLevel& b) { return b.special != 0; });
    // prologue
    if (t.Get<u32>(0x4480) != 4) {
        t.Set<u32>(0x4480, 4);
        t.Or8(0x447c, 1);
    }
    t.Set<u64>(0x68, 0);
    bool raster_flag = false; // bVar15: which of the two raster state objects the last primitive asked for
    for (const BatchLevel& level : plan.batches) {
        if (t.worker && level.special != 0) {
            ++t.refused; // P1b: the vertex modifier entry binds through the engine's own function (it reads the global device): not on the worker (none in the cascades)
            continue;
        }
        if (t.split && !level.main_any) {
            continue; // P1: the worker makes all of it
        }
        t.batch_owner = level.batch;
        t.Set<u64>(0x580, level.value_580);
        t.batch_owner = 0;
        if (level.only_580) {
            continue;
        }
        if (level.group && t.worker) {
            if (!GroupBindWorker(t, level)) {
                ++t.group_fail;
                t.refused += level.step_count;
                continue;
            }
        } else if (level.group) {
            if (Timed(CallId::GroupBind, [&] { return GroupBindNative(t, level); })) {
                // done natively (the context words below the engine's function would have written, and the uploader)
            } else if (t.live) {
                Timed(CallId::GroupBind, [&] { reinterpret_cast<GuestGroupBind>(Global(FunctionGroupBind))(level.inner, level.group_object, plan.view_type); });
            } else {
                t.Set<u64>(0x70, level.g70);
                t.Set<u64>(0x78, level.g78);
                t.Set<u32>(0x170, level.g170);
                t.Set<u32>(0x174, level.g174);
                t.Set<u64>(0x1f0, 0);
                if (level.g340_set) {
                    t.Set<u64>(0x340, level.g340);
                }
                t.Put(0x800, level.gtable.data(), level.table_dwords * 8);
            }
        }
        t.batch_owner = level.batch;
        t.Set<u64>(0x58, level.view_58);
        t.Set<u64>(0x60, 0x39);
        t.Set<u64>(0x288, level.pass_object_288);
        t.Set<u64>(0x308, level.pass_object_288);
        t.batch_owner = 0;
        for (u32 i = level.table_count; i-- > 0;) {
            t.table_entry = level.table_entry[i];
            t.Set<u64>(0x6c0 + level.table[i].first * 8, level.table[i].second);
            t.table_entry = 0;
        }
        for (u32 s = level.first_step; s < level.first_step + level.step_count; ++s) {
            const Step& step = plan.steps[s];
            if (t.split && step.lane == 1) {
                continue; // P1: the other thread's primitive
            }
            if (t.worker && (step.variant == 0 || R<u64>(step.variant + 8) == 0 || R<u64>(step.variant + 0x18) == 0)) {
                ++t.unlearned; // P1b: the scan has no variant for it (first use: the engine's loop runs the next frame of this view and learns it)
                continue;
            }
            t.step_failed = false;
            t.slot_material = step.material;
            if (step.flag != 0) {
                CopyEarlyBlock(t, step.material);
                if (!level.bvar54) {
                    Timed(CallId::ResourceSet, [&] { ResourceSetStep(t, rd, step.material, level.special, pure_pass); });
                }
            }
            if (level.special != 0) {
                // the loop copies the material block first when its own conditions say so, then binds the resource set through the special entry (always), then the block again below
                const u64 cached = R<u64>(step.material + 0x48);
                const bool early = cached == 0 ? (R<u32>(step.material + 0x158) != 0 || (R<u8>(step.material + 0xd5) & 1) != 0)
                                               : ((R<u32>(cached + 0x24) != 0 && R<u32>(cached + 0x28) != 0) || (R<u8>(step.material + 0xd5) & 1) != 0);
                if (early) {
                    BindMaterial(t, step.material);
                }
                Timed(CallId::ResourceSet, [&] { ResourceSetStep(t, rd, step.material, level.special, pure_pass); });
            }
            BindMaterial(t, step.material);
            t.slot_material = 0;
            if (t.live) {
                W<u64>(R<u64>(step.record + 0x28) + 0x48, step.variant); // the loop stores the variant it chose behind the primitive record
            }
            SetShader(t, 0x4698, R<u64>(step.variant + 8), 1);
            SetShader(t, 0x4690, R<u64>(step.variant + 0x18), 2);
            const u32 raster_bits = R<u32>(step.material + 0xd4) & 0x2e;
            const bool raster = raster_bits != 0;
            if (raster != raster_flag) {
                raster_flag = raster;
                const u64 object = R<u64>(rd + (raster ? 0x3bf8 : 0x3bf0));
                if (t.Get<u64>(0x46c8) != object) {
                    t.Set<u64>(0x46c8, object);
                    t.Set<u8>(0x46d8, 1);
                }
            }
            if (level.visibility) {
                if (t.split) {
                    if (plan.culled[s] != 0) {
                        continue; // P1: the answer read when the pass was handed over
                    }
                } else if (const u64 node = VisibilityNode(plan.visibility_owner, step.record); node != 0 && R<s32>(node + 8) == 2) {
                    continue; // culled by the cache: the loop leaves the primitive after its state is set, before the draw and the counters
                }
            }
            if (t.split && step.lane == 2) {
                continue; // P1: the state only; the worker draws it (and the counters were added when the pass was handed over)
            }
            if (t.worker && t.step_failed) {
                ++t.refused;
                continue;
            }
            const s32 triangles = R<s32>(step.record + 0x14);
            if (t.live) {
                Core::Vkb::Graphics::SetStepHint(level.batch, step.material); // R7 step B
            }
            if (level.group) {
                IssueDraw(t, 1, level.indirect_buffer, (u64{step.index} * 9 + plan.view_type) * 0x18);
            } else {
                IssueDraw(t, 0, static_cast<u32>(triangles * 3), R<u32>(step.record + 0x10));
            }
            if (t.live) {
                Core::Vkb::Graphics::ClearStepHint();
            }
            if (t.worker) {
                ++t.draws; // P1b: added to the device's counters by the engine's thread at its next hand-over
                t.triangles += triangles;
            } else if (t.live) {
                W<s32>(rd + 0x10, R<s32>(rd + 0x10) + 1);
                W<s32>(rd + 0x14, R<s32>(rd + 0x14) + triangles);
            } else {
                ++t.draws;
                t.triangles += triangles;
            }
        }
        if (level.group) {
            // the end of an instanced batch clears the instance state
            const std::array<u8, 0x30> zeros{};
            t.Put(0x70, zeros.data(), 16);
            t.Set<u64>(0x170, 0);
            t.Set<u64>(0x1f0, 0);
            t.Set<u64>(0x340, 0);
            t.Put(0x800, zeros.data(), 0x30);
        }
    }
    // epilogue
    t.Set<u64>(0x980, 0);
    const u64 object = R<u64>(rd + 0x3bf0);
    if (t.Get<u64>(0x46c8) != object) {
        t.Set<u64>(0x46c8, object);
        t.Set<u8>(0x46d8, 1);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------- learning

struct LearnArgs {
    const Plan* plan;
    State* state;
};

void LearnImpl(LearnArgs& args) {
    State& s = *args.state;
    std::unordered_set<u64> cache(s.cache_seen.begin(), s.cache_seen.end());
    for (const Step& step : args.plan->steps) {
        const u64 data = R<u64>(step.record + 0x28);
        const u64 variant = data != 0 ? R<u64>(data + 0x48) : 0;
        if (variant == 0) {
            continue;
        }
        if (cache.find(variant) == cache.end()) {
            ++s.learn_not_in_cache;
            continue;
        }
        const auto [it, fresh] = s.table.try_emplace(step.key, TableEntry{variant, false});
        if (fresh) {
            ++s.learned;
        } else if (it->second.variant != variant && !it->second.ambiguous) {
            it->second.ambiguous = true;
            ++s.conflicts;
        }
    }
}

struct CacheArgs {
    State* state;
    u64 render_device;
};
/// The engine's variant cache (renderDev + 0x1c60, count + 0x1c68) only grows: a learned object stays valid as long as the entries we saw are still there.
void CheckCacheImpl(CacheArgs& args) {
    State& s = *args.state;
    const u64 array = R<u64>(args.render_device + 0x1c60);
    const u32 count = R<u32>(args.render_device + 0x1c68);
    bool same = array != 0 && count >= s.cache_seen.size() && count < 100000;
    for (size_t i = 0; same && i < s.cache_seen.size(); ++i) {
        same = R<u64>(array + i * 8) == s.cache_seen[i];
    }
    if (!same) {
        if (!s.table.empty()) {
            ++s.table_resets;
        }
        s.table.clear();
        s.cache_seen.clear();
    }
    if (array != 0 && count < 100000) {
        s.cache_seen.resize(count);
        for (u32 i = 0; i < count; ++i) {
            s.cache_seen[i] = R<u64>(array + u64{i} * 8);
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------- the G-buffer loop (N6.6)
// RenderDev_DrawGBufferPrimitives(renderDev, mode, list, args, flags, query mode, p7), ghidra 0x01c76710. What the live scenes ask of it (audit /gbuffer): mode 0, query mode 0, the
// flags 0x20; batches with and without an instance group, with and without a second render data (batch + 0x28); no special entries; every primitive has its variant cached behind its
// data (+0x40) and a negative +0x24. Everything else is declined. The variant is the cached one, so there is no key table here.

constexpr u64 GBufferBegin = 0x01c76710, GBufferEnd = 0x01c779a0; // DrawGBufferPrimitives and the next function (DrawForwardPass)
constexpr u64 GBufferIndexedSite = 0x01c777a1, GBufferIndirectSite = 0x01c77788;
constexpr u64 FunctionHandleLookup = 0x01c59060; // FUN_01c59060(render device, object) -> the object's render data or 0 (creates it at first use)

// FUN_01c59060 is the engine's own lookup of an object's render data (creates it at first use). Calling it also sets up what two site generators need (with a native answer they
// declined every pass, 2026-10-30), so it always runs. The native answer for the case "the object already has its render data" (three reads) was removed 2026-11-06: it was off by default.
struct LeafStats {
    std::atomic<u64> prologue_native{}, block_compared{}, block_differing{};
};
LeafStats& Leaf() {
    static LeafStats stats;
    return stats;
}
/// P2: on a pass worker the lookup takes only a texture the engine has made (RenderDev_GetGpuTexture makes a missing one: engine state of the engine's thread); else it
/// answers 0 and notes it (the caller leaves the primitive out; the engine's thread makes the texture at its own next use)
thread_local bool t_par_worker = false, t_par_lookup_failed = false;
u64 HandleLookup(u64 render_device, u64 object) {
    if (t_par_worker) {
        if (object == 0) {
            return 0;
        }
        const u64 texture = R<u64>(object + 0x38);
        if (R<s32>(object + 0x20) != 0 || texture == 0) {
            t_par_lookup_failed = true;
            return 0;
        }
        return texture;
    }
    return reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64)>(Global(FunctionHandleLookup))(render_device, object);
}
constexpr u64 GlobalObjectTable = 0x0689eed8;    // DAT_0689eed8

struct GBatch {
    BatchLevel level;     // value_580, view_58, pass_object_288, the table; group fields (inner, group_object, g70.., gtable)
    u64 batch{};
    u32 flags_value{0};   // ((batch flags & 4) << 4) | the pass flags
    bool has_second{false};
    u64 v840{0};
    bool has_mesh{false};
    bool pair{false};     // no group: the first table entry is a valid object: ctx + 0x588 / 0x590
    u64 v588{0}, v590{0};
    u64 e588{0}, e590{0}; // the table entry objects the two values come from
    u64 v68_group{0};     // group: ctx + 0x68
};

struct GStep {
    u32 index{};
    u64 record{}, material{}, variant{};
};

struct GPlan {
    bool scanned{false};
    std::vector<const char*> reasons;
    u64 render_device{0}, ctx{0};
    u32 mode{0}, flags{0}, query_mode{0};
    s32 indirect_mode{0};
    u64 v_b60{0}, v_3c28{0}, v_68{0};
    std::vector<GBatch> batches;
    std::vector<GStep> steps;
    u32 group_batches{0};
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* r) {
        for (const char* x : reasons) {
            if (x == r) {
                return;
            }
        }
        reasons.push_back(r);
    }
};

struct GScanArgs {
    GPlan* plan;
    u64 render_device, list, args;
    bool worker{false}; // P2: the live context's shaders are the engine thread's
};

void ScanGBuffer(GScanArgs& a) {
    GPlan& plan = *a.plan;
    const u64 rd = a.render_device;
    plan.render_device = rd;
    plan.ctx = R<u64>(rd + 0x2d8);
    plan.indirect_mode = R<s32>(rd + 0x38e0);
    if (plan.ctx == 0 || a.list == 0) {
        plan.Decline("no list or no device context");
        return;
    }
    if (plan.mode == 1) {
        plan.Decline("mode 1 (the other binding order, never seen)");
    }
    if (plan.query_mode == 1 || plan.query_mode == 2 || plan.query_mode == 5) {
        plan.Decline("query mode 1, 2, 5 (queries or a target switch)");
    }
    for (const u32 offset : {0x4698u, 0x4690u}) {
        if (a.worker) {
            break;
        }
        if (const u64 shader = R<u64>(plan.ctx + offset); shader != 0 && R<s64>(shader + 8) <= 1) {
            plan.Decline("the context holds a shader with a single reference (releasing it destroys it)");
        }
    }
    plan.v_3c28 = R<u64>(rd + 0x3c28);
    plan.v_68 = R<u64>(rd + 0x3d90 + u64(plan.mode == 1) * 8);
    {
        const u64 table = R<u64>(Global(GlobalObjectTable));
        const u64 handle = R<u64>(R<u64>(a.args + 0x2d0) + 0x30);
        const u64 object = Timed(CallId::HandleLookup, [&] { return HandleLookup(table, handle); });
        plan.v_b60 = object != 0 ? R<u64>(object + 0x48) : 0;
    }
    const u64 context_global = R<u64>(Global(GlobalContext));
    const u64 default_object = R<u64>(R<u64>(context_global + 0x38) + 0x20);
    const u64 batch_array = R<u64>(a.list);
    const u32 batch_count = R<u32>(a.list + 8);
    if (batch_count != 0 && (batch_array == 0 || batch_count > 100000)) {
        plan.Decline("batch list unreadable");
        return;
    }
    for (u32 b = 0; b < batch_count; ++b) {
        const u64 batch = R<u64>(batch_array + u64{b} * 8);
        if (batch == 0) {
            plan.Decline("null batch");
            continue;
        }
        GBatch gb;
        gb.batch = batch;
        const u16 batch_flags = R<u16>(batch + 0x66);
        gb.flags_value = (u32{static_cast<u32>(batch_flags & 4)} << 4) | plan.flags;
        if (R<u64>(batch + 0x48) != 0) {
            plan.Decline("batch with a special entry");
            continue;
        }
        if (const u64 second = R<u64>(batch + 0x28); second != 0) {
            const u64 second_data = R<u64>(second + 0x38);
            if (second_data == 0 || R<u64>(second_data + 0x20) == 0) {
                plan.Decline("second render data not created yet");
                continue;
            }
            gb.has_second = true;
            gb.v840 = R<u64>(R<u64>(R<u64>(R<u64>(second_data + 0x20) + 0x48) + 0x38) + 0x28);
        }
        const u64 mesh = R<u64>(batch + 0x20);
        gb.has_mesh = mesh != 0;
        gb.level.first_step = static_cast<u32>(plan.steps.size());
        if (!gb.has_mesh) {
            plan.batches.push_back(gb);
            continue;
        }
        const u64 render_data = R<u64>(mesh + 0x38);
        if (render_data == 0 || R<u64>(render_data + 0x20) == 0) {
            plan.Decline("mesh render data not created yet (the loop creates it at first use)");
            continue;
        }
        FillBatchLevel(gb.level, batch, default_object);
        gb.level.first_step = static_cast<u32>(plan.steps.size());
        const u64 inner = gb.level.inner;
        const u64 group = R<u64>(batch + 0x38);
        if (group == 0) {
            const u32 count = R<u32>(inner + 0x68);
            if (count != 0) {
                const u64 array = R<u64>(inner + 0x60);
                const u64 first = R<u64>(array);
                if (first != default_object && first != 0 && R<u64>(first + 0x38) != 0) {
                    gb.pair = true;
                    gb.v588 = R<u64>(R<u64>(first + 0x38) + 0x28);
                    gb.e588 = first;
                    u64 second_entry = R<u64>(array + 8);
                    if (second_entry == default_object) {
                        second_entry = 0;
                    }
                    if (second_entry == 0) {
                        plan.Decline("table with a valid first entry and no second one (the loop reads through a null pointer)");
                        continue;
                    }
                    gb.v590 = R<u64>(R<u64>(second_entry + 0x38) + 0x28);
                    gb.e590 = second_entry;
                }
            }
        } else {
            const u64 group_inner = R<u64>(group + 8);
            if (group_inner == 0) {
                plan.Decline("instance group without its inner object");
                continue;
            }
            if (plan.indirect_mode == 0) {
                plan.Decline("instance group batch without the indirect mode (DrawIndexedInstanced is not hooked)");
                continue;
            }
            const u64 group_object = R<u64>(group_inner + 0x38);
            u32 instances = 0;
            if (group_object != 0) {
                instances = R<u32>(R<u64>(group_object + 0x28) + 0x48);
            }
            gb.level.group = true;
            ++plan.group_batches;
            if (group_object == 0 || instances == 0) {
                gb.level.only_580 = true; // the loop leaves the batch after ctx + 0x580
                plan.batches.push_back(gb);
                continue;
            }
            gb.level.group_object = group_object;
            gb.level.indirect_buffer = R<u64>(group_object + 0x110);
            const u32 views_count = R<u32>(group_object + 0xe0);
            const s32 ring_index = R<s32>(group_object + 0x20);
            if (views_count > 45 || (ring_index != 0 && ring_index != 1)) {
                plan.Decline("instance group with an unusual view count or ring index");
                continue;
            }
            const u64 ring_entry = R<u64>(R<u64>(group_object + 0xb8) + u64(ring_index) * 8);
            const u64 view_object = R<u64>(R<u64>(inner + 0x48) + 0x38);
            gb.level.g70 = R<u64>(view_object + 0x20);
            gb.level.g78 = R<u64>(ring_entry + 0x20);
            gb.level.g170 = R<u32>(view_object + 0x40);
            gb.level.g174 = R<u32>(ring_entry + 0x40);
            if (const u64 uploader = R<u64>(group_object + 0xf0); uploader != 0) {
                gb.level.g340 = R<u64>(R<u64>(uploader) + 0x10);
                gb.level.g340_set = true;
            }
            gb.level.gtable[0] = R<u64>(group_object + 0x1b0); // view type 0
            gb.level.gtable[1] = R<u64>(group_object + 0x1f8 + u64(ring_index) * 8);
            gb.level.gtable[2] = R<u64>(group_object + 0x1f8 + u64(1 - ring_index) * 8);
            const u64 views_array = R<u64>(group_object + 0xd8);
            for (u32 i = 0; i < views_count; ++i) {
                const u64 holder = R<u64>(views_array + u64{i} * 8);
                gb.level.gtable[3 + i] = holder != 0 ? R<u64>(holder + 0x28) : 0;
            }
            gb.level.table_dwords = 3 + views_count;
            gb.v68_group = plan.indirect_mode == 0 ? R<u64>(rd + 0x3da8) : R<u64>(rd + 0x3d90);
        }
        const u16 prim_count = R<u16>(batch + 100);
        const u64 prim_array = R<u64>(batch + 0x58);
        for (u32 p = 0; p < prim_count; ++p) {
            const u64 record = prim_array + u64{p} * 0x38;
            GStep step;
            step.index = p;
            step.record = record;
            step.material = R<u64>(record + 0x18);
            const u64 data = R<u64>(record + 0x28);
            if (data == 0 || R<s32>(record + 0x24) >= 0) {
                plan.Decline(data == 0 ? "primitive record without its per primitive data" : "primitive with a non-negative +0x24 (another draw path)");
                continue;
            }
            step.variant = R<u64>(data + 0x40);
            if (step.variant == 0 || R<u32>(record + 0x30) != 0) {
                plan.Decline("primitive without a cached variant (the loop creates it)");
                continue;
            }
            if (R<u64>(step.variant + 8) == 0 || R<u64>(step.variant + 0x18) == 0) {
                plan.Decline("variant without a shader pair");
                continue;
            }
            const u64 object = R<u64>(step.material + 0x38);
            if (object == 0) {
                plan.Decline("material without a render object");
                continue;
            }
            const u32 count_a = R<u32>(object + 0x4f8), count_b = R<u32>(object + 0x4e8);
            if ((R<u64>(object + 0x4f0) != 0 && (count_a == 0 || count_a > 62)) || (R<u64>(object + 0x4e0) != 0 && (count_b == 0 || count_b > 72)) || (R<u64>(object + 0x440) & 0xffffffffull) >= 64) {
                plan.Decline("material with a resource count or slot index the engine asserts on");
                continue;
            }
            plan.steps.push_back(step);
        }
        gb.level.step_count = static_cast<u32>(plan.steps.size()) - gb.level.first_step;
        plan.batches.push_back(gb);
    }
    plan.scanned = true;
}

void GenerateGBuffer(Target& t, const GPlan& plan, u32 pass_flags) {
    const u64 rd = plan.render_device;
    if (t.Get<u32>(0x4480) != 4) {
        t.Set<u32>(0x4480, 4);
        t.Or8(0x447c, 1);
    }
    t.Set<u64>(0x68, plan.v_68);
    t.Set<u64>(0xb60, plan.v_b60);
    bool raster_flag = false;
    u32 local_158 = 0, local_150 = pass_flags;
    for (const GBatch& gb : plan.batches) {
        const BatchLevel& level = gb.level;
        if (pass_flags != 0) {
            local_150 = local_158;
            if (local_158 != gb.flags_value) {
                t.Set<u64>(0x46e0, plan.v_3c28);
                t.Set<u32>(0x46e8, gb.flags_value);
                t.Set<u8>(0x46ec, 1);
                local_158 = gb.flags_value;
                local_150 = gb.flags_value;
            }
        }
        if (gb.has_second) {
            t.Set<u64>(0x840, gb.v840);
        }
        if (!gb.has_mesh) {
            continue;
        }
        t.batch_owner = level.batch;
        t.Set<u64>(0x580, level.value_580);
        t.batch_owner = 0;
        if (!level.group) {
            t.Set<u64>(0x68, 0);
            t.batch_owner = level.batch;
            t.Set<u64>(0x58, level.view_58);
            t.batch_owner = 0;
            t.Set<u64>(0x60, 0x39);
            if (gb.pair) {
                t.Set<u64>(0x580, level.value_580);
                t.table_entry = gb.e588;
                t.Set<u64>(0x588, gb.v588);
                t.table_entry = gb.e590;
                t.Set<u64>(0x590, gb.v590);
                t.table_entry = 0;
            }
        } else {
            if (level.only_580) {
                continue;
            }
            if (t.worker && !ParRealBuffers()) {
                if (!GroupBindWorker(t, level)) {
                    ++t.group_fail;
                    t.refused += level.step_count;
                    continue;
                }
            } else if (Timed(CallId::GroupBind, [&] { return GroupBindNative(t, level); })) {
                // done natively
            } else if (t.live) {
                Timed(CallId::GroupBind, [&] { reinterpret_cast<GuestGroupBind>(Global(FunctionGroupBind))(level.inner, level.group_object, 0); });
            } else {
                t.Set<u64>(0x70, level.g70);
                t.Set<u64>(0x78, level.g78);
                t.Set<u32>(0x170, level.g170);
                t.Set<u32>(0x174, level.g174);
                t.Set<u64>(0x1f0, 0);
                if (level.g340_set) {
                    t.Set<u64>(0x340, level.g340);
                }
                t.Put(0x800, level.gtable.data(), level.table_dwords * 8);
            }
            t.Set<u64>(0x68, gb.v68_group);
            t.batch_owner = level.batch;
            t.Set<u64>(0x58, level.view_58);
            t.batch_owner = 0;
            t.Set<u64>(0x60, 0x39);
        }
        if (t.moved_check != nullptr) { // P2 census: did the engine's thread move this batch's data after the hand-over?
            if (const auto it = t.moved_check->find(level.batch); it != t.moved_check->end()) {
                ++t.moved_checked;
                const u64 obj = level.pass_object_288;
                const u64 cb = obj != 0 ? (u64{R<u32>(obj + 0x60)} | u64{R<u32>(obj + 0x64) & 0xffu} << 32) : 0;
                const u64 view = level.value_580;
                const u64 vb = view != 0 && R<u64>(view + 0x30) != 0 ? R<u64>(R<u64>(view + 0x30) + 0x30) : 0;
                t.moved_cb += cb != it->second.first;
                t.moved_vb += vb != it->second.second;
            }
        }
        // FUN_01c932d0
        t.batch_owner = level.batch;
        t.Set<u64>(0x288, level.pass_object_288);
        t.Set<u64>(0x308, level.pass_object_288);
        t.batch_owner = 0;
        for (u32 i = level.table_count; i-- > 0;) {
            t.table_entry = level.table_entry[i];
            t.Set<u64>(0x6c0 + level.table[i].first * 8, level.table[i].second);
            t.table_entry = 0;
        }
        for (u32 s = level.first_step; s < level.first_step + level.step_count; ++s) {
            const GStep& step = plan.steps[s];
            t.step_failed = false;
            Timed(CallId::ResourceSet, [&] { ResourceSetStep(t, rd, step.material, 0, true); });
            const bool raster = (R<u32>(step.material + 0xd4) & 0x26) != 0;
            if (raster != raster_flag) {
                raster_flag = raster;
                const u64 object = R<u64>(rd + (raster ? 0x3bf8 : 0x3bf0));
                if (t.Get<u64>(0x46c8) != object) {
                    t.Set<u64>(0x46c8, object);
                    t.Set<u8>(0x46d8, 1);
                }
            }
            SetShader(t, 0x4698, R<u64>(step.variant + 8), 1);
            SetShader(t, 0x4690, R<u64>(step.variant + 0x18), 2);
            t.slot_material = step.material;
            BindMaterial(t, step.material);
            t.slot_material = 0;
            const s32 triangles = R<s32>(step.record + 0x14);
            if (t.worker && t.step_failed) {
                ++t.refused;
                continue;
            }
            if (level.group) {
                IssueDraw(t, 1, level.indirect_buffer, u64{step.index} * 0xd8);
            } else {
                IssueDraw(t, 0, static_cast<u32>(triangles * 3), R<u32>(step.record + 0x10));
            }
            if (t.worker) {
                ++t.draws;
                t.triangles += triangles;
            } else if (t.live) {
                W<s32>(rd + 0x10, R<s32>(rd + 0x10) + 1);
                W<s32>(rd + 0x14, R<s32>(rd + 0x14) + triangles);
            } else {
                ++t.draws;
                t.triangles += triangles;
            }
        }
        if (level.group) {
            const std::array<u8, 0x30> zeros{};
            t.Put(0x70, zeros.data(), 16);
            t.Set<u64>(0x170, 0);
            t.Set<u64>(0x1f0, 0);
            t.Set<u64>(0x340, 0);
            t.Put(0x800, zeros.data(), 0x30);
        }
    }
    // epilogue: memset(ctx + 0x580, 0, 0x400), memset(ctx + 0x980, 0, 0x400)
    {
        const std::array<u8, 0x400> zeros{};
        t.Put(0x580, zeros.data(), 0x400);
        t.Put(0x980, zeros.data(), 0x400);
    }
    const u64 object = R<u64>(rd + 0x3bf0);
    if (t.Get<u64>(0x46c8) != object) {
        t.Set<u64>(0x46c8, object);
        t.Set<u8>(0x46d8, 1);
    }
    t.Set<u64>(0x46e0, plan.v_3c28);
    t.Set<u32>(0x46e8, local_150);
    t.Set<u8>(0x46ec, 1);
}

struct GConfig {
    bool on{false};
};
const GConfig& GCfg() {
    static const GConfig config = [] {
        GConfig c;
        const char* e = std::getenv("TLG_GBUFFER_GEN");
        // under TLG_VKB: on for the G-buffer workers; without TLG_VKB: default on since 2026-10-27 (the owner's play test); TLG_GBUFFER_GEN=0 turns it off
        c.on = e != nullptr ? *e != '0' : (GeneratorsDefaultOn() || ParGbufferSwitch());
        return c;
    }();
    return config;
}

struct GStats {
    ViewStats v;
    std::map<u32, u64> arguments; // mode * 1000 + query mode -> passes
    u64 eligible{0};
};
GStats& GS() {
    static GStats stats;
    return stats;
}

// ------------------------------------------------------------------------------------------------------------------------------- the forward loop (N6.7)
// RenderDev_DrawForwardPass(renderDev, a1, a2, list, mode), ghidra 0x01c779a0. Batches with an instance group are skipped by the loop; the others are drawn plain. The variant of every
// primitive is found or created by the engine's own FUN_01c822c0 (called here, in both the live run and the verify run's shadow, so there is no key table). Live census: mode 0, no
// special entries, material + 0x158 == 0, the first table entry never a valid object.

constexpr u64 ForwardBegin = 0x01c779a0, ForwardEnd = 0x01c782a0;
constexpr u64 ForwardIndexedSite = 0x01c7819d;
constexpr u64 FunctionForwardPrologue = 0x01c710f0; // FUN_01c710f0(renderDev, a1, a2, mode)
constexpr u64 FunctionFindVariant = 0x01c822c0;     // FUN_01c822c0(renderDev + 1000, material, flags, 0, mesh word, &token, 0) -> variant or 0
std::atomic<u64> last_forward_ctx{0}; // study: the device context seen by the forward generator (the report lists its vtable)
// FUN_01c822c0: the engine's variant search (cache lists under its lock, creation). A native answer from `material + 0x1d0` was removed 2026-11-06 (off by default).
u64 FindVariant(u64 cache, u64 material, u32 flags, u32 flags2, u32 mesh_word, u32* token) {
    return reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u32, u32, u32, void*, u64)>(Global(FunctionFindVariant))(cache, material, flags, flags2, mesh_word, token, 0);
}
constexpr u64 FunctionHandleFree = 0x01e45a60;      // FUN_01e45a60(handle): the last reference went
constexpr u64 HandleRefTable = 0x02716240;          // DAT_02716240: 0x20 bytes per handle, the reference count first

void HandleRefInc(u64 handle) {
    reinterpret_cast<std::atomic<s32>*>(Global(HandleRefTable) + handle * 0x20)->fetch_add(1);
}
void HandleRefDec(u64 handle) {
    if (reinterpret_cast<std::atomic<s32>*>(Global(HandleRefTable) + handle * 0x20)->fetch_sub(1) == 1) {
        Timed(CallId::HandleFree, [&] { reinterpret_cast<void(PS4_SYSV_ABI*)(u64)>(Global(FunctionHandleFree))(handle); });
    }
}

// A3: FUN_01c92f30(renderDev + 0x2910, ctx, material, special), the per primitive resource set, read off the decompilation (stages/a3-engine-functions.md). `TLG_GEN_NATIVE_LEAF=4`
// makes it natively for a primitive without a special entry, with the constant buffer of the matrices updated the way the context's UpdateSubresource (vtable + 0xd0,
// FUN_02480f90) does it for the dynamic buffer kind; any other shape runs the engine's function. In the verify run's shadow the effects on the device (the matrices block, the
// buffer's descriptor and ring block) cannot be applied: the shadow computes the expected bytes, the first draw after the call compares them with what the engine's loop left
// (`NoteLoopDraw`), and a pass with a special entry is not modelled in the shadow at all (as before).
constexpr u64 FunctionAllocFrameGarlic = 0x02476350; // JsDevice_AllocFrameGarlic(allocator, size, align)
constexpr u64 GlobalDeviceHolder = 0x06b03558;       // DAT_06b03558: holder of the frame allocator (+ 0x10238)
u64 RenderDataOf(u64 render_device, u64 object);
struct RsStatsT {
    std::atomic<u64> native{}, engine_special{}, engine_shape{}, compared{}, differing{};
};
RsStatsT& RsStats() {
    static RsStatsT stats;
    return stats;
}
bool NativeResourceSetOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_GEN_NATIVE_LEAF");
        return e != nullptr && *e != 0 && std::atoi(e) >= 4;
    }();
    return on;
}
/// debug (TLG_VKB_PAR_REAL_BUFFERS=1, only with TLG_VKB_PAR_INLINE=1): the worker's matrices / instance data go into the engine's real buffer objects, as the serial path
bool ParRealBuffers() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_PAR_REAL_BUFFERS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
/// P1b: the worker's pieces of its ring block (256 byte aligned; null when the block is used up: the caller leaves the draw out)
constexpr u32 ParStageBytes = 4096;
constexpr u32 ParCloneBytes = 0x200;
u8* WorkerAlloc(Target& t, u64 size) {
    const u64 at = (t.arena_used + 255) & ~u64{255};
    t.arena_need = std::max(t.arena_need, at + size);
    if (t.ring == nullptr || at + size > t.arena_size) {
        t.arena_used = at + size;
        return nullptr;
    }
    t.arena_used = at + size;
    return t.ring + at;
}
/// What the context's UpdateSubresource of a dynamic buffer writes into the buffer object (the descriptor's address and size, the stamp, a new version)
void WriteRename(u64 object, u64 address, u32 size, u32 stamp) {
    const u64 holder = R<u64>(Global(GlobalDeviceHolder));
    W<u32>(object + 0x2c, stamp);
    W<u32>(object + 0x60, static_cast<u32>(address));
    W<u32>(object + 0x64, static_cast<u32>(address >> 32) & 0xfffu | 0x100000u);
    W<u32>(object + 0x6c, 0x20077facu);
    W<u32>(object + 0x68, (size + 0xfu) >> 4);
    W<s64>(object + 0x160, reinterpret_cast<std::atomic<s64>*>(holder + 0x302a0)->fetch_add(1));
}
void ResourceSetStep(Target& t, u64 rd, u64 material, u64 special, bool shadow_ok) {
    const u64 rd2910 = rd + 0x2910;
    const auto engine = [&] { reinterpret_cast<GuestBind>(Global(FunctionBindResources))(rd2910, reinterpret_cast<u64>(t.base), material, special); };
    const bool native = (NativeResourceSetOn() || t.worker) && special == 0;
    if (!native) {
        if (special != 0 && NativeResourceSetOn()) {
            RsStats().engine_special.fetch_add(1, std::memory_order_relaxed);
        }
        if (t.worker) {
            t.step_failed = true;
            ++t.rs_fail;
            return;
        }
        if (t.live) {
            engine();
        }
        return;
    }
    if (!t.live && !shadow_ok) {
        return;
    }
    if (!(R<s32>(material + 0x1bc) != 0 || R<u64>(material + 0x48) != 0)) {
        RsStats().native.fetch_add(1, std::memory_order_relaxed); // the engine's function does nothing for this material
        return;
    }
    const u64 buffer = R<u64>(rd2910 + 0xef0);
    if (buffer == 0 || R<s32>(buffer + 0x124) != 1 || (R<u8>(buffer + 0x188) & 4) == 0 || R<u32>(buffer + 0x180) < 112) {
        RsStats().engine_shape.fetch_add(1, std::memory_order_relaxed);
        if (t.worker) {
            t.step_failed = true;
            ++t.rs_fail;
            return;
        }
        if (t.live) {
            engine();
        }
        return;
    }
    const u64 table = R<u64>(Global(GlobalObjectTable));
    const u64 list = R<u64>(R<u64>(Global(GlobalContext)) + 0x38);
    std::array<u64, 8> slot;
    slot.fill(RenderDataOf(table, R<u64>(list + 0x30)));
    // the matrices: up to seven 16 byte entries from the material into the block at renderDev + 0x3790 (the shadow keeps its own copy)
    const u32 n = std::min<u32>(7, R<u32>(material + 0x180));
    std::array<u8, 112> matrices{};
    if (n != 0) {
        std::memcpy(matrices.data(), reinterpret_cast<const void*>(R<u64>(material + 0x178)), n * 16);
    }
    // the textures: the material's list is read under its spin lock
    auto* lock = reinterpret_cast<std::atomic<s32>*>(material + 0x1a4);
    for (;;) {
        s32 expected = 0;
        if (lock->compare_exchange_strong(expected, 1)) {
            break;
        }
        std::this_thread::yield();
    }
    const u32 count = std::min<u32>(8, R<u32>(material + 0x198));
    const u64 entries = R<u64>(material + 400);
    for (u32 i = 0; i < count; ++i) {
        u64 data = RenderDataOf(table, R<u64>(entries + u64{i} * 8));
        if (data == 0) {
            data = RenderDataOf(table, R<u64>(list + 0x38));
        }
        slot[i] = data;
    }
    {
        s32 expected = 1;
        lock->compare_exchange_strong(expected, 0);
    }
    if (t.worker && t_par_lookup_failed) {
        t_par_lookup_failed = false;
        t.step_failed = true;
        ++t.rs_fail;
        return;
    }
    const u32 mask = R<u32>(material + 0x170);
    if ((mask & 1) != 0) {
        for (u32 i = 0; i < 8; ++i) {
            t.Set<u64>(0xa40 + i * 8, slot[i]);
        }
    }
    if ((mask & 2) != 0) {
        for (u32 i = 0; i < 8; ++i) {
            t.Set<u64>(0x740 + i * 8, slot[i]);
        }
    }
    const u32 size = R<u32>(buffer + 0x180);
    static const bool real_rs = [] { const char* e = std::getenv("TLG_VKB_PAR_REAL_RS"); return e != nullptr && e[0] == '1'; }(); // debug, inline only
    if (t.worker && !ParRealBuffers() && !real_rs) {
        // P1b: the same into the worker's own copies: its staging block, a piece of its ring block, its copy of the buffer object (the engine's thread keeps renaming the real one)
        u8* block = size <= ParStageBytes && t.rs_clone != nullptr ? WorkerAlloc(t, size) : nullptr;
        if (block == nullptr) {
            t.step_failed = true;
            ++t.rs_fail;
            return;
        }
        if (n != 0) {
            std::memcpy(t.rs_stage, matrices.data(), n * 16);
        }
        std::memcpy(block, t.rs_stage, size);
        const u64 clone = reinterpret_cast<u64>(t.rs_clone);
        WriteRename(clone, reinterpret_cast<u64>(block), size, t.Get<u32>(0x47ac));
        t.Set<u64>(0x2e0, clone);
        t.Set<u64>(0x360, clone);
        RsStats().native.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (t.live) {
        // UpdateSubresource on the dynamic buffer: the whole block at renderDev + 0x3790 goes to a block of the frame ring, the buffer's descriptor points at it
        if (n != 0) {
            std::memcpy(reinterpret_cast<void*>(rd2910 + 0xe80), matrices.data(), n * 16);
        }
        W<u32>(buffer + 0x2c, t.Get<u32>(0x47ac));
        const u64 holder = R<u64>(Global(GlobalDeviceHolder));
        void* block = reinterpret_cast<void*>(reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u64)>(Global(FunctionAllocFrameGarlic))(R<u64>(holder + 0x10238), size, 0x10));
        std::memcpy(block, reinterpret_cast<const void*>(rd2910 + 0xe80), size);
        const u64 address = reinterpret_cast<u64>(block);
        W<u32>(buffer + 0x60, static_cast<u32>(address));
        W<u32>(buffer + 0x64, static_cast<u32>(address >> 32) & 0xfffu | 0x100000u);
        W<u32>(buffer + 0x6c, 0x20077facu);
        W<u32>(buffer + 0x68, (size + 0xfu) >> 4);
        const s64 version = reinterpret_cast<std::atomic<s64>*>(holder + 0x302a0)->fetch_add(1);
        W<s64>(buffer + 0x160, version);
    } else {
        t.rs_pending = true;
        t.rs_n = n;
        t.rs_size = size;
        t.rs_bytes = matrices;
    }
    t.Set<u64>(0x2e0, buffer);
    t.Set<u64>(0x360, buffer);
    RsStats().native.fetch_add(1, std::memory_order_relaxed);
}

// A3: FUN_01ee4e50(inner, group, view type), the instance group bind. The context words and the table are the values the scan already read (the shadow writes the same); what is
// left is the uploader: the staged instance data goes into the uploader's ring block, the node's buffer is updated like the context's UpdateSubresource does it for a dynamic
// buffer, and the double buffer index toggles. `TLG_GEN_NATIVE_LEAF=5`. A buffer of another shape runs the engine's function (returns false).
int LeafLevel() {
    static const int level = [] {
        const char* e = std::getenv("TLG_GEN_NATIVE_LEAF");
        return e != nullptr && *e != 0 ? std::atoi(e) : 0;
    }();
    return level;
}
struct GbStatsT {
    std::atomic<u64> native{}, engine_shape{}, compared{}, differing{};
};
GbStatsT& GbStats() {
    static GbStatsT stats;
    return stats;
}
bool GroupBindNative(Target& t, const BatchLevel& level) {
    if (LeafLevel() < 5) {
        return false;
    }
    const u64 uploader = R<u64>(level.group_object + 0xf0);
    u64 node = 0, buffer = 0, value = 0, staged = 0;
    s32 staged_bytes = 0;
    if (uploader != 0) {
        node = R<u64>(uploader);
        buffer = R<u64>(node + 0x10);
        const s32 index = R<s32>(uploader + 0x28);
        value = R<u64>(uploader + 0x18 + static_cast<u64>(index & 1) * 8);
        staged = R<u64>(uploader + 8);
        staged_bytes = R<s32>(uploader + 0x10);
        if ((index != 0 && index != 1) || buffer == 0 || R<s32>(buffer + 0x124) != 1 || (R<u8>(buffer + 0x188) & 4) == 0 || R<u32>(buffer + 0x180) == 0 || staged_bytes < 0 ||
            level.g340 != buffer) {
            GbStats().engine_shape.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }
    t.Set<u64>(0x70, level.g70);
    t.Set<u64>(0x78, level.g78);
    t.Set<u32>(0x170, level.g170);
    t.Set<u32>(0x174, level.g174);
    t.Set<u64>(0x1f0, 0);
    if (uploader != 0) {
        if (t.live) {
            std::memcpy(reinterpret_cast<void*>(value), reinterpret_cast<const void*>(staged), static_cast<size_t>(staged_bytes));
            W<u64>(node, value);
            const u32 size = R<u32>(buffer + 0x180);
            W<u32>(buffer + 0x2c, t.Get<u32>(0x47ac));
            const u64 holder = R<u64>(Global(GlobalDeviceHolder));
            void* block = reinterpret_cast<void*>(reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u64)>(Global(FunctionAllocFrameGarlic))(R<u64>(holder + 0x10238), size, 0x10));
            std::memcpy(block, reinterpret_cast<const void*>(value), size);
            const u64 address = reinterpret_cast<u64>(block);
            W<u32>(buffer + 0x60, static_cast<u32>(address));
            W<u32>(buffer + 0x64, static_cast<u32>(address >> 32) & 0xfffu | 0x100000u);
            W<u32>(buffer + 0x6c, 0x20077facu);
            W<u32>(buffer + 0x68, (size + 0xfu) >> 4);
            W<s64>(buffer + 0x160, reinterpret_cast<std::atomic<s64>*>(holder + 0x302a0)->fetch_add(1));
            W<u8>(uploader + 0x28, R<u8>(uploader + 0x28) ^ 1);
        } else {
            t.gb_buffer = buffer;
            t.gb_src = staged;
            t.gb_n = std::min<u32>(static_cast<u32>(staged_bytes), R<u32>(buffer + 0x180));
        }
        t.Set<u64>(0x340, level.g340);
    }
    t.Put(0x800, level.gtable.data(), level.table_dwords * 8);
    GbStats().native.fetch_add(1, std::memory_order_relaxed);
    return true;
}

/// P1b: FUN_01ee4e50 on the worker: the context words and the table as GroupBindNative writes them; the staged instance data goes straight into a piece of the worker's ring
/// block and the worker's copy of the node's buffer object points at it. The uploader's double buffer and node are the engine thread's (left as they are: they only stage CPU data).
bool GroupBindWorker(Target& t, const BatchLevel& level) {
    const u64 uploader = R<u64>(level.group_object + 0xf0);
    u64 buffer = 0, staged = 0, value = 0;
    s32 staged_bytes = 0;
    if (uploader != 0) {
        const u64 node = R<u64>(uploader);
        buffer = R<u64>(node + 0x10);
        const s32 index = R<s32>(uploader + 0x28);
        value = R<u64>(uploader + 0x18 + static_cast<u64>(index & 1) * 8);
        staged = R<u64>(uploader + 8);
        staged_bytes = R<s32>(uploader + 0x10);
        if ((index != 0 && index != 1) || buffer == 0 || R<s32>(buffer + 0x124) != 1 || (R<u8>(buffer + 0x188) & 4) == 0 || R<u32>(buffer + 0x180) == 0 || staged_bytes < 0 ||
            level.g340 != buffer || t.group_clone == nullptr) {
            return false;
        }
    }
    t.Set<u64>(0x70, level.g70);
    t.Set<u64>(0x78, level.g78);
    t.Set<u32>(0x170, level.g170);
    t.Set<u32>(0x174, level.g174);
    t.Set<u64>(0x1f0, 0);
    if (uploader != 0) {
        const u32 size = R<u32>(buffer + 0x180);
        u8* block = WorkerAlloc(t, size);
        if (block == nullptr) {
            return false;
        }
        const u32 from_staged = std::min<u32>(static_cast<u32>(staged_bytes), size);
        std::memcpy(block, reinterpret_cast<const void*>(staged), from_staged);
        if (from_staged < size && value != 0) {
            std::memcpy(block + from_staged, reinterpret_cast<const void*>(value + from_staged), size - from_staged); // (the rest of the engine's half, as its copy takes it)
        }
        std::memcpy(t.group_clone, reinterpret_cast<const void*>(buffer), ParCloneBytes);
        const u64 clone = reinterpret_cast<u64>(t.group_clone);
        WriteRename(clone, reinterpret_cast<u64>(block), size, t.Get<u32>(0x47ac));
        t.Set<u64>(0x340, clone);
    }
    t.Put(0x800, level.gtable.data(), level.table_dwords * 8);
    return true;
}

// A3: FUN_0198c730, the special entry's virtual +0x48 (forward pass): the handle token and its reference count. The case with the static VERTEX_MODIFIER handle already created
// is answered natively (`TLG_GEN_NATIVE_LEAF=5`); the first use (the function-static guard still clear) is the engine's.
bool SpecialTokenNative(u32* out, u64 special, u64 material);

// A3: FUN_01c710f0 and FUN_01c70cc0 (the forward prologue), read off the decompilation (documents/tlg/native plan/stages/a3-engine-functions.md). `TLG_GEN_NATIVE_LEAF=2`
// (the lookup too): the prologue is made natively, in the live run and in the verify run's shadow. The shadow cannot take a block from the frame ring: it builds the block in
// `Verify::forward_block` and the first draw of the engine's loop compares it with the block the engine's prologue put in the ring (`NoteLoopDraw`).
bool NativePrologueOn() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_GEN_NATIVE_LEAF");
        return e != nullptr && *e != 0 && std::atoi(e) >= 2;
    }();
    return on;
}
u64 RenderDataOf(u64 render_device, u64 object) {
    const u64 data = object != 0 ? HandleLookup(render_device, object) : 0;
    return data != 0 ? R<u64>(data + 0x48) : 0;
}
/// FUN_01c70cc0(renderDev, ctx, 0, desc): ten render targets -> ctx + 0xac0 .. 0xb08.
void TargetSlotsNative(Target& t, u64 rd, u64 desc) {
    const u64 list = R<u64>(R<u64>(Global(GlobalContext)) + 0x38); // DAT_069a1b40 holds the pointer
    const u32 count = R<u32>(list + 0x88);
    const u64 array = R<u64>(list + 0x80);
    std::array<u64, 10> slot{};
    for (u32 i = 0; i < 10; ++i) {
        slot[i] = RenderDataOf(rd, i < count ? R<u64>(array + u64{i} * 8) : 0);
    }
    if (const u64 d = R<u64>(desc + 0x458); d != 0 && R<s32>(d + 0x270) == 3 && R<s32>(d + 0x258) == 7) {
        const u64 colour = R<u64>(d + 0x250), depth = R<u64>(d + 0x268);
        for (u32 i = 0; i < 7; ++i) {
            slot[i] = RenderDataOf(rd, R<u64>(colour + u64{i} * 8));
        }
        for (u32 i = 0; i < 3; ++i) {
            slot[7 + i] = RenderDataOf(rd, R<u64>(depth + u64{i} * 8));
        }
    }
    for (u32 i = 0; i < 10; ++i) {
        t.Set<u64>(0xac0 + i * 8, slot[i]);
    }
}
/// FUN_01c710f0(renderDev, a1, a2, mode).
void ForwardPrologueNative(Target& t, u64 rd, u64 a1, u64 a2, u32 mode) {
    Leaf().prologue_native.fetch_add(1, std::memory_order_relaxed);
    TargetSlotsNative(t, rd, a2);
    const u64 second = mode == 2 ? 1 : 0;
    t.Set<u64>(0xa80, R<u64>(R<u64>(R<u64>(a1 + 0x20) + 0x20) + 0x48));
    t.Set<u64>(0xa88, R<u64>(rd + 0x4448));
    t.Set<u64>(0xa90, R<u64>(rd + 0x4458));
    t.Set<u64>(0xa98, R<u64>(R<u64>(a1 + 0x38) + 0x18));
    t.Set<u64>(0xaa0, R<u64>(rd + 0x87e8 + second * 0x28));
    t.Set<u64>(0xaa8, R<u64>(rd + 0x4460 + second * 8));
    t.Set<u64>(0xab0, R<u64>(R<u64>(a1 + 0x38) + 0x38 + second * 8));
    const u64 target_desc = R<u64>(a2 + 0x448);
    const u64 object5 = R<u64>(target_desc + 0x240), object6 = R<u64>(target_desc + 0x248);
    const u64 data5 = RenderDataOf(rd, object5), data6 = RenderDataOf(rd, object6);
    std::array<u8, 0x340> local{};
    u8* block = local.data();
    if (t.live) {
        const u64 holder = R<u64>(Global(GlobalDeviceHolder));
        block = reinterpret_cast<u8*>(reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u64)>(Global(FunctionAllocFrameGarlic))(R<u64>(holder + 0x10238), 0x340, 4));
    }
    std::memcpy(block, reinterpret_cast<const void*>(rd + 0x2910), 0x2b0);
    std::memcpy(block + 0x2b0, reinterpret_cast<const void*>(rd + 0x2dc0), 0x50);
    if (data5 != 0) {
        std::memcpy(block + 0x300, reinterpret_cast<const void*>(data5 + 0x58), 0x20);
    } else {
        std::memset(block + 0x300, 0, 0x20);
    }
    if (data6 != 0) {
        std::memcpy(block + 0x320, reinterpret_cast<const void*>(data6 + 0x58), 0x20);
    } else {
        std::memset(block + 0x320, 0, 0x20);
    }
    if (t.Get<u32>(0x13088) < 2) {
        t.Set<u32>(0x13088, 2);
    }
    if (t.Get<u32>(0x13060) < 2) {
        t.Set<u32>(0x13060, 2);
    }
    if (t.live) {
        t.Set<u64>(0x13070, reinterpret_cast<u64>(block));
        t.Set<u64>(0x13048, reinterpret_cast<u64>(block));
    }
    t.Set<u64>(0xa28, data5);
    t.Set<u64>(0xa30, data6);
    t.Set<u64>(0xbb0, 0);
    t.Set<u64>(0xbb8, R<u64>(rd + 0x87c0));
}

constexpr u64 GlobalModifierGuard = 0x06888938, GlobalModifierHandle = 0x06888940; // DAT_06888938 / DAT_06888940: the function-static VERTEX_MODIFIER handle
bool SpecialTokenNative(u32* out, u64 special, u64 material) {
    if (LeafLevel() < 5) {
        return false;
    }
    if (R<u64>(special + 0x40) == 0) {
        out[0] = R<u32>(material + 0x1bc);
    } else if (R<u8>(Global(GlobalModifierGuard)) != 0) {
        out[0] = R<u32>(Global(GlobalModifierHandle));
    } else {
        return false;
    }
    HandleRefInc(out[0]);
    return true;
}

struct FBatch {
    BatchLevel level;
    u64 batch{};
    bool has_second{false};
    u64 v840{0};
    bool valid_first{false}; // the mesh's first table entry is a valid object (the other branch of the loop)
    u64 view_g70{0};
    u32 view_g170{0};
    u32 mesh_word{0}, base_vertex{0};
    u32 variant_flags{0};
    u64 special{0}; // batch + 0x48 (class 0x26dbfc8)
};

struct FStep {
    u64 record{}, material{};
};

struct FPlan {
    bool scanned{false};
    std::vector<const char*> reasons;
    u64 render_device{0}, ctx{0}, a1{0}, a2{0};
    u32 mode{0};
    u64 v_3ca8{0}, v_3d90{0};
    std::vector<FBatch> batches;
    std::vector<FStep> steps;
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* r) {
        for (const char* x : reasons) {
            if (x == r) {
                return;
            }
        }
        reasons.push_back(r);
    }
};

struct FScanArgs {
    FPlan* plan;
    u64 list;
};

void ScanForwardPass(FScanArgs& a) {
    FPlan& plan = *a.plan;
    const u64 rd = plan.render_device;
    plan.ctx = R<u64>(rd + 0x2d8);
    if (plan.ctx == 0 || a.list == 0) {
        plan.Decline("no list or no device context");
        return;
    }
    for (const u32 offset : {0x4698u, 0x4690u}) {
        if (const u64 shader = R<u64>(plan.ctx + offset); shader != 0 && R<s64>(shader + 8) <= 1) {
            plan.Decline("the context holds a shader with a single reference (releasing it destroys it)");
        }
    }
    plan.v_3ca8 = R<u64>(rd + 0x3ca8);
    plan.v_3d90 = R<u64>(rd + 0x3d90);
    const u32 base_flags = plan.mode == 2 ? 0x10000010u : 0x10000000u;
    const u64 context_global = R<u64>(Global(GlobalContext));
    const u64 default_object = R<u64>(R<u64>(context_global + 0x38) + 0x20);
    const u64 batch_array = R<u64>(a.list);
    const u32 batch_count = R<u32>(a.list + 8);
    if (batch_count != 0 && (batch_array == 0 || batch_count > 100000)) {
        plan.Decline("batch list unreadable");
        return;
    }
    for (u32 b = 0; b < batch_count; ++b) {
        const u64 batch = R<u64>(batch_array + u64{b} * 8);
        if (batch == 0) {
            plan.Decline("null batch");
            continue;
        }
        if (R<u64>(batch + 0x38) != 0) {
            continue; // the forward loop leaves batches with an instance group alone
        }
        const u64 mesh = R<u64>(batch + 0x20);
        if (mesh == 0) {
            continue;
        }
        FBatch fb;
        fb.batch = batch;
        fb.variant_flags = base_flags;
        if (const u64 special = R<u64>(batch + 0x48); special != 0) {
            if (R<u64>(special) - NativeDraw::ImageBase() + 0x1000000 != 0x26dbfc8) {
                plan.Decline("batch with a special entry of an unknown class");
                continue;
            }
            fb.special = special;
            fb.variant_flags |= 0x40;
        }
        const u64 render_data = R<u64>(mesh + 0x38);
        if (render_data == 0 || R<u64>(render_data + 0x20) == 0) {
            plan.Decline("mesh render data not created yet (the loop creates it at first use)");
            continue;
        }
        if (const u64 second = R<u64>(batch + 0x28); second != 0) {
            const u64 second_data = R<u64>(second + 0x38);
            if (second_data == 0 || R<u64>(second_data + 0x20) == 0) {
                plan.Decline("second render data not created yet");
                continue;
            }
            fb.has_second = true;
            fb.variant_flags |= 0x80000000u;
            fb.v840 = R<u64>(R<u64>(R<u64>(R<u64>(second_data + 0x20) + 0x48) + 0x38) + 0x28);
        }
        FillBatchLevel(fb.level, batch, default_object);
        const u64 inner = fb.level.inner;
        {
            const u32 count = R<u32>(inner + 0x68);
            const u64 first = count != 0 ? R<u64>(R<u64>(inner + 0x60)) : 0;
            fb.valid_first = count != 0 && first != default_object && first != 0 && R<u64>(first + 0x38) != 0;
            if (fb.valid_first) {
                fb.variant_flags |= 0x100;
            } else {
                const u64 view_object = R<u64>(R<u64>(inner + 0x48) + 0x38);
                fb.view_g70 = R<u64>(view_object + 0x20);
                fb.view_g170 = R<u32>(view_object + 0x40);
            }
        }
        fb.mesh_word = R<u32>(mesh + 0x78);
        fb.base_vertex = R<u32>(batch + 0x60);
        fb.level.first_step = static_cast<u32>(plan.steps.size());
        const u16 prim_count = R<u16>(batch + 100);
        const u64 prim_array = R<u64>(batch + 0x58);
        for (u32 p = 0; p < prim_count; ++p) {
            const u64 record = prim_array + u64{p} * 0x38;
            FStep step;
            step.record = record;
            step.material = R<u64>(record + 0x18);
            const u32 selector = R<u32>(step.material + 0x158);
            if (selector > 3) {
                plan.Decline("material + 0x158 above 3");
                continue;
            }
            const u64 object = R<u64>(step.material + 0x38);
            if (object == 0) {
                plan.Decline("material without a render object");
                continue;
            }
            const u32 count_a = R<u32>(object + 0x4f8), count_b = R<u32>(object + 0x4e8);
            if ((R<u64>(object + 0x4f0) != 0 && (count_a == 0 || count_a > 62)) || (R<u64>(object + 0x4e0) != 0 && (count_b == 0 || count_b > 72)) || (R<u64>(object + 0x440) & 0xffffffffull) >= 64) {
                plan.Decline("material with a resource count or slot index the engine asserts on");
                continue;
            }
            plan.steps.push_back(step);
        }
        fb.level.step_count = static_cast<u32>(plan.steps.size()) - fb.level.first_step;
        plan.batches.push_back(fb);
    }
    plan.scanned = true;
}

void GenerateForward(Target& t, const FPlan& plan) {
    const u64 rd = plan.render_device;
    const bool pure_pass = std::none_of(plan.batches.begin(), plan.batches.end(), [](const FBatch& b) { return b.special != 0; });
    if (t.Get<u32>(0x4480) != 4) {
        t.Set<u32>(0x4480, 4);
        t.Or8(0x447c, 1);
    }
    if (NativePrologueOn() && R<u64>(rd + 0x2d8) == plan.ctx) {
        Timed(CallId::ForwardPrologue, [&] { ForwardPrologueNative(t, rd, plan.a1, plan.a2, plan.mode); });
    } else if (t.live) {
        Timed(CallId::ForwardPrologue, [&] { reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64, u64, u64)>(Global(FunctionForwardPrologue))(rd, plan.a1, plan.a2, plan.mode); });
    }
    const auto state_words = [&](u64 entry) {
        t.Set<u64>(0x46f0, entry);
        t.Set<u64>(0x46f8, 0x3f8000003f800000ull);
        t.Set<u64>(0x4700, 0x3f8000003f800000ull);
        t.Set<u32>(0x4708, 0xffffffffu);
        t.Set<u8>(0x470c, 1);
    };
    state_words(plan.v_3ca8);
    t.Set<u64>(0x68, 0);
    bool raster_flag = false;
    u32 last_state = 0;
    for (const FBatch& fb : plan.batches) {
        const BatchLevel& level = fb.level;
        if (fb.has_second) {
            t.Set<u64>(0x840, fb.v840);
        }
        t.batch_owner = level.batch;
        t.Set<u64>(0x580, level.value_580);
        if (!fb.valid_first) {
            t.Set<u64>(0x70, fb.view_g70);
            t.Set<u32>(0x170, fb.view_g170);
            t.Set<u32>(0x1f0, 0);
            t.Set<u64>(0x58, level.view_58);
            t.Set<u64>(0x60, 0x39);
            t.batch_owner = 0;
            t.Set<u64>(0x68, plan.v_3d90);
        } else {
            t.Set<u64>(0x68, 0);
            t.Set<u64>(0x58, level.view_58);
            t.batch_owner = 0;
            t.Set<u64>(0x60, 0x39);
        }
        t.batch_owner = level.batch;
        t.Set<u64>(0x288, level.pass_object_288);
        t.Set<u64>(0x308, level.pass_object_288);
        t.batch_owner = 0;
        for (u32 i = level.table_count; i-- > 0;) {
            t.table_entry = level.table_entry[i];
            t.Set<u64>(0x6c0 + level.table[i].first * 8, level.table[i].second);
            t.table_entry = 0;
        }
        for (u32 s = level.first_step; s < level.first_step + level.step_count; ++s) {
            const FStep& step = plan.steps[s];
            const u32 selector = R<u32>(step.material + 0x158);
            const bool cached_object = R<u64>(step.material + 0x48) != 0;
            u32 state = 0;
            if (selector == 3) {
                state = cached_object ? 0xf : 4;
            } else if (selector == 2) {
                state = cached_object ? 0xe : 3;
            } else if (selector == 1) {
                state = cached_object ? 0xd : 2;
            }
            if (state != last_state) {
                state_words(R<u64>(rd + 0x3ca8 + u64{state} * 8));
                last_state = state;
            }
            Timed(CallId::ResourceSet, [&] { ResourceSetStep(t, rd, step.material, fb.special, pure_pass); });
            // the handle sequence of the loop: the material's handle is counted, a special entry replaces it by the handle its virtual +0x48 returns (counted by the virtual), the variant search
            // gets the resulting handle
            u64 token = R<u32>(step.material + 0x1bc);
            HandleRefInc(token);
            if (fb.special != 0) {
                std::array<u32, 2> out{0, 0};
                Timed(CallId::SpecialToken, [&] {
                    if (!SpecialTokenNative(out.data(), fb.special, step.material)) {
                        reinterpret_cast<void(PS4_SYSV_ABI*)(void*, u64, u64)>(R<u64>(R<u64>(fb.special) + 0x48))(out.data(), fb.special, step.material);
                    }
                });
                u64 released = token;
                if (token != out[0]) {
                    HandleRefInc(out[0]);
                    if (token != 0) {
                        HandleRefDec(token);
                    }
                    released = out[0];
                    token = out[0];
                }
                if (released != 0) {
                    HandleRefDec(released);
                }
            }
            const bool raster = (R<u32>(step.material + 0xd4) & 0x26) != 0;
            if (raster != raster_flag) {
                raster_flag = raster;
                const u64 object = R<u64>(rd + (raster ? 0x3bf8 : 0x3bf0));
                if (t.Get<u64>(0x46c8) != object) {
                    t.Set<u64>(0x46c8, object);
                    t.Set<u8>(0x46d8, 1);
                }
            }
            HandleRefInc(token);
            std::array<s32, 2> token_slot{static_cast<s32>(token), 0};
            const u64 variant = Timed(CallId::FindVariant, [&] { return FindVariant(rd + 1000, step.material, fb.variant_flags, 0, fb.mesh_word, reinterpret_cast<u32*>(token_slot.data())); });
            if (token != 0) {
                HandleRefDec(token);
            }
            if (variant != 0) {
                SetShader(t, 0x4698, R<u64>(variant + 8), 1);
                SetShader(t, 0x4690, R<u64>(variant + 0x18), 2);
                t.slot_material = step.material;
                BindMaterial(t, step.material);
                t.slot_material = 0;
                const s32 triangles = R<s32>(step.record + 0x14);
                IssueDraw(t, 0, static_cast<u32>(triangles * 3), R<u32>(step.record + 0x10), (fb.variant_flags & 0x100) == 0 ? fb.base_vertex : 0);
                if (t.live) {
                    W<s32>(rd + 0x10, R<s32>(rd + 0x10) + 1);
                    W<s32>(rd + 0x14, R<s32>(rd + 0x14) + triangles);
                } else {
                    ++t.draws;
                    t.triangles += triangles;
                }
            }
            if (token != 0) {
                HandleRefDec(token);
            }
        }
    }
    {
        const std::array<u8, 0x400> zeros{};
        t.Put(0x980, zeros.data(), 0x400);
        t.Put(0x580, zeros.data(), 0x400);
    }
    const u64 object = R<u64>(rd + 0x3bf0);
    if (t.Get<u64>(0x46c8) != object) {
        t.Set<u64>(0x46c8, object);
        t.Set<u8>(0x46d8, 1);
    }
}

struct FConfig {
    bool on{false};
};
const FConfig& FCfg() {
    static const FConfig config = [] {
        FConfig c;
        const char* e = std::getenv("TLG_FORWARD_GEN");
        c.on = e != nullptr ? *e != '0' : GeneratorsDefaultOn(); // off under TLG_VKB (note 57); before: default on since 2026-10-27 (the owner's play test); TLG_FORWARD_GEN=0 turns it off
        return c;
    }();
    return config;
}

struct FStats {
    ViewStats v;
    u64 eligible{0};
};
FStats& FS() {
    static FStats stats;
    return stats;
}

// ------------------------------------------------------------------------------------------------------------------------------- the decal pass loop (A7)
// FUN_01c71680(renderDev, view, a3, a4, desc, flags, type, width, height) draws the decals of one view type: `desc + 0x480 + type * 0x18` batches (list at + 0x478), every primitive of a batch
// is one decal object (record + 0x18) drawn with an inlined `DrawIndexAuto`. The loop writes the pass constants once (the block at renderDev + 0x39f0 .. 0x3a3c and 0x3b40 .. 0x3b7c), then per
// primitive the block at renderDev + 0x3990 .. 0x3b3c from the decal object, updates the constant buffer at renderDev + 0x3b80 from the whole block (UpdateSubresource, the context's virtual +
// 0xd0), puts five render data views into the context (0xa38 .. 0xa58), asks the variant cache for the shader pair and draws. Decompiled in documents/tlg/native plan/stages/a7-decal-pass.md.
// The generator makes the same context writes (live: into the device context; verify: into the shadow copy, compared word by word at every draw), the same block (compared byte by byte at
// every draw against the block the engine's loop left) and the same draws. What stays the engine's: the handle lookups that create render data at first use (`FUN_01c59060`), the variant
// search, the buffer update (UpdateSubresource) and the index buffer call (`FUN_02488430`).
constexpr u64 DecalBegin = 0x01c71680, DecalEnd = 0x01c72a20; // the loop and the next function
constexpr u64 DecalDrawSite = 0x01c727a7;                     // the return address of its inlined PrepareDraw
constexpr u64 FunctionSetIndexBuffer = 0x02488430;            // FUN_02488430(ctx, index buffer): emits a SET_INDEX_BASE when the base changed (the loop passes 0)
constexpr u32 DecalStageBase = 0x3990;                        // renderDev + 0x3990 .. + 0x3b80: the constant block the buffer at renderDev + 0x3b80 is updated from

struct DStep {
    u64 record{}, decal{};
    u32 index{};
};
struct DBatch {
    u64 batch{};
    bool set_580{false};
    u64 value_580{};
    u32 first_step{0}, step_count{0};
};
struct DPlan {
    bool scanned{false};
    bool no_work{false}; // desc + 0x480 + type * 0x18 is zero: the function does nothing at all
    bool any_mesh{false};
    std::vector<const char*> reasons;
    u64 render_device{0}, ctx{0}, view{0}, a3{0}, a4{0}, desc{0};
    u32 flags{0}, width{0}, height{0};
    s32 type{0};
    std::vector<DBatch> batches;
    std::vector<DStep> steps;
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* r) {
        for (const char* x : reasons) {
            if (x == r) {
                return;
            }
        }
        reasons.push_back(r);
    }
};

struct DScanArgs {
    DPlan* plan;
};

void ScanDecalPass(DScanArgs& a) {
    DPlan& plan = *a.plan;
    const u64 rd = plan.render_device;
    const u64 desc = plan.desc;
    if (plan.type < 0 || plan.type > 64) {
        plan.Decline("view type out of range");
        return;
    }
    const u64 list_count_at = desc + 0x480 + u64{static_cast<u32>(plan.type)} * 0x18, list_array_at = desc + 0x478 + u64{static_cast<u32>(plan.type)} * 0x18;
    const u32 count = R<u32>(list_count_at);
    if (count == 0) {
        plan.no_work = true;
        plan.scanned = true;
        return;
    }
    plan.ctx = R<u64>(rd + 0x2d8);
    if (plan.ctx == 0) {
        plan.Decline("no device context");
        return;
    }
    for (const u32 offset : {0x4698u, 0x4690u}) {
        if (const u64 shader = R<u64>(plan.ctx + offset); shader != 0 && R<s64>(shader + 8) <= 1) {
            plan.Decline("the context holds a shader with a single reference (releasing it destroys it)");
        }
    }
    if (!NativeDraw::InlineAutoDrawTagged(1)) {
        plan.Decline("native draw execution is off (the draw hook does not take the inlined auto draw)");
    }
    if (R<u64>(desc + 0x298) == 0 || R<u64>(desc + 0x448) == 0 || R<u64>(R<u64>(desc + 0x448) + 0x2b0) == 0) {
        plan.Decline("pass description without its constants object or target description");
        return;
    }
    const u64 array = R<u64>(list_array_at);
    if (array == 0 || count > 100000) {
        plan.Decline("batch list unreadable");
        return;
    }
    u64 last_mesh = 0;
    for (u32 b = 0; b < count; ++b) {
        const u64 batch = R<u64>(array + u64{b} * 8);
        if (batch == 0) {
            plan.Decline("null batch");
            continue;
        }
        const u64 mesh = R<u64>(batch + 0x20);
        DBatch db;
        db.batch = batch;
        bool enter = false;
        if (mesh == last_mesh) {
            if (mesh == 0) {
                plan.Decline("first batch without a mesh");
                continue;
            }
            enter = true;
        } else if (mesh != 0) {
            const u64 render_data = R<u64>(mesh + 0x38);
            if (render_data == 0) {
                plan.Decline("mesh render data not created yet (the loop creates it at first use)");
                continue;
            }
            const u64 inner = R<u64>(render_data + 0x20);
            if (inner == 0 || R<u64>(inner + 0x48) == 0 || R<u64>(R<u64>(inner + 0x48) + 0x38) == 0) {
                plan.Decline("mesh render data without its inner object");
                continue;
            }
            if (R<u64>(inner + 0x50) == 0) {
                db.set_580 = true;
                db.value_580 = R<u64>(R<u64>(R<u64>(inner + 0x48) + 0x38) + 0x28);
                last_mesh = mesh;
                enter = true;
            }
        }
        if (!enter) {
            continue; // the loop leaves the batch alone
        }
        db.first_step = static_cast<u32>(plan.steps.size());
        const u16 prim_count = R<u16>(batch + 100);
        const u64 prim_array = R<u64>(batch + 0x58);
        for (u32 p = 0; p < prim_count; ++p) {
            const u64 record = prim_array + u64{p} * 0x38;
            if (R<s32>(record + 0x14) == 0) {
                continue;
            }
            const u64 decal = R<u64>(record + 0x18);
            if (decal == 0) {
                continue;
            }
            if (R<u64>(decal + 0x1e8) == 0 || R<u64>(decal + 0x230) == 0 || R<u64>(decal + 0x248) == 0 || R<u64>(decal + 0x218) == 0 || (R<u32>(decal + 0x2b8) >= 2 && R<u64>(decal + 0x200) == 0)) {
                plan.Decline("decal object without one of its resource lists");
                continue;
            }
            plan.steps.push_back({record, decal, p});
        }
        db.step_count = static_cast<u32>(plan.steps.size()) - db.first_step;
        plan.batches.push_back(db);
    }
    plan.any_mesh = last_mesh != 0;
    plan.scanned = true;
}

void GenerateDecal(Target& t, const DPlan& plan) {
    const u64 rd = plan.render_device, desc = plan.desc;
    const bool wide_state = plan.type == 10 || plan.type == 7; // the loop's two state object families
    std::array<u8, DecalStageBytes> shadow_stage{};
    u8* stage = shadow_stage.data();
    if (t.live) {
        stage = reinterpret_cast<u8*>(rd + DecalStageBase);
    } else {
        std::memcpy(stage, reinterpret_cast<const void*>(rd + DecalStageBase), DecalStageBytes);
    }
    const auto put32 = [&](u32 offset, u32 value) { std::memcpy(stage + (offset - DecalStageBase), &value, 4); };
    const auto putf = [&](u32 offset, float value) { std::memcpy(stage + (offset - DecalStageBase), &value, 4); };
    const auto put_from = [&](u32 offset, u64 source, u32 bytes) { std::memcpy(stage + (offset - DecalStageBase), reinterpret_cast<const void*>(source), bytes); };
    const auto raster_object = [&](u64 object) {
        if (t.Get<u64>(0x46c8) != object) {
            t.Set<u64>(0x46c8, object);
            t.Set<u8>(0x46d8, 1);
        }
    };
    const auto state_words = [&](u64 entry) {
        t.Set<u64>(0x46f0, entry);
        t.Set<u64>(0x46f8, 0x3f8000003f800000ull);
        t.Set<u64>(0x4700, 0x3f8000003f800000ull);
        t.Set<u32>(0x4708, 0xffffffffu);
        t.Set<u8>(0x470c, 1);
    };
    // ----- prologue
    raster_object(R<u64>(rd + 0x3bf8));
    state_words(R<u64>(rd + (wide_state ? 0x3cb0 : 0x3d30)));
    t.Set<u64>(0x46e0, R<u64>(rd + 0x3c40));
    t.Set<u32>(0x46e8, 0);
    t.Set<u8>(0x46ec, 1);
    t.Set<u64>(0x68, 0);
    const u64 object_list = R<u64>(R<u64>(Global(GlobalContext)) + 0x38);
    HandleLookup(rd, R<u64>(object_list + 0x50)); // the loop calls it for its side effect (the render data of the default object)
    u64 last_view_data = 0; // the render data the last view_of call read its view from (a word the engine's other threads can replace: tagged for the verify compare)
    const auto view_of = [&](u64 object) -> u64 {
        const u64 data = HandleLookup(rd, object);
        last_view_data = data;
        return data != 0 ? R<u64>(data + 0x48) : 0;
    };
    const u64 desc_targets = R<u64>(desc + 0x448);
    const u64 target_set = R<u64>(desc_targets + 0x2b0);
    const u64 w_988 = R<u64>(rd + 0x41b0);
    const u64 w_990 = R<u64>(R<u64>(R<u64>(plan.view + 0x40) + 0x90) + 0x48);
    const u64 w_998 = view_of(R<u64>(target_set + 0x140));
    const u64 w_9a0 = view_of(R<u64>(target_set + 0x148));
    u64 w_9c8 = 0;
    if (const u64 list = R<u64>(plan.view + 0x50); list != 0) {
        w_9c8 = R<u64>(R<u64>(list + 0x18 + u64{R<u32>(list + 0x14)} * 8) + 0x48);
    }
    const std::array<u64, 5> w_9d0 = {R<u64>(rd + 0x4448), R<u64>(rd + 0x4458), R<u64>(rd + 0x8898), R<u64>(rd + 0x88a0), R<u64>(rd + 0x4060)};
    const std::array<u64, 5> w_a10 = {R<u64>(rd + 0x87e8), R<u64>(rd + 0x4460), R<u64>(rd + 0x88a8), R<u64>(rd + 0x4080), R<u64>(R<u64>(R<u64>(rd + 0x3a8) + 0x50) + 0x48)};
    const std::array<u64, 4> w_9a8 = {R<u64>(rd + 0x4188), R<u64>(rd + 0x4190), R<u64>(rd + 0x4198), R<u64>(rd + 0x41a0)};
    // the pass constants
    const u64 constants = R<u64>(desc + 0x298);
    const float depth_a = R<float>(constants + 0x1f8), depth_b = R<float>(constants + 0x1fc);
    const std::array<u64, 6> q = {R<u64>(constants + 0x10), R<u64>(constants + 0x18), R<u64>(constants + 0x20), R<u64>(constants + 0x28), R<u64>(constants + 0x30), R<u64>(constants + 0x38)};
    const u32 one = 0x3f800000u;
    put32(0x39f0, static_cast<u32>(q[0]));
    put32(0x39f4, static_cast<u32>(q[0] >> 32));
    put32(0x39f8, static_cast<u32>(q[1]));
    put32(0x39fc, one);
    put32(0x3a00, static_cast<u32>(q[2]));
    put32(0x3a04, static_cast<u32>(q[2] >> 32));
    put32(0x3a08, static_cast<u32>(q[3]));
    put32(0x3a0c, one);
    put32(0x3a10, static_cast<u32>(q[4]));
    put32(0x3a14, static_cast<u32>(q[4] >> 32));
    put32(0x3a18, static_cast<u32>(q[5]));
    put32(0x3a1c, one);
    putf(0x3a20, depth_a);
    putf(0x3a24, depth_b);
    putf(0x3a28, 1.0f / depth_a);
    putf(0x3a2c, (depth_a - depth_b) / (depth_b * depth_a));
    put_from(0x3b40, desc + 0x210, 0x40);
    putf(0x3a30, static_cast<float>(plan.width));
    putf(0x3a34, static_cast<float>(plan.height));
    putf(0x3a38, 1.0f / static_cast<float>(plan.width));
    putf(0x3a3c, 1.0f / static_cast<float>(plan.height));
    {
        const u64 buffer_a = R<u64>(rd + 0x3980), buffer_b = R<u64>(rd + 0x3b80);
        t.Set<u64>(0x2e0, buffer_a);
        t.Set<u64>(0x2e8, buffer_b);
        t.Set<u64>(0x360, buffer_a);
        t.Set<u64>(0x368, buffer_b);
    }
    const float scale = (plan.flags & 2) != 0 ? 1.0f : 0.0f;
    if (NativePrologueOn() && R<u64>(rd + 0x2d8) == plan.ctx) {
        Timed(CallId::ForwardPrologue, [&] { ForwardPrologueNative(t, rd, plan.view, desc, 0); });
    } else if (t.live) {
        Timed(CallId::ForwardPrologue, [&] { reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64, u64, u64)>(Global(FunctionForwardPrologue))(rd, plan.view, desc, 0); });
    }
    const u64 desc_targets_after = R<u64>(desc + 0x448);
    const u64 w_9f8 = R<u64>(desc_targets_after + 0x240) != 0 ? view_of(R<u64>(desc_targets_after + 0x240)) : 0;
    const u64 w_a00 = R<u64>(desc_targets_after + 0x248) != 0 ? view_of(R<u64>(desc_targets_after + 0x248)) : 0;
    t.Set<u64>(0x980, plan.a3);
    t.Set<u64>(0x988, w_988);
    t.Set<u64>(0x990, w_990);
    t.Set<u64>(0x998, w_998);
    t.Set<u64>(0x9a0, w_9a0);
    for (u32 i = 0; i < 4; ++i) {
        t.Set<u64>(0x9a8 + i * 8, w_9a8[i]);
    }
    t.Set<u64>(0x9c8, w_9c8);
    for (u32 i = 0; i < 5; ++i) {
        t.Set<u64>(0x9d0 + i * 8, w_9d0[i]);
    }
    t.Set<u64>(0x9f8, w_9f8);
    t.Set<u64>(0xa00, w_a00);
    t.Set<u64>(0xa08, plan.a4);
    for (u32 i = 0; i < 5; ++i) {
        t.Set<u64>(0xa10 + i * 8, w_a10[i]);
    }
    // ----- the batches
    u32 last_state = 1;
    for (const DBatch& db : plan.batches) {
        if (db.set_580) {
            t.batch_owner = db.batch;
            t.Set<u64>(0x580, db.value_580);
            t.batch_owner = 0;
        }
        for (u32 s = db.first_step; s < db.first_step + db.step_count; ++s) {
            const DStep& step = plan.steps[s];
            const u64 d = step.decal, record = step.record;
            const u64 p = step.index;
            const u64 image_object = R<u64>(R<u64>(d + 0x1e8) + p * 8);
            const u64 view_0 = view_of(image_object);
            const u64 data_0 = last_view_data;
            const u64 view_1 = view_of(R<u64>(R<u64>(d + 0x230) + p * 8));
            const u64 data_1 = last_view_data;
            const u64 view_2 = view_of(R<u64>(R<u64>(d + 0x248) + p * 8));
            const u64 data_2 = last_view_data;
            if (view_0 == 0) {
                continue;
            }
            u64 view_d = view_0, data_d = data_0;
            if (R<u32>(d + 0x2b8) >= 2) {
                const u64 object = R<u64>(R<u64>(d + 0x200) + p * 8);
                if (object == 0) {
                    continue;
                }
                view_d = view_of(object);
                data_d = last_view_data;
            }
            const u64 data_e = HandleLookup(rd, R<u64>(R<u64>(d + 0x218) + p * 8));
            if (data_e == 0) {
                continue;
            }
            const u64 view_e = R<u64>(data_e + 0x48);
            if (view_e == 0) {
                continue;
            }
            // the state object the primitive's flag byte selects
            const u32 flag_byte = R<u8>(record + 0x21);
            if (flag_byte != last_state) {
                u64 object;
                if (wide_state) {
                    object = (flag_byte & 1) != 0 ? R<u64>(rd + 0x3cb0) : (flag_byte & 2) != 0 ? R<u64>(rd + 0x3b90) : (flag_byte & 4) != 0 ? R<u64>(rd + 0x3b98) : R<u64>(rd + 0x3ca8);
                } else {
                    object = (flag_byte & 1) != 0 ? R<u64>(rd + 0x3d30) : (flag_byte & 2) != 0 ? R<u64>(rd + 0x3d38) : (flag_byte & 4) != 0 ? R<u64>(rd + 0x3d40) : R<u64>(rd + 0x3d48);
                }
                state_words(object);
                last_state = flag_byte;
            }
            // the constant block
            put_from(0x39c0, d + 0x110, 16);
            const u64 target_view = R<u64>(image_object + 0x68);
            const u32 size_a = target_view != 0 ? R<u32>(target_view + 0x10) : R<u32>(image_object + 0x84);
            const u32 size_b = target_view != 0 ? R<u32>(target_view + 0x14) : R<u32>(image_object + 0x88);
            putf(0x39d0, static_cast<float>(R<s32>(d + 0x2cc)));
            putf(0x39d4, static_cast<float>(size_a));
            putf(0x39d8, static_cast<float>(size_b));
            put32(0x39dc, 0);
            const u32 vertices = R<u32>(d + (R<s32>(d + 0x2c4) != 0 ? 0x2c0 : 0x2b4));
            const float radius = R<float>(d + 0xa4);
            put32(0x3a40, R<u32>(d + 0x2c8));
            putf(0x3a44, static_cast<float>(vertices));
            putf(0x3a48, radius * radius);
            put32(0x3a4c, R<u32>(d + 0x5bc));
            put_from(0x3a50, d + 0x540, 16);
            put32(0x3a60, R<u32>(d + 0x5c0));
            put32(0x3a64, R<u32>(d + 0x5c4));
            put32(0x3a68, 0);
            put32(0x3a6c, 0);
            put_from(0x3a90, d + 0x2d0, 0x40);
            const u32 first = R<u32>(record + 0x10);
            put32(0x3990, R<u32>(d + 0x29c));
            put32(0x3994, R<u32>(d + 0x284));
            put32(0x3998, R<u32>(d + 0x294));
            put32(0x399c, R<u32>(d + 0x298));
            const u64 holder = R<u64>(d + 0x1d8);
            const s32 holder_value = holder != 0 ? R<s32>(holder + 0x24) : 0;
            const u64 word_288 = R<u64>(d + 0x288);
            put32(0x39a0, static_cast<u32>(word_288));
            put32(0x39a4, static_cast<u32>(word_288 >> 32));
            putf(0x39a8, static_cast<float>(holder_value));
            put32(0x39ac, R<u32>(d + 0x2a0));
            put32(0x39b0, R<u32>(target_set + 0x2d4));
            put32(0x39b4, R<u32>(target_set + 0x2d8));
            put32(0x39b8, R<u32>(target_set + 0x2dc));
            put32(0x39bc, R<u32>(d + 0x290));
            put32(0x39e0, R<u32>(d + 0x278));
            put32(0x39e4, R<u32>(d + 0x27c));
            put32(0x39e8, R<u32>(d + 0x280));
            putf(0x39ec, static_cast<float>(first));
            put32(0x3a70, R<u32>(d + 0x260));
            put32(0x3a74, R<u32>(d + 0x264));
            put32(0x3a78, R<u32>(d + 0x270));
            put32(0x3a7c, R<u32>(d + 0x274));
            putf(0x3a80, scale * R<float>(d + 0x268));
            putf(0x3a84, scale * R<float>(d + 0x26c));
            put32(0x3a88, R<u32>(d + 0x2a8));
            put32(0x3a8c, R<u32>(d + 0x2a4));
            put32(0x3ad0, R<u32>(d + 0x550));
            put32(0x3ad4, R<u32>(d + 0x554));
            put32(0x3ad8, R<u32>(d + 0x558));
            put32(0x3adc, 0);
            put32(0x3ae0, R<u32>(d + 0x55c));
            put32(0x3ae4, R<u32>(d + 0x560));
            put32(0x3ae8, R<u32>(d + 0x564));
            put32(0x3aec, 0);
            put_from(0x3af0, d + 0x570, 0x40);
            put32(0x3b30, first);
            put32(0x3b34, 0);
            put32(0x3b38, 0);
            put32(0x3b3c, 0);
            if (t.live) {
                Timed(CallId::DecalUpdate, [&] {
                    reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64)>(R<u64>(R<u64>(plan.ctx) + 0xd0))(plan.ctx, R<u64>(rd + 0x3b80), 0, 0, rd + DecalStageBase, 0, 0);
                });
            } else {
                t.stages.push_back(shadow_stage);
                t.stage_pending = static_cast<s32>(t.stages.size()) - 1;
            }
            t.direct_offset = 0x48;
            t.direct_object = data_0;
            t.Set<u64>(0xa38, view_0);
            t.direct_object = data_d;
            t.Set<u64>(0xa40, view_d);
            t.direct_object = data_e;
            t.Set<u64>(0xa48, view_e);
            t.direct_object = data_1;
            t.Set<u64>(0xa50, view_1);
            t.direct_object = data_2;
            t.Set<u64>(0xa58, view_2);
            t.direct_object = 0;
            t.Set<u64>(0x58, 0);
            t.Set<u64>(0x60, 0x39);
            if (t.Get<u32>(0x4480) != 5) {
                t.Set<u32>(0x4480, 5);
                t.Or8(0x447c, 1);
            }
            std::array<s32, 2> token{0, 0};
            const u64 variant = Timed(CallId::FindVariant, [&] {
                return reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u32, u32, u32, void*, u64)>(Global(FunctionFindVariant))(rd + 1000, d, 0, 0, 0, token.data(), 0);
            });
            if (variant == 0) {
                continue;
            }
            SetShader(t, 0x4698, R<u64>(variant + 8), 1);
            SetShader(t, 0x4690, R<u64>(variant + 0x18), 2);
            const s32 instances = R<s32>(record + 0x14);
            if (t.Get<s32>(0x4724) != instances) {
                t.Set<s32>(0x4724, instances);
            }
            if (t.live) {
                reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64)>(Global(FunctionSetIndexBuffer))(plan.ctx, 0);
            } else if (t.Get<u32>(0x278) != 0x200) {
                // FUN_02488430(ctx, 0): the index buffer base and type the loop last set go back to zero (the packets it emits are not modelled)
                if (t.Get<u64>(0x4470) != 0) {
                    t.Set<u64>(0x4470, 0);
                }
                if (t.Get<s32>(0x4478) != 0) {
                    t.Set<s32>(0x4478, 0);
                }
            }
            t.Set<u64>(0x479c, 0);
            if (t.live) {
                Timed(CallId::DrawIssue, [&] { NativeDraw::IssueInlineAutoDraw(plan.ctx, vertices, DecalDrawSite); });
            } else if (NativeDraw::InlineAutoDrawTagged(vertices)) {
                t.marks.push_back({t.arena.size(), 2, vertices, 0, static_cast<u64>(instances), t.Get<u64>(0x4698), t.Get<u64>(0x4690)});
                if (t.stage_pending >= 0) {
                    t.marks.back().stage = t.stage_pending;
                }
            }
            t.stage_pending = -1;
            if (t.live) {
                W<s32>(rd + 0x10, R<s32>(rd + 0x10) + 1);
                W<s32>(rd + 0x14, R<s32>(rd + 0x14) + instances * 2);
            } else {
                ++t.draws;
                t.triangles += instances * 2;
            }
        }
    }
    // ----- epilogue
    {
        const std::array<u8, 0x400> zeros{};
        t.Put(0x580, zeros.data(), 0x400);
        t.Put(0x980, zeros.data(), 0x400);
        if (plan.any_mesh) {
            t.Set<u64>(0x580, 0);
            t.Put(0x980, zeros.data(), 0xe0);
            raster_object(R<u64>(rd + 0x3bf0));
        }
    }
}

struct DConfig {
    bool on{false};
};
const DConfig& DCfg() {
    static const DConfig config = [] {
        DConfig c;
        const char* e = std::getenv("TLG_DECAL_GEN");
        c.on = e != nullptr ? *e != '0' : GeneratorsDefaultOn(); // off under TLG_VKB (note 57); before: default on since 2026-10-27 (the owner's play test); TLG_DECAL_GEN=0 turns it off
        return c;
    }();
    return config;
}

struct DStats {
    ViewStats v;
    u64 eligible{0};
    u64 loops_without_work{0};
};
DStats& DS() {
    static DStats stats;
    return stats;
}

// ------------------------------------------------------------------------------------------------------------------------------- the skinning loop (B3)
// FUN_01c93730(skin system S, scene F, async flag): for every skinned object of the scene (the list at F + 0x29e00) one skinning dispatch (CS_Skinning, 24 variants), then, for the objects
// with the normal flag, three more (CS_FixVertexNormal: clear, calculate, normalize). It talks to the device context only through virtual calls that are table writes
// (SetShader: ctx + 0x4670; SetConstantBuffers: ctx + 0x500; SetShaderResources: ctx + 0x1980; SetUnorderedAccessViews: ctx + 0x2098) and UpdateSubresource of its own constant block (S + 0x50),
// plus the dispatches. The generator makes the same table writes, the same constant blocks (compared byte by byte at every dispatch in the verify run), the same dispatches
// (through the dispatch hook, from the loop's own call sites). Decompiled and annotated in documents/tlg/native plan/stages/b3-skinning.md.
constexpr u64 SkinBegin = 0x01c93730, SkinEnd = 0x01c95550; // the loop and the next function
constexpr u64 GlobalDevice = 0x0689eed8;                    // DAT_0689eed8: the render device (the value)
constexpr u64 SkinSiteDispatch = 0x01c94f44, SkinSiteClear = 0x01c951ea, SkinSiteCalculate = 0x01c952f3, SkinSiteNormalize = 0x01c95426;
constexpr u32 SkinBlockOffset = 0x50, SkinBlockBytes = 0x120; // the constant block UpdateSubresource sends from S + 0x50 (the buffer's byte width is checked at the scan)

struct SkinNode {
    u64 node{};
    bool active{false}; // objects 1, 2 and 8 exist: the loop makes bone matrices and a dispatch for it
    u64 rd1{}, rd2{}, rd3{}, rd8{}, rd5{}, rd35{};
    u64 inner1{};
    u64 v20{}, v21{}, v59{}, v60{}, v63{}, v64{}, v65{}, v66{}, v69{}, v180{}, v50{};
    bool has_object4{false}, b70{false}, b74{false}, b75{false};
    u64 v_a0{0}; // SRV 8 (only with the object at +0x108)
    s32 count46{0};
    u32 variant{0};
    bool normals{false};
    u64 normals_rd{0}; // render data of the object at node + 0x20 (pass 3)
};

struct SkinPlan {
    bool scanned{false};
    bool no_work{false};
    std::vector<const char*> reasons;
    u64 device{0}, ctx{0}, system{0}, scene{0}, def{0}, context_global{0};
    u64 camera{0}, camera_object{0}, view{0};
    bool async{false};
    bool run_normals{false};
    std::vector<SkinNode> nodes;
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* r) {
        for (const char* x : reasons) {
            if (x == r) {
                return;
            }
        }
        reasons.push_back(r);
    }
};

struct SkinScanArgs {
    SkinPlan* plan;
};

void ScanSkin(SkinScanArgs& a) {
    SkinPlan& p = *a.plan;
    p.device = R<u64>(Global(GlobalDevice));
    p.context_global = R<u64>(Global(GlobalContext));
    p.def = R<u64>(R<u64>(p.context_global + 0x38) + 0x20);
    // the graphics device context, or the async compute context (its tables differ: shader + 0x1d10, constant buffers + 0x21f8, shader resources + 0x1d18, unordered access + 0x2118)
    p.ctx = p.async ? R<u64>(R<u64>(p.device + 0x2c8) + 0x30258) : R<u64>(p.device + 0x2d8);
    const u64 first = R<u64>(p.scene + 0x29e00);
    if (first == 0) {
        p.no_work = true;
        p.scanned = true;
        return;
    }
    if (p.ctx == 0) {
        p.Decline("no device context");
        return;
    }
    const u64 S = p.system;
    for (const u32 offset : {0x1d0u, 0x230u, 0x340u, 0x360u, 0x380u}) {
        if (R<u64>(S + offset) == 0) {
            p.Decline("the system's constant buffers are not created");
            return;
        }
    }
    if (R<u32>(R<u64>(S + 0x1d0) + 0x180) < SkinBlockBytes) {
        p.Decline("the constant buffer is smaller than the block");
        return;
    }
    p.camera_object = R<u64>(R<u64>(p.context_global + 0x108) + 0x2c8);
    p.camera = R<u64>(R<u64>(R<u64>(p.context_global + 0x108) + 0x2e0) + 0x2c0);
    if (p.camera == 0 || p.camera_object == 0) {
        p.Decline("no camera");
        return;
    }
    const u64 slot_count = R<u32>(p.scene + 0x29dd0);
    p.view = 0;
    if (slot_count != 0) {
        u64 slot = p.scene + 0x5d0;
        for (u64 k = 1;; ++k) {
            const bool taken = R<s32>(R<u64>(slot + 0x290) + 0x40) != 0;
            if (!taken) {
                p.view = slot;
                break;
            }
            if (slot_count <= k) {
                break;
            }
            slot += 0xa60;
        }
    }
    const auto render_data = [&](u64 object) { return object != 0 ? R<u64>(object + 0x38) : 0ull; };
    const auto zero_or_default = [&](u64 object) { return object == p.def ? 0ull : object != 0 ? R<u64>(object + 0x38) : 0ull; };
    u32 guard = 0;
    for (u64 node = first; node != 0 && guard < 100000; node = R<u64>(node), ++guard) {
        SkinNode n;
        n.node = node;
        const u64 o1 = R<u64>(node + 8), o2 = R<u64>(node + 0x10), o3 = R<u64>(node + 0x18), o4 = R<u64>(node + 0x20), o5 = R<u64>(node + 0x28), o8 = R<u64>(node + 0x40),
                  o35 = R<u64>(node + 0x118);
        if (o1 != 0) {
            n.rd1 = render_data(o1);
            if (n.rd1 == 0) {
                p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
            }
        }
        if (o1 != 0 && o2 != 0) {
            n.rd2 = render_data(o2);
            if (n.rd2 == 0) {
                p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
            }
            if (o3 != 0) {
                n.rd3 = render_data(o3);
                if (n.rd3 == 0) {
                    p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                }
            }
            if (o8 != 0) {
                n.rd8 = render_data(o8);
                if (n.rd8 == 0) {
                    p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                }
            }
        }
        n.active = o1 != 0 && o2 != 0 && o8 != 0;
        if (n.active && !p.reasons.empty()) {
            p.nodes.push_back(n);
            continue;
        }
        if (n.active) {
            if (o5 != 0) {
                n.rd5 = render_data(o5);
                if (n.rd5 == 0) {
                    p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                }
            }
            if (o35 != 0) {
                n.rd35 = render_data(o35);
                if (n.rd35 == 0) {
                    p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                }
            }
            u64 object4_data = 0;
            if (o4 != 0) {
                object4_data = render_data(o4);
                if (object4_data == 0) {
                    p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                }
            }
            if (!p.reasons.empty()) {
                p.nodes.push_back(n);
                continue;
            }
            n.inner1 = R<u64>(n.rd1 + 0x20);
            n.has_object4 = o4 != 0;
            u64 inner4 = 0;
            if (n.has_object4) {
                inner4 = R<u64>(object4_data + 0x20);
                const u32 count = R<u32>(inner4 + 0x68);
                if (count >= 9) {
                    const u64 array = R<u64>(inner4 + 0x60);
                    n.v69 = zero_or_default(R<u64>(array + 0x40));
                    if (count >= 0x24) {
                        n.v180 = zero_or_default(R<u64>(array + 0x118));
                    }
                    if (count >= 0xb) {
                        const u64 x = R<u64>(array + 0x50);
                        n.v50 = (x == p.def || x == 0) ? 0 : R<u64>(x + 0x38);
                    }
                }
            }
            if (n.inner1 == 0 || R<u64>(n.rd2 + 0x20) == 0) {
                p.Decline("render data without its inner object");
                p.nodes.push_back(n);
                continue;
            }
            if (R<u32>(n.inner1 + 0x68) > 0xb) {
                const u64 x = R<u64>(R<u64>(n.inner1 + 0x60) + 0x58);
                n.v63 = zero_or_default(x);
            }
            n.v20 = R<u64>(R<u64>(n.inner1 + 0x48) + 0x38);
            n.v21 = R<u64>(R<u64>(R<u64>(n.rd2 + 0x20) + 0x48) + 0x38);
            n.v59 = n.rd3 != 0 ? R<u64>(R<u64>(R<u64>(n.rd3 + 0x20) + 0x48) + 0x38) : 0;
            if (n.has_object4) {
                n.v66 = R<u64>(R<u64>(inner4 + 0x48) + 0x38);
                const u32 count = R<u32>(inner4 + 0x68);
                if (count >= 0x25) {
                    n.v60 = zero_or_default(R<u64>(R<u64>(inner4 + 0x60) + 0x120));
                    if (count >= 0x26) {
                        const u64 x = R<u64>(R<u64>(inner4 + 0x60) + 0x128);
                        n.v65 = (x == p.def || x == 0) ? 0 : R<u64>(x + 0x38);
                    }
                }
            }
            n.v64 = R<u64>(n.inner1 + 0x58) != 0 ? R<u64>(R<u64>(n.inner1 + 0x58) + 0x38) : 0;
            n.count46 = R<s32>(n.rd8 + 0x40);
            n.b74 = !p.async && R<u64>(node + 0xa0) != 0;
            n.b75 = n.v69 != 0 && R<u64>(n.v69 + 0x30) != 0;
            n.b70 = R<u64>(node + 0x108) != 0 && !n.b74;
            if (n.b70) {
                if (!n.has_object4) {
                    p.Decline("constants object without the object at +0x20");
                } else {
                    const u64 candidate = R<u64>(R<u64>(inner4 + 0x60) + 0x108);
                    if (candidate == p.def || candidate == 0 || R<u64>(candidate + 0x38) == 0) {
                        p.Decline("constants object without render data");
                    } else {
                        n.v_a0 = R<u64>(R<u64>(candidate + 0x38) + 0x28);
                    }
                }
            }
            const bool flag54 = (R<u32>(node + 0x54) >> 1 & 1) != 0;
            const bool o5set = n.rd5 != 0, o35set = n.rd35 != 0, o4set = n.has_object4;
            (void)o4set;
            u32 v;
            const bool all_three = n.rd5 != 0 && n.v64 != 0 && n.rd3 != 0;
            if (all_three) {
                v = (n.count46 == 0x40 ? 1u : 0u) | 0x5a;
            } else if (R<u64>(node + 0x108) == 0 || n.b74) {
                if (n.b74) {
                    if (n.b75) {
                        if (n.v180 == 0) {
                            v = (o35set ? 2u : 0u) + 0x4d;
                            if (n.count46 == 0x40) {
                                v = (o35set ? 2u : 0u) + 0x4e;
                            }
                        } else {
                            v = (o35set ? 2u : 0u) + 0x51;
                            if (n.count46 == 0x40) {
                                v = (o35set ? 2u : 0u) + 0x52;
                            }
                        }
                    } else {
                        v = 0x4c;
                        if (n.count46 != 0x40) {
                            v = (flag54 ? 1u : 0u) | 0x4a;
                        }
                    }
                } else if (n.count46 == 0x40) {
                    v = 0x49;
                    if (!flag54) {
                        u32 high = (n.v60 != 0 ? 1u : 0u) + 0x58;
                        v = (n.v60 != 0 ? 1u : 0u) | 0x56;
                        if (n.v180 != 0) {
                            v = high;
                        }
                    }
                } else {
                    v = 0x48;
                    if (!flag54) {
                        v = n.v60 != 0 ? 0x55u : 0x47u;
                    }
                }
                (void)o5set;
            } else {
                const u32 high = (n.v60 != 0 ? 1u : 0u) + 0x5e;
                v = (n.v60 != 0 ? 1u : 0u) | 0x5c;
                if (n.v180 != 0) {
                    v = high;
                }
            }
            n.variant = v;
            if (R<u64>(p.device + 0x1008 + u64{v} * 0x10) == 0) {
                p.Decline("shader variant not loaded");
            }
        }
        p.nodes.push_back(n);
    }
    if (guard >= 100000) {
        p.Decline("node list unreadable");
    }
    // pass 3: the normals
    p.run_normals = !p.async && R<u32>(p.camera + 0x24) != 0;
    if (p.run_normals) {
        for (SkinNode& n : p.nodes) {
            const u64 node = n.node;
            if (R<u64>(node + 0xa0) != 0 && (R<u8>(node + 0x54) & 8) != 0) {
                const u64 object = R<u64>(node + 0x20);
                if (object != 0) {
                    n.normals = true;
                    n.normals_rd = R<u64>(object + 0x38);
                    if (n.normals_rd == 0) {
                        p.Decline("mesh render data not created yet (the skinning loop creates it at first use)");
                    } else if (R<u64>(n.normals_rd + 0x20) == 0) {
                        p.Decline("render data without its inner object");
                    }
                }
            }
        }
        for (const u32 offset : {0x1448u, 0x1458u, 0x1468u}) {
            if (R<u64>(p.device + offset) == 0) {
                p.Decline("normal shader not loaded");
            }
        }
    }
    p.scanned = true;
}

/// The 4x4 inverse the loop makes of a node's matrix (node + 0xb0, row major) with the engine's own order of operations (read off the disassembly: cofactors as `(a*b - c*d) + e*f` per lane,
/// scaled by 1 / determinant with alternating signs); a determinant under 1e-5 gives the engine's constant block. `out` is 16 floats (S + 0xe0 .. 0x11f).
void SkinInverse(u64 node, float* out) {
    const float m00 = R<float>(node + 0xb0), m01 = R<float>(node + 0xb4), m02 = R<float>(node + 0xb8), m03 = R<float>(node + 0xbc);
    const float m10 = R<float>(node + 0xc0), m11 = R<float>(node + 0xc4), m12 = R<float>(node + 0xc8), m13 = R<float>(node + 0xcc);
    const float m20 = R<float>(node + 0xd0), m21 = R<float>(node + 0xd4), m22 = R<float>(node + 0xd8), m23 = R<float>(node + 0xdc);
    const float m30 = R<float>(node + 0xe0), m31 = R<float>(node + 0xe4), m32 = R<float>(node + 0xe8), m33 = R<float>(node + 0xec);
    const float a1 = m13 * m02 - m12 * m03;
    const float a2 = m23 * m02 - m22 * m03;
    const float x = m33 * m02 - m32 * m03;
    const float a3 = m23 * m12 - m22 * m13;
    const float b3 = m33 * m12 - m32 * m13;
    const float a5 = m33 * m22 - m32 * m23;
    const float c0 = (m11 * a5 - m21 * b3) + m31 * a3;
    const float c1 = (m31 * a2 - m21 * x) + m01 * a5;
    const float c2 = (m31 * a1 - m11 * x) + m01 * b3;
    const float c3 = (m21 * a1 - m11 * a2) + m01 * a3;
    const float sum_a = c2 * m20 + c0 * m00;
    const float sum_b = c1 * m10 + c3 * m30;
    const float det = sum_a - sum_b;
    const float abs_det = 0.0f < det ? det : sum_b - sum_a;
    const float threshold = R<float>(Global(0x024f15dc));
    if (!(threshold < abs_det)) {
        std::memcpy(out, reinterpret_cast<const void*>(Global(0x024ff720)), 32);      // S + 0xe0
        std::memcpy(out + 8, reinterpret_cast<const void*>(Global(0x024ff700)), 32); // S + 0x100
        return;
    }
    const float e1 = m11 * m00 - m01 * m10;
    const float e2 = m21 * m00 - m20 * m01;
    const float e3 = m31 * m00 - m30 * m01;
    const float e4 = m10 * m21 - m20 * m11;
    const float e5 = m10 * m31 - m30 * m11;
    const float e6 = m20 * m31 - m30 * m21;
    const float c4 = (m10 * a5 - m20 * b3) + m30 * a3;
    const float c5 = (m00 * a5 - m20 * x) + m30 * a2;
    const float c6 = (m00 * b3 - m10 * x) + m30 * a1;
    const float c7 = (m00 * a3 - m10 * a2) + m20 * a1;
    const float d0 = (e6 * m13 + e4 * m33) - e5 * m23;
    const float d1 = (e6 * m03 + e2 * m33) - e3 * m23;
    const float d2 = (e5 * m03 + e1 * m33) - e3 * m13;
    const float d3 = (e4 * m03 + e1 * m23) - e2 * m13;
    const float d4 = (e6 * m12 + e4 * m32) - e5 * m22;
    const float d5 = (e6 * m02 + e2 * m32) - e3 * m22;
    const float d6 = (e5 * m02 + e1 * m32) - e3 * m12;
    const float d7 = (e4 * m02 + e1 * m22) - e2 * m12;
    const float r = R<float>(Global(0x024f15e0)) / det;
    const float n = -r;
    out[0] = r * c0;
    out[1] = n * c1;
    out[2] = r * c2;
    out[3] = n * c3;
    out[4] = n * c4;
    out[5] = r * c5;
    out[6] = n * c6;
    out[7] = r * c7;
    out[8] = r * d0;
    out[9] = d1 * n;
    out[10] = r * d2;
    out[11] = d3 * n;
    out[12] = d4 * n;
    out[13] = r * d5;
    out[14] = d6 * n;
    out[15] = r * d7;
}

void GenerateSkin(Target& t, const SkinPlan& p) {
    const u64 device = p.device, scene = p.scene, S_live = p.system;
    const u64 ctx = p.ctx;
    // the system object: live, the loop's own; verify, a private copy of the first 0x400 bytes (its staging area is written by the loop)
    std::array<u8, 0x400> shadow_system{};
    u8* sb = reinterpret_cast<u8*>(S_live);
    if (!t.live) {
        std::memcpy(shadow_system.data(), reinterpret_cast<const void*>(S_live), shadow_system.size());
        sb = shadow_system.data();
    }
    const auto sget = [&]<class T>(u32 offset) { T v; std::memcpy(&v, sb + offset, sizeof(T)); return v; };
    const auto sput = [&]<class T>(u32 offset, T value) { std::memcpy(sb + offset, &value, sizeof(T)); };
    const auto sput_from = [&](u32 offset, u64 source, u32 bytes) { std::memcpy(sb + offset, reinterpret_cast<const void*>(source), bytes); };
    // the ring index and the bone matrices
    const s32 old_index = sget.template operator()<s32>(0x40);
    const s32 index = (old_index + 1) % 4;
    if (t.live) {
        W<s32>(S_live + 0x40, index);
    }
    const u64 ring_memory = R<u64>(R<u64>(S_live + u64{static_cast<u32>(index)} * 8) + 0x30);
    if (t.live) {
        u32 offset = 0;
        for (const SkinNode& n : p.nodes) {
            if (n.active) {
                std::memcpy(reinterpret_cast<void*>(ring_memory + u64{offset} * 0x40), reinterpret_cast<const void*>(R<u64>(n.node + 0x48)), u64{R<u32>(n.node + 0x50)} << 6);
                offset += R<u32>(n.node + 0x50);
            }
        }
    }
    const std::array<u64, 3> cb_array{R<u64>(S_live + 0x1d0), R<u64>(S_live + 0x230), R<u64>(S_live + 0x340)};
    const u32 set_shader_offset = p.async ? 0x1d10 : 0x4670, cb_table = p.async ? 0x21f8 : 0x500, srv_table = p.async ? 0x1d18 : 0x1980, uav_table = p.async ? 0x2118 : 0x2098;
    const auto update_subresource = [&](u64 buffer, u32 source_offset, u32 bytes) {
        if (t.live) {
            Timed(CallId::SkinUpdate, [&] {
                reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64)>(R<u64>(R<u64>(ctx) + 0xd0))(ctx, buffer, 0, 0, S_live + source_offset, 0, 0);
            });
        } else {
            std::vector<u8> block(bytes);
            std::memcpy(block.data(), sb + source_offset, bytes);
            t.blocks.push_back({S_live + source_offset, std::move(block)});
        }
    };
    const auto dispatch = [&](u32 x, bool fenced, u64 site) {
        if (t.live) {
            Timed(CallId::DrawIssue, [&] { NativeDraw::IssueDispatch(ctx, x, 1, 1, fenced, site, p.async); });
        } else {
            t.marks.push_back({t.arena.size(), p.async ? 4u : 3u, x, 1, 1, t.Get<u64>(set_shader_offset), 0});
            t.marks.back().blocks = std::move(t.blocks);
            t.blocks.clear();
            t.marks.back().flag_source = p.context_global + 0x19c;
            t.marks.back().flag_block = S_live + 0x50;
        }
        ++t.draws;
    };
    const auto flush_wait = [&] {
        if (t.live) {
            reinterpret_cast<void(PS4_SYSV_ABI*)(u64)>(R<u64>(R<u64>(ctx) + 0x78))(ctx);
        }
    };
    const auto set_cb = [&](u32 start, u32 count, const u64* values) { t.Put(cb_table + start * 8, values, count * 8); };
    const auto set_srv = [&](u32 start, u32 count, const u64* values) { t.Put(srv_table + start * 8, values, count * 8); };
    const auto set_uav = [&](u32 start, u32 count, const u64* values) {
        t.Put(uav_table + start * 8, values, count * 8);
        if (p.async && t.live) {
            for (u32 i = 0; i < count; ++i) {
                if (values[i] != 0) {
                    W<u32>(values[i] + 0x88, 0xffffffffu); // the async context resets the counter index of every view it is given (no counter array)
                }
            }
        }
    };
    const auto set_shader = [&](u64 shader) { t.Set<u64>(set_shader_offset, shader); };
    // pass 2: one dispatch per object
    u32 matrix_offset = 0;
    const float one_e5 = R<float>(Global(0x024f15dc));
    for (const SkinNode& n : p.nodes) {
        if (!n.active) {
            continue;
        }
        const u64 node = n.node;
        set_shader(R<u64>(device + 0x1008 + u64{n.variant} * 0x10));
        if (n.b70) {
            u32 dwords = R<u32>(node + 0x110);
            if (dwords > 0x3f) {
                dwords = 0x40;
            }
            sput_from(0x240, R<u64>(node + 0x108), dwords << 2);
            update_subresource(cb_array[2], 0x240, std::min<u32>(R<u32>(cb_array[2] + 0x180), 0x100));
        }
        if (p.view == 0) {
            const u64 cam = p.camera_object;
            const float f2 = R<float>(cam + 0x90), f3 = R<float>(cam + 0x98), f77 = -R<float>(cam + 0x94);
            sput.template operator()<float>(0x60, (f77 * R<float>(node + 0xb4) - f2 * R<float>(node + 0xb0)) - f3 * R<float>(node + 0xb8));
            sput.template operator()<float>(0x64, (f77 * R<float>(node + 0xc4) - f2 * R<float>(node + 0xc0)) - f3 * R<float>(node + 0xc8));
            sput.template operator()<float>(0x68, (f77 * R<float>(node + 0xd4) - f2 * R<float>(node + 0xd0)) - f3 * R<float>(node + 0xd8));
            sput.template operator()<u32>(0x6c, 0x3f800000u);
            sput_from(0x170, cam + 0xf0, 16);
        } else {
            const u64 view = p.view;
            sput_from(0x190, view + 0x80, 64);
            sput.template operator()<u32>(0x60, R<u32>(view + 0x2a0) ^ 0x80000000u);
            sput.template operator()<u32>(0x64, R<u32>(view + 0x2a4) ^ 0x80000000u);
            sput.template operator()<u32>(0x68, R<u32>(view + 0x2a8) ^ 0x80000000u);
            sput.template operator()<u32>(0x6c, R<u32>(view + 0x2ac));
            sput_from(0x170, view + 0x140, 16);
        }
        sput_from(0x180, p.camera + 0x30, 16);
        sput.template operator()<u32>(0x50, R<u32>(node + 0x30));
        sput.template operator()<u32>(0x54, R<u32>(node + 0x50));
        sput.template operator()<u32>(0x58, matrix_offset);
        sput.template operator()<u32>(0x5c, !(R<float>(p.context_global + 0x19c) < one_e5) ? 1u : 0u);
        const u64 camera_flags = R<u64>(p.camera + 0x40);
        sput.template operator()<u32>(0x70, static_cast<u32>(camera_flags));
        sput.template operator()<u32>(0x74, static_cast<u32>(camera_flags >> 32));
        sput.template operator()<u64>(0x78, 0);
        sput_from(0x90, node + 0xf0, 16);
        sput_from(0xa0, node + 0xb0, 64);
        {
            std::array<float, 16> inverse{};
            SkinInverse(node, inverse.data());
            std::memcpy(sb + 0xe0, inverse.data(), 64);
        }
        if (const u64 holder = R<u64>(node + 0xa0); holder != 0) {
            sput.template operator()<float>(0x80, 1.0f / R<float>(holder + 0x30));
            sput.template operator()<u32>(0x84, R<u32>(holder + 0x34));
            sput.template operator()<u32>(0x88, 0);
            sput.template operator()<u32>(0x8c, R<u32>(holder + 0x38));
            sput_from(0x120, node + 0x70, 16);
            const u64 w80 = R<u64>(node + 0x80), w88 = R<u64>(node + 0x88);
            sput.template operator()<u32>(0x130, static_cast<u32>(w80));
            sput.template operator()<u32>(0x134, static_cast<u32>(w80 >> 32));
            sput.template operator()<u32>(0x138, static_cast<u32>(w88));
            sput.template operator()<u32>(0x13c, R<u32>(holder + 0x48));
            sput_from(0x140, node + 0x90, 16);
            const u64 h40 = R<u64>(holder + 0x40), h58 = R<u64>(holder + 0x58);
            sput.template operator()<u32>(0x150, static_cast<u32>(h40));
            sput.template operator()<u32>(0x154, static_cast<u32>(h58));
            sput.template operator()<u32>(0x158, static_cast<u32>(h40 >> 32));
            sput.template operator()<u32>(0x15c, static_cast<u32>(h58 >> 32));
        }
        sput.template operator()<u32>(0x160, R<u32>(node + 0x100));
        sput.template operator()<u64>(0x164, 0);
        sput.template operator()<u32>(0x16c, 0);
        update_subresource(cb_array[0], 0x50, std::min<u32>(R<u32>(cb_array[0] + 0x180), SkinBlockBytes));
        matrix_offset += R<u32>(node + 0x50);
        set_cb(0, 3, cb_array.data());
        // the views and the targets
        std::array<u64, 11> srv{};
        srv[0] = R<u64>(n.v20 + 0x28);
        srv[1] = R<u64>(n.rd8 + 0x28);
        srv[2] = R<u64>(S_live + 0x20 + u64{static_cast<u32>(index)} * 8);
        srv[3] = n.v64 != 0 ? R<u64>(n.v64 + 0x28) : 0;
        srv[4] = n.v59 != 0 ? R<u64>(n.v59 + 0x28) : 0;
        if (const u64 holder = R<u64>(node + 0xa0); holder != 0) {
            const u64 data = HandleLookup(device, R<u64>(holder + 0x50));
            srv[5] = R<u64>(data + 0x48);
        }
        srv[6] = n.v63 != 0 ? R<u64>(n.v63 + 0x28) : 0;
        srv[7] = n.rd35 != 0 ? R<u64>(n.rd35 + 0x28) : 0;
        srv[8] = n.b70 ? n.v_a0 : 0;
        srv[9] = n.v180 != 0 ? R<u64>(n.v180 + 0x28) : 0;
        srv[10] = n.v60 != 0 ? R<u64>(n.v60 + 0x28) : 0;
        set_srv(4, 11, srv.data());
        std::array<u64, 8> uav{};
        uav[0] = R<u64>(n.v21 + 0x30);
        uav[2] = n.has_object4 ? R<u64>(n.v66 + 0x30) : 0;
        uav[3] = n.rd5 != 0 ? R<u64>(n.rd5 + 0x30) : 0;
        uav[5] = n.v69 != 0 ? R<u64>(n.v69 + 0x30) : 0;
        uav[6] = n.b70 ? R<u64>(n.v50 + 0x30) : 0;
        uav[7] = n.v65 != 0 ? R<u64>(n.v65 + 0x30) : 0;
        set_uav(0, 8, uav.data());
        dispatch((R<u32>(node + 0x34) + 0x3f) >> 6, false, SkinSiteDispatch);
    }
    flush_wait();
    {
        const std::array<u64, 11> zeros{};
        set_srv(0, 4, zeros.data());
        set_srv(4, 11, zeros.data());
        set_uav(0, 8, zeros.data());
        set_cb(0, 3, zeros.data());
        set_shader(0);
    }
    // pass 3
    if (p.run_normals) {
        for (const SkinNode& n : p.nodes) {
            if (!n.normals) {
                continue;
            }
            const u64 node = n.node;
            const u64 object = R<u64>(node + 0x20);
            const u64 rd = n.normals_rd;
            const u64 w38 = R<u64>(node + 0x38);
            const u32 triangles = static_cast<u32>(w38 >> 32), vertices = R<u32>(node + 0x34);
            sput.template operator()<u32>(0x350, static_cast<u32>(w38));
            sput.template operator()<u32>(0x354, 0);
            const u64 cb360 = R<u64>(S_live + 0x360);
            update_subresource(cb360, 0x350, std::min<u32>(R<u32>(cb360 + 0x180), 0x10));
            set_cb(0, 1, &cb360);
            const u64 uav380 = R<u64>(S_live + 0x380);
            set_uav(0, 1, &uav380);
            set_shader(R<u64>(device + 0x1448));
            const u32 groups = (vertices + 0x3f) >> 6;
            dispatch(groups, true, SkinSiteClear);
            const u64 inner = R<u64>(rd + 0x20);
            const std::array<u64, 2> srv_a{R<u64>(R<u64>(R<u64>(inner + 0x50) + 0x38) + 0x28), R<u64>(R<u64>(R<u64>(inner + 0x48) + 0x38) + 0x28)};
            set_srv(0, 2, srv_a.data());
            set_shader(R<u64>(device + 0x1458));
            dispatch(((triangles / 3) + 0x3f) >> 6, true, SkinSiteCalculate);
            const u64 zero = 0;
            set_uav(0, 1, &zero);
            const std::array<u64, 2> srv_b{R<u64>(S_live + 0x378), 0};
            set_srv(0, 2, srv_b.data());
            const u64 uav_normal = R<u64>(R<u64>(R<u64>(inner + 0x48) + 0x38) + 0x30);
            set_uav(0, 1, &uav_normal);
            set_shader(R<u64>(device + 0x1468));
            dispatch(groups, true, SkinSiteNormalize);
            set_uav(0, 1, &zero);
            const std::array<u64, 2> srv_c{};
            set_srv(0, 2, srv_c.data());
            (void)object;
        }
        flush_wait();
        const std::array<u64, 11> zeros{};
        set_srv(0, 11, zeros.data());
        set_uav(0, 8, zeros.data());
        const u64 zero = 0;
        set_cb(0, 1, &zero);
        set_shader(0);
    }
}

struct SConfig {
    bool on{false};
};
const SConfig& SCfg() {
    static const SConfig config = [] {
        SConfig c;
        const char* e = std::getenv("TLG_SKIN_GEN");
        c.on = e != nullptr ? *e != '0' : GeneratorsDefaultOn(); // off under TLG_VKB (note 57); before: default on since 2026-10-27 (the owner's play test); TLG_SKIN_GEN=0 turns it off
        return c;
    }();
    return config;
}

struct SStats {
    ViewStats v;
    u64 eligible{0};
    u64 loops_without_work{0};
};
SStats& SS() {
    static SStats stats;
    return stats;
}

// ------------------------------------------------------------------------------------------------------------------------------- B5: the wet simulation
// FUN_0199bf00(wet system W, device context C, scene F), a virtual of the wet system called once per frame per context (graphics or async compute): the wetness of the characters and
// objects in three phases (CS_WetElapse: the time step; CS_WetIntersect: every wet object's box against the system's boxes; CS_WetForce: the water modifier's force) and, when the
// CPU asked for it (system + 0x1b8 bit 2), the readback of the result (CopyResource into a staging resource, Map, memcpy, Unmap). Like the skinning loop it talks to the context through
// virtuals that are table writes (SetConstantBuffers + 0x28, SetShader + 0x38, SetShaderResources + 0x40, SetUnorderedAccessViews + 0x48), UpdateSubresource (+ 0xd0) of its own constant
// blocks and the dispatches (+ 0x58, return addresses 0x199c529, 0x199d14b, 0x199d728); what is not a table write (a PM4 packet through + 0xb8, the wait + 0x78, CopyResource / Map /
// Unmap + 0x88 / 0x90 / 0xc8) is called through the same virtuals in the live run. Its locks (the system's at + 0x70, the job lists' at + 0x1b4, a ticket at + 0x250) are taken only in the
// live run. Decompiled and annotated in documents/tlg/native plan/stages/b5-wet.md.
constexpr u64 WetBegin = 0x0199bf00, WetEnd = 0x0199dba5;
constexpr u64 WetSiteElapse = 0x0199c529, WetSiteIntersect = 0x0199d14b, WetSiteForce = 0x0199d728;

struct WetPlan {
    bool scanned{false};
    bool no_work{false};
    bool async{false};
    std::vector<const char*> reasons;
    u64 system{0}, ctx{0}, scene{0}, device{0}, context_global{0};
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* r) {
        for (const char* x : reasons) {
            if (x == r) {
                return;
            }
        }
        reasons.push_back(r);
    }
};

struct WetScanArgs {
    WetPlan* plan;
};

constexpr u32 WetSystemsMax = 64, WetJobsMax = 4096;

void ScanWet(WetScanArgs& a) {
    WetPlan& p = *a.plan;
    const u64 Wsys = p.system;
    p.scanned = true;
    p.device = R<u64>(Global(GlobalDevice));
    p.context_global = R<u64>(Global(GlobalContext));
    if (R<s32>(Wsys + 0x68) == 0 || !(1e-5f <= R<float>(p.context_global + 0x19c))) {
        p.no_work = true; // the function does nothing
        return;
    }
    if (p.device == 0 || p.context_global == 0) {
        p.Decline("no device");
        return;
    }
    // the context the system is called with is one of the frame's contexts, not always the device's current one: its class (graphics or async compute) is its vtable
    const u64 graphics = R<u64>(p.device + 0x2d8);
    const u64 asc_holder = R<u64>(p.device + 0x2c8);
    const u64 asc = asc_holder != 0 ? R<u64>(asc_holder + 0x30258) : 0;
    const u64 vtable = R<u64>(p.ctx);
    if (graphics != 0 && vtable == R<u64>(graphics)) {
        p.async = false;
    } else if (asc != 0 && vtable == R<u64>(asc)) {
        p.async = true;
    } else {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 3) {
            TLG_LOG(GEN, Warn, "wet: context {:#x} (vtable {:#x}), graphics {:#x} (vtable {:#x}), async {:#x}", p.ctx, vtable, graphics, graphics != 0 ? R<u64>(graphics) : 0, asc);
        }
        p.Decline("a device context the generator does not know");
        return;
    }
    if (R<u64>(Wsys + 0x90) == 0) {
        p.Decline("the wet system's constant buffer is not created");
        return;
    }
    const u32 count = R<u32>(Wsys + 0x58);
    const u64 systems = R<u64>(Wsys + 0x50);
    if (count > WetSystemsMax || (count != 0 && systems == 0)) {
        p.Decline("the system list is not plausible");
        return;
    }
    const u64 def = R<u64>(R<u64>(p.context_global + 0x38) + 0x20);
    const u64 water = R<u64>(p.scene + 0x31ed0);
    const bool water_on = water != 0 && R<s32>(water + 0x48) != 0;
    if (water_on && R<u64>(water + 0x38) == 0) {
        p.Decline("render data not created yet (the wet loop creates it at first use)");
    }
    const auto has_rd = [&](u64 object) { return object != 0 && R<u64>(object + 0x38) != 0; };
    for (u32 i = 0; i < count; ++i) {
        const u64 s = R<u64>(systems + i * 8);
        if (s == 0) {
            p.Decline("the system list is not plausible");
            return;
        }
        if (R<s32>(s + 0x30) == 0) {
            continue;
        }
        if (R<u64>(s + 0x120) == 0 || R<u64>(s + 0x190) == 0) {
            p.Decline("a system's constant buffer is not created");
            return;
        }
        if (R<s32>(s + 0x1d4) == 0) {
            continue;
        }
        const u32 jobs = R<u32>(s + 0x1a8);
        const u64 list = R<u64>(s + 0x1a0);
        if (jobs > WetJobsMax || (jobs != 0 && list == 0)) {
            p.Decline("the job list is not plausible");
            return;
        }
        const bool readback = (R<u32>(s + 0x1b8) & 4) != 0;
        if (readback) {
            // the CPU readback (CopyResource, Map, memcpy, Unmap) is written in GenerateWet but no scene of the regression set asks for it, so it has never been verified: the engine keeps it
            Core::RareEvents::Note("wet_cpu_readback", fmt::format("wet system {:#x}, flags {:#x}, {} jobs: the generator declined, the engine's loop did the readback", s, R<u32>(s + 0x1b8), jobs));
            p.Decline("a CPU readback is pending (the engine's loop does it until the generator's is verified)");
            return;
        }
        for (u32 j = 0; j < jobs; ++j) {
            const u64 e = list + u64{j} * 0x20;
            const u64 object = R<u64>(e);
            if (object == 0) {
                p.Decline("a job without its object");
                return;
            }
            const u64 x = R<u64>(object + 0x48);
            const u64 y = R<u64>(R<u64>(object + 0x60) + 0x60);
            if (!has_rd(x) || !has_rd(y)) {
                p.Decline("render data not created yet (the wet loop creates it at first use)");
                return;
            }
            if (R<u32>(object + 0x68) > 0xd) {
                const u64 c = R<u64>(R<u64>(object + 0x60) + 0x68);
                if (c != def && c != 0) {
                    if (!has_rd(c)) {
                        p.Decline("render data not created yet (the wet loop creates it at first use)");
                        return;
                    }
                    if (readback) {
                        const u64 staging = R<u64>(e + 8);
                        if (staging == 0 || R<u64>(e + 0x10) == 0) {
                            p.Decline("the readback destination is not allocated yet (the wet loop allocates it at first use)");
                            return;
                        }
                        if (!has_rd(staging)) {
                            p.Decline("render data not created yet (the wet loop creates it at first use)");
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// The ticket the engine's FUN_0120d030 takes on a system (+ 0x250): the counter moves only while its high half is zero.
void WetTicketAcquire(u64 address) {
    auto* word = reinterpret_cast<std::atomic<u32>*>(address);
    for (u32 spin = 0;; ++spin) {
        u32 expected = word->load(std::memory_order_relaxed) & 0xffffu;
        if (word->compare_exchange_strong(expected, (expected & 0xffffu) + 1u, std::memory_order_acq_rel)) {
            return;
        }
        if (spin > 64) {
            SwitchToThread();
        }
    }
}

/// A lock word the engine spins on (0 free, 1 taken): the wet system's (+ 0x70) and the job lists' (+ 0x1b4, FUN_0199dbd0).
void WetLockAcquire(u64 address) {
    auto* word = reinterpret_cast<std::atomic<s32>*>(address);
    for (u32 spin = 0;; ++spin) {
        s32 expected = 0;
        if (word->compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
            return;
        }
        if (spin > 64) {
            SwitchToThread();
        }
    }
}

void WetLockRelease(u64 address) {
    auto* word = reinterpret_cast<std::atomic<s32>*>(address);
    s32 expected = 1;
    word->compare_exchange_strong(expected, 0, std::memory_order_acq_rel);
}

std::atomic<u64> wet_skipped_jobs{0};
std::array<std::atomic<u64>, 4> wet_dispatches{}; // by phase: elapse, intersect, force; and the readbacks done
std::atomic<u64> s_raced_wet_color{0};

/// The blocks of the wet loop as functions of the engine's objects, so the verify run can build them again when it compares (the objects are rewritten by the game's thread).
void WetBuildStep(u64 s, u64 G, std::array<u8, 16>& block) {
    block = {};
    const u32 step = R<u32>(s + 0x1c8);
    const u32 time = R<u32>(G + 0x19c);
    std::memcpy(block.data(), &step, 4);
    std::memcpy(block.data() + 4, &time, 4);
}

void WetBuildIntersect(u64 s, u64 P, u64 object, u64 B, std::array<u8, 0xc0>& block) {
    block = {};
    // the block at s + 0x60 .. 0x11f
    const auto put32 = [&](u32 at, u32 v) { std::memcpy(block.data() + (at - 0x60), &v, 4); };
    const auto putf = [&](u32 at, float v) { std::memcpy(block.data() + (at - 0x60), &v, 4); };
    const auto put64 = [&](u32 at, u64 v) { std::memcpy(block.data() + (at - 0x60), &v, 8); };
    const auto Pf = [&](u32 o) { return R<float>(P + o); };
    const auto Bf = [&](u32 o) { return R<float>(B + o); };
    const u64 c1c0 = R<u64>(s + 0x1c0);
    put32(0x60, R<u32>(s + 0x1bc));
    put32(0x64, static_cast<u32>(c1c0));
    put32(0x68, static_cast<u32>(c1c0 >> 32));
    put64(0x6c, 0);
    put64(0x74, 0);
    put32(0x7c, 0);
    putf(0x80, 1.0f / R<float>(object + 0xa0));
    putf(0x84, 1.0f / R<float>(object + 0xa4));
    put32(0x88, R<u32>(object + 0x60));
    put32(0x8c, R<u32>(object + 0x68));
    // the four rows of the object's matrix and the system's two 4x4 factors, in the engine's operation order
    float row[4][4];
    for (u32 r = 0; r < 4; ++r) {
        for (u32 c = 0; c < 4; ++c) {
            row[r][c] = Bf(0xb0 + r * 0x10 + c * 4);
        }
    }
    const float P70 = Pf(0x70), P74 = Pf(0x74), P78 = Pf(0x78), P7c = Pf(0x7c), P80 = Pf(0x80), P84 = Pf(0x84), P88 = Pf(0x88), P8c = Pf(0x8c), P90 = Pf(0x90), P94 = Pf(0x94),
                P98 = Pf(0x98), P9c = Pf(0x9c), Pa0 = Pf(0xa0), Pa4 = Pf(0xa4), Pa8 = Pf(0xa8), Pac = Pf(0xac);
    // the engine adds the four products as (row 1 + row 0) + (row 2 + row 3) (read off the disassembly: ymm lanes and scalar chains, not the decompiler's left to right sum)
    const auto tree = [](float r1, float r0, float r2, float r3) { return (r1 + r0) + (r2 + r3); };
    for (u32 c = 0; c < 4; ++c) {
        putf(0x90 + c * 4, tree(row[1][c] * P74, row[0][c] * P70, row[2][c] * P78, row[3][c] * P7c));
        putf(0xa0 + c * 4, tree(P84 * row[1][c], P80 * row[0][c], P88 * row[2][c], P8c * row[3][c]));
    }
    const float Pm[4][2] = {{P94, P90}, {P98, P9c}, {Pa4, Pa0}, {Pa8, Pac}};
    for (u32 c = 0; c < 4; ++c) {
        putf(0xb0 + c * 4, tree(P94 * row[1][c], P90 * row[0][c], P98 * row[2][c], P9c * row[3][c]));
        putf(0xc0 + c * 4, tree(Pa4 * row[1][c], Pa0 * row[0][c], Pa8 * row[2][c], Pac * row[3][c]));
    }
    (void)Pm;
    for (u32 k = 0; k < 4; ++k) {
        put64(0xd0 + k * 8, R<u64>(B + 0x70 + k * 8));
        put64(0xf0 + k * 8, R<u64>(B + 0x90 + k * 8));
    }
    put32(0x110, R<u32>(B + 0x150));
    put32(0x114, R<u32>(B + 0x160));
    put32(0x118, R<u32>(B + 0x158));
    put32(0x11c, R<u32>(B + 0x168));
}

void WetBuildForce(u64 s, u64 P, u64 water, std::array<u8, 0x54>& block) {
    block = {};
    // the block at s + 0x130: the system's parameters, the water's value and the colour
    for (u32 k = 0; k < 8; ++k) {
        const u32 from = k < 4 ? 0x70 + k * 8 : 0x90 + (k - 4) * 8; // 0x70 0x78 0x80 0x88, then 0x90 0x98 0xa0 0xa8
        const u64 v = R<u64>(P + from);
        std::memcpy(block.data() + k * 8, &v, 8);
    }
    const u32 water_value = R<u32>(water + 0x48);
    std::memcpy(block.data() + 0x50, &water_value, 4);
    const float f = R<float>(s + 0x258) * std::bit_cast<float>(0x3b808081u);
    const u32 packed = R<u32>(s + 0x254);
    const std::array<u32, 4> channels{static_cast<u32>(R<u8>(s + 0x256)), static_cast<u32>(R<u8>(s + 0x255)), packed, packed >> 24};
    const float scale[4] = {f, f, f, std::bit_cast<float>(0x3b808081u)};
    for (u32 c = 0; c < 4; ++c) {
        const float v = scale[c] * static_cast<float>(static_cast<s32>(channels[c] & 0xffu));
        std::memcpy(block.data() + 0x40 + c * 4, &v, 4);
    }
}

void GenerateWet(Target& t, const WetPlan& p) {
    const u64 Wsys = p.system, F = p.scene, device = p.device, ctx = p.ctx, G = p.context_global;
    const bool live = t.live;
    using Virtual = u64(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64);
    const auto vcall = [&](u32 slot, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) { return reinterpret_cast<Virtual>(R<u64>(R<u64>(ctx) + slot))(ctx, a1, a2, a3, a4, a5, 0); };
    const u32 set_shader_offset = p.async ? 0x1d10 : 0x4670, cb_table = p.async ? 0x21f8 : 0x500, srv_table = p.async ? 0x1d18 : 0x1980, uav_table = p.async ? 0x2118 : 0x2098;
    const auto set_cb = [&](u32 start, u32 count, const u64* values) { t.Put(cb_table + start * 8, values, count * 8); };
    const auto set_srv = [&](u32 start, u32 count, const u64* values) { t.Put(srv_table + start * 8, values, count * 8); };
    const auto set_uav = [&](u32 start, u32 count, const u64* values) {
        t.Put(uav_table + start * 8, values, count * 8);
        if (p.async && live) {
            for (u32 i = 0; i < count; ++i) {
                if (values[i] != 0) {
                    W<u32>(values[i] + 0x88, 0xffffffffu); // the async context resets the counter index of every view it is given (no counter array)
                }
            }
        }
    };
    // table entries with the object word each was read from, so the verify run can tell a word the engine's other threads replaced between the two reads
    struct Src {
        u64 value;
        u64 owner;
        u32 offset;
    };
    const auto put_table = [&](u32 table, u32 start, std::initializer_list<Src> sources, bool uav) {
        u32 i = 0;
        for (const Src& src : sources) {
            t.direct_object = src.owner;
            t.direct_offset = src.offset;
            t.Put(table + (start + i) * 8, &src.value, 8);
            t.direct_object = 0;
            if (uav && p.async && live && src.value != 0) {
                W<u32>(src.value + 0x88, 0xffffffffu);
            }
            ++i;
        }
    };
    const auto put_cb = [&](u32 start, std::initializer_list<Src> sources) { put_table(cb_table, start, sources, false); };
    const auto put_srv = [&](u32 start, std::initializer_list<Src> sources) { put_table(srv_table, start, sources, false); };
    const auto put_uav = [&](u32 start, std::initializer_list<Src> sources) { put_table(uav_table, start, sources, true); };
    const auto set_shader = [&](u64 shader) { t.Set<u64>(set_shader_offset, shader); };
    u32 wet_kind = 0;
    u64 wet_system = 0, wet_object = 0, wet_water = 0, wet_block = 0;
    const auto dispatch = [&](u32 x, u64 site) {
        wet_dispatches[site == WetSiteElapse ? 0 : site == WetSiteIntersect ? 1 : 2].fetch_add(1, std::memory_order_relaxed);
        if (live) {
            Timed(CallId::DrawIssue, [&] { NativeDraw::IssueDispatch(ctx, x, 1, 1, false, site, p.async); });
        } else {
            t.marks.push_back({t.arena.size(), p.async ? 4u : 3u, x, 1, 1, t.Get<u64>(set_shader_offset), 0});
            t.marks.back().blocks = std::move(t.blocks);
            t.blocks.clear();
            Mark& mk = t.marks.back();
            mk.wet_kind = wet_kind;
            mk.wet_system = wet_system;
            mk.wet_object = wet_object;
            mk.wet_water = wet_water;
            mk.wet_block = wet_block;
            mk.wet_context_global = G;
        }
        ++t.draws;
    };
    const auto wait = [&] {
        if (live) {
            vcall(0x78, 0, 0, 0, 0, 0);
        }
    };
    // UpdateSubresource of a staged block: the engine writes its fields into its own memory and the virtual copies the destination's byte width from there. The live run does the same;
    // the verify run keeps the bytes for the comparison at the next dispatch (a block staged again at the same address replaces the pending one: no dispatch consumed it)
    const auto update = [&](u64 resource, u64 source, const u8* bytes, u32 size) {
        if (live) {
            std::memcpy(reinterpret_cast<void*>(source), bytes, size);
            Timed(CallId::WetUpdate, [&] { vcall(0xd0, resource, 0, 0, source, 0); });
        } else {
            const u32 n = std::min<u32>(R<u32>(resource + 0x180), size);
            for (auto& [address, block] : t.blocks) {
                if (address == source) {
                    block.assign(bytes, bytes + n);
                    return;
                }
            }
            t.blocks.push_back({source, std::vector<u8>(bytes, bytes + n)});
        }
    };
    const auto lock_list = [&](u64 s) {
        if (live) {
            WetLockAcquire(s + 0x1b4);
        }
    };
    const auto unlock_list = [&](u64 s) {
        if (live) {
            WetLockRelease(s + 0x1b4);
        }
    };
    const auto render_data = [&](u64 object) -> u64 {
        const u64 rd = object != 0 ? R<u64>(object + 0x38) : 0;
        if (rd == 0 && live) {
            wet_skipped_jobs.fetch_add(1, std::memory_order_relaxed); // created between the scan and now: the engine would create it here
        }
        return rd;
    };
    const auto skip_gt = [](float a, float b) { return a > b; };
    const auto skip_lt = [](float a, float b) { return !(a >= b); };

    if (live) {
        W<s32>(Wsys + 0x74, 2);
    }
    const float dt = R<float>(G + 0x19c);
    const u64 ring = static_cast<u64>(static_cast<s64>(R<s32>(Wsys + 0x78)) ^ 1);
    const u64 list_count_at = Wsys + 0xa8 + ring * 0x18;
    // the objects with render data that is in use (the list the engine builds before it starts)
    std::vector<u64> wet_objects;
    if (const u32 n = R<u32>(list_count_at); n != 0) {
        const u64 array = R<u64>(Wsys + 0xa0 + ring * 0x18);
        for (u32 i = 0; i < n; ++i) {
            const u64 object = R<u64>(array + u64{i} * 0x10);
            const u64 rd = R<u64>(object + 0x38);
            if (rd != 0 && R<s32>(rd + 0x30) != 0) {
                wet_objects.push_back(object);
            }
        }
    }
    const u64 systems = R<u64>(Wsys + 0x50);
    const u32 system_count = R<u32>(Wsys + 0x58);
    const u64 constants = R<u64>(Wsys + 0x90);
    put_cb(0, {{constants, Wsys, 0x90}});
    if (live) {
        WetLockAcquire(Wsys + 0x70);
        vcall(0xb8, 0xd8, 0, 0, 0, 0);
    }
    set_shader(R<u64>(device + 0x1168));
    // ---- phase 1: the time step
    if (system_count != 0) {
        bool any = false;
        for (u32 i = 0; i < system_count; ++i) {
            const u64 s = R<u64>(systems + u64{i} * 8);
            if (R<s32>(s + 0x30) == 0) {
                continue;
            }
            std::array<u8, 16> block{};
            WetBuildStep(s, G, block);
            update(constants, Wsys + 0x80, block.data(), 16);
            wet_kind = 1;
            wet_system = s;
            wet_block = Wsys + 0x80;
            any = true;
            if (R<s32>(s + 0x1d4) != 0) {
                lock_list(s);
                const u32 jobs = R<u32>(s + 0x1a8);
                const u64 list = R<u64>(s + 0x1a0);
                for (u32 j = 0; j < jobs; ++j) {
                    const u64 e = list + u64{j} * 0x20;
                    const u64 object = R<u64>(e);
                    const s32 amount = R<s32>(e + 0x18);
                    const u64 rd_a = render_data(R<u64>(object + 0x48));
                    const u64 rd_b = render_data(R<u64>(R<u64>(object + 0x60) + 0x60));
                    if (rd_a == 0 || rd_b == 0) {
                        continue;
                    }
                    put_cb(1, {{R<u64>(s + 0x120), s, 0x120}});
                    put_srv(0, {{R<u64>(rd_a + 0x28), rd_a, 0x28}});
                    put_uav(0, {{R<u64>(rd_b + 0x30), rd_b, 0x30}});
                    dispatch((static_cast<u32>(amount) + 0xffu) >> 8, WetSiteElapse);
                }
                unlock_list(s);
            }
        }
        if (any) {
            wait();
        }
    }
    (void)dt;
    // ---- phase 2: the objects' boxes against the systems' boxes
    if (R<u32>(list_count_at) != 0) {
        set_shader(R<u64>(device + 0x1178));
        if (system_count != 0) {
            bool any = false;
            for (u32 i = 0; i < system_count; ++i) {
                const u64 s = R<u64>(systems + u64{i} * 8);
                if (R<s32>(s + 0x30) == 0) {
                    continue;
                }
                const u64 P = R<u64>(s + 0x28);
                if (R<s32>(s + 0x1d4) != 0 && P != 0 && !wet_objects.empty()) {
                    const u64 box = (static_cast<u64>(static_cast<s64>(R<s32>(s + 0x54)) ^ 1)) * 0x20;
                    const u64 hi = s + 0x220 + box, lo = s + 0x210 + box;
                    for (const u64 object : wet_objects) {
                        const u64 B = R<u64>(R<u64>(object + 0x38) + 0x28);
                        if (B == 0 || (R<u8>(B + 0x24) & 1) == 0) {
                            continue;
                        }
                        if (skip_gt(R<float>(B + 0x170), R<float>(hi)) || skip_lt(R<float>(B + 0x180), R<float>(lo)) || skip_gt(R<float>(B + 0x174), R<float>(hi + 4)) ||
                            skip_lt(R<float>(B + 0x184), R<float>(lo + 4)) || skip_gt(R<float>(B + 0x178), R<float>(hi + 8)) || skip_lt(R<float>(B + 0x188), R<float>(lo + 8))) {
                            continue;
                        }
                        std::array<u8, 0xc0> block{};
                        WetBuildIntersect(s, P, object, B, block);
                        update(R<u64>(s + 0x120), s + 0x60, block.data(), 0xc0);
                        wet_kind = 2;
                        wet_system = s;
                        wet_object = object;
                        wet_block = s + 0x60;
                        // the second shader resource: the object's wetness texture or the default one
                        u64 handle;
                        const u64 texture = R<u64>(object + 0x40);
                        if (texture == 0 || static_cast<u32>(R<u64>(texture + 0x50)) == 0 || (R<u64>(texture + 0x50) >> 32) == 0 || R<s32>(texture + 0x30) == 0 || R<s32>(texture + 0x58) == 0) {
                            handle = R<u64>(object + 0xf0);
                        } else {
                            handle = R<u64>(R<u64>(texture + 0x108) + 0x108);
                        }
                        if (handle == 0) {
                            handle = R<u64>(R<u64>(G + 0x38) + 0x38);
                        }
                        const u64 looked = HandleLookup(device, handle);
                        lock_list(s);
                        const u32 jobs = R<u32>(s + 0x1a8);
                        const u64 list = R<u64>(s + 0x1a0);
                        for (u32 j = 0; j < jobs; ++j) {
                            const u64 e = list + u64{j} * 0x20;
                            const u64 job = R<u64>(e);
                            const s32 amount = R<s32>(e + 0x18);
                            const u64 rd_a = render_data(R<u64>(job + 0x48));
                            const u64 rd_b = render_data(R<u64>(R<u64>(job + 0x60) + 0x60));
                            if (rd_a == 0 || rd_b == 0 || looked == 0) {
                                continue;
                            }
                            put_cb(1, {{R<u64>(s + 0x120), s, 0x120}});
                            put_srv(0, {{R<u64>(rd_a + 0x28), rd_a, 0x28}, {R<u64>(looked + 0x48), looked, 0x48}});
                            put_uav(0, {{R<u64>(rd_b + 0x30), rd_b, 0x30}});
                            dispatch((static_cast<u32>(amount) + 0xffu) >> 8, WetSiteIntersect);
                        }
                        unlock_list(s);
                    }
                }
                any = true;
            }
            if (any) {
                wait();
            }
        }
    }
    // ---- phase 3: the water modifier's force, and the readback
    const u64 water = R<u64>(F + 0x31ed0);
    if (water != 0 && R<s32>(water + 0x48) != 0 && system_count != 0) {
        const u64 def = R<u64>(R<u64>(G + 0x38) + 0x20);
        bool any = false;
        for (u32 i = 0; i < system_count; ++i) {
            const u64 s = R<u64>(systems + u64{i} * 8);
            if (R<s32>(s + 0x30) == 0) {
                continue;
            }
            any = true;
            const u64 P = R<u64>(s + 0x28);
            if (R<s32>(s + 0x1d4) == 0 || P == 0) {
                continue;
            }
            const u64 box = (static_cast<u64>(static_cast<s64>(R<s32>(s + 0x54)) ^ 1)) * 0x20;
            const u64 hi = s + 0x220 + box, lo = s + 0x210 + box;
            const bool inside = !(skip_gt(R<float>(F + 0x31eb0), R<float>(hi)) || skip_lt(R<float>(F + 0x31ec0), R<float>(lo)) || skip_gt(R<float>(F + 0x31eb4), R<float>(hi + 4)) ||
                                  skip_lt(R<float>(F + 0x31ec4), R<float>(lo + 4)) || skip_gt(R<float>(F + 0x31eb8), R<float>(hi + 8)) || skip_lt(R<float>(F + 0x31ec8), R<float>(lo + 8)));
            u32 flags = 0;
            if (inside || ((flags = R<u32>(s + 0x1b8)) & 4) != 0) {
                if (live) {
                    WetTicketAcquire(s + 0x250);
                }
                lock_list(s);
                const u32 jobs = R<u32>(s + 0x1a8);
                const u64 list = R<u64>(s + 0x1a0);
                for (u32 j = 0; j < jobs; ++j) {
                    const u64 e = list + u64{j} * 0x20;
                    const u64 object = R<u64>(e);
                    u64 second = 0;
                    if (R<u32>(object + 0x68) > 0xd) {
                        const u64 c = R<u64>(R<u64>(object + 0x60) + 0x68);
                        second = c != def ? c : 0;
                    }
                    set_shader(R<u64>(device + 0x1188 + (second != 0 ? 0x10 : 0)));
                    const s32 amount = R<s32>(e + 0x18);
                    std::array<u8, 0x54> block{};
                    WetBuildForce(s, P, water, block);
                    update(R<u64>(s + 0x190), s + 0x130, block.data(), 0x54);
                    const u64 rd_a = render_data(R<u64>(object + 0x48));
                    const u64 rd_w = render_data(water);
                    const u64 rd_b = render_data(R<u64>(R<u64>(object + 0x60) + 0x60));
                    const u64 rd_c = second != 0 ? render_data(second) : 0;
                    if (rd_a == 0 || rd_w == 0 || rd_b == 0 || (second != 0 && rd_c == 0)) {
                        continue;
                    }
                    put_cb(1, {{R<u64>(s + 0x190), s, 0x190}});
                    put_uav(0, {{R<u64>(rd_b + 0x30), rd_b, 0x30}, {second != 0 ? R<u64>(rd_c + 0x30) : 0, second != 0 ? rd_c : 0, 0x30}});
                    put_srv(0, {{R<u64>(rd_a + 0x28), rd_a, 0x28}, {R<u64>(rd_w + 0x28), rd_w, 0x28}});
                    wet_kind = 3;
                    wet_system = s;
                    wet_water = water;
                    wet_block = s + 0x130;
                    dispatch((static_cast<u32>(amount) + 0xffu) >> 8, WetSiteForce);
                }
                if (live) {
                    reinterpret_cast<std::atomic<s32>*>(s + 0x250)->fetch_sub(1, std::memory_order_acq_rel);
                }
                if ((R<u32>(s + 0x1b8) & 4) != 0) {
                    wait();
                }
                unlock_list(s);
                flags = R<u32>(s + 0x1b8);
            }
            if ((flags & 4) != 0) {
                lock_list(s);
                const u32 jobs = R<u32>(s + 0x1a8);
                const u64 list = R<u64>(s + 0x1a0);
                for (u32 j = 0; live && j < jobs; ++j) {
                    const u64 e = list + u64{j} * 0x20;
                    const u64 object = R<u64>(e);
                    if (R<u32>(object + 0x68) <= 0xd) {
                        continue;
                    }
                    const u64 c = R<u64>(R<u64>(object + 0x60) + 0x68);
                    if (c == def || c == 0) {
                        continue;
                    }
                    const u64 rd_c = render_data(c);
                    const u64 staging = R<u64>(e + 8);
                    const u64 destination = R<u64>(e + 0x10);
                    const u32 dwords = R<u32>(e + 0x18);
                    const u64 rd_s = staging != 0 ? render_data(staging) : 0;
                    if (rd_c == 0 || rd_s == 0 || destination == 0) {
                        continue;
                    }
                    wet_dispatches[3].fetch_add(1, std::memory_order_relaxed);
                    vcall(0x88, R<u64>(rd_s + 0x20), R<u64>(rd_c + 0x20), 0, 0, 0);
                    u64 mapped = 0;
                    const u64 status = reinterpret_cast<Virtual>(R<u64>(R<u64>(ctx) + 0x90))(ctx, R<u64>(rd_s + 0x20), 0, 1, 0, reinterpret_cast<u64>(&mapped), 0);
                    if (static_cast<s32>(status) >= 0) {
                        std::memcpy(reinterpret_cast<void*>(destination), reinterpret_cast<const void*>(mapped), u64{dwords} << 2);
                        vcall(0xc8, R<u64>(rd_s + 0x20), 0, 0, 0, 0);
                    }
                }
                if (live) {
                    W<u8>(s + 0x1b8, static_cast<u8>(R<u8>(s + 0x1b8) & 0xfbu));
                }
                unlock_list(s);
            }
        }
        if (any) {
            wait();
            if (live) {
                vcall(0xb8, 0, 0, 0, 0, 0);
            }
        }
    }
    // ---- the end: the lock, and the views are cleared
    if (live) {
        WetLockRelease(Wsys + 0x70);
    }
    const std::array<u64, 2> zeros{};
    set_srv(0, 2, zeros.data());
    set_uav(0, 2, zeros.data());
}

struct WConfig {
    bool on{false};
};
const WConfig& WCfg() {
    static const WConfig config = [] {
        WConfig c;
        const char* e = std::getenv("TLG_WET_GEN");
        c.on = e != nullptr ? *e != '0' : GeneratorsDefaultOn(); // off under TLG_VKB (note 57); before: default on since 2026-10-27 (the owner's play test); TLG_WET_GEN=0 turns it off
        return c;
    }();
    return config;
}

struct WStats {
    ViewStats v;
    u64 eligible{0};
    u64 loops_without_work{0};
};
WStats& WS() {
    static WStats stats;
    return stats;
}

// ------------------------------------------------------------------------------------------------------------------------------- B4: GPU instancing (cull and sort)
// FUN_01ee5130(H = the instance holder, F = the scene): the systems of the scene's instance list (F + 0x29e10) feed the GPU instancing path. Per frame: the holder's constant block, the
// per system ring updates (a memcpy of the CPU copy into the next mapped buffer of a ring of two and UpdateSubresource of it), the systems' uploads (FUN_01c458b0), then the dispatches:
// loop 1 (per system, shader index system + 0xa0), loop 2 (system + 0xa4, with the views and the shader cleared after it and the resource copies the flags 0x20000000 / 0x2000000 ask for),
// and, when the scene's pass mask F[0] is not zero, the scene constant block (F + 0 .. 0x5af copied to H + 0x10 and sent), loop 3 (one dispatch of 9 x count groups per system,
// device + 0x1368), the nine-pass loop 4 (a counter block H + 0x5d0 per pass and system, device + 0x1008 + system + 0xac), a wait and loop 5 (device + 0x1378). The context virtuals are
// table writes (graphics context: shader + 0x4670, constant buffers + 0x500, shader resources + 0x1980, unordered access views + 0x2098), UpdateSubresource and the dispatches
// (return addresses 0x1ee8a76, 0x1ee8ec4, 0x1ee94fc, 0x1ee95c5, 0x1ee9a35). First uses (the holder's objects, a system's render object and buffers, the rebuild flag) are the engine's.
// Decompiled and annotated in documents/tlg/native plan/stages/b4-instance.md.
constexpr u64 InstBegin = 0x01ee5130, InstEnd = 0x01ee9ac8;
constexpr u64 InstSite1 = 0x01ee8a76, InstSite2 = 0x01ee8ec4, InstSite3 = 0x01ee94fc, InstSite4 = 0x01ee95c5, InstSite5 = 0x01ee9a35;

struct InstPlan {
    bool scanned{false};
    bool no_work{false};
    std::vector<const char*> reasons;
    u64 holder{0}, scene{0}, device{0}, ctx{0};
    bool Ok() const { return scanned && reasons.empty(); }
    void Decline(const char* r) {
        for (const char* x : reasons) {
            if (x == r) {
                return;
            }
        }
        reasons.push_back(r);
    }
};

struct InstScanArgs {
    InstPlan* plan;
};

constexpr u32 InstSystemsMax = 4096;
std::array<std::atomic<u64>, 5> inst_dispatches{}; // by loop: 1, 2, 3, 4, 5

void ScanInst(InstScanArgs& a) {
    InstPlan& p = *a.plan;
    const u64 H = p.holder, F = p.scene;
    p.scanned = true;
    if (R<u64>(F + 0x29e10) == 0) {
        p.no_work = true;
        return;
    }
    p.device = R<u64>(Global(GlobalDevice));
    p.ctx = p.device != 0 ? R<u64>(p.device + 0x2d8) : 0;
    if (p.device == 0 || p.ctx == 0) {
        p.Decline("no device context");
        return;
    }
    const u64 object = R<u64>(H);
    const u64 block = R<u64>(H + 8);
    if (object == 0 || block == 0 || R<u64>(object + 0x38) == 0 || R<u64>(block + 0x10) == 0 || R<u64>(block) == 0) {
        p.Decline("the instance holder is not set up yet (the loop creates its objects at first use)");
        return;
    }
    u32 guard = 0;
    for (u64 node = R<u64>(F + 0x29e10); node != 0; node = R<u64>(node), ++guard) {
        if (guard > InstSystemsMax) {
            p.Decline("the system list is not plausible");
            return;
        }
        const u64 sys = R<u64>(node + 8);
        if (sys == 0) {
            p.Decline("a system without its object");
            return;
        }
        const u32 flags = R<u32>(sys + 0x40);
        const u64 rd = R<u64>(sys + 0x38);
        if (static_cast<s32>(flags) < 0 || rd == 0 || (flags & 0x40000000u) != 0) {
            p.Decline("a system's render object is not ready (the loop creates or rebuilds it)");
            return;
        }
        if (R<u64>(rd + 0x28) == 0) {
            p.Decline("a system's render object is not ready (the loop creates or rebuilds it)");
            return;
        }
        if ((flags & 0x20000000u) != 0 && R<u64>(rd + 0x2a0) == 0) {
            p.Decline("a resource copy target is not created yet (the loop creates it at first use)");
            return;
        }
        if ((flags & 0x2000000u) != 0 && R<u64>(rd + 0x2a8) == 0) {
            p.Decline("a resource copy target is not created yet (the loop creates it at first use)");
            return;
        }
        const s32 count = R<s32>(R<u64>(rd + 0x28) + 0x48);
        if (count != 0 && (R<u8>(sys + 0x43) & 1) == 0 && (R<u32>(sys + 0xa0) != 0 || R<u32>(sys + 0xa4) != 0)) {
            // loops 1 and 2: the shapes the engine asserts on are declined (no views, too many views)
            const u32 srvs = R<u32>(rd + 0xa8), uavs = R<u32>(rd + 0xc0);
            if (srvs == 0 || srvs >= 0x80 || uavs == 0 || uavs > 8) {
                p.Decline("a system with a view count the engine asserts on");
                return;
            }
        }
        if (count != 0 && R<u32>(F) != 0) {
            // loop 4: the views of the nine passes
            const s32 n = R<s32>(rd + 0x38);
            const u64 current = R<u64>(R<u64>(rd + 0xb8) + u64{R<u32>(rd + 0x20)} * 8);
            if (n < 0 || n >= 0x7f || R<u32>(rd + 0x50) + 2 > 8 || current == 0) {
                p.Decline("a system with a view count the engine asserts on");
                return;
            }
        }
    }
}

void GenerateInst(Target& t, const InstPlan& p) {
    const u64 H = p.holder, F = p.scene, D = p.device, ctx = p.ctx;
    const bool live = t.live;
    using Virtual = u64(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64);
    const auto vcall = [&](u32 slot, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) { return reinterpret_cast<Virtual>(R<u64>(R<u64>(ctx) + slot))(ctx, a1, a2, a3, a4, a5, 0); };
    constexpr u32 ShaderWord = 0x4670, CbTable = 0x500, SrvTable = 0x1980, UavTable = 0x2098;
    struct Src {
        u64 value;
        u64 owner;
        u32 offset;
    };
    // a table entry with the engine word it was read from, so the verify run can tell a word the engine's other threads replaced between the two reads
    const auto put = [&](u32 address, const Src& src) {
        t.direct_object = src.owner;
        t.direct_offset = src.offset;
        t.Put(address, &src.value, 8);
        t.direct_object = 0;
    };
    const auto set_shader = [&](u64 shader) { t.Set<u64>(ShaderWord, shader); };
    u32 inst_kind = 0;
    // the verify run does not toggle the systems' current buffer (the loop does, before its dispatches): the generator keeps its own copy of the value it would have left
    std::unordered_map<u64, u32> current_buffer;
    const auto current_of = [&](u64 rd) -> u32 {
        const auto it = current_buffer.find(rd);
        return it != current_buffer.end() ? it->second : R<u32>(rd + 0x20);
    };
    const auto dispatch = [&](u32 x, u32 y, u32 z, u64 site) {
        inst_dispatches[site == InstSite1 ? 0 : site == InstSite2 ? 1 : site == InstSite3 ? 2 : site == InstSite4 ? 3 : 4].fetch_add(1, std::memory_order_relaxed);
        if (site == InstSite1) {
            Core::RareEvents::Note("instancing_loop1", fmt::format("a dispatch of loop 1 ({}, {} x {} x {}), shader {:#x}: written from the engine's code, never verified before", live ? "generated" : "verify shadow", x, y, z, t.Get<u64>(ShaderWord)));
        }
        if (live) {
            Timed(CallId::DrawIssue, [&] { NativeDraw::IssueDispatch(ctx, x, y, z, false, site, false); });
        } else {
            t.marks.push_back({t.arena.size(), 3u, x, y, z, t.Get<u64>(ShaderWord), 0});
            t.marks.back().blocks = std::move(t.blocks);
            t.blocks.clear();
            t.marks.back().site = site;
        }
        ++t.draws;
        (void)inst_kind;
    };
    const auto wait = [&] {
        if (live) {
            vcall(0x78, 0, 0, 0, 0, 0);
        }
    };
    // UpdateSubresource of a staged block: live, the bytes are in the engine's memory and the virtual copies from there; the verify run keeps (address, bytes) for the comparison at the next
    // dispatch (a block staged again at the same address replaces the pending one)
    const auto stage = [&](u64 resource, u64 source, const u8* bytes, u32 size) {
        if (!live) {
            const u32 n = std::min<u32>(R<u32>(resource + 0x180), size);
            for (auto& [address, block] : t.blocks) {
                if (address == source) {
                    block.assign(bytes, bytes + n);
                    return;
                }
            }
            t.blocks.push_back({source, std::vector<u8>(bytes, bytes + n)});
        }
    };
    const auto update_live = [&](u64 resource, u64 source) { vcall(0xd0, resource, 0, 0, source, 0); };
    const auto upload_flush = [&](u64 rd, u64 object) {
        if (live) {
            reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u64, u64)>(Global(0x01c458b0))(rd, ctx, object, 0);
        }
    };
    const auto nodes = [&](auto&& fn) {
        for (u64 node = R<u64>(F + 0x29e10); node != 0; node = R<u64>(node)) {
            fn(node, R<u64>(node + 8));
        }
    };

    // ---- the holder's constant block and the first two table words
    const u64 holder_object = R<u64>(H);
    const u64 holder_block = R<u64>(H + 8);
    const u64 holder_rd = R<u64>(holder_object + 0x38);
    upload_flush(holder_rd, holder_object);
    {
        const u64 destination = R<u64>(holder_block + 0x10);
        const u64 source = R<u64>(holder_block);
        const u32 size = R<u32>(holder_block + 8);
        if (live) {
            update_live(destination, source);
        } else {
            std::vector<u8> bytes(size);
            std::memcpy(bytes.data(), reinterpret_cast<const void*>(source), size);
            stage(destination, source, bytes.data(), size);
        }
        put(CbTable, {destination, holder_block, 0x10});
        put(SrvTable, {R<u64>(holder_rd + 0x28), holder_rd, 0x28});
    }
    // ---- the systems' ring updates and uploads
    nodes([&](u64, u64 sys) {
        const u64 rd = R<u64>(sys + 0x38);
        if (R<s32>(R<u64>(rd + 0x28) + 0x48) == 0) {
            return;
        }
        const u32 flags = R<u32>(sys + 0x40);
        if ((flags & 0x1000000u) == 0) {
            if (live) {
                W<u8>(rd + 0x20, static_cast<u8>(R<u8>(rd + 0x20) ^ 1));
            } else {
                current_buffer[rd] = R<u32>(rd + 0x20) ^ 1u;
            }
        }
        const u64 rings[3] = {R<u64>(rd + 0xd0), R<u64>(rd + 0x98), R<u64>(rd + 0x60)};
        for (u32 i = 0; i < 3; ++i) {
            const u64 A = rings[i];
            if (A == 0 || (i > 0 && A == rings[0])) {
                continue;
            }
            const s32 index = R<s32>(A + 0x28);
            const u64 destination = R<u64>(A + 0x18 + static_cast<u64>(static_cast<s64>(index)) * 8);
            const u64 source = R<u64>(A + 8);
            const s32 size = R<s32>(A + 0x10);
            const u64 block = R<u64>(A);
            if (live) {
                std::memcpy(reinterpret_cast<void*>(destination), reinterpret_cast<const void*>(source), static_cast<size_t>(static_cast<s64>(size)));
                W<u64>(block, destination);
                update_live(R<u64>(block + 0x10), destination);
                W<u8>(A + 0x28, static_cast<u8>(R<u8>(A + 0x28) ^ 1));
            } else {
                std::vector<u8> bytes(static_cast<size_t>(size));
                std::memcpy(bytes.data(), reinterpret_cast<const void*>(source), bytes.size());
                stage(R<u64>(block + 0x10), destination, bytes.data(), static_cast<u32>(size));
            }
        }
        for (const u32 array : {0xa0u, 0x68u, 0x30u}) {
            const u32 count = R<u32>(rd + array + 8);
            for (u32 k = 0; k < count; ++k) {
                const u64 entry = R<u64>(R<u64>(rd + array) + u64{k} * 8);
                if (entry != 0) {
                    upload_flush(R<u64>(entry + 0x38), entry);
                }
            }
        }
    });
    // ---- loops 1 and 2: per system, its views, its shader, one dispatch
    const auto fill_views = [&](u64 sys, u64 rd, u32 shader_index, bool clear_after, u64 site) {
        const u64 ring0 = R<u64>(rd + 0xd0);
        const u64 cb1 = ring0 != 0 ? R<u64>(R<u64>(ring0) + 0x10) : 0;
        const u32 n = R<u32>(rd + 0xa8);
        std::array<Src, 128> srv{};
        for (u32 i = 0; i < n; ++i) {
            const u64 entry = R<u64>(R<u64>(rd + 0xa0) + u64{i} * 8);
            srv[i] = {entry != 0 ? R<u64>(R<u64>(entry + 0x38) + 0x28) : 0, entry != 0 ? R<u64>(entry + 0x38) : 0, 0x28};
        }
        const u32 m = R<u32>(rd + 0xc0);
        std::array<Src, 8> uav{};
        for (u32 i = 0; i < m && i < 8; ++i) {
            const u32 index = i < 2 ? (current_of(rd) ^ i) : i;
            const u64 entry = R<u64>(R<u64>(rd + 0xb8) + u64{index} * 8);
            uav[i] = {entry != 0 ? R<u64>(entry + 0x30) : 0, entry, 0x30};
        }
        put(CbTable + 8, {cb1, ring0 != 0 ? R<u64>(ring0) : 0, 0x10});
        for (u32 i = 0; i < m && i < 8; ++i) {
            put(UavTable + i * 8, uav[i]);
        }
        for (u32 i = 0; i < n; ++i) {
            put(SrvTable + 8 + i * 8, srv[i]);
        }
        set_shader(R<u64>(D + 0x1008 + u64{shader_index} * 0x10));
        const s32 count = R<s32>(R<u64>(rd + 0x28) + 0x48);
        dispatch((static_cast<u32>(count) + 0x3fu) >> 6, 1, 1, site);
        if (clear_after) {
            const u64 zero = 0;
            put(CbTable + 8, {0, 0, 0});
            for (u32 i = 0; i < n; ++i) {
                put(SrvTable + 8 + i * 8, {0, 0, 0});
            }
            for (u32 i = 0; i < m && i < 8; ++i) {
                put(UavTable + i * 8, {0, 0, 0});
            }
            (void)zero;
            set_shader(0);
        }
    };
    nodes([&](u64, u64 sys) {
        const u64 rd = R<u64>(sys + 0x38);
        if ((R<u8>(sys + 0x43) & 1) != 0 || R<s32>(R<u64>(rd + 0x28) + 0x48) == 0 || R<u32>(sys + 0xa0) == 0) {
            return;
        }
        fill_views(sys, rd, R<u32>(sys + 0xa0), false, InstSite1);
        if ((R<u8>(sys + 0x42) & 0x80) != 0 && live) {
            W<u32>(sys + 0xa0, 0);
        }
    });
    nodes([&](u64, u64 sys) {
        const u64 rd = R<u64>(sys + 0x38);
        if ((R<u8>(sys + 0x43) & 1) != 0 || R<s32>(R<u64>(rd + 0x28) + 0x48) == 0 || R<u32>(sys + 0xa4) == 0) {
            return;
        }
        fill_views(sys, rd, R<u32>(sys + 0xa4), true, InstSite2);
        u32 flags = R<u32>(sys + 0x40);
        if ((flags & 0x20000000u) != 0) {
            if (live) {
                W<u32>(sys + 0x40, flags & ~0x20000000u);
                vcall(0x88, R<u64>(rd + 0x2a0), R<u64>(rd + 0x2b0), 0, 0, 0);
                W<u32>(rd + 0x2d0, 1);
            }
            flags &= ~0x20000000u;
        }
        if ((flags & 0x2000000u) != 0 && live) {
            Core::RareEvents::Note("instancing_flag_0x2000000", fmt::format("system {:#x}, flags {:#x}: the second resource copy of loop 2 (never verified before)", sys, flags));
            W<u32>(sys + 0x40, flags & ~0x2000000u);
            vcall(0x88, R<u64>(rd + 0x2a8), R<u64>(rd + 0x2b8), 0, 0, 0);
            W<u32>(rd + 0x2d0, 1);
        }
    });
    // ---- the scene's pass mask: the scene block and loops 3 to 5
    const u32 mask = R<u32>(F);
    if (mask != 0) {
        {
            const u64 destination = R<u64>(H + 0x5c0);
            g_inst_scene_block = H + 0x10;
            g_inst_scene_source = F;
            if (live) {
                std::memcpy(reinterpret_cast<void*>(H + 0x10), reinterpret_cast<const void*>(F), 0x5b0);
                update_live(destination, H + 0x10);
            } else {
                stage(destination, H + 0x10, reinterpret_cast<const u8*>(F), 0x5b0);
            }
            put(CbTable + 0x20, {destination, H, 0x5c0});
            put(CbTable + 0x28, {R<u64>(H + 0x5f0), H, 0x5f0});
        }
        nodes([&](u64, u64 sys) {
            if (sys == 0) {
                return;
            }
            const u64 rd = R<u64>(sys + 0x38);
            if (rd == 0 || R<s32>(R<u64>(rd + 0x28) + 0x48) == 0) {
                return;
            }
            set_shader(R<u64>(D + 0x1368));
            put(UavTable, {R<u64>(rd + 0x118), rd, 0x118});
            dispatch(9, R<u32>(rd + 0x298), 1, InstSite3);
        });
        wait();
        for (u32 k = 0; k < 9; ++k) {
            if (((mask >> (k & 0x1f)) & 1) == 0) {
                continue;
            }
            nodes([&](u64, u64 sys) {
                if (sys == 0) {
                    return;
                }
                const u64 rd = R<u64>(sys + 0x38);
                if (rd == 0) {
                    return;
                }
                const s32 count = R<s32>(R<u64>(rd + 0x28) + 0x48);
                if (count == 0) {
                    return;
                }
                // the counter block: H + 0x5d4 = count - 1, H + 0x5d8 = the pass
                const u64 destination = R<u64>(H + 0x5f0);
                const u32 width = std::min<u32>(R<u32>(destination + 0x180), 0x20);
                std::array<u8, 0x20> block{};
                std::memcpy(block.data(), reinterpret_cast<const void*>(H + 0x5d0), 0x20);
                const u32 counter = static_cast<u32>(count - 1);
                std::memcpy(block.data() + 4, &counter, 4);
                std::memcpy(block.data() + 8, &k, 4);
                if (live) {
                    W<u32>(H + 0x5d4, counter);
                    W<u32>(H + 0x5d8, k);
                    update_live(destination, H + 0x5d0);
                } else {
                    stage(destination, H + 0x5d0, block.data(), width);
                }
                // views: shader resource 0 is the object's current buffer, then its list; unordered access 0 / 1 / then its list
                const u32 current = current_of(rd);
                const u64 first = R<u64>(R<u64>(rd + 0xb8) + u64{current} * 8);
                put(SrvTable, {R<u64>(first + 0x28), first, 0x28});
                const s32 n = R<s32>(rd + 0x38);
                for (s32 i = 0; i < n; ++i) {
                    const u64 entry = R<u64>(R<u64>(rd + 0x30) + u64{static_cast<u32>(i)} * 8);
                    put(SrvTable + 8 + static_cast<u32>(i) * 8, {entry != 0 ? R<u64>(R<u64>(entry + 0x38) + 0x28) : 0, entry != 0 ? R<u64>(entry + 0x38) : 0, 0x28});
                }
                put(UavTable, {R<u64>(rd + 0x118), rd, 0x118});
                put(UavTable + 8, {R<u64>(rd + 0x168 + u64{k} * 8), rd, 0x168 + k * 8});
                const u32 extra = R<u32>(rd + 0x50);
                for (u32 i = 0; i < extra; ++i) {
                    const u32 index = i < 2 ? (current ^ i) : i;
                    const u64 entry = R<u64>(R<u64>(rd + 0x48) + u64{index} * 8);
                    put(UavTable + (i + 2) * 8, {entry != 0 ? R<u64>(entry + 0x30) : 0, entry, 0x30});
                }
                if (const u64 ring = R<u64>(rd + 0x60); ring != 0) {
                    put(CbTable + 0x30, {R<u64>(R<u64>(ring) + 0x10), R<u64>(ring), 0x10});
                }
                u32 shader_index = R<u32>(sys + 0xac);
                if (shader_index == 0) {
                    shader_index = 0x33;
                }
                set_shader(R<u64>(D + 0x1008 + u64{shader_index} * 0x10));
                dispatch(1, 1, 1, InstSite4);
            });
        }
        wait();
        nodes([&](u64, u64 sys) {
            if (sys == 0) {
                return;
            }
            const u64 rd = R<u64>(sys + 0x38);
            if (rd == 0 || R<s32>(R<u64>(rd + 0x28) + 0x48) == 0 || R<u32>(rd + 0x298) <= 1) {
                return;
            }
            set_shader(R<u64>(D + 0x1378));
            put(UavTable, {R<u64>(rd + 0x118), rd, 0x118});
            dispatch(9, R<u32>(rd + 0x298) - 1, 1, InstSite5);
        });
        put(SrvTable, {0, 0, 0});
        put(UavTable, {0, 0, 0});
        put(UavTable + 8, {0, 0, 0});
    }
    put(CbTable, {0, 0, 0});
    put(SrvTable, {0, 0, 0});
}

struct IConfig {
    bool on{false};
};
const IConfig& ICfg() {
    static const IConfig config = [] {
        IConfig c;
        const char* e = std::getenv("TLG_INST_GEN");
        c.on = e != nullptr ? *e != '0' : GeneratorsDefaultOn(); // off under TLG_VKB (note 57); before: default on since 2026-10-27 (the owner's play test); TLG_INST_GEN=0 turns it off
        return c;
    }();
    return config;
}

struct IStats {
    ViewStats v;
    u64 eligible{0};
    u64 loops_without_work{0};
};
IStats& IS() {
    static IStats stats;
    return stats;
}




} // namespace

bool Enabled() {
    return Cfg().on;
}



// note 107 census (debug, TLG_VKB_ZPASS_CENSUS=1, from the Z pass hook): the generator's scan, read only and guarded, on the pass about to run (nothing is generated; the engine's loop
// draws). Counts, per view type and pass kind, the primitives of each batch shape (plain, instance group, vertex modifier entry, visibility cache), the pass-level reasons the
// generator gives up for, and the draws and time of the engine's loop that follows (the caller passes them). What a direct path for one ordinary batch shape could take.
namespace {
struct ZCensusRow {
    u64 passes{}, ns{}, draws{}, steps_plain{}, steps_group{}, steps_modifier{}, steps_visibility{}, batches{}, group_batches{}, only_580{};
    std::map<std::string, u64> reasons;
};
std::mutex zc_mutex;
std::map<u64, ZCensusRow> zc_rows; // (view type << 32) | pass kind
std::atomic<u64> zc_calls{0};
const std::unordered_map<u64, TableEntry> zc_empty_table;
} // namespace

bool ZPassCensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_ZPASS_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}

void ZPassCensus(u64 render_device, u64 desc, const std::function<std::pair<u64, u64>()>& run) {
    Plan plan;
    ScanArgs args{&plan, render_device, desc, &zc_empty_table};
    const bool scanned = NativeDraw::GuardedCall([](void* p) { ScanImpl(*static_cast<ScanArgs*>(p)); }, &args); // before the loop: what it is about to draw
    const auto [draws, ns] = run();
    std::scoped_lock lock{zc_mutex};
    ZCensusRow& row = zc_rows[(u64{plan.view_type} << 32) | plan.kind];
    ++row.passes;
    row.ns += ns;
    row.draws += draws;
    if (!scanned) {
        ++row.reasons["scan faulted"];
    }
    for (const char* r : plan.reasons) {
        const std::string reason{r};
        if (reason.find("variant") == std::string::npos) { // (the census has no learned variants: every step says so)
            ++row.reasons[reason];
        }
    }
    row.batches += plan.batches.size();
    for (const BatchLevel& level : plan.batches) {
        row.group_batches += level.group ? 1 : 0;
        row.only_580 += level.only_580 ? 1 : 0;
        u64& bucket = level.group ? row.steps_group : level.special != 0 ? row.steps_modifier : level.visibility ? row.steps_visibility : row.steps_plain;
        bucket += level.step_count;
    }
    if ((zc_calls.fetch_add(1, std::memory_order_relaxed) & 63) == 0) {
        nlohmann::json out = nlohmann::json::object();
        for (const auto& [key, r] : zc_rows) {
            nlohmann::json j = {{"passes", r.passes}, {"ms", static_cast<double>(r.ns) / 1e6}, {"engine_draws", r.draws}, {"batches", r.batches}, {"group_batches", r.group_batches},
                                {"only_580_batches", r.only_580}, {"steps_plain", r.steps_plain}, {"steps_instance_group", r.steps_group}, {"steps_vertex_modifier", r.steps_modifier},
                                {"steps_visibility", r.steps_visibility}, {"pass_reasons", r.reasons}};
            out[fmt::format("view{}_kind{}", key >> 32, key & 0xffffffffull)] = std::move(j);
        }
        DebugServer::Publish("zpass_census", std::move(out));
    }
}


// ------------------------------------------------------------------------------------------------------------------------------- R7 research census
// Scene on the GPU R7 census (debug, TLG_VKB_R7_CENSUS=1, shadow cascades = Z pass views 1, 2): before the engine's loop runs, the generator's scan predicts each draw
// (in order: the batch's vertex view = ctx + 0x580, its table = ctx + 0x6c0 + 8 j, the per-object constant buffer = ctx + 0x288, the index view = ctx + 0x58, the material's
// constant slot = ctx + 0x290, count / first index); each draw the engine then makes is compared field by field. And: does one (engine variant, raster flag, view) always come
// to the same pipeline variant and fixed state of ours? What a submitter that writes rows from the scan's data, without the engine's context, relies on.
namespace {
struct Prediction {
    u64 record{}, material{}, srv0{}, cb1{}, index_view{}, cb2{};
    u32 count{}, first{}, view{}, ntable{};
    bool group{};
    std::array<std::pair<u32, u64>, 16> table{};
};
thread_local std::vector<Prediction> t_pred;
thread_local size_t t_pred_at = 0;
thread_local bool t_pred_on = false;
struct R7Stats {
    std::atomic<u64> passes{0}, scan_faults{0}, predicted{0}, drawn{0}, extra{0}, missing{0}, kind_diff{0}, srv0_diff{0}, table_diff{0}, cb1_diff{0}, index_diff{0}, cb2_diff{0},
        args_diff{0}, all_equal{0}, map_keys{0}, map_ambiguous{0}, map_checked{0}, map_variant_diff{0}, map_fixed_diff{0}, cb2_same_at_draw{0}, record_key_ambiguous{0}, record_keys{0};
};
R7Stats r7;
std::mutex r7_map_mutex;
std::unordered_map<u64, std::pair<const void*, const void*>> r7_map; // (engine variant, raster flag, view) -> (our variant, our fixed state)
u64 MaterialSlot(u64 material) {
    const u64 object = R<u64>(material + 0x38);
    const u64 index = (R<u64>(object + 0x440) & 0xffffffffull) * 0xb0;
    return R<u64>(object + 0xc0 + index);
}
} // namespace

u32 R7FixedBytes = 0; // the size of our fixed state (set by the backend)
void R7SetFixedBytes(u32 bytes) {
    R7FixedBytes = bytes;
}
bool R7CensusOn() {
    static const bool on = [] { const char* e = std::getenv("TLG_VKB_R7_CENSUS"); return e != nullptr && e[0] == '1'; }();
    return on;
}

void R7Predict(u64 render_device, u64 desc) {
    t_pred.clear();
    t_pred_at = 0;
    t_pred_on = false;
    Plan plan;
    ScanArgs args{&plan, render_device, desc, &zc_empty_table};
    r7.passes.fetch_add(1, std::memory_order_relaxed);
    if (!NativeDraw::GuardedCall([](void* p) { ScanImpl(*static_cast<ScanArgs*>(p)); }, &args)) {
        r7.scan_faults.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    struct BuildArgs {
        Plan* plan;
    } build{&plan};
    const bool ok = NativeDraw::GuardedCall([](void* p) {
        Plan& pl = *static_cast<BuildArgs*>(p)->plan;
        for (const BatchLevel& level : pl.batches) {
            if (level.only_580) {
                continue;
            }
            for (u32 s = level.first_step; s < level.first_step + level.step_count; ++s) {
                const Step& step = pl.steps[s];
                if (level.visibility) {
                    if (const u64 node = VisibilityNode(pl.visibility_owner, step.record); node != 0 && R<s32>(node + 8) == 2) {
                        continue; // culled: the loop sets the state and draws nothing
                    }
                }
                Prediction pr;
                pr.record = step.record;
                pr.material = step.material;
                pr.srv0 = level.value_580;
                pr.cb1 = level.pass_object_288;
                pr.index_view = level.view_58;
                pr.cb2 = MaterialSlot(step.material);
                pr.count = static_cast<u32>(R<s32>(step.record + 0x14) * 3);
                pr.first = R<u32>(step.record + 0x10);
                pr.view = pl.view_type;
                pr.group = level.group;
                pr.ntable = level.table_count;
                for (u32 i = 0; i < level.table_count; ++i) {
                    pr.table[i] = level.table[i];
                }
                t_pred.push_back(pr);
            }
        }
    }, &build);
    if (!ok) {
        r7.scan_faults.fetch_add(1, std::memory_order_relaxed);
        t_pred.clear();
        return;
    }
    r7.predicted.fetch_add(t_pred.size(), std::memory_order_relaxed);
    t_pred_on = true;
}

void R7Check(u64 ctx, u32 kind, u64 a1, u64 a2, const void* our_variant, const void* our_fixed) {
    if (!t_pred_on) {
        return;
    }
    r7.drawn.fetch_add(1, std::memory_order_relaxed);
    if (t_pred_at >= t_pred.size()) {
        r7.extra.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const Prediction& p = t_pred[t_pred_at++];
    bool all = true;
    const auto diff = [&](bool d, std::atomic<u64>& counter) {
        if (d) {
            counter.fetch_add(1, std::memory_order_relaxed);
            all = false;
        }
    };
    diff(kind != (p.group ? 1u : 0u) && kind != (p.group ? 2u : 0u), r7.kind_diff);
    diff(R<u64>(ctx + 0x580) != p.srv0, r7.srv0_diff);
    bool table_same = true;
    for (u32 i = 0; i < p.ntable; ++i) {
        table_same = table_same && R<u64>(ctx + 0x6c0 + p.table[i].first * 8) == p.table[i].second;
    }
    diff(!table_same, r7.table_diff);
    diff(R<u64>(ctx + 0x288) != p.cb1, r7.cb1_diff);
    diff(R<u64>(ctx + 0x58) != p.index_view, r7.index_diff);
    diff(R<u64>(ctx + 0x290) != p.cb2, r7.cb2_diff);
    if (R<u64>(ctx + 0x290) != p.cb2 && R<u64>(ctx + 0x290) == MaterialSlot(p.material)) {
        r7.cb2_same_at_draw.fetch_add(1, std::memory_order_relaxed); // the material's slot word changed between the scan and the draw (read at draw time it is the same)
    }
    if (R<u64>(ctx + 0x290) != p.cb2 && R<u64>(ctx + 0x290) != MaterialSlot(p.material)) {
        static std::atomic<u32> shown{0};
        if (shown.fetch_add(1, std::memory_order_relaxed) < 8) {
            const u64 object = R<u64>(p.material + 0x38);
            LOG_WARNING(Render_Vulkan, "R7 census cb2: material {:#x} kind {:#x} flags d4 {:#x} d5 {:#x} object {:#x} index word {:#x}: predicted {:#x}, ctx 0x290 {:#x} 0x298 {:#x} 0x310 {:#x}",
                        p.material, R<u64>(p.material + 0x18), R<u32>(p.material + 0xd4), R<u8>(p.material + 0xd5), object, R<u64>(object + 0x440), p.cb2,
                        R<u64>(ctx + 0x290), R<u64>(ctx + 0x298), R<u64>(ctx + 0x310));
        }
    }
    if (!p.group) {
        diff(static_cast<u32>(a1) != p.count || static_cast<u32>(a2) != p.first, r7.args_diff);
    }
    if (all) {
        r7.all_equal.fetch_add(1, std::memory_order_relaxed);
    }
    // the pipeline map
    const u64 data = R<u64>(p.record + 0x28);
    const u64 engine_variant = data != 0 ? R<u64>(data + 0x48) : 0;
    const bool raster = (R<u32>(p.material + 0xd4) & 0x2e) != 0;
    const u64 key = engine_variant ^ (u64{raster} << 60) ^ (u64{p.view} << 61) ^ (u64{p.group} << 63);
    std::scoped_lock lock{r7_map_mutex};
    {
        static std::unordered_map<u64, std::pair<const void*, const void*>> by_record;
        const u64 rkey = p.record ^ (u64{p.view} << 61) ^ (u64{p.group} << 63) ^ (u64{raster} << 60);
        auto [rit, rfresh] = by_record.try_emplace(rkey, our_variant, our_fixed);
        if (rfresh) {
            r7.record_keys.fetch_add(1, std::memory_order_relaxed);
        } else if (rit->second.first != our_variant || rit->second.second != our_fixed) {
            r7.record_key_ambiguous.fetch_add(1, std::memory_order_relaxed);
            rit->second = {our_variant, our_fixed};
        }
    }
    r7.map_checked.fetch_add(1, std::memory_order_relaxed);
    auto [it, fresh] = r7_map.try_emplace(key, our_variant, our_fixed);
    if (fresh) {
        r7.map_keys.fetch_add(1, std::memory_order_relaxed);
    } else if (it->second.first != our_variant || it->second.second != our_fixed) {
        r7.map_ambiguous.fetch_add(1, std::memory_order_relaxed);
        if (it->second.first != our_variant) {
            r7.map_variant_diff.fetch_add(1, std::memory_order_relaxed);
        }
        if (it->second.second != our_fixed) {
            r7.map_fixed_diff.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<u32> shown{0};
            if (shown.fetch_add(1, std::memory_order_relaxed) < 6) {
                // which bytes of the interned fixed state differ (offset ranges)
                const u8* x = static_cast<const u8*>(it->second.second);
                const u8* y = static_cast<const u8*>(our_fixed);
                std::string where;
                u32 shown_ranges = 0;
                for (u32 i = 0; i < R7FixedBytes && shown_ranges < 12; ++i) {
                    if (x[i] != y[i]) {
                        u32 j = i;
                        while (j + 1 < R7FixedBytes && x[j + 1] != y[j + 1]) {
                            ++j;
                        }
                        where += fmt::format(" +{:#x}..{:#x}", i, j);
                        i = j;
                        ++shown_ranges;
                    }
                }
                LOG_WARNING(Render_Vulkan, "R7 census fixed differs: view {} raster {} group {} engine variant {:#x}: bytes{}", p.view, raster ? 1 : 0, p.group ? 1 : 0, engine_variant, where);
            }
        }
        it->second = {our_variant, our_fixed};
    }
}

void R7Finish() {
    if (t_pred_on && t_pred_at < t_pred.size()) {
        r7.missing.fetch_add(t_pred.size() - t_pred_at, std::memory_order_relaxed);
    }
    t_pred_on = false;
    static std::atomic<u32> n{0};
    if ((n.fetch_add(1, std::memory_order_relaxed) & 15) == 0) {
        DebugServer::Publish("r7_census", {{"passes", r7.passes.load()}, {"scan_faults", r7.scan_faults.load()}, {"predicted", r7.predicted.load()}, {"drawn", r7.drawn.load()},
                                           {"extra_draws", r7.extra.load()}, {"missing_draws", r7.missing.load()}, {"all_fields_equal", r7.all_equal.load()},
                                           {"diff_kind", r7.kind_diff.load()}, {"diff_srv0", r7.srv0_diff.load()}, {"diff_table", r7.table_diff.load()},
                                           {"diff_cb1", r7.cb1_diff.load()}, {"diff_index_view", r7.index_diff.load()}, {"diff_cb2", r7.cb2_diff.load()},
                                           {"diff_args", r7.args_diff.load()}, {"map_keys", r7.map_keys.load()}, {"map_ambiguous", r7.map_ambiguous.load()},
                                           {"map_checked", r7.map_checked.load()}, {"map_variant_diff", r7.map_variant_diff.load()}, {"map_fixed_diff", r7.map_fixed_diff.load()}, {"cb2_same_at_draw", r7.cb2_same_at_draw.load()}, {"record_keys", r7.record_keys.load()}, {"record_key_ambiguous", r7.record_key_ambiguous.load()}});
    }
}

// ------------------------------------------------------------------------------------------------------------------------------- P1: parallel passes
// TLG_VKB_PAR_ZPASS=1 (experiment, branch experimental-native-passes; with TLG_ZPASS_GEN=1 TLG_ZPASS_GEN_VIEWS=1+2): a shadow cascade pass (Z pass, view types 1 and 2) runs on a
// worker thread. The engine's thread only copies the device context (the pass's start state), makes the context's end state (the loop's prologue and epilogue words: what the next
// pass relies on), adds the counters of the worker's earlier passes, takes a block of the frame ring for the worker, puts a placeholder into our stream (Stream::OpenSplice) and
// returns. The worker scans the pass, writes it into its private context copy and draws from it: its jobs fill the placeholder; the resolver waits there, so the GPU order is the
// API order. The per-primitive matrices (FUN_01c92f30) and the instance data (FUN_01ee4e50) go into the worker's own copies of the engine's buffer objects and its ring block
// (the engine's thread keeps renaming the real ones for its own passes). What the scan cannot take yet (a variant not learned, mesh render data or a visibility cache entry not
// made: the engine's loop makes them at first use) is left out of this frame's pass and the next pass of that view runs on the engine's thread (the loop, and the generator learns).
// Another reason the scan declines switches the view back to the engine's thread for good.
namespace {
bool ParOn() { // default on under TLG_VKB since 2026-11-21 (owner play: heaviest and spears saves); TLG_VKB_PAR_ZPASS=0: the engine's loop
    return ParZpassSwitch();
}
std::atomic<u64> c_par_group_publish{0}; // P2: instance group binds made on the engine's thread at hand-overs
constexpr size_t ParCtxBytes = 0x13100; // the device context's words our draw path reads (the highest: + 0x130e8)
constexpr u32 ParSlots = 4;
// the render device's words the worker reads (the highest: the visibility lists at + 0x10988, 0x18 each, list index at + 0x10978): copied at the hand-over like the context.
// The worker used to read the live render device after the engine's thread had moved on to its next pass (a cascade drawn with the next pass's setup: the sun's shadow
// map wrong for a frame or two, owner play 2026-11-21)
constexpr size_t ParRdBytes = 0x11000;
constexpr u32 ParRdMaxList = static_cast<u32>((ParRdBytes - 0x10988) / 0x18) - 1;
constexpr u64 ParArenaMin = 64 << 10, ParArenaMax = 2 << 20;
struct ParTask {
    u64 rd{}, desc{};
    u32 kind{0}; // 0 a shadow cascade (Z pass), 1 a G-buffer pass
    u64 mode{}, list{}, args{}, flags{}, query_mode{};
    u32 view{}, slot{};
    std::shared_ptr<std::unordered_map<u64, std::pair<u64, u64>>> moved; // P2 census (TLG_VKB_PAR_MOVED=1)
    u64 v_b60{0}; // P2: the G-buffer pass's texture at ctx + 0xb60, looked up on the engine's thread (RenderDev_GetGpuTexture refreshes this texture at each lookup)
    Core::Vkb::Stream::Splice* splice{};
    u8* arena{};
    u64 arena_size{};
};
struct ParState {
    std::mutex m;
    std::condition_variable cv;
    std::deque<ParTask> tasks;
    std::array<std::vector<u8>, ParSlots> snapshots;
    std::array<std::vector<u8>, ParSlots> rd_snapshots; ///< the render device at the hand-over (ParRdBytes)
    std::array<std::array<u8, 0x200>, ParSlots> rs_objects{}; ///< the resource set's buffer object at the hand-over
    std::array<u32, ParSlots> rs_stage_bytes{};
    // the pass's draw list as the hand-over left it: the engine may rebuild or reuse the list once its loop has returned (a cascade scanned late missed 34 or 58 casters:
    // the sun's shadow map wrong for two frames, owner play 2026-11-21). desc (0x40 bytes) -> list header (0x20) -> the batch pointer array, copied; the batches are the
    // engine's objects (read in place)
    struct ListCopy {
        std::array<u8, 0x40> desc{};
        std::array<u8, 0x20> list{};
        std::vector<u64> batches;
        u64 live_list{}; ///< the engine's list (census: changed by the time the worker reads it?)
    };
    std::array<ListCopy, ParSlots> lists;
    // the constant buffers bound in the context at the hand-over, frozen: the draw path reads a constant buffer's data address from its engine object (+ 0x60) at the draw,
    // and the engine's thread renames the same object for its next pass (both sun cascades drawn with one cascade's view constants, owner play 2026-11-21, spears).
    // Per slot: up to ParCbClones copies of engine buffer objects in the game heap (the backend tells engine objects by address)
    static constexpr u32 ParCbClones = 80;
    std::array<u8*, ParSlots> cb_clones{};
    std::atomic<u64> cbs_frozen{0};
    std::atomic<u64> list_changed{0}, list_checked{0};
    std::array<std::atomic<bool>, ParSlots> busy{};
    std::array<std::atomic<bool>, 16> serial_next{}, view_off{};
    // issue 130 (floors): a G-buffer frame has several passes, all on view 15, so one per-view "run the engine's loop next" flag was taken by whichever pass came next
    // and the pass that needed the loop (to create its variants / render data) never got it: its pieces stayed out for good (s17_SentinelTower roof floors). The request
    // is kept per pass (the engine's list header) instead; g_held counts passes the engine's thread kept (pieces to learn found before the hand-over)
    std::mutex g_serial_mutex;
    std::vector<u64> g_serial;
    std::atomic<u64> g_held{0}, g_learn_asked{0};
    std::array<std::atomic<u64>, 16> need{};
    std::atomic<s64> pend_draws{0}, pend_triangles{0};
    std::atomic<u64> calls{0}, passes{0}, no_slot{0}, copy_faults{0}, scan_faults{0}, worker_faults{0}, learn_frames{0}, views_off{0};
    std::atomic<u64> steps{0}, drawn{0}, refused{0}, unlearned{0}, rs_fail{0}, group_fail{0}, arena_short{0}, worker_ns{0};
    std::atomic<u64> moved_checked{0}, moved_cb{0}, moved_vb{0};
    std::mutex reasons_mutex;
    std::map<std::string, u64> reasons;
    struct WorkerMem {
        std::vector<u8> rs_clone = std::vector<u8>(0x200), group_clone = std::vector<u8>(0x200), rs_stage = std::vector<u8>(4096);
        // the copies of engine buffer objects live in the game's heap: our backend tells engine objects from anything else by address (IsHeapPointer; ObjectClass of
        // anything else is "not an engine object", which picks another pipeline and leaves the slot unbound)
        u8* rs_object{nullptr};
        u8* group_object{nullptr};
    };
    std::array<WorkerMem, 4> mem; // per worker thread
    bool started{false};
};
ParState& PS() {
    static ParState state;
    return state;
}
bool ParGOn() { // P2: the G-buffer passes on the workers too; default under TLG_VKB again since 2026-11-27, TLG_VKB_PAR_GBUFFER=0: off
    return ParGbufferSwitch();
}
constexpr u32 ParGView = 15; // the G-buffer's entry in the per-view switches
bool Learnable(const char* reason) {
    for (const char* prefix : {"mesh render data not created yet", "variant key not learned yet", "visibility cache entry missing", "second render data not created yet",
                               "primitive without a cached variant"}) {
        if (std::strncmp(reason, prefix, std::strlen(prefix)) == 0) {
            return true;
        }
    }
    return false;
}
u32 ParWorkers() { // TLG_VKB_PAR_WORKERS (1 to 4, default 2 since 2026-11-21: shadows alone gained nothing from 2, 121.0 vs 124.3; with the G-buffer passes 2 is the played set)
    static const u32 n = [] { const char* e = std::getenv("TLG_VKB_PAR_WORKERS"); return e != nullptr ? std::clamp<u32>(static_cast<u32>(std::atoi(e)), 1, 4) : 2u; }();
    return n;
}
void ParTarget(Target& t, const ParTask& task, u32 index) {
    ParState& ps = PS();
    t.live = true;
    t.worker = true;
    t.base = ps.snapshots[task.slot].data();
    t.ring = task.arena;
    t.arena_size = task.arena_size;
    t.rs_clone = ps.mem[index].rs_object != nullptr ? ps.mem[index].rs_object : ps.mem[index].rs_clone.data();
    t.group_clone = ps.mem[index].group_object != nullptr ? ps.mem[index].group_object : ps.mem[index].group_clone.data();
    t.rs_stage = ps.mem[index].rs_stage.data();
}
void ParAskLoopFor(u64 list) { // the G-buffer pass with this list runs the engine's loop at its next hand-over
    ParState& ps = PS();
    std::scoped_lock lock{ps.g_serial_mutex};
    if (std::find(ps.g_serial.begin(), ps.g_serial.end(), list) == ps.g_serial.end() && ps.g_serial.size() < 64) {
        ps.g_serial.push_back(list);
    }
    ps.g_learn_asked.fetch_add(1, std::memory_order_relaxed);
}
bool ParTakeLoopFor(u64 list) {
    ParState& ps = PS();
    std::scoped_lock lock{ps.g_serial_mutex};
    if (const auto it = std::find(ps.g_serial.begin(), ps.g_serial.end(), list); it != ps.g_serial.end()) {
        ps.g_serial.erase(it);
        return true;
    }
    return false;
}
void ParReasons(const std::vector<const char*>& reasons, u32 view, u64 g_list = 0) {
    ParState& ps = PS();
    bool learn = false, off = false;
    for (const char* r : reasons) {
        (Learnable(r) ? learn : off) = true;
        std::scoped_lock lock{ps.reasons_mutex};
        ++ps.reasons[r];
    }
    if (off) {
        ps.view_off[view].store(true, std::memory_order_relaxed);
        ps.views_off.fetch_add(1, std::memory_order_relaxed);
    } else if (learn) {
        if (g_list != 0) {
            ParAskLoopFor(g_list);
        } else {
            ps.serial_next[view].store(true, std::memory_order_relaxed);
        }
        ps.learn_frames.fetch_add(1, std::memory_order_relaxed);
    }
}
void ParDone(const Target& t, const ParTask& task, u64 steps) {
    ParState& ps = PS();
    ps.pend_draws.fetch_add(t.draws, std::memory_order_relaxed);
    ps.pend_triangles.fetch_add(t.triangles, std::memory_order_relaxed);
    ps.need[task.view].store(t.arena_need, std::memory_order_relaxed);
    if (t.arena_need > t.arena_size) {
        ps.arena_short.fetch_add(1, std::memory_order_relaxed);
    }
    ps.steps.fetch_add(steps, std::memory_order_relaxed);
    ps.drawn.fetch_add(static_cast<u64>(t.draws), std::memory_order_relaxed);
    ps.refused.fetch_add(t.refused, std::memory_order_relaxed);
    ps.unlearned.fetch_add(t.unlearned, std::memory_order_relaxed);
    ps.rs_fail.fetch_add(t.rs_fail, std::memory_order_relaxed);
    ps.group_fail.fetch_add(t.group_fail, std::memory_order_relaxed);
}
u64 ParListFor(u32 slot, bool desc);
void ParWorkG(ParTask& task, u32 index) {
    ParState& ps = PS();
    GPlan plan;
    plan.mode = static_cast<u32>(task.mode);
    plan.flags = static_cast<u32>(task.flags);
    plan.query_mode = static_cast<u32>(task.query_mode);
    const u64 rd = reinterpret_cast<u64>(ps.rd_snapshots[task.slot].data()); // the render device as the hand-over left it
    GScanArgs scan_args{&plan, rd, ParListFor(task.slot, false), task.args, true};
    if (!NativeDraw::GuardedCall([](void* p) { ScanGBuffer(*static_cast<GScanArgs*>(p)); }, &scan_args)) {
        plan.scanned = false;
    }
    t_par_lookup_failed = false;
    plan.v_b60 = task.v_b60;
    if (!plan.scanned) {
        ps.scan_faults.fetch_add(1, std::memory_order_relaxed);
        ParAskLoopFor(task.list);
        return;
    }
    ParReasons(plan.reasons, ParGView, task.list);
    struct Args {
        Target t;
        const GPlan* plan;
        u64 rd;
        const u8* rs_object;
    } args{};
    ParTarget(args.t, task, index);
    args.t.moved_check = task.moved.get();
    args.plan = &plan;
    args.rd = rd;
    args.rs_object = ps.rs_objects[task.slot].data();
    if (!NativeDraw::GuardedCall(
            [](void* p) {
                auto* a = static_cast<Args*>(p);
                const u64 rs_buffer = R<u64>(a->rd + 0x2910 + 0xef0);
                if (rs_buffer != 0) {
                    std::memcpy(a->t.rs_clone, a->rs_object, 0x200); // (the object as the hand-over left it)
                    const u32 size = std::min<u32>(R<u32>(reinterpret_cast<u64>(a->rs_object) + 0x180), 4096);
                    std::memcpy(a->t.rs_stage, reinterpret_cast<const void*>(a->rd + 0x2910 + 0xe80), size);
                } else {
                    a->t.rs_clone = nullptr;
                }
                GenerateGBuffer(a->t, *a->plan, a->plan->flags);
            },
            &args)) {
        ps.worker_faults.fetch_add(1, std::memory_order_relaxed);
    }
    ps.moved_checked.fetch_add(args.t.moved_checked, std::memory_order_relaxed);
    ps.moved_cb.fetch_add(args.t.moved_cb, std::memory_order_relaxed);
    ps.moved_vb.fetch_add(args.t.moved_vb, std::memory_order_relaxed);
    ParDone(args.t, task, plan.steps.size());
}
void ParWork(ParTask& task, u32 index) {
    if (task.kind == 1) {
        ParWorkG(task, index);
        return;
    }
    ParState& ps = PS();
    State& s = S();
    Plan plan;
    const u64 rd = reinterpret_cast<u64>(ps.rd_snapshots[task.slot].data()); // the render device as the hand-over left it
    ScanArgs scan_args{&plan, rd, ParListFor(task.slot, true), &s.table, true};
    {
        std::scoped_lock lock{s.mutex};
        CacheArgs cache_args{&s, rd};
        (void)NativeDraw::GuardedCall([](void* p) { CheckCacheImpl(*static_cast<CacheArgs*>(p)); }, &cache_args);
        if (!NativeDraw::GuardedCall([](void* p) { ScanImpl(*static_cast<ScanArgs*>(p)); }, &scan_args)) {
            plan.scanned = false;
        }
    }
    if (!plan.scanned) {
        ps.scan_faults.fetch_add(1, std::memory_order_relaxed);
        ps.serial_next[task.view].store(true, std::memory_order_relaxed);
        return;
    }
    t_par_lookup_failed = false;
    ParReasons(plan.reasons, task.view);
    struct Args {
        Target t;
        const Plan* plan;
        u64 rd;
        const u8* rs_object;
    } args{};
    Target& t = args.t;
    ParTarget(t, task, index);
    args.plan = &plan;
    args.rd = rd;
    args.rs_object = ps.rs_objects[task.slot].data();
    if (!NativeDraw::GuardedCall(
            [](void* p) {
                auto* a = static_cast<Args*>(p);
                // the worker's copies of the resource set's buffer object and of its staging block (renamed / filled per primitive below)
                const u64 rs_buffer = R<u64>(a->rd + 0x2910 + 0xef0);
                if (rs_buffer != 0) {
                    std::memcpy(a->t.rs_clone, a->rs_object, 0x200); // (the object as the hand-over left it)
                    const u32 size = std::min<u32>(R<u32>(reinterpret_cast<u64>(a->rs_object) + 0x180), 4096);
                    std::memcpy(a->t.rs_stage, reinterpret_cast<const void*>(a->rd + 0x2910 + 0xe80), size);
                } else {
                    a->t.rs_clone = nullptr;
                }
                Generate(a->t, *a->plan);
            },
            &args)) {
        ps.worker_faults.fetch_add(1, std::memory_order_relaxed);
    }
    ParDone(t, task, plan.steps.size());
}
void ParWorkerMain(u32 index) {
#ifdef _WIN32
    SetThreadDescription(GetCurrentThread(), index == 0 ? L"vkb:Pass1" : index == 1 ? L"vkb:Pass2" : index == 2 ? L"vkb:Pass3" : L"vkb:Pass4");
#endif
    ParState& ps = PS();
    t_par_worker = true;
    for (;;) {
        ParTask task;
        {
            std::unique_lock lock{ps.m};
            ps.cv.wait(lock, [&] { return !ps.tasks.empty(); });
            task = ps.tasks.front();
            ps.tasks.pop_front();
        }
        const auto t0 = std::chrono::steady_clock::now();
        Core::Vkb::Graphics::SetDepthPass(task.kind == 0);
        Core::Vkb::Stream::SinkTo(task.splice);
        ParWork(task, index);
        Core::Vkb::Graphics::FlushBatch();
        Core::Vkb::Graphics::SetDepthPass(false);
        Core::Vkb::Stream::SinkTo(nullptr);
        Core::Vkb::Stream::CloseSplice(task.splice);
        ps.busy[task.slot].store(false, std::memory_order_release);
        ps.worker_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
    }
}
void ParPublish() {
    ParState& ps = PS();
    nlohmann::json reasons = nlohmann::json::object();
    {
        std::scoped_lock lock{ps.reasons_mutex};
        for (const auto& [r, n] : ps.reasons) {
            reasons[r] = n;
        }
    }
    nlohmann::json out = {{"calls", ps.calls.load()}, {"passes", ps.passes.load()}, {"no_slot", ps.no_slot.load()}, {"copy_faults", ps.copy_faults.load()},
                          {"scan_faults", ps.scan_faults.load()}, {"worker_faults", ps.worker_faults.load()}, {"learn_frames", ps.learn_frames.load()}, {"g_held", ps.g_held.load()}, {"g_learn_asked", ps.g_learn_asked.load()},{"list_checked", ps.list_checked.load()}, {"cbs_frozen", ps.cbs_frozen.load()}, {"list_changed", ps.list_changed.load()},
                          {"views_off", ps.views_off.load()}, {"steps", ps.steps.load()}, {"drawn", ps.drawn.load()}, {"refused", ps.refused.load()},
                          {"unlearned", ps.unlearned.load()}, {"rs_fail", ps.rs_fail.load()}, {"group_fail", ps.group_fail.load()}, {"arena_short", ps.arena_short.load()},
                          {"worker_ms", static_cast<double>(ps.worker_ns.load()) / 1e6}, {"reasons", reasons},
                          {"groups_published", c_par_group_publish.load()}, {"moved_checked", ps.moved_checked.load()}, {"moved_cb", ps.moved_cb.load()}, {"moved_vb", ps.moved_vb.load()}, {"draw", NativeDraw::WorkerDrawStats()}};
    DebugServer::Publish("par_zpass", std::move(out));
}
/// P2: the engine-visible part of the pass's instance group binds (FUN_01ee4e50's uploader: the staged data into a new ring block, the node's buffer renamed, the
/// double buffer toggled), made on the engine's thread at the hand-over, in the pass's batch order: later passes read a group's buffer as the pass left it. The worker draws
/// from its own copies of the same staged data.
void ParPublishGroups(u64 list, u32 stamp) {
    struct Args {
        u64 list;
        u32 stamp;
    } args{list, stamp};
    (void)NativeDraw::GuardedCall(
        [](void* p) {
            auto* a = static_cast<Args*>(p);
            const u64 array = R<u64>(a->list);
            const u32 count = R<u32>(a->list + 8);
            const u64 holder = R<u64>(Global(GlobalDeviceHolder));
            for (u32 b = 0; b < count && b < 100000; ++b) {
                const u64 batch = R<u64>(array + u64{b} * 8);
                const u64 group = batch != 0 && R<u64>(batch + 0x20) != 0 ? R<u64>(batch + 0x38) : 0;
                const u64 group_inner = group != 0 ? R<u64>(group + 8) : 0;
                const u64 group_object = group_inner != 0 ? R<u64>(group_inner + 0x38) : 0;
                if (group_object == 0 || R<u32>(R<u64>(group_object + 0x28) + 0x48) == 0) {
                    continue;
                }
                const u64 uploader = R<u64>(group_object + 0xf0);
                if (uploader == 0) {
                    continue;
                }
                const u64 node = R<u64>(uploader);
                const u64 buffer = R<u64>(node + 0x10);
                const s32 index = R<s32>(uploader + 0x28);
                const u64 value = R<u64>(uploader + 0x18 + static_cast<u64>(index & 1) * 8);
                const u64 staged = R<u64>(uploader + 8);
                const s32 staged_bytes = R<s32>(uploader + 0x10);
                if ((index != 0 && index != 1) || buffer == 0 || R<s32>(buffer + 0x124) != 1 || (R<u8>(buffer + 0x188) & 4) == 0 || R<u32>(buffer + 0x180) == 0 ||
                    staged_bytes < 0 || value == 0) {
                    continue;
                }
                std::memcpy(reinterpret_cast<void*>(value), reinterpret_cast<const void*>(staged), static_cast<size_t>(staged_bytes));
                W<u64>(node, value);
                const u32 size = R<u32>(buffer + 0x180);
                void* block = reinterpret_cast<void*>(reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u64)>(Global(FunctionAllocFrameGarlic))(R<u64>(holder + 0x10238), size, 0x10));
                std::memcpy(block, reinterpret_cast<const void*>(value), size);
                WriteRename(buffer, reinterpret_cast<u64>(block), size, a->stamp);
                W<u8>(uploader + 0x28, R<u8>(uploader + 0x28) ^ 1);
                c_par_group_publish.fetch_add(1, std::memory_order_relaxed);
            }
        },
        &args);
}
/// A free context copy slot with the context copied into it (ParSlots: none)
u32 ParSnapshot(u64 ctx, u64 rd) {
    ParState& ps = PS();
    u32 slot = ParSlots;
    for (u32 i = 0; i < ParSlots; ++i) {
        bool expected = false;
        if (ps.busy[i].compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            slot = i;
            break;
        }
    }
    if (slot == ParSlots) {
        ps.no_slot.fetch_add(1, std::memory_order_relaxed);
        return ParSlots;
    }
    std::vector<u8>& snap = ps.snapshots[slot];
    if (snap.size() != ParCtxBytes) {
        snap.assign(ParCtxBytes, 0);
    }
    std::vector<u8>& rd_snap = ps.rd_snapshots[slot];
    if (rd_snap.size() != ParRdBytes) {
        rd_snap.assign(ParRdBytes, 0);
    }
    struct CopyArgs {
        u8* dst;
        u64 src;
        u8* rd_dst;
        u64 rd;
        u8* rs_dst;
        bool ok;
    } copy{snap.data(), ctx, rd_snap.data(), rd, ps.rs_objects[slot].data(), false};
    if (!NativeDraw::GuardedCall(
            [](void* p) {
                auto* c = static_cast<CopyArgs*>(p);
                std::memcpy(c->dst, reinterpret_cast<const void*>(c->src), ParCtxBytes);
                std::memcpy(c->rd_dst, reinterpret_cast<const void*>(c->rd), ParRdBytes);
                if (const u64 rs_buffer = R<u64>(c->rd + 0x2910 + 0xef0); rs_buffer != 0) {
                    std::memcpy(c->rs_dst, reinterpret_cast<const void*>(rs_buffer), 0x200);
                }
                c->ok = R<u32>(c->rd + 0x10978) <= ParRdMaxList;
            },
            &copy) ||
        !copy.ok) {
        ps.busy[slot].store(false, std::memory_order_release);
        ps.copy_faults.fetch_add(1, std::memory_order_relaxed);
        return ParSlots;
    }
    {
        std::scoped_lock lock{ps.m};
        if (!ps.started) {
            ps.started = true;
            using GuestMalloc = u64(PS4_SYSV_ABI*)(u64);
            for (auto& mem : ps.mem) { // the game's malloc (libc thunk 0x024bdc18), on the engine's thread
                const u64 a = reinterpret_cast<GuestMalloc>(Global(0x024bdc18))(0x200);
                const u64 b = reinterpret_cast<GuestMalloc>(Global(0x024bdc18))(0x200);
                mem.rs_object = IsHeapPointer(a) ? reinterpret_cast<u8*>(a) : nullptr;
                mem.group_object = IsHeapPointer(b) ? reinterpret_cast<u8*>(b) : nullptr;
                LOG_INFO(Core, "P2: worker buffer object copies in the game heap: {:#x} {:#x}", a, b);
            }
            for (auto& pool : ps.cb_clones) {
                const u64 a = reinterpret_cast<GuestMalloc>(Global(0x024bdc18))(u64{ParState::ParCbClones} * 0x200);
                pool = IsHeapPointer(a) ? reinterpret_cast<u8*>(a) : nullptr;
            }
            for (u32 i = 0; i < ParWorkers(); ++i) {
                std::thread{ParWorkerMain, i}.detach();
            }
        }
    }
    if (u8* pool = ps.cb_clones[slot]; pool != nullptr) {
        struct FreezeArgs {
            u8* snap;
            u8* pool;
            u32 used;
            std::array<u64, ParState::ParCbClones> originals;
        } f{snap.data(), pool, 0, {}};
        (void)NativeDraw::GuardedCall(
            [](void* p) {
                auto* f = static_cast<FreezeArgs*>(p);
                for (u32 stage = 0; stage < 5; ++stage) {
                    for (u32 i = 0; i < 16; ++i) {
                        const u64 at = 0x280 + stage * 0x80 + i * 8;
                        u64 object = 0;
                        std::memcpy(&object, f->snap + at, 8);
                        if (object == 0 || !IsHeapPointer(object) || (object & 7) != 0) {
                            continue;
                        }
                        u64 clone = 0;
                        for (u32 k = 0; k < f->used; ++k) { // (the same object in two slots: one clone)
                            if (f->originals[k] == object) {
                                clone = reinterpret_cast<u64>(f->pool + k * 0x200);
                                break;
                            }
                        }
                        if (clone == 0) {
                            if (f->used == ParState::ParCbClones) {
                                continue;
                            }
                            u8* c = f->pool + f->used * 0x200;
                            std::memcpy(c, reinterpret_cast<const void*>(object), 0x200);
                            f->originals[f->used++] = object;
                            clone = reinterpret_cast<u64>(c);
                        }
                        std::memcpy(f->snap + at, &clone, 8);
                    }
                }
            },
            &f);
        ps.cbs_frozen.fetch_add(f.used, std::memory_order_relaxed);
    }
    return slot;
}
/// Copies the pass's draw list into the slot (the engine's thread, at the hand-over). `desc` 0: a G-buffer pass (only the list). False: unreadable (no hand-over).
bool ParCopyList(u32 slot, u64 desc, u64 list) {
    ParState::ListCopy& c = PS().lists[slot];
    struct Args {
        ParState::ListCopy* c;
        u64 desc, list;
        bool ok;
    } a{&c, desc, list, false};
    (void)NativeDraw::GuardedCall(
        [](void* p) {
            auto* a = static_cast<Args*>(p);
            ParState::ListCopy& c = *a->c;
            if (a->desc != 0) {
                std::memcpy(c.desc.data(), reinterpret_cast<const void*>(a->desc), c.desc.size());
                a->list = R<u64>(a->desc);
            }
            c.live_list = a->list;
            if (a->list == 0) {
                a->ok = true; // (the scan declines it)
                return;
            }
            std::memcpy(c.list.data(), reinterpret_cast<const void*>(a->list), c.list.size());
            const u64 array = R<u64>(a->list);
            const u32 count = R<u32>(a->list + 8);
            if (array == 0 || count > 100000) {
                a->ok = true; // (the scan declines it)
                return;
            }
            c.batches.resize(count);
            std::memcpy(c.batches.data(), reinterpret_cast<const void*>(array), u64{count} * 8);
            const u64 copied = reinterpret_cast<u64>(c.batches.data());
            std::memcpy(c.list.data(), &copied, 8);
            a->ok = true;
        },
        &a);
    if (!a.ok) {
        return false;
    }
    if (desc != 0) {
        const u64 list_copy = c.live_list != 0 ? reinterpret_cast<u64>(c.list.data()) : 0;
        std::memcpy(c.desc.data(), &list_copy, 8);
    }
    return true;
}
/// The worker: the copied desc / list for the scan; counts whether the engine's list changed since the hand-over
u64 ParListFor(u32 slot, bool desc) {
    ParState& ps = PS();
    ParState::ListCopy& c = ps.lists[slot];
    struct Args {
        ParState::ListCopy* c;
        bool changed;
    } a{&c, false};
    if (c.live_list != 0 && NativeDraw::GuardedCall(
                                [](void* p) {
                                    auto* a = static_cast<Args*>(p);
                                    const ParState::ListCopy& c = *a->c;
                                    const u64 array = R<u64>(c.live_list);
                                    const u32 count = R<u32>(c.live_list + 8);
                                    a->changed = count != c.batches.size() ||
                                                 (array != 0 && count <= 100000 && std::memcmp(reinterpret_cast<const void*>(array), c.batches.data(), u64{count} * 8) != 0);
                                },
                                &a)) {
        ps.list_checked.fetch_add(1, std::memory_order_relaxed);
        if (a.changed) {
            ps.list_changed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (desc) {
        return reinterpret_cast<u64>(c.desc.data());
    }
    return c.live_list != 0 ? reinterpret_cast<u64>(c.list.data()) : 0;
}
/// The shared tail of a hand-over: the counters of the worker's earlier passes, its ring block, the placeholder, the task
void ParSend(ParTask task) {
    ParState& ps = PS();
    const u64 rd = task.rd;
    if (const s64 d = ps.pend_draws.exchange(0, std::memory_order_relaxed); d != 0) {
        W<s32>(rd + 0x10, R<s32>(rd + 0x10) + static_cast<s32>(d));
    }
    if (const s64 tr = ps.pend_triangles.exchange(0, std::memory_order_relaxed); tr != 0) {
        W<s32>(rd + 0x14, R<s32>(rd + 0x14) + static_cast<s32>(tr));
    }
    const u64 bytes = std::clamp<u64>(ps.need[task.view].load(std::memory_order_relaxed) * 2, ParArenaMin, ParArenaMax);
    const u64 holder = R<u64>(Global(GlobalDeviceHolder));
    task.arena = reinterpret_cast<u8*>(reinterpret_cast<u64(PS4_SYSV_ABI*)(u64, u64, u64)>(Global(FunctionAllocFrameGarlic))(R<u64>(holder + 0x10238), bytes, 0x100));
    task.arena_size = task.arena != nullptr ? bytes : 0;
    // debug (TLG_VKB_PAR_INLINE=1): the worker's code runs here, on the engine's thread, without a placeholder (the worker's inputs without another thread)
    static const bool inline_run = [] { const char* e = std::getenv("TLG_VKB_PAR_INLINE"); return e != nullptr && e[0] == '1'; }();
    if (inline_run) {
        const bool was = t_par_worker;
        t_par_worker = true;
        Core::Vkb::Graphics::SetDepthPass(task.kind == 0);
        ParWork(task, 3);
        static const bool copy_back = [] { const char* e = std::getenv("TLG_VKB_PAR_COPYBACK"); return e != nullptr && e[0] == '1'; }();
        if (copy_back) { // debug: the context words the pass loop writes (0 .. 0xd80) as the worker's copy left them (is the end state the difference?)
            std::memcpy(reinterpret_cast<void*>(R<u64>(task.rd + 0x2d8)), ps.snapshots[task.slot].data(), 0xd80);
        }
        Core::Vkb::Graphics::FlushBatch();
        Core::Vkb::Graphics::SetDepthPass(false);
        t_par_worker = was;
        ps.busy[task.slot].store(false, std::memory_order_release);
        ps.passes.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    task.splice = Core::Vkb::Stream::OpenSplice();
    {
        std::scoped_lock lock{ps.m};
        ps.tasks.push_back(task);
    }
    ps.cv.notify_one();
    ps.passes.fetch_add(1, std::memory_order_relaxed);
    // debug (TLG_VKB_PAR_SYNC=1, no gain by design): the engine's thread waits until the worker is done (rules a timing race in or out)
    static const bool sync = [] { const char* e = std::getenv("TLG_VKB_PAR_SYNC"); return e != nullptr && e[0] == '1'; }();
    if (sync) {
        while (ps.busy[task.slot].load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
}
/// P2: a G-buffer pass (RenderDev_DrawGBufferPrimitives) handed to the worker. The context's end state as the loop leaves it: the stage mode word, ctx + 0x68 (the last
/// batch with a mesh: its group's word or 0), the epilogue (ctx + 0x580 and + 0x980, 0x400 bytes each, cleared; the default raster state; the flags state object and the last
/// batch's flags value when the pass has flags)
bool ParKickG(u64 rd, u64 mode, u64 list, u64 args, u64 flags, u64 query_mode) {
    ParState& ps = PS();
    if ((ps.calls.fetch_add(1, std::memory_order_relaxed) & 255) == 0) {
        ParPublish();
    }
    if (mode == 1 || query_mode == 1 || query_mode == 2 || query_mode == 5 || list == 0) {
        return false;
    }
    if (ps.view_off[ParGView].load(std::memory_order_relaxed) || ParTakeLoopFor(list)) {
        return false;
    }
    struct Head {
        u64 rd, list, flags, ctx, v68;
        u32 last_flags;
        bool ok, learn;
    } head{rd, list, flags, 0, 0, 0, false, false};
    if (!NativeDraw::GuardedCall(
            [](void* p) {
                auto* h = static_cast<Head*>(p);
                h->ctx = R<u64>(h->rd + 0x2d8);
                const u64 array = R<u64>(h->list);
                const u32 count = R<u32>(h->list + 8);
                if (h->ctx == 0 || (count != 0 && (array == 0 || count > 100000))) {
                    return;
                }
                const s32 indirect_mode = R<s32>(h->rd + 0x38e0);
                h->v68 = R<u64>(h->rd + 0x3d90);
                h->last_flags = static_cast<u32>(h->flags);
                for (u32 b = count; b-- > 0;) {
                    const u64 batch = R<u64>(array + u64{b} * 8);
                    if (batch == 0) {
                        continue;
                    }
                    if (b == count - 1) {
                        h->last_flags = (static_cast<u32>(R<u16>(batch + 0x66) & 4) << 4) | static_cast<u32>(h->flags);
                    }
                    if (R<u64>(batch + 0x20) != 0) {
                        h->v68 = R<u64>(batch + 0x38) != 0 ? (indirect_mode == 0 ? R<u64>(h->rd + 0x3da8) : R<u64>(h->rd + 0x3d90)) : 0;
                        break;
                    }
                }
                // issue 130 (floors): what ScanGBuffer leaves out because the engine's loop creates it at first use (render data, the primitive's cached variant). A worker
                // would draw the pass without those pieces and the loop would never run to create them, so such a pass stays on this thread this time (the loop then makes
                // them and the worker takes the pass from the next frame on). Same tests as the scan, any pass, read on the live list before the hand-over
                for (u32 b = 0; b < count && !h->learn; ++b) {
                    const u64 batch = R<u64>(array + u64{b} * 8);
                    if (batch == 0 || R<u64>(batch + 0x48) != 0) {
                        continue;
                    }
                    if (const u64 second = R<u64>(batch + 0x28); second != 0) {
                        const u64 second_data = R<u64>(second + 0x38);
                        if (second_data == 0 || R<u64>(second_data + 0x20) == 0) {
                            h->learn = true;
                            break;
                        }
                    }
                    const u64 mesh = R<u64>(batch + 0x20);
                    if (mesh == 0) {
                        continue;
                    }
                    const u64 render_data = R<u64>(mesh + 0x38);
                    if (render_data == 0 || R<u64>(render_data + 0x20) == 0) {
                        h->learn = true;
                        break;
                    }
                    if (const u64 group = R<u64>(batch + 0x38); group != 0) {
                        const u64 group_inner = R<u64>(group + 8);
                        const u64 group_object = group_inner != 0 ? R<u64>(group_inner + 0x38) : 0;
                        if (group_object == 0 || R<u32>(R<u64>(group_object + 0x28) + 0x48) == 0) {
                            continue; // the scan reads no primitives of this batch
                        }
                    }
                    const u16 prim_count = R<u16>(batch + 100);
                    const u64 prim_array = R<u64>(batch + 0x58);
                    for (u32 p = 0; p < prim_count; ++p) {
                        const u64 record = prim_array + u64{p} * 0x38;
                        const u64 data = R<u64>(record + 0x28);
                        if (data == 0 || R<s32>(record + 0x24) >= 0) {
                            continue;
                        }
                        if (R<u64>(data + 0x40) == 0 || R<u32>(record + 0x30) != 0) {
                            h->learn = true;
                            break;
                        }
                    }
                }
                h->ok = true;
            },
            &head) ||
        !head.ok) {
        return false;
    }
    if (head.learn) {
        ps.g_held.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const u64 ctx = head.ctx;
    const u32 slot = ParSnapshot(ctx, rd);
    if (slot == ParSlots) {
        return false;
    }
    if (!ParCopyList(slot, 0, list)) {
        PS().busy[slot].store(false, std::memory_order_release);
        return false;
    }
    NativeDraw::HandOverTargets(ctx);
    ParPublishGroups(list, R<u32>(ctx + 0x47ac));
    if (R<u32>(ctx + 0x4480) != 4) {
        W<u32>(ctx + 0x4480, 4);
        W<u8>(ctx + 0x447c, R<u8>(ctx + 0x447c) | 1);
    }
    W<u64>(ctx + 0x68, head.v68);
    std::memset(reinterpret_cast<void*>(ctx + 0x580), 0, 0x400);
    std::memset(reinterpret_cast<void*>(ctx + 0x980), 0, 0x400);
    if (const u64 raster = R<u64>(rd + 0x3bf0); R<u64>(ctx + 0x46c8) != raster) {
        W<u64>(ctx + 0x46c8, raster);
        W<u8>(ctx + 0x46d8, 1);
    }
    W<u64>(ctx + 0x46e0, R<u64>(rd + 0x3c28));
    W<u32>(ctx + 0x46e8, flags != 0 ? head.last_flags : 0);
    W<u8>(ctx + 0x46ec, 1);
    ParTask task;
    static const bool moved_census = [] { const char* e = std::getenv("TLG_VKB_PAR_MOVED"); return e != nullptr && e[0] == '1'; }();
    if (moved_census) {
        struct Walk {
            u64 list;
            std::unordered_map<u64, std::pair<u64, u64>>* out;
        } walk{list, nullptr};
        task.moved = std::make_shared<std::unordered_map<u64, std::pair<u64, u64>>>();
        walk.out = task.moved.get();
        (void)NativeDraw::GuardedCall(
            [](void* p) {
                auto* w = static_cast<Walk*>(p);
                const u64 array = R<u64>(w->list);
                const u32 count = R<u32>(w->list + 8);
                for (u32 b = 0; b < count && b < 100000; ++b) {
                    const u64 batch = R<u64>(array + u64{b} * 8);
                    const u64 mesh = batch != 0 ? R<u64>(batch + 0x20) : 0;
                    const u64 render_data = mesh != 0 ? R<u64>(mesh + 0x38) : 0;
                    const u64 inner = render_data != 0 ? R<u64>(render_data + 0x20) : 0;
                    if (inner == 0) {
                        continue;
                    }
                    const u64 obj = R<u64>(R<u64>(batch + 0x68) + 0x130);
                    const u64 cb = obj != 0 ? (u64{R<u32>(obj + 0x60)} | u64{R<u32>(obj + 0x64) & 0xffu} << 32) : 0;
                    const u64 view = R<u64>(R<u64>(R<u64>(inner + 0x48) + 0x38) + 0x28);
                    const u64 vb = view != 0 && R<u64>(view + 0x30) != 0 ? R<u64>(R<u64>(view + 0x30) + 0x30) : 0;
                    (*w->out)[batch] = {cb, vb};
                }
            },
            &walk);
    }
    {
        struct B60 {
            u64 args, value;
        } b60{args, 0};
        (void)NativeDraw::GuardedCall(
            [](void* p) {
                auto* b = static_cast<B60*>(p);
                const u64 object = HandleLookup(R<u64>(Global(GlobalObjectTable)), R<u64>(R<u64>(b->args + 0x2d0) + 0x30));
                b->value = object != 0 ? R<u64>(object + 0x48) : 0;
            },
            &b60);
        task.v_b60 = b60.value;
    }
    task.rd = rd;
    task.kind = 1;
    task.mode = mode, task.list = list, task.args = args, task.flags = flags, task.query_mode = query_mode;
    task.view = ParGView;
    task.slot = slot;
    ParSend(task);
    return true;
}
/// The engine's thread: hand the pass to the worker. False: nothing done (the caller runs the pass as before).
bool ParKick(u64 rd, u64 desc, u32 view) {
    ParState& ps = PS();
    if ((ps.calls.fetch_add(1, std::memory_order_relaxed) & 255) == 0) {
        ParPublish();
    }
    if (ps.view_off[view].load(std::memory_order_relaxed) || ps.serial_next[view].exchange(false, std::memory_order_relaxed)) {
        return false;
    }
    struct Head {
        u64 rd, desc, ctx;
        bool ok;
    } head{rd, desc, 0, false};
    // what the scan would refuse for the whole pass, read here (the worker has no way back once the pass is handed over)
    if (!NativeDraw::GuardedCall(
            [](void* p) {
                auto* h = static_cast<Head*>(p);
                h->ctx = R<u64>(h->rd + 0x2d8);
                h->ok = h->ctx != 0 && R<u64>(h->desc) != 0 && R<u32>(h->desc + 0x10) == 3 && R<u64>(h->rd + 0x8930) == 0;
            },
            &head) ||
        !head.ok) {
        return false;
    }
    const u32 slot = ParSnapshot(head.ctx, rd);
    if (slot == ParSlots) {
        return false;
    }
    if (!ParCopyList(slot, desc, 0)) {
        PS().busy[slot].store(false, std::memory_order_release);
        return false;
    }
    NativeDraw::HandOverTargets(head.ctx);
    const u64 ctx = head.ctx;
    ParPublishGroups(R<u64>(desc), R<u32>(ctx + 0x47ac));
    // the context as the loop leaves it: its prologue (the stage mode word and its dirty bit, no input layout) and its epilogue (no material block, the default raster state)
    if (R<u32>(ctx + 0x4480) != 4) {
        W<u32>(ctx + 0x4480, 4);
        W<u8>(ctx + 0x447c, R<u8>(ctx + 0x447c) | 1);
    }
    W<u64>(ctx + 0x68, 0);
    W<u64>(ctx + 0x980, 0);
    if (const u64 raster = R<u64>(rd + 0x3bf0); R<u64>(ctx + 0x46c8) != raster) {
        W<u64>(ctx + 0x46c8, raster);
        W<u8>(ctx + 0x46d8, 1);
    }
    ParTask task;
    task.rd = rd;
    task.desc = desc;
    task.kind = 0;
    task.view = view;
    task.slot = slot;
    ParSend(task);
    return true;
}
} // namespace

bool HandlePass(u64 render_device, u64 desc, u64 original_loop) {
    const Config& cfg = Cfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    using Loop = void(PS4_SYSV_ABI*)(u64, u64);
    if (ParOn() && Core::Vkb::Stream::RecorderOn()) { // P1: the cascades on the worker
        struct ViewArgs {
            u64 desc;
            u32 view;
        } va{desc, 0xffffffffu};
        if (NativeDraw::GuardedCall([](void* p) { auto* v = static_cast<ViewArgs*>(p); v->view = static_cast<u32>(R<u64>(v->desc + 8)); }, &va) && (va.view == 1 || va.view == 2) &&
            cfg.views[va.view] && ParKick(render_device, desc, va.view)) {
            return true;
        }
    }
    State& s = S();
    Plan plan;
    ScanArgs scan_args{&plan, render_device, desc, &s.table};
    // the scan and the learning read the engine's objects: guarded, and under the state lock for the table
    bool handled_view = false;
    {
        std::scoped_lock lock{s.mutex};
        u32 view = 0xffffffffu;
        struct HeaderArgs {
            u64 desc;
            u32 view;
        } header{desc, view};
        if (!NativeDraw::GuardedCall([](void* p) { auto* h = static_cast<HeaderArgs*>(p); h->view = static_cast<u32>(R<u64>(h->desc + 8)); }, &header)) {
            return false;
        }
        view = header.view;
        handled_view = view < cfg.views.size() && cfg.views[view];
        if (!handled_view) {
            return false;
        }
        CacheArgs cache_args{&s, render_device};
        if (!NativeDraw::GuardedCall([](void* p) { CheckCacheImpl(*static_cast<CacheArgs*>(p)); }, &cache_args)) {
            ++s.scan_faults;
        }
        ViewStats& vs = s.views[view];
        ++vs.passes;
        struct KindArgs {
            u64 desc;
            u32 kind;
        } kind_args{desc, 0xffffffffu};
        if (NativeDraw::GuardedCall([](void* p) { auto* k = static_cast<KindArgs*>(p); k->kind = R<u32>(k->desc + 0x10); }, &kind_args)) {
            ++vs.kinds[kind_args.kind];
        }
        if (vs.disabled) {
            plan.Decline("generator switched off for this view type by a verify difference");
        } else if (!Timed(CallId::ScanDepth, [&] { return NativeDraw::GuardedCall([](void* p) { ScanImpl(*static_cast<ScanArgs*>(p)); }, &scan_args); })) {
            ++s.scan_faults;
            ++vs.not_scanned;
            plan.scanned = false;
            plan.steps.clear();
            plan.batches.clear();
            plan.Decline("scan faulted");
        }
        plan.view_type = view;
        if (!plan.Ok()) {
            for (const char* r : plan.reasons) {
                ++vs.declines[r];
            }
        }
    }
    ViewStats* stats = nullptr;
    {
        std::scoped_lock lock{s.mutex};
        stats = &s.views[plan.view_type];
        if (plan.Ok()) {
            ++stats->eligible;
        }
    }
    if (plan.Ok()) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        Timed(CallId::PassDepth, [&] { Generate(live, plan); });
        std::scoped_lock lock{s.mutex};
        ++stats->generated;
        stats->generated_batches += plan.batches.size();
        stats->generated_draws += plan.steps.size();
        stats->group_batches += plan.group_batches;
        stats->group_batches_skipped += plan.group_batches_skipped;
        return true;
    }
    reinterpret_cast<Loop>(original_loop)(render_device, desc);
    // learn what the loop chose
    {
        std::scoped_lock lock{s.mutex};
        ++stats->loop_passes;
        if (plan.scanned && !plan.steps.empty()) {
            CacheArgs cache_args{&s, render_device};
            if (!NativeDraw::GuardedCall([](void* p) { CheckCacheImpl(*static_cast<CacheArgs*>(p)); }, &cache_args)) {
                ++s.scan_faults;
            }
            LearnArgs learn_args{&plan, &s};
            if (!NativeDraw::GuardedCall([](void* p) { LearnImpl(*static_cast<LearnArgs*>(p)); }, &learn_args)) {
                ++s.learn_faults;
            }
        }
    }
    return true;
}

bool GBufferEnabled() {
    return GCfg().on;
}

bool HandleGBuffer(u64 render_device, u64 mode, u64 list, u64 args, u64 flags, u64 query_mode, u64 param7, u64 original_loop) {
    const GConfig& cfg = GCfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    (void)param7;
    using Loop = void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64);
    if (ParOn() && ParGOn() && Core::Vkb::Stream::RecorderOn() && ParKickG(render_device, mode, list, args, flags, query_mode)) {
        return true; // P2: the pass is the worker's
    }
    State& s = S();
    GStats& gs = GS();
    GPlan plan;
    plan.mode = static_cast<u32>(mode);
    plan.flags = static_cast<u32>(flags);
    plan.query_mode = static_cast<u32>(query_mode);
    GScanArgs scan_args{&plan, render_device, list, args};
    {
        std::scoped_lock lock{s.mutex};
        ++gs.v.passes;
        ++gs.arguments[static_cast<u32>(mode) * 1000 + static_cast<u32>(query_mode)];
        if (gs.v.disabled) {
            plan.Decline("generator switched off by a verify difference");
        } else if (!Timed(CallId::ScanGBuffer, [&] { return NativeDraw::GuardedCall([](void* p) { ScanGBuffer(*static_cast<GScanArgs*>(p)); }, &scan_args); })) {
            ++gs.v.not_scanned;
            plan.scanned = false;
            plan.Decline("scan faulted");
        }
        if (!plan.Ok()) {
            for (const char* r : plan.reasons) {
                ++gs.v.declines[r];
            }
        }
    }
    if (plan.Ok()) {
        std::scoped_lock lock{s.mutex};
        ++gs.eligible;
    }
    if (plan.Ok()) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        live.indexed_site = GBufferIndexedSite;
        live.indirect_site = GBufferIndirectSite;
        live.indexed_a4 = 0;
        Timed(CallId::PassGBuffer, [&] { GenerateGBuffer(live, plan, plan.flags); });
        std::scoped_lock lock{s.mutex};
        ++gs.v.generated;
        gs.v.generated_batches += plan.batches.size();
        gs.v.generated_draws += plan.steps.size();
        gs.v.group_batches += plan.group_batches;
        return true;
    }
    reinterpret_cast<Loop>(original_loop)(render_device, mode, list, args, flags, query_mode, param7);
    std::scoped_lock lock{s.mutex};
    ++gs.v.loop_passes;
    return true;
}

bool ForwardEnabled() {
    return FCfg().on;
}

bool HandleForward(u64 render_device, u64 a1, u64 a2, u64 list, u64 mode, u64 original_loop) {
    const FConfig& cfg = FCfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    using Loop = void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64);
    State& s = S();
    FStats& fs = FS();
    FPlan plan;
    plan.render_device = render_device;
    plan.a1 = a1;
    plan.a2 = a2;
    plan.mode = static_cast<u32>(mode);
    last_forward_ctx.store(R<u64>(render_device + 0x2d8), std::memory_order_relaxed);
    FScanArgs scan_args{&plan, list};
    {
        std::scoped_lock lock{s.mutex};
        ++fs.v.passes;
        if (fs.v.disabled) {
            plan.Decline("generator switched off by a verify difference");
        } else if (!Timed(CallId::ScanForward, [&] { return NativeDraw::GuardedCall([](void* p) { ScanForwardPass(*static_cast<FScanArgs*>(p)); }, &scan_args); })) {
            ++fs.v.not_scanned;
            plan.scanned = false;
            plan.Decline("scan faulted");
        }
        if (!plan.Ok()) {
            for (const char* r : plan.reasons) {
                ++fs.v.declines[r];
            }
        }
    }
    if (plan.Ok()) {
        std::scoped_lock lock{s.mutex};
        ++fs.eligible;
    }
    if (plan.Ok()) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        live.indexed_site = ForwardIndexedSite;
        live.indexed_a4 = 1;
        Timed(CallId::PassForward, [&] { GenerateForward(live, plan); });
        std::scoped_lock lock{s.mutex};
        ++fs.v.generated;
        fs.v.generated_batches += plan.batches.size();
        fs.v.generated_draws += plan.steps.size();
        return true;
    }
    reinterpret_cast<Loop>(original_loop)(render_device, a1, a2, list, mode);
    std::scoped_lock lock{s.mutex};
    ++fs.v.loop_passes;
    return true;
}

bool DecalEnabled() {
    return DCfg().on;
}

bool HandleDecal(u64 render_device, u64 view, u64 a3, u64 a4, u64 desc, u32 flags, s32 type, u32 width, u32 height, u64 original_loop) {
    const DConfig& cfg = DCfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    using Loop = void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64, u64, u64);
    const auto run_loop = [&] {
        reinterpret_cast<Loop>(original_loop)(render_device, view, a3, a4, desc, flags, static_cast<u64>(static_cast<u32>(type)), width, height);
    };
    State& s = S();
    DStats& ds = DS();
    DPlan plan;
    plan.render_device = render_device;
    plan.view = view;
    plan.a3 = a3;
    plan.a4 = a4;
    plan.desc = desc;
    plan.flags = flags;
    plan.type = type;
    plan.width = width;
    plan.height = height;
    DScanArgs scan_args{&plan};
    {
        std::scoped_lock lock{s.mutex};
        if (ds.v.disabled) {
            plan.Decline("generator switched off by a verify difference");
        } else if (!Timed(CallId::ScanDecal, [&] { return NativeDraw::GuardedCall([](void* p) { ScanDecalPass(*static_cast<DScanArgs*>(p)); }, &scan_args); })) {
            ++ds.v.not_scanned;
            plan.scanned = false;
            plan.Decline("scan faulted");
        }
        if (plan.no_work) {
            ++ds.loops_without_work;
        } else {
            ++ds.v.passes;
            if (!plan.Ok()) {
                for (const char* r : plan.reasons) {
                    ++ds.v.declines[r];
                }
            }
        }
    }
    if (plan.no_work) {
        run_loop(); // the engine's function does nothing for an empty list
        return true;
    }
    if (plan.Ok()) {
        std::scoped_lock lock{s.mutex};
        ++ds.eligible;
    }
    if (plan.Ok()) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        Timed(CallId::PassDecal, [&] { GenerateDecal(live, plan); });
        std::scoped_lock lock{s.mutex};
        ++ds.v.generated;
        ds.v.generated_batches += plan.batches.size();
        ds.v.generated_draws += plan.steps.size();
        return true;
    }
    run_loop();
    std::scoped_lock lock{s.mutex};
    ++ds.v.loop_passes;
    return true;
}

bool SkinEnabled() {
    return SCfg().on;
}

bool HandleSkin(u64 system, u64 scene, u64 async_flag, u64 original_loop) {
    const SConfig& cfg = SCfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    using Loop = void(PS4_SYSV_ABI*)(u64, u64, u64);
    State& s = S();
    SStats& ss = SS();
    SkinPlan plan;
    plan.system = system;
    plan.scene = scene;
    plan.async = (async_flag & 0xff) != 0;
    const auto run_loop = [&] { reinterpret_cast<Loop>(original_loop)(system, scene, async_flag); };
    SkinScanArgs scan_args{&plan};
    {
        std::scoped_lock lock{s.mutex};
        if (ss.v.disabled) {
            plan.Decline("generator switched off by a verify difference");
        } else if (!Timed(CallId::ScanSkin, [&] { return NativeDraw::GuardedCall([](void* p) { ScanSkin(*static_cast<SkinScanArgs*>(p)); }, &scan_args); })) {
            ++ss.v.not_scanned;
            plan.scanned = false;
            plan.Decline("scan faulted");
        }
        if (plan.no_work) {
            ++ss.loops_without_work;
        } else {
            ++ss.v.passes;
            if (!plan.Ok()) {
                for (const char* r : plan.reasons) {
                    ++ss.v.declines[r];
                }
            }
        }
    }
    if (plan.no_work) {
        run_loop();
        return true;
    }
    if (plan.Ok()) {
        std::scoped_lock lock{s.mutex};
        ++ss.eligible;
    }
    if (plan.Ok()) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        Timed(CallId::PassSkin, [&] { GenerateSkin(live, plan); });
        std::scoped_lock lock{s.mutex};
        ++ss.v.generated;
        ss.v.generated_batches += plan.nodes.size();
        ss.v.generated_draws += live.draws;
        return true;
    }
    run_loop();
    std::scoped_lock lock{s.mutex};
    ++ss.v.loop_passes;
    return true;
}

bool WetEnabled() {
    return WCfg().on;
}

bool HandleWet(u64 system, u64 ctx, u64 scene, u64 original_loop) {
    const WConfig& cfg = WCfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    using Loop = void(PS4_SYSV_ABI*)(u64, u64, u64);
    State& s = S();
    WStats& ws = WS();
    WetPlan plan;
    plan.system = system;
    plan.ctx = ctx;
    plan.scene = scene;
    const auto run_loop = [&] { reinterpret_cast<Loop>(original_loop)(system, ctx, scene); };
    WetScanArgs scan_args{&plan};
    {
        std::scoped_lock lock{s.mutex};
        if (ws.v.disabled) {
            plan.scanned = true;
            plan.Decline("generator switched off by a verify difference");
        } else if (!Timed(CallId::ScanWet, [&] { return NativeDraw::GuardedCall([](void* p) { ScanWet(*static_cast<WetScanArgs*>(p)); }, &scan_args); })) {
            ++ws.v.not_scanned;
            plan.scanned = false;
            plan.Decline("scan faulted");
        }
        if (plan.no_work) {
            ++ws.loops_without_work;
        } else {
            ++ws.v.passes;
            if (!plan.Ok()) {
                for (const char* r : plan.reasons) {
                    ++ws.v.declines[r];
                }
            }
        }
    }
    if (plan.no_work) {
        run_loop();
        return true;
    }
    if (plan.Ok()) {
        std::scoped_lock lock{s.mutex};
        ++ws.eligible;
    }
    if (plan.Ok()) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        Timed(CallId::PassWet, [&] { GenerateWet(live, plan); });
        std::scoped_lock lock{s.mutex};
        ++ws.v.generated;
        ws.v.generated_draws += live.draws;
        return true;
    }
    run_loop();
    std::scoped_lock lock{s.mutex};
    ++ws.v.loop_passes;
    return true;
}

bool InstEnabled() {
    return ICfg().on;
}

bool HandleInst(u64 holder, u64 scene, u64 original_loop) {
    const IConfig& cfg = ICfg();
    if (!cfg.on || original_loop == 0) {
        return false;
    }
    using Loop = void(PS4_SYSV_ABI*)(u64, u64);
    State& s = S();
    IStats& is = IS();
    InstPlan plan;
    plan.holder = holder;
    plan.scene = scene;
    const auto run_loop = [&] { reinterpret_cast<Loop>(original_loop)(holder, scene); };
    InstScanArgs scan_args{&plan};
    {
        std::scoped_lock lock{s.mutex};
        if (is.v.disabled) {
            plan.scanned = true;
            plan.Decline("generator switched off by a verify difference");
        } else if (!Timed(CallId::ScanInst, [&] { return NativeDraw::GuardedCall([](void* p) { ScanInst(*static_cast<InstScanArgs*>(p)); }, &scan_args); })) {
            ++is.v.not_scanned;
            plan.scanned = false;
            plan.Decline("scan faulted");
        }
        if (plan.no_work) {
            ++is.loops_without_work;
        } else {
            ++is.v.passes;
            if (!plan.Ok()) {
                for (const char* r : plan.reasons) {
                    ++is.v.declines[r];
                }
            }
        }
    }
    if (plan.no_work) {
        run_loop();
        return true;
    }
    if (plan.Ok()) {
        std::scoped_lock lock{s.mutex};
        ++is.eligible;
    }
    if (plan.Ok()) {
        Target live;
        live.live = true;
        live.base = reinterpret_cast<u8*>(plan.ctx);
        Timed(CallId::PassInst, [&] { GenerateInst(live, plan); });
        std::scoped_lock lock{s.mutex};
        ++is.v.generated;
        is.v.generated_draws += live.draws;
        return true;
    }
    run_loop();
    std::scoped_lock lock{s.mutex};
    ++is.v.loop_passes;
    return true;
}

nlohmann::json Report() {
    auto& s = S();
    std::scoped_lock lock{s.mutex};
    const Config& cfg = Cfg();
    nlohmann::json out = nlohmann::json::object();
    out["enabled"] = cfg.on;
    out["job_loop"] = NativeJobGen::Report(); // B7 (core/native_job_gen.cpp)
    out["site_generators"] = NativeSiteGen::Report(); // B7 (core/native_site_gen.cpp)
    nlohmann::json views_on = nlohmann::json::array();
    for (u32 i = 0; i < cfg.views.size(); ++i) {
        if (cfg.views[i]) {
            views_on.push_back(i);
        }
    }
    out["view_types_handled"] = views_on;
    u64 ambiguous = 0;
    for (const auto& [key, entry] : s.table) {
        ambiguous += entry.ambiguous ? 1 : 0;
    }
    out["variant_table"] = {{"keys", s.table.size()}, {"ambiguous_keys", ambiguous}, {"learned", s.learned}, {"conflicts", s.conflicts}, {"resets_after_a_cache_change", s.table_resets},
                            {"variants_not_in_the_engine_cache", s.learn_not_in_cache}, {"learn_faults", s.learn_faults}, {"scan_faults", s.scan_faults}, {"cache_entries", s.cache_seen.size()}};
    auto& views = out["views"] = nlohmann::json::object();
    for (const auto& [view, v] : s.views) {
        nlohmann::json entry = {{"passes", v.passes},
                                {"generated_passes_without_the_loop", v.generated},
                                {"generated_batches", v.generated_batches},
                                {"generated_draws", v.generated_draws},
                                {"passes_run_by_the_loop", v.loop_passes},
                                {"switched_off_by_verify", v.disabled},
                                {"scan_faults", v.not_scanned},
                                {"instance_group_batches_generated", v.group_batches},
                                {"instance_group_batches_with_no_instances", v.group_batches_skipped}};
        nlohmann::json kinds = nlohmann::json::object();
        for (const auto& [kind, count] : v.kinds) {
            kinds[std::to_string(kind)] = count;
        }
        entry["passes_by_kind_desc_0x10"] = kinds;
        nlohmann::json declines = nlohmann::json::object();
        for (const auto& [reason, count] : v.declines) {
            declines[reason] = count;
        }
        entry["declined_passes_by_reason"] = declines;
        views[std::to_string(view)] = entry;
    }
    {
        const GStats& gs = GS();
        const ViewStats& v = gs.v;
        nlohmann::json g = {{"enabled", GCfg().on},
                            {"passes", v.passes},
                            {"generated_passes_without_the_loop", v.generated},
                            {"generated_batches", v.generated_batches},
                            {"generated_draws", v.generated_draws},
                            {"instance_group_batches", v.group_batches},
                            {"passes_run_by_the_loop", v.loop_passes},
                            {"switched_off_by_verify", v.disabled},
                            {"scan_faults", v.not_scanned},
                            {"passes_by_mode_times_1000_plus_query_mode", gs.arguments}};
        nlohmann::json declines = nlohmann::json::object();
        for (const auto& [reason, count] : v.declines) {
            declines[reason] = count;
        }
        g["declined_passes_by_reason"] = declines;
        out["gbuffer"] = g;
        const ViewStats& f = FS().v;
        nlohmann::json fj = {{"enabled", FCfg().on},
                             {"passes", f.passes},
                             {"generated_passes_without_the_loop", f.generated},
                             {"generated_batches", f.generated_batches},
                             {"generated_draws", f.generated_draws},
                             {"passes_run_by_the_loop", f.loop_passes},
                             {"switched_off_by_verify", f.disabled},
                             {"scan_faults", f.not_scanned}};
        nlohmann::json fdeclines = nlohmann::json::object();
        for (const auto& [reason, count] : f.declines) {
            fdeclines[reason] = count;
        }
        fj["declined_passes_by_reason"] = fdeclines;
        out["forward"] = fj;
        const ViewStats& dj = DS().v;
        nlohmann::json dd = {{"enabled", DCfg().on},
                             {"passes", dj.passes},
                             {"loops_without_work", DS().loops_without_work},
                             {"generated_passes_without_the_loop", dj.generated},
                             {"generated_batches", dj.generated_batches},
                             {"generated_draws", dj.generated_draws},
                             {"passes_run_by_the_loop", dj.loop_passes},
                             {"switched_off_by_verify", dj.disabled},
                             {"scan_faults", dj.not_scanned}};
        nlohmann::json ddeclines = nlohmann::json::object();
        for (const auto& [reason, count] : dj.declines) {
            ddeclines[reason] = count;
        }
        dd["declined_passes_by_reason"] = ddeclines;
        out["decal"] = dd;
        const ViewStats& sj = SS().v;
        nlohmann::json sk = {{"enabled", SCfg().on},
                             {"passes", sj.passes},
                             {"loops_without_work", SS().loops_without_work},
                             {"generated_passes_without_the_loop", sj.generated},
                             {"generated_dispatches", sj.generated_draws},
                             {"passes_run_by_the_loop", sj.loop_passes},
                             {"switched_off_by_verify", sj.disabled},
                             {"scan_faults", sj.not_scanned}};
        nlohmann::json sdeclines = nlohmann::json::object();
        for (const auto& [reason, count] : sj.declines) {
            sdeclines[reason] = count;
        }
        sk["declined_passes_by_reason"] = sdeclines;
        out["skinning"] = sk;
        const ViewStats& wj = WS().v;
        nlohmann::json wt = {{"enabled", WCfg().on},
                             {"passes", wj.passes},
                             {"loops_without_work", WS().loops_without_work},
                             {"generated_passes_without_the_loop", wj.generated},
                             {"generated_dispatches", wj.generated_draws},
                             {"passes_run_by_the_loop", wj.loop_passes},
                             {"jobs_skipped_for_render_data_created_late", wet_skipped_jobs.load()},
                             {"dispatches_by_phase", {{"elapse", wet_dispatches[0].load()}, {"intersect", wet_dispatches[1].load()}, {"force", wet_dispatches[2].load()}, {"readbacks_done_live", wet_dispatches[3].load()}}},
                             {"switched_off_by_verify", wj.disabled},
                             {"scan_faults", wj.not_scanned}};
        nlohmann::json wdeclines = nlohmann::json::object();
        for (const auto& [reason, count] : wj.declines) {
            wdeclines[reason] = count;
        }
        wt["declined_passes_by_reason"] = wdeclines;
        out["wet"] = wt;
        const ViewStats& ij = IS().v;
        nlohmann::json inst = {{"enabled", ICfg().on},
                               {"passes", ij.passes},
                               {"loops_without_work", IS().loops_without_work},
                               {"generated_passes_without_the_loop", ij.generated},
                               {"generated_dispatches", ij.generated_draws},
                               {"passes_run_by_the_loop", ij.loop_passes},
                               {"switched_off_by_verify", ij.disabled},
                               {"scan_faults", ij.not_scanned},
                               {"dispatches_by_loop", {{"loop1", inst_dispatches[0].load()}, {"loop2", inst_dispatches[1].load()}, {"loop3_counts", inst_dispatches[2].load()}, {"loop4_nine_passes", inst_dispatches[3].load()}, {"loop5", inst_dispatches[4].load()}}}};
        nlohmann::json ideclines = nlohmann::json::object();
        for (const auto& [reason, count] : ij.declines) {
            ideclines[reason] = count;
        }
        inst["declined_passes_by_reason"] = ideclines;
        out["instancing"] = inst;
    }
    if (CensusOn()) {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        nlohmann::json census = nlohmann::json::object();
        const double to_ms = 1000.0 / static_cast<double>(frequency.QuadPart);
        for (u32 i = 0; i < static_cast<u32>(CallId::Count); ++i) {
            const u64 calls = Census()[i].calls.load(), ticks = Census()[i].ticks.load();
            if (calls != 0) {
                census[CallNames[i]] = {{"calls", calls}, {"total_ms", ticks * to_ms}, {"microseconds_per_call", ticks * to_ms * 1000.0 / static_cast<double>(calls)}};
            }
        }
        for (u32 kind = 0; kind < 2; ++kind) {
            u64 calls, tag, original, taken;
            NativeDraw::IssueCensusRead(kind, calls, tag, original, taken);
            if (calls != 0) {
                census[kind == 0 ? "draw issue, indexed: the tag (TagIndexed)" : "draw issue, indirect: the tag (TagIndirect)"] = {{"calls", calls}, {"total_ms", tag * to_ms}, {"microseconds_per_call", tag * to_ms * 1000.0 / static_cast<double>(calls)}, {"draws_the_tag_took_over", taken}};
                census[kind == 0 ? "draw issue, indexed: the engine's draw function after the tag" : "draw issue, indirect: the engine's draw function after the tag"] = {{"calls", calls - taken}, {"total_ms", original * to_ms}, {"microseconds_per_call", original * to_ms * 1000.0 / static_cast<double>(std::max<u64>(1, calls - taken))}};
            }
        }
        {
            std::array<u64, 5> ticks{};
            u64 calls = 0;
            NativeDraw::TagCensusRead(ticks, calls);
            constexpr const char* names[5] = {"tag phase 0: probes, policy, class checks", "tag phase 1: resource tracking (the engine's, RefreshTargets) and the registry", "tag phase 2: signature of the context (Compute)",
                                              "tag phase 3: record build (MakeRecord, SignRecord)", "tag phase 4: tag words, execution eligibility, native execution side effects"};
            for (u32 i = 0; i < 5 && calls != 0; ++i) {
                census[names[i]] = {{"calls", calls}, {"total_ms", ticks[i] * to_ms}, {"microseconds_per_call", ticks[i] * to_ms * 1000.0 / static_cast<double>(calls)}};
            }
        }
        {
            std::vector<std::pair<u64, u32>> changed;
            for (u32 i = 0; i < reuse_changed_words.size(); ++i) {
                if (const u64 n = reuse_changed_words[i].load()) {
                    changed.emplace_back(n, i * 8);
                }
            }
            std::sort(changed.rbegin(), changed.rend());
            nlohmann::json words = nlohmann::json::object();
            for (size_t i = 0; i < changed.size() && i < 40; ++i) {
                words[fmt::format("{:#x}", changed[i].second)] = changed[i].first;
            }
            census["context words (0 to 0xd80) that changed between consecutive draws: offset -> draws"] = words;
        }
        census["context state equal to the previous draw's (only the draw arguments differ)"] = {{"draws", Reuse().draws.load()}, {"same_as_previous", Reuse().same_as_previous.load()}, {"same_as_one_of_the_last_8", Reuse().same_as_one_of_last_8.load()}};
        {
            std::array<u64, 6> ticks{};
            u64 calls = 0;
            NativeDraw::MakeRecordCensusRead(ticks, calls);
            constexpr const char* names[6] = {"record build 0: setup", "record build 1: Synthesize (registers)", "record build 2: Fixed::Build (fixed function state)", "record build 3: user data bindings, vertex stage",
                                              "record build 4: user data bindings, pixel stage", "record build 5: draw argument registers and the rest"};
            for (u32 i = 0; i < 6 && calls != 0; ++i) {
                census[names[i]] = {{"calls", calls}, {"total_ms", ticks[i] * to_ms}, {"microseconds_per_call", ticks[i] * to_ms * 1000.0 / static_cast<double>(calls)}};
            }
        }
        {
            u64 lookups, hits_big, hits_small;
            NativeDraw::RecordCacheCensusRead(lookups, hits_big, hits_small);
            census["a cache of records keyed by the whole signature (all groups but the draw arguments), direct mapped"] = {{"lookups", lookups}, {"hits_with_1M_entries", hits_big}, {"hits_with_64K_entries", hits_small}};
        }
        {
            u64 scan, fill, tables, slots;
            NativeDraw::UserDataCensusRead(scan, fill, tables, slots);
            if (tables != 0) {
                census["user data tables (resource, constant buffer, sampler pointers): the slot scan"] = {{"tables", tables}, {"total_ms", scan * to_ms}, {"microseconds_per_table", scan * to_ms * 1000.0 / static_cast<double>(tables)}};
                census["user data tables: filling the descriptors"] = {{"tables", tables}, {"slots_filled", slots}, {"total_ms", fill * to_ms}, {"microseconds_per_table", fill * to_ms * 1000.0 / static_cast<double>(tables)},
                                                                       {"nanoseconds_per_slot", fill * to_ms * 1e6 / static_cast<double>(std::max<u64>(1, slots))}};
            }
        }
        {
            std::array<u64, 6> m{};
            NativeDraw::MergeCensusRead(m);
            if (m[0] != 0) {
                census["records equal to an earlier record of the thread except for the draw arguments"] = {{"draws", m[0]}, {"equal_to_the_previous_record", m[1]}, {"equal_to_one_of_the_last_16", m[2]},
                                                                                                                   {"registers_and_fixed_state_equal_to_the_previous", m[3]}, {"registers_equal_to_the_previous", m[4]}};
            }
        }
        out["engine_call_census"] = census;
    }
    if (CensusOn()) {
        nlohmann::json audit = nlohmann::json::object();
        NativeDraw::InputTableAuditRead(audit);
        out["input_table_audit"] = audit;
    }
    {
        nlohmann::json sites = nlohmann::json::object();
        NativeDraw::DrawSiteCensusRead(sites);
        if (sites["total_draws"].get<u64>() != 0) {
            out["draw_site_census"] = std::move(sites);
        }
    }
    {
        nlohmann::json list = nlohmann::json::object();
        NativeDraw::NativeListRead(list);
        out["native_list"] = std::move(list);
    }
    out["forward_prologue_FUN_01c710f0"] = {{"made_natively_live_and_shadow", Leaf().prologue_native.load()},
                                            {"blocks_compared_with_the_engines", Leaf().block_compared.load()},
                                            {"blocks_differing", Leaf().block_differing.load()}};
    if (const u64 ctx = last_forward_ctx.load(); ctx != 0) {
        nlohmann::json vt = nlohmann::json::object();
        const u64 table = R<u64>(ctx);
        for (u32 slot = 0; slot < 0x120 / 8; ++slot) {
            vt[fmt::format("{:#x}", slot * 8)] = fmt::format("{:#x}", R<u64>(table + slot * 8) - NativeDraw::ImageBase() + 0x1000000);
        }
        out["study_device_context_vtable_as_ghidra_addresses"] = vt;
    }
    out["native_resource_set_FUN_01c92f30"] = {{"made_natively", RsStats().native.load()}, {"engine_special_entry", RsStats().engine_special.load()},
                                              {"engine_other_buffer_shape", RsStats().engine_shape.load()}, {"verify_draws_compared", RsStats().compared.load()},
                                              {"verify_draws_differing", RsStats().differing.load()}};
    out["native_group_bind_FUN_01ee4e50"] = {{"made_natively", GbStats().native.load()}, {"engine_other_buffer_shape", GbStats().engine_shape.load()},
                                            {"verify_draws_compared", GbStats().compared.load()}, {"verify_draws_differing", GbStats().differing.load()}};
    out["verify_difference_kinds"] = s.mismatch_kinds;
    out["special_entry_classes"] = s.special_classes;
    out["verify_words_that_moved_by_view_kind_offset"] = s.raced_words;
    out["verify_difference_samples"] = s.samples;
    return out;
}

} // namespace Core::NativeZPassGen
