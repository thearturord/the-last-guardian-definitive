// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>

#include <string>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"
#include "native/native_device.h"
#include "native/shadps4/video_core/amdgpu/resource.h"
#include "native/shadps4/video_core/renderer_vulkan/vk_common.h"

namespace VkbVulkan {
class HostImport;
}

/// V2 of the engine-to-Vulkan backend (documents/tlg/native plan/engine-vulkan-backend.md), `TLG_VKB=1`: our Vulkan objects for the engine's objects, made when the engine makes
/// its own and keyed by the engine's id (+0x10, never reused).
///  - textures: a `VkImage` from the texture's Direct3D 11 description (size, levels, layers, format, bind flags); depth targets get a depth format
///  - texture views (shader resource, unordered access, render target, depth stencil): a `VkImageView` from the view's Direct3D 11 description
///  - buffers: the GPU reads them where they are in guest memory (`HostImport`, the whole backing imported in 256 MiB chunks): (chunk buffer, offset)
///  - V2f: a buffer the GPU writes (the unordered access bind flag) is our own device-local `VkBuffer` instead, as a Direct3D 11 default buffer is on a PC: its initial data and
///    its UpdateSubresource writes are copied in; the CPU never reads it but through the engine's copies into staging buffers (V0c), which are our copies (below)
/// An object the engine deletes is destroyed three graphics submits later (the frames that may still use it).
namespace Core::Vkb {

void TextureCreated(const NativeDevice::Object& texture);
void ViewCreated(const NativeDevice::Object& view, const NativeDevice::Object& resource);
void BufferCreated(const NativeDevice::Object& buffer);
void ObjectDeleted(u64 id);
/// Once per graphics submit: the deferred destructions that are due
void ObjectsTick();

/// The in-place access to guest memory (null before the device exists)
[[nodiscard]] VkbVulkan::HostImport* InPlace();

/// Our view of engine view `view_id` for a compute dispatch, of the type the module declares (`want`: the view itself, or a second view of the same range in that type, made
/// once, when Vulkan allows it: a cube as a 2D array, a 2D view as a one-layer array, and back); its image is moved to the general layout (the barrier recorded into `cmd`) when
/// it is not there yet. False: no such view, or the type cannot be had (counted).
bool ViewForCompute(u64 view_id, vk::ImageViewType want, vk::CommandBuffer cmd, vk::ImageView& out, vk::Image* image = nullptr);
/// The view type an image binding of a recompiled module declares, for its T#'s host view type (`GetHostViewType`): the type `ViewForCompute` is asked for. Draws and
/// dispatches both use it. A 1D image is declared as 2D (note 41); a cube as a 2D array of all its faces (spirv_emit_context.cpp ImageType: the module computes the layer,
/// slice x 6 + face, itself). `TLG_VKB_CUBE_AS_ARRAY=0` (issue 34; the old way) asks for a cube view instead, which `ViewForCompute` cuts to the first cube of the range.
[[nodiscard]] vk::ImageViewType ModuleViewType(VkbAmdGpu::ImageType type);
/// R5 TLG_VKB_DRAW_DUMP: what a view and a sampler object are (image size, formats, mips; the sampler's engine description)
std::string DescribeView(u64 view_id);
std::string DescribeSampler(u64 object);

/// V4: an attachment of a draw: our view of engine view `view_id` in `format` (the view itself when it has that format; else a view of the same range in that format, made once:
/// the image allows other formats of its size class), its image moved to the general layout; the extent of its first mip and its layer count. False: no such view, or the
/// format cannot view the image (counted).
struct Attachment {
    vk::ImageView view{};
    vk::Extent2D extent{};
    u32 layers{};
    bool has_stencil{};
    vk::Image image{}; ///< V4d: for the barriers
};
bool AttachmentFor(u64 view_id, vk::Format format, vk::CommandBuffer cmd, Attachment& out);
/// note 46 / issue 82: moves whenever a view or an image goes or an image's layout is set back; while it stays, an AttachmentFor answer for the same view id and format stays
[[nodiscard]] u64 ViewEpoch();
/// The format of our view of engine view `view_id` (eUndefined: none)
[[nodiscard]] vk::Format ViewFormat(u64 view_id);
/// Debug: the size of the first subresource of our view of engine view `view_id` (false: none)
bool ViewExtent(u64 view_id, u32& width, u32& height, u32* depth = nullptr, u64* texture_id = nullptr);
/// note 41, debug only: the first subresource of our view of engine view `view_id` to log/vkb_image_dump/<name>_<w>x<h>_<format>.bin once the GPU has it (recording thread, outside a pass)
void DumpView(vk::CommandBuffer cmd, u64 view_id, const std::string& name);
/// note 41, debug only: `size` bytes of `buffer` at `offset` (the GPU's copy) to log/vkb_image_dump/<name>_<size>B.gpu once the GPU has it
void DumpBoundView(vk::CommandBuffer cmd, vk::ImageView handle, const std::string& name); ///< debug: the image an image view handle shows
void DumpBuffer(vk::CommandBuffer cmd, vk::Buffer buffer, u64 offset, u64 size, const std::string& name);
/// note 91: the image `image` (a pass attachment, by its Vulkan handle) at full size, as DumpView
void DumpImageHandle(vk::CommandBuffer cmd, vk::Image image, const std::string& name);
/// F1: the format, size and kind of our image `image` (false: not one of ours); a scan, once a frame at most
bool ImageFacts(vk::Image image, vk::Format& format, u32& width, u32& height, bool& depth);
/// A view recorded into the graphics stream, destroyed once the GPU has done the submit that may use it (as views of deleted engine objects)
void RetireView(vk::ImageView view);
/// note 91: dump files queued and not yet written; waits (up to `timeout_ms`) until none is left
[[nodiscard]] u64 DumpsPending();
bool WaitDumps(u64 timeout_ms);
/// Our sampler for an engine sampler object, made once from its Direct3D 11 description (V2d)
/// note 40: `compare` for a shader that samples it with a depth compare (the S# compares per instruction on GCN; a Vulkan sampler compares for all its reads)
/// Issue 42: `bias_q` an extra mip bias in quarter steps, `aniso` the anisotropic filtering setting (0: the game's own filter), both from Aa::LookFor
[[nodiscard]] vk::Sampler SamplerFor(u64 sampler_object, bool compare = false, s32 bias_q = 0, u32 aniso = 0);
/// V4: a clear of engine render target view `view_id` (ClearRenderTargetView) / depth stencil view (ClearDepthStencilView: flags 1 depth, 2 stencil) as our image clear in the
/// frame stream. False: counted.
bool ClearColor(u64 view_id, const float* color);
bool ClearDepthStencil(u64 view_id, u32 flags, float depth, u32 stencil);

/// V3c for textures: the engine's CopyResource of one subresource (index = mip + layer x mips, the engine's numbering) as our image copy in the frame stream. False: counted.
bool CopyTexture(u64 destination_id, u32 destination_sub, u64 source_id, u32 source_sub);
/// Engine API map, data paths (note 42): a staging texture (D3D11 usage staging) is read by the CPU through Map(READ). After a copy into one, `StagingCopied` copies that
/// subresource of our image into a host buffer in the same submission; `StagingRead` (at the engine's Map for reading) waits for it and writes its rows to `out` with
/// `row_pitch` bytes per row (a volume mip: its slices `depth_pitch` bytes apart, the pitches the engine's Map returned). False: nothing pending for that texture.
void StagingCopied(u64 texture_id, u32 sub);
bool StagingRead(u64 texture_id, u32 sub, u8* out, u32 row_pitch, u32 depth_pitch = 0);
/// Every subresource the two images share (mips and layers), as CopyTexture; the depth copy of the engine (ExCsDepthCopy)
void CopyTextureAll(u64 destination_id, u64 source_id);

/// Where [address, address + size) of an engine buffer is for the GPU: our device buffer when the range is in one (V2f), else in place. False: neither (counted).
struct BufferWindow {
    vk::Buffer buffer{};
    u64 offset{};
    bool device{};
};
bool FindBuffer(u64 address, u64 size, BufferWindow& out);
/// tier-2 gap pass (debug): the buffer lives in our device memory (its engine memory is not updated by the GPU). No side effects.
bool BufferOnDevice(u64 address, u64 size);
/// Rows by pointer (G3): a generation per engine object pointer (hashed), bumped whenever an engine object is created or deleted at that pointer (the creation and
/// delete hooks of native_device.cpp). A table entry made for an object pointer holds while the pointer's generation is the one it was made with, so a draw finds its
/// rows from the pointers in the context without reading the engine objects. Collisions only bump more often (an entry is remade, never kept wrongly).
inline std::array<std::atomic<u32>, 1u << 16> g_object_gen{};
inline u32 ObjectGenSlot(u64 object) {
    return static_cast<u32>((object * 0x9e3779b97f4a7c15ull) >> 48);
}
inline void ObjectPointerChanged(u64 object) {
    g_object_gen[ObjectGenSlot(object)].fetch_add(1, std::memory_order_release);
}
inline u32 ObjectPointerGen(u64 object) {
    return g_object_gen[ObjectGenSlot(object)].load(std::memory_order_acquire);
}
/// Lantern census (debug): the device buffer under the range is one the GPU writes (bound as an unordered access view); false for a dynamic buffer (the CPU writes it, uploaded at Unmap)
bool BufferGpuWrites(u64 address, u64 size);
/// R3a: the window of engine memory at [address, address + size) as a copy source: our frame ring in device memory, else the in-place import (null buffer: none)
BufferWindow SourceWindow(u64 address, u64 size);
/// V4h census: a bind of a found window by kind (0 compute descriptor, 1 graphics descriptor, 2 vertex buffer, 3 index buffer); counted per frame, in place or device, unique bytes
void NoteBind(u32 kind, const BufferWindow& w, u64 size, bool written);
/// The CPU wrote [address, address + size) of an engine buffer (UpdateSubresource): when it is ours in device memory the bytes are copied in, in the frame stream's order
void BufferCpuWritten(u64 address, u64 size);
/// True when [address, address + size) lies in a buffer we draw from our own device copy (R3d / V4e, mode 2): a CPU write into its memory with no API call
/// (the fur results, open item N94b) would not reach the GPU.
[[nodiscard]] bool BufferHasDeviceCopy(u64 address, u64 size);
/// census: the caller of the next BufferCpuWritten (0 Unmap, 1 UpdateSubresource, 2 other)
void SetCpuWriteKind(u32 kind);
nlohmann::json UploadCensusReport();
bool UploadCensusOn();
/// V2e: the CPU wrote a texture's memory after its creation (UpdateSubresource, the dynamic texture fill): our image gets the contents again (the upload batch, which the
/// queue runs before the frame stream: a dispatch earlier in the same frame sees the new contents too)
void TextureCpuWritten(u64 texture_id, u32 subresource = ~0u); ///< subresource: the one UpdateSubresource named (~0u: the whole texture)
/// api-contract.md Y6: the engine handed the CPU a plain pointer to this buffer range (AsyncMap): its bytes live where the pointer writes (our device copy is retired)
void BufferCpuAddressable(u64 buffer_id);
/// api-contract.md W5: the engine moved texture `old_id` to new memory as texture `new_id` (from mip `first_mip` on): our new image gets the old image's contents
void TextureMoved(u64 new_id, u64 old_id, u32 first_mip);
/// api-contract.md W5b: the engine's buffer move (GpuHeap_MoveOneResource): the new memory gets the old contents now when the old memory holds them (false: copy in the stream)
bool MoveBufferNow(u64 destination, u64 source, u64 bytes);
/// api-contract.md W16: the engine copied mip `mip`, layer `src_layer` of texture `src_id` as raw memory into layer `dst_layer` of texture `dst_id` (probe cubemaps)
void TextureLayerCopied(u64 dst_id, u32 dst_layer, u64 src_id, u32 src_layer, u32 mip);
/// API contract census (debug, TLG_VKB_CONTRACT_CENSUS=1): counts by key, in the objects report under "contract_census"
bool ContractCensusOn();
void ContractNote(const std::string& key);
/// api-contract.md V4 / review A6: an engine buffer view's byte range inside its buffer (`bytes` long), from the view's own V# (SRV at + 0x78, UAV at + 0x74).
/// False with offset 0, size = bytes: the whole buffer.
bool ViewRange(u64 view, bool srv, u64 buffer, u64 bytes, u64& offset, u64& size);
/// C10: a texture bound as target 7 (the target-7 mode), and a shader reading a texture (counted when it was a target-7 texture)
void ContractTarget7(u64 texture_id, const std::string& key);
void ContractSampled(u64 texture_id);
/// Y6: where our bytes of an engine range live ("in place" or a device copy the CPU cannot see)
std::string ContractWhereLives(u64 address, u64 size);
/// The engine read a file into [address, address + bytes) (its file layer, note 81): every texture there gets its contents again; returns how many
u32 MemoryLoaded(u64 address, u64 bytes);
/// debug (tint hunt): every 3D texture uploaded again from the engine's memory; returns how many
u32 ReuploadVolumes();
/// V3c for buffers: the engine's buffer copies and fills (the compute copy, the SDK fill, DMA copies) as copies and fills in the frame stream. False: a range not found (counted).
bool CopyBuffer(u64 destination, u64 source, u64 bytes);
bool FillBuffer(u64 destination, u64 bytes, u32 value);
/// V5c: destination[i] = source[i mod dwords] over `bytes` (the SDK's compute copy with wrap): the pattern copied from the source, then doubled, in the frame stream
bool FillPattern(u64 destination, u64 bytes, u64 source, u32 dwords);
/// note 49 census (TLG_VKB_FILL_READERS=1): the tag of the work being recorded on this thread, a fill target's engine caller
bool FillReadersOn();
void CensusTag(u64 tag);
void R3dDumpTraced(const std::string& label); ///< debug R3d trace: the device buffers of the traced size dumped now
void R3dStaleCheck(const std::string& label); ///< debug (TLG_VKB_R3D_STALE_CHECK=<submit>): device copies vs engine memory, at the census trigger
u64 R3dTraceSize(); ///< debug: TLG_VKB_R3D_TRACE=<bytes> (0: off)
void R3dTraceMapRead(u64 memory, u64 bytes, u64 caller); ///< debug R3d trace: a Map READ against the fur copies into that memory
/// R7 census (TLG_VKB_R7_CENSUS=1, note 56): a buffer binding as today's path resolved it (address from the descriptor words), against the engine object it came from.
/// `bind`: 0 constant buffer, 1 buffer view (SRV), 2 UAV view, 3 vertex, 4 index; `object` the slot's object (a buffer, or a view for 1 and 2)
bool R7CensusOn();
/// R7: our window of engine buffer object `buffer` at byte `offset` for `size` bytes. Resolved once per object (its whole memory) and kept until our buffers change
/// (the buffer epoch); the engine id at +0x10 keys it, so a reused object is a new entry. False: not ours (counted; the caller may try the old path).
bool BufferOfObject(u64 buffer, u64 offset, u64 size, BufferWindow& out);
/// R2b (note 64): the same from fields captured at the draw (the engine object itself is not read: it may be gone when the recorder resolves)
bool BufferOfObjectFields(u64 buffer, u64 id, u64 memory, u64 bytes, u64 offset, u64 size, BufferWindow& out);
/// R7: a constant buffer's current data (the engine renames every constant buffer, note 56): in the frame ring, our window by arithmetic. False: elsewhere (the caller
/// copies its bytes at the draw, as a D3D11 driver versions a constant buffer per draw).
bool ConstantInRing(u64 address, u64 size, BufferWindow& out);
/// R2b (note 64): a buffer range in the frame ring, by arithmetic (resolved at the capture: ring windows are never resolved later)
bool BufferInRing(u64 address, u64 size, BufferWindow& out);
void NoteConstantCopied();
/// note 56 bisect (TLG_VKB_R7_NO_CB_COPY=1): constant buffers outside the frame ring bound in place again, not copied at the draw
inline bool R7NoCopy() { static const bool on = [] { const char* e = std::getenv("TLG_VKB_R7_NO_CB_COPY"); return e != nullptr && e[0] == '1'; }(); return on; }
/// O3 L-CB (TLG_VKB_CB_IN_PLACE=1, note 98): the engine's per-object constant buffers (a pool of 0x130-byte buffers per frame slot, 3 slots in rotation,
/// FrameData_AddRenderable / FrameData_BuildBatches / the pool fill in Engine_RegisterModules) are read by the GPU in place, where the engine wrote them, instead
/// of being copied at every draw: the engine's own contract (on PS4 the GPU reads them in place), safe while the GPU is less than 3 frames behind the writer
[[nodiscard]] bool CbInPlaceOn();
void NotePoolConstantBuffer(u64 object);
[[nodiscard]] bool PoolConstantBuffer(u64 object);
void NotePoolInPlace();
/// at each graphics submit (engine thread): counts submits where the GPU has not finished the submit two back (a pool slot could be rewritten while read)
void PoolSubmitCheck(bool gpu_two_back_done);
void R7Census(u32 bind, u64 object, u64 address, u64 size);
void FillReadersTarget(u64 destination, u64 bytes, u64 caller, const char* what);
/// The async context's counter read-back: bytes of our GDS at `gds_offset` copied to guest memory (or our device buffer) in the frame stream
bool CopyFromGds(u64 gds_offset, u64 destination, u64 bytes);

/// V4c: our image of the engine texture whose memory is display buffer `address` (the swap chain's render target), for the presenter: the image, its format and size. Our
/// images are kept in the general layout; the presenter's barriers go from it and back. False: no such image.
struct DisplayImage {
    vk::Image image{};
    vk::Format format{};
    u32 width{}, height{};
    bool mutable_format{}; ///< note 42: a view of another format (its sRGB twin) is allowed
};
bool DisplayImageAt(u64 address, DisplayImage& out);

/// /stats `vkb_objects`
void PublishObjects();
/// The figures PublishObjects publishes (debugger mode: also snapshot every frame, native_vkb.cpp FrameSnapshots)
[[nodiscard]] nlohmann::json ObjectsReport();
/// note 45: time in our object lookups by kind
[[nodiscard]] nlohmann::json LookupReport();

} // namespace Core::Vkb
