# Texture packs

The port can replace the game's textures with sharper, AI-upscaled versions while it runs. Packs
are made from **your own copy of the game** with
[ps4-texturepack-generator](https://github.com/tabnul/ps4-texturepack-generator) (profile `tlg`),
which reads the game folder without writing to it. Packs are not distributed with the port.

Packs are optional. The game runs exactly as before when no pack is installed or **Texture pack** is
set to *Off* (the default).

## Making a pack

**We advise building it from the launcher**: it downloads everything it needs, nothing has to be
installed, and it checks disk space first. Building by hand with the generator is the fallback, for
example when the launcher build does not work on your system.

### From the launcher (experimental, advised)

*Image* > *Texture pack* > **Build a texture pack… (experimental)**. Pick the graphics card kind
and a pack name, then press **Start**. The launcher does the rest:

1. It downloads a ready-made generator kit for your system and card, once. The kit holds Python,
   PyTorch, the generator and texconv, and nothing gets installed. The kits come from the
   generator's [`kit-v1` release](https://github.com/tabnul/ps4-texturepack-generator/releases/tag/kit-v1).
2. It downloads the upscale model, 4x-UltraSharp, from its author
   [Kim2091](https://huggingface.co/Kim2091/UltraSharp). Its license is CC BY-NC-SA 4.0, which allows
   no commercial use.
3. It builds the pack into the texture pack folder and selects it when it is done.

| Card | Kit | Download / on disk |
|---|---|---|
| NVIDIA RTX 20 or newer | CUDA | Windows 1.9 / 4.7 GB, Linux 2.6 / 7.5 GB |
| AMD (experimental) | Windows DirectML, Linux ROCm | Windows 0.2 / 1.4 GB, Linux 5.7 / 15.8 GB |
| Other (GTX 10 and older, Intel) | CPU only | 0.1-0.2 / 0.7-1.4 GB; a build takes days |

Before you start:
- There is only **FULL** for now, which upscales *every* texture. **Light** (only the textures the
  game draws larger than they are) is planned and will be the advised choice; the plan and progress
  are in [issue #43](https://github.com/tabnul/the-last-guardian-definitive/issues/43).
- **Disk**: about **160 GB** free while building. About **80 GB** stays once the pack is done.
- **Time**: about 3 hours on an RTX 5090. AMD has not been timed yet, and the CPU kit takes days.
  **Stop** ends the build; **Start** again later continues where it stopped.
- The build uses the GPU fully and a lot of RAM (tested with 32 GB). Don't play while it runs.
- The build log is `<pack name>.build.log` in the texture pack folder (*Open log*).

### By hand (fallback)

Use this if the launcher build does not work for you. Follow the generator's README: set it up once, then
```
texpack make "D:\Games\CUSA03745" "D:\Packs\UltraSharp 4x"
```
A full pack (colour 4x, normal maps 2x) takes a few hours on a recent NVIDIA card. The pack folder's
name is the name you see in the launcher and in game. Don't play while it builds: both want the GPU.

## Installing and using it

1. Put the pack folder in the texture pack folder: `texpack\` next to the port's exe by default, or
   the folder set under *Texture packs folder* in the launcher (environment variable `TLG_TEXPACK_DIR`).
   ```
   <texture pack folder>\UltraSharp 4x\index.txt
   <texture pack folder>\UltraSharp 4x\game\...
   ```
2. Pick it:
   - **Launcher**: *Image* > *Texture pack* lists the packs in that folder.
   - **In game**: the overlay menu (Insert) > *Image* > *Textures* > *Texture pack*. It switches
     live; textures appear as they load (a second or two).
   - **Presets** store the pack too: *Stock (as shipped)* is the game without any pack.

Several packs can sit side by side (for example a 2x and a 4x one); you pick one at a time.

### Video memory

Pack textures load when the game first draws them and are released again:
- textures not drawn for **30 seconds** are freed (they reload when needed),
- switching packs or setting *Off* frees all of them,
- the pack only uses video memory that is **free**. It loads new textures while the card is under
  65 % full and frees pack textures again above 70 %, checking every 5 seconds. Beyond that,
  textures stay at the original quality, and the game's own textures always come first.
  `TLG_TEXPACK_BUDGET_MB` sets a fixed limit instead.

On cards with 8 GB or less, on integrated graphics and on the Steam Deck, few pack textures may
fit. A 2x pack (`--scale 2`) is the safer choice there (a quarter of the memory per texture).

## Troubleshooting

| Problem | Fix |
|---|---|
| The pack does not show up | The folder must contain `index.txt` and sit directly in the texture pack folder (no extra folder level) |
| Some textures look unchanged | They are not in the pack (see the generator's *What goes in the pack*), or were not drawn long enough to load |

## For developers

`index.txt` lines are `<key> <width> <height> <dds path>`: the key is XXH3-64 of the texture's
tiled mip 0 as it sits in GPU memory, so a pack matches the game whatever file the texture came
from. The full format is in the generator's README.

The loader is `Vulkan::TexturePack` (`gpu/shadps4/video_core/renderer_vulkan/vk_texture_pack.cpp`).
Debug variables: `TLG_TEXPACK_LOG=1` (every load), `TLG_TEXPACK_TINT=1` (tints pack textures),
`TLG_TEXPACK_VMA=1` (video memory every 10 s).
