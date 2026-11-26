#!/usr/bin/env python3
"""Shader list (Build shaders ahead, tlg.ini shader_prebuild): turns the shader recipes the game recorded
(user/cache/<serial>-recipes) into a list that holds no shader program bytes.

A recipe holds a shader's program bytes (game code) and the pipeline state it was built with. In
the list each program is replaced by where the player's own game files hold it: an entry of the
ps4_shader_cache/*.sb files inside the game's .psarc archives (named by the MD5 of its path, the
archive's own key), or a range of a file of the game folder (eboot.bin). The length and a CRC-32
let the game check that it read the right bytes. A fetch shader (the small program the game
builds at run time to load vertices) is replaced by its vertex input layout, the form the shader
compiler reads it in. Pipeline keys are copied as they are (state and hashes).

The game reads the list at start (shader_recipe.cpp, ImportChild) and builds the missing shaders
in the background before they are needed.

Recipes keep two compiler structs (the shader's run-time state, its binding start numbers) as
raw bytes, and the pipeline keys are raw structs too; how they are laid out depends on the
compiler (Windows lays some bit fields out otherwise than Linux). The list (version 3) holds them
in the portable form instead, every member as a number, which every build reads into its own
structs. That form is made by the game's own code, from the build that recorded the recipes:
  tlg-probe --portable-recipes RECIPE_DIR PORTABLE_FILE
(it reads each recipe and key, writes it in the portable form and checks that it comes back the
same). The list carries the portable id of that form; a build with other field lists does not
read it.

--raw writes the old version 2 list instead (raw struct bytes and the layout id of the recipes;
read only by a build with the same layout). Version 1 recipes carry no layout id: with --raw,
--v1-layout-id gives theirs, once it is proven that they were recorded by a build with that
layout (the import prints its id: "layout id 0x...").
Version 3 recipes (log_recipes.py: a shader the game built while playing that the list lacked,
from the line the game writes into its log) hold the two structs in the portable form already, the
fetch shader as its vertex input layout, and the program's length and CRC-32.
Recipes of several game sessions are merged with merge_recipes.py first, never by copying the
folders over each other (the same recipe name can be another shader in another session).
Usage: make_list.py RECIPE_DIR GAME_DIR OUT --portable PORTABLE_FILE [--report FILE]
       make_list.py RECIPE_DIR GAME_DIR OUT --raw [--v1-layout-id ID] [--report FILE]
Exit 0 when the list was written; prints which programs were not found in the game files.
No game data in the output other than state (registers, descriptors, layouts) and hashes."""
import argparse
import hashlib
import pathlib
import struct
import sys
import zlib

PORT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PORT / 'launcher'))
from soundtrack import Psarc  # noqa: E402

RECIPE_MAGIC, RECIPE_VERSIONS, PORTABLE_RECIPE = 0x52474C54, (1, 2, 3), 3  # TLGR, shader_recipe.cpp
LIST_MAGIC, LIST_VERSION, RAW_LIST_VERSION = b'TLGSLIST', 3, 2  # shader_list.cpp
PORTABLE_MAGIC, PORTABLE_VERSION = b'TLGPORTB', 1  # shader_recipe.cpp, PortableRecipes
SRC_PSARC_SB, SRC_GAME_FILE = 1, 2
ARCHIVES = ['trico_bootp.psarc', 'trico_boot.psarc']  # the update first, as the game reads them


def parse_recipe(data):
    p = 0

    def pod(fmt):
        nonlocal p
        v = struct.unpack_from('<' + fmt, data, p)
        p += struct.calcsize('<' + fmt)
        return v

    def blob():
        nonlocal p
        (n,) = pod('Q')
        b = data[p:p + n]
        p += n
        return b

    magic, ver, sri, sb = pod('IIII')
    if magic != RECIPE_MAGIC or ver not in RECIPE_VERSIONS:
        raise ValueError('not a version 1, 2 or 3 recipe')
    r = {'sri': sri, 'sb': sb, 'version': ver, 'fetch': None}
    # Version 2: the layout id; version 3: the portable id of its two structs.
    r['layout'] = pod('Q')[0] if ver >= 2 else None
    r['hw'], r['sw'] = pod('II')
    r['hash'], r['perm'], r['base'] = pod('QQQ')
    r['ud'], r['vs'], r['flat'] = blob(), blob(), blob()
    if ver == PORTABLE_RECIPE:
        r['rt'], r['start'] = blob(), blob()
        r['length'], r['crc'] = pod('II')
        (has_fetch,) = pod('B')
        if has_fetch:
            size, vofs, iofs, n = pod('IbbB')
            r['fetch'] = {'size': size, 'vofs': vofs, 'iofs': iofs,
                          'attrs': [data[p + 9 * i:p + 9 * i + 9] for i in range(n)]}
            p += 9 * n
    else:
        r['rt'] = data[p:p + sri]
        p += sri
        r['start'] = data[p:p + sb]
        p += sb
    (count,) = pod('Q')
    r['regions'] = []
    for _ in range(count):
        (a,) = pod('Q')
        r['regions'].append((a, blob()))
    if p != len(data):
        raise ValueError('trailing bytes')
    return r


def read_portable(path):
    """The file of tlg-probe --portable-recipes: its portable id, the runtime info and bindings
    per recipe name, and the pipeline keys (file name, portable key)."""
    data = path.read_bytes()
    if data[:8] != PORTABLE_MAGIC or struct.unpack_from('<I', data, 8)[0] != PORTABLE_VERSION:
        raise ValueError('not a portable recipe file of version 1')
    portable_id, layout_id = struct.unpack_from('<QQ', data, 12)
    p = 28

    def u32():
        nonlocal p
        (v,) = struct.unpack_from('<I', data, p)
        p += 4
        return v

    def blob():
        nonlocal p
        n = u32()
        if p + n > len(data):
            raise ValueError('damaged portable recipe file')
        b = data[p:p + n]
        p += n
        return b

    recipes = {}
    for _ in range(u32()):
        name = blob().decode()
        recipes[name] = (blob(), blob())
    keys = []
    for _ in range(u32()):
        name = blob().decode()
        keys.append((name, blob()))
    if p != len(data):
        raise ValueError('trailing bytes in the portable recipe file')
    return portable_id, layout_id, recipes, keys


def decode_fetch(code):
    """Vertex input layout of a fetch shader, as Shader::Gcn::ParseFetchShader reads it, or None
    when it holds an instruction this decoder does not know (the recipe is then left out)."""
    words = [int.from_bytes(code[i:i + 4], 'little') for i in range(0, len(code) - 3, 4)]
    loads = {}  # SMRD destination sgpr -> (dword offset, base sgpr)
    attrs, vofs, iofs, i, semantic = [], -1, -1, 0, 0
    while i < len(words):
        w = words[i]
        if w & 0xFFFFFF00 == 0xBE802000:  # s_setpc_b64 s[0:1]: back to the vertex shader
            return {'size': (i + 1) * 4, 'vofs': vofs, 'iofs': iofs, 'attrs': attrs}
        if w >> 23 == 0x17F:  # SOPP: s_waitcnt and other control, no layout
            if (w >> 16) & 0x7F != 0x0C:
                return None
            i += 1
        elif w >> 27 == 0x18:  # SMRD
            op, sdst, sbase, imm, offset = (w >> 22) & 0x1F, (w >> 15) & 0x7F, (w >> 9) & 0x3F, (w >> 8) & 1, w & 0xFF
            if op != 2 or not imm:  # s_load_dwordx4 with an immediate offset only
                return None
            loads[sdst] = (offset, sbase * 2)
            i += 1
        elif w >> 31 == 0 and (w >> 25) & 0x3F == 0x25:  # VOP2 v_add_i32 vdst, vcc, sN, vdst
            vdst, src0 = (w >> 17) & 0xFF, w & 0x1FF
            if src0 >= 104 or vdst not in (0, 3):
                return None
            if vdst == 0:
                vofs = src0
            else:
                iofs = src0
            i += 1
        elif w >> 26 == 0x38:  # MUBUF
            if i + 1 >= len(words):
                return None
            w1 = words[i + 1]
            op = (w >> 18) & 0x7F
            if op > 3:  # buffer_load_format_x .. _xyzw only
                return None
            srsrc = ((w1 >> 16) & 0x1F) * 4
            if srsrc not in loads:
                return None
            if w & 0xFFF > 0xFF:  # the layout keeps the offset in a byte
                return None
            dword_offset, base_sgpr = loads[srsrc]
            attrs.append(bytes([semantic, (w1 >> 8) & 0xFF, op + 1, base_sgpr, dword_offset,
                                w1 & 0xFF, w & 0xFFF, 0, 0]))
            semantic += 1
            i += 2
        else:  # MTBUF and anything else: not seen in this game's recordings yet
            return None
    return None


def looks_like_code(address, data, base, length):
    """Memory the compiler read that may be game code, the rule of shader_recipe.cpp LooksLikeCode
    and LogUnlisted: more than 64 bytes, the shader header signature, an s_setpc_b64 (the end of a
    fetch shader) in a word aligned in memory, or a place inside the program's range."""
    if len(data) > 64 or data.find(b'OrbShdr') >= 0 or (address < base + length and address + len(data) > base):
        return True
    return any((int.from_bytes(data[i:i + 4], 'little') & 0xFFFFFF00) == 0xBE802000
               for i in range((4 - address % 4) % 4, len(data) - 3, 4))


def blob(b):
    return struct.pack('<I', len(b)) + bytes(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recipes', type=pathlib.Path)
    ap.add_argument('game', type=pathlib.Path)
    ap.add_argument('out', type=pathlib.Path)
    ap.add_argument('--report', type=pathlib.Path)
    ap.add_argument('--portable', type=pathlib.Path,
                    help='the file of tlg-probe --portable-recipes RECIPE_DIR FILE (version 3 list)')
    ap.add_argument('--raw', action='store_true',
                    help='the old version 2 list: raw struct bytes, read only by a build of the same layout')
    ap.add_argument('--v1-layout-id', type=lambda v: int(v, 0),
                    help='with --raw: the layout id of version 1 recipes (they carry none)')
    ap.add_argument('--keys-from', type=pathlib.Path,
                    help='a version 3 list whose pipeline keys go into the new list as they are '
                         '(list_recipes.py: the recipe folder of that list holds no keys)')
    a = ap.parse_args()
    if a.keys_from is not None and a.raw:
        sys.exit('--keys-from: with --portable only')
    if a.raw == (a.portable is not None):
        sys.exit('give --portable FILE (the list for every build) or --raw (see the usage)')
    if a.v1_layout_id is not None and (not a.raw or not 0 <= a.v1_layout_id < 1 << 64):
        sys.exit('--v1-layout-id: with --raw only, a 64-bit number as the import prints it')

    recipes = []
    for f in sorted(a.recipes.glob('*.in')):
        try:
            recipes.append(parse_recipe(f.read_bytes()) | {'name': f.stem})
        except (ValueError, struct.error) as e:
            print(f'{f.name}: skipped ({e})')
    if not recipes:
        sys.exit('no recipes')
    v1 = sum(r['layout'] is None for r in recipes)
    unportable = []
    if a.raw:
        if any(r['version'] == PORTABLE_RECIPE for r in recipes):
            sys.exit('version 3 recipes (portable form): with --portable only')
        keys = [(k.name, k.read_bytes()) for k in sorted(a.recipes.glob('*.key'))]
        # One layout for the whole list: the game checks one id, so every recipe must have it.
        if v1 and a.v1_layout_id is None:
            sys.exit(f'{v1} version 1 recipes carry no layout id: give --v1-layout-id (see the usage)')
        for r in recipes:
            if r['layout'] is None:
                r['layout'] = a.v1_layout_id
        layouts = {}
        for r in recipes:
            layouts.setdefault((r['sri'], r['sb'], r['layout']), []).append(r)
        if len(layouts) > 1:
            sys.exit('recipes of several struct layouts, not mixed into one list:\n' + '\n'.join(
                f'  {len(rs)} recipes: struct sizes {k[0]} and {k[1]}, layout id {k[2]:#018x}'
                for k, rs in layouts.items()))
        sri, sbz, list_id = next(iter(layouts))
    else:
        # The recipes' structs and the keys in the portable form, as the game's code wrote them.
        try:
            list_id, portable_layout, portable, keys = read_portable(a.portable)
        except (ValueError, struct.error, UnicodeDecodeError) as e:
            sys.exit(f'{a.portable}: {e}')
        for r in recipes:
            if r['name'] in portable:
                r['rt'], r['start'] = portable[r['name']]
            else:
                unportable.append(r)
        recipes = [r for r in recipes if r['name'] in portable]
        if not recipes:
            sys.exit('no recipe of the folder is in the portable file')
        if a.keys_from is not None:
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
            from list_recipes import read_list  # noqa: E402 (beside this file)
            try:
                from_id, _sources, _entries, from_keys = read_list(a.keys_from)
            except (ValueError, struct.error, zlib.error) as e:
                sys.exit(f'{a.keys_from}: {e}')
            if from_id != list_id:
                sys.exit(f'--keys-from: that list has portable id {from_id:#018x}, the recipes {list_id:#018x}')
            have = {data for _name, data in keys}
            keys = keys + [('', data) for data in from_keys if data not in have]

    # The game's shader files, and where each program hash has its header (OrbShdr) in them.
    sb_files, footers = {}, {}
    for name in ARCHIVES:
        archive = Psarc(a.game / name)
        for n in archive.names:
            if n.startswith('ps4_shader_cache/') and n.endswith('.sb'):
                data = archive.read(n)
                sb_files[(name, n)] = data
                at = data.find(b'OrbShdr')
                while at >= 0:
                    footers.setdefault(int.from_bytes(data[at + 16:at + 24], 'little'), []).append((name, n))
                    at = data.find(b'OrbShdr', at + 1)
        archive.close()
    eboot = (a.game / 'eboot.bin').read_bytes()

    sources, source_index = [], {}

    def source(kind, ident):
        if (kind, ident) not in source_index:
            source_index[(kind, ident)] = len(sources)
            sources.append((kind, ident))
        return source_index[(kind, ident)]

    located, missing, left_out = {}, [], []
    for r in recipes:
        program = dict(r['regions']).get(r['base'])
        if program is None or r['hash'] in located:
            continue
        ref = None
        for archive_name, n in footers.get(r['hash'], []):
            at = sb_files[(archive_name, n)].find(program)
            if at >= 0:
                ref = (source(SRC_PSARC_SB, hashlib.md5(n.upper().encode()).digest()), at)
                break
        if ref is None and (at := eboot.find(program)) >= 0:
            ref = (source(SRC_GAME_FILE, b'eboot.bin'), at)
        if ref is None:
            missing.append(r)
            located[r['hash']] = None
        else:
            located[r['hash']] = (ref[0], ref[1], len(program), zlib.crc32(program))

    out = bytearray(struct.pack('<II', sri, sbz) if a.raw else b'')
    out += struct.pack('<I', len(sources))
    for kind, ident in sources:
        out += struct.pack('<B', kind) + blob(ident)
    entries = bytearray()
    count = fetch_count = 0
    for r in recipes:
        loc = located.get(r['hash'])
        if loc is None:
            continue
        ud = [int.from_bytes(r['ud'][i:i + 4], 'little') for i in range(0, len(r['ud']), 4)]
        pointers = {ud[i] | ud[i + 1] << 32 for i in range(len(ud) - 1)}
        fetch, data_regions, bad = r['fetch'], [], False
        for address, data in r['regions']:
            if address == r['base']:
                continue
            if address in pointers and (layout := decode_fetch(data)) is not None and fetch is None:
                fetch = layout
            elif not looks_like_code(address, data, r['base'], loc[2]):
                data_regions.append((address, data))  # resource descriptors (V#), state
            else:
                bad = True  # looks like code: never copied into the list
        if r['vs']:
            bad = True  # a geometry shader's copy shader is code
        if bad:
            left_out.append(r)
            continue
        e = struct.pack('<IIQQQ', r['hw'], r['sw'], r['hash'], r['perm'], r['base'])
        e += blob(r['ud']) + blob(r['vs']) + blob(r['flat'])
        e += r['rt'] + r['start'] if a.raw else blob(r['rt']) + blob(r['start'])
        e += struct.pack('<IQII', *loc)
        if fetch:
            fetch_count += 1
            e += struct.pack('<BIbbB', 1, fetch['size'], fetch['vofs'], fetch['iofs'], len(fetch['attrs']))
            e += b''.join(fetch['attrs'])
        else:
            e += b'\0'
        e += struct.pack('<I', len(data_regions))
        for address, data in data_regions:
            e += struct.pack('<Q', address) + blob(data)
        entries += e
        count += 1
    out += struct.pack('<I', count) + entries
    out += struct.pack('<I', len(keys))
    for name, data in keys:
        # Version 3: the game names a key file itself (by its key's hash in that build).
        out += blob(name.encode()) + blob(data) if a.raw else blob(data)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    version = RAW_LIST_VERSION if a.raw else LIST_VERSION
    a.out.write_bytes(LIST_MAGIC + struct.pack('<IIQQ', version, 0, len(out), list_id) +
                      zlib.compress(bytes(out), 9))
    programs = len(located)
    if a.raw:
        ident = (f'layout id {list_id:#018x}'
                 + (f' ({v1} version 1 recipes given it by --v1-layout-id)' if v1 else ''))
    else:
        ident = (f'portable id {list_id:#018x} (made by a build of layout id {portable_layout:#018x})')
    lines = [f'list version {version}: {len(recipes)} recipes ({programs} programs), {len(keys)} pipeline keys, {ident}',
             f'list: {count} recipes, {fetch_count} with a fetch layout, {len(sources)} game file sources, '
             f'{a.out.stat().st_size} bytes',
             f'programs found in the game files: {programs - len(missing)} of {programs}']
    for r in missing:
        lines.append(f'  not found: program {r["hash"]:#018x} (hw stage {r["hw"]}, {len(dict(r["regions"])[r["base"]])} bytes)')
    for r in unportable:
        lines.append(f'  left out (not in the portable file: the build could not read it): {r["name"]}')
    for r in left_out:
        lines.append(f'  left out (memory that looks like code): {r["hash"]:#018x}_{r["perm"]}')
    print('\n'.join(lines))
    if a.report:
        a.report.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
