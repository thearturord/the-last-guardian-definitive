#!/usr/bin/env python3
"""Shader recipes from game logs: the lines the game writes for a shader it built while playing that
the shader list lacked (shader_recipe.cpp, LogUnlisted), turned back into recipe files, so
merge_recipes.py and make_list.py can add them to the list.

A line is
  Shader recipe (not in list): <program hash>_<permutation> <base64 of the zlib-packed recipe>
The recipe is version 3 (the portable form of its state, the same for every build and OS) and
holds no game code: of the program only its length and CRC-32. This tool reads the program back
out of the player's own game files (the ps4_shader_cache/*.sb entries of the .psarc archives
whose shader header names the program hash, or eboot.bin), checks length and CRC-32, and writes
the full recipe. A recipe file holds the program, like the recipes the game records: it is never
shipped; only the list made from it is.

The permutation number is only valid within one game session, so the recipes of each log go into
a folder of their own (OUT/<log name>; a second shader of the same name within one log starts a
further folder), for merge_recipes.py to merge them with the other recordings:
  log_recipes.py GAME_DIR OUT LOG [LOG ...] [--payloads FILE]
  merge_recipes.py --probe out/tlg-probe --profile <cache>/profile.bin MERGED RECORDED OUT/*
  tlg-probe --portable-recipes MERGED PORTABLE_FILE
  make_list.py MERGED GAME_DIR LIST --portable PORTABLE_FILE
A line
  Pipeline key (not in list): <key file name> <base64 of the zlib-packed portable key>
(shader_recipe.cpp, LogUnlistedKey) names a pipeline the game made whose key the list lacks: it
goes into the log's folder as <key file name>.pkey (the portable form; stage names and pipeline
state, no game code). add_from_logs.py turns them into key files (tlg-probe --keys-from-portable)
before merge_recipes.py, which keeps a key whose stages are this log's recipes or the list's.
--payloads FILE writes the unpacked recipes exactly as the logs hold them, one after the other,
for check_no_code.py --file FILE GAME_DIR (proof that the log lines hold no game code).
Exit 0 when every recipe line was turned into a recipe file."""
import argparse
import base64
import binascii
import pathlib
import re
import struct
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_list as ml  # noqa: E402

LINE = re.compile(r'Shader recipe \(not in list\): (0x[0-9a-f]{16})_(\d+) ([A-Za-z0-9+/]+=*)\s*$')
# c_/g_ ...key: keys this build reads; v_ ...vkey: keys of another kind, already final.
KEY_LINE = re.compile(r'Pipeline key \(not in list\): ([cg]_0x[0-9a-f]{16}\.key|v_0x[0-9a-f]{16}\.vkey) ([A-Za-z0-9+/]+=*)\s*$')
KEY_LEFT_OUT = re.compile(r'Pipeline key \(not in list\): ([cgv]_0x[0-9a-f]{16}\.v?key) left out: (.*)$')
LEFT_OUT = re.compile(r'Shader recipe \(not in list\): (0x[0-9a-f]{16}_\d+) left out: (.*)$')
SCAN = 0x10000  # how far before a shader header in eboot.bin its program may start
MAX_RECIPE = 1 << 20  # a logged recipe unpacks to a few KB at most; more is a damaged or forged line


def unpack(text):
    d = zlib.decompressobj()
    data = d.decompress(base64.b64decode(text, validate=True), MAX_RECIPE)
    if d.unconsumed_tail or not d.eof:
        raise ValueError('cut short, or does not unpack within 1 MB')
    return data


def write_v3(r, program):
    """The version 3 recipe file (shader_recipe.cpp, Parse) of a logged recipe with its program."""
    def blob(b):
        return struct.pack('<Q', len(b)) + bytes(b)
    out = struct.pack('<IIIIQ', ml.RECIPE_MAGIC, ml.PORTABLE_RECIPE, 0, 0, r['layout'])
    out += struct.pack('<II', r['hw'], r['sw']) + struct.pack('<QQQ', r['hash'], r['perm'], r['base'])
    out += blob(r['ud']) + blob(r['vs']) + blob(r['flat']) + blob(r['rt']) + blob(r['start'])
    out += struct.pack('<II', r['length'], r['crc'])
    f = r['fetch']
    if f:
        out += struct.pack('<BIbbB', 1, f['size'], f['vofs'], f['iofs'], len(f['attrs'])) + b''.join(f['attrs'])
    else:
        out += b'\0'
    regions = sorted(r['regions'] + [(r['base'], program)])
    out += struct.pack('<Q', len(regions))
    for address, data in regions:
        out += struct.pack('<Q', address) + blob(data)
    return out


class GameShaders:
    """Programs of the game folder by hash: the .sb entries whose shader header (OrbShdr) names
    the hash, and the places in eboot.bin where a header names it."""

    def __init__(self, game):
        self.sb = {}  # hash -> [.sb bytes]
        for name in ml.ARCHIVES:
            archive = ml.Psarc(game / name)
            for n in archive.names:
                if n.startswith('ps4_shader_cache/') and n.endswith('.sb'):
                    data = archive.read(n)
                    at = data.find(b'OrbShdr')
                    while at >= 0:
                        self.sb.setdefault(int.from_bytes(data[at + 16:at + 24], 'little'), []).append(data)
                        at = data.find(b'OrbShdr', at + 1)
            archive.close()
        self.eboot = (game / 'eboot.bin').read_bytes()

    def program(self, pgm_hash, length, crc):
        """The program bytes of that length and CRC-32, or None."""
        def check(data, start):
            code = data[start:start + length]
            return code if start >= 0 and len(code) == length and zlib.crc32(code) == crc else None
        for data in self.sb.get(pgm_hash, []):
            if len(data) >= 64 and (code := check(data, 52 + struct.unpack_from('<I', data, 60)[0])):
                return code  # where the .sb header puts the code
            for start in range(0, len(data) - length + 1, 4):
                if code := check(data, start):
                    return code
        key = pgm_hash.to_bytes(8, 'little')
        at = self.eboot.find(b'OrbShdr')
        while at >= 0:
            if self.eboot[at + 16:at + 24] == key:
                for start in range(at - length, max(-1, at - length - SCAN), -4):
                    if code := check(self.eboot, start):
                        return code
            at = self.eboot.find(b'OrbShdr', at + 1)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('game', type=pathlib.Path)
    ap.add_argument('out', type=pathlib.Path)
    ap.add_argument('logs', type=pathlib.Path, nargs='+')
    ap.add_argument('--payloads', type=pathlib.Path)
    a = ap.parse_args()
    shaders = None
    payloads = bytearray()
    sizes, folders, problems = [], [], 0
    written = left_out = keys = 0
    for log in a.logs:
        stem = log.stem
        folder_of = {}  # name -> bytes written there, per folder of this log
        log_folders = []
        for number, line in enumerate(log.read_text(errors='replace').splitlines(), 1):
            if m := KEY_LEFT_OUT.search(line):
                print(f'{log.name}:{number}: key {m[1]} left out by the game ({m[2]})')
                continue
            if m := KEY_LINE.search(line):
                try:
                    key = unpack(m[2])
                except (binascii.Error, zlib.error, ValueError) as e:
                    print(f'{log.name}:{number}: key {m[1]}: damaged line ({e})')
                    problems += 1
                    continue
                if not log_folders:
                    log_folders.append(a.out / stem)
                log_folders[0].mkdir(parents=True, exist_ok=True)
                # a .vkey is kept as it is (merge_recipes.py and the list take it unchanged)
                target = log_folders[0] / (m[1] if m[1].endswith('.vkey') else f'{m[1]}.pkey')
                if not target.exists():
                    target.write_bytes(key)
                    keys += 1
                continue
            if m := LEFT_OUT.search(line):
                left_out += 1
                print(f'{log.name}:{number}: {m[1]} left out by the game ({m[2]})')
                continue
            m = LINE.search(line)
            if not m:
                continue
            name = f'{m[1]}_{m[2]}'
            sizes.append(len(line.encode()) + 1)
            try:
                payload = unpack(m[3])
                r = ml.parse_recipe(payload)
            except (binascii.Error, zlib.error, ValueError, struct.error) as e:
                print(f'{log.name}:{number}: {name}: damaged line ({e})')
                problems += 1
                continue
            if r['version'] != ml.PORTABLE_RECIPE or f'{r["hash"]:#018x}_{r["perm"]}' != name:
                print(f'{log.name}:{number}: {name}: not a version 3 recipe of that name')
                problems += 1
                continue
            payloads += payload
            if shaders is None:
                shaders = GameShaders(a.game)
            program = shaders.program(r['hash'], r['length'], r['crc'])
            if program is None:
                print(f'{log.name}:{number}: {name}: program ({r["length"]} bytes, CRC-32 '
                      f'{r["crc"]:#010x}) not in the game files')
                problems += 1
                continue
            data = write_v3(r, program)
            # A further shader of the same name is of another session: the next folder.
            index = 0
            while index < len(log_folders) and folder_of.get((index, name), data) != data:
                index += 1
            if index == len(log_folders):
                log_folders.append(a.out / (stem if index == 0 else f'{stem}-{index + 1}'))
            if (index, name) in folder_of:
                continue
            folder_of[(index, name)] = data
            log_folders[index].mkdir(parents=True, exist_ok=True)
            (log_folders[index] / f'{name}.in').write_bytes(data)
            written += 1
        folders += log_folders
    if a.payloads:
        a.payloads.write_bytes(bytes(payloads))
    print(f'{len(sizes)} recipe lines in {len(a.logs)} logs: {written} recipe files written into '
          f'{len(folders)} folders, {keys} pipeline keys, {problems} not usable, {left_out} shaders '
          f'left out by the game')
    if sizes:
        typical = sorted(sizes)[len(sizes) // 2]
        print(f'line size: largest {max(sizes)} bytes, typical (median) {typical} bytes')
    for folder in folders:
        print(f'  {folder}')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
