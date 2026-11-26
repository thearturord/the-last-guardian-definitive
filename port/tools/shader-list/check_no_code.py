#!/usr/bin/env python3
"""Checks that a shader list holds no bytes of the game's shader programs.

Every 32-byte window of the unpacked list (at every byte position) that is not low in content
(at least 12 different byte values: zero runs, small integers and repeated words are left out)
is looked up at every 4-byte position of every ps4_shader_cache/*.sb file of the game's .psarc
archives and of eboot.bin. Shader code is 4-byte aligned. A hit prints where it is; exit 1 then.
Also: the list must not contain the shader header signature "OrbShdr".
--file: FILE is not a list but checked as it is (log_recipes.py --payloads: the recipes as the
game logs hold them). Exit 2 when nothing could be checked (an empty or low-content file).

Usage: check_no_code.py LIST GAME_DIR
       check_no_code.py --file FILE GAME_DIR"""
import pathlib
import struct
import sys
import zlib

PORT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PORT / 'launcher'))
from soundtrack import Psarc  # noqa: E402

W = 32


def main():
    args = sys.argv[1:]
    plain = args[:1] == ['--file']
    if plain:
        args = args[1:]
    if len(args) != 2:
        sys.exit(__doc__)
    raw = pathlib.Path(args[0]).read_bytes()
    if not plain:
        if raw[:8] != b'TLGSLIST':
            sys.exit('not a shader list (use --file for another file)')
        (version,) = struct.unpack_from('<I', raw, 8)
        (size,) = struct.unpack_from('<Q', raw, 16)
        payload = zlib.decompress(raw[24 if version == 1 else 32:])  # version 2: + layout id, 3: + portable id
        assert len(payload) == size
    else:
        payload = raw
    windows = {}
    skipped = 0
    for at in range(0, len(payload) - W + 1):  # every byte position of the list
        w = payload[at:at + W]
        if len(set(w)) < 12:
            skipped += 1
            continue
        windows.setdefault(w, at)
    print(f'{"file" if plain else "list"}: {len(payload)} bytes unpacked, {len(windows)} distinct windows checked, '
          f'{skipped} low-content windows left out')
    hits = 0
    if b'OrbShdr' in payload:
        print('HIT: the list holds the shader header signature OrbShdr')
        hits += 1
    if not windows:
        print('nothing to check: no window with enough content')
        sys.exit(1 if hits else 2)
    game = pathlib.Path(args[1])
    sources = []
    for name in ['trico_bootp.psarc', 'trico_boot.psarc']:
        archive = Psarc(game / name)
        for n in archive.names:
            if n.startswith('ps4_shader_cache/'):
                sources.append((f'{name}:{n}', archive.read(n)))
        archive.close()
    sources.append(('eboot.bin', (game / 'eboot.bin').read_bytes()))
    scanned = 0
    for label, data in sources:
        for at in range(0, len(data) - W + 1, 4):
            w = data[at:at + W]
            if w in windows:
                hits += 1
                if hits <= 20:
                    print(f'HIT: list offset {windows[w]} = {label} offset {at}')
        scanned += len(data)
    print(f'game files scanned: {len(sources)} files, {scanned} bytes; hits: {hits}')
    sys.exit(1 if hits else 0)


if __name__ == '__main__':
    main()
