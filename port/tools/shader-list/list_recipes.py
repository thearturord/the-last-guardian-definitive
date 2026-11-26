#!/usr/bin/env python3
"""Recipe folder from a shader list: the way back of make_list.py, so shaders can be added to a
list without the recordings it was made from.

Each list entry (version 3: the portable form) becomes a version 3 recipe file, the form
log_recipes.py writes: its program is read back out of the player's own game files (the source
and offset the entry names), checked against the entry's length and CRC-32. Recipe names are the
list's own (<program hash>_<permutation>), so the list's pipeline keys keep naming the right
shaders. The keys are not written (the list holds them in the portable form only); make_list.py
--keys-from LIST copies them into the new list as they are.

A recipe file holds the program, like the recipes the game records: it is never shipped.

Adding shaders from game logs (the lines "Shader recipe (not in list)"):
  list_recipes.py LIST GAME_DIR FROM_LIST
  log_recipes.py GAME_DIR LOGGED LOG [LOG ...]
  merge_recipes.py --probe out/tlg-probe --profile <cache>/profile.bin MERGED FROM_LIST LOGGED/*
  tlg-probe --portable-recipes MERGED PORTABLE_FILE
  make_list.py MERGED GAME_DIR NEW_LIST --portable PORTABLE_FILE --keys-from LIST
  check_no_code.py NEW_LIST GAME_DIR
The list folder comes first in the merge, so its names stay as they are (merge_recipes.py rule 1).
Exit 0 when every entry was written."""
import argparse
import hashlib
import pathlib
import struct
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_list as ml  # noqa: E402
from log_recipes import write_v3  # noqa: E402


def read_list(path):
    """(portable id, sources, entries, keys) of a version 3 list (make_list.py writes it)."""
    data = path.read_bytes()
    if data[:8] != ml.LIST_MAGIC:
        raise ValueError('not a shader list')
    version, _zero, size, list_id = struct.unpack_from('<IIQQ', data, 8)
    if version != ml.LIST_VERSION:
        raise ValueError(f'list version {version}: only version {ml.LIST_VERSION} (portable) is read')
    body = zlib.decompress(data[8 + 24:])
    if len(body) != size:
        raise ValueError('unpacked size differs from the header')
    p = 0

    def pod(fmt):
        nonlocal p
        v = struct.unpack_from('<' + fmt, body, p)
        p += struct.calcsize('<' + fmt)
        return v

    def blob():
        nonlocal p
        (n,) = pod('I')
        if p + n > len(body):
            raise ValueError('damaged list')
        b = body[p:p + n]
        p += n
        return b

    sources = []
    for _ in range(pod('I')[0]):
        (kind,) = pod('B')
        sources.append((kind, blob()))
    entries = []
    for _ in range(pod('I')[0]):
        e = dict(zip(('hw', 'sw', 'hash', 'perm', 'base'), pod('IIQQQ')))
        e['ud'], e['vs'], e['flat'], e['rt'], e['start'] = blob(), blob(), blob(), blob(), blob()
        e['src'], e['offset'], e['length'], e['crc'] = pod('IQII')
        (has_fetch,) = pod('B')
        e['fetch'] = None
        if has_fetch:
            fsize, vofs, iofs, n = pod('IbbB')
            e['fetch'] = {'size': fsize, 'vofs': vofs, 'iofs': iofs,
                          'attrs': [body[p + 9 * i:p + 9 * i + 9] for i in range(n)]}
            p += 9 * n
        e['regions'] = []
        for _ in range(pod('I')[0]):
            (address,) = pod('Q')
            e['regions'].append((address, blob()))
        entries.append(e)
    keys = [blob() for _ in range(pod('I')[0])]
    if p != len(body):
        raise ValueError('trailing bytes in the list')
    return list_id, sources, entries, keys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('list', type=pathlib.Path)
    ap.add_argument('game', type=pathlib.Path)
    ap.add_argument('out', type=pathlib.Path)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f'{a.out} exists and is not empty')
    try:
        list_id, sources, entries, keys = read_list(a.list)
    except (ValueError, struct.error, zlib.error) as e:
        sys.exit(f'{a.list}: {e}')

    # The game's files the sources name: .sb entries by the MD5 of their upper-case path, or a
    # file of the game folder.
    wanted = {ident for kind, ident in sources if kind == ml.SRC_PSARC_SB}
    sb = {}
    for name in ml.ARCHIVES:
        archive = ml.Psarc(a.game / name)
        for n in archive.names:
            digest = hashlib.md5(n.upper().encode()).digest()
            if digest in wanted and digest not in sb:  # the update first, as the game reads them
                sb[digest] = archive.read(n)
        archive.close()
    files = {}
    a.out.mkdir(parents=True, exist_ok=True)
    written, problems = 0, []
    for e in entries:
        name = f'{e["hash"]:#018x}_{e["perm"]}'
        kind, ident = sources[e['src']]
        if kind == ml.SRC_PSARC_SB:
            data = sb.get(ident)
        elif kind == ml.SRC_GAME_FILE:
            rel = ident.decode()
            if rel not in files:
                path = a.game / rel
                files[rel] = path.read_bytes() if path.is_file() else None
            data = files[rel]
        else:
            data = None
        program = data[e['offset']:e['offset'] + e['length']] if data is not None else b''
        if len(program) != e['length'] or zlib.crc32(program) != e['crc']:
            problems.append(f'{name}: program not in the game files (length {e["length"]}, CRC-32 {e["crc"]:#010x})')
            continue
        (a.out / f'{name}.in').write_bytes(write_v3(e | {'layout': list_id}, program))
        written += 1
    print(f'{a.list}: portable id {list_id:#018x}, {len(entries)} entries, {len(keys)} pipeline keys '
          f'(not written: make_list.py --keys-from)')
    print(f'{written} recipe files written into {a.out}, {len(problems)} not usable')
    for line in problems:
        print('  ' + line)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
