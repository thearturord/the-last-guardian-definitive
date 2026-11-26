#!/usr/bin/env python3
"""Merges the shader recipe folders (user/cache/<serial>-recipes) of several game sessions into one
folder without mixing the sessions' permutation numbers.

Why: a recipe is named <program hash>_<permutation number>, and the permutation number is only
the order in which that session met the program's settings (binding start numbers, run-time
state). The same name can be another shader in another session. A pipeline key names its stages
by program and permutation number, so a key of one session with a same-named recipe of another
session pairs shader code with a descriptor layout or pipeline state it was not built for. The
driver does not check that and crashes while the warm-up builds the pipeline (a 219-scene
recording laid over an older recording crashed the AMD Vulkan driver in 3 of 3 starts; only the
81 recipes whose binding start numbers, run-time state or bytes differ did it, the 792 that
differ only in guest addresses never did).

The rule (every name in the result means what the session that wrote it meant):
  0. A recipe this build cannot build (the import skips it) is left out and named: it gives
     no shader, and without one it cannot be compared.
  1. The first folder is taken whole, names unchanged (its keys as in rule 4).
  2. A program of a further folder that the result does not have yet: all its recipes and the
     keys that use only such programs keep their names.
  3. A recipe of a further folder whose program the result already has: when the game would
     take it for one of the result's permutations (the same settings by the compare the game
     uses to pick a permutation, tlg-probe --same-permutation on the import's output), it is
     that permutation; otherwise it gets the next free permutation number of that program (the
     number inside the file too).
  4. Every key of a further folder gets its stage names translated by the folder's own map
     (rule 2 or 3). A stage the folder has no recipe for keeps its name when the first folder
     has it (the list: a session that used the list names its shaders as the list does); a key
     with any other stage is left out. A
     changed key gets a new file name (g_/c_ and a hash of the new file; the warm-up reads keys
     by content, the name only has to be unique; another key under the same name gets such a
     name too). A key whose content is already in the result is not added twice.
The settings are compared on what this build makes of each recipe: every folder is imported
(tlg-probe --import-shaders) into a scratch folder first, which needs the build's profile.bin
(user/cache/<serial>-<build>/profile.bin, written by any start of that build). The recipe and key
files are read at the byte offsets of that build's formats (recipe version 1 or 2, key version as
written; version 3 recipes, log_recipes.py, hold the portable form that every build reads); use
the tlg-probe of the build that recorded them or reads them. On Linux the probe's
out/lib is put on LD_LIBRARY_PATH (as the launcher does); on Windows its DLLs sit beside it.

Usage: merge_recipes.py --probe TLG_PROBE --profile PROFILE_BIN OUT FIRST_DIR [MORE_DIR ...]
OUT must not exist or be empty. Prints what was taken, renumbered and left out; exit 0 when
written."""
import argparse
import hashlib
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

MASK = (1 << 64) - 1
STAGES = 6  # Shader::SwStage::NumLogicalStages (runtime_info.h)
KEY_HEADER = 8  # u32 version, u32 is_compute (vk_pipeline_serialization.cpp, ReadWarmEntry)
RECIPE_MAGIC = 0x52474C54  # TLGR, shader_recipe.cpp


def hash_combine(seed, value):
    """common/hash.h HashCombine (u64): the stage name a key uses for program + permutation."""
    return seed ^ ((value + 0x9e3779b9 + ((seed << 12) & MASK) + (seed >> 4)) & MASK)


def key_stages(data):
    """(offset, count) of the stage names in a key file: one for a compute key (its value),
    six for a graphics key (GraphicsPipelineKey starts with its stage_hashes array)."""
    if len(data) < KEY_HEADER:
        raise ValueError('key too short')
    _version, is_compute = struct.unpack_from('<II', data, 0)
    count = 1 if is_compute else STAGES
    if len(data) < KEY_HEADER + 8 * count:
        raise ValueError('key too short')
    return bool(is_compute), list(struct.unpack_from('<%dQ' % count, data, KEY_HEADER))


def perm_offset(data):
    """Where a recipe file keeps its permutation number (shader_recipe.cpp, Parse)."""
    magic, version = struct.unpack_from('<II', data, 0)
    if magic != RECIPE_MAGIC or version not in (1, 2, 3):
        raise ValueError('not a version 1, 2 or 3 recipe')
    return 16 + (8 if version >= 2 else 0) + 8 + 8  # header, layout or portable id, stages, program hash


def recipes_of(folder):
    """{program: {perm: bytes}} of a recipe folder."""
    programs = {}
    for f in folder.glob('*.in'):
        program, perm = f.stem.split('_')
        programs.setdefault(int(program, 16), {})[int(perm)] = f.read_bytes()
    return programs


def import_folder(probe, profile, folder, scratch):
    """The folder's recipes built by the import child; returns the cache folder."""
    cache = scratch / f'cache{len(list(scratch.iterdir()))}'
    cache.mkdir()
    shutil.copy2(profile, cache / 'profile.bin')
    env = dict(os.environ)
    lib = probe.parent / 'lib'
    if lib.is_dir():
        env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, [str(lib), env.get('LD_LIBRARY_PATH')]))
    run = subprocess.run([str(probe), '--import-shaders', str(cache), str(folder.resolve())],
                         capture_output=True, text=True, env=env)
    built = [line for line in run.stdout.splitlines() if 'shaders built' in line]
    if run.returncode or not built:
        sys.exit(f'import of {folder} failed (exit {run.returncode}): {run.stdout[-500:]}')
    print(f'{folder}: {built[-1].strip()}')
    metas = {int(f.stem, 16) for f in cache.glob('*.meta')}
    return cache, env, metas


def same_permutation(probe, env, cache_a, cache_b):
    """{(program, perm of b): perm of a} for the shaders of b the game takes for one of a's."""
    run = subprocess.run([str(probe), '--same-permutation', str(cache_a), str(cache_b)],
                         capture_output=True, text=True, env=env)
    if run.returncode:
        sys.exit(f'same-permutation failed (exit {run.returncode}): {run.stderr[-500:]}')
    same = {}
    for line in run.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == 'same':
            same[(int(parts[1], 16), int(parts[2]))] = int(parts[3])
    return same


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--probe', required=True, type=pathlib.Path)
    ap.add_argument('--profile', required=True, type=pathlib.Path)
    ap.add_argument('out', type=pathlib.Path)
    ap.add_argument('dirs', nargs='+', type=pathlib.Path)
    args = ap.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        sys.exit(f'{out} exists and is not empty')
    result = {}   # program -> {perm: recipe bytes}
    keys = {}     # file name -> bytes
    other_keys = {}  # .vkey file name -> bytes (keys of another kind, as they are)
    caches = []   # per folder: its import's cache
    maps = []     # per folder: {(program, own perm): perm in the result}
    with tempfile.TemporaryDirectory() as tmp:
        scratch = pathlib.Path(tmp)
        for n, folder in enumerate(args.dirs):
            programs = recipes_of(folder)
            if programs:
                cache, env, metas = import_folder(args.probe, args.profile, folder, scratch)
                same = [same_permutation(args.probe, env, caches[k], cache) if caches[k] else {}
                        for k in range(n)]
            else:  # keys only (log_recipes.py: a log that named keys, no new shader)
                cache, metas, same = None, set(), [{}] * n
            caches.append(cache)
            own = {}
            kept = matched = renumbered = 0
            unbuilt = []
            for program, perms in sorted(programs.items()):
                have = result.setdefault(program, {})
                fresh = not have
                for perm, data in sorted(perms.items()):
                    if hash_combine(program, perm) not in metas:
                        unbuilt.append(f'{program:#018x}_{perm}')
                        continue
                    target = None
                    if fresh:
                        target = perm
                        kept += 1
                    else:
                        for k in range(n):
                            earlier = same[k].get((program, perm))
                            if earlier is not None and (program, earlier) in maps[k]:
                                target = maps[k][(program, earlier)]
                                matched += 1
                                break
                    if target is None:
                        target = max(have) + 1
                        at = perm_offset(data)
                        data = data[:at] + struct.pack('<Q', target) + data[at + 8:]
                        renumbered += 1
                    have.setdefault(target, data)
                    own[(program, perm)] = target
            maps.append(own)
            stage_map = {hash_combine(p, perm): hash_combine(p, target)
                         for (p, perm), target in own.items()}
            if n:
                # A stage this folder has no recipe for that the first folder has: the same shader
                # (a session that used the list names the list's shaders as the list does: the
                # game loads them from the list's import under the list's names). Logged keys
                # (log_recipes.py) mostly pair a new shader with stages of the list.
                for p, perm in maps[0]:
                    stage_map.setdefault(hash_combine(p, perm), hash_combine(p, perm))
            taken = changed = unknown = dup = damaged = renamed = 0
            left_out = []
            for f in sorted(folder.glob('*.key')):
                data = f.read_bytes()
                try:
                    compute, stages = key_stages(data)
                except ValueError:
                    damaged += 1
                    continue
                if any(s and s not in stage_map for s in stages):
                    unknown += 1
                    left_out.append(f.name)
                    continue
                mapped = [stage_map[s] if s else 0 for s in stages]
                name = f.name
                if mapped != stages:
                    data = (data[:KEY_HEADER] + struct.pack('<%dQ' % len(mapped), *mapped) +
                            data[KEY_HEADER + 8 * len(mapped):])
                    name = (f'c_{mapped[0]:#018x}.key' if compute else
                            f'g_0x{hashlib.blake2b(data, digest_size=8).hexdigest()}.key')
                    changed += 1
                if data in keys.values():
                    dup += 1
                    continue
                if name in keys:  # another key under this name: name this one by its content
                    name = f'{name[:2]}0x{hashlib.blake2b(data, digest_size=8).hexdigest()}.key'
                    renamed += 1
                keys[name] = data
                taken += 1
            # Keys of another kind (.vkey, log_recipes.py): they name the list's own shaders, which keep
            # their names here; taken as they are, once.
            for f in sorted(folder.glob('*.vkey')):
                data = f.read_bytes()
                if data not in other_keys.values():
                    other_keys.setdefault(f.name, data)
            print(f'{folder}: recipes {kept} as they are, {matched} the same shader as one '
                  f'already there, {renumbered} new settings renumbered; keys {taken} taken '
                  f'({changed} with stages renamed), {dup} already there, {unknown} left out (a '
                  f'stage without a recipe), {damaged} unreadable, {renamed} renamed (name taken)')
            for what, names in (('recipes the import could not build, left out', unbuilt),
                                ('keys left out', left_out)):
                if names:
                    print(f'  {what} ({len(names)}): {" ".join(names[:8])}'
                          f'{" ..." if len(names) > 8 else ""}')
    out.mkdir(parents=True, exist_ok=True)
    result = {p: perms for p, perms in result.items() if perms}
    for program, perms in result.items():
        for perm, data in perms.items():
            (out / f'{program:#018x}_{perm}.in').write_bytes(data)
    for name, data in keys.items():
        (out / name).write_bytes(data)
    for name, data in other_keys.items():
        (out / name).write_bytes(data)
    print(f'{out}: {len(result)} programs, {sum(len(p) for p in result.values())} recipes, '
          f'{len(keys)} keys, {len(other_keys)} keys of another kind')


if __name__ == '__main__':
    main()
