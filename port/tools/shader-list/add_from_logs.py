#!/usr/bin/env python3
"""Adds the shaders that players' game logs name as missing to a shader list: logs in, a checked
list out. One command for the steps in list_recipes.py:

  1. the current list back into a recipe folder (list_recipes.py; programs from the game files);
  2. the log lines "Shader recipe (not in list)" and "Pipeline key (not in list)" into recipe
     folders (log_recipes.py), the keys as this build's key files (tlg-probe
     --keys-from-portable);
  3. both merged, the list first, so its names stay (merge_recipes.py; renumbers same-named
     shaders of other sessions);
  4. the portable form (tlg-probe --portable-recipes) and the new list with the old list's
     pipeline keys (make_list.py --keys-from);
  5. the game-code check of the new list (check_no_code.py): it must find nothing.

The logs hold no shader code: every program comes from GAME_DIR, the player's own game files.
A logged shader whose program is not in the game files (made by the game while it runs, or of
another game version) is named and left out. The current list is not changed; NEW_LIST must
not exist. Recipe folders hold programs and are never shipped: they stay in a scratch folder
that is removed at the end (--keep DIR keeps them there instead).

PROBE is the tlg-probe of this build; PROFILE is its profile.bin
(user/cache/<serial>-<build>/profile.bin, written by any start of that build). On Windows the
probe's DLLs must be found (beside it, or e.g. the MSYS2 clang64 bin folder on PATH).

Usage: add_from_logs.py --probe PROBE --profile PROFILE LIST GAME_DIR NEW_LIST LOG [LOG ...]
Then: replay a scene from the logs on an empty cache with NEW_LIST and check that the count of
"not in list" lines drops. Exit 0 when the new list is written and checked; 3 when the logs name
no shader that can be added (no new list)."""
import argparse
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from list_recipes import read_list  # noqa: E402


def step(title, cmd, ok=(0,), env=None):
    print(f'== {title}', flush=True)
    run = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    out = (run.stdout + run.stderr).rstrip()
    if out:
        print('   ' + out.replace('\n', '\n   '), flush=True)
    if run.returncode not in ok:
        sys.exit(f'{title}: failed (exit {run.returncode})')
    return run.returncode


def tool(name):
    return [sys.executable, '-I', HERE / name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probe', required=True, type=pathlib.Path)
    ap.add_argument('--profile', required=True, type=pathlib.Path)
    ap.add_argument('--keep', type=pathlib.Path, help='keep the recipe folders here (must not exist or be empty)')
    ap.add_argument('list', type=pathlib.Path)
    ap.add_argument('game', type=pathlib.Path)
    ap.add_argument('new', type=pathlib.Path)
    ap.add_argument('logs', type=pathlib.Path, nargs='+')
    a = ap.parse_args()
    for path, what in ((a.probe, 'probe'), (a.profile, 'profile'), (a.list, 'list')):
        if not path.is_file():
            sys.exit(f'{what} {path}: not a file')
    if not a.game.is_dir():
        sys.exit(f'game folder {a.game}: not a folder')
    for log in a.logs:
        if not log.is_file():
            sys.exit(f'log {log}: not a file')
    if a.new.exists():
        sys.exit(f'{a.new} exists: give a new file name (the current list is never overwritten)')
    try:
        list_id, _sources, old_entries, old_keys = read_list(a.list)
    except (ValueError, struct.error, zlib.error) as e:
        sys.exit(f'{a.list}: {e}')

    if a.keep:
        if a.keep.exists() and any(a.keep.iterdir()):
            sys.exit(f'{a.keep} exists and is not empty')
        a.keep.mkdir(parents=True, exist_ok=True)
        work, cleanup = a.keep, None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix='tlg-addlogs-')
        work = pathlib.Path(cleanup.name)
    try:
        from_list, logged, merged = work / 'from-list', work / 'logged', work / 'merged'
        portable = work / 'merged.portable'
        step('1. current list -> recipes', tool('list_recipes.py') + [a.list, a.game, from_list])
        # Exit 1 = some lines not usable (named in the output): the rest is still added.
        step('2. log lines -> recipes', tool('log_recipes.py') + [a.game, logged] + a.logs, ok=(0, 1))
        folders = sorted(p for p in logged.iterdir() if p.is_dir()) if logged.is_dir() else []
        if not folders:
            print('The logs name no shader or pipeline key that can be added: no new list.')
            sys.exit(3)
        for folder in folders:
            if any(folder.glob('*.pkey')):
                step(f'2b. keys of {folder.name}', [a.probe, '--keys-from-portable', folder])
        step('3. merge (list first)', tool('merge_recipes.py') +
             ['--probe', a.probe, '--profile', a.profile, merged, from_list] + folders)
        step('4a. portable form', [a.probe, '--portable-recipes', merged, portable])
        step('4b. new list', tool('make_list.py') +
             [merged, a.game, a.new, '--portable', portable, '--keys-from', a.list])
        step('5. game-code check', tool('check_no_code.py') + [a.new, a.game])
    finally:
        if cleanup:
            cleanup.cleanup()

    new_id, _s, new_entries, new_keys = read_list(a.new)
    if new_id != list_id:
        sys.exit(f'{a.new}: portable id {new_id:#018x} differs from the list {list_id:#018x}')
    def programs(entries):
        return {e['hash'] for e in entries}
    print(f'Done: {a.new}')
    print(f'  shaders  {len(old_entries)} -> {len(new_entries)} (+{len(new_entries) - len(old_entries)})')
    print(f'  programs {len(programs(old_entries))} -> {len(programs(new_entries))}')
    print(f'  pipeline keys {len(old_keys)} -> {len(new_keys)}')
    print(f'  size {a.list.stat().st_size} -> {a.new.stat().st_size} bytes; game-code check: nothing found')
    if a.keep:
        print(f'  recipe folders kept in {a.keep}: they hold game code, never ship or share them')


if __name__ == '__main__':
    main()
