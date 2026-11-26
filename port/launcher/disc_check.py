# SPDX-License-Identifier: GPL-2.0-or-later
"""Disc check: is this copy of the game technically the same as the discs the port was tested with?

Measures checksums only, no game content:
  * the loaded parts of eboot.bin and of the two system modules (a SELF dump is unwrapped first, so
    the dump tool does not matter);
  * the file table of every trico_*.psarc (names, sizes and packed block sizes of all its files);
  * the real contents of the shaders and Lua scripts in the boot archives.
The port ships no checksums of the game: the player's values are sent in and compared by the
developers with the values of the tested US (CUSA03627) and EU (CUSA03745) 1.03 dumps (a reference
file kept outside the port). Those two are identical in everything but their language archives, so
a disc whose game program, shared archives, shaders and scripts all match is the same game with
other languages.

The "Report a problem" zip carries the text as disc-check.txt, which is how a player with a disc
nobody tested sends it in (#105). By hand (--reference compares with a reference file):

    python disc_check.py [--reference FILE] <game folder or .zar>
"""
import hashlib
import json
from pathlib import Path
import struct
import sys

import game_image

DIGEST = 16                                        # hex digits kept of each SHA-256
LOADED_SEGMENTS = (1, 0x61000010)                  # PT_LOAD, PT_SCE_RELRO: what the loader maps
BINARIES = ('eboot.bin', 'sce_module/libc.prx', 'sce_module/libSceFios2.prx')
# The archives every region has; the others hold one language each.
SHARED_ARCHIVES = ('trico_boot.psarc', 'trico_bootp.psarc', 'trico_001.psarc', 'trico_001p.psarc',
                   'trico_002.psarc', 'trico_002p.psarc', 'trico_003.psarc', 'trico_003p.psarc')
# Shaders and scripts live in the boot archive and its update; the update's copy of a file wins.
CONTENT_ARCHIVES = ('trico_boot.psarc', 'trico_bootp.psarc')
CONTENT_SETS = {'shaders': lambda name: name.startswith('ps4_shader_cache/'),
                'scripts': lambda name: name.endswith('.lua')}
OWN_ARCHIVES = ('trico_other.psarc',)              # built by the port (horns, toy), not on a disc



def digest(data):
    return hashlib.sha256(data).hexdigest()[:DIGEST]


def binary_parts(data, name):
    """{'<name> <address> <rwx>': checksum} of the segments the loader maps, for a plain ELF or a
    SELF dump; {} when the file is neither."""
    sys.path.insert(0, str(game_image.PORT_DIR / 'scripts'))
    from prepare import parse_self
    try:
        elf, _, headers, _, _ = parse_self(data)
    except (ValueError, struct.error, IndexError):
        return {}
    parts = {}
    for p in headers:
        if p['type'] in LOADED_SEGMENTS and p['filesz']:
            access = ''.join(letter if p['flags'] & bit else '-' for letter, bit in (('r', 4), ('w', 2), ('x', 1)))
            parts[f'{name} {p["vaddr"]:#x} {access}'] = digest(bytes(elf[p['offset']:p['offset'] + p['filesz']]))
    return parts


def archive_table(path):
    """'<file size>/<files>/<checksum of the file table>' of a PSARC; None when it is not one."""
    with path.open('rb') as file:
        head = file.read(32)
        if len(head) < 32 or head[:4] != b'PSAR':
            return None
        table_size, _, count = struct.unpack('>III', head[12:24])
        table = head + file.read(table_size - 32)
    return f'{path.stat().st_size}/{count - 1}/{digest(table)}'


def content_sets(archives):
    """{'shaders': '<files>/<checksum>', 'scripts': ...} over the contents of the files each set
    picks from the archives (in order: a later archive's file replaces an earlier one's)."""
    import soundtrack
    found = {key: {} for key in CONTENT_SETS}
    for path in archives:
        archive = soundtrack.Psarc(path)
        try:
            for name in archive.names:
                for key, picks in CONTENT_SETS.items():
                    if picks(name):
                        found[key][name] = digest(archive.read(name))
        finally:
            archive.close()
    return {key: f'{len(files)}/{digest("".join(f"{n} {h}\n" for n, h in sorted(files.items())).encode())}'
            for key, files in found.items() if files}


def measure(game_dir):
    """Every checksum of the game folder or .zar as {key: value}; what cannot be read is left out."""
    values = {}
    files = Path(game_image.safe_files_dir(game_dir))
    for name in BINARIES:
        try:
            values.update(binary_parts((files / name).read_bytes(), Path(name).name))
        except OSError:
            pass
    try:
        archives = {path.name.lower(): path for path in game_image.root_files(game_dir, 'trico_*.psarc')}
    except (OSError, RuntimeError, ValueError):
        archives = {}
    for name, path in archives.items():
        try:
            table = archive_table(path)
        except (OSError, RuntimeError, ValueError, struct.error):
            table = None
        if table:
            values[name] = table
    try:
        values.update(content_sets([archives[name] for name in CONTENT_ARCHIVES if name in archives]))
    except Exception:   # a damaged archive: the tables above still tell what differs
        pass
    return values


def disc(game_dir):
    """(title id, app version) of sce_sys/param.sfo; ('', '') without it."""
    from savemanager import read_sfo
    try:
        sfo = read_sfo((Path(game_image.safe_files_dir(game_dir)) / 'sce_sys' / 'param.sfo').read_bytes())
    except OSError:
        return '', ''
    return str(sfo.get('TITLE_ID', '')), str(sfo.get('APP_VER', ''))


def load_reference(path):
    """(what it was measured on, {key: value}) of a reference file."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return data['tested'], data['values']


def groups(values, reference=None):
    """[(group, [(key, value, state)])] with state 'same', 'DIFFERENT', 'MISSING' (in the
    reference group, not on this disc) or '' (not compared: without a reference, or no tested
    disc has it)."""
    reference = reference or {}
    binaries = tuple(Path(name).name + ' ' for name in BINARIES)
    belongs = {
        'Game program': lambda key: key.startswith(binaries),
        'Shared archives': lambda key: key in SHARED_ARCHIVES,
        'Shaders and scripts': lambda key: key in CONTENT_SETS,
        'Language archives': lambda key: key.endswith('.psarc') and key not in SHARED_ARCHIVES + OWN_ARCHIVES,
    }
    out = []
    for group, picks in belongs.items():
        rows = [(key, value, ('same' if reference[key] == value else 'DIFFERENT') if key in reference else '')
                for key, value in sorted(values.items()) if picks(key)]
        if reference and group != 'Language archives':   # a disc has only its own languages
            rows += [(key, '', 'MISSING') for key in sorted(reference) if picks(key) and key not in values]
        out.append((group, rows))
    return out


def equal(found):
    """True when every compared group but the language archives is complete and the same."""
    return all(rows and all(state == 'same' for _, _, state in rows)
               for group, rows in found if group != 'Language archives')


def report(game_dir, values=None, reference=None):
    """The text of disc-check.txt. It names files of the game, never the folder they are in.
    Without `reference` ((tested, values) of load_reference) it lists the values only."""
    values = measure(game_dir) if values is None else values
    title, version = disc(game_dir)
    lines = ['The Last Guardian disc check', f'Disc: {title or "unknown"}, update {version or "unknown"}']
    if not reference:
        lines += ['Send this text to the developers: they compare it with the tested discs.', '']
        for group, rows in groups(values):
            lines.append(f'[{group}]')
            lines += [f'{key:<34} {value}' for key, value, _ in rows] or ['NOT FOUND']
            lines.append('')
        return '\n'.join(lines)
    tested, reference = reference
    found = groups(values, reference)
    lines += [f'Compared with: {tested}', '']
    for group, rows in found:
        if group == 'Language archives':
            verdict = f'{len(rows)} on this disc (they differ per region)'
        elif not rows:
            verdict = 'NOT FOUND'
        else:
            same = sum(state == 'same' for _, _, state in rows)
            verdict = 'same' if same == len(rows) else f'DIFFERENT ({same} of {len(rows)} the same)'
        lines.append(f'{group}: {verdict}')
    lines += ['', 'Result: ' + ('technically the same game as the tested discs.' if equal(found) else
                                'NOT the same as the tested discs, see the lines marked below.'), '']
    for group, rows in found:
        lines.append(f'[{group}]')
        lines += [f'{key:<34} {value:<40} {state}'.rstrip() for key, value, state in rows]
        lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    reference = None
    if len(argv) == 3 and argv[0] == '--reference':
        reference, argv = load_reference(argv[1]), argv[2:]
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    print(report(argv[0], reference=reference))
    return 0


if __name__ == '__main__':
    sys.exit(main())
