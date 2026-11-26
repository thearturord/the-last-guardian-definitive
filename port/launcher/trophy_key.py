"""The PS4 trophy key check for the launcher (issue #60).

The trophy names in sce_sys/trophy/trophy00.trp are encrypted: each ESFM file is a 16-byte IV and
AES-128-CBC data, keyed with the NP communication id (from sce_sys/npbind.dat, zero-padded to 16
bytes) AES-encrypted with the player's PS4 trophy key. The same scheme as the game side
(gpu/shim/tlg_trophy.cpp). The key is never shipped; this only tells the player whether theirs
opens the game's trophy file. Checking the key at once is an idea from @SulphurS16's TLG-Port.
Plain Python AES (FIPS-197), so the launcher needs no extra package.
"""
from pathlib import Path
import re
import string


def _xtime(a):
    a <<= 1
    return (a ^ 0x11B) if a & 0x100 else a


def _mul(a, b):
    out = 0
    while b:
        if b & 1:
            out ^= a
        a, b = _xtime(a), b >> 1
    return out


def _make_sbox():
    sbox, inv = [0] * 256, [0] * 256
    for x in range(256):
        y = 0 if x == 0 else next(c for c in range(1, 256) if _mul(x, c) == 1)  # the inverse in GF(2^8)
        s = y
        for shift in range(1, 5):
            s ^= ((y << shift) | (y >> (8 - shift))) & 0xFF
        s ^= 0x63
        sbox[x], inv[s] = s, x
    return sbox, inv


SBOX, INV_SBOX = _make_sbox()


def _expand(key):
    words = [list(key[i:i + 4]) for i in range(0, 16, 4)]
    rcon = 1
    for i in range(4, 44):
        t = list(words[i - 1])
        if i % 4 == 0:
            t = [SBOX[b] for b in t[1:] + t[:1]]
            t[0] ^= rcon
            rcon = _xtime(rcon) & 0xFF
        words.append([a ^ b for a, b in zip(words[i - 4], t)])
    return [sum(words[4 * r:4 * r + 4], []) for r in range(11)]


def _add(state, round_key):
    return [a ^ b for a, b in zip(state, round_key)]


def _shift_rows(s, inverse=False):
    out = [0] * 16
    for col in range(4):
        for row in range(4):
            src = (col + row) % 4 if not inverse else (col - row) % 4
            out[4 * col + row] = s[4 * src + row]
    return out


def _mix(s, m):
    out = []
    for col in range(4):
        c = s[4 * col:4 * col + 4]
        for row in range(4):
            v = 0
            for k in range(4):
                v ^= _mul(m[row][k], c[k])
            out.append(v)
    return out


_MIX = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]
_INV_MIX = [[14, 11, 13, 9], [9, 14, 11, 13], [13, 9, 14, 11], [11, 13, 9, 14]]


def aes_encrypt_block(key, block):
    keys = _expand(key)
    s = _add(list(block), keys[0])
    for r in range(1, 11):
        s = _shift_rows([SBOX[b] for b in s])
        if r < 10:
            s = _mix(s, _MIX)
        s = _add(s, keys[r])
    return bytes(s)


def aes_decrypt_block(key, block, keys=None):
    keys = keys or _expand(key)
    s = _add(list(block), keys[10])
    for r in range(9, -1, -1):
        s = [INV_SBOX[b] for b in _shift_rows(s, inverse=True)]
        s = _add(s, keys[r])
        if r > 0:
            s = _mix(s, _INV_MIX)
    return bytes(s)


def parse_key(text):
    """16 bytes from 32 hex digits (spaces allowed), or None."""
    digits = ''.join(text.split())
    if len(digits) != 32 or any(c not in string.hexdigits for c in digits):
        return None
    return bytes.fromhex(digits)


def trp_files(data):
    """{name: bytes} of a trophy00.trp (big-endian, magic DC A2 4D 00)."""
    if len(data) < 0x60 or data[:4] != b'\xdc\xa2\x4d\x00':
        return {}
    count, size = int.from_bytes(data[0x10:0x14], 'big'), int.from_bytes(data[0x14:0x18], 'big')
    out = {}
    for i in range(min(count, 4096)):
        at = 0x60 + i * size
        if size < 0x30 or at + 0x30 > len(data):
            break
        name = data[at:at + 32].split(b'\0', 1)[0].decode('ascii', 'replace')
        offset, length = int.from_bytes(data[at + 32:at + 40], 'big'), int.from_bytes(data[at + 40:at + 48], 'big')
        if offset + length <= len(data):
            out[name] = data[offset:offset + length]
    return out


def decrypt_esfm(key, np_id, esfm):
    """The XML text of one ESFM, or None when the key does not open it."""
    if len(esfm) < 32 or (len(esfm) - 16) % 16:
        return None
    file_key = aes_encrypt_block(key, np_id.encode('ascii').ljust(16, b'\0'))
    keys = _expand(file_key)
    out, prev = bytearray(), esfm[:16]
    for at in range(16, len(esfm), 16):
        block = esfm[at:at + 16]
        out += bytes(a ^ b for a, b in zip(aes_decrypt_block(file_key, block, keys), prev))
        prev = block
    text = bytes(out).split(b'\0', 1)[0].decode('utf-8', 'replace')
    return text if '<trophy' in text else None


def np_id_of(npbind):
    match = re.search(rb'NPWR\d{5}_\d{2}', npbind)
    return match.group().decode('ascii') if match else ''


def check(sce_sys, text):
    """Whether the key text opens the game's trophy file: (status, names). Status: 'cleared'
    (empty), 'format' (not 32 hex digits), 'nofile' (no trophy file to check against),
    'wrong' (does not open it), 'ok' (names = how many trophy names it holds)."""
    if not ''.join(text.split()):
        return 'cleared', 0
    key = parse_key(text)
    if key is None:
        return 'format', 0
    try:
        files = trp_files((Path(sce_sys) / 'trophy' / 'trophy00.trp').read_bytes())
        np_id = np_id_of((Path(sce_sys) / 'npbind.dat').read_bytes())
    except OSError:
        return 'nofile', 0
    esfm = files.get('TROP.ESFM') or files.get('TROPCONF.ESFM') or next(
        (v for k, v in sorted(files.items()) if k.endswith('.ESFM')), None)
    if esfm is None or not np_id:
        return 'nofile', 0
    xml = decrypt_esfm(key, np_id, esfm)
    if xml is None:
        return 'wrong', 0
    return 'ok', len(re.findall(r'<trophy\s', xml))
