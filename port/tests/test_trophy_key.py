# SPDX-License-Identifier: GPL-2.0-or-later
"""The launcher's trophy key check (launcher/trophy_key.py, issue #60): AES against the FIPS-197
vector, and a made-up trophy file (a made-up key, not the console's) checked with right, wrong,
badly typed and empty keys, and without game files."""
import sys
import tempfile
import unittest
from pathlib import Path

from paths import ROOT
sys.path.insert(0, str(ROOT / 'launcher'))
import trophy_key  # noqa: E402

KEY = bytes(range(16))
NP_ID = 'NPWR00000_00'
XML = ('<?xml version="1.0"?><trophyconf><title-name>Test</title-name>'
       '<trophy id="000" ttype="P"><name>One</name></trophy>'
       '<trophy id="001" ttype="B"><name>Two</name></trophy></trophyconf>')


def make_esfm(key, np_id, xml):
    file_key = trophy_key.aes_encrypt_block(key, np_id.encode().ljust(16, b'\0'))
    data = xml.encode()
    data += b'\0' * (-len(data) % 16)
    prev = out = bytes(range(0x30, 0x40))
    for at in range(0, len(data), 16):
        prev = trophy_key.aes_encrypt_block(file_key, bytes(a ^ b for a, b in zip(data[at:at + 16], prev)))
        out += prev
    return out


def make_trp(files):
    entry = 0x40
    head = bytearray(0x60 + len(files) * entry)
    head[0:4] = b'\xdc\xa2\x4d\x00'
    head[0x10:0x14] = len(files).to_bytes(4, 'big')
    head[0x14:0x18] = entry.to_bytes(4, 'big')
    body = bytearray()
    for i, (name, data) in enumerate(files):
        at = 0x60 + i * entry
        head[at:at + len(name)] = name.encode()
        head[at + 32:at + 40] = (len(head) + len(body)).to_bytes(8, 'big')
        head[at + 40:at + 48] = len(data).to_bytes(8, 'big')
        body += data
    return bytes(head + body)


class TrophyKeyTests(unittest.TestCase):
    def test_aes_fips197(self):
        plain = bytes(i * 0x11 for i in range(16))
        out = trophy_key.aes_encrypt_block(KEY, plain)
        self.assertEqual(out.hex(), '69c4e0d86a7b0430d8cdb78070b4c55a')
        self.assertEqual(trophy_key.aes_decrypt_block(KEY, out), plain)

    def test_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            sce_sys = Path(tmp)
            self.assertEqual(trophy_key.check(sce_sys, KEY.hex()), ('nofile', 0))
            (sce_sys / 'trophy').mkdir()
            (sce_sys / 'trophy' / 'trophy00.trp').write_bytes(make_trp(
                [('TROP000.PNG', b'png'), ('TROP.ESFM', make_esfm(KEY, NP_ID, XML))]))
            (sce_sys / 'npbind.dat').write_bytes(b'\0' * 20 + NP_ID.encode() + b'\0' * 20)
            self.assertEqual(trophy_key.check(sce_sys, KEY.hex()), ('ok', 2))
            self.assertEqual(trophy_key.check(sce_sys, ' 00010203 04050607\n08090A0B 0C0D0E0F '), ('ok', 2))
            self.assertEqual(trophy_key.check(sce_sys, 'ff' * 16), ('wrong', 0))
            self.assertEqual(trophy_key.check(sce_sys, '0102'), ('format', 0))
            self.assertEqual(trophy_key.check(sce_sys, 'g' * 32), ('format', 0))
            self.assertEqual(trophy_key.check(sce_sys, '  '), ('cleared', 0))


if __name__ == '__main__':
    unittest.main()
