# SPDX-License-Identifier: GPL-2.0-or-later
"""The models archive with the foxtail toy (launcher/foxtail.py, on top of the ICO horns) is built on
the player's PC: every machine must get the same bytes, and the generator must not read any file."""
import builtins
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from paths import ROOT
sys.path.insert(0, str(ROOT / 'launcher'))
import foxtail  # noqa: E402
import ico_horns  # noqa: E402

APPROVED = '91402b843f189cba7ddaf9af6d7b98896ed0972cb63e38247e75d10df37c50fb'


class Foxtail(unittest.TestCase):
    def test_same_bytes_everywhere(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trico_other.psarc'
            self.assertEqual(foxtail.write(str(path)), APPROVED)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), APPROVED)

    def test_horns_unchanged(self):
        horns, files = ico_horns.build(), foxtail.build()
        for name, data in horns.items():
            self.assertEqual(files[name], data, name)

    def test_reads_no_file(self):
        real, opened = builtins.open, []

        def watched(file, mode='r', *args, **kwargs):
            opened.append((str(file), mode))
            return real(file, mode, *args, **kwargs)

        builtins.open = watched
        try:
            files = foxtail.build()
        finally:
            builtins.open = real
        self.assertEqual(opened, [])
        self.assertIn('game/assets/props/level/foxtail_toy/foxtail_toy.bod', files)


if __name__ == '__main__':
    unittest.main()
