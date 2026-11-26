"""launcher/disc_check.py: a disc is 'the same game' only when every compared part matches."""
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from paths import ROOT
sys.path.insert(0, str(ROOT / 'launcher'))
import disc_check  # noqa: E402
import report  # noqa: E402


# Made-up values in the layout of the real reference (which stays outside the port).
REFERENCE = {**{f'{name} {address} {access}': '0' * 16
                for name in ('eboot.bin', 'libc.prx', 'libSceFios2.prx')
                for address, access in (('0x0', 'r-x'), ('0x1000', 'r--'), ('0x2000', 'rw-'))},
             **{name: '1/1/' + '0' * 16 for name in disc_check.SHARED_ARCHIVES},
             'scripts': '1/' + '0' * 16, 'shaders': '1/' + '0' * 16}
TESTED = ('the test discs', REFERENCE)


def same_values():
    """What a disc equal to the tested ones measures, plus one language archive."""
    return dict(REFERENCE, **{'trico_kor.psarc': '1/1/00'})


def groups(values):
    return disc_check.groups(values, REFERENCE)


class DiscCheckTest(unittest.TestCase):
    def test_reference_covers_every_group(self):
        found = dict(groups(REFERENCE))
        for group in ('Game program', 'Shared archives', 'Shaders and scripts'):
            self.assertTrue(found[group], group)

    def test_the_port_ships_no_checksums(self):
        self.assertFalse(hasattr(disc_check, 'REFERENCE'))
        text = disc_check.report('', same_values())
        self.assertIn('Send this text', text)
        self.assertNotIn('same', text.replace('the same', ''))
        self.assertIn('0' * 16, text)

    def test_equal_disc(self):
        self.assertTrue(disc_check.equal(groups(same_values())))

    def test_other_program_is_not_equal(self):
        values = same_values()
        values['eboot.bin 0x0 r-x'] = 'ffffffffffffffff'
        found = groups(values)
        self.assertFalse(disc_check.equal(found))
        self.assertIn(('eboot.bin 0x0 r-x', 'ffffffffffffffff', 'DIFFERENT'), dict(found)['Game program'])

    def test_other_shaders_are_not_equal(self):
        values = same_values()
        values['shaders'] = '1/ffff'
        self.assertFalse(disc_check.equal(groups(values)))

    def test_missing_archive_is_not_equal(self):
        values = same_values()
        del values['trico_002.psarc']
        found = groups(values)
        self.assertFalse(disc_check.equal(found))
        self.assertIn(('trico_002.psarc', '', 'MISSING'), dict(found)['Shared archives'])

    def test_language_archives_are_not_compared(self):
        found = dict(groups(same_values()))
        self.assertEqual(found['Language archives'], [('trico_kor.psarc', '1/1/00', '')])

    def test_port_archive_is_left_out(self):
        values = dict(same_values(), **{'trico_other.psarc': '1/1/00'})
        self.assertTrue(disc_check.equal(groups(values)))
        self.assertNotIn('trico_other.psarc', disc_check.report('', values, TESTED))

    def test_archive_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'trico_x.psarc')
            head = struct.pack('>4sHH4sIIIII', b'PSAR', 1, 4, b'zlib', 40, 30, 3, 65536, 0)
            path.write_bytes(head + b'12345678' + b'rest')
            self.assertEqual(disc_check.archive_table(path).split('/')[:2], ['44', '2'])
            path.write_bytes(b'not an archive')
            self.assertIsNone(disc_check.archive_table(path))

    def test_report_names_no_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = disc_check.report(tmp, same_values(), TESTED)
        self.assertIn('technically the same game', text)
        self.assertNotIn(tmp, text)

    def test_zip_carries_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'r.zip')
            report.write_zip(path, 'report', None, '', 'disc text')
            with zipfile.ZipFile(path) as z:
                self.assertEqual(z.read('disc-check.txt'), b'disc text')
            report.write_zip(path, 'report', None, '')
            with zipfile.ZipFile(path) as z:
                self.assertNotIn('disc-check.txt', z.namelist())


if __name__ == '__main__':
    unittest.main()
