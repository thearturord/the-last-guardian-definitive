# SPDX-License-Identifier: GPL-2.0-or-later
"""The new engine's shader store (TLG_VKB_SHADER_PREBUILD) names its folder after every setting the
translator reads, so a start with another value never takes a stage translated with the old one.
This test reads the translator's sources (the engine's copy of the shader recompiler and the
descriptor headers it uses) and fails when they read a setting the store's list does not name."""
import re
import unittest

from paths import ROOT

GPU = ROOT / 'gpu' / 'native'
STORE = GPU / 'native_vkb_shader_store.cpp'
SOURCES = [GPU / 'shadps4' / 'shader_recompiler', GPU / 'shadps4' / 'video_core' / 'amdgpu', GPU / 'ps4_data_layout']
ENV = re.compile(r'(?:getenv|Flag)\(\s*"([A-Z0-9_]+)"')
SETTING = re.compile(r'EmulatorSettings(?:Impl)?(?:\.|::|->)\s*([A-Za-z0-9_]+)\s*\(')


def settings_read():
    found = {}
    for folder in SOURCES:
        for path in sorted(folder.rglob('*')):
            if path.suffix not in ('.cpp', '.h', '.inc'):
                continue
            text = path.read_text(errors='replace')
            for name in ENV.findall(text) + SETTING.findall(text):
                found.setdefault(name, path.relative_to(ROOT))
    return found


class StoreSettingsTest(unittest.TestCase):
    def test_every_translator_setting_is_in_the_store_id(self):
        store = STORE.read_text()
        listed = set()
        for array in ('TranslatorEnv', 'TranslatorEmulatorSettings', 'TranslatorWithoutEffect'):
            body = re.search(r'constexpr const char\* ' + array + r'\[\] = \{([^}]*)\}', store)
            self.assertIsNotNone(body, array)
            listed |= set(re.findall(r'"([A-Za-z0-9_]+)"', body.group(1)))
        # the EmulatorSettings values must also be read where the id is made
        made = store[store.index('std::string TranslatorSettings() {'):store.index('void OpenFolder(')]
        for name in re.findall(r'"([A-Za-z0-9_]+)"', re.search(r'TranslatorEmulatorSettings\[\] = \{([^}]*)\}', store).group(1)):
            self.assertIn('EmulatorSettings.' + name + '()', made, name)
        missing = [f'{name} ({where})' for name, where in settings_read().items() if name not in listed]
        self.assertEqual(missing, [], 'the translator reads settings the shader store does not name '
                         '(add them to TranslatorEnv / TranslatorSettings in native_vkb_shader_store.cpp, '
                         'or to its list of settings without effect on the result)')

    def test_the_scan_finds_the_known_ones(self):
        found = settings_read()
        for name in ('TLG_CONST_OFFSET_FLOATRULE', 'IsDirectMemoryAccessEnabled'):
            self.assertIn(name, found)


if __name__ == '__main__':
    unittest.main()
