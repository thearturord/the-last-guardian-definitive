// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: trophies (issue #60). The game reports each unlock to the runtime (sceNpTrophyUnlockTrophy);
// this keeps them in user/trophies.txt, and reads the list (icons, names) from the player's own
// sce_sys/trophy/trophy00.trp. Names and descriptions are encrypted in that file: they show only
// when the player enters the PS4 trophy key (tlg.ini trophy_key, 32 hex digits); without it the
// trophies are numbered. Idea from SulphurS16's TLG-Port (achievements); own code.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace TlgTrophy {

struct Entry {
    int id = 0;
    char grade = 0;          // 'P' platinum, 'G' gold, 'S' silver, 'B' bronze, 0 = unknown (no key)
    bool hidden = false;     // a secret trophy: its text shows only once unlocked
    std::string name;        // empty without the key
    std::string detail;
    std::vector<unsigned char> icon_png; // TROPnnn.PNG from the trophy file
    int64_t unlocked_at = 0; // Unix time, 0 = locked
};

// Key and NP communication id -> the AES key of the ESFM files (exposed for the unit test).
bool DecryptEsfm(const unsigned char key[16], const std::string& np_comm_id, const unsigned char* data,
                 size_t size, std::string& xml);
// FIPS-197 AES-128 single block, exposed for the unit test.
void AesEncryptBlock(const unsigned char key[16], const unsigned char in[16], unsigned char out[16]);
void AesDecryptBlock(const unsigned char key[16], const unsigned char in[16], unsigned char out[16]);
bool ParseKey(const std::string& hex, unsigned char out[16]);

// Runtime side (any thread). trp/npbind: the game's files; store: the unlock file to use.
void Setup(std::vector<unsigned char> trp, std::vector<unsigned char> npbind, std::string store, int language);
// Returns true for a new unlock (false: already unlocked or an unknown id).
bool Unlock(int id);
void UnlockFlags(uint32_t flags[4]);
int Count();
int NextLocked(); // -1 = all unlocked

// UI side: a copy of the list, and whether the key worked (0 no key, 1 works, -1 wrong key).
std::vector<Entry> List(int* key_state = nullptr, std::string* title = nullptr);
// List() leaves icon_png empty; the icon of one trophy:
std::vector<unsigned char> IconPng(int id);
// After a key change in the menu: read the names again.
void Reload();
// Pop-ups waiting to be shown (taken by the overlay).
bool TakeUnlockToast(Entry& out);
bool HasToast();
std::string DisplayName(const Entry& e); // the name, or "Trophy n"
// The unlock chime (tlg_trophy_sound.cpp) on the system's default sound device. Any thread;
// returns at once.
void PlayChime();

} // namespace TlgTrophy
