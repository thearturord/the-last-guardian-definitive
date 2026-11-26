// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: trophies (issue #60): AES against the FIPS-197 vector, the key parser, and a made-up trophy
// file (two icons + one encrypted ESFM) through Setup, unlock, a restart and a wrong key.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "tlg_settings.h"
#include "tlg_trophy.h"

namespace {

void PutBe32(std::vector<unsigned char>& v, size_t at, uint32_t x) {
    for (int i = 0; i < 4; ++i) v[at + i] = (unsigned char)(x >> (24 - 8 * i));
}
void PutBe64(std::vector<unsigned char>& v, size_t at, uint64_t x) {
    PutBe32(v, at, uint32_t(x >> 32));
    PutBe32(v, at + 4, uint32_t(x));
}

// ESFM as the PS4 writes it: IV + AES-CBC(file key) of the zero-padded XML.
std::vector<unsigned char> MakeEsfm(const unsigned char key[16], const std::string& np, const std::string& xml) {
    unsigned char np16[16] = {}, file_key[16];
    std::memcpy(np16, np.data(), np.size());
    TlgTrophy::AesEncryptBlock(key, np16, file_key);
    std::vector<unsigned char> out(16);
    for (int i = 0; i < 16; ++i) out[i] = (unsigned char)(0x30 + i);
    std::string padded = xml;
    padded.resize((xml.size() + 15) / 16 * 16, '\0');
    unsigned char prev[16];
    std::memcpy(prev, out.data(), 16);
    for (size_t at = 0; at < padded.size(); at += 16) {
        unsigned char block[16], enc[16];
        for (int i = 0; i < 16; ++i) block[i] = (unsigned char)(padded[at + i] ^ prev[i]);
        TlgTrophy::AesEncryptBlock(file_key, block, enc);
        out.insert(out.end(), enc, enc + 16);
        std::memcpy(prev, enc, 16);
    }
    return out;
}

std::vector<unsigned char> MakeTrp(const std::vector<std::pair<std::string, std::vector<unsigned char>>>& files) {
    const size_t entry = 0x40, head = 0x60 + files.size() * entry;
    std::vector<unsigned char> trp(head, 0);
    PutBe32(trp, 0, 0xDCA24D00u);
    PutBe32(trp, 0x10, uint32_t(files.size()));
    PutBe32(trp, 0x14, uint32_t(entry));
    for (size_t i = 0; i < files.size(); ++i) {
        const size_t at = 0x60 + i * entry;
        std::memcpy(&trp[at], files[i].first.c_str(), files[i].first.size());
        PutBe64(trp, at + 32, trp.size());
        PutBe64(trp, at + 40, files[i].second.size());
        trp.insert(trp.end(), files[i].second.begin(), files[i].second.end());
    }
    return trp;
}

} // namespace

int main() {
    // FIPS-197 appendix C.1.
    unsigned char key[16], plain[16], out[16], back[16];
    for (int i = 0; i < 16; ++i) { key[i] = (unsigned char)i; plain[i] = (unsigned char)(i * 0x11); }
    static const unsigned char expect[16] = {0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30,
                                             0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a};
    TlgTrophy::AesEncryptBlock(key, plain, out);
    assert(std::memcmp(out, expect, 16) == 0);
    TlgTrophy::AesDecryptBlock(key, out, back);
    assert(std::memcmp(back, plain, 16) == 0);

    unsigned char parsed[16];
    assert(TlgTrophy::ParseKey("000102030405060708090a0b0c0d0e0f", parsed) && std::memcmp(parsed, key, 16) == 0);
    assert(TlgTrophy::ParseKey(" 00010203 04050607\t08090A0B 0C0D0E0F\n", parsed) && std::memcmp(parsed, key, 16) == 0);
    assert(!TlgTrophy::ParseKey("000102030405060708090a0b0c0d0e", parsed));   // too short
    assert(!TlgTrophy::ParseKey("000102030405060708090a0b0c0d0e0g", parsed)); // not hex

    // A made-up key and trophy file (not the game's).
    const std::string np = "NPWR00000_00";
    const std::string xml =
        "<?xml version=\"1.0\"?><trophyconf><title-name>Test &amp; Game</title-name>"
        "<trophy id=\"000\" hidden=\"no\" ttype=\"P\"><name>All of them</name><detail>Get all</detail></trophy>"
        "<trophy id=\"001\" hidden=\"yes\" ttype=\"B\"><name>Secret</name><detail>A &lt;hidden&gt; one</detail></trophy>"
        "</trophyconf>";
    const std::vector<unsigned char> esfm = MakeEsfm(key, np, xml);
    std::string text;
    assert(TlgTrophy::DecryptEsfm(key, np, esfm.data(), esfm.size(), text) && text == xml);
    unsigned char wrong[16];
    std::memcpy(wrong, key, 16);
    wrong[0] ^= 1;
    assert(!TlgTrophy::DecryptEsfm(wrong, np, esfm.data(), esfm.size(), text));

    const std::vector<unsigned char> png0 = {0x89, 'P', 'N', 'G', 0}, png1 = {0x89, 'P', 'N', 'G', 1};
    const std::vector<unsigned char> trp = MakeTrp({{"TROPCONF.ESFM", {}}, {"TROP000.PNG", png0},
                                                    {"TROP001.PNG", png1}, {"TROP_01.ESFM", esfm}});
    std::vector<unsigned char> npbind(64, 0);
    std::memcpy(&npbind[20], np.data(), np.size());
    const std::string store = "trophy_test_store.txt";
    std::remove(store.c_str());

    // No key: numbered trophies, icons there.
    TlgSettings::Get().trophy_key.clear();
    TlgTrophy::Setup(trp, npbind, store, 1);
    int state = 9;
    std::vector<TlgTrophy::Entry> list = TlgTrophy::List(&state);
    assert(state == 0 && list.size() == 2 && list[0].name.empty());
    assert(TlgTrophy::DisplayName(list[1]) == "Trophy 2");
    assert(TlgTrophy::IconPng(1) == png1);
    assert(TlgTrophy::Count() == 2 && TlgTrophy::NextLocked() == 0);

    // Unlock: once only, a pop-up, flags, saved.
    assert(TlgTrophy::Unlock(1) && !TlgTrophy::Unlock(1) && !TlgTrophy::Unlock(5));
    TlgTrophy::Entry toast;
    assert(TlgTrophy::HasToast() && TlgTrophy::TakeUnlockToast(toast) && toast.id == 1);
    assert(!TlgTrophy::HasToast());
    uint32_t flags[4];
    TlgTrophy::UnlockFlags(flags);
    assert(flags[0] == 2u && flags[1] == 0 && TlgTrophy::NextLocked() == 0);

    // Restart with the key: the unlock is kept, names and grades read.
    TlgSettings::Get().trophy_key = "000102030405060708090A0B0C0D0E0F";
    TlgTrophy::Setup(trp, npbind, store, 1);
    std::string title;
    list = TlgTrophy::List(&state, &title);
    assert(state == 1 && title == "Test & Game");
    assert(list[0].name == "All of them" && list[0].grade == 'P' && !list[0].hidden && !list[0].unlocked_at);
    assert(list[1].name == "Secret" && list[1].detail == "A <hidden> one" && list[1].grade == 'B' && list[1].hidden);
    assert(list[1].unlocked_at > 0);

    // A wrong key: numbered again, flagged as wrong.
    TlgSettings::Get().trophy_key = "ffffffffffffffffffffffffffffffff";
    TlgTrophy::Reload();
    list = TlgTrophy::List(&state);
    assert(state == -1 && list[0].name.empty() && list[1].unlocked_at > 0);
    TlgSettings::Get().trophy_key = "not a key";
    TlgTrophy::Reload();
    TlgTrophy::List(&state);
    assert(state == -1);

    std::remove(store.c_str());
    std::printf("trophy test: ok\n");
    return 0;
}
