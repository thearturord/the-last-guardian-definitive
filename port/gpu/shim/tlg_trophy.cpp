// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: trophies (issue #60), see tlg_trophy.h.
#include "tlg_trophy.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <mutex>

#include "tlg_settings.h"

namespace TlgTrophy {
namespace {

// ---- AES-128 (FIPS-197), tables made once ----
struct Tables {
    unsigned char sbox[256], inv[256];
    Tables() {
        unsigned char p = 1, q = 1;
        do { // p walks the multiplicative group by 3, q by its inverse
            p = p ^ (unsigned char)(p << 1) ^ (p & 0x80 ? 0x1b : 0);
            q ^= q << 1; q ^= q << 2; q ^= q << 4;
            if (q & 0x80) q ^= 0x09;
            const unsigned char x = q ^ (unsigned char)((q << 1) | (q >> 7)) ^ (unsigned char)((q << 2) | (q >> 6)) ^
                                    (unsigned char)((q << 3) | (q >> 5)) ^ (unsigned char)((q << 4) | (q >> 4));
            sbox[p] = x ^ 0x63;
        } while (p != 1);
        sbox[0] = 0x63;
        for (int i = 0; i < 256; ++i) inv[sbox[i]] = (unsigned char)i;
    }
};
const Tables& T() { static const Tables t; return t; }

unsigned char Mul(unsigned char a, unsigned char b) {
    unsigned char r = 0;
    while (b) {
        if (b & 1) r ^= a;
        a = (unsigned char)((a << 1) ^ (a & 0x80 ? 0x1b : 0));
        b >>= 1;
    }
    return r;
}

void Expand(const unsigned char key[16], unsigned char w[176]) {
    std::memcpy(w, key, 16);
    unsigned char rcon = 1;
    for (int i = 16; i < 176; i += 4) {
        unsigned char t[4] = {w[i - 4], w[i - 3], w[i - 2], w[i - 1]};
        if (i % 16 == 0) {
            const unsigned char first = t[0];
            t[0] = T().sbox[t[1]] ^ rcon; t[1] = T().sbox[t[2]]; t[2] = T().sbox[t[3]]; t[3] = T().sbox[first];
            rcon = Mul(rcon, 2);
        }
        for (int j = 0; j < 4; ++j) w[i + j] = w[i - 16 + j] ^ t[j];
    }
}

void Encrypt(const unsigned char w[176], const unsigned char in[16], unsigned char out[16]) {
    unsigned char s[16];
    for (int i = 0; i < 16; ++i) s[i] = in[i] ^ w[i];
    for (int round = 1; round <= 10; ++round) {
        unsigned char t[16];
        for (int i = 0; i < 16; ++i) t[i] = T().sbox[s[(i + 4 * (i % 4)) % 16]]; // SubBytes + ShiftRows
        if (round < 10) {
            for (int c = 0; c < 4; ++c) {
                unsigned char* col = t + 4 * c;
                const unsigned char a0 = col[0], a1 = col[1], a2 = col[2], a3 = col[3];
                col[0] = Mul(a0, 2) ^ Mul(a1, 3) ^ a2 ^ a3;
                col[1] = a0 ^ Mul(a1, 2) ^ Mul(a2, 3) ^ a3;
                col[2] = a0 ^ a1 ^ Mul(a2, 2) ^ Mul(a3, 3);
                col[3] = Mul(a0, 3) ^ a1 ^ a2 ^ Mul(a3, 2);
            }
        }
        for (int i = 0; i < 16; ++i) s[i] = t[i] ^ w[16 * round + i];
    }
    std::memcpy(out, s, 16);
}

void Decrypt(const unsigned char w[176], const unsigned char in[16], unsigned char out[16]) {
    unsigned char s[16];
    for (int i = 0; i < 16; ++i) s[i] = in[i] ^ w[160 + i];
    for (int round = 9; round >= 0; --round) {
        unsigned char t[16];
        for (int i = 0; i < 16; ++i) t[(i + 4 * (i % 4)) % 16] = T().inv[s[i]]; // InvShiftRows + InvSubBytes
        for (int i = 0; i < 16; ++i) t[i] ^= w[16 * round + i];
        if (round > 0) {
            for (int c = 0; c < 4; ++c) {
                unsigned char* col = t + 4 * c;
                const unsigned char a0 = col[0], a1 = col[1], a2 = col[2], a3 = col[3];
                col[0] = Mul(a0, 14) ^ Mul(a1, 11) ^ Mul(a2, 13) ^ Mul(a3, 9);
                col[1] = Mul(a0, 9) ^ Mul(a1, 14) ^ Mul(a2, 11) ^ Mul(a3, 13);
                col[2] = Mul(a0, 13) ^ Mul(a1, 9) ^ Mul(a2, 14) ^ Mul(a3, 11);
                col[3] = Mul(a0, 11) ^ Mul(a1, 13) ^ Mul(a2, 9) ^ Mul(a3, 14);
            }
        }
        std::memcpy(s, t, 16);
    }
    std::memcpy(out, s, 16);
}

// ---- the trophy file ----
uint32_t Be32(const unsigned char* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint64_t Be64(const unsigned char* p) { return uint64_t(Be32(p)) << 32 | Be32(p + 4); }

struct FileEntry { std::string name; uint64_t offset, size; };

std::vector<FileEntry> TrpEntries(const std::vector<unsigned char>& trp) {
    std::vector<FileEntry> out;
    if (trp.size() < 0x60 || Be32(trp.data()) != 0xDCA24D00u) return out;
    const uint32_t count = Be32(&trp[0x10]), size = Be32(&trp[0x14]);
    if (size < 0x30 || count > 4096) return out;
    for (uint32_t i = 0; i < count; ++i) {
        const size_t at = 0x60 + size_t(i) * size;
        if (at + 0x30 > trp.size()) break;
        FileEntry e;
        e.name.assign(reinterpret_cast<const char*>(&trp[at]), strnlen(reinterpret_cast<const char*>(&trp[at]), 32));
        e.offset = Be64(&trp[at + 32]);
        e.size = Be64(&trp[at + 40]);
        if (e.offset + e.size <= trp.size()) out.push_back(std::move(e));
    }
    return out;
}

// The text between <tag ...> and </tag> starting at pos; XML entities decoded.
std::string Inner(const std::string& xml, size_t from, size_t to, const char* tag) {
    const std::string open = std::string("<") + tag, close = std::string("</") + tag + ">";
    size_t a = xml.find(open, from);
    if (a == std::string::npos || a >= to) return {};
    a = xml.find('>', a);
    const size_t b = xml.find(close, a);
    if (a == std::string::npos || b == std::string::npos || b > to) return {};
    std::string text = xml.substr(a + 1, b - a - 1), out;
    static const struct { const char* from; char to; } ents[] = {{"&amp;", '&'}, {"&lt;", '<'}, {"&gt;", '>'},
                                                                 {"&quot;", '"'}, {"&apos;", '\''}};
    for (size_t i = 0; i < text.size();) {
        bool done = false;
        for (const auto& e : ents) {
            const size_t n = std::strlen(e.from);
            if (text.compare(i, n, e.from) == 0) { out += e.to; i += n; done = true; break; }
        }
        if (!done) out += text[i++];
    }
    return out;
}

std::string Attr(const std::string& xml, size_t tag_at, const char* name) {
    const size_t end = xml.find('>', tag_at);
    const std::string key = std::string(" ") + name + "=\"";
    const size_t a = xml.find(key, tag_at);
    if (a == std::string::npos || a > end) return {};
    const size_t b = xml.find('"', a + key.size());
    return b == std::string::npos ? std::string{} : xml.substr(a + key.size(), b - a - key.size());
}

std::mutex mutex;
std::vector<unsigned char> trp_file;
std::string np_id, store_path, game_title;
int lang = 1;
int key_state = 0;
std::vector<Entry> entries;
std::deque<Entry> toasts;

void LoadStore() {
    FILE* f = std::fopen(store_path.c_str(), "r");
    if (!f) return;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        int id = -1; long long at = 0;
        if (line[0] == '#' || std::sscanf(line, "%d %lld", &id, &at) != 2) continue;
        for (auto& e : entries) if (e.id == id && !e.unlocked_at) e.unlocked_at = at > 0 ? at : 1;
    }
    std::fclose(f);
}

void SaveStore() {
    const std::string tmp = store_path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) { std::printf("Trophies: cannot write %s\n", store_path.c_str()); return; }
    std::fprintf(f, "# trophy id, unlock time (Unix seconds), name; written by the game\n");
    for (const auto& e : entries)
        if (e.unlocked_at) std::fprintf(f, "%d %lld %s\n", e.id, (long long)e.unlocked_at, DisplayName(e).c_str());
    std::fclose(f);
    std::remove(store_path.c_str());
    std::rename(tmp.c_str(), store_path.c_str());
}

// Names, grades and hidden flags from the ESFM files (key needed). Called with the mutex held.
void ReadNames() {
    key_state = 0;
    for (auto& e : entries) { e.name.clear(); e.detail.clear(); e.grade = 0; e.hidden = false; }
    unsigned char key[16];
    const std::string& hex = TlgSettings::Get().trophy_key;
    if (hex.empty()) return;
    if (!ParseKey(hex, key) || np_id.empty()) { key_state = -1; return; }
    const auto files = TrpEntries(trp_file);
    auto find = [&](const std::string& name) -> const FileEntry* {
        for (const auto& f : files) if (f.name == name) return &f;
        return nullptr;
    };
    char lang_name[32];
    std::snprintf(lang_name, sizeof(lang_name), "TROP_%02d.ESFM", lang);
    const FileEntry* file = find(lang_name);
    if (!file) file = find("TROP.ESFM");
    std::string xml;
    if (!file || !DecryptEsfm(key, np_id, &trp_file[file->offset], file->size, xml)) {
        key_state = -1;
        std::printf("Trophies: the trophy key does not open the trophy file\n");
        return;
    }
    key_state = 1;
    game_title = Inner(xml, 0, xml.size(), "title-name");
    for (size_t at = xml.find("<trophy "); at != std::string::npos; at = xml.find("<trophy ", at + 1)) {
        const size_t end = xml.find("</trophy>", at);
        if (end == std::string::npos) break;
        const int id = std::atoi(Attr(xml, at, "id").c_str());
        for (auto& e : entries) {
            if (e.id != id) continue;
            const std::string type = Attr(xml, at, "ttype");
            e.grade = type.empty() ? 0 : type[0];
            e.hidden = Attr(xml, at, "hidden") == "yes";
            e.name = Inner(xml, at, end, "name");
            e.detail = Inner(xml, at, end, "detail");
        }
    }
    std::printf("Trophies: names read (%s)\n", file->name.c_str());
}

} // namespace

void AesEncryptBlock(const unsigned char key[16], const unsigned char in[16], unsigned char out[16]) {
    unsigned char w[176]; Expand(key, w); Encrypt(w, in, out);
}
void AesDecryptBlock(const unsigned char key[16], const unsigned char in[16], unsigned char out[16]) {
    unsigned char w[176]; Expand(key, w); Decrypt(w, in, out);
}

bool ParseKey(const std::string& text, unsigned char out[16]) {
    std::string hex;
    for (char c : text) if (!std::isspace((unsigned char)c)) hex += c;
    if (hex.size() != 32) return false;
    for (int i = 0; i < 16; ++i) {
        unsigned v = 0;
        if (std::sscanf(hex.c_str() + 2 * i, "%2x", &v) != 1 || !std::isxdigit((unsigned char)hex[2 * i]) ||
            !std::isxdigit((unsigned char)hex[2 * i + 1]))
            return false;
        out[i] = (unsigned char)v;
    }
    return true;
}

// The same scheme as shadPS4's TRP reader: the NP id (zero-padded to 16 bytes), AES-CBC encrypted
// with the trophy key and a zero IV, is the file key; each ESFM is its IV (16 bytes) + AES-CBC data.
bool DecryptEsfm(const unsigned char key[16], const std::string& np_comm_id, const unsigned char* data, size_t size,
                 std::string& xml) {
    if (size < 32 || (size - 16) % 16 || np_comm_id.size() > 16) return false;
    unsigned char np[16] = {}, file_key[16], w[176];
    std::memcpy(np, np_comm_id.data(), np_comm_id.size());
    AesEncryptBlock(key, np, file_key);
    Expand(file_key, w);
    xml.assign(size - 16, '\0');
    unsigned char prev[16];
    std::memcpy(prev, data, 16);
    for (size_t at = 16; at < size; at += 16) {
        unsigned char block[16];
        Decrypt(w, data + at, block);
        for (int i = 0; i < 16; ++i) xml[at - 16 + i] = char(block[i] ^ prev[i]);
        std::memcpy(prev, data + at, 16);
    }
    xml.resize(strnlen(xml.data(), xml.size()));
    return xml.find("<trophy") != std::string::npos;
}

void Setup(std::vector<unsigned char> trp, std::vector<unsigned char> npbind, std::string store, int language) {
    std::scoped_lock lock{mutex};
    trp_file = std::move(trp);
    store_path = std::move(store);
    lang = language;
    np_id.clear();
    for (size_t i = 0; i + 12 <= npbind.size(); ++i) {
        if (std::memcmp(&npbind[i], "NPWR", 4) == 0 && npbind[i + 9] == '_') {
            np_id.assign(reinterpret_cast<const char*>(&npbind[i]), 12);
            break;
        }
    }
    entries.clear();
    for (const auto& f : TrpEntries(trp_file)) {
        int id = -1;
        if (f.name.size() == 11 && std::sscanf(f.name.c_str(), "TROP%3d.PNG", &id) == 1 && id >= 0 && id < 128) {
            Entry e;
            e.id = id;
            e.icon_png.assign(trp_file.begin() + f.offset, trp_file.begin() + f.offset + f.size);
            entries.push_back(std::move(e));
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.id < b.id; });
    ReadNames();
    LoadStore();
    int done = 0;
    for (const auto& e : entries) done += e.unlocked_at != 0;
    std::printf("Trophies: %zu in the trophy file, %d unlocked (%s), NP id %s\n", entries.size(), done,
                store_path.c_str(), np_id.empty() ? "missing" : np_id.c_str());
}

bool Unlock(int id) {
    std::scoped_lock lock{mutex};
    for (auto& e : entries) {
        if (e.id != id) continue;
        if (e.unlocked_at) return false;
        e.unlocked_at = (int64_t)std::time(nullptr);
        SaveStore();
        Entry toast = e;
        toast.icon_png.clear();
        toasts.push_back(std::move(toast));
        return true;
    }
    return false;
}

void UnlockFlags(uint32_t flags[4]) {
    std::scoped_lock lock{mutex};
    std::memset(flags, 0, 16);
    for (const auto& e : entries)
        if (e.unlocked_at) flags[e.id / 32] |= 1u << (e.id % 32);
}

int Count() {
    std::scoped_lock lock{mutex};
    return (int)entries.size();
}

int NextLocked() {
    std::scoped_lock lock{mutex};
    for (const auto& e : entries)
        if (!e.unlocked_at) return e.id;
    return -1;
}

std::vector<Entry> List(int* state, std::string* title) {
    std::scoped_lock lock{mutex};
    if (state) *state = key_state;
    if (title) *title = game_title;
    std::vector<Entry> out = entries;
    for (auto& e : out) e.icon_png.clear(); // icons: IconPng, once per texture
    return out;
}

std::vector<unsigned char> IconPng(int id) {
    std::scoped_lock lock{mutex};
    for (const auto& e : entries)
        if (e.id == id) return e.icon_png;
    return {};
}

void Reload() {
    std::scoped_lock lock{mutex};
    if (!trp_file.empty()) ReadNames();
}

bool HasToast() {
    std::scoped_lock lock{mutex};
    return !toasts.empty();
}

bool TakeUnlockToast(Entry& out) {
    std::scoped_lock lock{mutex};
    if (toasts.empty()) return false;
    out = std::move(toasts.front());
    toasts.pop_front();
    return true;
}

std::string DisplayName(const Entry& e) {
    if (!e.name.empty()) return e.name;
    return "Trophy " + std::to_string(e.id + 1);
}

} // namespace TlgTrophy
