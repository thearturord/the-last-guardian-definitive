# SPDX-License-Identifier: GPL-2.0-or-later
"""Mods > Foxtail toy: a foxtail grass stalk (green foxtail, Setaria viridis) for the boy, the toy the
developers cut (the game's PlayToy act still asks for it). Written from scratch like the ICO horns:
nothing is read from the game, shape and colours are computed here and the file writers are the ones
of ico_horns.py. The launcher builds one archive with the horns and the toy (tlg_run.py
build_mod_models); tests/test_ico_horns.py checks that every machine gets the same bytes.
Model space: metres, stem along +Y, hand grip at the origin; the game shows it at 1.5x (tlg_lua.cpp).
  python foxtail.py out.psarc"""
import hashlib
import math
import random
import struct
import sys

import ico_horns as H

MODEL = 'foxtail_toy'   # game/assets/props/level/foxtail_toy/foxtail_toy (tlg_lua.cpp FoxtailLua)

# ---- shape (real plant: stem 30-60 cm, head 5-10 cm long, ~1 cm thick, bristles ~1 cm, head nods over)
STEM_LEN, STEM_R, BELOW = 0.46, 0.0022, 0.05      # stem length above grip, radius, part below the hand
NOD = 0.10                                        # sideways bend of the top (m) -> head nods
HEAD_LEN, HEAD_R, BRISTLE = 0.10, 0.0085, 0.015  # spike length, core radius, bristle length
N_BRISTLE, BR_W = 900, 0.0016                     # bristle count, bristle half-width
LEAVES = ((0.10, 0.17, 0.0075, 0.0), (0.22, 0.14, 0.0065, 2.6))   # (height on stem, length, width, turn)

# ---- colours (sRGB) by texture row band: stem, leaf, head core, bristle base -> tip
STEM_C, LEAF_C, CORE_C, BR0, BR1 = (92, 118, 52), (78, 112, 44), (118, 140, 62), (128, 150, 70), (196, 204, 140)
BANDS = {'stem': (0.02, 0.18), 'leaf': (0.22, 0.42), 'core': (0.46, 0.60), 'brist': (0.64, 0.98)}


def colours():
    """512x512 BC1, one colour per 4x4 block, placed in the game's 8x8-block tile order."""
    rnd = random.Random(5)
    want = [[None] * 128 for _ in range(128)]
    def lerp(a, b, k): return tuple(x + (y - x) * k for x, y in zip(a, b))
    for Y in range(128):
        v = (Y + 0.5) / 128
        for X in range(128):
            n = 0.92 + 0.16 * rnd.random()
            if v < 0.20: c = STEM_C
            elif v < 0.44: c = lerp(LEAF_C, (98, 132, 58), abs((X + 0.5) / 128 - 0.5) * 2)   # paler leaf edges
            elif v < 0.62: c = CORE_C
            else: c = lerp(BR0, BR1, min(1.0, (v - 0.64) / 0.34))
            want[Y][X] = tuple(min(255, int(q * n)) for q in c)
    out = bytearray(128 * 128 * 8)
    for Y in range(128):
        for X in range(128):
            ty, py, tx, px = Y // 8, Y % 8, X // 8, X % 8
            sy, sx = ty * 8 + tx // 2, (tx % 2) * 64 + py * 8 + px
            c = H.rgb565(*want[Y][X]); out[(sy * 128 + sx) * 8:(sy * 128 + sx) * 8 + 8] = struct.pack('<HHI', c, c, 0)
    return bytes(out)


def norm(v): return H.norm(v)
def add(a, b): return tuple(x + y for x, y in zip(a, b))
def mul(a, k): return tuple(x * k for x in a)


def stem_pt(t):
    """t 0..1 from below the hand to the head base; bends sideways (+X) more towards the top."""
    y = -BELOW + t * (STEM_LEN + BELOW)
    s = max(0.0, (y / STEM_LEN)) ** 2.2
    return (NOD * s, y, 0.0)


def head_pt(t):
    """continues the stem curve over the spike: the head hangs over."""
    a0 = stem_pt(1.0); d = norm(H.sub(stem_pt(1.0), stem_pt(0.98)))
    ang = math.atan2(d[0], d[1]) + 0.9 * t          # keeps turning down
    L = HEAD_LEN * t
    # integrate the arc approximately
    n = 12; x, y = a0[0], a0[1]
    base = math.atan2(d[0], d[1])
    for i in range(n):
        a = base + 0.9 * t * (i + 0.5) / n
        x += math.sin(a) * L / n; y += math.cos(a) * L / n
    return (x, y, 0.0)


def build_mesh():
    pos, nrm, tan, uv, tris = [], [], [], [], []
    def vert(p, n, t, u, v):
        pos.append(p); nrm.append(norm(n)); tan.append(norm(t)); uv.append((u, v)); return len(pos) - 1
    def tube(f, rad, rings, sides, v0, v1, cap_top=True):
        W = sides + 1; start = len(pos)
        for r in range(rings):
            t = r / (rings - 1); c = f(t); T = norm(H.sub(f(min(1, t + 1e-3)), f(max(0, t - 1e-3))))
            B = norm(H.cross(T, (0, 0, 1))); N = H.cross(B, T); rr = rad(t)
            for s in range(W):
                a = 2 * math.pi * s / sides; d = add(mul(B, math.cos(a)), mul(N, math.sin(a)))
                vert(add(c, mul(d, rr)), d, T, 0.1 + 0.8 * s / sides, v0 + (v1 - v0) * t)
        for r in range(rings - 1):
            for s in range(sides):
                a, b, c, d = start + r*W+s, start + r*W+s+1, start + (r+1)*W+s, start + (r+1)*W+s+1
                tris.extend([(a, b, c), (b, d, c)])
        if cap_top:
            c = f(1.0); T = norm(H.sub(f(1.0), f(0.99))); k = vert(add(c, mul(T, rad(1.0) * 0.8)), T, (1, 0, 0), 0.5, v1)
            for s in range(sides): tris.append((start + (rings-1)*W+s, start + (rings-1)*W+s+1, k))
        b0 = len(pos); c = f(0.0); T = norm(H.sub(f(0.0), f(0.01)))
        k = vert(c, T, (1, 0, 0), 0.5, v0)
        for s in range(sides): tris.append((start + s + 1, start + s, k))
    def card(p0, p1, p2, v0, v1, u):
        """one thin triangle, both sides (own vertices per side)."""
        n = norm(H.cross(H.sub(p1, p0), H.sub(p2, p0))); T = norm(H.sub(p2, p0))
        for sgn in (1, -1):
            nn = mul(n, sgn)
            a = vert(p0, nn, T, u - 0.03, v0); b = vert(p1, nn, T, u + 0.03, v0); c = vert(p2, nn, T, u, v1)
            tris.append((a, b, c) if sgn > 0 else (a, c, b))
    def quad2(ps, vs, us):
        """strip of quads (list of left/right point pairs), both sides."""
        for side in (1, -1):
            idx = []
            for (l, r), v in zip(ps, vs):
                n = norm(H.cross(H.sub(r, l), (0, 0, 1) if True else 0)); n = mul(n, side)
                idx.append((vert(l, n, H.sub(r, l), us[0], v), vert(r, n, H.sub(r, l), us[1], v)))
            for i in range(len(idx) - 1):
                (a, b), (c, d) = idx[i], idx[i + 1]
                tris.extend([(a, b, c), (b, d, c)] if side > 0 else [(a, c, b), (b, c, d)])

    S0, S1 = BANDS['stem']; tube(stem_pt, lambda t: STEM_R * (1.15 - 0.3 * t), 22, 6, S0, S1, cap_top=False)
    C0, C1 = BANDS['core']
    hr = lambda t: HEAD_R * (0.75 + 0.35 * math.sin(math.pi * min(1, t * 1.2)) ) * (1 - 0.6 * max(0, t - 0.7) / 0.3)
    tube(head_pt, hr, 14, 8, C0, C1)
    # leaves: flat blades leaving the stem at an angle, arching outward, tapering to a point
    L0, L1 = BANDS['leaf']
    for h, ln, w, turn in LEAVES:
        base = stem_pt((h + BELOW) / (STEM_LEN + BELOW))
        out = (math.cos(turn), 0, math.sin(turn)); side = (-math.sin(turn), 0, math.cos(turn))
        ps, vs = [], []
        for i in range(9):
            t = i / 8; reach = ln * t
            p = add(base, add(mul(out, reach * 0.55), (0, reach * 0.8 - 0.35 * ln * t * t, 0)))
            ww = w * (0.6 + 0.4 * math.sin(math.pi * min(1, t * 1.6))) * (1 - t) + 0.0004
            ps.append((add(p, mul(side, -ww / 2)), add(p, mul(side, ww / 2)))); vs.append(L0 + (L1 - L0) * t)
        quad2(ps, vs, (0.15, 0.85))
    # bristles: many thin spikes all round the head, leaning towards the tip
    rnd = random.Random(17); B0, B1 = BANDS['brist']
    for i in range(N_BRISTLE):
        t = (i + rnd.random()) / N_BRISTLE
        c = head_pt(t); T = norm(H.sub(head_pt(min(1, t + 0.01)), head_pt(max(0, t - 0.01))))
        Bv = norm(H.cross(T, (0, 0, 1))); N = H.cross(Bv, T); a = rnd.random() * 2 * math.pi
        d = add(mul(Bv, math.cos(a)), mul(N, math.sin(a)))
        r0 = hr(t) * 0.9; ln = BRISTLE * (0.7 + 0.5 * rnd.random()) * (1 - 0.5 * max(0, t - 0.8) / 0.2)
        root = add(c, mul(d, r0)); dirn = norm(add(mul(d, 1.0), mul(T, 0.9 + 0.4 * rnd.random())))
        tip = add(root, mul(dirn, ln)); wv = norm(H.cross(dirn, d)); hw = BR_W
        card(add(root, mul(wv, -hw)), add(root, mul(wv, hw)), tip, B0, B1, 0.2 + 0.6 * rnd.random())
    return pos, nrm, tan, uv, tris


def build():
    """The horns (unchanged) plus the foxtail toy: {path in the archive: bytes}."""
    files = H.build()
    D = 'game/textures/props/foxtail_'; MAT = 'game/materials/SM_foxtail'
    files[MAT.lower() + '.bod'] = H.material(MAT, D + 'cl', D + 'nm', D + 'sp', D + 'rg', 0.2, 0.15)
    files[D + 'cl.gnf'] = H.gnf(512, 512, H.BC1, False, (4, 5, 6, 7), colours())
    files[D + 'nm.gnf'] = H.gnf(64, 64, H.BC5, True, (4, 5, 0, 1), bytes([0x80, 0x80] + [0] * 6) * 2 * 256)
    files[D + 'sp.gnf'] = H.gnf(64, 64, H.BC1, False, (4, 5, 6, 7),
                                struct.pack('<HHI', H.rgb565(14, 14, 14), H.rgb565(14, 14, 14), 0) * 256)
    files[D + 'rg.gnf'] = H.gnf(64, 64, H.BC4, False, (4, 0, 0, 1), bytes([210, 210] + [0] * 6) * 256)
    path = 'game/assets/props/level/' + MODEL + '/' + MODEL
    pos, nrm, tan, uv, tris = build_mesh()
    assert len(pos) < 65536, len(pos)
    vb = b''.join(struct.pack('<3f', *pos[i]) + H.sn8(nrm[i]) + b'\x00' + b'\xff' * 4 + H.sn8(tan[i]) + b'\x7f'
                  + struct.pack('<2f', *uv[i]) for i in range(len(pos)))
    bmin = tuple(min(p[k] for p in pos) for k in range(3)); bmax = tuple(max(p[k] for p in pos) for k in range(3))
    files[path + '.bod'] = H.model(path, MODEL, 'tlg_', MAT, len(pos), len(tris), bmin, bmax)
    files[path + '_gbuf_verts.data'] = H.cdat(0xc0000025, 32, vb)
    files[path + '_gbuf_elems.data'] = H.cdat(0xc0000029, 2, b''.join(struct.pack('<3H', *t) for t in tris))
    files[path + '_gbuf_uv1.data'] = H.cdat(0xc0000021, 8, b''.join(struct.pack('<2f', *q) for q in uv))
    return files


def write(path):
    """Builds the archive (horns and foxtail toy) at path; returns its sha256."""
    H.write_psarc(path, build())
    with open(path, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()


if __name__ == '__main__':
    print(write(sys.argv[1]))
