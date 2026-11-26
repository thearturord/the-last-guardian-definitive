# SPDX-License-Identifier: GPL-2.0-or-later
# tlg: draws the Fun move bar's icons (our own art, simple shapes) and writes them as one PNG atlas
# embedded in gpu/shim/tlg_fun_icons.h. Order = every move in TlgFun::Actions (tlg_fun.h). Run: python -I make_icons.py [preview.png]  (needs Pillow)
import io
import math
import os
import random
import re
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter

S = 96  # icon size in the atlas
K = 4   # supersampling
W = S * K
COLS = 9

GROUPS = {  # background: inner, outer colour
    'sound': ((222, 168, 72), (70, 40, 10)),
    'body': ((70, 190, 180), (10, 45, 50)),
    'play': ((235, 120, 110), (70, 20, 30)),
    'rest': ((120, 130, 235), (20, 22, 70)),
    'tummy': ((150, 200, 80), (25, 50, 10)),
    'sealed': ((214, 120, 230), (45, 15, 70)), # Sealed moves (cut or marketing only)
    'maybe': ((150, 150, 160), (30, 30, 38)),  # May not work: plain stone grey
    'unique': ((255, 236, 140), (140, 96, 0)),   # gold: our own recreated moves (UniqueHelp), in the Sealed group
}
INK = (255, 248, 230, 255)
SHADOW = (0, 0, 0, 150)


def bg(group):
    inner, outer = GROUPS[group]
    small = Image.new('RGBA', (S, S))
    px = small.load()
    for y in range(S):
        for x in range(S):
            d = min(1.0, math.hypot(x - S * 0.42, y - S * 0.38) / (S * 0.75))
            px[x, y] = tuple(int(inner[i] * (1 - d) + outer[i] * d) for i in range(3)) + (255,)
    return small.resize((W, W), Image.BILINEAR)


class Pen:
    """Light ink on a transparent layer; finish() adds a soft shadow under it."""

    def __init__(self):
        self.ink = Image.new('RGBA', (W, W), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.ink)

    @staticmethod
    def p(v):  # icon units 0..96 -> pixels
        return v * K

    def line(self, pts, w=6):
        pts = [(self.p(x), self.p(y)) for x, y in pts]
        self.d.line(pts, fill=INK, width=int(w * K), joint='curve')
        r = w * K / 2
        for x, y in (pts[0], pts[-1]):
            self.d.ellipse((x - r, y - r, x + r, y + r), fill=INK)

    def circle(self, x, y, r, fill=True, w=6):
        self.ellipse(x - r, y - r, x + r, y + r, fill, w)

    def ellipse(self, x0, y0, x1, y1, fill=True, w=6):
        box = (self.p(x0), self.p(y0), self.p(x1), self.p(y1))
        if fill:
            self.d.ellipse(box, fill=INK)
        else:
            self.d.ellipse(box, outline=INK, width=int(w * K))

    def arc(self, x, y, r, a0, a1, w=6):
        box = (self.p(x - r), self.p(y - r), self.p(x + r), self.p(y + r))
        self.d.arc(box, a0, a1, fill=INK, width=int(w * K))

    def poly(self, pts):
        self.d.polygon([(self.p(x), self.p(y)) for x, y in pts], fill=INK)

    def cut(self, x, y, r):  # erase a circle (crescent moon, ear hole)
        self.d.ellipse((self.p(x - r), self.p(y - r), self.p(x + r), self.p(y + r)), fill=(0, 0, 0, 0))

    def arrow(self, x0, y0, x1, y1, w=7, head=13):
        a = math.atan2(y1 - y0, x1 - x0)
        self.line([(x0, y0), (x1 - math.cos(a) * head * 0.8, y1 - math.sin(a) * head * 0.8)], w)
        l, r = a + 2.5, a - 2.5
        self.poly([(x1, y1), (x1 + math.cos(l) * head, y1 + math.sin(l) * head),
                   (x1 + math.cos(r) * head, y1 + math.sin(r) * head)])

    def paw(self, x, y, s=1.0):
        self.ellipse(x - 13 * s, y - 4 * s, x + 13 * s, y + 16 * s)
        for dx, dy in ((-15, -12), (-6, -20), (6, -20), (15, -12)):
            self.circle(x + dx * s, y + dy * s, 5.2 * s)

    def feather(self, x, y, ang, length, w=9):
        a = math.radians(ang)
        ex, ey = x + math.cos(a) * length, y + math.sin(a) * length
        nx, ny = -math.sin(a) * w / 2, math.cos(a) * w / 2
        mx, my = (x + ex) / 2, (y + ey) / 2
        self.poly([(x, y), (mx + nx, my + ny), (ex, ey), (mx - nx, my - ny)])

    def wing(self, x, y, side, spread, n=5, length=34):
        a0, a1 = spread
        for i in range(n):
            ang = a0 + (a1 - a0) * i / (n - 1)
            if side < 0:
                ang = 180 - ang
            self.feather(x, y, ang, length - abs(i - n // 2) * 2)

    def waves(self, x, y, n=3, r0=10, step=8, a0=-40, a1=40, w=5):
        for i in range(n):
            self.arc(x, y, r0 + i * step, a0, a1, w)

    def bar(self, y, x0=16, x1=80):
        self.line([(x0, y), (x1, y)], 6)

    def finish(self, base):
        shadow = Image.new('RGBA', (W, W), (0, 0, 0, 0))
        shadow.paste(Image.new('RGBA', (W, W), SHADOW), (0, 0), self.ink)
        shadow = shadow.filter(ImageFilter.GaussianBlur(3 * K))
        out = base.copy()
        out.alpha_composite(shadow, (K * 2, K * 3))
        out.alpha_composite(self.ink)
        return out


# The painted look (user pick 2026-11-15, "too smooth" before): light rays and grain on the group
# background, the shapes raised with a dark edge, a cream-to-group-colour fill, a gloss and a rim.
def _solid(col, mask):
    im = Image.new('RGBA', (W, W), tuple(col) + (0,))
    im.putalpha(mask)
    return im


def _grow(mask, px):
    while px > 0:
        n = min(31, int(px) | 1)
        mask = mask.filter(ImageFilter.MaxFilter(n))
        px -= n - 1
    return mask


def _vgrad(top, bot):
    im = Image.new('RGBA', (W, W))
    d = ImageDraw.Draw(im)
    for y in range(W):
        t = y / (W - 1)
        d.line([(0, y), (W, y)], fill=tuple(int(top[i] * (1 - t) + bot[i] * t) for i in range(3)) + (255,))
    return im


def _lit(col, f):
    return tuple(min(255, int(c * f)) for c in col)


def paint(p, group):
    inner, outer = GROUPS[group]
    m = p.ink.split()[3]
    base = bg(group)
    rays = Image.new('L', (W, W), 0)
    d = ImageDraw.Draw(rays)
    cx, cy = W * 0.42, W * 0.38
    for i in range(14):
        a = i * math.pi * 2 / 14
        d.polygon([(cx, cy), (cx + math.cos(a - .09) * W * 1.5, cy + math.sin(a - .09) * W * 1.5),
                   (cx + math.cos(a + .09) * W * 1.5, cy + math.sin(a + .09) * W * 1.5)], fill=34)
    base.alpha_composite(_solid(_lit(inner, 1.35), rays.filter(ImageFilter.GaussianBlur(2 * K))))
    # grain: fixed seed per group, so a rebuild gives the same atlas
    noise = Image.frombytes('L', (W, W), random.Random(group).randbytes(W * W))
    grain = noise.filter(ImageFilter.GaussianBlur(1.2)).point(lambda v: 200 + (v - 128) // 3)
    grained = Image.merge('RGBA', [ImageChops.multiply(c, grain) for c in base.split()[:3]] + [base.split()[3]])
    base = Image.blend(base, grained, 0.6)
    vig = Image.new('L', (W, W), 0)
    ImageDraw.Draw(vig).ellipse((-W * .2, -W * .25, W * 1.2, W * 1.15), fill=255)
    vig = ImageChops.invert(vig.filter(ImageFilter.GaussianBlur(14 * K)))
    base.alpha_composite(_solid((0, 0, 0), vig.point(lambda v: int(v * 0.7))))
    sh = _grow(m, 5 * K).filter(ImageFilter.GaussianBlur(4 * K))
    base.alpha_composite(_solid((0, 0, 0), sh.point(lambda v: int(v * 0.6))), (2 * K, 4 * K))
    base.alpha_composite(_solid(_lit(outer, 0.6), _grow(m, 4 * K)))
    ink = _vgrad((255, 252, 238), _lit(inner, 1.12))
    ink.putalpha(m)
    base.alpha_composite(ink)
    a = m.filter(ImageFilter.GaussianBlur(3 * K))
    o = 3 * K // 2
    hi = ImageChops.multiply(ImageChops.subtract(a, ImageChops.offset(a, o, o)), m)
    lo = ImageChops.multiply(ImageChops.subtract(a, ImageChops.offset(a, -o, -o)), m)
    base.alpha_composite(_solid((255, 255, 255), hi.point(lambda v: min(255, v * 3))))
    base.alpha_composite(_solid(_lit(outer, 0.8), lo.point(lambda v: min(255, v * 2))))
    gl = Image.new('L', (W, W), 0)
    ImageDraw.Draw(gl).ellipse((-W * .3, -W * .75, W * 1.3, W * .42), fill=48)
    base.alpha_composite(_solid((255, 255, 255), gl.filter(ImageFilter.GaussianBlur(3 * K))))
    rim = Image.new('L', (W, W), 0)
    ImageDraw.Draw(rim).rounded_rectangle((K, K, W - K - 1, W - K - 1), 13 * K, outline=200, width=2 * K)
    base.alpha_composite(_solid(_lit(inner, 1.5), rim))
    shape = Image.new('L', (W, W), 0)
    ImageDraw.Draw(shape).rounded_rectangle((0, 0, W - 1, W - 1), 14 * K, fill=255)
    base.putalpha(ImageChops.multiply(base.split()[3], shape))
    return base


def drop(p, x, y, r):  # a water drop, point up
    p.circle(x, y, r)
    p.poly([(x - r * 0.9, y - r * 0.4), (x, y - r * 2.4), (x + r * 0.9, y - r * 0.4)])


def nose(p, x, y):  # Trico's two nostrils with the scent lines going up
    p.circle(x - 7, y, 4.5); p.circle(x + 7, y, 4.5)


def scent(p, x, y, n=3, h=30):
    for i in range(n):
        xi = x + (i - (n - 1) / 2) * 12
        p.line([(xi, y), (xi - 4, y - h * 0.33), (xi + 4, y - h * 0.66), (xi, y - h)], 4)


def boy(p, x, y, s=1.0):  # the boy: a head and a body
    p.circle(x, y - 14 * s, 7 * s)
    p.poly([(x - 9 * s, y + 18 * s), (x - 6 * s, y - 4 * s), (x + 6 * s, y - 4 * s), (x + 9 * s, y + 18 * s)])


def zz(p, x, y, s=1.0):
    p.line([(x, y), (x + 12 * s, y), (x, y + 12 * s), (x + 12 * s, y + 12 * s)], 4 * s)


def star(p, x, y, r):
    pts = []
    for i in range(8):
        a = math.pi / 4 * i - math.pi / 2
        rr = r if i % 2 == 0 else r * 0.32
        pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr))
    p.poly(pts)


def moon(p):
    p.circle(30, 30, 16)
    p.cut(38, 24, 14)


def draw(name, p):
    if name == 'A_sneeze':
        p.circle(30, 54, 14); p.circle(44, 46, 12); p.circle(40, 62, 10)
        for x, y, r in ((64, 34, 4), (74, 48, 5), (66, 58, 3.5), (80, 30, 3), (82, 62, 4)):
            p.circle(x, y, r)
    elif name == 'A_yawn':
        p.ellipse(28, 26, 68, 78, fill=False, w=7)
        p.poly([(36, 34), (42, 34), (39, 44)]); p.poly([(54, 34), (60, 34), (57, 44)])
        p.line([(74, 20), (86, 20), (74, 32), (86, 32)], 4)
    elif name == 'roar':
        p.poly([(14, 44), (44, 30), (40, 48), (44, 66), (14, 52)])
        p.waves(46, 48, 3, 14, 10, -45, 45, 7)
    elif name == 'howling':
        moon(p); p.waves(52, 80, 3, 12, 10, -100, -30, 6)
    elif name == 'howling_calm':
        moon(p); p.waves(52, 80, 2, 12, 9, -95, -35, 3.5)
    elif name == 'flap_wings':
        p.wing(46, 60, 1, (-100, -20)); p.wing(50, 60, -1, (-100, -20))
        p.line([(22, 76), (32, 82)], 4); p.line([(74, 76), (64, 82)], 4)
    elif name == 'A_wing_stretchA':
        p.wing(46, 50, 1, (-40, 30), 6, 38); p.wing(50, 50, -1, (-40, 30), 6, 38)
    elif name == 'A_wing_stretchB':
        p.wing(48, 52, 1, (-70, 0), 5, 34); p.wing(48, 52, -1, (-70, 0), 5, 34)
        p.arrow(48, 66, 48, 88, 5, 9)
    elif name == 'stretch_front':
        p.paw(24, 60, 0.75); p.arrow(40, 52, 86, 52)
    elif name == 'stretch_back':
        p.paw(72, 60, 0.75); p.arrow(56, 52, 10, 52)
    elif name == 'stretch_up':
        p.paw(48, 76, 0.75); p.arrow(48, 58, 48, 10)
    elif name == 'stand_shake_high':
        p.ellipse(34, 30, 62, 70)
        for s in (-1, 1):
            p.line([(48 + s * 22, 30), (48 + s * 28, 40), (48 + s * 22, 50), (48 + s * 28, 60),
                    (48 + s * 22, 70)], 4)
        for x, y in ((14, 24), (82, 26), (16, 76), (80, 74)):
            p.circle(x, y, 3.5)
    elif name in ('scratch_left', 'scratch_right'):
        s = -1 if name == 'scratch_left' else 1
        for i in (-1, 0, 1):
            x = 48 + i * 16
            p.line([(x - s * 12, 22), (x + s * 12, 74)], 6)
    elif name == 'kneading':
        p.paw(30, 58, 0.6); p.paw(66, 46, 0.6)
        p.arrow(30, 84, 30, 72, 4, 7); p.arrow(66, 12, 66, 24, 4, 7)
    elif name == 'P_eye_wink_left':
        p.arc(28, 36, 16, 20, 160, 6)
        p.ellipse(52, 28, 84, 60); p.cut(68, 44, 7); p.circle(68, 44, 4)
    elif name == 'P_ear_shake_left':
        p.poly([(34, 80), (48, 14), (62, 80)]); p.cut(48, 62, 8)
        p.arc(48, 44, 30, 200, 240, 4); p.arc(48, 44, 30, 300, 340, 4)
        p.arc(48, 44, 38, 205, 235, 3); p.arc(48, 44, 38, 305, 335, 3)
    elif name == 'roll_on_ground':
        p.arc(48, 46, 26, 40, 330, 7); p.arrow(70, 30, 74, 50, 6, 12); p.bar(84, 18, 78)
    elif name == 'sit_plays':
        p.circle(64, 34, 14, fill=False, w=5); p.line([(52, 30), (76, 38)], 3); p.paw(34, 66, 0.7)
    elif name == 'toilet':
        p.ellipse(28, 62, 68, 80); p.ellipse(34, 48, 62, 66); p.ellipse(40, 36, 56, 52)
        for x in (22, 74):
            p.line([(x, 40), (x - 4, 30), (x, 20)], 3.5)
    elif name == 'tlg_gag':  # a queasy mouth and a swirl
        p.line([(20, 66), (32, 58), (44, 66), (56, 58), (68, 66), (80, 58)], 6)
        p.arc(48, 32, 16, 0, 300, 5); p.arc(48, 32, 7, 60, 360, 4)
    elif name == 'tlg_throw_up':  # an open beak with drops falling out
        p.line([(16, 22), (56, 30), (16, 42)], 7)
        for x, y, r in ((50, 50, 6), (62, 60, 7), (52, 72, 5), (68, 78, 6), (40, 84, 4)):
            p.circle(x, y, r)
    elif name == 'sit':
        p.arrow(48, 12, 48, 62); p.bar(76)
    elif name == 'sit_shake':
        p.arrow(48, 12, 48, 58); p.bar(76)
        for s in (-1, 1):
            p.line([(48 + s * 18, 30), (48 + s * 24, 38), (48 + s * 18, 46)], 4)
    elif name == 'sit_end':
        p.bar(80); p.arrow(48, 70, 48, 14)
    elif name == 'lie_down':
        p.arrow(48, 14, 48, 50); p.line([(14, 68), (82, 68)], 9); p.bar(82)
    elif name == 'lie_down_relax':
        p.line([(14, 68), (82, 68)], 9); p.bar(82)
        p.line([(18, 30), (26, 24), (34, 30), (42, 24), (50, 30)], 4)
        p.line([(46, 46), (54, 40), (62, 46), (70, 40), (78, 46)], 4)
    elif name == 'rise':
        p.line([(14, 78), (82, 78)], 9); p.arrow(34, 66, 34, 16, 6, 12); p.arrow(62, 66, 62, 16, 6, 12)
    elif name == 'stand':
        p.ellipse(18, 30, 78, 56)
        for x in (26, 38, 58, 70):
            p.line([(x, 50), (x, 80)], 6)
        p.circle(80, 30, 9)
    elif name == 'A_groanA':  # a low rumbling sound
        p.poly([(14, 40), (40, 34), (40, 58), (14, 52)])
        p.line([(46, 46), (54, 40), (62, 52), (70, 40), (78, 52), (86, 46)], 5)
    elif name == 'A_face_rejoice':  # happy eyes and a smile
        p.arc(32, 44, 11, 200, 340, 6); p.arc(64, 44, 11, 200, 340, 6)
        p.arc(48, 52, 22, 20, 160, 6)
    elif name == 'A_inclines_neck':  # a tilted head with a curious question
        p.ellipse(18, 40, 58, 72); p.poly([(22, 46), (20, 24), (36, 40)]); p.poly([(44, 38), (56, 22), (56, 46)])
        p.arc(72, 30, 10, 180, 400, 5); p.line([(72, 40), (72, 48)], 5); p.circle(72, 58, 3.5)
    elif name == 'A_scolded':  # ears flat, a sad mouth, a drop
        p.ellipse(22, 34, 74, 76)
        p.poly([(26, 42), (6, 56), (24, 52)]); p.poly([(70, 42), (90, 56), (72, 52)])
        p.cut(36, 50, 4); p.cut(60, 50, 4)
        p.d.arc((p.p(36), p.p(58), p.p(60), p.p(78)), 200, 340, fill=(0, 0, 0, 0), width=4 * K)
        drop(p, 80, 26, 5)
    elif name == 'A_refuse':  # a head shaking no
        p.ellipse(32, 30, 64, 66)
        p.arrow(26, 48, 8, 48, 5, 9); p.arrow(70, 48, 88, 48, 5, 9)
        p.line([(38, 78), (58, 92)], 5); p.line([(58, 78), (38, 92)], 5)
    elif name == 'threat_posture':  # a hunched back with spiky fur, angry eyes
        p.line([(10, 74), (24, 52), (36, 36), (48, 30), (60, 36), (72, 52), (86, 74)], 7)
        for x, y, a in ((24, 52, -150), (36, 36, -120), (48, 30, -90), (60, 36, -60), (72, 52, -30)):
            p.feather(x, y, a, 18, 7)
        p.line([(36, 62), (44, 66)], 5); p.line([(60, 62), (52, 66)], 5)
    elif name == 'stand_shake_drainsWater':  # shaking, water flying off
        p.ellipse(34, 34, 62, 70)
        for s_ in (-1, 1):
            p.line([(48 + s_ * 22, 34), (48 + s_ * 28, 44), (48 + s_ * 22, 54), (48 + s_ * 28, 64)], 4)
        for x, y, r in ((14, 30, 4), (82, 28, 4.5), (12, 70, 3.5), (84, 72, 4), (48, 18, 3.5)):
            drop(p, x, y, r)
    elif name == 'excite_shake_head':  # a head shaking with stars
        p.ellipse(30, 36, 66, 74)
        p.arc(48, 56, 30, 200, 250, 5); p.arc(48, 56, 30, 290, 340, 5)
        star(p, 18, 22, 9); star(p, 80, 20, 8); star(p, 84, 70, 6)
    elif name == 'grooming_left':  # a feather being smoothed, a sparkle
        p.feather(20, 78, -45, 64, 18); p.line([(20, 78), (66, 32)], 3)
        p.arc(62, 62, 14, 300, 420, 4); star(p, 78, 22, 9)
    elif name == 'grooming_wing_left':
        p.wing(30, 66, 1, (-80, -10), 5, 44); star(p, 76, 66, 9); star(p, 82, 30, 6)
    elif name == 'P_fur_bristle':  # a puffed-up fluffy ball
        p.circle(48, 50, 18)
        for i in range(16):
            a = i * math.pi * 2 / 16
            p.line([(48 + math.cos(a) * 22, 50 + math.sin(a) * 22),
                    (48 + math.cos(a) * 34, 50 + math.sin(a) * 34)], 4)
    elif name == 'sit_plays_pecks_nose':  # a beak pecking at a dot
        p.poly([(14, 20), (54, 50), (22, 40)]); p.circle(64, 62, 8)
        for a in (-20, 20, 60):
            r = math.radians(a)
            p.line([(64 + math.cos(r) * 14, 62 + math.sin(r) * 14), (64 + math.cos(r) * 24, 62 + math.sin(r) * 24)], 4)
        p.bar(84, 30, 86)
    elif name == 'joy_jump':  # a hop over the ground with sparkles
        p.arc(48, 70, 30, 190, 350, 6); p.arrow(76, 58, 80, 72, 6, 12); p.bar(84)
        star(p, 48, 22, 10); star(p, 22, 30, 6); star(p, 76, 28, 6)
    elif name == 'mud_playing':  # mud splashes
        p.ellipse(18, 64, 78, 86)
        for x, y, r in ((24, 40, 7), (48, 28, 9), (72, 42, 6), (36, 54, 5), (62, 56, 5)):
            p.circle(x, y, r)
    elif name == 'cat_toy':  # a ball of yarn and its thread
        p.circle(54, 46, 22, fill=False, w=6)
        p.arc(54, 46, 13, 200, 340, 4); p.arc(54, 46, 13, 20, 160, 4); p.line([(36, 34), (72, 58)], 4)
        p.line([(34, 62), (24, 74), (36, 82), (16, 88)], 4)
    elif name == 'tlg_foxtail':  # a foxtail grass stalk: a bent stem, a fuzzy nodding head, one leaf
        p.line([(30, 90), (32, 66), (38, 44), (50, 28), (60, 22)], 5)
        p.line([(33, 64), (22, 52), (14, 46)], 4)
        head = [(60, 22), (68, 22), (75, 26), (80, 33), (82, 42)]
        for i in range(len(head) - 1):
            p.line([head[i], head[i + 1]], 12)
        for i, (x, y) in enumerate(head):
            for a in (-60, -20, 20, 60, 100, 140, 180, 220):
                r = math.radians(a + i * 17)
                p.line([(x, y), (x + math.cos(r) * 10, y + math.sin(r) * 10)], 2.5)
    elif name == 'tlg_foxtail_play':  # the foxtail toy waved, a paw reaching for its head
        p.line([(70, 90), (68, 68), (60, 48), (48, 34)], 4)
        head = [(48, 34), (40, 28), (32, 26), (24, 28)]
        for i in range(len(head) - 1):
            p.line([head[i], head[i + 1]], 9)
        for i, (x, y) in enumerate(head):
            for a in (-60, -20, 20, 60, 100, 140, 180, 220):
                r = math.radians(a + i * 17)
                p.line([(x, y), (x + math.cos(r) * 7, y + math.sin(r) * 7)], 2)
        p.arc(84, 22, 8, 100, 200, 3); p.arc(84, 22, 14, 100, 200, 3)
        p.paw(28, 70, 0.7)
    elif name == 'roll_in_nose':  # a nose with a roll around it
        nose(p, 48, 52); p.arc(48, 52, 30, 30, 320, 6); p.arrow(74, 34, 76, 52, 6, 12)
    elif name == 'smell':  # nostrils and scent
        nose(p, 48, 74); scent(p, 48, 60, 3, 40)
    elif name == 'touch_boy_right':  # a big paw touching the boy
        boy(p, 70, 58, 1.2); p.paw(30, 54, 0.85)
    elif name == 'touch_left_side_soft':  # a paw giving a soft pat
        p.paw(46, 60, 1.0); p.arc(46, 54, 36, 210, 250, 4); p.arc(46, 54, 36, 290, 330, 4)
        p.arc(46, 54, 44, 215, 245, 3); p.arc(46, 54, 44, 295, 325, 3)
    elif name == 'EX_photo':  # a camera with a flash
        p.poly([(14, 36), (30, 36), (36, 26), (60, 26), (66, 36), (82, 36), (82, 78), (14, 78)])
        p.cut(48, 56, 15); p.circle(48, 56, 8); p.cut(74, 44, 3.5)
        star(p, 84, 16, 9)
    elif name == 'EX_photo_claws':  # three claws slashing
        for i in (-1, 0, 1):
            x = 48 + i * 18
            p.poly([(x - 8, 16), (x + 2, 16), (x + 10, 52), (x + 2, 84), (x - 2, 52)])
    elif name == 'A_groanD':  # a lion roar: a mane of spikes and a big open beak
        for i in range(12):
            a = i * math.pi * 2 / 12
            p.feather(40 + math.cos(a) * 14, 48 + math.sin(a) * 14, math.degrees(a), 20, 9)
        p.circle(40, 48, 16)
        p.waves(56, 48, 3, 14, 10, -40, 40, 6)
    elif name == 'A_excite':  # an exclamation mark with stars
        p.poly([(40, 14), (56, 14), (52, 60), (44, 60)]); p.circle(48, 76, 7)
        star(p, 20, 30, 8); star(p, 78, 34, 8)
    elif name == 'A_hungry':  # a food barrel with a rumble
        p.ellipse(28, 20, 68, 32); p.poly([(28, 26), (68, 26), (66, 72), (30, 72)])
        p.cut(48, 46, 0.1); p.line([(30, 44), (66, 44)], 3)
        p.line([(14, 84), (22, 78), (30, 84), (38, 78), (46, 84), (54, 78), (62, 84), (70, 78), (78, 84)], 4)
    elif name == 'puts_sand':  # a claw digging, sand flying
        p.poly([(10, 80), (30, 60), (66, 60), (86, 80)])
        p.paw(40, 32, 0.6)
        for x, y, r in ((64, 36, 4), (74, 26, 3.5), (80, 42, 3), (70, 50, 2.5)):
            p.circle(x, y, r)
    elif name == 'smellA':  # sniffing round: nose, scent, a circling arrow
        nose(p, 48, 62); scent(p, 48, 50, 2, 24)
        p.arc(48, 52, 36, 200, 520, 5); p.arrow(78, 70, 70, 82, 5, 10)
    elif name == 'TC_reactionA_to_boy':  # the boy and a surprised mark
        boy(p, 38, 60, 1.3)
        p.poly([(68, 18), (80, 18), (77, 54), (71, 54)]); p.circle(74, 68, 5.5)
    elif name == 'liedown_smell':  # lying, sniffing
        p.line([(14, 74), (82, 74)], 9); p.bar(86)
        nose(p, 48, 58); scent(p, 48, 46, 3, 26)
    elif name == 'liedown_to_sleep':  # lying, eyes closing, a z
        p.line([(14, 74), (82, 74)], 9); p.bar(86)
        p.arc(32, 46, 10, 20, 160, 5); p.arc(56, 46, 10, 20, 160, 5); zz(p, 70, 18, 1.0)
    elif name == 'sleep':  # a moon and Zz
        moon(p); zz(p, 54, 46, 1.4); zz(p, 74, 26, 0.9)
    elif name == 'sleep_to_liedown':  # wake up: an eye opening with rays
        p.ellipse(26, 40, 70, 68); p.cut(48, 54, 9); p.circle(48, 54, 5)
        for a in (-150, -120, -90, -60, -30):
            r = math.radians(a)
            p.line([(48 + math.cos(r) * 28, 54 + math.sin(r) * 28), (48 + math.cos(r) * 40, 54 + math.sin(r) * 40)], 5)
    else:
        raise SystemExit('no icon for ' + name)


def works(header):
    """Every move (command, group) from tlg_fun.h, in menu order (all groups, May not work too)."""
    out, group = [], None
    names = {'Sounds and moods': 'sound', 'Body': 'body', 'Play': 'play', 'Tummy': 'tummy', 'Rest': 'rest',
             'Sealed': 'sealed', 'MayNotWork': 'maybe'}
    text = open(header, encoding='utf-8').read()
    text = text[text.index('Actions[] = {'):]
    text = text[:text.index('};')]
    for m in re.finditer(r'\{(nullptr|"[^"]*"), ("[^"]*"|MayNotWork|Sealed)[^}]*', text):
        cmd, name = m.group(1), m.group(2).strip('"')
        if cmd == 'nullptr':
            if name not in names:
                break
            group = names[name]
            continue
        out.append((cmd.strip('"'), 'unique' if 'UniqueHelp' in m.group(0) else group))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    shim = os.path.join(here, '..', '..', 'gpu', 'shim')
    moves = works(os.path.join(shim, 'tlg_fun.h'))
    rows = (len(moves) + COLS - 1) // COLS
    atlas = Image.new('RGBA', (COLS * S, rows * S), (0, 0, 0, 0))
    for i, (cmd, group) in enumerate(moves):
        p = Pen()
        draw(cmd, p)
        icon = paint(p, group).resize((S, S), Image.LANCZOS)
        atlas.paste(icon, ((i % COLS) * S, (i // COLS) * S))
    if len(sys.argv) > 1:
        atlas.save(sys.argv[1])
    buf = io.BytesIO()
    atlas.save(buf, 'PNG', optimize=True)
    data = buf.getvalue()
    lines = ['// SPDX-License-Identifier: GPL-2.0-or-later',
             '// tlg: generated by tools/fun-icons/make_icons.py (our own art), do not edit.',
             '#pragma once', '', 'namespace TlgFunIcons {',
             f'inline constexpr int Size = {S}, Columns = {COLS}, Count = {len(moves)};',
             f'inline constexpr unsigned char Png[{len(data)}] = {{']
    for i in range(0, len(data), 24):
        lines.append('    ' + ','.join(str(b) for b in data[i:i + 24]) + ',')
    lines += ['};', '} // namespace TlgFunIcons', '']
    with open(os.path.join(shim, 'tlg_fun_icons.h'), 'w', newline='\n') as f:
        f.write('\n'.join(lines))
    print(len(moves), 'icons,', len(data), 'bytes')


main()
