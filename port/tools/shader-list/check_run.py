#!/usr/bin/env python3
"""check_run.py RUN_DIR [MARKS_TSV] [--json OUT] [--md OUT] [--renderer vkb0|vkb1]: per scene of a shot.py/sweep run with SHOT_PAD_PATTERN,
counted inside the pattern window (from "Lua: shot pad pattern start" to "... end", which starts 8 s after the load):
  built: shaders the game translated itself (misses with the list A+B), split: before the background build ended;
  slow: of those, built inside slow frames (the game's rule: over twice the median of the 61 frames around, and
  1/60 s over it); frames over 40 ms (meaningless on the box, which draws most scenes far below 25 FPS); uses: shaders used from the cache (first use), by list mark A/B;
  misses_A = built + uses of B programs; misses_B = built + uses of A programs; misses_AB = built;
  frames over 40 ms, worst frame ms, lowest 5 s FPS ("Frame stats" lines in the window).
MARKS_TSV: lines "0xHASH<TAB>A|B" (the list's marks); without it every use counts as A.
--renderer: which takes' logs are read: vkb0 (the normal renderer, default) or vkb1 (the new engine, TLG_VKB=1, whose shader store
prints the same lines: native_vkb_shader_store.h)."""
import json, re, sys, glob, os
args = sys.argv[1:]
def opt(name):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
jout, mout, renderer = opt('--json'), opt('--md'), opt('--renderer') or 'vkb0'
if renderer not in ('vkb0', 'vkb1'):
    sys.exit('--renderer: vkb0 or vkb1')
run = args[0]
marks = {}
if len(args) > 1:
    for l in open(args[1]):
        h, m = l.split()[:2]; marks[int(h, 16)] = m
RB = re.compile(r'Shader built while playing: (0x[0-9a-f]+)_(\d+) \(hw stage (\d+)\)(.*)')
RU = re.compile(r'Shader used from the cache: (0x[0-9a-f]+)_(\d+) \(hw stage (\d+)\), (.*)')
RF = re.compile(r'Frame time: ([\d.]+) ms, (\d+) shader builds')
RS = re.compile(r'Frame stats: ([\d.]+) FPS, worst frame ([\d.]+) ms')
rows = []
for log in sorted(glob.glob(os.path.join(run, '*', renderer + '_take*.log'))):
    scene = log.split('/')[-2]
    ls = open(log, errors='replace').read().splitlines()
    s = next((i for i, l in enumerate(ls) if 'Lua: shot pad pattern start' in l), None)
    if s is None:
        rows.append({'scene': scene, 'take': os.path.basename(log), 'error': 'no pattern start'}); continue
    e = next((i for i, l in enumerate(ls) if 'Lua: shot pad pattern end' in l and i > s), len(ls))
    r = {'scene': scene, 'take': os.path.basename(log), 'complete': e < len(ls), 'built': [], 'built_bg': [],
         'slow_builds': 0, 'uses': [], 'over40': 0, 'worst_ms': 0.0, 'fps5': []}
    frames = [(i, float(m[1]), int(m[2])) for i, l in enumerate(ls) if (m := RF.search(l))]
    times = [f[1] for f in frames]
    r['slow'] = 0
    for k, (i, ms, nb) in enumerate(frames):
        if not s <= i < e:
            continue
        around = sorted(times[max(0, k - 30):k + 31]); med = around[len(around) // 2]
        if ms > 2 * med and ms > med + 16.667:
            r['slow'] += 1; r['slow_builds'] += nb
        r['over40'] += ms > 40; r['worst_ms'] = max(r['worst_ms'], ms)
    for l in ls[s:e]:
        if m := RB.search(l):
            (r['built_bg'] if 'background build still running' in m[4] else r['built']).append(f'{m[1]}_{m[2]}')
        elif m := RU.search(l):
            r['uses'].append((int(m[1], 16), 'lookup' in m[4]))
        elif m := RS.search(l):
            r['fps5'].append(float(m[1])); r['worst_ms'] = max(r['worst_ms'], float(m[2]))
    useA = sum(marks.get(h, 'A') == 'A' for h, _ in r['uses']); useB = len(r['uses']) - useA
    n = len(r['built'])
    r.update(useA=useA, useB=useB, lookups=sum(lk for _, lk in r['uses']), misses_AB=n, misses_A=n + useB,
             misses_B=n + useA, min_fps5=min(r['fps5']) if r['fps5'] else None, uses=len(r['uses']))
    rows.append(r)
ok = [r for r in rows if 'error' not in r]
tot = {k: sum(r[k] for r in ok) for k in ('misses_A', 'misses_B', 'misses_AB', 'useA', 'useB', 'slow_builds', 'slow', 'over40')}
tot['built_bg'] = sum(len(r['built_bg']) for r in ok)
tot['scenes'] = len(ok); tot['no_pattern'] = len(rows) - len(ok)
print('totals', json.dumps(tot))
if jout:
    json.dump({'totals': tot, 'scenes': rows}, open(jout, 'w'), indent=0)
if mout:
    with open(mout, 'w') as f:
        f.write('| | misses with A alone | B alone | A+B |\n|---|---|---|---|\n')
        f.write(f"| all {tot['scenes']} scenes | {tot['misses_A']} | {tot['misses_B']} | {tot['misses_AB']} |\n\n")
        f.write('| scene | A alone | B alone | A+B | built inside slow frames | slow frames | built before the background build ended | frames >40 ms | worst ms | lowest 5 s FPS |\n|---|---|---|---|---|---|---|---|---|---|\n')
        for r in sorted(ok, key=lambda r: (-r['misses_AB'], -r['misses_A']))[:25]:
            f.write(f"| {r['scene']} | {r['misses_A']} | {r['misses_B']} | {r['misses_AB']} | {r['slow_builds']} | {r['slow']} | {len(r['built_bg'])} | {r['over40']} | {r['worst_ms']:.0f} | {r['min_fps5'] or '-'} |\n")
        zero = [r['scene'] for r in ok if r['misses_AB'] == 0]
        f.write(f"\nScenes with 0 misses (A+B): {len(zero)}: " + ', '.join(zero) + '\n')
        bad = [f"{r['scene']} ({r['error']})" for r in rows if 'error' in r]
        if bad:
            f.write('\nNot counted: ' + ', '.join(bad) + '\n')
