#!/usr/bin/env python3
"""Compare two fixed-256K batch result directories (old vs new SRAM library).

Usage:  python3 compare_batches.py <old_batch_dir> <new_batch_dir>

Per-(config, layer) parse of the SRAM Optimizer Report:
  M,N,K -> ops = 2*M*N*K ; TOTAL energy (pJ, first "TOTAL :" in the report) ;
  area (um^2) ; cycles ; SRAM buffer energies (parsed but not used in the reports).
Only (config, layer) pairs present in both batches are compared, unweighted
(no layer-repetition weighting).
Aggregates per config: TOPS/W = sum(ops)/sum(energy_pJ)  [ops/pJ equals ops/s per W,
so the plain ratio is TOPS/W], area-norm = TOPS/W / area_mm2.  Reports:
  (a) overall TOPS/W uplift (per-config and fleet-wide)
  (b) area-normalized ranking shift (top 10, both batches)
  (c) BurstMode vs ReuseB prefill energy gap, old vs new
"""
import glob, os, re, sys
from collections import defaultdict

def parse(path):
    """Parse one report into a dict, or return None if dims/TOTAL/area are missing."""
    txt = open(path).read()
    m = re.search(r'\(M=(\d+), N=(\d+), K=(\d+)\)', txt)
    tot = re.search(r'TOTAL\s+:\s+([0-9.]+)', txt)
    area = re.search(r'-- Area \(um\^2\) --.*?TOTAL\s+:\s+([0-9.]+)', txt, re.S)
    cyc = re.search(r'cycles\s+:\s+(\d+)', txt)
    if not (m and tot and area):
        return None
    M, N, K = map(int, m.groups())
    d = dict(ops=2.0*M*N*K, total_pJ=float(tot.group(1)),
             area_um2=float(area.group(1)), cycles=int(cyc.group(1)) if cyc else 0)
    for role in ('Input', 'Weight', 'Output'):
        b = re.search(r'-- %s buffer\s*\(SRAM\) --.*?total energy\s+:\s+([0-9.]+)' % role, txt, re.S)
        d[role.lower()+'_pJ'] = float(b.group(1)) if b else 0.0
    sd = re.search(r'memory \(SRAM\)\s+:\s+([0-9.]+)\s+dynamic\s+:\s+([0-9.]+)', txt, re.S)
    d['sram_pJ'] = float(sd.group(1)) if sd else 0.0
    d['sram_dyn_pJ'] = float(sd.group(2)) if sd else 0.0
    return d

def load(root):
    """Parse root/<config>/*.txt into {(config, layer_stem): parsed dict}."""
    out = {}
    for f in glob.glob(os.path.join(root, '*', '*.txt')):
        cfg = os.path.basename(os.path.dirname(f))
        lay = os.path.basename(f)[:-4]
        r = parse(f)
        if r:
            out[(cfg, lay)] = r
    return out

def agg(res, keys=None):
    """Per config (optionally restricted to `keys`): TOPS/W = sum ops / sum pJ;
    area is the max over layers (expected constant per cfg)."""
    a = defaultdict(lambda: dict(ops=0.0, pJ=0.0, area=0.0, n=0))
    for (cfg, lay), r in res.items():
        if keys is not None and (cfg, lay) not in keys:
            continue
        d = a[cfg]
        d['ops'] += r['ops']; d['pJ'] += r['total_pJ']
        d['area'] = max(d['area'], r['area_um2']); d['n'] += 1
    out = {}
    for cfg, d in a.items():
        tpw = d['ops'] / d['pJ']              # ops per pJ == TOPS/W (ops/s divided by J/s; the 1e12 factors cancel)
        out[cfg] = dict(tpw=tpw, anorm=tpw / (d['area'] / 1e6), n=d['n'], pJ=d['pJ'], ops=d['ops'])
    return out

def main(old_dir, new_dir):
    """Print sections (a)-(c) comparing the old and new batch."""
    old, new = load(old_dir), load(new_dir)
    common = set(old) & set(new)
    print('parsed: old=%d new=%d common=%d' % (len(old), len(new), len(common)))
    ao, an = agg(old, common), agg(new, common)

    # (a) fleet-wide + per-config uplift
    to = sum(v['pJ'] for v in ao.values()); tn = sum(v['pJ'] for v in an.values())
    print('\n== (a) energy & TOPS/W ==')
    print('fleet total energy: old=%.4g new=%.4g  (%+.2f%%)' % (to, tn, 100*(tn-to)/to))
    ups = sorted(((an[c]['tpw']/ao[c]['tpw']-1)*100, c) for c in ao)
    print('per-config TOPS/W uplift: median=%+.2f%% min=%+.2f%% (%s) max=%+.2f%% (%s)'
          % (ups[len(ups)//2][0], ups[0][0], ups[0][1], ups[-1][0], ups[-1][1]))

    # (b) rankings
    for name, key in (('TOPS/W (energy)', 'tpw'), ('TOPS/W/mm2 (area-norm)', 'anorm')):
        ro = sorted(ao, key=lambda c: -ao[c][key])[:10]
        rn = sorted(an, key=lambda c: -an[c][key])[:10]
        print('\n== (b) top-10 by %s ==' % name)
        print('%-4s %-28s %-12s | %-28s %-12s' % ('rank', 'old', 'value', 'new', 'value'))
        for i in range(10):
            print('%-4d %-28s %-12.6g | %-28s %-12.6g'
                  % (i+1, ro[i], ao[ro[i]][key], rn[i], an[rn[i]][key]))

    # (c) BurstMode vs ReuseB prefill gap (matched A3 geometries, prefill layers only)
    pref = {k for k in common if '_prefill_' in k[1]}
    po, pn = agg(old, pref), agg(new, pref)
    print('\n== (c) prefill energy: A3_BurstMode vs A3_ReuseB (matched geometry) ==')
    gaps_o, gaps_n = [], []
    for cfg in sorted(po):
        if not cfg.startswith('A3_BurstMode_'):
            continue
        peer = cfg.replace('BurstMode', 'ReuseB')
        if peer not in po:
            continue
        go = (po[cfg]['pJ']/po[peer]['pJ']-1)*100
        gn = (pn[cfg]['pJ']/pn[peer]['pJ']-1)*100
        gaps_o.append(go); gaps_n.append(gn)
        print('  %-32s old %+7.2f%%   new %+7.2f%%' % (cfg[len('A3_BurstMode_'):], go, gn))
    if gaps_o:
        med = lambda x: sorted(x)[len(x)//2]
        print('  median gap: old %+0.2f%%  ->  new %+0.2f%%' % (med(gaps_o), med(gaps_n)))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
