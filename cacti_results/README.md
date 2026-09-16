# cacti_results_lowleak_rebank

An SRAM library on the low-leakage (itrs-lstp/itrs-lop) + ECC=false basis in which **the bank
count (N) and sub-array structure are re-selected to match the objective**. It replaces
`cacti_results_lowleak_noecc`.

Generator: [`cacti/bw_sweep/rebank_cacti_results.py`](../cacti/bw_sweep/rebank_cacti_results.py)
(The original `build_cacti_results.sh` was left untouched. The output schema is the same 21-column UHDR.)

```
python3 cacti/bw_sweep/rebank_cacti_results.py
# tune via env vars: OUT / BASE_CFG / BURST_BASE_CFG / NWAYS / TARGET_NS / NORMAL_TARGET_NS / ONLY / FORCE_INPUT
```

## What changed

The two selection rules in `build_cacti_results.sh` were unrelated to the objective (TOPS/W/mm^2):

| | Rule | Problem |
|---|---|---|
| `emit_auto` (normal) | **first (= smallest) N** in WAYS meeting cycle<=2ns | ignores energy and area; access unchecked |
| `lock_structure` (burst_\*) | candidate score = **leak x area** | dynamic energy is missing. In this batch leakage is 1% of total energy |

The script itself criticizes min-N in its burst_weight comments ("min-N is unrelated to the
objective"), yet that rule remained in `normal`. Both rules under-select N.

Changes:
1. Score replaced by **fJ/bit x area** (evaluated at the provisioning point = the series' max width).
2. N candidates extended **up to 128** (the old WAYS went up to 16; effective picks were 2~8).
3. Gate on **both access and cycle** <= 2ns. The old gate only checked cycle.

## Invariants kept (verified)

* `burst_input` / `burst_weight` : all bandwidths in a series share **one** sub-array structure.
  Verified that `(Ndwl, Ndbl, Nspd, num_subarrays, block_B)` is single-valued in each directory.
* `normal` : each width is optimized independently (no ladder lock). The first version used
  per-width free optimization (Force cache config=false), as in the original script; the current
  `build_normal` instead sweeps a forced Ndwl/Ndbl/Nspd grid per width with the same fJ/bit x area
  score and a tighter `NORMAL_TARGET_NS` gate (default 1.9ns), because the delay-only free optimum
  loses 31~54% on energy x area.
* burst family area/leak : the max-width (provisioning) value is charged to every row.
* schema and (macro, band) coverage : same as `cacti_results_lowleak_noecc`, plus 6 `max256_min8` rows.
* **0 rows violate 2ns.**

## Selection results

| Source | Before (noecc) | New | fJ/bit @max | area mm^2 |
|---|---|---|---|---|
| `burst_input_256K` | N=8, 2/2/1 | **N=64, 2/2/1** | 178.5 -> **49.8** | 3.216 -> 1.946 |
| `burst_weight_max256_min8` | N=1, 4/4/8 (violates 2ns) | **N=2, 2/8/8** | — -> 631.1 | — -> 2.023 |
| `burst_weight_max512_min32` | N=4, 2/2/8 | **N=8, 8/2/8** | 510.9 -> **306.3** | 1.584 -> 1.937 |
| `burst_weight_max1024_min128` | N=4, 2/2/4 | **N=32, 4/2/4** | 331.1 -> **151.4** | 1.687 -> 1.859 |
| `burst_weight_max2048_min512` | N=4, 2/2/2 | **N=32, 2/2/1** | 240.6 -> **99.0** | 1.894 -> 1.832 |
| `burst_weight_max4096_min2048` | N=4, 2/2/1 | **N=64, 2/2/1** | 195.7 -> **62.0** | 2.315 -> 1.767 |
| `normal` @512 | N=2, 2/8/4 | **N=4, 2/4/8** | 352.4 -> 337.0 | 2.470 -> 1.904 |
| `normal` @1024 | N=2, 2/4/2 | **N=8, 2/2/8** | 285.8 -> 264.8 | 2.572 -> 1.724 |
| `normal` @2048 | N=4, 2/4/2 | **N=32, 2/2/8** | 218.6 -> **160.4** | 3.203 -> 1.911 |
| `normal` @4096 | N=4, 2/2/1 | **N=32, 2/2/4** | 125.8 -> 121.4 | 2.256 -> 2.138 |

The physics is simple. Splitting into banks makes each bank's macro smaller and its bus
narrower, shortening the drive distance of the sub-array output driver — **energy, area and
timing improve at the same time** (this term is 86% of read energy). The limit is the per-bank
peripheral circuitry that gets replicated, which shows up first in leakage, but since leakage
is 1% of total energy it barely restrains the objective.

Some weight macros grew in area (`max512_min32` 1.584 -> 1.937, +22%). That was traded for a
2.5x energy improvement, which is exactly the trade-off the score is meant to make.

## 8192-bit additions for the new Toroidal RTL (2026-07-30)

The 11 dims of the new RTL (`Toroidal_DualMode_*` -> `A3_BurstMode_*.cfg`,
`Toroidal_ReuseB_*` -> `A3_ReuseB_*.cfg`):

```
1x1x1x32x32  1x2x2x32x8  1x2x2x8x32  1x4x4x2x32  1x4x4x32x2
2x2x1x8x32   2x2x1x32x8  2x4x2x2x32  2x4x2x32x2  4x4x1x2x32  4x4x1x32x2
```

Computing the required bandwidths exhaustively (ArrayConfig.hpp rules, dims = oR x oC x lR x lC x m,
rowDim=oR·lR·m, colDim=oC·lC·m) shows that colDim*8 falls into three groups {2048, 4096, 8192};
2048/4096 are covered by the existing series (max2048_min512, max4096_min2048 — min(U) is also
unchanged). **Only `1x1x1x32x32` (colDim=1024) needs an 8192-bit weight**, so two rows were
added (generator: [`cacti/bw_sweep/add_toroidal_8192.py`](../cacti/bw_sweep/add_toroidal_8192.py);
the score and 2ns gate are the same as rebank):

| Addition | Used by | Selection | fJ/bit | area mm^2 |
|---|---|---|---|---|
| `normal` @8192 (append) | `Toroidal_ReuseB_1x1x1x32x32` weight | N=32, 2/2/2 | 81.3 | 2.538 |
| `burst_weight_max8192_min8192` (new) | `Toroidal_DualMode_1x1x1x32x32` weight | N=128, 2/2/1, blk=8B | 42.0 | 1.830 |

In the max8192 series U = lC·m·8 = 8192 = R, so there is a single band [8192] (GEMV and GEMM
use the same row). All other roles are satisfied by existing rows: DualMode input GEMV is 8192
for every config (= 1024 total MACs x 8, present in `burst_input_256K`), and output requirements
of 184~2432 bits are served by the normal 512~4096 rows. 8192 was also added to the default bands
of `build_normal`, so it survives re-synthesis. However, for a full rerun of
`rebank_cacti_results.py` to regenerate `burst_weight_max8192_min8192`, the new
`A3_BurstMode_*.cfg` files must be present in `array_config/` (`ru_pairs()` derives R/U from the
cfgs) — until then, use `add_toroidal_8192.py`.

## `burst_weight_max256_min8` revived

Because the `wbank_for` table hard-coded N=1, this series was a single 256KB bank with access
2.49~2.71ns, violating 2ns in both the current and noecc libraries (it passed only because the
gate checked cycle alone). Banking with N=2 brings access to 1.921ns. This revives the 6 configs
that use this series, and the batch is back to all 76 configs:

| config | Previous objective | rebank estimate | Rank |
|---|---|---|---|
| `A3_BurstMode_1x32x32x1x1` | 0.0157 | 0.0265 | 47 -> 45 |
| `A3_BurstMode_2x32x16x1x1` | 0.0156 | 0.0262 | 48 -> 48 |
| `A3_BurstMode_4x32x8x1x1` | 0.0153 | 0.0257 | 51 -> 49 |
| `A3_BurstMode_8x32x4x1x1` | 0.0148 | 0.0247 | 54 -> 50 |
| `A3_BurstMode_16x32x2x1x1` | 0.0139 | 0.0229 | 61 -> 53 |
| `A3_BurstMode_32x32x1x1x1` | 0.0125 | 0.0203 | 69 -> 66 |

## Estimated batch impact

Over all of `batch_fixed256_lowleak_results`: 76 configs x 124 layers = 9424 runs
(traffic and cycles unchanged; only SRAM energy and area replaced):

- Total energy **0.917x**, total area **0.701x**, objective **+55.4%**
- BurstMode **+64.0%**, ReuseB **+48.1%**
- **Regressions: 0 / 9424**
- Total area 14.43~20.42 -> **11.49~13.47 mm^2**, SRAM share of area 93.1% -> 90.2%
- Config rank correlation Spearman **0.975** — the top ordering is nearly unchanged

| config | Before | rebank | Rank |
|---|---|---|---|
| `A3_BurstMode_2x2x1x16x16` | 0.0255 | **0.0444** | 2 -> **1** |
| `A3_BurstMode_1x2x2x16x16` | 0.0255 | 0.0443 | 3 -> 2 |
| `A3_BurstMode_1x4x4x4x16` | 0.0256 | 0.0436 | 1 -> 3 |
| `A3_BurstMode_2x4x2x4x16` | 0.0254 | 0.0432 | 4 -> 4 |

The top 5 are within 2% of each other, so the reshuffle of ranks 1~3 is not meaningful. **BurstMode keeps its lead.**

The two-regime split also holds, but the representative of each camp changes:

| phase | Before | rebank |
|---|---|---|
| prefill | `A3_ReuseB_4x4x1x4x16` 0.0451 | `A3_ReuseB_8x8x1x4x4` **0.0614** |
| vision | `A3_ReuseB_4x4x1x4x16` 0.0427 | `A3_ReuseB_8x8x1x4x4` 0.0586 |
| decode | `A3_BurstMode_4x4x1x4x16` 0.0151 | `A3_BurstMode_2x2x1x16x16` 0.0285 |
| action | `A3_BurstMode_2x2x1x16x16` 0.0182 | `A3_BurstMode_2x2x1x16x16` 0.0329 |

## leakage power

With up to 64 banks one might expect leakage to explode from replicated peripherals, but it does
not. Total cell count is fixed at 256KB and only the per-bank peripheral circuitry is replicated,
whose share is small with itrs-lop peripherals. At the provisioning point (mW):

| macro | itrs-hp original | lowleak | rebank | rebank/lowleak |
|---|---|---|---|---|
| `burst_mode_256K`@8192 | 426.8 | 4.012 | 4.245 (N=64) | 1.06x |
| `burst_max512_min32`@512 | 334.8 | 1.578 | 3.820 (N=8) | **2.42x** |
| `normal_results`@2048 | 336.1 | 1.394 | 2.340 (N=32) | 1.68x |
| `normal_results`@4096 | 344.5 | 1.559 | 2.509 (N=32) | 1.61x |
| `burst_max1024_min128`@1024 | 340.7 | 1.578 | 2.120 (N=32) | 1.34x |
| `burst_max2048_min512`@2048 | 372.8 | 1.923 | 1.906 (N=32) | 0.99x |
| `normal_results`@1024 | 332.5 | 1.520 | 1.205 (N=8) | 0.79x |
| `burst_max4096_min2048`@4096 | 384.7 | 2.867 | 2.145 (N=64) | **0.75x** |

Over the 9424-run batch:

- Absolute SRAM leakage energy **1.068x** (by role: input 1.007x / weight 1.139x / output 1.035x)
- Share within SRAM energy 3.38% -> **5.27%** — mostly because dynamic energy fell, not because leakage grew
- **Share of total energy 1.03% -> 1.19%** (max 1.79% -> 2.05%)

So leakage is still ~1% of total energy and does not constrain the design. Even the config with
the highest leakage share, `A3_BurstMode_1x16x16x4x1`, is at 1.68%. For reference, with the
original itrs-hp library the input macro leakage alone was 219x (median).

**Caution**: these picks are tuned for the "leakage is negligible" regime. `burst_max512_min32`
paying 2.42x leakage for 0.40~0.60x energy only makes sense because leakage is 1%. If BASE_CFG is
reverted to itrs-hp and the library re-synthesized, a leakage term must be added to the score (`fJ/bit x area`).

## Limitations

* The numbers above are **estimates**, computed with traffic and cycles fixed and only SRAM energy
  and area replaced. Since the cd selection depends on SRAM characteristics, actual values are only
  confirmed by rerunning the batch (`SRAM_DIR=cacti_results_lowleak_rebank`).
* The `fJ/bit x area` score is a self-contained proxy. For `burst_input` it exactly matched the
  result of optimizing directly on the batch objective (N=64, 2/2/1), but there is no guarantee it
  matches for every series.
* Write energy and the narrow bands do not enter the score directly (only read at the provisioning
  point is considered). In practice this was not a problem (0 regressions), but it needs
  re-checking for other workloads.
