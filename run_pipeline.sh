#!/usr/bin/env bash
# End-to-end pipeline: newSim  ->  DRAMsim3 trace  ->  DRAMsim3  ->  merged results.
#
# Usage:   ./run_pipeline.sh <sim args...>
#   The arguments are passed to sim unchanged:
#     arrayConfig layer inputCapKB weightCapKB outputCapKB [sramDir]
#   The *CapKB values are accepted for compatibility but ignored (the CACTI SRAM
#   library is fixed at 256KB); bandwidth is auto-selected from the array config.
#   e.g.   ./run_pipeline.sh array_config/A3_ReuseB_64x64x1x1x1.cfg \
#                            layer/huggyllama__llama-7b/prefill/00_q_proj.cfg \
#                            256 512 1024 cacti_results
#
# sim must run with DRAM_MODEL unset or =dramsim3; in analytic mode it writes no
# trace and this script stops at step 1.
#
# Env overrides:
#   DRAM_CFG   (default configs/HBM2_8Gb_x128.ini)   DRAM config, relative to DRAMsim3/
#   DRAM_LABEL (default HBM2_8Gb_x128)               label printed in the merged file
#   DRAM_MEM_TCK (default 1.0)                        DRAM tCK in ns (must match the config)
#   DRAM_CYCLES (default: auto from trace)            #cycles to simulate
#   OUT_DIR    (default DRAMsim3/out_newsim)          DRAMsim3 output directory
set -euo pipefail
cd "$(dirname "$0")"   # run from the repo root (relative paths below)

SIM=./sim
DRAMSIM_DIR=DRAMsim3
DRAM_CFG="${DRAM_CFG:-configs/HBM2_8Gb_x128.ini}"
DRAM_LABEL="${DRAM_LABEL:-HBM2_8Gb_x128}"
DRAM_MEM_TCK="${DRAM_MEM_TCK:-1.0}"
OUT_DIR="${OUT_DIR:-$DRAMSIM_DIR/out_newsim}"

# Make sure sim is up to date.
make -s sim

# 1) run newSim; it writes <name>.txt + <name>.trace and prints the trace path.
LOG=$(mktemp)
"$SIM" "$@" | tee "$LOG"
TRACE=$(grep -oP 'DRAM trace written: \K[^ ]+' "$LOG" || true)
rm -f "$LOG"
[ -n "$TRACE" ] || { echo "ERROR: could not find trace path in sim output" >&2; exit 1; }
RESULT="${TRACE%.trace}.txt"

# 2) size the DRAM run: last trace timestamp (column 3) + 10% + 100000 cycles of
#    margin (unless DRAM_CYCLES is given).
if [ -z "${DRAM_CYCLES:-}" ]; then
    LAST=$(tail -n 1 "$TRACE" | awk '{print $3}')
    DRAM_CYCLES=$(( LAST + LAST / 10 + 100000 ))
fi
echo ">> trace   : $TRACE"
echo ">> result  : $RESULT"
echo ">> DRAM    : $DRAM_CFG  for $DRAM_CYCLES cycles"

# 3) run DRAMsim3 on the trace (paths must be absolute since we cd into DRAMsim3/).
mkdir -p "$OUT_DIR"
ABS_TRACE="$(realpath "$TRACE")"
ABS_OUT="$(realpath "$OUT_DIR")"
( cd "$DRAMSIM_DIR" && ./build/dramsim3main "$DRAM_CFG" -c "$DRAM_CYCLES" \
      -t "$ABS_TRACE" -o "$ABS_OUT" >/dev/null )

# 4) merge DRAMsim3 final results into the newSim result file (in place, idempotent).
python3 merge_results.py "$RESULT" "$OUT_DIR/dramsim3.json" \
      --label "$DRAM_LABEL" --mem-tck "$DRAM_MEM_TCK"
echo ">> DONE. Merged results written to: $RESULT"
