#!/usr/bin/env python3
"""Merge DRAMsim3 final results into a newSim results file.

Usage:
  merge_results.py <newSim_result.txt> <dramsim3.json> [--label NAME] [--mem-tck NS]

With DRAM_MODEL=dramsim3 (the default), newSim reports only compute + SRAM energy
(DRAM is counts-only) and writes a DRAM trace for DRAMsim3. This script
reads the DRAMsim3 stats, aggregates them across channels, and appends a
"DRAM (DRAMsim3)" section plus DRAM-inclusive combined totals/efficiency to the
newSim results file, in place. It is idempotent (re-running replaces the merged
block rather than duplicating it). Normally invoked by run_pipeline.sh.

Options:
  --label NAME   DRAM configuration label shown in the block (default HBM2_8Gb_x128)
  --mem-tck NS   DRAM clock period in ns, used to convert cycles to time (default 1.0)
"""
import json
import re
import sys

# Header line of the merged block; everything from it to EOF is replaced on rerun.
MARKER = "==================  DRAM (DRAMsim3) + Combined Totals  =================="


# ----------------------------- DRAMsim3 side -----------------------------
def load_channels(path):
    """Load dramsim3.json and return the per-channel stat dicts ordered by channel id."""
    with open(path) as f:
        data = json.load(f)
    return [v for k, v in sorted(data.items(), key=lambda kv: int(kv[0]))]


def num(ch, key, default=0.0):
    """Return a numeric stat from a channel dict, or default if missing/non-numeric."""
    v = ch.get(key, default)
    return v if isinstance(v, (int, float)) else default


def aggregate(channels):
    """Combine channel stats: counts/energy/power/bandwidth are summed, num_cycles
    is the max, row-hit rates are over all commands, and latencies are averaged
    weighted by completed reads/writes (write latency is None if not reported).
    """
    rd = sum(num(c, "num_reads_done") for c in channels)
    wr = sum(num(c, "num_writes_done") for c in channels)
    rcmd = sum(num(c, "num_read_cmds") for c in channels)
    wcmd = sum(num(c, "num_write_cmds") for c in channels)
    has_wlat = any("average_write_latency" in c for c in channels)
    return {
        "channels": len(channels),
        "num_cycles": max(num(c, "num_cycles") for c in channels),
        "reads_done": rd,
        "writes_done": wr,
        "read_cmds": rcmd,
        "write_cmds": wcmd,
        "read_row_hit_rate": (sum(num(c, "num_read_row_hits") for c in channels) / rcmd)
        if rcmd else 0.0,
        "write_row_hit_rate": (sum(num(c, "num_write_row_hits") for c in channels) / wcmd)
        if wcmd else 0.0,
        "act_cmds": sum(num(c, "num_act_cmds") for c in channels),
        "pre_cmds": sum(num(c, "num_pre_cmds") for c in channels),
        "total_energy_pj": sum(num(c, "total_energy") for c in channels),
        "read_energy_pj": sum(num(c, "read_energy") for c in channels),
        "write_energy_pj": sum(num(c, "write_energy") for c in channels),
        "bandwidth_gbps": sum(num(c, "average_bandwidth") for c in channels),
        "power_mw": sum(num(c, "average_power") for c in channels),
        "avg_read_latency": (sum(num(c, "average_read_latency") * num(c, "num_reads_done")
                                 for c in channels) / rd) if rd else 0.0,
        "avg_write_latency": (sum(num(c, "average_write_latency") * num(c, "num_writes_done")
                                  for c in channels) / wr) if (wr and has_wlat) else None,
    }


# ----------------------------- newSim side -------------------------------
def grab(text, pattern):
    """Return the first capture of a multiline regex as float, or None."""
    m = re.search(pattern, text, re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_newsim(text):
    """Extract compute+SRAM TOTAL energy, throughput and efficiencies from a newSim result."""
    # take the compute+SRAM TOTAL (the Energy section's TOTAL comes before Area's)
    return {
        "compute_sram_total_pj": grab(text, r"^    TOTAL\s*:\s*([\d.eE+-]+)\s*$"),
        "tops": grab(text, r"^    throughput\s*:\s*([\d.eE+-]+)\s*TOPS"),
        "energy_eff": grab(text, r"^    energy eff\s*:\s*([\d.eE+-]+)\s*TOPS/W"),
        "area_norm_eff": grab(text, r"^    area-norm eff\s*:\s*([\d.eE+-]+)"),
    }


# ------------------------------- merge -----------------------------------
def build_block(a, ns, label, mem_tck):
    """Format the DRAM section and the DRAM-inclusive combined totals as text.

    a: aggregate() output, ns: parse_newsim() output. Combined totals are
    skipped if the newSim TOTAL could not be parsed.
    """
    # simulated time in microseconds (cycles * tCK[ns])
    sim_us = a["num_cycles"] * mem_tck * 1e-9 * 1e6
    wlat = ("n/a (posted writes)" if a["avg_write_latency"] is None
            else f"{a['avg_write_latency']:.2f} cyc  ({a['avg_write_latency']*mem_tck:.2f} ns)")

    lines = [MARKER,
             f"  -- DRAM (DRAMsim3: {label}) --",
             f"    channels          : {a['channels']}",
             f"    simulated cycles  : {int(a['num_cycles'])}  "
             f"(tCK={mem_tck} ns -> {sim_us:.3f} us)",
             f"    reads / writes    : {int(a['reads_done'])} / {int(a['writes_done'])}  (done)",
             f"    READ  cmds        : {int(a['read_cmds'])}  "
             f"(row-buffer hit {a['read_row_hit_rate']*100:.1f}%)",
             f"    WRITE cmds        : {int(a['write_cmds'])}  "
             f"(row-buffer hit {a['write_row_hit_rate']*100:.1f}%)",
             f"    ACT / PRE cmds    : {int(a['act_cmds'])} / {int(a['pre_cmds'])}",
             f"    avg read latency  : {a['avg_read_latency']:.2f} cyc  "
             f"({a['avg_read_latency']*mem_tck:.2f} ns)",
             f"    avg write latency : {wlat}",
             f"    bandwidth         : {a['bandwidth_gbps']:.3f} GB/s",
             f"    average power     : {a['power_mw']:.2f} mW",
             f"    DRAM energy       : {a['total_energy_pj']:.6g} pJ  "
             f"(read {a['read_energy_pj']:.4g} / write {a['write_energy_pj']:.4g})"]

    e_cpu = ns.get("compute_sram_total_pj")
    e_dram = a["total_energy_pj"]
    if e_cpu is not None:
        e_grand = e_cpu + e_dram
        dram_pct = 100.0 * e_dram / e_grand if e_grand else 0.0
        lines += ["  -- Combined Totals (compute + SRAM + DRAM) --",
                  f"    compute + SRAM    : {e_cpu:.6g} pJ",
                  f"    DRAM (DRAMsim3)   : {e_dram:.6g} pJ  ({dram_pct:.1f}% of total)",
                  f"    TOTAL energy      : {e_grand:.6g} pJ"]
        eff = ns.get("energy_eff")
        if eff is not None and e_grand > 0:
            opcount = eff * e_cpu               # energy_eff = opCount / E_compute_sram
            grand_eff = opcount / e_grand       # DRAM-inclusive TOPS/W
            lines.append(f"    energy eff        : {grand_eff:.6g} TOPS/W")
            an = ns.get("area_norm_eff")
            if an and an > 0:
                area_mm2 = eff / an             # back out area from the two efficiencies
                lines.append(f"    area-norm eff     : {grand_eff/area_mm2:.6g} TOPS/W/mm^2")
        if ns.get("tops") is not None:
            lines.append(f"    throughput        : {ns['tops']:.6g} TOPS  (unchanged)")
    else:
        lines.append("  (could not parse compute+SRAM TOTAL from newSim file; "
                     "combined totals skipped)")

    lines.append("=" * len(MARKER))
    return "\n".join(lines) + "\n"


def main():
    """Parse arguments, merge the DRAMsim3 block into the result file in place, and echo it."""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    ns_path, json_path = sys.argv[1], sys.argv[2]
    label, mem_tck = "HBM2_8Gb_x128", 1.0
    args = sys.argv[3:]
    for i, x in enumerate(args):
        if x == "--label" and i + 1 < len(args):
            label = args[i + 1]
        elif x == "--mem-tck" and i + 1 < len(args):
            mem_tck = float(args[i + 1])

    with open(ns_path) as f:
        text = f.read()
    # idempotent: drop any previously merged block
    text = text.split(MARKER)[0].rstrip("\n") + "\n"

    a = aggregate(load_channels(json_path))
    ns = parse_newsim(text)
    block = build_block(a, ns, label, mem_tck)

    with open(ns_path, "w") as f:
        f.write(text + block)

    print(f"Merged DRAMsim3 results into {ns_path}\n")
    print(block, end="")


if __name__ == "__main__":
    main()
