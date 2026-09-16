#!/usr/bin/env python3
"""Fill array_config/*.cfg power/area fields from Design Compiler reports.

Reads the reports in synthesis_results/ that follow either flat naming
convention

    0_<Design>_area.rpt            /  1_<Design>_power.rpt
    <Design>_0_<Design>_area.rpt   /  <Design>_1_<Design>_power.rpt

where <Design> is e.g. A2_ReuseB_2x4x4x4x8 / A3_ReuseB_1x8x8x4x4 /
Toroidal_DualMode_1x8x8x4x4. Toroidal_DualMode_<dims> maps to
array_config/A3_BurstMode_<dims>.cfg (the BurstMode configs are the Toroidal
architecture), Toroidal_ReuseB_<dims> maps to array_config/A3_ReuseB_<dims>.cfg;
every other design maps to array_config/<Design>.cfg.
Updates the matching cfg
*in place*, rewriting only these four values and leaving the header comments,
kind/dataflow and the dims untouched:

    arrayArea          <- "Total cell area:"                     (um^2)
    arraySwitchPower   <- Switch column of the top-level row     (mW)
    arrayInternalPower <- Int    column of the top-level row     (mW)
    arrayLeakagePower  <- Leak   column of the top-level row     (nW)

Those units are what SimulationResults.hpp expects (switchPowerMw /
internalPowerMw / leakagePowerNw), so the values are written through raw.

It also writes

    arrayClockPowerFraction   (default 0.4, override with env CLK_FRACTION)

the clock-tree/register share of total streaming power for the two-rate
energy model (Fix 1 of the array-energy fix plan; see SimulationResults.hpp): E_dyn = clkFrac*P*busyCycles
+ (1-clkFrac)*P*usefulStreamCycles. The DC reports are run without SAIF
(-analysis_effort low), so their power assumes typical toggle activity;
billing fill/zero-pad/drain cycles at that full rate double-counts idle
cycles as active. 0.4 is a first-cut for a register-heavy 65nm systolic
array (defensible range 0.3-0.5); replace with measured idle/stream rates
if SAIF-annotated power runs become available. The line is inserted after
arrayInternalPower when the cfg does not have it yet.

Error-stub reports ("Error: Current design is not defined.") are skipped, as
are designs with no matching cfg or no power report.

Usage:  [CLK_FRACTION=0.4] python3 parse_synthesis.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT, "synthesis_results")
CFG_DIR = os.path.join(ROOT, "array_config")

# Report designs whose cfg lives under a different name.
CFG_ALIAS_RE = [
    (re.compile(r"^Toroidal_DualMode_([0-9x]+)$"), r"A3_BurstMode_\1"),
    (re.compile(r"^Toroidal_ReuseB_([0-9x]+)$"), r"A3_ReuseB_\1"),
]

# "arrayArea                : 1.0"  ->  key, separator-with-padding, value
FIELD_RE = re.compile(r"^(\s*(arraySwitchPower|arrayInternalPower|"
                      r"arrayLeakagePower|arrayArea|arrayClockPowerFraction)"
                      r"\s*:\s*)(\S+)\s*$")

# Clock-tree/register share of streaming power (two-rate energy model).
CLK_FRACTION = float(os.environ.get("CLK_FRACTION", "0.4"))


def is_error_stub(text):
    """True if a DC report only contains the "design not defined" error."""
    return "Current design is not defined" in text


def parse_area(path):
    """Return total cell area in um^2, or None if the report is an error stub."""
    with open(path) as f:
        text = f.read()
    if is_error_stub(text):
        return None
    m = re.search(r"Total cell area:\s*([0-9.eE+\-]+)", text)
    if not m:
        raise ValueError(f"no 'Total cell area' in {path}")
    return float(m.group(1))


def parse_power(path, design):
    """Return (switch_mW, internal_mW, leakage_nW) for the top-level design row,
    or None if the report is an error stub.

    The hierarchical power table looks like

                                           Switch   Int      Leak     Total
        Hierarchy                          Power    Power    Power    Power    %
        ----------------------------------------------------------------------
        A2_ReuseB_2x4x4x4x8                 11.623  393.175 5.65e+06  410.450 100.0
          postProcessorC (TypeC_...)      3.79e-02    5.809 3.39e+04    5.881   1.4

    so the root row is the one whose first token is the design name; it is also
    the first data row after the separator. The search takes the first data row
    with at least 4 tokens and reads the last five columns; rows with fewer
    tokens are skipped. Reaching an indented row that does not start with the
    design name means the root row was not found, which raises ValueError.
    """
    with open(path) as f:
        lines = f.readlines()
    if any(is_error_stub(ln) for ln in lines):
        return None

    for i, ln in enumerate(lines):
        if ln.strip().startswith("Hierarchy") and "Power" in ln:
            # lines[i + 1] is the '----' separator; data rows follow.
            for row in lines[i + 2:]:
                if not row.strip() or row.strip().startswith("-"):
                    continue
                toks = row.split()
                if toks[0] != design and row[:1].isspace():
                    # An indented row means we passed the root without matching.
                    break
                if len(toks) < 4:
                    continue
                # trailing tokens: Switch Int Leak Total %
                sw, it, lk = toks[-5], toks[-4], toks[-3]
                return float(sw), float(it), float(lk)
            break
    raise ValueError(f"no top-level power row for {design} in {path}")


def update_cfg(path, switch, internal, leakage, area):
    """Rewrite the power/area values in an existing cfg.

    arrayClockPowerFraction is inserted after arrayInternalPower when the
    cfg predates the two-rate energy model; the other four must exist."""
    values = {
        "arraySwitchPower": switch,
        "arrayInternalPower": internal,
        "arrayLeakagePower": leakage,
        "arrayArea": area,
        "arrayClockPowerFraction": CLK_FRACTION,
    }
    with open(path) as f:
        lines = f.readlines()

    seen = set()
    for i, ln in enumerate(lines):
        m = FIELD_RE.match(ln)
        if m:
            key = m.group(2)
            lines[i] = f"{m.group(1)}{values[key]:.6f}\n"
            seen.add(key)

    missing = set(values) - seen - {"arrayClockPowerFraction"}
    if missing:
        raise ValueError(f"{path}: missing field(s) {sorted(missing)}")

    if "arrayClockPowerFraction" not in seen:
        for i, ln in enumerate(lines):
            if FIELD_RE.match(ln) and "arrayInternalPower" in ln:
                lines.insert(i + 1,
                             f"arrayClockPowerFraction  : {CLK_FRACTION:.6f}\n")
                break

    with open(path, "w") as f:
        f.writelines(lines)


def discover_jobs():
    """Return a list of (cfg_name, design_in_report, area_rpt, power_rpt), one
    per area report found in RESULT_DIR (file names, not full paths). The power
    report may not exist; main() checks that."""
    jobs = []
    for fn in sorted(os.listdir(RESULT_DIR)):
        m = re.match(r"0_(.+)_area\.rpt$", fn)
        if m:
            design = m.group(1)
            power_fn = f"1_{design}_power.rpt"
        else:
            m = re.match(r"(.+)_0_\1_area\.rpt$", fn)
            if not m:
                continue
            design = m.group(1)
            power_fn = f"{design}_1_{design}_power.rpt"
        cfg_name = design
        for pat, repl in CFG_ALIAS_RE:
            cfg_name = pat.sub(repl, cfg_name)
        jobs.append((cfg_name, design, fn, power_fn))
    return jobs


def main():
    """Update every matching cfg and print what was written or skipped."""
    jobs = discover_jobs()
    if not jobs:
        print(f"No *_area.rpt found in {RESULT_DIR}", file=sys.stderr)
        return 1

    written, skipped = [], []
    for cfg_name, design, area_fn, power_fn in jobs:
        area_rpt = os.path.join(RESULT_DIR, area_fn)
        power_rpt = os.path.join(RESULT_DIR, power_fn)
        cfg_path = os.path.join(CFG_DIR, f"{cfg_name}.cfg")

        if not os.path.exists(cfg_path):
            skipped.append((design, "no array_config/%s.cfg" % cfg_name))
            continue
        if not os.path.exists(power_rpt):
            skipped.append((design, "missing power report"))
            continue

        area = parse_area(area_rpt)
        power = parse_power(power_rpt, design)
        if area is None or power is None:
            skipped.append((design, "synthesis error stub"))
            continue

        sw, it, lk = power
        update_cfg(cfg_path, sw, it, lk, area)
        written.append((cfg_name, sw, it, lk, area))

    print(f"Updated {len(written)} config(s) in array_config/")
    for design, sw, it, lk, area in written:
        print(f"  {design:28s} switch={sw:>9.3f}mW  int={it:>10.3f}mW  "
              f"leak={lk:>12.4g}nW  area={area:>13.3f}um^2")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for design, why in skipped:
            print(f"  {design:28s} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
