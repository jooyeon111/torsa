#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <memory>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <unordered_map>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>

#include "Layer.hpp"
#include "LayerParser.hpp"
#include "ArrayConfig.hpp"
#include "ArrayConfigParser.hpp"
#include "Array.hpp"
#include "Dram.hpp"
#include "DoubleBuffer.hpp"
#include "Scheduler.hpp"
#include "Simulator.hpp"
#include "SramResultParser.hpp"
#include "SramSelector.hpp"
#include "ContractingDimension.hpp"

// ============================================================================
// Optimizer
//
// Design-space search over SRAM choices and the contracting dimension (cd) for
// one (array config, layer) pair.
//
// Inputs : an array config file (Array_cfg) + a layer file (layer_cfg), and a
//          directory of CACTI SRAM characterization CSVs (default "cacti_results").
// Flow   : for each role (input / weight / output) the SramLibrary is filtered to
//          the SRAMs whose bandwidth satisfies the array geometry (proper band),
//          optionally restricted by capacity (OPT_CAP_MIN/OPT_CAP_MAX). Every
//          (input x weight x output) combination is enumerated, and for each one
//          cd is swept (see cdCandidates()); every (combination, cd) point is fully
//          simulated and the best cd is kept per combination. Combinations are then
//          ranked by the objective (default: minimum total energy).
//          Note: SramSelector currently returns a single 256 KB proper row per role,
//          so the enumeration collapses to 1x1x1 and only cd is actually searched.
//
// Two-stage DRAM energy:
//   stage 1 ranks all combinations with an analytic per-access DRAM model;
//   stage 2 re-simulates the top OPT_TOPK (default 8) with trace output and runs
//   DRAMsim3 on each trace to replace the analytic DRAM term with a measured one.
//
// Robustness:
//   Some (config, SRAM) combinations abort the process inside checkSramSize()/run()
//   (assert, division by zero, parts of the output-SRAM sizing that are still
//   placeholders). Running them in the main process would kill the whole optimizer.
//   Therefore each point is simulated in a fork()ed child: if the child exits
//   normally (= the point is actually simulatable) its results are read back over a
//   pipe; if it dies by signal / non-zero exit the point is marked infeasible. The
//   optimizer picks the best among the surviving points.
// ============================================================================
class Optimizer {
public:
    // Optimization objective.
    enum class Objective {
        TotalEnergy,         // minimize total energy (pJ)
        Cycles,              // minimize cycle count
        EnergyDelayProduct,  // minimize energy x cycles (EDP)
        TopsPerWattPerMm2    // maximize TOPS/W/mm^2 (area-normalized energy efficiency)
    };

    // One SRAM choice for a role: display/identity path + parsed characterization.
    struct SramOption {
        std::string path;   // "<csv path> [band<bits>]" identifier; also the lookup key in findSpec().
        SramSpec    spec;
    };

    // Access/leakage breakdown of one SRAM buffer (for the detailed report).
    struct BufStat {
        long   readCount   = 0;
        long   writeCount  = 0;
        long   readBytes   = 0;
        long   writeBytes  = 0;
        double readEnergy    = 0.0; // pJ
        double writeEnergy   = 0.0; // pJ
        double leakageEnergy = 0.0; // pJ
        double dynamicEnergy() const { return readEnergy + writeEnergy; }
        double totalEnergy()   const { return dynamicEnergy() + leakageEnergy; }
    };

    // Result of evaluating one (input, weight, output) SRAM combination at its best cd.
    struct Candidate {
        std::string inputSram;
        std::string weightSram;
        std::string outputSram;

        // The cd (free/streaming tile length) this candidate actually used. cd is no
        // longer uniquely determined by the input/weight capacities; evaluate() searches
        // for it, so it must travel with the candidate. refineCandidate (stage 2) reuses
        // this exact value to rebuild the same plan as stage 1 (re-deriving it would make
        // the two stages diverge).
        int    cd = 0;

        bool   feasible = false;  // Did the simulation run to completion (no abort)?
        bool   finished = false;  // Were all C tiles produced?
        long   cycles = 0;
        double totalEnergy  = 0.0; // pJ (DRAM-inclusive once DRAM is merged)
        double memoryEnergy = 0.0; // pJ (SRAM dynamic + leakage)
        double arrayEnergy  = 0.0; // pJ
        double tops         = 0.0; // throughput (TOPS)
        double topsPerWatt  = 0.0; // energy efficiency (TOPS/W)
        double topsPerWattPerMm2 = 0.0; // area-normalized efficiency (TOPS/W/mm^2)

        // ---- Detailed breakdown (expanded per component in the best-combination report) ----
        int    arrayActiveCount   = 0;   // number of tile ops (MAC) executed
        long   arrayBusyCycles    = 0;   // fill+stream+drain cycles (basis for clock power)
        long   usefulStreamCycles = 0;   // stream cycles with real data flowing (basis for datapath power)
        double arrayClockEnergy   = 0.0; // pJ (P_clk x busy) -- two-rate split
        double arrayDataEnergy    = 0.0; // pJ (P_data x useful)
        double arrayDynamicEnergy = 0.0; // pJ
        double arrayLeakageEnergy = 0.0; // pJ
        double arrayArea          = 0.0; // um^2 (systolic array area)
        double sramArea           = 0.0; // um^2 (sum of input+weight+output SRAM area)
        // Total on-chip area (mm^2). Denominator of the area-normalized efficiency.
        double totalAreaMm2() const { return (arrayArea + sramArea) / 1e6; }
        BufStat inBufStat;               // input  SRAM read/write/leakage breakdown
        BufStat wBufStat;                // weight SRAM
        BufStat outBufStat;              // output SRAM
        long   dramReads  = 0;           // DRAM 4096-bit access counts (analytic DRAM energy and
        long   dramWrites = 0;           //   the cd early-stop heuristic are based on these)

        double memoryDynamicEnergy() const {
            return inBufStat.dynamicEnergy() + wBufStat.dynamicEnergy() + outBufStat.dynamicEnergy();
        }
        double memoryLeakageEnergy() const {
            return inBufStat.leakageEnergy + wBufStat.leakageEnergy + outBufStat.leakageEnergy;
        }

        // ---- DRAM (analytic in stage 1, DRAMsim3-measured in stage 2; filled by the parent after the pipe) ----
        double energyComputeSram = 0.0; // memory (SRAM) + array energy, excluding DRAM, pJ
        bool   dramMerged  = false;     // Has DRAM energy been added (analytic or DRAMsim3)?
        bool   dramMeasured = false;    // Was DRAM measured by stage-2 DRAMsim3 (false if analytic)?
        double dramEnergy      = 0.0;   // DRAM dynamic energy (pJ): read+write (analytic) or read+write+act (DRAMsim3).
        double dramReadEnergy  = 0.0;   // pJ
        double dramWriteEnergy = 0.0;   // pJ
        double dramActEnergy   = 0.0;   // pJ (DRAMsim3: row-activation dynamic energy; 0 if analytic)
        // DRAM static (background standby + refresh) energy -- analogous to SRAM/array leakage.
        //   Same constant/formula as SimulationResults: 0.65 W x cycles x clockPeriod
        //   (total simulated cycles, not arrayBusyCycles).
        //   The dynamic term (read+write+act) excludes background/refresh, so this term adds it back.
        //   Cross-check: datasheet 0.653 W vs. DRAMsim3 honest-background 0.649 W (~1% agreement).
        static constexpr double dramStaticPowerW = 0.65;   // W, HBM2_8Gb_x128, 8 channels
        static constexpr double clockPeriodS     = 2e-9;   // s, one newSim cycle
        double dramStaticEnergy = 0.0;  // pJ (0.65 W x cycles x clockPeriod)
        double dramBandwidthGbps = 0.0; // GB/s (sum over channels; DRAMsim3 only)
        double dramPowerMw       = 0.0; // mW (sum over channels; DRAMsim3 only)
        long   dramReadsDone  = 0;      // DRAMsim3 completed read transactions
        long   dramWritesDone = 0;      // DRAMsim3 completed write transactions
        std::string traceRel;           // trace path relative to workDir_ (transient)

        double dramFraction() const {   // Share of total energy spent in DRAM (dynamic + static).
            return (totalEnergy > 0.0) ? (dramEnergy + dramStaticEnergy) / totalEnergy : 0.0;
        }

        // Sort key: "smaller is better". Maximization objectives (TOPS/W/mm^2) are negated.
        double score(Objective o) const {
            switch (o) {
                case Objective::TotalEnergy:        return totalEnergy;
                case Objective::Cycles:             return static_cast<double>(cycles);
                case Objective::EnergyDelayProduct: return totalEnergy * static_cast<double>(cycles);
                case Objective::TopsPerWattPerMm2:  return -topsPerWattPerMm2;
            }
            return totalEnergy;
        }
    };

    // Parses the array config and layer, loads the SRAM candidate lists from sramDir,
    // reads the OPT_CD_* knobs, creates the temp work dir and locates DRAMsim3.
    // Throws std::runtime_error if some role has no SRAM satisfying the array bandwidth.
    Optimizer(const std::string& arrayConfigPath,
              const std::string& layerConfigPath,
              const std::string& sramDir = "cacti_results")
        : arraySpec_(parseArrayConfig(arrayConfigPath)),
          layer_(parseLayer(layerConfigPath)),
          // config_ is a geometry placeholder used by the bandwidth guard (loadSrams)
          // and by cd auto-derivation. cd differs per candidate (input/weight SRAM
          // capacity and the cd search), so the schedule (tiles/ops) is not built once
          // here but per candidate in evaluate().
          //   makeResolvedConfig finalizes arraySpec_ before building the config:
          //   ToroidalDual with layer M>1 is replaced by Toroidal ReuseB (prevents the
          //   GEMV-only schedule from folding a general GEMM into GEMV mode). All later
          //   makeConfig(cd)/isBurst()/cd derivation see the finalized kind.
          config_(makeResolvedConfig(arraySpec_, layer_))
    {
        loadSrams(sramDir);
        readCdKnobs();
        prepareWorkDir();
        setupDramsim3();
    }

    void      setObjective(Objective o) { objective_ = o; }
    Objective objective() const         { return objective_; }
    bool      isBurst() const           { return arraySpec_.isBurst(); }

    // All surviving candidates (one per SRAM combination that ran at some cd), sorted
    // best-first after optimize().
    const std::vector<Candidate>& candidates() const { return candidates_; }

    // True if at least one valid (feasible && finished) combination exists.
    bool hasFeasible() const { return feasibleCount() > 0; }

    // Evaluates all combinations and returns the best (feasible && finished) one.
    // If no valid combination exists, returns an empty Candidate with feasible=false
    // (no exception is thrown, so callers check hasFeasible() or the result's feasible).
    Candidate optimize() {
        candidates_.clear();
        evaluated_ = skipped_ = crashed_ = 0;

        enumerate();

        // Sort so smaller scores come first (valid+finished first). -- stage 1 (analytic) ranking.
        std::sort(candidates_.begin(), candidates_.end(),
                  [this](const Candidate& a, const Candidate& b) { return rankLess(a, b); });

        // ---- stage 2: refine the top topK_ with DRAMsim3 ----
        // candidates_ is already sorted best-first, so "top-K" = the first K feasible ones.
        // Refinement only makes the DRAM term more accurate (energy/efficiency update), so
        // re-sort afterwards with the same comparator to reflect any reordering in the report.
        if (topK_ > 0 && dramEnabled_) {
            int refined = 0;
            for (Candidate& c : candidates_) {
                if (!(c.feasible && c.finished)) continue;   // only rankable candidates.
                if (refined >= topK_) break;
                refineCandidate(c);                          // re-sim (DramSim3) + mergeDram
                ++refined;
            }
            std::sort(candidates_.begin(), candidates_.end(),
                      [this](const Candidate& a, const Candidate& b) { return rankLess(a, b); });
        }

        for (const Candidate& c : candidates_)
            if (c.feasible && c.finished) return c;
        return Candidate{};
    }

    // Human-readable result report: search statistics, top-N table, and a detailed
    // per-component breakdown of the best combination.
    void printReport(std::ostream& os, int topN = 10) const {
        os << "================== SRAM Optimizer Report ==================\n";
        os << "  layer            : " << layer_.layerName
           << "  (M=" << layer_.gemmDimenion.dimensionM
           << ", N=" << layer_.gemmDimenion.dimensionN
           << ", K=" << layer_.gemmDimenion.dimensionK << ")\n";
        os << "  mode             : " << (isBurst() ? "ToroidalDual (input + weight + output)"
                                                    : "Grouped/Toroidal (input + weight + output)") << "\n";
        os << "  objective        : " << objectiveName(objective_) << "\n";
        // Note: "feasible" here is candidates_.size(), i.e. combinations that ran at some
        // cd; it may include unfinished ones (the table below only lists finished ones).
        os << "  combinations     : evaluated=" << evaluated_
           << ", skipped(size)=" << skipped_
           << ", infeasible(run)=" << crashed_
           << ", feasible=" << candidates_.size() << "\n";
        // cd is a search axis swept per (in,w,out). The number of cd points per combination
        // is the simulation cost, so report it too (can be bounded with OPT_CD_MIN/OPT_CD_MAX).
        os << "  cd search        : plans=" << cdEvaluated_
           << ", skipped(size)=" << cdSkipped_
           << ", simulated=" << (cdEvaluated_ - cdSkipped_)
           << ", early-stop=" << cdEarlyStopped_
           << (arraySpec_.hasManualCd ? "  (manual cd: no search)"
                                      : (cdFull_ ? "  (OPT_CD_FULL: exhaustive)" : "")) << "\n";
        os << "  DRAM energy      : analytic (read=" << Dram::readEnergyPerAccess
           << " pJ/access, write=" << Dram::writeEnergyPerAccess
           << " pJ/access, 4096-bit access)  [filled=" << dramMergedCount_ << "]\n";
        // Emit the extra line only when stage 2 ran (so OPT_TOPK=0 output stays
        // byte-identical to the analytic-only baseline).
        if (dramMeasuredCount_ > 0)
            os << "  top-K refine     : K=" << topK_ << ", refined=" << dramMeasuredCount_
               << ", dram=DRAMsim3 (measured)\n";

        if (candidates_.empty()) {
            os << "  result           : NO FEASIBLE COMBINATION\n";
            os << "==========================================================\n";
            return;
        }

        const int col = 18;   // SRAM name column width.
        const int ruleLen = 4 + col * 3 + 14 + 18 + 18;

        // Raise the minimum number of rows to K so every refined top-K candidate appears
        // in the table (measured candidates are not cut off even if OPT_TOPK > topN). With
        // dramMeasuredCount_==0 (analytic-only) effTopN==topN, so OPT_TOPK=0 output stays
        // identical to the baseline.
        const int effTopN = std::max(topN, dramMeasuredCount_);

        os << "\n  Top " << std::min<int>(effTopN, (int)feasibleCount())
           << " combinations (best first):\n";
        os << "  " << std::string(ruleLen, '-') << "\n";
        os << "  " << std::left << std::setw(4) << "#"
           << std::setw(col) << "input";
        os << std::setw(col) << "weight"
           << std::setw(col) << "output"
           << std::right << std::setw(14) << "cycles"
           << std::setw(18) << "energy(pJ)*"
           << std::setw(18) << "TOPS/W/mm^2" << "\n";
        os << "  " << std::string(ruleLen, '-') << "\n";

        int shown = 0;
        for (const Candidate& c : candidates_) {
            if (shown >= effTopN) break;
            if (!(c.feasible && c.finished)) continue;
            os << "  " << std::left << std::setw(4) << (shown + 1)
               << std::setw(col) << shortName(c.inputSram);
            os << std::setw(col) << shortName(c.weightSram)
               << std::setw(col) << shortName(c.outputSram)
               << std::right << std::setw(14) << c.cycles
               << std::setw(18) << std::fixed << std::setprecision(1) << c.totalEnergy
               << std::setw(18) << std::setprecision(6) << c.topsPerWattPerMm2;
            // Mark measured rows only when refinement ran (no marker with OPT_TOPK=0 -> identical to baseline).
            if (dramMeasuredCount_ > 0 && c.dramMeasured) os << "  [DRAMsim3]";
            os << "\n";
            ++shown;
        }
        os << "  " << std::string(ruleLen, '-') << "\n";
        if (dramMeasuredCount_ > 0) {
            os << "  * energy / TOPS/W/mm^2 are DRAM-inclusive (compute + SRAM + DRAM;\n";
            os << "    [DRAMsim3] rows use measured DRAM energy, others analytic)\n";
        } else {
            os << "  * energy / TOPS/W/mm^2 are DRAM-inclusive (compute + SRAM + DRAM analytic)\n";
        }

        // candidates_ is sorted with valid+finished first, so front() is the best when any exists.
        const Candidate& best = candidates_.front();
        os << "\n  BEST combination:\n";
        os << "    input          : " << baseName(best.inputSram) << "\n";
        os << "    weight         : " << baseName(best.weightSram) << "\n";
        os << "    output         : " << baseName(best.outputSram) << "\n";
        os << "    cd (searched)  : " << best.cd << "\n";
        os << "    cycles         : " << best.cycles << "\n";

        // ---- Per-component breakdown (dynamic/leakage, read/write split) ----
        os << std::fixed << std::setprecision(2);
        os << "\n    -- Array --\n";
        os << "      active ops    : " << best.arrayActiveCount << "\n";
        os << "      busy cycles   : " << best.arrayBusyCycles << "\n";
        os << "      useful cycles : " << best.usefulStreamCycles
           << "  (stream w/ data; excludes fill/drain/zero-pad)\n";
        os << "      dynamic energy: " << best.arrayDynamicEnergy << " pJ\n";
        os << "        clk  (busy) : " << best.arrayClockEnergy << " pJ\n";
        os << "        data (usefl): " << best.arrayDataEnergy  << " pJ\n";
        os << "      leakage energy: " << best.arrayLeakageEnergy << " pJ\n";
        os << "      total energy  : " << best.arrayEnergy        << " pJ\n";
        os << "      area          : " << best.arrayArea          << " um^2\n";

        printBufDetail(os, "Input buffer  (SRAM)", best.inBufStat);
        printBufDetail(os, "Weight buffer (SRAM)", best.wBufStat);
        printBufDetail(os, "Output buffer (SRAM)", best.outBufStat);

        if (best.dramMeasured) {
            os << "\n    -- DRAM (DRAMsim3 measured, dynamic: read+write+act) --\n";
            os << "      accesses r / w: " << best.dramReads << " / "
               << best.dramWrites << "  (4096-bit, analytic count)\n";
            os << "      DRAMsim3 txns : " << best.dramReadsDone << " r / "
               << best.dramWritesDone << " w  (done)\n";
            os << "      read energy   : " << best.dramReadEnergy    << " pJ\n";
            os << "      write energy  : " << best.dramWriteEnergy   << " pJ\n";
            os << "      act energy    : " << best.dramActEnergy     << " pJ\n";
            os << "      DRAM dynamic  : " << best.dramEnergy        << " pJ  (read+write+act; bg/refresh via static)\n";
            os << "      DRAM static   : " << best.dramStaticEnergy  << " pJ  (0.65W bg+refresh x busy time)\n";
            os << "      bandwidth     : " << best.dramBandwidthGbps << " GB/s\n";
            os << "      power         : " << best.dramPowerMw       << " mW\n";
        } else {
            os << "\n    -- DRAM (analytic: constant per 4096-bit access) --\n";
            os << "      accesses r / w: " << best.dramReads << " / "
               << best.dramWrites << "  (4096-bit)\n";
            os << "      read energy   : " << best.dramReadEnergy    << " pJ\n";
            os << "      write energy  : " << best.dramWriteEnergy   << " pJ\n";
            os << "      DRAM dynamic  : " << best.dramEnergy        << " pJ  (read+write)\n";
            os << "      DRAM static   : " << best.dramStaticEnergy  << " pJ  (0.65W bg+refresh x busy time)\n";
        }

        os << "\n    -- Energy (pJ) --\n";
        os << "      memory (SRAM) : " << best.memoryEnergy          << "\n";
        os << "        dynamic     : " << best.memoryDynamicEnergy() << "\n";
        os << "        leakage     : " << best.memoryLeakageEnergy() << "\n";
        os << "      array         : " << best.arrayEnergy           << "\n";
        os << "        dynamic     : " << best.arrayDynamicEnergy    << "\n";
        os << "        leakage     : " << best.arrayLeakageEnergy    << "\n";
        os << "      compute+SRAM  : " << best.energyComputeSram     << "\n";
        os << (best.dramMeasured ? "      DRAM dyn(sim3) : " : "      DRAM dyn(anlyt): ")
           << best.dramEnergy << "\n";
        os << "      DRAM static   : " << best.dramStaticEnergy
           << "  (0.65W bg+refresh)\n";
        os << "      DRAM total    : " << (best.dramEnergy + best.dramStaticEnergy)
           << "  (" << std::setprecision(1) << best.dramFraction() * 100.0 << "% of total)"
           << std::setprecision(2) << "\n";
        os << "      TOTAL         : " << best.totalEnergy           << "\n";

        os << "\n    -- Area (um^2) --\n";
        os << "      array         : " << best.arrayArea            << "\n";
        os << "      SRAM total    : " << best.sramArea             << "\n";
        os << "      TOTAL         : " << (best.arrayArea + best.sramArea) << "\n";

        os << "\n    -- Efficiency (" << (best.dramMerged ? "DRAM-inclusive"
                                                          : "compute + SRAM only") << ") --\n";
        os << "      throughput    : " << std::setprecision(6) << best.tops << " TOPS\n";
        os << "      energy eff    : " << best.topsPerWatt        << " TOPS/W\n";
        os << "      area-norm eff : " << best.topsPerWattPerMm2   << " TOPS/W/mm^2  (array+SRAM)\n";
        os << "==========================================================\n";
    }

    // Prints the read/write/dynamic/leakage breakdown of one SRAM buffer.
    static void printBufDetail(std::ostream& os, const char* name, const BufStat& s) {
        os << "\n    -- " << name << " --\n";
        os << "      reads / writes: " << s.readCount << " / " << s.writeCount << "\n";
        os << "      read / write B: " << s.readBytes << " / " << s.writeBytes << "\n";
        os << "      read energy   : " << s.readEnergy     << " pJ\n";
        os << "      write energy  : " << s.writeEnergy    << " pJ\n";
        os << "      dynamic energy: " << s.dynamicEnergy() << " pJ\n";
        os << "      leakage energy: " << s.leakageEnergy  << " pJ\n";
        os << "      total energy  : " << s.totalEnergy()  << " pJ\n";
    }

private:
    // Finalizes the spec for the layer before building the config (ToroidalDual & M>1 -> Toroidal ReuseB).
    // Called from the constructor initializer list after arraySpec_/layer_ and before config_.
    static std::shared_ptr<const SystolicArrayConfig>
    makeResolvedConfig(ArrayConfigSpec& spec, const Layer& layer) {
        spec.resolveBurstForLayerM(layer.gemmDimenion.dimensionM);
        return spec.makeConfig();
    }

    // ----------------------------- Members ---------------------------------
    // Member declaration order matters: arraySpec_ and layer_ must be initialized before config_.
    ArrayConfigSpec arraySpec_;
    Layer           layer_;
    // cd-independent geometry placeholder. Used by the bandwidth guard (loadSrams) and cd auto-derivation.
    std::shared_ptr<const SystolicArrayConfig> config_;

    // Per-role SRAM candidates: bandwidth pinned to the proper value derived from the
    // array config, varying (at most) in capacity.
    // (Previously every (capacity x bandwidth) combination was enumerated; now the
    //  bandwidth is guarded to the proper value. With the current SramSelector each list
    //  holds a single 256 KB row, so no capacity sweep actually happens.)
    std::vector<SramOption> inputSrams_;   // input  proper-band candidates.
    std::vector<SramOption> weightSrams_;  // weight proper-band candidates.
    std::vector<SramOption> outputSrams_;  // output proper-band candidates.

    // Bounds on the cd search axis (0 = unset), given via OPT_CD_MIN / OPT_CD_MAX. The cd
    // axis multiplies the number of simulations per combination, so batch runs short on
    // time can use these knobs to reduce the number of points (same nature as
    // OPT_CAP_MIN/OPT_CAP_MAX).
    int cdMinOverride_ = 0;
    int cdMaxOverride_ = 0;
    // OPT_CD_FULL=1 -> disable the cd-descent early stop (unimodality assumption) and simulate the whole range.
    bool cdFull_ = false;

    std::vector<Candidate> candidates_;
    Objective objective_ = Objective::TotalEnergy;
    std::string workDir_;                  // Temp directory where child simulations write result/trace files.

    // Statistics.
    int evaluated_ = 0;   // number of (in,w,out) combinations attempted.
    int skipped_   = 0;   // combinations that failed the capacity pre-check at every cd (or had cdCeil < 1).
    int crashed_   = 0;   // combinations whose simulation aborted (infeasible) at every cd that was tried.
    // cd search-axis statistics (separate from the (in,w,out) stats; one combination runs several cd points).
    int cdEvaluated_ = 0; // number of (combination x cd) plans built.
    int cdSkipped_   = 0; // of those, filtered by the capacity pre-check (usually psumTile > output face).
    int cdEarlyStopped_ = 0; // combinations whose cd descent was stopped early by dominance.
    int dramMergedCount_ = 0;   // candidates with DRAM energy merged (analytic included; counts stage-1 runs).
    int dramMeasuredCount_ = 0; // candidates measured (refined) with stage-2 DRAMsim3.
    int dramFailedCount_ = 0;   // candidates whose DRAMsim3 run/parse failed.

    // ----------------------------- top-K refine knob -----------------------
    // Two-stage optimization: stage 1 cheaply ranks all candidates with the analytic DRAM
    // model, stage 2 re-simulates only the top topK_ with DRAMsim3 and updates them with
    // measured DRAM energy.
    //   OPT_TOPK=0  => skip stage 2 (pure analytic, legacy behavior).
    //   dramEnabled_==false => (binary/config missing or OPT_DRAMSIM=0) stage 2 is a no-op.
    int topK_ = 8;

    // ----------------------------- DRAMsim3 integration --------------------
    // Runs each refined candidate's trace through DRAMsim3 to obtain measured DRAM energy.
    // Supports the same overrides as run_pipeline.sh (DRAMSIM_DIR/DRAM_CFG).
    bool        dramEnabled_ = false;   // Binary/config present and not disabled by the user.
    std::string dramsimDir_;            // DRAMsim3 root (absolute path).
    std::string dramMainBin_;          // Absolute path of dramsim3main.
    std::string dramCfgRel_;           // Config path relative to the DRAMsim3 root.
    std::string dramLabel_ = "HBM2_8Gb_x128";   // DRAM_LABEL override; stored but not otherwise used here.

    // DRAMsim3 aggregate result for one trace (summed over channels).
    struct DramResult {
        bool   ok = false;
        // Energies in pJ as reported by DRAMsim3; totalEnergy = read + write + act (dynamic only).
        double totalEnergy = 0.0, readEnergy = 0.0, writeEnergy = 0.0, actEnergy = 0.0;
        double bandwidthGbps = 0.0, powerMw = 0.0;
        long   readsDone = 0, writesDone = 0;
    };
    // Trace-content hash -> result. Avoids re-running DRAMsim3 on identical traces.
    std::unordered_map<uint64_t, DramResult> dramCache_;

    // ----------------------------- SRAM loading ----------------------------
    // Pins the bandwidth to the proper value derived from the array config (guard) and
    // builds, per role, the candidate list that differs only in capacity. The optimizer
    // sweeps only this capacity axis. Throws if any role ends up empty.
    void loadSrams(const std::string& sramDir) {
        // All roles share the same library (SramSelector.hpp).
        SramLibrary lib(sramDir);

        // Capacity sweep range limit (KB). Default is unbounded (legacy behavior).
        //   OPT_CAP_MIN / OPT_CAP_MAX (KB) drop capacities outside [min,max] from the
        //   candidates entirely. (Historically, with a multi-capacity library, the best
        //   combinations for objective=tops clustered in 256~1024 KB, so batch runs used
        //   OPT_CAP_MIN=256 OPT_CAP_MAX=2048 to shrink 8^3 -> 4^3 combinations.)
        int capMinKB = 0, capMaxKB = INT_MAX;
        if (const char* v = std::getenv("OPT_CAP_MIN")) capMinKB = std::max(0, std::atoi(v));
        if (const char* v = std::getenv("OPT_CAP_MAX")) { int m = std::atoi(v); if (m > 0) capMaxKB = m; }

        auto toOptions = [&](const std::vector<SramFile>& v) {
            std::vector<SramOption> out;
            for (const SramFile& f : v) {
                const int capKB = static_cast<int>(f.spec.capacityBytes / 1024);
                if (capKB < capMinKB || capKB > capMaxKB) continue;
                out.push_back({ f.path, f.spec });
            }
            return out;
        };

        auto candidatesFor = [&](SramRole role) {
            return toOptions(properSramsPerCapacity(lib.forRole(*config_, role),
                                                    *config_, role));
        };

        inputSrams_  = candidatesFor(SramRole::Input);
        weightSrams_ = candidatesFor(SramRole::Weight);
        outputSrams_ = candidatesFor(SramRole::Output);

        // Report which role is empty and why. Without "required bandwidth / source actually
        // read / bands present in that source" one cannot tell whether to fix the CSV or the guard.
        auto requireNonEmpty = [&](const std::vector<SramOption>& v, SramRole role) {
            if (!v.empty()) return;
            const int needBits = requiredSramBandwidthBytes(*config_, role) * 8;
            const std::string src = cactiSourceFor(lib.dir(), *config_, role);
            std::string bands;
            for (const SramFile& f : lib.forRole(*config_, role))
                bands += (bands.empty() ? "" : ", ") + std::to_string(f.spec.dataWidthBytes * 8);
            throw std::runtime_error(
                std::string("Optimizer: ") + sram_select_detail::roleName(role) +
                " no SRAM satisfies the array bandwidth -- required band >= " +
                std::to_string(needBits) + " bits, bands in source \"" + src +
                "\": [" + (bands.empty() ? "(none)" : bands) + "]" +
                ", capacity filter " + std::to_string(capMinKB) + "~" + std::to_string(capMaxKB) + "KB");
        };
        requireNonEmpty(inputSrams_,  SramRole::Input);
        requireNonEmpty(weightSrams_, SramRole::Weight);
        requireNonEmpty(outputSrams_, SramRole::Output);
    }

    // cd search-range knobs (OPT_CD_MIN / OPT_CD_MAX) and OPT_CD_FULL. Unset -> 0, meaning
    // the default range [naturalMinCd(), deriveContractingDimensionSize(...)] is used as is.
    void readCdKnobs() {
        if (const char* v = std::getenv("OPT_CD_MIN")) cdMinOverride_ = std::max(0, std::atoi(v));
        if (const char* v = std::getenv("OPT_CD_MAX")) cdMaxOverride_ = std::max(0, std::atoi(v));
        if (const char* v = std::getenv("OPT_CD_FULL")) cdFull_ = (std::atoi(v) != 0);
    }

    // ----------------------------- Combination enumeration -----------------
    // Bandwidth is already pinned to the proper value, so only the capacity axes are swept
    // (full cartesian product of the three role lists).
    void enumerate() {
        for (const SramOption& in  : inputSrams_)
        for (const SramOption& w   : weightSrams_)
        for (const SramOption& out : outputSrams_)
            evaluate(in, w, out);
    }

    // Lower bound of the cd search axis (= the array's "natural minimum" streaming length).
    // If cd is shorter than the array fill depth (cycles the stream needs to fill the
    // pipeline), utilization collapses and it cannot win, so powers of two are scanned
    // from there upward.
    // (The lower bound is only a heuristic -- it can be overridden with OPT_CD_MIN.)
    int naturalMinCd() const {
        switch (config_->kind()) {
            // For ToroidalDual, cd counts stream steps, not N elements (one step consumes
            // totalVectorPeCol elements of N -- Scheduler::buildTilesToroidalDual).
            // To keep the old "minimum N-axis length = systolicArrayRowDimension" heuristic
            // it must be converted to steps (ceil division). Otherwise the lower bound is
            // inflated by the port count and small-cd candidates (= small psum tiles ->
            // large nb) are never considered.
            case ConfigKind::ToroidalDual:
                return std::max(1, (config_->systolicArrayRowDimension +
                                    config_->totalVectorPeCol - 1) / config_->totalVectorPeCol);
            case ConfigKind::Grouped:
            case ConfigKind::Toroidal:
                switch (config_->dataflow) {
                    case Dataflow::ReuseB: return config_->systolicArrayRowDimension;
                    case Dataflow::ReuseA: return config_->systolicArrayColDimension;
                    case Dataflow::ReuseC: return 1;   // cd unused.
                }
                break;
        }
        return 1;
    }

    // List of cd values to try for this candidate. cdCeil is the "feasibility upper bound"
    // from deriveContractingDimensionSize, not the optimum (see ContractingDimension.hpp).
    //
    // Why search: in ReuseB, cd -> tileM -> psumTile = cd*tileN*psumBytes, so growing cd to
    // the ceiling inflates psumTile in proportion to the input face capacity (measured: exactly
    // 4x), and gTotal = outputFace/psumTile starves nb, multiplying A's DRAM traffic.
    // I.e. cd is a free variable trading utilization (array) against nb (DRAM + output SRAM).
    // That trade-off is hard to solve in closed form (it needs SRAM per-access energy), and
    // the Optimizer already runs a full simulation per candidate, so choosing it "by
    // measurement" here is the cheapest and most accurate approach.
    // Candidates: hi (= cdCeil after OPT_CD_MAX and de-skew rounding) plus every power of two
    // in [lo, hi/2].
    // Returned in descending order (= cheapest simulations first, and the order in which
    // the early stop is valid).
    std::vector<int> cdCandidates(int cdCeil) const {
        std::vector<int> v;
        if (cdCeil < 1) return v;
        // A manual cd override is honored as is (no search).
        if (arraySpec_.hasManualCd) { v.push_back(cdCeil); return v; }

        int lo = std::max(1, std::min(naturalMinCd(), cdCeil));
        int hi = cdCeil;
        if (cdMinOverride_ > 0) lo = std::max(1, std::min(cdMinOverride_, cdCeil));
        if (cdMaxOverride_ > 0) hi = std::min(hi, cdMaxOverride_);
        if (hi < lo) hi = lo;

        // For Grouped ReuseA/ReuseB, cd is the AccSram depth and must satisfy the slot
        // constraint: depth % accDeskewSpan == 0 and at least 2 slots (1 de-skew flush + 1 data).
        //   If span is a power of two (true for all current Grouped configs), power-of-two
        //   candidates >= 2*span are automatically valid, so in practice only hi gets rounded.
        //   If no valid depth exists, this candidate cannot build a panel -> empty list (skip).
        const bool deskew = config_->hasDeskewFlush() && config_->accDeskewSpan > 1;
        if (deskew) {
            hi = config_->roundDownFreeDimensionSize(hi);
            if (hi < 1) return v;
            if (lo > hi) lo = hi;   // If the lower-bound heuristic exceeds the ceiling, lower it.
        }

        v.push_back(hi);                                   // The upper bound is always included.
        for (int p = 1; p <= hi / 2; p *= 2)               // Powers of two, collected ascending.
            if (p >= lo && (!deskew || config_->isValidFreeDimensionSize(p)))
                v.push_back(p);
        std::sort(v.begin(), v.end(), std::greater<int>());
        v.erase(std::unique(v.begin(), v.end()), v.end()); // Dedupe (defensive; p <= hi/2 never equals hi).
        return v;
    }

    // Evaluates one (input, weight, output) combination. cd is swept inside as a search
    // axis and only the "best cd" per combination is kept as a candidate (one ranking-table
    // row = one SRAM combination).
    //
    // Statistics (evaluated_/skipped_/crashed_) are counted per (in,w,out) as before -- the
    // cd axis is reported by separate counters (cdEvaluated_/cdSkipped_/cdEarlyStopped_).
    void evaluate(const SramOption& in, const SramOption& w, const SramOption& out) {
        ++evaluated_;

        // The cd ceiling depends on this candidate's input/weight SRAM capacities (differs per
        // candidate). cdCeil<1 means not even a cd==1 minimal tile fits, so skip the combination.
        const int cdCeil = arraySpec_.hasManualCd
            ? arraySpec_.contractingDimensionSize
            : deriveContractingDimensionSize(*config_, layer_,
                                             in.spec.faceCapacityBytes(),
                                             w.spec.faceCapacityBytes());
        if (cdCeil < 1) { ++skipped_; return; }

        Candidate best;
        bool haveBest = false, anyFit = false, anyRan = false;
        long prevDramReads = LONG_MAX;   // DRAM read count of the previous (= next larger) cd.

        // cd is in descending order. The early stop (see dominance below) requires this order.
        for (const int cd : cdCandidates(cdCeil)) {
            // Build a candidate-specific config/plan for this cd. Pass this candidate's output
            // SRAM face capacity so the Scheduler splits the accumulation axis into G
            // partial-sum tile blocks (output capacity differs per candidate, so the plan does too).
            std::shared_ptr<const SystolicArrayConfig> cfg = arraySpec_.makeConfig(cd);
            SchedulePlan plan = Scheduler(layer_, cfg, out.spec.faceCapacityBytes(),
                                          w.spec.faceCapacityBytes()).plan();
            ++cdEvaluated_;

            // Cheap necessary-condition pre-check (a single tile does not even fit) -> saves the fork.
            // Large cds whose psumTile exceeds the output face are pruned here (= cd-axis pruning).
            if (!passesQuickCheck(plan, *cfg, in.spec, w.spec, out.spec)) { ++cdSkipped_; continue; }
            anyFit = true;

            Candidate cand;
            cand.inputSram  = in.path;
            cand.weightSram = w.path;
            cand.outputSram = out.path;
            cand.cd         = cd;

            runSim(cfg, plan, in.spec, w.spec, out.spec, cand);
            if (!cand.feasible) continue;
            anyRan = true;

            const long dramReads = cand.dramReads;
            const bool better    = (!haveBest || rankLess(cand, best));
            if (better) { best = std::move(cand); haveBest = true; }

            // Early stop (dominance): if halving cd (a) made the score worse and (b) did not
            // reduce DRAM traffic, shrinking further cannot help -- the only gain from a smaller
            // cd is nb headroom (= less DRAM), which has already saturated, while the loss
            // (utilization -> cycles/array energy) is monotone in cd.
            // Measured (q_proj, in128/w128/out2048): DRAM reads drop 3.61M -> 2.41M from cd
            // 2048 -> 1024 and stay flat at 2.41M for 512/256, while utilization keeps falling
            // 94% -> 89% -> 80% -> 67%. I.e. the score is unimodal in cd.
            // When the unimodality assumption is not trusted (validating a new geometry/dataflow,
            // brute-force comparison via Makefile.exp_cd, etc.), set OPT_CD_FULL=1 to disable it
            // and sweep the whole range.
            if (!better && dramReads >= prevDramReads && !cdFull_) { ++cdEarlyStopped_; break; }
            prevDramReads = dramReads;
        }

        // No cd fits the faces -> capacity skip. Some fit but all aborted -> infeasible.
        if (!anyFit) { ++skipped_; return; }
        if (!anyRan) { ++crashed_; return; }
        candidates_.push_back(std::move(best));
    }

    // Checks only cheap "must hold" conditions, e.g. a single tile fits in a single buffer face.
    // (Output-buffer modulo / accumulation-group conditions depend on model formulas and are
    //  left to the child run.)
    bool passesQuickCheck(const SchedulePlan& plan, const SystolicArrayConfig& cfg,
                          const SramSpec& in, const SramSpec& w, const SramSpec& out) const {
        if (!plan.tilesA.empty() &&
            plan.tilesA.front().shape.sizeBytes() > in.faceCapacityBytes()) return false;
        if (!plan.tilesB.empty() &&
            plan.tilesB.front().shape.sizeBytes() > w.faceCapacityBytes()) return false;
        if (!plan.tilesC.empty()) {
            const TileC& c = plan.tilesC.front();
            const int needOut = (cfg.dataflow == Dataflow::ReuseC)
                                    ? c.shape.sizeBytes()        // ReuseC: complete TileC
                                    : c.partialSumSizeBytes();   // RMW: one partial-sum tile
            if (needOut > out.faceCapacityBytes()) return false;
        }
        return true;
    }

    // ----------------------------- Child simulation ------------------------
    // Simulates one (combination, cd) point in a fork()ed child. If the child exits normally
    // (and the result payload parses), the candidate is marked feasible; on signal / abnormal
    // exit it is marked infeasible.
    // mode==Analytic (stage 1): the child disables tracing and only reports access counts ->
    //   the parent fills constant per-access DRAM energy via applyDramEnergy.
    // mode==DramSim3 (stage 2 refine): the child writes the real trace under workDir_ and
    //   cand.traceRel receives its relative path. DRAM energy is not filled here; the caller
    //   (refineCandidate) measures it with mergeDram.
    void runSim(const std::shared_ptr<const SystolicArrayConfig>& cfg,
                const SchedulePlan& plan,
                const SramSpec& in,
                const SramSpec& w, const SramSpec& out, Candidate& cand,
                DramEnergyMode mode = DramEnergyMode::Analytic) {
        int fds[2];
        if (pipe(fds) != 0) throw std::runtime_error("Optimizer: pipe() failed");

        pid_t pid = fork();
        if (pid < 0) throw std::runtime_error("Optimizer: fork() failed");

        if (pid == 0) {
            close(fds[0]);
            childRun(cfg, plan, in, w, out, fds[1], mode);   // Calls _exit() internally on success/exception.
            _exit(0);
        }

        // ---- parent ----
        // Read the whole payload (single line, well under 2 KB) until EOF.
        close(fds[1]);
        char buf[2048];
        ssize_t total = 0, n;
        while (total < (ssize_t)sizeof(buf) - 1 &&
               (n = read(fds[0], buf + total, sizeof(buf) - 1 - total)) > 0)
            total += n;
        buf[total > 0 ? total : 0] = '\0';
        close(fds[0]);

        int status = 0;
        waitpid(pid, &status, 0);

        const bool exitedOk = WIFEXITED(status) && WEXITSTATUS(status) == 0;
        if (exitedOk && parsePayload(buf, cand)) {
            cand.feasible = true;
            // Stage 1 fills analytic DRAM here. Stage 2 (DramSim3) only obtains the trace and
            // mergeDram measures DRAM, so nothing is filled here in that mode.
            if (mode == DramEnergyMode::Analytic)
                applyDramEnergy(cand);   // Fill analytic DRAM energy from the access counts received over the pipe.
        } else {
            cand.feasible = false;   // Aborted by assert/SIGFPE/exception etc. -> infeasible combination.
        }
    }

    // Deserializes the detailed result that childRun() sent over the pipe into a Candidate.
    // Field order must exactly match the serialization in childRun().
    // Returns true on success (all fields parsed). Resets totalEnergy to compute+SRAM only.
    static bool parsePayload(const char* text, Candidate& cand) {
        std::istringstream is(text);
        auto readBuf = [&is](BufStat& b) {
            is >> b.readCount >> b.writeCount >> b.readBytes >> b.writeBytes
               >> b.readEnergy >> b.writeEnergy >> b.leakageEnergy;
        };

        int fin = 0;
        is >> fin >> cand.cycles
           >> cand.arrayActiveCount
           >> cand.arrayBusyCycles >> cand.usefulStreamCycles
           >> cand.arrayClockEnergy >> cand.arrayDataEnergy
           >> cand.arrayDynamicEnergy
           >> cand.arrayLeakageEnergy >> cand.arrayArea >> cand.sramArea;
        readBuf(cand.inBufStat);
        readBuf(cand.wBufStat);
        readBuf(cand.outBufStat);
        is >> cand.dramReads >> cand.dramWrites
           >> cand.tops >> cand.topsPerWatt >> cand.topsPerWattPerMm2
           >> cand.traceRel;   // Last token: trace path relative to workDir_.

        if (!is) return false;

        cand.finished     = (fin != 0);
        cand.arrayEnergy  = cand.arrayDynamicEnergy + cand.arrayLeakageEnergy;
        cand.memoryEnergy = cand.memoryDynamicEnergy() + cand.memoryLeakageEnergy();
        // DRAM is not known yet. Compute+SRAM only for now; applyDramEnergy()/mergeDram() add DRAM.
        cand.energyComputeSram = cand.memoryEnergy + cand.arrayEnergy;
        cand.totalEnergy       = cand.energyComputeSram;
        return true;
    }

    // Child-process body: builds the array/DRAM/SRAM buffers, runs the simulation and writes
    // a single-line, space-separated result payload to writeFd. Never returns: _exit(0) on
    // success, _exit(2) on a C++ exception (asserts/signals terminate the child directly).
    void childRun(const std::shared_ptr<const SystolicArrayConfig>& config,
                  const SchedulePlan& plan,
                  const SramSpec& in,
                  const SramSpec& w, const SramSpec& out, int writeFd,
                  DramEnergyMode mode) {
        // Suppress the simulator's heavy stdout/stderr output and isolate result files in the temp dir.
        silenceStdio();
        if (!workDir_.empty()) { if (chdir(workDir_.c_str()) != 0) { /* ignored */ } }

        try {
            Array array(config,
                        arraySpec_.arraySwitchPower, arraySpec_.arrayInternalPower,
                        arraySpec_.arrayLeakagePower, arraySpec_.arrayArea,
                        arraySpec_.arrayClockPowerFraction);
            Dram dram;

            DoubleBuffer inputBuffer  = makeBuffer(in);
            DoubleBuffer weightBuffer = makeBuffer(w);
            DoubleBuffer outputBuffer = makeBuffer(out);

            SchedulePlan planCopy = plan;
            // Determined by mode: Analytic (stage 1) disables tracing and only reports access
            // counts; DramSim3 (stage 2 refine) writes the real trace under workDir_/results and
            // reports its relative path via traceFilePath() (the parent measures it with DRAMsim3).
            Simulator sim(planCopy, array, dram, inputBuffer, outputBuffer, weightBuffer,
                          mode);
            sim.checkSramSize();
            sim.run();

            const SimulationResults& r = sim.results();

            // Send the detailed result over the pipe. Order must exactly match parsePayload().
            //   finished cycles
            //   arrayActive busyCycles usefulCycles clkE dataE arrayDyn arrayLeak arrayArea sramArea
            //   <for each of input/weight/output: rc wc rB wB rE wE leakE>
            //   dramReads dramWrites
            //   tops topsPerWatt topsPerWattPerMm2
            //   traceRel
            const auto& ib = r.inputBuffer;
            const auto& wb = r.weightBuffer;
            const auto& ob = r.outputBuffer;

            std::ostringstream ss;
            ss.precision(10);
            auto emitBuf = [&ss, &r](const SimulationResults::AccessStats& s) {
                ss << ' ' << s.readCount << ' ' << s.writeCount
                   << ' ' << s.readBytes << ' ' << s.writeBytes
                   << ' ' << s.readEnergy << ' ' << s.writeEnergy
                   << ' ' << r.bufferLeakage(s);
            };
            ss << (r.finished ? 1 : 0) << ' ' << r.cycles
               << ' ' << r.arrayActiveCount
               << ' ' << r.arrayBusyCycles << ' ' << r.usefulStreamCycles
               << ' ' << r.arrayClockEnergy() << ' ' << r.arrayDataEnergy()
               << ' ' << r.arrayDynamicEnergy()
               << ' ' << r.arrayLeakageEnergy()
               << ' ' << r.arrayArea
               << ' ' << r.sramArea();
            emitBuf(ib);
            emitBuf(wb);
            emitBuf(ob);
            ss << ' ' << r.dram.readCount << ' ' << r.dram.writeCount
               << ' ' << r.tops() << ' ' << r.topsPerWatt() << ' ' << r.topsPerWattPerMm2()
               << ' ' << sim.traceFilePath()   // Relative path (inside workDir_). Contains no spaces.
               << '\n';

            const std::string payload = ss.str();
            ssize_t wr = write(writeFd, payload.data(), payload.size()); (void)wr;
            _exit(0);
        } catch (...) {
            _exit(2);   // Exception -> infeasible combination.
        }
    }

    // Redirects stdout/stderr of the current (child) process to /dev/null.
    static void silenceStdio() {
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            if (devnull > STDERR_FILENO) close(devnull);
        }
    }

    // Creates <system temp dir>/sram_optimizer_work and stores it in workDir_ (left empty on failure).
    void prepareWorkDir() {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path d = fs::temp_directory_path(ec) / "sram_optimizer_work";
        fs::create_directories(d, ec);
        if (!ec) workDir_ = d.string();
    }

    // ----------------------------- DRAMsim3 integration --------------------
    // Resolves the DRAMsim3 binary/config paths and decides availability (dramEnabled_).
    // Overrides (same as run_pipeline.sh):
    //   DRAMSIM_DIR (default DRAMsim3), DRAM_CFG (default configs/HBM2_8Gb_x128.ini),
    //   DRAM_LABEL  (default HBM2_8Gb_x128), OPT_TOPK (stage-2 size, default 8).
    //   OPT_DRAMSIM=0 disables DRAMsim3 (stage 2). Analytic DRAM energy is still applied in
    //   stage 1 regardless, so ranking remains DRAM-inclusive.
    void setupDramsim3() {
        namespace fs = std::filesystem;
        // Stage-2 refine size. Negative values clamp to 0 (= stage 2 disabled).
        if (const char* k = std::getenv("OPT_TOPK")) topK_ = std::max(0, std::atoi(k));

        const char* off = std::getenv("OPT_DRAMSIM");
        if (off && std::string(off) == "0") { dramEnabled_ = false; return; }

        const char* dirEnv = std::getenv("DRAMSIM_DIR");
        const char* cfgEnv = std::getenv("DRAM_CFG");
        const char* lblEnv = std::getenv("DRAM_LABEL");
        if (lblEnv && *lblEnv) dramLabel_ = lblEnv;

        std::error_code ec;
        dramsimDir_ = fs::absolute(dirEnv && *dirEnv ? dirEnv : "DRAMsim3", ec).string();
        dramCfgRel_ = (cfgEnv && *cfgEnv) ? cfgEnv : "configs/HBM2_8Gb_x128.ini";
        dramMainBin_ = (fs::path(dramsimDir_) / "build" / "dramsim3main").string();

        const bool haveBin = fs::exists(dramMainBin_, ec);
        const bool haveCfg = fs::exists(fs::path(dramsimDir_) / dramCfgRel_, ec);
        dramEnabled_ = haveBin && haveCfg;
        if (!dramEnabled_) {
            // Note: "energy excludes DRAM" is inaccurate -- only the measured (stage-2) DRAM
            // term is lost; stage-1 analytic DRAM energy is still included.
            std::cerr << "Optimizer: DRAMsim3 disabled ("
                      << (!haveBin ? "dramsim3main not found" : "config not found")
                      << ") -> energy excludes DRAM.\n";
        }
    }

    // Analytic DRAM model: multiplies the 4096-bit access counts received over the pipe
    // (dramReads/dramWrites) by constant per-access energies, adds DRAM static energy, and
    // updates the candidate's total energy/efficiency to DRAM-inclusive values (so ranking
    // becomes DRAM-inclusive).
    // Uses the same constants/access counts as the simulator in Analytic mode, so both tools
    // report the same DRAM energy.
    void applyDramEnergy(Candidate& cand) {
        cand.dramReadEnergy  = cand.dramReads  * Dram::readEnergyPerAccess;   // dramReads = 4096-bit accesses
        cand.dramWriteEnergy = cand.dramWrites * Dram::writeEnergyPerAccess;
        cand.dramEnergy      = cand.dramReadEnergy + cand.dramWriteEnergy;
        cand.dramMerged      = true;
        // (C) DRAM static (background+refresh): like SRAM leakage, scaled by total simulated
        // cycles. W x cycles x s/cycle = J, x1e12 = pJ.
        cand.dramStaticEnergy = Candidate::dramStaticPowerW * cand.cycles
                              * Candidate::clockPeriodS * 1e12;
        cand.totalEnergy     = cand.energyComputeSram + cand.dramEnergy + cand.dramStaticEnergy;
        ++dramMergedCount_;

        // Recompute efficiency with the DRAM-inclusive energy.
        //   opCount = 2*M*N*K (fixed per layer); tops is cycle-based and therefore unchanged.
        //   TOPS/W = 1e-12 x ops/J = ops/pJ (Tera and pico cancel), so opCount / totalEnergy[pJ].
        const double opCount = 2.0
            * static_cast<double>(layer_.gemmDimenion.dimensionM)
            * static_cast<double>(layer_.gemmDimenion.dimensionN)
            * static_cast<double>(layer_.gemmDimenion.dimensionK);
        cand.topsPerWatt = (cand.totalEnergy > 0.0) ? opCount / cand.totalEnergy : 0.0;
        const double areaMm2 = cand.totalAreaMm2();
        cand.topsPerWattPerMm2 = (areaMm2 > 0.0) ? cand.topsPerWatt / areaMm2 : 0.0;
    }

    // [stage 2 refine] Re-simulates one top candidate with DRAMsim3 and updates it with
    // measured DRAM energy. cfg/plan must be rebuilt "exactly as" in evaluate() so the trace
    // corresponds to stage 1 (same cd and same output-capacity-based Scheduler plan). After
    // rebuilding, re-sim in DramSim3 mode -> obtain the real trace, and mergeDram replaces the
    // DRAM term with the measured one.
    void refineCandidate(Candidate& c) {
        const SramSpec* in  = findSpec(c.inputSram,  inputSrams_);
        const SramSpec* w   = findSpec(c.weightSram, weightSrams_);
        const SramSpec* out = findSpec(c.outputSram, outputSrams_);
        if (!in || !w || !out) return;   // Cannot happen: paths come from stage 1.

        // cd is not "re-derived"; the value stage 1 searched and chose (c.cd) is reused.
        // When cd was a function of the (in,w) capacities re-derivation gave the same value,
        // but cd is now a search axis and cannot be recovered that way -- using c.cd is the only
        // way to keep the two stages from diverging.
        // (evaluate() is the only path that fills stage-1 candidates, so c.cd>=1 always holds.)
        const int cd = c.cd;
        if (cd < 1) { assert(false && "refineCandidate: stage-1 candidate has no cd"); return; }
        std::shared_ptr<const SystolicArrayConfig> cfg = arraySpec_.makeConfig(cd);
        SchedulePlan plan = Scheduler(layer_, cfg, out->faceCapacityBytes(),
                                      w->faceCapacityBytes()).plan();

        // Re-sim in DramSim3 mode -> c.traceRel receives the real trace path.
        // (parsePayload refills c's compute+SRAM terms identically to stage 1 and resets
        //  totalEnergy to the DRAM-exclusive value; mergeDram then adds measured DRAM.
        //  If mergeDram fails, the candidate is left with DRAM-exclusive totals.)
        runSim(cfg, plan, *in, *w, *out, c, DramEnergyMode::DramSim3);
        if (!(c.feasible && c.finished)) return;   // Normally always true since stage 1 passed.
        mergeDram(c);
    }

    // Runs one candidate's trace through DRAMsim3 to obtain measured DRAM energy and updates
    // the candidate's total energy/efficiency to DRAM-inclusive (measured) values.
    // Called only from refineCandidate.
    void mergeDram(Candidate& cand) {
        if (!dramEnabled_ || cand.traceRel.empty()) return;

        const std::string absTrace = workDir_ + "/" + cand.traceRel;
        const DramResult dr = dramForTrace(absTrace, cand.cycles);
        if (!dr.ok) { ++dramFailedCount_; return; }

        cand.dramMerged      = true;
        cand.dramMeasured    = true;   // Mark as stage-2 measured (distinguishes from analytic).
        cand.dramEnergy      = dr.totalEnergy;   // Dynamic component only (read+write+act).
        cand.dramReadEnergy  = dr.readEnergy;
        cand.dramWriteEnergy = dr.writeEnergy;
        cand.dramActEnergy   = dr.actEnergy;
        cand.dramBandwidthGbps = dr.bandwidthGbps;
        cand.dramPowerMw       = dr.powerMw;
        cand.dramReadsDone   = dr.readsDone;
        cand.dramWritesDone  = dr.writesDone;
        // Note: this candidate already incremented dramMergedCount_ once in stage 1 via
        // applyDramEnergy (analytic). Only the refine counter is bumped here to avoid double counting.
        ++dramMeasuredCount_;

        // Update to DRAM-inclusive total energy/efficiency (ranking becomes DRAM-inclusive).
        //   opCount = 2*M*N*K (fixed per layer); tops is cycle-based and therefore unchanged.
        //   DRAMsim3 dynamic (read+write+act) also excludes background/refresh, so add the static
        //   term (same definition as analytic -> stage-1/stage-2 consistent).
        cand.dramStaticEnergy = Candidate::dramStaticPowerW * cand.cycles
                              * Candidate::clockPeriodS * 1e12;
        cand.totalEnergy = cand.energyComputeSram + cand.dramEnergy + cand.dramStaticEnergy;
        const double opCount = 2.0
            * static_cast<double>(layer_.gemmDimenion.dimensionM)
            * static_cast<double>(layer_.gemmDimenion.dimensionN)
            * static_cast<double>(layer_.gemmDimenion.dimensionK);
        cand.topsPerWatt = (cand.totalEnergy > 0.0) ? opCount / cand.totalEnergy : 0.0;
        const double areaMm2 = cand.totalAreaMm2();
        cand.topsPerWattPerMm2 = (areaMm2 > 0.0) ? cand.topsPerWatt / areaMm2 : 0.0;
    }

    // Looks up the cache by trace-content hash; on a miss, actually runs DRAMsim3.
    // Only successful results are cached (and not when hashing failed, key==0).
    DramResult dramForTrace(const std::string& absTrace, long simCycles) {
        uint64_t key = 0;
        if (hashFile(absTrace, key)) {
            auto it = dramCache_.find(key);
            if (it != dramCache_.end()) return it->second;
        }
        const DramResult dr = runDramsim3(absTrace, simCycles);
        if (dr.ok && key != 0) dramCache_.emplace(key, dr);
        return dr;
    }

    // Runs DRAMsim3 and parses dramsim3.json.
    //   -c budget: the trace's last timestamp is <= 2*simCycles, so the budget is set generously.
    DramResult runDramsim3(const std::string& absTrace, long simCycles) {
        namespace fs = std::filesystem;
        DramResult dr;
        std::error_code ec;
        if (!fs::exists(absTrace, ec)) return dr;

        const std::string absOut = workDir_ + "/dram_out";
        fs::create_directories(absOut, ec);

        // memCycle = 2*newSimCycle. Add 10% + a fixed margin so all transactions drain.
        const long budget = static_cast<long>(simCycles * 2.2) + 100000;

        std::string cmd =
            "cd '" + dramsimDir_ + "' && ./build/dramsim3main '" + dramCfgRel_ + "'"
            " -c " + std::to_string(budget) +
            " -t '" + absTrace + "'"
            " -o '" + absOut + "' >/dev/null 2>&1";
        const int rc = std::system(cmd.c_str());
        if (rc != 0) return dr;

        return parseDramJson(absOut + "/dramsim3.json");
    }

    // Reads dramsim3.json (one object per channel) and sums the needed fields over channels.
    static DramResult parseDramJson(const std::string& jsonPath) {
        DramResult dr;
        std::ifstream ifs(jsonPath);
        if (!ifs) return dr;
        std::string json((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());
        if (json.empty()) return dr;

        dr.readEnergy    = sumJsonField(json, "read_energy");
        dr.writeEnergy   = sumJsonField(json, "write_energy");
        dr.actEnergy     = sumJsonField(json, "act_energy");
        // DRAM energy aggregates only the "dynamic (access-driven)" component: read + write + activation.
        // Background/standby/refresh (pre_stb/act_stb/refresh/self-refresh etc.) accumulate with time
        // in proportion to the -c budget (sim length) -- they keep growing in the idle tail after the
        // trace drains -- so their definition differs from the per-access analytic model and they
        // are not comparable -> excluded (static energy is modelled separately via dramStaticPowerW).
        // (Unifying this definition removes the cause of the Step 7.4 "wild mismatch" (45x, dominated by background).)
        dr.totalEnergy   = dr.readEnergy + dr.writeEnergy + dr.actEnergy;
        dr.bandwidthGbps = sumJsonField(json, "average_bandwidth");
        dr.powerMw       = sumJsonField(json, "average_power");
        dr.readsDone     = static_cast<long>(sumJsonField(json, "num_reads_done"));
        dr.writesDone    = static_cast<long>(sumJsonField(json, "num_writes_done"));
        dr.ok = true;
        return dr;
    }

    // Sums the values of a flat numeric field "key":<num> across all channels. Anchoring on the
    // leading quote prevents mixing with keys that share a suffix (e.g. refresh_read_energy).
    // Nested values ("key": { or [) are skipped.
    static double sumJsonField(const std::string& json, const std::string& key) {
        const std::string needle = "\"" + key + "\":";
        double sum = 0.0;
        size_t pos = 0;
        while ((pos = json.find(needle, pos)) != std::string::npos) {
            pos += needle.size();
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
            if (pos < json.size() && (json[pos] == '{' || json[pos] == '[')) continue;
            const char* start = json.c_str() + pos;
            char* end = nullptr;
            const double v = std::strtod(start, &end);
            if (end != start) sum += v;
        }
        return sum;
    }

    // Hashes the trace file contents with 64-bit FNV-1a (standard offset basis; file size is
    // not mixed in). Returns false if the file cannot be opened.
    static bool hashFile(const std::string& path, uint64_t& out) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        uint64_t h = 1469598103934665603ULL;   // FNV offset basis
        char chunk[65536];
        while (ifs) {
            ifs.read(chunk, sizeof(chunk));
            const std::streamsize got = ifs.gcount();
            for (std::streamsize i = 0; i < got; ++i) {
                h ^= static_cast<unsigned char>(chunk[i]);
                h *= 1099511628211ULL;          // FNV prime
            }
        }
        out = h;
        return true;
    }

    // ----------------------------- Buffer construction (= same as main.cpp) -----------
    // Builds a ping/pong DoubleBuffer from an SRAM characterization (face capacity, access
    // width, per-access energies in pJ, leakage in W, area in um^2).
    static DoubleBuffer makeBuffer(const SramSpec& s) {
        return DoubleBuffer(
            s.faceCapacityBytes(), s.dataWidthBytes,
            s.readEnergyPerAccess, s.writeEnergyPerAccess, s.totalLeakagePowerW(),
            s.totalAreaUm2());
    }

    // ----------------------------- Display helpers -------------------------
    // Filename component of an SRAM path ("-" if empty). Note: the " [band<bits>]" suffix, if
    // present, is kept since it is part of the filename string.
    static std::string baseName(const std::string& path) {
        if (path.empty()) return "-";
        return std::filesystem::path(path).filename().string();
    }

    // Short name for the table. Path format: "<...>/<file>.csv [band<bits>]".
    //   -> "<file>@<bits>"  e.g. "cacti_results/normal/normal_results.csv [band2048]"
    //      -> "normal_results@2048".
    static std::string shortName(const std::string& path) {
        std::string p = path, bandTag;
        const std::size_t lb = p.find(" [band");
        if (lb != std::string::npos) {
            std::size_t rb = p.find(']', lb);
            bandTag = p.substr(lb + 6, (rb == std::string::npos ? p.size() : rb) - (lb + 6));
            p = p.substr(0, lb);
        }
        std::string n = baseName(p);
        const std::string ext = ".csv";
        if (n.size() >= ext.size() &&
            n.compare(n.size() - ext.size(), ext.size(), ext) == 0)
            n.erase(n.size() - ext.size());
        if (!bandTag.empty()) n += "@" + bandTag;
        return n;
    }

    // Number of candidates that are both feasible and finished (i.e. rankable).
    int feasibleCount() const {
        int n = 0;
        for (const Candidate& c : candidates_)
            if (c.feasible && c.finished) ++n;
        return n;
    }

    // Ranking comparator (shared by stage 1/2): valid+finished first, then ascending objective score.
    bool rankLess(const Candidate& a, const Candidate& b) const {
        const bool aOk = a.feasible && a.finished;
        const bool bOk = b.feasible && b.finished;
        if (aOk != bOk) return aOk;
        return a.score(objective_) < b.score(objective_);
    }

    // Recovers the SramSpec from the role's candidate list by the candidate's SRAM path (for refine rebuild).
    static const SramSpec* findSpec(const std::string& path,
                                    const std::vector<SramOption>& opts) {
        for (const SramOption& o : opts)
            if (o.path == path) return &o.spec;
        return nullptr;
    }

    // Human-readable objective description for the report.
    static const char* objectiveName(Objective o) {
        switch (o) {
            case Objective::TotalEnergy:        return "minimize total energy (pJ)";
            case Objective::Cycles:             return "minimize cycles";
            case Objective::EnergyDelayProduct: return "minimize energy-delay product";
            case Objective::TopsPerWattPerMm2:  return "maximize TOPS/W/mm^2";
        }
        return "minimize total energy (pJ)";
    }
};
