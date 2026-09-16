#pragma once
#include <ostream>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include "Array.hpp"
#include "Dram.hpp"
#include "DoubleBuffer.hpp"

// SimulationResults: final per-layer results aggregated from each component
// (systolic array / DRAM / SRAM double buffers) after a simulation run, plus the
// derived throughput, energy-efficiency and area metrics (TOPS, TOPS/W, TOPS/W/mm^2).
//
// Units: energies are in pJ, areas in um^2, time in clock cycles (clockPeriod = 2 ns).
//
// The Simulator owns one instance, fills it via collect() when run() finishes, and prints it.
class SimulationResults {
public:
    // ---- Access/energy totals for one component (DRAM or an SRAM buffer) ----
    struct AccessStats {
        long   readCount    = 0;      // DRAM: # of 4096-bit read accesses. SRAM buffer: # of read transfers (addRead calls).
        long   writeCount   = 0;      // Same as readCount, for writes.
        long   readBytes    = 0;      // Bytes actually read.
        long   writeBytes   = 0;      // Bytes actually written.
        double readEnergy   = 0.0;    // pJ; per-access energy x # of word-width accesses (ceil(bytes / access width)).
        double writeEnergy  = 0.0;    // pJ; same model as readEnergy.
        double activationEnergy = 0.0; // (B) DRAM row-miss (ACT) energy, pJ. Always 0 for SRAM buffers.
        long   activations  = 0;       // (B) Estimated # of DRAM activations (diagnostic).
        double leakagePowerW = 0.0;   // Static leakage power (W), ping+pong macros. 0 for DRAM.
        double areaUm2       = 0.0;   // Physical area (um^2), ping+pong macros. 0 for DRAM (not area-normalized).

        // Access-driven dynamic energy (pJ). Includes the activation term for DRAM.
        double dynamicEnergy() const { return readEnergy + writeEnergy + activationEnergy; }
        // Same as dynamicEnergy(): leakage is NOT included here, because it depends on the
        // run length (see SimulationResults::bufferLeakage()).
        double totalEnergy() const { return dynamicEnergy(); }
    };

    long cycles = 0;   // Total simulated clock cycles (end-to-end, including memory stalls).

    // Whether DRAM energy is included in the totals/efficiency metrics.
    //   true only in Analytic mode  -> DRAM energy enters totalEnergy()/TOPS/W.
    //   false in DramSim3 mode      -> merge_results.py adds it later (avoids double counting).
    // Must be set before collect().
    bool dramEnergyIncluded = false;

    AccessStats dram;
    AccessStats inputBuffer;
    AccessStats weightBuffer;
    AccessStats outputBuffer;

    // ---- Array activity/power ----
    int    arrayActiveCount    = 0;   // # of tile ops (MAC) executed (reference only; not used for energy)
    long   arrayBusyCycles     = 0;   // array-busy cycles (fill+stream+drain) - basis for clock power
    long   usefulStreamCycles  = 0;   // stream cycles in which data actually flowed - basis for datapath power
    int    stationaryLoadCount = 0;   // # of stationary-operand (re)loads
    long   arrayMacUnits       = 0;   // # of physical MAC (multiplier) units = peak MACs per cycle
    double switchPowerMw       = 0.0; // Array switching power (mW)
    double internalPowerMw     = 0.0; // Array internal power (mW)
    double leakagePowerNw      = 0.0; // Array leakage power (nW)
    double clockPowerFraction  = 1.0; // P_clk fraction (.cfg arrayClockPowerFraction). 1.0 = flat charge over busy cycles
    double arrayArea           = 0.0; // Total array area (um^2)

    bool finished = false;   // true if all C tiles were generated

    // ---- Target layer identification ----
    std::string layerName;   // Name of the layer these results belong to

    // ---- Logical GEMM dimensions (layer dimensions, used for throughput) ----
    long gemmM = 0;
    long gemmN = 0;
    long gemmK = 0;

    // ---- Area (um^2) ----
    // Sum of the physical areas of all SRAM buffers (input/weight/output).
    double sramArea() const {
        return inputBuffer.areaUm2 + weightBuffer.areaUm2 + outputBuffer.areaUm2;
    }
    // Total on-chip area = systolic array + SRAM buffers. (DRAM is off-chip and excluded.)
    double totalArea() const { return arrayArea + sramArea(); }

    // Temporal utilization of the systolic array = array-busy cycles / total cycles.
    // Fraction of time the array was working rather than idle (e.g. waiting on memory). [0, 1].
    double arrayUtilization() const {
        return (cycles > 0) ? static_cast<double>(arrayBusyCycles) / cycles : 0.0;
    }

    // Effective MAC count = M*N*K (one multiplication = one MAC). operationCount() doubles
    // this per the TOPS convention, so utilization uses this multiply-only count instead.
    double macCount() const {
        return static_cast<double>(gemmM) * gemmN * gemmK;
    }

    // MAC (spatial x temporal) utilization = effective MACs / (MAC units x total cycles).
    // Directly reflects under-utilization when the layer is smaller than the array.
    //   e.g. 64x64 array (4096 MACs) with an M=1 GEMV -> only one row is mapped, ~1/64.
    // This is end-to-end utilization, including stalls (memory waits).
    double arrayMacUtilization() const {
        const double peak = static_cast<double>(arrayMacUnits) * cycles;
        return (peak > 0.0) ? macCount() / peak : 0.0;
    }

    // Mapping (spatial-only) efficiency = effective MACs / (MAC units x busy cycles).
    // Fraction of physical PEs actually used during busy cycles. M=1 GEMV -> exactly 1/64.
    double arrayMappingEfficiency() const {
        const double peak = static_cast<double>(arrayMacUnits) * arrayBusyCycles;
        return (peak > 0.0) ? macCount() / peak : 0.0;
    }

    // ===================== Throughput/efficiency metrics =====================
    // One MAC = 1 multiply + 1 add = 2 ops (industry TOPS convention).
    double operationCount() const {
        return 2.0 * static_cast<double>(gemmM) * static_cast<double>(gemmN)
                   * static_cast<double>(gemmK);
    }

    // Execution time (s) = cycles x clockPeriod.
    double seconds() const { return static_cast<double>(cycles) * clockPeriod; }

    // Area (mm^2). totalArea() is in um^2 (array + SRAM), so divide by 1e6.
    double areaMm2() const { return totalArea() / 1e6; }

    // Throughput (TOPS = 1e12 ops/s).
    double tops() const {
        const double s = seconds();
        return (s > 0.0) ? operationCount() / s / 1e12 : 0.0;
    }

    // Energy efficiency (TOPS/W). TOPS/W = (ops/s x 1e-12) / (J/s) = 1e-12 x ops/J = ops/pJ
    // (the Tera and pico factors cancel). Hence (logical ops) / (total energy [pJ]) is TOPS/W.
    double topsPerWatt() const {
        const double e = totalEnergy();   // pJ
        return (e > 0.0) ? operationCount() / e : 0.0;
    }

    // Area-normalized efficiency (TOPS/W/mm^2).
    double topsPerWattPerMm2() const {
        const double a = areaMm2();
        return (a > 0.0) ? topsPerWatt() / a : 0.0;
    }

    // Sum of dynamic (access) energy of all SRAM buffers (pJ).
    // DRAM dynamic energy is not included here (added in totalEnergy() in Analytic mode,
    // or merged from DRAMsim3 results in DramSim3 mode).
    double totalMemoryDynamicEnergy() const {
        return inputBuffer.dynamicEnergy()
             + weightBuffer.dynamicEnergy()
             + outputBuffer.dynamicEnergy();
    }

    // Sum of leakage energy of all SRAM buffers (pJ).
    // DRAM static energy is accounted separately (dramStaticEnergy()).
    double totalMemoryLeakageEnergy() const {
        return bufferLeakageEnergy(inputBuffer)
             + bufferLeakageEnergy(weightBuffer)
             + bufferLeakageEnergy(outputBuffer);
    }

    // Total SRAM buffer energy: dynamic + leakage (pJ). Excludes DRAM.
    double totalMemoryEnergy() const {
        return totalMemoryDynamicEnergy() + totalMemoryLeakageEnergy();
    }

    // Total array energy: dynamic + leakage (pJ).
    double totalArrayEnergy() const {
        return getArrayDynamicEnergy() + getArrayLeakageEnergy();
    }

    // Public accessors so detailed reports (e.g. Optimizer) can obtain per-component
    // energies without duplicating the clockPeriod-based formulas.
    double arrayDynamicEnergy() const { return getArrayDynamicEnergy(); }
    double arrayLeakageEnergy() const { return getArrayLeakageEnergy(); }
    // Two-rate split: clk (all busy cycles) / data (useful stream cycles only).
    double arrayClockEnergy()   const { return getArrayClockEnergy(); }
    double arrayDataEnergy()    const { return getArrayDataEnergy(); }
    // Leakage energy of one SRAM buffer (AccessStats), pJ.
    double bufferLeakage(const AccessStats& s) const { return bufferLeakageEnergy(s); }

    // (C) DRAM static (leakage + refresh/standby) energy. Same form as SRAM leakage:
    //   E = P_dram_static x (cycles x clockPeriod) = P_dram_static x seconds().
    //   Grows with slower designs (more cycles), so stage 1 reflects the slow-design penalty.
    //   Returns 0 unless dramEnergyIncluded (Analytic mode).
    //   P_dram_static = 0.65 W (HBM2_8Gb_x128, 8 channels); two methods agree within 1%:
    //     datasheet : VDD 1.2 V x (IDD3N 55 mA active-standby + (IDD5AB-IDD3N) x tRFC/tREFI
    //                 refresh 15.6 mW) x 8 ch = 0.653 W.
    //     measured  : honest background (-c = LAST x 1.05) / busy (cycles x clockPeriod)
    //                 = 0.649 W (same for q_proj and qk_t).
    //   NOTE: multiplied by the simulated 'cycles', not the DRAMsim3 -c budget (keeps the
    //   time base consistent with the other leakage terms).
    static constexpr double dramStaticPowerW = 0.65;
    double dramStaticEnergy() const {
        return dramEnergyIncluded ? dramStaticPowerW * cycles * clockPeriod * J_TO_PJ : 0.0;
    }

    // Total energy consumption: SRAM memory + array (pJ).
    // In Analytic mode (dramEnergyIncluded) DRAM dynamic + static (C) energy is added too.
    // In DramSim3 mode DRAM is excluded (merge_results.py adds it later).
    double totalEnergy() const {
        double e = totalMemoryEnergy() + totalArrayEnergy();
        if (dramEnergyIncluded) e += dram.dynamicEnergy() + dramStaticEnergy();
        return e;
    }

    // Aggregate directly from the live components at the end of a run.
    //   finalCycle         : total simulated cycles
    //   isFinished         : whether all output tiles were generated
    //   dimM/dimN/dimK     : logical GEMM dimensions of the layer (for throughput)
    //   lName              : layer name (for the report)
    // dramEnergyIncluded must already be set: DRAM energies are copied only in Analytic mode.
    void collect(long finalCycle,
                 bool isFinished,
                 const Array& array,
                 const Dram& dramRef,
                 const DoubleBuffer& inBuf,
                 const DoubleBuffer& wBuf,
                 const DoubleBuffer& outBuf,
                 long dimM = 0, long dimN = 0, long dimK = 0,
                 const std::string& lName = "") {
        cycles   = finalCycle;
        finished = isFinished;
        layerName = lName;
        gemmM = dimM;
        gemmN = dimN;
        gemmK = dimK;

        // DRAM access counts (4096-bit accesses) and bytes.
        dram.readCount    = dramRef.getReadCount();
        dram.writeCount   = dramRef.getWriteCount();
        dram.readBytes    = dramRef.getReadBytes();
        dram.writeBytes   = dramRef.getWriteBytes();
        if (dramEnergyIncluded) {            // Analytic mode: fill with constant per-access energies.
            dram.readEnergy       = dramRef.getReadEnergy();
            dram.writeEnergy      = dramRef.getWriteEnergy();
            dram.activationEnergy = dramRef.getActivationEnergy();   // (B) row-miss term
            dram.activations      = dramRef.getActivations();
        }                                    // DramSim3 mode: leave at 0; merge adds it.

        inputBuffer  = bufferStats(inBuf);
        weightBuffer = bufferStats(wBuf);
        outputBuffer = bufferStats(outBuf);

        arrayActiveCount    = array.getArrayActiveCount();
        arrayBusyCycles     = array.getArrayBusyCycles();
        usefulStreamCycles  = array.getUsefulStreamCycles();
        stationaryLoadCount = array.getStationaryLoadCount();
        arrayMacUnits       = array.getMacUnitCount();
        switchPowerMw       = array.getSwitchPower();
        internalPowerMw     = array.getInternalPower();
        leakagePowerNw      = array.getLeakagePower();
        clockPowerFraction  = array.getClockPowerFraction();
        arrayArea           = array.getArrayArea();
    }

    // Print a human-readable report (energies in pJ unless noted).
    void print(std::ostream& os) const {
        os << "================ Simulation Results ================\n";
        os << "  layer             : "
           << (layerName.empty() ? "(unnamed)" : layerName) << "\n";
        os << "  dimensions M/N/K  : "
           << gemmM << " / " << gemmN << " / " << gemmK << "\n";
        os << "  cycles            : " << cycles << "\n";
        os << "  status            : "
           << (finished ? "finished" : "INCOMPLETE (some C tiles not generated)")
           << "\n";

        os << "  -- Array --\n";
        os << "    active ops      : " << arrayActiveCount << "\n";
        os << "    busy cycles     : " << arrayBusyCycles << "\n";
        os << "    useful cycles   : " << usefulStreamCycles
           << "  (stream w/ data; excludes fill/drain/zero-pad)\n";
        os << "    MAC units       : " << arrayMacUnits << "\n";
        os << "    MAC utilization : " << (arrayMacUtilization() * 100.0) << " %"
           << "  (" << macCount() << " MACs / " << arrayMacUnits
           << " units x " << cycles << " cycles)\n";
        os << "    mapping eff     : " << (arrayMappingEfficiency() * 100.0) << " %"
           << "  (spatial, over busy cycles)\n";
        os << "    time  util      : " << (arrayUtilization() * 100.0) << " %"
           << "  (busy " << arrayBusyCycles << " / " << cycles << " cycles)\n";
        // Stationary (re)load count. Exposed for the same reason the Pulse Scala driver's
        // summary line reports `loads=`: stationary reuse must be visible in the log, not inferred.
        // Expected value in dual mode = numPanels x numKPasses = nTiles x kTiles.
        os << "    stationary loads: " << stationaryLoadCount << "\n";
        // os << "    switch power    : " << switchPowerMw   << " mW\n";
        // os << "    internal power  : " << internalPowerMw << " mW\n";
        // os << "    leakage power   : " << leakagePowerNw  << " nW\n";
        os << "    dynamic energy  : " << getArrayDynamicEnergy() << " pJ\n";
        os << "      clk  (busy)   : " << getArrayClockEnergy()
           << " pJ  (clkFrac " << clockPowerFraction << " x "
           << arrayBusyCycles << " cycles)\n";
        os << "      data (useful) : " << getArrayDataEnergy()
           << " pJ  (x " << usefulStreamCycles << " cycles)\n";
        os << "    leakage energy  : " << getArrayLeakageEnergy() << " pJ\n";
        os << "    total energy    : " << totalArrayEnergy() << " pJ\n";
        os << "    area            : " << arrayArea << " um^2\n";

        if (dramEnergyIncluded) {
            os << "  -- DRAM (analytic: constant per 4096-bit access) --\n";
            os << "    accesses r / w  : " << dram.readCount << " / " << dram.writeCount
               << "  (4096-bit)\n";
            os << "    read / write B  : " << dram.readBytes << " / " << dram.writeBytes << "\n";
            os << "    read energy     : " << dram.readEnergy  << "\n";
            os << "    write energy    : " << dram.writeEnergy << "\n";
            os << "    activations     : " << dram.activations
               << "  (act energy " << dram.activationEnergy << " pJ)\n";
            os << "    dynamic energy  : " << dram.dynamicEnergy() << "\n";
        } else {
            os << "  -- DRAM (counts only; energy/latency/bandwidth from DRAMsim3) --\n";
            os << "    accesses r / w  : " << dram.readCount << " / " << dram.writeCount
               << "  (4096-bit)\n";
        }
        printAccess(os, "Input buffer",  inputBuffer,  bufferLeakageEnergy(inputBuffer));
        printAccess(os, "Weight buffer", weightBuffer, bufferLeakageEnergy(weightBuffer));
        printAccess(os, "Output buffer", outputBuffer, bufferLeakageEnergy(outputBuffer));

        os << (dramEnergyIncluded
                   ? "  -- Energy (pJ) [compute + SRAM + DRAM (analytic)] --\n"
                   : "  -- Energy (pJ) [compute + SRAM; DRAM added by merge_results.py] --\n");
        os << "    memory (SRAM)   : " << totalMemoryEnergy()        << "\n";
        os << "      dynamic       : " << totalMemoryDynamicEnergy() << "\n";
        os << "      leakage       : " << totalMemoryLeakageEnergy() << "\n";
        os << "    array           : " << totalArrayEnergy()       << "\n";
        os << "      dynamic       : " << getArrayDynamicEnergy()  << "\n";
        os << "      leakage       : " << getArrayLeakageEnergy()  << "\n";
        if (dramEnergyIncluded) {
            os << "    DRAM dyn(anlyt) : " << dram.dynamicEnergy()  << "\n";
            os << "    DRAM static (C) : " << dramStaticEnergy()
               << "  (" << dramStaticPowerW << " W x " << seconds() << " s)\n";
        } else
            os << "    DRAM            : 0  (via DRAMsim3 merge)\n";
        os << "    TOTAL           : " << totalEnergy()            << "\n";

        os << "  -- Area (um^2) --\n";
        os << "    array           : " << arrayArea               << "\n";
        os << "    SRAM (in/w/out) : " << inputBuffer.areaUm2 << " / "
           << weightBuffer.areaUm2 << " / " << outputBuffer.areaUm2 << "\n";
        os << "    SRAM total      : " << sramArea()              << "\n";
        os << "    TOTAL           : " << totalArea()             << "\n";

        os << (dramEnergyIncluded
                   ? "  -- Efficiency (DRAM-inclusive: compute + SRAM + DRAM analytic) --\n"
                   : "  -- Efficiency (compute + SRAM; DRAM-inclusive values from merge) --\n");
        os << "    throughput      : " << tops()              << " TOPS\n";
        os << "    energy eff      : " << topsPerWatt()       << " TOPS/W\n";
        os << "    area-norm eff   : " << topsPerWattPerMm2() << " TOPS/W/mm^2\n";
        os << "===================================================\n";
    }

    // Save the report to a text file, creating parent directories if needed.
    // Returns false on failure.
    bool writeToFile(const std::string& path) const {
        std::filesystem::path p(path);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());

        std::ofstream ofs(p);
        if (!ofs) {
            std::cerr << "Error: could not open '" << path
                      << "' for writing results\n";
            return false;
        }
        print(ofs);
        return true;
    }

private:

    double clockPeriod = 2e-9;   // Clock period in seconds (2 ns)

    // Memory energies are in pJ, so array energies are converted to pJ as well.
    //   dynamic: mW x cycles x s = mJ -> x1e9  = pJ
    //   leakage: nW x cycles x s = nJ -> x1e3  = pJ
    static constexpr double MJ_TO_PJ = 1e9;
    static constexpr double NJ_TO_PJ = 1e3;
    // SRAM leakage and DRAM static power are in W, so W x cycles x s = J -> x1e12 = pJ.
    static constexpr double J_TO_PJ  = 1e12;

    // Leakage energy of one SRAM buffer (pJ).
    // Like array leakage, computed as power x (powered-on time = whole run).
    //   W x cycles x clockPeriod(s) = J -> x J_TO_PJ = pJ
    double bufferLeakageEnergy(const AccessStats& s) const {
        return s.leakagePowerW * cycles * clockPeriod * J_TO_PJ;
    }

    // Two-rate dynamic energy model:
    //   E_dyn = P_clk x arrayBusyCycles x T  +  P_data x usefulStreamCycles x T
    //   P_clk  = clockPowerFraction x (switch+internal) - clock tree / register internal power.
    //            Charged on every busy (fill+stream+drain) cycle regardless of data.
    //   P_data = the remainder - MAC/operand datapath switching. Charged only on
    //            usefulStreamCycles where data actually flowed (zero-pad/fill/drain have
    //            zero inputs -> no toggling).
    // History: (1) proportional to the tile-op count (arrayActiveCount) -> the streaming MAC
    //   term was missing and cd efficiency was overestimated. (2) flat full power x busy
    //   cycles -> for low-utilization (M=1 decode) layers fill/zero-pad/drain were double
    //   charged at typical-activity power, ~127x inflation. The
    //   current model keeps busy-cycle proportionality (unit-consistent with leakage;
    //   zero-pad waste stays visible as clock power) while separating datapath activity onto
    //   useful cycles only. With clockPowerFraction = 1.0 it is identical to (2).
    double getArrayClockEnergy() const {
        const double pClk = clockPowerFraction * (switchPowerMw + internalPowerMw);
        return pClk * arrayBusyCycles * clockPeriod * MJ_TO_PJ;
    }

    // Datapath part of the two-rate model (pJ); see getArrayClockEnergy().
    double getArrayDataEnergy() const {
        const double pData = (1.0 - clockPowerFraction) * (switchPowerMw + internalPowerMw);
        return pData * usefulStreamCycles * clockPeriod * MJ_TO_PJ;
    }

    double getArrayDynamicEnergy() const {
        return getArrayClockEnergy() + getArrayDataEnergy();
    }

    // Array leakage over the whole run (all cycles, not just busy ones), pJ.
    double getArrayLeakageEnergy() const {
        return (leakagePowerNw) * cycles * clockPeriod * NJ_TO_PJ;
    }

    // Snapshot counters, energies, leakage power and area of one SRAM double buffer.
    static AccessStats bufferStats(const DoubleBuffer& buf) {
        AccessStats s;
        s.readCount    = buf.getReadCount();
        s.writeCount   = buf.getWriteCount();
        s.readBytes    = buf.getReadBytes();
        s.writeBytes   = buf.getWriteBytes();
        s.readEnergy   = buf.getReadEnergy();
        s.writeEnergy  = buf.getWriteEnergy();
        s.leakagePowerW = buf.getLeakagePower();
        s.areaUm2       = buf.getArea();
        return s;
    }

    // Print one SRAM buffer section of the report.
    static void printAccess(std::ostream& os, const char* name,
                            const AccessStats& s, double leakageEnergy) {
        os << "  -- " << name << " --\n";
        os << "    reads / writes  : " << s.readCount << " / " << s.writeCount << "\n";
        os << "    read / write B  : " << s.readBytes << " / " << s.writeBytes << "\n";
        os << "    read energy     : " << s.readEnergy  << "\n";
        os << "    write energy    : " << s.writeEnergy << "\n";
        os << "    dynamic energy  : " << s.dynamicEnergy() << "\n";
        os << "    leakage power   : " << s.leakagePowerW << " W\n";
        os << "    leakage energy  : " << leakageEnergy << "\n";
        os << "    total energy    : " << (s.dynamicEnergy() + leakageEnergy) << "\n";
    }
};
