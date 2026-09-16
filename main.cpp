// Single-run entry point: simulate one layer on one array configuration.
// Parses the array and layer configs, auto-selects the input/weight/output SRAMs from the
// CACTI results, derives cd (unless overridden in the .cfg), builds the schedule and runs
// the cycle-level Simulator. DRAM energy backend is chosen via the DRAM_MODEL env variable.
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
#include <memory>
#include <iostream>
#include <string>
#include <cstdlib>

namespace {

// Build a DoubleBuffer from two ordinary (single-access) SRAM macros (ping/pong).
// One face = one macro, so the face capacity equals the macro capacity, while area and
// leakage are charged for both macros (see the double-buffer note in SramSpec).
DoubleBuffer makeBuffer(const SramSpec& s) {
    return DoubleBuffer(
        s.faceCapacityBytes(),
        s.dataWidthBytes,
        s.readEnergyPerAccess,
        s.writeEnergyPerAccess,
        s.totalLeakagePowerW(),
        s.totalAreaUm2()
    );
}

void printUsage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " <arrayConfig> <layer> <inputCapKB> <weightCapKB> <outputCapKB> [sramDir]\n"
        << "\n"
        << "  arrayConfig      : array_config/*.cfg (array configuration file). Burst mode is selected by its kind\n"
        << "                     (ToroidalDual), and is used only for GEMV layers (M==1).\n"
        << "  layer            : layer/*.cfg (layer = GEMM configuration file). Reads name/dimensionM/N/K.\n"
        << "  *CapKB           : (compatibility only) ignored, since CACTI sources are fixed at 256KB. The\n"
        << "                     bandwidth is derived from the array configuration and the matching CACTI CSV\n"
        << "                     row is selected automatically.\n"
        << "  sramDir          : CACTI results root (default: cacti_results). The library routes internally to\n"
        << "                     the sub-CSV for each role/mode (normal/burst_input/burst_weight).\n";
}

int parseCapKB(const char* s) {
    return std::stoi(std::string(s));
}

// Select the DRAM energy backend from the DRAM_MODEL environment variable.
//   analytic           -> constant per-access energy, self-contained (no trace/merge needed).
//   anything else/unset -> dramsim3: a trace is written and DRAMsim3 + merge_results.py
//                          supply the DRAM energy.
// The default is dramsim3 to preserve the original pipeline.
DramEnergyMode parseDramMode() {
    const char* env = std::getenv("DRAM_MODEL");
    if (env && std::string(env) == "analytic") return DramEnergyMode::Analytic;
    return DramEnergyMode::DramSim3;
}

// Human-readable description of the DRAM backend, printed in the run header.
const char* dramModeName(DramEnergyMode m) {
    return (m == DramEnergyMode::Analytic)
        ? "analytic (constant per 4096-bit access; DRAM in totals, no trace/merge)"
        : "dramsim3 (trace -> DRAMsim3 + merge_results.py; DRAM excluded from totals)";
}

} // namespace

int main(int argc, char** argv) {
    // argv[1] = array configuration file. Parse the array configuration first.
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    const std::string arrayConfigPath = argv[1];

    try {
        ArrayConfigSpec arraySpec = parseArrayConfig(arrayConfigPath);

        // Argument layout:
        //   arrayConfig layer inputCapKB weightCapKB outputCapKB [sramDir]
        //   - sramDir : optional (last). Defaults to cacti_results when absent.
        // Required argc = 1 (prog) + 1 (array) + 1 (layer) + 3 capacities.
        const int requiredArgc = 1 + 1 + 1 + 3;
        if (argc != requiredArgc && argc != requiredArgc + 1) {
            std::cerr << "Error: a layer config and 3 capacity arguments (+ optional sramDir) are required.\n";
            printUsage(argv[0]);
            return 1;
        }

        // If the optional last argument is present, treat it as sramDir.
        const std::string sramDir =
            (argc == requiredArgc + 1) ? argv[argc - 1] : "cacti_results";

        int idx = 2;
        Layer layer = parseLayer(argv[idx++]);

        // ToroidalDual is GEMV-only (M==1). For a general GEMM (M>1) it is resolved here to
        // Toroidal ReuseB, so cd derivation, SRAM selection, config and schedule all use ReuseB.
        arraySpec.resolveBurstForLayerM(layer.gemmDimenion.dimensionM);

        // The derived cd depends on the selected input/weight SRAM capacities, so the SRAMs
        // are chosen before building the config/Scheduler. First build a placeholder config
        // for the bandwidth guard (bandwidth and geometry do not depend on cd).
        std::shared_ptr<const SystolicArrayConfig> geomConfig = arraySpec.makeConfig();

        // Parse the SRAM synthesis results once and reuse them for all roles.
        SramLibrary srams(sramDir);

        // Argument layout: inputCapKB weightCapKB outputCapKB (currently ignored by selectSram).
        const int inputCapKB  = parseCapKB(argv[idx++]);
        const int weightCapKB = parseCapKB(argv[idx++]);
        const int outputCapKB = parseCapKB(argv[idx++]);

        // Auto-select, per role, the narrowest SRAM whose data width meets the array's required
        // bandwidth (throws if none does). The capacity argument is accepted but ignored.
        SramFile inputSram  = selectSram(srams, *geomConfig, SramRole::Input,  inputCapKB);
        SramFile weightSram = selectSram(srams, *geomConfig, SramRole::Weight, weightCapKB);
        SramFile outputSram = selectSram(srams, *geomConfig, SramRole::Output, outputCapKB);

        // cd: use the .cfg override if present; otherwise derive the largest value that fits
        // one face of the input/weight SRAM. cd<1 means not even the smallest tile fits: fail.
        const int cd = arraySpec.hasManualCd
            ? arraySpec.contractingDimensionSize
            : deriveContractingDimensionSize(*geomConfig, layer,
                                             inputSram.spec.faceCapacityBytes(),
                                             weightSram.spec.faceCapacityBytes());
        if (cd < 1)
            throw std::runtime_error(
                "contractingDimensionSize: even cd=1 does not fit the streamed tile "
                "into the chosen input/weight SRAM face. Provide a larger SRAM.");

        std::shared_ptr<const SystolicArrayConfig> config = arraySpec.makeConfig(cd);
        // Pass the output SRAM face capacity so the Scheduler can split the RMW accumulation
        // axis (ReuseB: M) into G partial-sum tile blocks (output-side blocking, symmetric to
        // cd fixing the input-side tiling via the input/weight SRAM). The weight SRAM face
        // capacity is passed as well.
        Scheduler schedule = Scheduler(layer, config, outputSram.spec.faceCapacityBytes(),
                                       weightSram.spec.faceCapacityBytes());
        Array array = Array(config,
                            arraySpec.arraySwitchPower,
                            arraySpec.arrayInternalPower,
                            arraySpec.arrayLeakagePower,
                            arraySpec.arrayArea,
                            arraySpec.arrayClockPowerFraction);
        Dram dram;

        std::cout << "Selected SRAMs:\n"
                  << "  input      : " << inputSram.path  << "\n";

        DoubleBuffer inputBuffer = makeBuffer(inputSram.spec);

        std::cout << "  weight     : " << weightSram.path << "\n"
                  << "  output     : " << outputSram.path << "\n"
                  << (arraySpec.hasManualCd ? "  cd(manual) : " : "  cd(derived): ")
                  << cd << "\n";

        DoubleBuffer outputBuffer = makeBuffer(outputSram.spec);
        DoubleBuffer weightBuffer = makeBuffer(weightSram.spec);

        const DramEnergyMode dramMode = parseDramMode();
        std::cout << "  DRAM model : " << dramModeName(dramMode) << "\n";

        Simulator sim(schedule.plan(), array, dram,
                      inputBuffer, outputBuffer, weightBuffer, dramMode);
        sim.checkSramSize();
        sim.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
