#pragma once
#include <string>
#include <fstream>
#include <stdexcept>
#include <cctype>
#include <memory>
#include "ArrayConfig.hpp"
#include "Dataflow.hpp"

// Parser for array_config/*.cfg (systolic array configuration).
// Reads the SystolicArrayConfig parameters that used to be hard-coded in main.cpp.
//
// File format: one "key : value" (or "key = value") per line.
//   - Lines starting with '#' and blank lines are ignored.
//   - Leading/trailing whitespace around key and value is trimmed.
//   - Unknown keys and empty values are errors.
//
// Required keys:
//   kind                       Grouped | Toroidal | ToroidalDual
//                              (legacy aliases A2 | A3 | A3Burst | A3ReuseBBurst are accepted)
//   outerPeRow outerPeCol localPeRow localPeCol multPerPe
//   dataflow                   ReuseA | ReuseB | ReuseC   (required only for Grouped/Toroidal; ToroidalDual is fixed to ReuseB and ignores it)
//
// Optional (override) keys:
//   contractingDimensionSize   free/streaming tile length. By default it is not taken
//     from the file: the program derives the *largest* value that fits one face of the
//     selected input/weight SRAM (ContractingDimension.hpp). Giving this key disables
//     auto-derivation and uses the value as a manual override (for sweep experiments).
//     If the override overflows the SRAM, downstream checkSramSize/quick-check catches
//     it; that is the user's responsibility.
//   maxContractionSize         maximum GEMM K this array is provisioned for (must be > 0).
//     For ReuseA/ReuseB it only sets the accumulator data width (tile shapes unchanged),
//     so it has no effect on cycles/traffic in this model. It is used only as the tileK
//     upper bound of Grouped ReuseC. If omitted, the historical value
//     (systolicArrayColDimension) is used, preserving existing results.
//     Constraint: must be a multiple of multPerPe (require in Scala SystolicArrayConfig).
//   arraySwitchPower, arrayInternalPower (mW), arrayLeakagePower (nW), arrayArea (um^2)
//   arrayClockPowerFraction    in [0,1], default 1.0 (see ArrayConfigSpec)
//   maximumTileNumber          deprecated, accepted and ignored
//
// Accumulation is done by RMW on the output SRAM rather than by a separate
// accumulator, so this file has no accumulator parameters.
// Parsed contents of one .cfg file. Turned into a SystolicArrayConfig via makeConfig().
struct ArrayConfigSpec {
    ConfigKind kind = ConfigKind::Grouped;
    Dataflow   dataflow = Dataflow::ReuseA;
    int        outerPeRow = 0;
    int        outerPeCol = 0;
    int        localPeRow = 0;
    int        localPeCol = 0;
    int        multPerPe  = 0;

    // Free/streaming tile length. If the .cfg has this key, hasManualCd is set to true
    // and the value is used as a manual override. Otherwise (the default) the program
    // derives it to fit one SRAM face and this value is ignored.
    // (Auto-derivation: ContractingDimension.hpp)
    int        contractingDimensionSize = 0;
    bool       hasManualCd = false;

    // Maximum GEMM K this array is provisioned for. 0 = unspecified (= systolicArrayColDimension).
    int        maxContractionSize = 0;

    // Array power parameters, passed unchanged to the Array constructor.
    // Switch/internal power in mW, leakage power in nW.
    double arraySwitchPower   = 0.0;
    double arrayInternalPower = 0.0;
    double arrayLeakagePower  = 0.0;
    // Total array area (um^2).
    double arrayArea          = 0.0;
    // Fraction of the total streaming power (switch+internal) that is burned on every
    // clocked cycle regardless of data, such as clock tree and registers (two-rate
    // energy model):
    //   E_dyn = clkFrac*P_total*busyCycles + (1-clkFrac)*P_total*usefulStreamCycles.
    // The DC report gives typical-activity power without SAIF (-analysis_effort low);
    // this split avoids charging full power for fill/zero-pad/drain cycles.
    // If the key is absent, 1.0 = the old flat charging (backward compatible). The
    // first-order default for a 65nm register-heavy systolic array is 0.4
    // (written by parse_synthesis.py).
    double arrayClockPowerFraction = 1.0;

    // ToroidalDual requires burst SRAM inputs (passed as separate arguments).
    bool isBurst() const { return kind == ConfigKind::ToroidalDual; }

    // Whether the kind declared in the .cfg was ToroidalDual. Set by
    // resolveBurstForLayerM and kept even after it overwrites kind with Toroidal for
    // GEMM; SRAM source selection reads it.
    bool declaredBurst = false;

    // ToroidalDual is a GEMV-only (M==1) schedule. Using it as-is for general GEMM (M>1)
    // would fold the whole M axis into a single tile (the "GEMV mode tiling
    // transition"), so once the layer's M is known (before building SRAMs/config) the
    // spec is resolved to Toroidal ReuseB. A burst config already has ReuseB geometry
    // (outerPeRow*localPeRow == outerPeCol), so it passes ToroidalConfig(ReuseB)
    // validation unchanged, and this single substitution makes cd derivation,
    // bandwidths, schedule, array cycles and Simulator labels all consistently behave
    // as Toroidal ReuseB.
    //
    // Even when demoted, declaredBurst records that "this .cfg was originally
    // ToroidalDual". SRAM source selection reads it: there is physically only one
    // weight buffer, so both GEMM and GEMV must pick from the same burst_weight macro.
    // Looking only at kind, a demoted config is indistinguishable from Toroidal and
    // would leak into the normal SRAM set (= a physically impossible model with a
    // different SRAM per mode). Bandwidths still differ per mode:
    //   GEMV (ToroidalDual)              weight bw = localPeCol*multPerPe*8 = series min
    //   GEMM (demoted Toroidal ReuseB)   weight bw = colDimension*8         = series max
    // These match exactly the R/U in the macro name burst_weight_max{R}_min{U}.
    void resolveBurstForLayerM(int dimensionM) {
        if (kind == ConfigKind::ToroidalDual) {
            declaredBurst = true;
            if (dimensionM != 1) {
                kind     = ConfigKind::Toroidal;
                dataflow = Dataflow::ReuseB;
            }
        }
    }

    // Builds the matching SystolicArrayConfig subclass for the given cd.
    // (cd is either the auto-derived value or the manual override; the caller decides.)
    // Throws std::runtime_error on an unhandled kind.
    std::shared_ptr<const SystolicArrayConfig> makeConfig(int cd) const {
        std::shared_ptr<SystolicArrayConfig> c;
        switch (kind) {
            case ConfigKind::Grouped:
                c = std::make_shared<GroupedConfig>(
                    outerPeRow, outerPeCol, localPeRow, localPeCol,
                    multPerPe, dataflow, cd, maxContractionSize);
                break;
            case ConfigKind::Toroidal:
                c = std::make_shared<ToroidalConfig>(
                    outerPeRow, outerPeCol, localPeRow, localPeCol,
                    multPerPe, dataflow, cd, maxContractionSize);
                break;
            case ConfigKind::ToroidalDual:
                c = std::make_shared<ToroidalDualConfig>(
                    outerPeRow, outerPeCol, localPeRow, localPeCol,
                    multPerPe, cd, maxContractionSize);
                break;
        }
        if (!c) throw std::runtime_error("array config: unhandled ConfigKind");
        // Carry the original ToroidalDual declaration into the config even if kind was demoted.
        c->declaredBurst = declaredBurst || (kind == ConfigKind::ToroidalDual);
        return c;
    }

    // Convenience overload without an explicit cd. Used to build a "placeholder config"
    // for computations whose result does not depend on cd (geometry/bandwidth). Uses
    // the parsed manual override if present, otherwise 1 as the placeholder cd.
    std::shared_ptr<const SystolicArrayConfig> makeConfig() const {
        return makeConfig(hasManualCd ? contractingDimensionSize : 1);
    }
};

namespace array_cfg_detail {

// Strips leading/trailing spaces, tabs, CR and LF.
inline std::string trim(const std::string& s) {
    std::size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    std::size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Maps a kind string to ConfigKind; throws on unknown values.
inline ConfigKind parseKind(const std::string& v, const std::string& path) {
    if (v == "Grouped") return ConfigKind::Grouped;
    if (v == "Toroidal") return ConfigKind::Toroidal;
    if (v == "ToroidalDual") return ConfigKind::ToroidalDual;
    // Legacy names (A2/A3/A3Burst/A3ReuseBBurst) are still accepted so existing .cfg files keep working.
    if (v == "A2") return ConfigKind::Grouped;
    if (v == "A3") return ConfigKind::Toroidal;
    if (v == "A3Burst" || v == "A3ReuseBBurst") return ConfigKind::ToroidalDual;
    throw std::runtime_error("array config: unknown kind \"" + v + "\" in \"" + path + "\"");
}

// Maps a dataflow string to Dataflow; throws on unknown values.
inline Dataflow parseDataflow(const std::string& v, const std::string& path) {
    if (v == "ReuseA") return Dataflow::ReuseA;
    if (v == "ReuseB") return Dataflow::ReuseB;
    if (v == "ReuseC") return Dataflow::ReuseC;
    throw std::runtime_error("array config: unknown dataflow \"" + v + "\" in \"" + path + "\"");
}

} // namespace array_cfg_detail

// Parses an array configuration file. Throws std::runtime_error if the file cannot be
// opened, a line is malformed, a key is unknown/invalid, or a required field is missing.
inline ArrayConfigSpec parseArrayConfig(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("array config: cannot open file \"" + path + "\"");

    ArrayConfigSpec spec;
    bool haveKind = false, haveDataflow = false;
    bool haveORow = false, haveOCol = false, haveLRow = false, haveLCol = false;
    bool haveMult = false;

    std::string line;
    while (std::getline(in, line)) {
        std::string trimmed = array_cfg_detail::trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        std::size_t sep = trimmed.find_first_of(":=");
        if (sep == std::string::npos)
            throw std::runtime_error(
                "array config: ':' or '=' expected in \"" + line + "\"");

        std::string key = array_cfg_detail::trim(trimmed.substr(0, sep));
        std::string val = array_cfg_detail::trim(trimmed.substr(sep + 1));
        if (val.empty())
            throw std::runtime_error(
                "array config: empty value for \"" + key + "\" in \"" + path + "\"");

        if (key == "kind") {
            spec.kind = array_cfg_detail::parseKind(val, path); haveKind = true;
        } else if (key == "dataflow") {
            spec.dataflow = array_cfg_detail::parseDataflow(val, path); haveDataflow = true;
        } else if (key == "outerPeRow") {
            spec.outerPeRow = std::stoi(val); haveORow = true;
        } else if (key == "outerPeCol") {
            spec.outerPeCol = std::stoi(val); haveOCol = true;
        } else if (key == "localPeRow") {
            spec.localPeRow = std::stoi(val); haveLRow = true;
        } else if (key == "localPeCol") {
            spec.localPeCol = std::stoi(val); haveLCol = true;
        } else if (key == "multPerPe") {
            spec.multPerPe = std::stoi(val); haveMult = true;
        } else if (key == "contractingDimensionSize") {
            // Optional override: if present, disables auto-derivation and uses this value as-is.
            spec.contractingDimensionSize = std::stoi(val); spec.hasManualCd = true;
        } else if (key == "maxContractionSize") {
            spec.maxContractionSize = std::stoi(val);
            if (spec.maxContractionSize <= 0)
                throw std::runtime_error(
                    "array config: maxContractionSize must be positive in \"" + path + "\"");
        } else if (key == "maximumTileNumber") {
            // Deprecated / no-op. It used to set the output SRAM depth
            // (= cd * maximumTileNumber), but the output SRAM size is now checked
            // directly against the plan's actual tile count (Simulator::checkSramSize).
            // The key is still accepted for compatibility with old/sweep-generated .cfg
            // files, but its value is ignored.
        } else if (key == "arraySwitchPower") {
            spec.arraySwitchPower = std::stod(val);
        } else if (key == "arrayInternalPower") {
            spec.arrayInternalPower = std::stod(val);
        } else if (key == "arrayLeakagePower") {
            spec.arrayLeakagePower = std::stod(val);
        } else if (key == "arrayArea") {
            spec.arrayArea = std::stod(val);
        } else if (key == "arrayClockPowerFraction") {
            spec.arrayClockPowerFraction = std::stod(val);
            if (spec.arrayClockPowerFraction < 0.0 || spec.arrayClockPowerFraction > 1.0)
                throw std::runtime_error(
                    "array config: arrayClockPowerFraction must be in [0,1] in \"" + path + "\"");
        } else {
            throw std::runtime_error(
                "array config: unknown key \"" + key + "\" in \"" + path + "\"");
        }
    }

    // contractingDimensionSize / maximumTileNumber are no longer required
    // (cd is auto-derived; maximumTileNumber was removed).
    if (!haveKind || !haveORow || !haveOCol || !haveLRow || !haveLCol ||
        !haveMult)
        throw std::runtime_error(
            "array config: missing required field(s) in \"" + path + "\"");

    // ToroidalDual is fixed to ReuseB, so dataflow may be omitted.
    // Grouped/Toroidal require dataflow.
    if (!haveDataflow && spec.kind != ConfigKind::ToroidalDual)
        throw std::runtime_error(
            "array config: 'dataflow' required for Grouped/Toroidal in \"" + path + "\"");

    return spec;
}
