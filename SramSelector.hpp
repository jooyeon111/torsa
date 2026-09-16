#pragma once
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <stdexcept>
#include <cstdlib>

#include "ArrayConfig.hpp"
#include "SramResultParser.hpp"

// ============================================================================
// SramSelector
//
// The user does not pass SRAM "bandwidth" directly. The program derives the "required
// bandwidth" of each SRAM role (input/weight/output) from the array configuration and
// picks the CACTI result row that fits that bandwidth.
//
// The data source (under cacti_results/) depends on role and mode:
//   .cfg is not burst (neither ToroidalDual nor declaredBurst) -> all roles use normal/normal_results.csv
//   .cfg is burst (ToroidalDual kind or declaredBurst), regardless of M:
//        input   -> burst_input_256K/*.csv
//        weight  -> burst_weight_max{colDim*8}_min<digits>/*.csv
//        output  -> normal/normal_results.csv (output does not use a burst macro)
//
// The burst test also checks declaredBurst (the kind as declared, before demotion), not
// only kind(). There is physically only one input buffer and one weight buffer, so GEMM
// and GEMV must use the same macro. Testing kind() alone would let GEMM (M>1), after
// resolveBurstForLayerM() demotes it to Toroidal, leak into normal, producing a
// (physically impossible) model with a different SRAM per mode.
//
// Only the bandwidth depends on the mode, so a different row of the same macro is picked:
//   input   GEMV = rowDim*outerPeCol*localPeCol*8 = 8192 in the current design space (1024 MACs x 8)
//           GEMM = colDim*8                       = 256..4096
//   weight  GEMV = localPeCol*multPerPe*8         = min(U) of the series
//           GEMM = colDim*8                       = max(R) of the series
// Hence the burst_weight series name max{R}_min{U}. colDim is identical in both modes
// (= oCol*lCol*mult), so the demoted config still points to the same series.
//
// Both burst macros are CSVs that "lock the structure at the maximum bandwidth and sweep
// downward". Access energy differs per row, but area/leakage are identical on every row
// (the value at the maximum point), because there is only one physical macro provisioned
// at the maximum point (build_cacti_results.sh:emit_row).
//
// Capacity is fixed at 256 KB for every source, so there is no capacity axis;
// only bandwidth is selected below.
//
// Required bandwidth (bytes/cycle): operands are 8-bit (= 1 byte), so the array's
// port bit width / 8 = byte bandwidth.
//   - input  : cfg.systolicArrayInputBandwidthBtis()  / 8
//   - weight : cfg.systolicArrayWeightBandwidthBtis() / 8
//   - output : cfg.systolicArrayOutputBandwidthBtis() / 8  (optionally raised by OPT_OUT_ACCESS_BITS)
//
// Selection rule (= guard): among rows with dataWidth (bytes) >= required, pick the
// smallest. Wider than required wastes bandwidth, narrower is a bottleneck, so
// "smallest that meets the requirement" is the proper choice. selectSram() throws if no
// row qualifies. (buswidth_bits = dataWidth(bytes) x 8.)
// ============================================================================

// Which buffer an SRAM macro serves.
enum class SramRole { Input, Weight, Output };

// One selectable SRAM row: a display path (CSV path + " [band<bits>]") and its parsed spec.
struct SramFile {
    std::string path;
    SramSpec    spec;
};

namespace sram_select_detail {

// Lowercase role name for error messages.
inline const char* roleName(SramRole role) {
    switch (role) {
        case SramRole::Input:      return "input";
        case SramRole::Weight:     return "weight";
        case SramRole::Output:     return "output";
    }
    return "?";
}

// Path of the first *.csv file in a directory (non-recursive, so files under raw/ are
// ignored). Iteration order is unspecified, so each directory is expected to hold exactly
// one CSV. Throws if the directory is missing or contains no CSV.
inline std::string firstCsvIn(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    if (!fs::exists(dir))
        throw std::runtime_error("SRAM select: directory not found: " + dir.string());
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".csv")
            return e.path().string();
    throw std::runtime_error("SRAM select: no CSV file in " + dir.string());
}

} // namespace sram_select_detail

// Output-coalescing experiment switch: env OPT_OUT_ACCESS_BITS (an absolute access width in
// bits, not k x port width), read once and cached.
// Unset/0 keeps the default rule (access width = port width). When set, the output SRAM
// access width is raised to max(port requirement, this value). The array still emits at
// port width per cycle, but a staging register collects consecutive addresses and accesses
// one wide row. The same absolute width must be applied to every config so the hardware
// budget (register = access width) is identical; this is the same principle as unifying
// input GEMV at 8192 bits across all configs.
inline int outputAccessBitsOverride() {
    static const int bits = [] {
        const char* s = std::getenv("OPT_OUT_ACCESS_BITS");
        return s ? std::atoi(s) : 0;
    }();
    return bits;
}

// Required bandwidth (bytes/cycle) of a role, derived from the array configuration.
inline int requiredSramBandwidthBytes(const SystolicArrayConfig& cfg, SramRole role) {
    switch (role) {
        case SramRole::Input:      return cfg.systolicArrayInputBandwidthBtis() / 8;
        case SramRole::Weight:     return cfg.systolicArrayWeightBandwidthBtis() / 8;
        case SramRole::Output: {
            const int need = cfg.systolicArrayOutputBandwidthBtis() / 8;
            const int unified = outputAccessBitsOverride() / 8;
            return unified > need ? unified : need;
        }
    }
    return 0;
}

// (root, cfg, role) -> path of the CACTI CSV this role reads.
// Throws if the burst input/weight source cannot be found.
inline std::string cactiSourceFor(const std::string& root,
                                  const SystolicArrayConfig& cfg, SramRole role) {
    namespace fs = std::filesystem;
    const std::string normalCsv = (fs::path(root) / "normal" / "normal_results.csv").string();

    // For input/weight, burst is decided by declaredBurst as well as kind(). Each buffer
    // exists physically only once, so GEMM and GEMV must use the same macro. Deciding by
    // kind() alone would, for GEMM, demote ToroidalDual to Toroidal and leak into normal,
    // producing a (physically impossible) model with a different SRAM per mode. Bandwidth
    // does depend on the mode, so a different row of the same macro is picked:
    //   input  GEMV = rowDim*outerPeCol*localPeCol*8 = 8192 in the current design space (1024 MACs x 8; the series maximum)
    //          GEMM = colDimension*8                 = 256..4096
    //   weight GEMV = localPeCol*multPerPe*8         = min(U) of the series
    //          GEMM = colDimension*8                 = max(R) of the series
    // colDimension is the same in both modes (= oCol*lCol*mult), so weight points to the same series.
    const bool declared = (cfg.kind() == ConfigKind::ToroidalDual) || cfg.declaredBurst;

    if (role == SramRole::Input && declared)
        return sram_select_detail::firstCsvIn(fs::path(root) / "burst_input_256K");

    if (role == SramRole::Weight && declared) {
        // Series selection: the burst_weight_* directory whose max == colDim*8.
        const int maxBits = cfg.systolicArrayColDimension * 8;
        const std::string prefix = "burst_weight_max" + std::to_string(maxBits) + "_min";
        for (const auto& e : fs::directory_iterator(root)) {
            if (!e.is_directory()) continue;
            const std::string name = e.path().filename().string();
            if (name.rfind(prefix, 0) != 0) continue;
            // Only the min bandwidth (digits) may follow the prefix. Matching the prefix
            // alone would let a backup directory such as "..._min128.bak" shadow the real series.
            const std::string rest = name.substr(prefix.size());
            if (rest.empty() ||
                rest.find_first_not_of("0123456789") != std::string::npos) continue;
            return sram_select_detail::firstCsvIn(e.path());
        }
        throw std::runtime_error(
            "SRAM select: no burst_weight series with max=" + std::to_string(maxBits) +
            " bits (colDim=" + std::to_string(cfg.systolicArrayColDimension) +
            ") under " + root);
    }

    // Everything else uses normal:
    //   - all roles of non-burst configs (Grouped/Toroidal without declaredBurst)
    //   - output of burst configs (output does not use a burst macro)
    return normalCsv;
}

// Parse one CACTI CSV and return each row as an SramFile. path is set to the CSV path
// with the row's bandwidth appended (so rows are distinguishable in reports).
inline std::vector<SramFile> loadCactiCsv(const std::string& csvPath) {
    std::vector<SramFile> out;
    for (const SramSpec& s : parseCactiCsv(csvPath)) {
        SramFile f;
        f.path = csvPath + " [band" + std::to_string(s.dataWidthBytes * 8) + "]";
        f.spec = s;
        out.push_back(f);
    }
    return out;
}

// Library: takes the cacti_results root, routes each (cfg, role) to its source CSV, and
// parses each source only once (cached by CSV path).
class SramLibrary {
public:
    explicit SramLibrary(std::string root) : root_(std::move(root)) {}

    // SRAM rows from the source routed for this role/mode. The returned reference stays
    // valid for the lifetime of the library (std::map nodes are stable).
    const std::vector<SramFile>& forRole(const SystolicArrayConfig& cfg, SramRole role) {
        const std::string src = cactiSourceFor(root_, cfg, role);
        auto it = cache_.find(src);
        if (it == cache_.end())
            it = cache_.emplace(src, loadCactiCsv(src)).first;
        return it->second;
    }

    // cacti_results root directory.
    const std::string& dir() const { return root_; }

private:
    std::string root_;
    std::map<std::string, std::vector<SramFile>> cache_;   // CSV path -> parsed rows
};

// From an already-parsed list, pick the one proper SRAM row for the role's required
// bandwidth, or nullptr if none qualifies. Capacity is fixed at 256 KB and is not an axis;
// capacityKB is ignored (kept for signature compatibility).
inline const SramFile* pickProperSram(const std::vector<SramFile>& all,
                                      const SystolicArrayConfig& cfg,
                                      SramRole role, int /*capacityKB*/ = -1) {
    const int needBytes = requiredSramBandwidthBytes(cfg, role);

    const SramFile* best = nullptr;
    for (const SramFile& f : all) {
        // proper: smallest dataWidth among those with dataWidth >= required.
        if (f.spec.dataWidthBytes < needBytes) continue;
        if (!best || f.spec.dataWidthBytes < best->spec.dataWidthBytes)
            best = &f;
    }
    return best;
}

// Find and return the proper SRAM for the role's required bandwidth. Throws if none exists (guard).
inline SramFile selectSram(const std::vector<SramFile>& all,
                           const SystolicArrayConfig& cfg,
                           SramRole role, int capacityKB = -1) {
    const SramFile* f = pickProperSram(all, cfg, role, capacityKB);
    if (f) return *f;

    const int needBytes = requiredSramBandwidthBytes(cfg, role);
    throw std::runtime_error(
        std::string("SRAM select: no ") + sram_select_detail::roleName(role) +
        " SRAM meets the array's required bandwidth (" + std::to_string(needBytes) +
        " bytes/cycle = band >= " + std::to_string(needBytes * 8) + " bits) in the "
        "selected CACTI source. Provide a wider row at 256KB.");
}

// Convenience overload: find the proper SRAM for (cfg, role) directly from a library.
inline SramFile selectSram(SramLibrary& lib,
                           const SystolicArrayConfig& cfg,
                           SramRole role, int capacityKB = -1) {
    return selectSram(lib.forRole(cfg, role), cfg, role, capacityKB);
}

// List of proper SRAMs meeting the role's required bandwidth (empty if none; does not throw).
// Since capacity became fixed at 256 KB the capacity sweep is gone: the list
// holds at most the single proper row, so the optimizer's (in x w x out) enumeration
// collapses to 1x1x1 (i.e. only the contracting-dimension sweep remains).
inline std::vector<SramFile> properSramsPerCapacity(const std::vector<SramFile>& all,
                                                    const SystolicArrayConfig& cfg,
                                                    SramRole role) {
    std::vector<SramFile> out;
    if (const SramFile* f = pickProperSram(all, cfg, role))
        out.push_back(*f);
    return out;
}
