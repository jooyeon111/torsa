#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

// Parser for cacti_results/**/*.csv (CACTI SRAM synthesis results).
// Extracts only the values needed to build a DoubleBuffer.
//
// Previously one sram_sim_results/*.txt file (NVSim text report) described one face.
// Now one CACTI CSV row describes one face. The three sources (normal / burst_input /
// burst_weight) share the unified 21-column header below, so a single parser
// reads all of them without per-source branching:
//
//   buswidth_bits, block_B, num_subarrays, access_ns, cycle_ns, bw_GBps,
//   read_nJ, write_nJ, leak_mW, area_mm2,
//   htree_nJ, out_driver_nJ, bitlines_nJ, senseamp_nJ, wordline_nJ, decoder_nJ,
//   Ndwl, Ndbl, Nspd, acc_le_2ns, cyc_le_2ns
//
// Only columns 0..9 (buswidth_bits .. area_mm2) are consumed; the rest are ignored.
//
// num_subarrays is informational (number of banks) and differs per source: normal uses
// the minimum N that meets 2 ns at each width (1/2/2/4), burst_weight uses the N chosen
// by emit_auto, and burst_input uses the structural constant NBANK (currently 8). In
// every source read/write/leak/area are already N-way totals
// (build_cacti_results.sh:emit_row multiplies by $NB), so the parser does not multiply again.

// One SRAM macro (one CACTI CSV row), with units already converted for the simulator.
struct SramSpec {
    int    capacityBytes        = 0;   // Face capacity (bytes); fixed 256 KB for CACTI sources
    int    dataWidthBytes       = 0;   // Data width = bandwidth of one access (bytes) = buswidth_bits / 8
    double readEnergyPerAccess  = 0.0; // Read  dynamic energy (pJ / access)
    double writeEnergyPerAccess = 0.0; // Write dynamic energy (pJ / access)
    double readBandwidthTBs     = 0.0; // Read  bandwidth (TB/s)
    double writeBandwidthTBs    = 0.0; // Write bandwidth (TB/s); CACTI gives one value, same as read
    double leakagePowerW        = 0.0; // Leakage power of ONE macro (W)
    double areaUm2              = 0.0; // Total area of ONE macro (um^2). 0 if unknown (ignored by area normalization).

    // ---- Double-buffer construction -----------------------------------------
    // A double buffer is built from two "instances" of this macro (ping and pong). A single
    // single-port macro cannot serve a ping read and a pong write at the same time, so two
    // physical macros are required. One CSV row (= one face) therefore maps to:
    //   - face capacity  = capacityBytes          (fixed 256 KB for CACTI)
    //   - total storage  = 2 x capacityBytes      (512 KB total)
    //   - area/leakage   = two macros             (synthesis result x 2)
    // An N-way burst_input row is also treated as "one face": the N banks together
    // form one 256 KB face, and that face is instantiated twice (ping/pong), so area and
    // leakage are doubled.
    //
    // Counting area twice is the key point. Previously only the face capacity was used as
    // capacityBytes while area/leakage were charged for a single macro, which effectively
    // gave twice the storage for free.
    static constexpr int pingPongMacros = 2;

    // Every place that needs the face capacity (DoubleBuffer::singleBufferSize,
    // deriveContractingDimensionSize, Scheduler::outputFaceCapBytes, the tile-load
    // pre-check) should use this helper, so call sites make it visible that the
    // macro capacity equals the face capacity.
    int faceCapacityBytes() const { return capacityBytes; }

    // There is no matching helper for access energy (read/writeEnergyPerAccess): one
    // access hits only one macro, so it is not doubled.
    double totalAreaUm2()       const { return areaUm2       * pingPongMacros; }
    double totalLeakagePowerW() const { return leakagePowerW * pingPongMacros; }
};

namespace sram_detail {

// CACTI CSVs have no capacity column. All sources are fixed at 256 KB.
constexpr int kCactiCapacityBytes = 262144;   // 256 KB

// Split one CSV line on commas (no quoting support; CACTI CSVs contain none).
inline std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

// Strip leading/trailing whitespace (space, tab, CR, LF) from a cell.
inline std::string trimCell(const std::string& s) {
    std::size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    std::size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

} // namespace sram_detail

// Parse one CACTI CSV file and return one SramSpec per data row.
// The schema is unified, so there is no per-source branching. Column positions are
// fixed, so unit conversions are constants:
//   read/write : nJ -> pJ      (x1000)
//   leak       : mW -> W       (/1000)
//   area       : mm^2 -> um^2  (x1e6)
//   bw         : GB/s -> TB/s  (/1000)
// Throws if the file cannot be opened, a data row has fewer than 10 columns, or there
// are no data rows.
inline std::vector<SramSpec> parseCactiCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("CACTI parse: cannot open file \"" + path + "\"");

    std::vector<SramSpec> out;
    std::string line;
    bool headerSeen = false;

    while (std::getline(in, line)) {
        if (sram_detail::trimCell(line).empty()) continue;

        // Any non-blank line whose first cell does not start with a digit or sign is treated
        // as a header/comment and skipped (normally just the header line).
        std::vector<std::string> cols = sram_detail::splitCsv(line);
        const std::string first = sram_detail::trimCell(cols.empty() ? "" : cols[0]);
        const bool numeric = !first.empty() &&
                             (std::isdigit(static_cast<unsigned char>(first[0])) ||
                              first[0] == '+' || first[0] == '-');
        if (!numeric) { headerSeen = true; continue; }

        // Required columns: buswidth_bits(0) .. area_mm2(9).
        if (cols.size() < 10)
            throw std::runtime_error(
                "CACTI parse: expected >=10 columns in \"" + path + "\" row \"" + line + "\"");

        auto num = [&](int i) { return std::stod(sram_detail::trimCell(cols[i])); };

        SramSpec spec;
        spec.capacityBytes        = sram_detail::kCactiCapacityBytes;
        spec.dataWidthBytes       = static_cast<int>(num(0)) / 8;   // buswidth_bits / 8
        spec.readBandwidthTBs     = num(5) / 1000.0;                // bw_GBps -> TB/s
        spec.writeBandwidthTBs    = spec.readBandwidthTBs;
        spec.readEnergyPerAccess  = num(6) * 1000.0;               // read_nJ -> pJ
        spec.writeEnergyPerAccess = num(7) * 1000.0;               // write_nJ -> pJ
        spec.leakagePowerW        = num(8) / 1000.0;               // leak_mW -> W
        spec.areaUm2              = num(9) * 1e6;                   // area_mm2 -> um^2
        // num_subarrays (col 2) is informational only. Values are already N-way totals, so do not multiply.
        out.push_back(spec);
    }

    (void)headerSeen;   // currently unused; kept for diagnostics
    if (out.empty())
        throw std::runtime_error("CACTI parse: no data rows in \"" + path + "\"");
    return out;
}
