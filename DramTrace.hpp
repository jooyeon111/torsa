#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include "Tile.hpp"

// Converts newSim DRAM accesses (per-tile reads/writes) into a trace that DRAMsim3 can
// consume. Each trace line has the form:
//     0x<address> <READ|WRITE> <memory-clock-cycle>
//
// newSim accounts a tile transfer as a single read/write, but DRAMsim3 operates on
// bursts (= one transaction each). A tile transfer of sizeBytes bytes is therefore
// expanded into multiple BURST_BYTES (64 B) transactions.
//
// Address mapping: each tile type (A/B/C) gets its own region, and each tile is assigned a
// stable base address from that region's cursor the first time it appears (reloading the
// same tile hits the same addresses, so row-buffer reuse is modelled realistically).
class DramTrace {
public:
    // Duration of one newSim cycle (ns). Matches SimulationResults::clockPeriod (2e-9 s).
    static constexpr double NEWSIM_TCK_NS = 2.0;
    // tCK of the target DRAM (ns). HBM2_8Gb_x128 has tCK = 1 ns.
    // When using another config, changing only this value rescales the timestamps.
    static constexpr double MEM_TCK_NS = 1.0;

    static constexpr uint64_t BURST_BYTES = 64;          // HBM x128, BL=4 -> 64 B per transaction
    static constexpr uint64_t REGION_A    = 0x10000000ULL; // input (A)
    static constexpr uint64_t REGION_B    = 0x40000000ULL; // weight (B)
    static constexpr uint64_t REGION_C    = 0x80000000ULL; // output (C)

    // Open (truncate) the trace file at `path`, creating parent directories, and reset
    // all address/channel cursors. Returns false if the file could not be opened.
    // bandwidthBytesPerCycle: newSim DRAM bandwidth (= Dram::bandwidth, bytes per newSim cycle).
    // The transfer span (in memory cycles) is derived from this value so that the two models
    // (trace timeline and cycle_ accounting) stay consistent. No separate constant is
    // introduced.
    bool open(const std::string& path, int bandwidthBytesPerCycle) {
        std::filesystem::path p(path);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());
        ofs_.open(p);
        path_ = path;
        bandwidth_ = bandwidthBytesPerCycle > 0 ? bandwidthBytesPerCycle : accessBytesFallback_;
        cursorA_ = REGION_A;
        cursorB_ = REGION_B;
        cursorC_ = REGION_C;
        issueMemCycle_ = 0;   // reset the DRAM channel issue cursor
        return static_cast<bool>(ofs_);
    }

    bool isOpen() const { return ofs_.is_open(); }
    const std::string& path() const { return path_; }
    // Number of burst transactions written so far.
    long transactionCount() const { return txCount_; }

    // Flush and close the stream. Must be called at the end of run(): Optimizer child
    // processes terminate via _exit(), which skips destructors and the ofstream flush, so
    // without an explicit close the last buffered transactions (notably the write-backs
    // that cluster at the end) would never reach the file and the trace would be truncated.
    void close() { if (ofs_.is_open()) ofs_.close(); }

    // Record one tile's DRAM access as a sequence of burst transactions.
    //   cycle  : newSim cycle at which the access is requested (request arrival time)
    //   isWrite: true for a write-back, false for a load
    //
    // Previously all bursts of a transfer were stamped with the same memCycle, which
    // collapsed the trace into a delta function. Now:
    //   - the span (memory cycles) for which the transfer occupies the DRAM channel is
    //     derived from Dram::bandwidth,
    //   - the channel is shared serially through the monotonic issue cursor issueMemCycle_,
    //   - the bursts are spread evenly over [start, start+span).
    // The burst count (= txCount_) and addresses are unchanged, so dynamic energy is unaffected.
    void record(const Tile& tile, bool isWrite, long cycle) {
        if (!ofs_) return;

        const uint64_t base  = baseAddressOf(tile);
        const uint64_t bytes = static_cast<uint64_t>(tile.shape.sizeBytes());
        if (bytes == 0) return;
        const char* op = isWrite ? "WRITE" : "READ";

        // span: duration of this transfer in memory cycles. Derived from the same formula
        //       (ceilDiv(bytes, bandwidth)) newSim already uses to charge DRAM occupancy to cycle_.
        const long newSimCycles = ceilDivL(static_cast<long>(bytes), bandwidth_);
        const long span = std::max<long>(1, std::llround(
            static_cast<double>(newSimCycles) * NEWSIM_TCK_NS / MEM_TCK_NS));

        // The channel is shared serially: start at the later of the request arrival time
        // and the time the channel becomes free. If the request is later than the cursor
        // (compute-bound phase) the channel stays idle until then, preserving the real
        // idle interval (legitimate background time).
        const long start = std::max<long>(issueMemCycle_, toMemCycle(cycle));

        const long nBursts =
            static_cast<long>((bytes + BURST_BYTES - 1) / BURST_BYTES);

        // Spread the nBursts bursts evenly over [start, start+span).
        // If span < nBursts, several bursts share the same memory cycle (realistic, since
        // HBM serves multiple transactions per quantum).
        for (long i = 0; i < nBursts; ++i) {
            const long ts = start + (i * span) / nBursts;
            ofs_ << "0x" << std::hex
                 << (base + static_cast<uint64_t>(i) * BURST_BYTES) << std::dec
                 << ' ' << op << ' ' << ts << '\n';
            ++txCount_;
        }

        issueMemCycle_ = start + span;   // advance the channel cursor (monotonic, non-overlapping)
    }

private:
    // newSim cycle -> memory clock cycle (rounded to nearest).
    static long toMemCycle(long newSimCycle) {
        return static_cast<long>(std::llround(
            static_cast<double>(newSimCycle) * NEWSIM_TCK_NS / MEM_TCK_NS));
    }

    static long ceilDivL(long a, long b) { return b > 0 ? (a + b - 1) / b : a; }

    static uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

    // Address-region cursor for the given tile type.
    uint64_t& cursorFor(TileType t) {
        switch (t) {
            case TileType::TypeA: return cursorA_;
            case TileType::TypeB: return cursorB_;
            case TileType::TypeC: return cursorC_;
        }
        return cursorA_;
    }

    // Assign a stable base address to a tile object exactly once, keyed by its address
    // (pointer identity). The region cursor advances by the tile size rounded up to a burst
    // (at least one burst for empty tiles).
    uint64_t baseAddressOf(const Tile& tile) {
        const void* key = static_cast<const void*>(&tile);
        auto it = tileBase_.find(key);
        if (it != tileBase_.end()) return it->second;

        uint64_t& cur = cursorFor(tile.tileType);
        const uint64_t base = cur;
        const uint64_t span = alignUp(
            static_cast<uint64_t>(tile.shape.sizeBytes()), BURST_BYTES);
        cur += (span == 0 ? BURST_BYTES : span);
        tileBase_.emplace(key, base);
        return base;
    }

    // Safe fallback when open() did not receive a valid Dram::bandwidth (512 = 4096-bit access).
    static constexpr int accessBytesFallback_ = 512;

    std::ofstream ofs_;
    std::string   path_;
    long          txCount_ = 0;
    int           bandwidth_ = accessBytesFallback_;  // bytes/newSim-cycle (Dram::bandwidth)
    long          issueMemCycle_ = 0;                 // DRAM channel issue cursor (memory cycles)
    uint64_t      cursorA_ = REGION_A;
    uint64_t      cursorB_ = REGION_B;
    uint64_t      cursorC_ = REGION_C;
    std::unordered_map<const void*, uint64_t> tileBase_;  // tile address -> base DRAM address
};
