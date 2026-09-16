#pragma once

// DRAM access and energy model.
//
// (A) HBM2 access-unit counting: one DRAM access moves a fixed 4096-bit (= 512-byte)
//     burst. A tile transfer is counted as ceil(bytes/512) accesses, proportional to
//     its size (previously it was one event per tile regardless of size).
// (B) Constant per-access energy: DRAM dynamic energy is computed in closed form as a
//     fixed pJ per read/write access plus a per-activation term (analytic mode), so
//     DRAMsim3 does not have to be run.
//
// The energy backend is selectable (DramEnergyMode):
//   - DramSim3 : a trace is written and DRAMsim3 + merge_results.py supply the DRAM
//                energy (original pipeline).
//   - Analytic : accesses x per-access constants, self-contained (no trace/merge needed).
// The access counters below are maintained in both modes.

// DRAM energy backend switch. Shared by Simulator/SimulationResults/Optimizer/main.
enum class DramEnergyMode { DramSim3, Analytic };

// Accumulates DRAM read/write traffic (bytes, 4096-bit accesses, estimated row
// activations) and converts it to dynamic energy in pJ for the analytic backend.
class Dram {
public:
    // HBM2 access unit: 4096 bits = 512 bytes. Must equal the timing quantum (bandwidth)
    // and, for calibration, the transaction size of the DRAMsim3 HBM2 config.
    static constexpr int accessBits  = 4096;
    static constexpr int accessBytes = accessBits / 8;   // 512
    const int bandwidth = accessBytes;                   // bytes/quantum (was a bare 512 literal)

    // Per-access *command* energy (excluding activation), in pJ per 4096-bit (512 B) access.
    // Step A/B calibration (2026-07): DRAMsim3 HBM2_8Gb_x128 measurements were split into
    //   command and activation parts. Measured read_energy/write_energy (pure command)
    //   = 6432 / 8544 pJ per 512 B (identical for both workloads). Activation energy is
    //   charged separately via activationEnergy below.
    static constexpr double readEnergyPerAccess  = 6432.0;
    static constexpr double writeEnergyPerAccess = 8544.0;

    // (B) Activation/locality term: energy per row miss (one ACT command), in pJ.
    //   Measured DRAMsim3 act_energy / num_act_cmds = 828 pJ/ACT.
    static constexpr double activationEnergy = 828.0;

    // HBM2 x128, BL=4 -> 64 B per transaction (burst). One 512 B access = 8 bursts.
    static constexpr int    burstBytes = 64;
    // Average number of bursts served per activation (row miss) when a tile is streamed
    // sequentially under the OPEN_PAGE policy with DRAMsim3's bank-interleaved address mapping.
    //   Measured read/write row-hit = 96.8% => miss rate 0.032 => 1/0.032 = 31.25 bursts/ACT.
    //   (datasheet row = 64 columns x 128-bit device = 1024 B = 16 bursts; 2-bank interleave ~= 2 KB)
    //   31.25 = 125/4, stored as a rational so the activation count can use integer ceil.
    static constexpr long   burstsPerActNum = 4;    // burstsPerActivation = 125/4 = 31.25
    static constexpr long   burstsPerActDen = 125;

    // Record one DRAM read (load) / write (write-back) transfer of `bytes` bytes.
    void addRead (long bytes) {
        readAccesses_ += accesses(bytes); readBytes_ += bytes;
        activations_  += activationsFor(bytes);
    }
    void addWrite(long bytes) {
        writeAccesses_ += accesses(bytes); writeBytes_ += bytes;
        activations_   += activationsFor(bytes);
    }

    long   getReadCount()   const { return readAccesses_; }   // now: # of 4096-bit accesses (was tiles)
    long   getWriteCount()  const { return writeAccesses_; }
    long   getReadBytes()   const { return readBytes_; }
    long   getWriteBytes()  const { return writeBytes_; }
    long   getActivations() const { return activations_; }    // (B) estimated number of row misses
    // Energies below are in pJ.
    double getReadEnergy()  const { return readAccesses_  * readEnergyPerAccess; }
    double getWriteEnergy() const { return writeAccesses_ * writeEnergyPerAccess; }
    double getActivationEnergy() const { return activations_ * activationEnergy; }
    double getEnergy() const {
        return getReadEnergy() + getWriteEnergy() + getActivationEnergy();
    }

    // Estimated row-hit rate (diagnostics/reporting): 1 - activations / total bursts.
    // Total bursts are computed from the summed byte count, so this is approximate.
    double estimatedHitRate() const {
        const long bursts = (readBytes_ + writeBytes_ + burstBytes - 1) / burstBytes;
        return bursts > 0 ? 1.0 - static_cast<double>(activations_) / bursts : 0.0;
    }

private:
    static long accesses(long bytes) {                     // ceilDiv; a partial burst still costs 1
        return bytes > 0 ? (bytes + accessBytes - 1) / accessBytes : 0;
    }
    // (B) Derive the activation count of one tile transfer (bytes) from its tiling.
    //   The transfer streams over contiguous addresses, so activations ~= ceil(nBursts/31.25).
    //   Large tiles converge to the 96.8% hit rate (~= nBursts/31.25); tiles smaller than a
    //   row (~2 KB) still cost at least one activation per transfer, so worse locality
    //   automatically yields higher energy.
    static long activationsFor(long bytes) {
        if (bytes <= 0) return 0;
        const long nBursts = (bytes + burstBytes - 1) / burstBytes;
        // ceil(nBursts / (burstsPerActDen/burstsPerActNum))
        //   = ceil(nBursts * burstsPerActNum / burstsPerActDen)
        return (nBursts * burstsPerActNum + burstsPerActDen - 1) / burstsPerActDen;
    }
    long readAccesses_  = 0;
    long writeAccesses_ = 0;
    long readBytes_     = 0;
    long writeBytes_    = 0;
    long activations_   = 0;   // (B) accumulated estimated row-miss count
};
