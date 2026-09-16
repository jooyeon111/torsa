#pragma once
#include "ArrayConfig.hpp"
#include "Tile.hpp"
#include <cassert>

// On-chip SRAM double (ping/pong) buffer model: two faces of equal capacity, one being
// read by the computation while the other is prefetched. Tracks per-face occupancy and
// accumulates access counts for energy accounting.
class DoubleBuffer{

public:
    //byte (capacity of one face, i.e. one SRAM macro)
    const int singleBufferSize;
    //byte (both faces combined = 2 x singleBufferSize)
    const int totalBuffersize;
    //byte (bytes moved per access = SRAM data width)
    const int singleBufferBandwidth;
    //pJ per access
    const double readEnergyPerAccess;
    //pJ per access
    const double writeEnergyPerAccess;
    //W (static leakage power; multiplied by cycles x clockPeriod to obtain leakage energy)
    const double leakagePowerW;
    //um^2 (physical area; enters the denominator of the area-normalized efficiency topsPerWattPerMm2)
    const double areaUm2;

    DoubleBuffer(
            int sBufferSize,
            int sBufferBandwidth,
            double rAccessEnergy,
            double wAccessEnergy,
            double leakagePower,
            double area = 0.0
        )
        : singleBufferSize(sBufferSize),
          totalBuffersize(sBufferSize * 2),
          singleBufferBandwidth(sBufferBandwidth),
          readEnergyPerAccess(rAccessEnergy),
          writeEnergyPerAccess(wAccessEnergy),
          leakagePowerW(leakagePower),
          areaUm2(area)
    { validate(); }

    // ---- ping/pong read-face state ----------------------------------------
    // Face currently read by the computation. Initially ping, because firstFillUp() fills ping first.
    bool isReadingPing() const { return readingPing_; }

    TileLocation currentReadLocation() const {
        return readingPing_ ? TileLocation::PingBuffer
                            : TileLocation::PongBuffer;
    }

    // The face opposite the read face (prefetch target).
    TileLocation currentWriteLocation() const {
        return readingPing_ ? TileLocation::PongBuffer
                            : TileLocation::PingBuffer;
    }

    // Exchange the roles of the read and write faces.
    void swap() { readingPing_ = !readingPing_; }

    // ---- per-face occupancy (bytes) ---------------------------------------
    // Every load/unload must be paired with a matching occupy()/release()/freeBytes() update.
    int usedOf(TileLocation side) const {
        return side == TileLocation::PingBuffer ? pingUsed_ : pongUsed_;
    }

    int freeSpace(TileLocation side) const {
        return singleBufferSize - usedOf(side);
    }

    bool canFit(int bytes, TileLocation side) const {
        return usedOf(side) + bytes <= singleBufferSize;
    }

    // Mark `bytes` as occupied on the given face (no capacity check; call canFit() first).
    void occupy(int bytes, TileLocation side) {
        usedRef(side) += bytes;
    }

    void release(TileLocation side) {   // empty the whole face after write-back/consumption
        usedRef(side) = 0;
    }

    // When unloading a single tile from a face (e.g. LRU eviction), subtract only that
    // tile's bytes (clamped at 0).
    void freeBytes(int bytes, TileLocation side) {
        usedRef(side) -= bytes;
        if (usedRef(side) < 0) usedRef(side) = 0;
    }

    // ---- counters ---------------------------------------------------------
    // Access energy is proportional to the number of real accesses (one data-width word =
    // singleBufferBandwidth bytes). readEnergyPerAccess is the energy of one access, so a
    // transfer of `bytes` is converted to ceil(bytes / singleBufferBandwidth) accesses.
    // (Same ceil-div model as Dram.hpp. Computing bytes x perByte would count partial-word
    //  accesses as fractional accesses and misestimate energy relative to real hardware.)
    //   addRead/addWrite(bytes): accumulate the transfer as accesses (bytes kept for reference).
    void addRead(long bytes)  { readAccesses_  += accessesFor(bytes); readBytes_  += bytes; ++readCount;  }
    void addWrite(long bytes) { writeAccesses_ += accessesFor(bytes); writeBytes_ += bytes; ++writeCount; }

    // Number of addRead/addWrite calls (transfers), not accesses.
    int  getReadCount(void)  const { return readCount; }
    int  getWriteCount(void) const { return writeCount; }
    long getReadAccesses(void)  const { return readAccesses_; }
    long getWriteAccesses(void) const { return writeAccesses_; }
    long getReadBytes(void)  const { return readBytes_; }
    long getWriteBytes(void) const { return writeBytes_; }

    // Static leakage power (W). Consumed continuously while powered, independent of access count.
    double getLeakagePower(void) const { return leakagePowerW; }

    // Physical area (um^2). Summed into the denominator of the area-normalized efficiency (topsPerWattPerMm2).
    double getArea(void) const { return areaUm2; }

    // Dynamic access energies in pJ.
    double getReadEnergy()  const { return readAccesses_  * readEnergyPerAccess; }
    double getWriteEnergy() const { return writeAccesses_ * writeEnergyPerAccess; }
    double getTotalEnergy() const { return getReadEnergy() + getWriteEnergy(); }

private:

    int& usedRef(TileLocation side) {
        return side == TileLocation::PingBuffer ? pingUsed_ : pongUsed_;
    }

    // ceilDiv: a partial-word access still counts as one access (same as Dram::accesses).
    long accessesFor(long bytes) const {
        return bytes > 0 ? (bytes + singleBufferBandwidth - 1) / singleBufferBandwidth : 0;
    }

    // Sanity checks on construction (active only when NDEBUG is not defined).
    void validate() const {
        assert(singleBufferSize      > 0   && "single buffer size must be positive");
        assert(totalBuffersize == singleBufferSize * 2 && "total buffer size must be twice the single buffer size");
        assert(singleBufferBandwidth > 0   && "single buffer bandwidth must be positive");
        assert(readEnergyPerAccess       >= 0.0 && "read energy per access must be non-negative");
        assert(writeEnergyPerAccess      >= 0.0 && "write energy per access must be non-negative");
        assert(leakagePowerW >= 0.0 && "leakage power must be non-negative");
        assert(areaUm2 >= 0.0 && "area must be non-negative");

        // Per-face occupancy must stay within [0, singleBufferSize].
        assert(pingUsed_ >= 0 && pingUsed_ <= singleBufferSize && "ping buffer occupancy out of range");
        assert(pongUsed_ >= 0 && pongUsed_ <= singleBufferSize && "pong buffer occupancy out of range");

        assert(readCount  >= 0 && "read count must be non-negative");
        assert(writeCount >= 0 && "write count must be non-negative");
    }

    int  readCount  = 0;    // number of addRead calls (reference only)
    int  writeCount = 0;    // number of addWrite calls (reference only)
    long readAccesses_  = 0;  // accumulated read accesses (basis for energy)
    long writeAccesses_ = 0;  // accumulated write accesses
    long readBytes_  = 0;   // accumulated bytes read (reference only)
    long writeBytes_ = 0;   // accumulated bytes written

    bool readingPing_ = true;
    int  pingUsed_ = 0;
    int  pongUsed_ = 0;
};