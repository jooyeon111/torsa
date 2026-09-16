#pragma once
// Tile model: a rectangular block of an operand (A, B) or of the output (C)
// together with its shape, byte size and current storage location
// (DRAM or one of the ping/pong on-chip SRAM buffers).
#include <ostream>
#include "TileType.hpp"

// Where a tile currently resides. None = not materialised yet (e.g. a TileC
// that has not started accumulating).
enum class TileLocation { None, DRAM, PingBuffer, PongBuffer };

inline const char* toString(TileLocation loc) {
    switch (loc) {
        case TileLocation::None:       return "None";
        case TileLocation::DRAM:       return "DRAM";
        case TileLocation::PingBuffer: return "Ping";
        case TileLocation::PongBuffer: return "Pong";
    }
    return "Unknown";
}

// Tile index in the tile grid. Meaning depends on the tile type:
// A = (m, k), B = (k, n), C = (m, n).
struct TileCoord { int row; int col; };

// Physical (possibly reshaped) tile dimensions in elements.
struct TileShape {
    int rows = 0;
    int cols = 0;
    int bytesPerElement = 1;                 // 8-bit input/weight = 1 byte
    int numel()     const { return rows * cols; }                  // number of elements
    int sizeBytes() const { return rows * cols * bytesPerElement; } // actual size in bytes
};

// Operand/output tile with a mutable storage location.
class Tile {
public:
    const TileCoord coord;
    const TileType tileType;
    const TileShape shape;
    TileLocation location;

    Tile(TileCoord c, TileType tType, TileShape s,
         TileLocation loc = TileLocation::DRAM)
        : coord(c), tileType(tType), shape(s), location(loc) {}

    // Operand-A tile of shape (tileM x tileK), initially in DRAM.
    static Tile makeA(TileCoord c, int tileM, int tileK, int bytesPerElement = 1) {
        return Tile(c, TileType::TypeA, {tileM, tileK, bytesPerElement}, TileLocation::DRAM);
    }

    // Operand-B tile of shape (tileK x tileN), initially in DRAM.
    static Tile makeB(TileCoord c, int tileK, int tileN, int bytesPerElement = 1) {
        return Tile(c, TileType::TypeB, {tileK, tileN, bytesPerElement}, TileLocation::DRAM);
    }

    bool isInDram()  const { return location == TileLocation::DRAM; }
    bool isInPing()  const { return location == TileLocation::PingBuffer; }
    bool isInPong()  const { return location == TileLocation::PongBuffer; }
    // True if the tile is held in either on-chip (ping/pong) buffer.
    bool isOnChip()  const {
        return location == TileLocation::PingBuffer
            || location == TileLocation::PongBuffer;
    }
    // Location-only state transitions; transfer costs are accounted elsewhere.
    void loadToPing()      { location = TileLocation::PingBuffer; }
    void loadToPong()      { location = TileLocation::PongBuffer; }
    void writeBackToDram() { location = TileLocation::DRAM; }
};

//We assumed TileC as completed output which means it is regarded as requantized
//In ReuseA and ReuseB it is re-quantized when DRAM reads SRAM
//In ReuseC it is re-quantized inside of PE
//
// A TileC is complete once partialSumCount partial sums (one per K tile) have
// been accumulated into it.
class TileC final : public Tile {
public:
    const int partialSumCount;    // number of partial sums needed to complete this tile
    // Bytes per element of the partial sum (accumulator) while accumulating.
    // shape.bytesPerElement is the output width of the *completed*, requantized
    // TileC, but until then the output SRAM holds the wider partial sum
    // (Scheduler::partialSumBytesPerElement: A bits + B bits + ceil(log2(K)) guard
    // bits, rounded up to whole bytes).
    const int partialSumBytesPerElement;
    int generatedCount = 0;       // partial sums accumulated so far
    bool isGenerated = false;     // true once generatedCount >= partialSumCount

    TileC(TileCoord c, int psCount, int tileM, int tileN,
          int psumBytesPerElement,
          int bytesPerElement = 1,
          TileLocation loc = TileLocation::None)
        : Tile(c, TileType::TypeC, {tileM, tileN, bytesPerElement}, loc),
          partialSumCount(psCount),
          partialSumBytesPerElement(psumBytesPerElement) {}

    // Maximum byte size of the partial-sum tile accumulated up to requantization
    // (i.e. right before the output is produced). Use this value when placing
    // (occupying) or reserving space for the accumulator in the output SRAM;
    // shape.sizeBytes() is the smaller requantized TileC size and does not match
    // the footprint during accumulation.
    int partialSumSizeBytes() const {
        return shape.numel() * partialSumBytesPerElement;
    }

    // Record one more accumulated partial sum; marks the tile complete when done.
    void addPartialSum() {
        ++generatedCount;
        if (generatedCount >= partialSumCount) isGenerated = true;
    }
};

// Debug print: tile type with its grid coordinate and location.
inline std::ostream& operator<<(std::ostream& os, const Tile& t) {
    switch (t.tileType) {
        case TileType::TypeA:
            os << "A(m=" << t.coord.row << ", k=" << t.coord.col << ")";
            break;
        case TileType::TypeB:
            os << "B(k=" << t.coord.row << ", n=" << t.coord.col << ")";
            break;
        case TileType::TypeC:
            os << "C(m=" << t.coord.row << ", n=" << t.coord.col << ")";
            break;
    }
    os << "  location=" << toString(t.location);
    return os;
}

// Debug print for TileC: coordinate, accumulation progress, completion and location.
inline std::ostream& operator<<(std::ostream& os, const TileC& t) {
    os << "C(m=" << t.coord.row << ", n=" << t.coord.col << ")"
       << "  " << t.generatedCount << "/" << t.partialSumCount
       << "  done=" << (t.isGenerated ? "true" : "false")
       << "  location=" << toString(t.location);
    return os;
}
