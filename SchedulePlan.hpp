#pragma once
#include <vector>
#include <ostream>
#include <string>
#include "Tile.hpp"
#include "PartialSum.hpp"

// SchedulePlan: static plan produced by the Scheduler. It has no notion of time;
// it only describes *what* must be done (which tiles exist and the ordered list
// of partial-sum ops). The Simulator copies it and assigns cycles/energy.
struct SchedulePlan {
    std::vector<Tile>        tilesA;   // A(m, k): input tiles
    std::vector<Tile>        tilesB;   // B(k, n): weight tiles
    std::vector<TileC>       tilesC;   // C(m, n): output tiles
    std::vector<PartialSumId> ops;      // unit ops C[m][n][k], in execution order

    // Name of the layer this plan targets (used to label result reports).
    std::string layerName;

    // Logical GEMM dimensions (before tiling). Used for throughput metrics (e.g. TOPS).
    int dimM = 0;
    int dimN = 0;
    int dimK = 0;

    // Tile count per axis. tilesA/tilesB/tilesC are stored densely (no gaps) in
    // the order below, so they can be looked up by index in O(1) (Simulator::findTile*):
    //   tilesA[m*kTiles + k]  (m outer, k inner)
    //   tilesB[k*nTiles + n]  (k outer, n inner)
    //   tilesC[m*nTiles + n]  (m outer, n inner)
    // Exception: for ToroidalDual, tilesB holds a single canonical tile per k,
    // indexed as tilesB[k] (see Scheduler::emitTiles).
    // Filled by Scheduler::emitTiles. 0 means unset (empty plan).
    int mTiles = 0;
    int nTiles = 0;
    int kTiles = 0;

    // Maximum bytes one RMW accumulation block can occupy on a single output SRAM
    // face: min(g, mTiles) * min(nBlock, nTiles) * (largest partial-sum tile bytes).
    // At a block boundary (PartialSumId::endsBlock) the Simulator checks whether the
    // next block fits on the write face and drains the face only if it does not.
    // Because g and nBlock are derived from psumFront (the largest tile) with
    // g*nBlock <= face capacity / psumFront, this value never exceeds one face
    // (as long as a single partial-sum tile fits, which checkSramSize asserts).
    long maxResidentBlockBytes = 0;   // 0 = unset (e.g. empty plan); only consulted on endsBlock ops (ReuseB)

    // Debug dump of all tiles and ops.
    void print(std::ostream& os) const {
        printSection(os, "tilesA", tilesA);
        printSection(os, "tilesB", tilesB);
        printSection(os, "tilesC", tilesC);
        printSection(os, "ops",    ops);
    }

private:
    template <class T>
    static void printSection(std::ostream& os, const char* name,
                             const std::vector<T>& items) {
        os << name << " (" << items.size() << ")\n";
        for (const auto& x : items) os << "  " << x << "\n";
    }
};