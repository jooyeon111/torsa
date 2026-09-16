#pragma once
#include <algorithm>
#include <climits>
#include "ArrayConfig.hpp"
#include "Layer.hpp"

// ============================================================================
// Derivation of the feasibility ceiling for contractingDimensionSize (= cd).
//
// cd is not K (the contraction axis) but the free/streaming tile length. The value
// returned here is the largest cd for which the front tile of the streamed operand still
// fits into one face (single buffer) of its SRAM, i.e. an **upper bound**, not an optimum.
//
// Note (correction of an earlier premise): an older comment claimed "a larger cd always
// helps, because longer streaming raises utilization". That is **false for ReuseB RMW**.
// In ReuseB, cd -> tileM, so psumTile = cd * tileN * psumBytes grows linearly with cd,
// which shrinks the number of partial-sum tiles that fit on the output face,
// gTotal = outputFace / psumTile, and starves the n-blocking nb (Scheduler::deriveBlocking).
// nb is the only means of amortizing A along the n axis, so A's DRAM traffic grows by a
// multiple. Meanwhile cd does not reduce A traffic itself
// (aTile * mTiles ~= M * rowBytes is invariant in cd) -- cd only buys a few %p of utilization.
// Measured: input cap128->cap256 (cd 2048->4096) gave nb 4->2 and 1.83x more DRAM reads.
//
// cd is therefore a free variable to be chosen together with (nb, g) under this ceiling.
// The Optimizer uses this value only as the upper bound of its search axis and picks the
// actual cd by per-candidate simulation (Optimizer::cdCandidates / Optimizer::evaluate).
// A single run (main) uses the ceiling as-is unless the .cfg overrides cd; for the best cd,
// run a cd sweep or the optimizer.
//
// What cd actually binds depends on (kind, dataflow) (derived in the Scheduler reshape*
// leaves):
//   ReuseB (Grouped/Toroidal): cd -> tileM               -> tile A (colA) -> INPUT SRAM,  streamDim = M
//   ToroidalDual             : cd -> tileN = cd * ports  -> tile A (colA) -> INPUT SRAM,  streamDim = ceil(N/ports)
//   ReuseA (Grouped/Toroidal): cd -> tileN               -> tile B (rowB) -> WEIGHT SRAM, streamDim = N
//   ReuseC                   : cd unused (tileM/tileN fixed by array geometry) -> any value works (returns 1)
//
// The front tile (m=0, k=0) is rowBytes * colBytes bytes, and cd forms one of those axes.
// The other axis (rowBytes) is independent of cd -- it is the "bytes per unit of cd" -- so:
//   maxCd = min(streamDim, floor(faceCap / rowBytes))
// rounded down. cd beyond streamDim is clamped by tileExtent in emitTiles and gives no
// benefit, hence the cap at streamDim.
//
// rowBytes derivation (must match the reshape* leaves exactly):
//   ReuseB      : tile A rowA = min(systolicArrayRowDimension, dimK)
//   ToroidalDual: tile A rowA = min(systolicArrayRowDimension, dimK) * ports
//   ReuseA      : tile B colB = min(systolicArrayColDimension, dimK)
//   (bytesPerElement of tiles A/B is the makeA/makeB default = inputOperandBitWidth/8.)
//
// For Grouped ReuseA/ReuseB, cd is the AccSram depth and the free-axis length covered by a
// tile is depth - slotStride (one slot is reserved for the de-skew flush), so the extent
// ceiling computed above is converted back into a depth ceiling. In that case the return
// value is always a valid depth (a multiple of span, >= 2*span).
//
// Returns: the largest feasible cd (= search upper bound). Returns 0 when even cd==1 (or,
//   for Grouped ReuseA/ReuseB, even one span) does not fit into one SRAM face; the caller
//   decides how to handle it (main throws, the Optimizer skips the candidate).
// ============================================================================
inline int deriveContractingDimensionSize(const SystolicArrayConfig& geom,
                                          const Layer& layer,
                                          long inputFaceCapBytes,
                                          long weightFaceCapBytes) {
    const int dimM = layer.gemmDimenion.dimensionM;
    const int dimN = layer.gemmDimenion.dimensionN;
    const int dimK = layer.gemmDimenion.dimensionK;
    const int bpe  = geom.inputOperandBitWidth / 8;   // bytes per tile A/B element (1 for 8-bit operands)

    int  streamDim = 0;
    long rowBytes  = 0;   // bytes per unit of cd = size of the streamed tile's "other" axis
    long faceCap   = 0;

    switch (geom.kind()) {
        case ConfigKind::ToroidalDual: {
            // cd is the number of stream steps (= AccSram depth), and each step consumes
            // numPorts (= totalVectorPeCol) elements of N -> tileN = cd x ports.
            // Hence (a) the stream-axis bound is ceil(dimN/ports) rather than dimN, and
            // (b) each unit of cd costs ports times as many bytes. Omitting the ports factor
            // would overestimate the feasible cd by a factor of ports, and the front tile
            // would overflow one face of the input SRAM.
            const long ports = geom.totalVectorPeCol;
            streamDim = static_cast<int>((dimN + ports - 1) / ports);
            rowBytes  = static_cast<long>(std::min(geom.systolicArrayRowDimension, dimK)) * bpe * ports;
            faceCap   = inputFaceCapBytes;
            break;
        }
        case ConfigKind::Grouped:
        case ConfigKind::Toroidal:
            switch (geom.dataflow) {
                case Dataflow::ReuseB:
                    streamDim = dimM;
                    rowBytes  = static_cast<long>(std::min(geom.systolicArrayRowDimension, dimK)) * bpe;
                    faceCap   = inputFaceCapBytes;
                    break;
                case Dataflow::ReuseA:
                    streamDim = dimN;
                    rowBytes  = static_cast<long>(std::min(geom.systolicArrayColDimension, dimK)) * bpe;
                    faceCap   = weightFaceCapBytes;
                    break;
                case Dataflow::ReuseC:
                    return 1;   // ReuseC does not use cd when building tiles.
            }
            break;
    }

    if (rowBytes <= 0) return 0;
    const long fit = faceCap / rowBytes;                 // round down
    const long cd  = std::min<long>(streamDim, fit);
    if (cd < 1) return 0;

    // Grouped ReuseA/ReuseB: cd is the AccSram depth (freeDimensionSize), and the free-axis
    // length actually covered by one tile (= panel) is depth - slotStride. The value computed
    // above is "the longest tile that fits into one SRAM face" = the ceiling on that extent,
    // so add back the slot reserved for the de-skew flush to turn it into a depth ceiling.
    //   depth = floor((extentCeil + span)/span)*span  ->  extent = depth - span <= extentCeil
    // If that valid depth is below the minimum of 2 slots (2*span), raise it to the minimum
    // panel. The extent is then span, so if even span does not fit into a face (fit < span)
    // the configuration is infeasible.
    //   (Even when streamDim < span on layers with small M (e.g. GEMV), depth is a hardware
    //    provisioning value, so the minimum panel can still be built -- the tile is clamped to M.)
    if (geom.hasDeskewFlush() && geom.accDeskewSpan > 1) {
        const long span  = geom.accDeskewSpan;
        const long depth = geom.roundDownFreeDimensionSize(
                               static_cast<int>(std::min<long>(cd + span, INT_MAX)));
        if (depth >= 2 * span) return static_cast<int>(depth);
        return fit >= span ? static_cast<int>(2 * span) : 0;
    }
    return static_cast<int>(cd);
}
