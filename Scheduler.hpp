#pragma once
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <climits>
#include <cstdlib>

#include "ArrayConfig.hpp"
#include "Layer.hpp"
#include "Tile.hpp"
#include "PartialSum.hpp"
#include "SchedulePlan.hpp"

// Scheduler.hpp
//
// Builds the static SchedulePlan for one GEMM layer on a given systolic-array
// configuration:
//   1) picks the GEMM-space tile sizes (tileM/tileN/tileK) from the array kind
//      (Grouped / Toroidal / ToroidalDual) and dataflow (ReuseA / ReuseB / ReuseC),
//   2) emits the A(m,k), B(k,n) and C(m,n) tiles with their physical (reshaped)
//      row/col shapes,
//   3) derives the output-side blocking factors (g, nBlock) from SRAM face
//      capacities, and
//   4) emits the ordered list of partial-sum ops C[m][n][k].
// The plan carries no timing; the Simulator assigns cycles and energy.

class Scheduler {
public:
    // outputFaceCapBytes: byte capacity of one output SRAM face (single buffer).
    //   Used to derive G (= maxResidentPsumTiles), which blocks the RMW accumulation
    //   (ReuseB) so that the number of simultaneously resident partial-sum tiles fits
    //   in this capacity. It is the output-side counterpart of cd, which sets the
    //   input-side tiling from the input/weight SRAM.
    //   Default (LONG_MAX) means "no blocking" (G >= mTiles) -> bit-for-bit identical
    //   to the unblocked schedule.
    // weightFaceCapBytes: byte capacity of one weight SRAM face. With ReuseB n-blocking,
    //   the nb B(k,n) tiles of one block must be resident at the same time, so this
    //   bounds nb from above.
    //   Default (LONG_MAX) means "no n-blocking" (nb = 1) -> bit-for-bit identical
    //   to the unblocked schedule.
    Scheduler(Layer l, std::shared_ptr<const SystolicArrayConfig> c,
              long outputFaceCapBytes = LONG_MAX,
              long weightFaceCapBytes = LONG_MAX)
        : layer_(std::move(l)), config_(std::move(c)),
          outputFaceCapBytes_(outputFaceCapBytes),
          weightFaceCapBytes_(weightFaceCapBytes) {
        buildTiles();
    }

    const Layer& layer() const { return layer_; }
    const SystolicArrayConfig& config() const { return *config_; }

    // The Simulator copies this plan and owns its copy.
    const SchedulePlan& plan() const { return plan_; }

    // Debug dump of the generated plan.
    void print(std::ostream& os = std::cout) const { plan_.print(os); }

private:
    Layer layer_;
    std::shared_ptr<const SystolicArrayConfig> config_;
    long outputFaceCapBytes_;   // bytes per output SRAM face (LONG_MAX = unbounded)
    long weightFaceCapBytes_;   // bytes per weight SRAM face (LONG_MAX = n-blocking off)
    SchedulePlan plan_;

    static int ceilDiv(int a, int b) { return (a + b - 1) / b; }

    // Extent actually covered by the idx-th tile along a GEMM axis of length dim.
    // The last tile uses the remainder (dim - idx*tile) and is not padded to tile.
    // (Interior tiles get the full tile size; a small axis with dim < tile returns dim.)
    static int tileExtent(int idx, int tile, int dim) {
        const int start = idx * tile;
        return std::min(tile, dim - start);
    }

    // Byte width of one element of the in-progress partial sum (accumulator), i.e. the
    // maximum partial-sum width just before requantization (output generation).
    // The product of input A (aBytes) and weight B (bBytes) is aBits+bBits wide, and
    // accumulating over K contraction elements adds ceil(log2(K)) guard bits.
    // The result is that bit count rounded up to whole bytes.
    static int partialSumBytesPerElement(int contractionK, int aBytes, int bBytes) {
        const int productBits = aBytes * 8 + bBytes * 8;          // bit width of the A*B product
        const int guardBits   = detail::log2Up(contractionK);     // growth from summing K products
        const int elemBits    = productBits + guardBits;
        return (elemBits + 7) / 8;                                // round up to bytes
    }

    // Physical shapes actually loaded into the array.
    // Distinct from the tileM/tileN/tileK used for counting (GEMM space): these hold the
    // row/col of each cut tile after it is reshaped for the systolic array.
    struct ReshapedShapes {
        int rowA = 0, colA = 0;
        int rowB = 0, colB = 0;
        int rowC = 0, colC = 0;
    };

    // Dispatch tile-size selection on the array kind.
    void buildTiles() {
        switch (config_->kind()) {
            case ConfigKind::Grouped: buildTilesGrouped(); break;
            case ConfigKind::Toroidal: buildTilesToroidal(); break;
            case ConfigKind::ToroidalDual: buildTilesToroidalDual(); break;
        }
    }

    void buildTilesGrouped() {
        // Note: contractingDimensionSize is NOT K (contraction) but the free dimension
        //   (= Scala freeDimensionSize = AccSram depth D). The actual K is handled by
        //   tileK (spatial axis) and kTiles (accumulation passes).
        //
        // Grouped ReuseA/ReuseB reserve the last slot of each panel for the de-skew
        // zero-flush (tilesPerPanel = numSlots - 1), so the free-axis length a panel
        // really covers is D - slotStride rather than D -> freeAxisTileExtent().
        //   Here one tile = one panel, so the tile count ceil(dim/(D-s)) matches Scala's
        //   numPanels = ceil(ceil(dim/s) / (D/s - 1)) to within one panel.
        //   Cycles are charged as D per panel by Array::cyclesForTileOp.
        int tileM = 0, tileN = 0, tileK = 0;
        switch (config_->dataflow) {
            case Dataflow::ReuseA:
                tileM = config_->totalVectorPeRow;
                tileN = config_->freeAxisTileExtent();
                tileK = config_->systolicArrayColDimension;
                break;
            case Dataflow::ReuseB:
                tileM = config_->freeAxisTileExtent();
                tileN = config_->totalVectorPeCol;
                tileK = config_->systolicArrayRowDimension;
                break;
            case Dataflow::ReuseC:
                // Grouped ReuseC contracts along the time axis (the number of cycles
                // accEnableC is held). No array dimension bounds that cycle count, so the
                // former tileK = systolicArrayColDimension was a verification artifact,
                // not an architectural limit (K=64 streamed in one pass on a colDim-16
                // array and PASSed). It is now read from the provisioned
                // maxContractionSize.
                tileM = config_->totalVectorPeRow;
                tileN = config_->totalVectorPeCol;
                tileK = config_->effectiveMaxContractionSize();
                break;
            default:
                assert(false && "unsupported dataflow");
        }
        emitTiles(tileM, tileN, tileK);
    }

    void buildTilesToroidal() {
        // Note: contractingDimensionSize is NOT K but the free dimension
        //   (ReuseA -> tileN, ReuseB -> tileM). See the note in buildTilesGrouped.
        //   Toroidal has slotStride == depth (numSlots == 1), so no slot is reserved for
        //   the de-skew flush and the free-axis tile length is the full depth -> unlike
        //   Grouped, no correction is applied.
        int tileM = 0, tileN = 0, tileK = 0;
        switch (config_->dataflow) {
            case Dataflow::ReuseA:
                // tileM = config_->systolicArrayColDimension;
                // tileN = config_->contractingDimensionSize;
                // tileK = config_->totalVectorPeCol;
                tileM = config_-> totalVectorPeRow;      // FIX: M-output = row dim
                tileN = config_-> contractingDimensionSize;
                tileK = config_-> systolicArrayColDimension;
                break;
            case Dataflow::ReuseB:
                // tileM = config_->systolicArrayRowDimension;
                // tileN = config_->totalVectorPeCol;
                // tileK = config_->contractingDimensionSize;
                tileM = config_-> contractingDimensionSize;
                tileN = config_-> totalVectorPeCol;      // FIX: N-output = col dim
                tileK = config_-> systolicArrayRowDimension;
                break;
            case Dataflow::ReuseC:
                // Toroidal ReuseC k-tile depth is the cluster-loop length circulated by
                // the wrap-around = contractionPassDepth = outerPeRow * multPerPe
                // (Scala GoldenMatrix.scala:327 -> SystolicArrayConfig.scala:94).
                // Broadcast (localPeRow/Col > 1) only widens output rows/cols in parallel
                // and does not deepen K. Using systolicArrayRowDimension (= oRow*lRow*mult)
                // would overstate the depth by lRow and undercount kTiles by lRow.
                tileM = config_->totalVectorPeRow;
                tileN = config_->totalVectorPeCol;
                tileK = config_->outerPeRow * config_->multPerPe;
                break;
            default:
                assert(false && "unsupported dataflow");
        }
        emitTiles(tileM, tileN, tileK);
    }

    void buildTilesToroidalDual() {
        int tileM = 0, tileN = 0, tileK = 0;
        switch (config_->dataflow) {
            case Dataflow::ReuseB:
                // ToroidalDual is GEMV-only (M==1). General GEMM (M>1) is resolved to
                // Toroidal ReuseB before the config is built
                // (ArrayConfigSpec::resolveBurstForLayerM), so only M==1 reaches this
                // path. The invariant is enforced here.
                assert(layer_.gemmDimenion.dimensionM == 1 &&
                       "ToroidalDual schedule requires M==1 (GEMV); M>1 must be resolved "
                       "to Toroidal ReuseB before config construction");
                tileM = layer_.gemmDimenion.dimensionM;
                // cd is Scala's freeDimensionSize (= AccSram depth = stream steps per
                // K-pass). Dual mode emits numPorts (= totalVectorPeCol) C elements per
                // stream step, so the N covered by one panel is depth * numPorts
                // (ToroidalDualModeDriver.runAcc: span = depth * g.numPorts).
                // tileN counts N-axis elements, hence the multiplication by ports here;
                // Array::cyclesForTileOp divides by ports again to recover depth cycles.
                tileN = config_->contractingDimensionSize * config_->totalVectorPeCol;
                tileK = config_->systolicArrayRowDimension;
                break;

            default:
                assert(false && "unsupported dataflow for ToroidalDual");
        }
        emitTiles(tileM, tileN, tileK);
    }

    // Populate plan_ (tiles, blocking, ops) for GEMM-space tile sizes tileM/N/K.
    void emitTiles(int tileM, int tileN, int tileK) {
        const GemmDimension gd = layer_.gemmDimenion;

        // Store the layer name and logical GEMM dims in the plan (for reports/throughput).
        plan_.layerName = layer_.layerName;
        plan_.dimM = gd.dimensionM;
        plan_.dimN = gd.dimensionN;
        plan_.dimK = gd.dimensionK;

        // Tile counts are taken in logical GEMM space (M, N, K).
        const int mTiles = ceilDiv(gd.dimensionM, tileM);
        const int nTiles = ceilDiv(gd.dimensionN, tileN);
        const int kTiles = ceilDiv(gd.dimensionK, tileK);

        // Keep per-axis tile counts for dense indexing (O(1) lookup in Simulator::findTile*).
        // The push order below must match them (m*kTiles+k / k*nTiles+n / m*nTiles+n).
        plan_.mTiles = mTiles;
        plan_.nTiles = nTiles;
        plan_.kTiles = kTiles;

        plan_.tilesA.clear();
        plan_.tilesB.clear();
        plan_.tilesC.clear();
        plan_.ops.clear();
        plan_.tilesA.reserve(mTiles * kTiles);
        plan_.tilesB.reserve(kTiles * nTiles);
        plan_.tilesC.reserve(mTiles * nTiles);
        plan_.ops.reserve(mTiles * nTiles * kTiles);

        // Small dimensions (dim < tile): tile *counts* are computed above in tile units
        // (ceilDiv), but the tile size passed to the physical reshape is clamped to the
        // actual layer dimension.
        //   On a streaming axis the tile (e.g. ReuseB tileM = contractingDimensionSize) is
        //   a large budget meaning the maximum streamable cycles. Without clamping, layers
        //   with small M/N/K (e.g. decode with M=1) would get physical tiles much larger
        //   than the real data, overstating SRAM load size, streaming cycles and DRAM
        //   bytes; in the worst case a single tile exceeds one input SRAM face and the
        //   simulation fails.
        //   (For axes with dim >= tile, min returns tile, so behavior is unchanged.)
        const int effTileM = std::min(tileM, gd.dimensionM);
        const int effTileN = std::min(tileN, gd.dimensionN);
        const int effTileK = std::min(tileK, gd.dimensionK);

        // Each tile's physical shape is computed from the GEMM range it actually covers.
        // Previously every tile was uniformly effTile, so the last tile was "padded" to
        // the full tile size (if M was not divisible by cd, streaming/accumulation bytes
        // were overstated as ceil(M/cd)*cd). Giving the boundary tile the remainder makes
        // the per-axis byte sum equal the logical dimension exactly, removing the padding
        // artifact.
        //   Axes a tile is not indexed by (N for A, M for B, K for C) receive the
        //   representative value (effTile). Most leaves do not use that axis for the
        //   corresponding row/col; the exception is reshapeToroidalDual (colA = tileN,
        //   colB = tileM*mult), where it is the clamped layer dimension.
        for (int m = 0; m < mTiles; ++m) {
            const int em = tileExtent(m, tileM, gd.dimensionM);
            for (int k = 0; k < kTiles; ++k) {
                const int ek = tileExtent(k, tileK, gd.dimensionK);
                const ReshapedShapes rs = reshapeTilesPhysically(em, effTileN, ek);
                plan_.tilesA.push_back(Tile::makeA({m, k}, rs.rowA, rs.colA));
            }
        }

        // In ToroidalDual (GEMV), tilesB are physically k-slices of the input vector A
        // (reshapeToroidalDual: rowB = tileK/mult, colB = tileM*mult, independent of n).
        // Building a (k,n) grid would store the same data as nTiles separate tiles and
        // charge a DRAM reload per n block (vector traffic overstated nTiles times).
        // Instead one canonical tile per k is built, and Simulator::findTileB / loadTiles
        // look it up by the compact index k in dual mode. Non-dual keeps the (k,n) grid.
        const int nTilesB =
            (config_->kind() == ConfigKind::ToroidalDual) ? 1 : nTiles;
        for (int k = 0; k < kTiles; ++k) {
            const int ek = tileExtent(k, tileK, gd.dimensionK);
            for (int n = 0; n < nTilesB; ++n) {
                const int en = tileExtent(n, tileN, gd.dimensionN);
                const ReshapedShapes rs = reshapeTilesPhysically(effTileM, en, ek);
                plan_.tilesB.push_back(Tile::makeB({k, n}, rs.rowB, rs.colB));
            }
        }

        // Byte width of one accumulator element: the maximum width just before
        // requantization, grown from the A/B byte widths plus accumulation over the full
        // contraction K (not per-tile K).
        const int aBytes = plan_.tilesA.empty() ? 1 : plan_.tilesA.front().shape.bytesPerElement;
        const int bBytes = plan_.tilesB.empty() ? 1 : plan_.tilesB.front().shape.bytesPerElement;
        const int psumBytes = partialSumBytesPerElement(gd.dimensionK, aBytes, bBytes);

        // Each C(m,n) accumulates the partial sums for k = 0..kTiles-1, so its
        // partialSumCount must be kTiles. (The Simulator calls addPartialSum() once per
        // op, and exactly kTiles ops target each C(m,n).)
        for (int m = 0; m < mTiles; ++m) {
            const int em = tileExtent(m, tileM, gd.dimensionM);
            for (int n = 0; n < nTiles; ++n) {
                const int en = tileExtent(n, tileN, gd.dimensionN);
                const ReshapedShapes rs = reshapeTilesPhysically(em, en, effTileK);
                plan_.tilesC.push_back(TileC({m, n}, kTiles, rs.rowC, rs.colC, psumBytes));
            }
        }

        const Blocking blk = deriveBlocking(mTiles, nTiles, kTiles);
        const int g  = blk.g;
        const int nb = blk.nBlock;

        // Optional debug trace of tiling and blocking (enable with SCHED_DEBUG=1).
        if (std::getenv("SCHED_DEBUG")) {
            std::cerr << "[sched] tileM=" << tileM << " tileN=" << tileN
                      << " tileK=" << tileK
                      << " | mTiles=" << mTiles << " nTiles=" << nTiles
                      << " kTiles=" << kTiles
                      << " | Atile=" << plan_.tilesA.front().shape.sizeBytes()
                      << "B Btile=" << plan_.tilesB.front().shape.sizeBytes()
                      << "B psumTile=" << plan_.tilesC.front().partialSumSizeBytes()
                      << "B | outFaceCap=" << outputFaceCapBytes_
                      << " wgtFaceCap=" << weightFaceCapBytes_
                      << " g=" << g << " nb=" << nb << "\n";
        }

        emitOps(mTiles, nTiles, kTiles, g, nb);

        // Maximum bytes one RMW block can occupy on a face. The Simulator uses it at block
        // boundaries to decide whether to drain ("does the next block fit on the write
        // face?"). With n-blocking, one block is (g tiles on m) x (nb tiles on n) partial
        // sums, all resident simultaneously while the k accumulation runs. By construction
        // g*nb*psumFront <= one face (given a single tile fits), so after a drain the
        // next block always fits.
        if (!plan_.tilesC.empty()) {
            const long psumFront = plan_.tilesC.front().partialSumSizeBytes();
            const long blkTiles  = static_cast<long>(std::min(g, mTiles))
                                 * static_cast<long>(std::min(nb, nTiles));
            plan_.maxResidentBlockBytes = blkTiles * psumFront;
        }
    }

    // The two ReuseB blocking factors. They share the same output face
    // (g*nBlock <= G_total).
    struct Blocking {
        int g      = 1;   // block size on the m axis (partial-sum tiles)
        int nBlock = 1;   // block size on the n axis (times each A(m,k) is reused while resident)
    };

    // Derive g and nBlock from the SRAM capacities.
    //
    // Background: if n is the outermost loop, every fixed n sweeps the whole A(m,k)
    // space and the next n sweeps it again from the start. All of A (hundreds of MB)
    // fits on no face, so the LRU in ensureResident misses every time -> A is reloaded
    // from DRAM nTiles times (input SRAM amortization = 1.0x).
    // Splitting n into chunks of nBlock and moving it inward lets nBlock B(k,n) tiles of
    // the block pass through the array while one A(m,k) tile stays resident, cutting A
    // reloads to ceil(nTiles/nBlock).
    //
    // The two axes' costs move in opposite directions:
    //   A DRAM bytes = Atile * mTiles*kTiles * ceil(nTiles/nBlock)   -> decreases with nBlock
    //   B DRAM bytes = Btile * nTiles*kTiles * ceil(mTiles/g)        -> decreases with g
    // But all g*nBlock partial sums of a block must be resident on the output face while
    // the k accumulation runs, so g*nBlock <= G_total and they compete for capacity.
    // Hence nBlock is searched exhaustively minimizing total DRAM bytes, with the
    // remaining capacity given to g (g = min(G_total/nBlock, mTiles)).
    //
    // nBlock has two upper bounds (plus nTiles):
    //   - weight face: the nBlock B(k,n) tiles of one block must be resident together.
    //   - output face: g >= 1 requires nBlock <= G_total.
    // If weightFaceCapBytes_ is the default (LONG_MAX), n-blocking is disabled
    // (nBlock = 1) and the plain G derivation is used -> bit-for-bit identical to the
    // unblocked schedule. Non-ReuseB dataflows also return nBlock = 1.
    Blocking deriveBlocking(int mTiles, int nTiles, int kTiles) const {
        Blocking b;
        b.g = deriveGroupBlock();

        // Only ReuseB wires n-blocking (ReuseA has no production config; ReuseC is not RMW).
        if (config_->dataflow != Dataflow::ReuseB) return b;
        if (weightFaceCapBytes_ == LONG_MAX) return b;
        if (plan_.tilesA.empty() || plan_.tilesB.empty() || plan_.tilesC.empty()) return b;

        const long aTile = plan_.tilesA.front().shape.sizeBytes();
        const long bTile = plan_.tilesB.front().shape.sizeBytes();
        const long psum  = plan_.tilesC.front().partialSumSizeBytes();
        if (aTile <= 0 || bTile <= 0 || psum <= 0) return b;

        const long gTotal = outputFaceCapBytes_ / psum;      // total partial-sum tiles that fit on one face
        if (gTotal < 1) return b;                            // not even one tile fits -> fall back to plain G

        const long nbFitWeight = weightFaceCapBytes_ / bTile;
        long nbMax = std::min<long>({nbFitWeight, gTotal, nTiles});
        if (nbMax < 1) return b;

        long bestBytes = LONG_MAX;
        for (long nb = 1; nb <= nbMax; ++nb) {
            const long g = std::min<long>(gTotal / nb, mTiles);
            if (g < 1) continue;
            const long aBytes = aTile * mTiles * kTiles * ceilDivL(nTiles, nb);
            const long bBytes = bTile * nTiles * kTiles * ceilDivL(mTiles, g);
            const long total  = aBytes + bBytes;
            if (total < bestBytes) {
                bestBytes = total;
                b.nBlock  = static_cast<int>(nb);
                b.g       = static_cast<int>(g);
            }
        }
        return b;
    }

    // Integer ceil(a/b) for positive long operands.
    static long ceilDivL(long a, long b) { return (a + b - 1) / b; }

    // Number of partial-sum tiles that can be resident on one output SRAM face at once
    // (G = maxResidentPsumTiles). Instead of requiring a whole accumulation group (all m
    // for a fixed n) to be resident during ReuseB RMW accumulation, the accumulation axis
    // is blocked so that only G tiles are resident at a time. (ReuseA would be the
    // symmetric case but is not wired; see emitOps.) Blocking only reorders ops (tile
    // shapes/counts are unchanged), so it is orthogonal to cd tiling and never
    // double-counts.
    //   psumTileBytes uses the front tile (= largest), so G is a safe (conservative)
    //   lower bound; Simulator::checkSramSize asserts G >= 1 using the same front tile.
    //   Returns 1 if the plan has no C tiles or not even one tile fits; clamps to INT_MAX.
    // With the default outputFaceCapBytes_ (LONG_MAX), G >= mTiles -> blocking is a no-op.
    int deriveGroupBlock() const {
        if (plan_.tilesC.empty()) return 1;
        const long psumTileBytes = plan_.tilesC.front().partialSumSizeBytes();
        if (psumTileBytes <= 0) return 1;
        const long g = outputFaceCapBytes_ / psumTileBytes;   // floor
        if (g < 1) return 1;                                  // at least 1 even if no tile fits
        if (g > INT_MAX) return INT_MAX;
        return static_cast<int>(g);
    }

    // ===================== reshape dispatch =====================
    // The reshape depends on three axes: (kind, dataflow, tile type).
    // kind and dataflow are independent, so dispatch is two-level; each leaf then
    // fills all six (row,col) values for A/B/C at once. ToroidalDual has a single
    // leaf since it only supports ReuseB.

    ReshapedShapes reshapeTilesPhysically(int tileM, int tileN, int tileK) const {
        switch (config_->kind()) {
            case ConfigKind::Grouped: return reshapeGrouped(tileM, tileN, tileK);
            case ConfigKind::Toroidal: return reshapeToroidal(tileM, tileN, tileK);
            case ConfigKind::ToroidalDual: return reshapeToroidalDual(tileM, tileN, tileK);
        }
        assert(false && "unsupported kind");
        return {};
    }

    ReshapedShapes reshapeGrouped(int tileM, int tileN, int tileK) const {
        switch (config_->dataflow) {
            case Dataflow::ReuseA: return reshapeGroupedReuseA(tileM, tileN, tileK);
            case Dataflow::ReuseB: return reshapeGroupedReuseB(tileM, tileN, tileK);
            case Dataflow::ReuseC: return reshapeGroupedReuseC(tileM, tileN, tileK);
        }
        assert(false && "unsupported dataflow");
        return {};
    }

    ReshapedShapes reshapeToroidal(int tileM, int tileN, int tileK) const {
        switch (config_->dataflow) {
            case Dataflow::ReuseA: return reshapeToroidalReuseA(tileM, tileN, tileK);
            case Dataflow::ReuseB: return reshapeToroidalReuseB(tileM, tileN, tileK);
            case Dataflow::ReuseC: return reshapeToroidalReuseC(tileM, tileN, tileK);
        }
        assert(false && "unsupported dataflow");
        return {};
    }

    // ===================== reshape leaves =====================
    // Each leaf fills all six (row,col) values for A/B/C, mapping the GEMM-space tile
    // (tileM, tileN, tileK) to the physical array layout. The operand named by the
    // dataflow (A for ReuseA, B for ReuseB, both for ReuseC) is folded by multPerPe:
    // its K extent becomes ceil(tileK/mult) (padded up, see ceilToMultiple) and its
    // other extent is multiplied by mult. The other operand keeps its GEMM extents
    // (possibly transposed). C is always (tileM, tileN).
    // (config_ gives access to systolicArrayRow/ColDimension, multPerPe, etc.)
    //
    // Note: A.cols and B.rows need not match after physical mapping. The Simulator
    //       (loadStationary etc.) costs each tile from its own actual shape rather
    //       than assuming a shared contracting dim. The colA == rowB assert is only
    //       active in the ReuseC leaves, where both A and B are streamed and must agree.
    //       The commented-out lines in each leaf are the original identity mapping.

    ReshapedShapes reshapeGroupedReuseA(int tileM, int tileN, int tileK) const {
        ReshapedShapes s;
        // Original identity mapping for ReuseA(Grouped) (kept for reference):
        // s.rowA = tileM; s.colA = tileK;
        // s.rowB = tileK; s.colB = tileN;
        // s.rowC = tileM; s.colC = tileN;
        // assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");

        s.rowA = tileM * config_->multPerPe;
        s.colA = ceilToMultiple(tileK, config_->multPerPe) / config_->multPerPe;

        s.rowB = tileN;
        s.colB = tileK;

        s.rowC = tileM;
        s.colC = tileN;

        // assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");

        return s;
    }

    ReshapedShapes reshapeGroupedReuseB(int tileM, int tileN, int tileK) const {
        ReshapedShapes s;
        // Original identity mapping for ReuseB(Grouped) (kept for reference):
        // s.rowA = tileM; s.colA = tileK;
        // s.rowB = tileK; s.colB = tileN;
        // s.rowC = tileM; s.colC = tileN;
        // assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");

        s.rowA = tileK;
        s.colA = tileM;

        s.rowB = ceilToMultiple(tileK, config_->multPerPe) / config_->multPerPe;
        s.colB = tileN * config_->multPerPe;

        s.rowC = tileM;
        s.colC = tileN;

        // assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");
        return s;
    }

    ReshapedShapes reshapeGroupedReuseC(int tileM, int tileN, int tileK) const {
        ReshapedShapes s;
        // Original identity mapping for ReuseC(Grouped) (kept for reference):
        // s.rowA = tileM; s.colA = tileK;
        // s.rowB = tileK; s.colB = tileN;
        // s.rowC = tileM; s.colC = tileN;
        // assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");
        s.rowA = tileM * config_->multPerPe;
        s.colA = ceilToMultiple(tileK, config_->multPerPe) / config_->multPerPe;

        s.rowB = ceilToMultiple(tileK, config_->multPerPe) / config_->multPerPe;
        s.colB = tileN * config_->multPerPe;

        s.rowC = tileM;
        s.colC = tileN;

        assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");
        return s;
    }

    ReshapedShapes reshapeToroidalReuseA(int tileM, int tileN, int tileK) const {
        ReshapedShapes s;
        // Original identity mapping for ReuseA(Toroidal) (kept for reference):
        // s.rowA = tileM; s.colA = tileK;
        // s.rowB = tileK; s.colB = tileN;
        // s.rowC = tileM; s.colC = tileN;
        // assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");
        s.rowA = tileM * config_->multPerPe;
        s.colA = (ceilToMultiple(tileK, config_->multPerPe)) / config_->multPerPe;

        s.rowB = tileN;
        s.colB = tileK;

        s.rowC = tileM;
        s.colC = tileN;

        // assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");
        return s;
    }

    ReshapedShapes reshapeToroidalReuseB(int tileM, int tileN, int tileK) const {
        ReshapedShapes s;
        // Original identity mapping for ReuseB(Toroidal) (kept for reference):
        // s.rowA = tileM; s.colA = tileK;
        // s.rowB = tileK; s.colB = tileN;
        // s.rowC = tileM; s.colC = tileN;
        // assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");
        s.rowA = tileK;
        s.colA = tileM;

        s.rowB = ceilToMultiple(tileK, config_->multPerPe) / config_->multPerPe;
        s.colB = tileN * config_->multPerPe ;

        s.rowC = tileM;
        s.colC = tileN;

        // assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");
        return s;
    }

    ReshapedShapes reshapeToroidalReuseC(int tileM, int tileN, int tileK) const {
        ReshapedShapes s;
        // Original identity mapping for ReuseC(Toroidal) (kept for reference):
        // s.rowA = tileM; s.colA = tileK;
        // s.rowB = tileK; s.colB = tileN;
        // s.rowC = tileM; s.colC = tileN;
        // assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");
        s.rowA = tileM * config_->multPerPe;
        s.colA = (ceilToMultiple(tileK, config_->multPerPe)) / config_->multPerPe;

        s.rowB = (ceilToMultiple(tileK, config_->multPerPe)) / config_->multPerPe;
        s.colB = tileN * config_->multPerPe;

        s.rowC = tileM;
        s.colC = tileN;
        
        assert(s.colA == s.rowB && "reshaped A.col must match B.row (contracting dim)");
        return s;
    }

    ReshapedShapes reshapeToroidalDual(int tileM, int tileN, int tileK) const {
        // M>1 is resolved to Toroidal ReuseB before config construction, so this path is
        // GEMV-only (M==1). A: (tileK, tileN); B (vector slice): (tileK/mult, tileM*mult).
        // Note tileK/mult is a floor division here, unlike the ceil used in other leaves.
        ReshapedShapes s;

        // Previous mapping with A and B swapped (kept for reference):
        // s.rowA = tileK / config_->multPerPe;
        // s.colA = tileM * config_->multPerPe;

        // s.rowB = tileK;
        // s.colB = tileN;

        s.rowA = tileK;
        s.colA = tileN;

        s.rowB = tileK / config_->multPerPe;
        s.colB = tileM * config_->multPerPe;

        s.rowC = tileM;
        s.colC = tileN;

        return s;
    }

    // Emit the ordered partial-sum ops into plan_.ops.
    //
    // groupBlock (G): number of partial-sum tiles on ReuseB's accumulation axis (M)
    // kept resident at once. nBlock (nb): number of n tiles grouped into one block.
    // ReuseB splits N into chunks of nb and, within each, M into chunks of G; it finishes
    // the K accumulation of one (mBlock x nBlock) block (only that block's partial-sum
    // tiles are resident in output SRAM), marks the last op with endsBlock so the
    // Simulator can drain, then moves to the next block:
    //   for nBlock in chunks(nTiles, nb): for mBlock in chunks(mTiles, G):
    //     for k: for m in mBlock: for n in nBlock
    // Stationary B(k,n) is re-streamed at each block boundary (the Simulator charges the
    // reload naturally), so output SRAM never holds more than G*nb partial-sum tiles.
    // With G >= mTiles and nb == 1 there is one m-block per n, identical to the
    // unblocked order (for n: for k: for m).
    // ReuseA: for m: for k: for n (no blocking). ReuseC: for m: for n: for k.
    void emitOps(int mTiles, int nTiles, int kTiles, int groupBlock, int nBlock) {
        const int g  = std::max(1, groupBlock);
        const int nb = std::max(1, nBlock);
        switch (config_->dataflow) {
            case Dataflow::ReuseA:
                // Symmetric case: the accumulation axis is N. There is no production
                // ReuseA config yet, so blocking is not wired. If needed, split nTiles
                // into chunks of G as ReuseB does for mTiles below.
                for (int m = 0; m < mTiles; ++m)
                    for (int k = 0; k < kTiles; ++k)
                        for (int n = 0; n < nTiles; ++n)
                            plan_.ops.push_back(PartialSumId{m, n, k});
                break;

            case Dataflow::ReuseB:
                // Split n into chunks of nb and place it innermost. While one A(m,k)
                // tile is resident, the nb B(k,n) tiles of the block pass through the
                // array, cutting A's DRAM reloads from nTiles to ceil(nTiles/nb) (so the
                // input SRAM actually amortizes).
                // One block = partial-sum tiles (m in [mBlk,mEnd)) x (n in [nBlk,nEnd)),
                // all completed by the last op of the k accumulation -> that op is the
                // drain point (endsBlock).
                for (int nBlk = 0; nBlk < nTiles; nBlk += nb) {
                    const int nEnd = std::min(nBlk + nb, nTiles);
                    for (int mBlk = 0; mBlk < mTiles; mBlk += g) {
                        const int mEnd = std::min(mBlk + g, mTiles);
                        for (int k = 0; k < kTiles; ++k)
                            for (int m = mBlk; m < mEnd; ++m)
                                for (int n = nBlk; n < nEnd; ++n) {
                                    const bool endsBlock = (k == kTiles - 1)
                                                        && (m == mEnd - 1)
                                                        && (n == nEnd - 1);
                                    plan_.ops.push_back(PartialSumId{m, n, k, endsBlock});
                                }
                    }
                }
                break;

            case Dataflow::ReuseC:
                for (int m = 0; m < mTiles; ++m)
                    for (int n = 0; n < nTiles; ++n)
                        for (int k = 0; k < kTiles; ++k)
                            plan_.ops.push_back(PartialSumId{m, n, k});
                break;
            
            default:
                assert(false && "unsupported dataflow");
        }

        // Former single loop order for all dataflows (kept for reference):
        // for (int m = 0; m < mTiles; ++m)
        //     for (int n = 0; n < nTiles; ++n)
        //         for (int k = 0; k < kTiles; ++k)
        //             plan_.ops.push_back(PartialSumId{m, n, k});
    }


    // Round value up to the nearest multiple of `multiple`.
    // Pulse's shiftUp/shiftLeft/blockTranspose pad the contraction (tileK) up to a
    // multiple of multPerPe (`if (c % p == 0) c else ((c/p)+1)*p`). To match that padding
    // convention this rounds with ceil rather than to-nearest.
    //   (For current configs tileK is always a multiple of mult so the result is the
    //    same, but when it is not, rounding to nearest could round down and diverge
    //    from Pulse.)
    static int ceilToMultiple(int value, int multiple) {
        assert(multiple > 0 && "multiple must be positive");
        assert(value >= 0 && "value must be non-negative");
        const int q = (value + multiple - 1) / multiple;   // ceil div
        return q * multiple;
    }

};