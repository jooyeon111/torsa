#pragma once
// Cycle/energy model of the systolic array itself: per-operation cycle costs
// (stationary load, streaming tile op, output drain) and the activity counters
// and power/area parameters used by the array energy model.

#include <algorithm>
#include <memory>
#include "ArrayConfig.hpp"
#include "Tile.hpp"

// Systolic array instance. Returns cycle costs for each array operation and
// accumulates activity statistics (op counts, busy cycles, useful stream cycles).
class Array{
public:

    // switchPower/internalPower in mW, leakagePower in nW, arrayAreaUm2 in um^2,
    // clockPowerFraction in [0,1] (see ArrayConfigSpec::arrayClockPowerFraction).
    Array(std::shared_ptr<const SystolicArrayConfig> c,
          double switchPower, double internalPower, double leakagePower,
          double arrayAreaUm2, double clockPowerFraction = 1.0)
        : switchPowerMw(switchPower),
          internalPowerMw(internalPower),
          leakagePowerNw(leakagePower),
          arrayArea(arrayAreaUm2),
          clockPowerFraction_(clockPowerFraction),
          config_(std::move(c))
    {
        validate();
    }

    Dataflow dataflow(void) const {
        return config_->dataflow;
    }

    const SystolicArrayConfig& config(void) const {
        return *config_;
    }

    // Number of streaming tile ops executed (incremented by cyclesForTileOp).
    int getArrayActiveCount(void) const {
        return arrayActiveCount;
    }

    // Number of physical MAC (multiplier) units in the array = peak MAC throughput per cycle.
    //   = (vector PE count) x (multipliers per PE) = totalVectorPeRow*totalVectorPeCol*multPerPe.
    // Note: systolicArrayRow/ColDimension each already include mult, so their product
    // overcounts by a factor of mult. Iso-MAC design points (e.g. 64x64x1x1x1, 4x8x4x4x8)
    // all agree at 4096.
    long getMacUnitCount(void) const {
        return static_cast<long>(config_->totalVectorPeRow)
             * config_->totalVectorPeCol
             * config_->multPerPe;
    }

    // Accumulated actual array-busy cycles (fill + stream + drain). Dynamic energy must
    // be proportional to these activity cycles, not to the *number* of tile ops (the
    // streaming MAC term stays, only the fill/drain overhead scales with tile count).
    // The Simulator passes in the cycles it computed via addBusyCycles (same values as
    // the timing cycle_ accumulation, plus drains that are hidden from wall-clock time).
    long getArrayBusyCycles(void) const {
        return arrayBusyCycles;
    }

    void addBusyCycles(long c) {
        arrayBusyCycles += c;
    }

    // Stream cycles in which data actually flowed (energy-only statistic). A subset of
    // arrayBusyCycles that excludes fill/drain and panel zero-pad tails, counting only
    // cycles with valid MAC activity. Used by the two-rate energy model
    // (SimulationResults::getArrayDynamicEnergy) as
    //   E = P_clk * busy + P_data * useful.
    // Never use it for timing (cycle_); the padded value from cyclesForTileOp remains
    // the correct timing value.
    long getUsefulStreamCycles(void) const {
        return usefulStreamCycles;
    }

    double getClockPowerFraction(void) const {
        return clockPowerFraction_;
    }

    // Switching power (mW).
    double getSwitchPower(void) const {
        return switchPowerMw;
    }

    // Internal power (mW).
    double getInternalPower(void) const {
        return internalPowerMw;
    }

    // Leakage power (nW).
    double getLeakagePower(void) const {
        return leakagePowerNw;
    }

    // Total array area (um^2).
    double getArrayArea(void) const {
        return arrayArea;
    }

    // Number of stationary-operand loads (incremented by cyclesForStationaryLoad).
    int getStationaryLoadCount(void) const {
        return stationaryLoadCount;
    }

    // Cost of filling the stationary operand into the array = preload + systolic fill latency.
    // Matches the HDL test's "1. Load Tile" loop (preload over the stationary tile's
    // contracting axis) followed by the fill wait:
    //   ReuseA: loadTile.cols + totalVectorPeCol - outerPeCol
    //   ReuseB: loadTile.rows + totalVectorPeRow - outerPeRow
    //   ReuseC (output-stationary) has no stationary load, only an accumulator clear -> 1.
    // Also increments the stationary load counter.
    long cyclesForStationaryLoad(Tile loadTile) {
        stationaryLoadCount ++;
        switch (config_->kind()) {
            case ConfigKind::Grouped:
            case ConfigKind::Toroidal:
                switch (config_->dataflow) {
                    case Dataflow::ReuseA: return loadTile.shape.cols + config_->totalVectorPeCol - config_->outerPeCol;
                    case Dataflow::ReuseB: return loadTile.shape.rows + config_->totalVectorPeRow - config_->outerPeRow;
                    case Dataflow::ReuseC: return 1;
                }
                break;
            // ToroidalDual: identical to Scala ToroidalDualModeDriver.Poker.loadStationary.
            //   for (step <- 0 until chainLen) { drive; waitRisingEdge }   -> chainLen
            //   for (_ <- 0 until g.settle) waitRisingEdge                 -> chainLen - 1
            // chainLen = totalVectorPeRow for ReuseB (the ReuseA mirror would be
            // totalVectorPeCol, but ToroidalDualConfig is fixed to ReuseB).
            //
            // Unlike stock Toroidal, the whole chain is always driven regardless of tile
            // shape; a short K-tile is zero-padded (golden.a is 0 outside K). So loadTile
            // is not read and chainLen takes the place of shape.rows.
            //
            // settle = chainLen - 1 (after the Pulse adder-tree reorg): the chain seals slot j at cycle
            // j+chainLen-1 (data has 2 register stages per VectorPE hop, capture enable
            // has 1), so the deepest slot is sealed only at 2*chainLen-2 and the first
            // stream starts on the next cycle. With the old per-PE skew
            // (Delay(inputA2, outerIndex.r)), outer hop r consumed its first stream r
            // cycles later and got the same -outerPeRow discount as stock; since dual
            // result reduction moved to the array-level DualModeAdderTree, that skew was
            // removed and every PE consumes its lane on the first stream cycle, so
            // streaming can only start after the whole chain is sealed. (Scala comment:
            // this bound is TIGHT; removing even one cycle overlaps the deepest slot that
            // is still latching and breaks the first stream step. Verified 2026-07-28:
            // 216/216, and 312/312 PASS including settle-stress/heavy.)
            case ConfigKind::ToroidalDual:
                (void)loadTile;
                return 2L * config_->totalVectorPeRow - 1;
        }
        return 1;
    }

    // Streaming (MAC) cost of one tile op = streamCycles =
    //   length of the "streaming axis" of the streamed operand as reshaped by the Scheduler.
    // Also increments the active-op counter and accumulates useful stream cycles.
    //
    // The Scheduler's reshape places the streamed operand with the same (row,col)
    // convention for Grouped and Toroidal:
    //   ReuseA: streamed = B = (tileN, tileK)  -> streaming cycles = rows = tileN
    //   ReuseB: streamed = A = (tileK, tileM)  -> streaming cycles = cols = tileM
    //   ReuseC: streamed = A = (tileM*mult, tileK/mult) -> streaming cycles = cols = tileK/mult
    // ReuseC streams the contraction along the time axis, so cols = tileK/mult already
    // reflects the mult packing via the reshape (a constant cDS could not express this).
    //
    // Exception - Grouped ReuseA/ReuseB de-skew flush:
    //   The AccSram address counter is free-running and wraps at depth, so one panel
    //   drives psumValid for numSlots * slotStride = depth cycles. The last slotStride
    //   cycles are the zero-flush that empties the de-skew chain, and short slots are
    //   zero-padded, so **a panel is always exactly depth cycles**. The tileM/tileN given
    //   by the Scheduler is already (depth - slotStride), and boundary tiles are cut
    //   further, so instead of reading the shape we return panelStreamCycles() (= depth).
    //   Toroidal has no de-skew flush (hasDeskewFlush() is false), so it keeps the
    //   original shape-based computation (Toroidal is bit-for-bit unchanged).
    //
    // Exception - fixed burst of ToroidalDual:
    //   The same free-running counter constraint applies to the dual-mode AccSram. There
    //   is no de-skew flush reservation (slotStride), but "one K-pass = exactly depth
    //   psumValid cycles" still holds, so a partial panel is zero-padded to depth as
    //   well -> panelStreamCycles() is used instead of the shape here too. See the
    //   ToroidalDual case comment below.
    //
    // Note: the streamed operand's axis is determined by the dataflow (rows = ReuseA,
    //   cols = ReuseB). Older code flipped the axis for Toroidal (ReuseA->cols,
    //   ReuseB->rows) and returned tileK (the systolic dimension) - a bug. The reshape
    //   itself was correct; only the axis being read was wrong.
    long cyclesForTileOp(Tile streamTile) {
        arrayActiveCount ++;
        usefulStreamCycles += usefulCyclesForTileOp(streamTile);   // energy only (timing unaffected)

        switch (config_->kind()) {
            case ConfigKind::Grouped:
                switch (config_->dataflow) {
                    // Grouped accumulator: one panel = depth cycles (last slot is the flush).
                    case Dataflow::ReuseA:
                    case Dataflow::ReuseB: return config_->panelStreamCycles();
                    case Dataflow::ReuseC: return streamTile.shape.cols;   // A=(.., tileK/mult)
                }
                break;

            case ConfigKind::Toroidal:
                switch (config_->dataflow) {
                    case Dataflow::ReuseA: return streamTile.shape.rows;   // B=(tileN, tileK)
                    case Dataflow::ReuseB: return streamTile.shape.cols;   // A=(tileK, tileM)
                    case Dataflow::ReuseC: return streamTile.shape.cols;   // A=(.., tileK/mult)
                }
                break;

            // ToroidalDual (ReuseB family): streamed = A = (tileK, tileN), streaming axis = cols = tileN.
            //
            // Dual mode emits numPorts (= totalVectorPeCol) C elements per cycle
            // (Scala ToroidalDualModeDriver: streamCycles = ceil(N / numPorts); one panel
            //  covers depth * numPorts elements of N). Every multiplier does valid MAC work
            // on every cycle, so a full panel is tileN/numPorts = depth cycles.
            // (Even older code returned tileN directly, underestimating throughput by numPorts.)
            //
            // However, the burst length is **always depth, regardless of how much N the
            // panel actually covers**:
            //   the AccSram address counter is free-running, wraps at depth and is not reset
            //   by firstLoop either (AccSram.scala:81-84), so one K-pass must drive exactly
            //   depth cycles of psumValid. For a tail panel where N is not divisible by
            //   span (= depth*numPorts), stream steps whose output index exceeds N are fed
            //   B=0 so those slots accumulate and drain 0 - the **content** is padded, the
            //   **burst length stays the same**.
            //   (2026-07-28 Pulse change: the `N % span != 0` skip in runAcc was removed, so
            //    partial panels became legal. newSim already counted panels with ceilDiv, so
            //    the panel count was right; only the tail panel's cycles were cut by the
            //    shape and undercounted.)
            //
            // Therefore, exactly like Grouped, the shape is not read and panelStreamCycles()
            // (= depth) is returned. For full panels this equals ceil(tileN/ports), so the
            // change only affects (a) tail panels and (b) small GEMVs with N < span that
            // have a single panel - in both, the wasted zero-pad cycles are now visible.
            // (So over-provisioning cd now shows up as a cost: the Optimizer's cd sweep
            //  actually measures that waste.)
            case ConfigKind::ToroidalDual:
                (void)streamTile;
                return config_->panelStreamCycles();

        }
        return 1;
    }

    // Number of stream cycles in one tile op in which data *actually flows* (energy only).
    // Unlike cyclesForTileOp, which returns the full panel (including zero-pad) for
    // timing, this counts the shape-based value without the zero-pad tail:
    //   Grouped ReuseA/ReuseB : min(streamed axis length, panel length) - the shape value
    //     cyclesForTileOp would return for Toroidal. The rest of the panel has zero
    //     input, so datapath switching is dead (busy-but-idle cycles).
    //   Grouped/Toroidal ReuseC, Toroidal ReuseA/ReuseB : no padding, same as timing.
    //   ToroidalDual : min(ceil(coveredN / numPorts), panel length) - the zero-pad tail of
    //     the fixed depth burst (stream steps beyond N) is busy-but-idle.
    // Fill/drain cycles are not counted here (zero useful stream).
    long usefulCyclesForTileOp(const Tile& streamTile) const {
        const long panel = config_->panelStreamCycles();
        switch (config_->kind()) {
            case ConfigKind::Grouped:
                switch (config_->dataflow) {
                    case Dataflow::ReuseA:
                        return std::min<long>(streamTile.shape.rows, panel);
                    case Dataflow::ReuseB:
                        return std::min<long>(streamTile.shape.cols, panel);
                    case Dataflow::ReuseC:
                        return streamTile.shape.cols;
                }
                break;

            case ConfigKind::Toroidal:
                switch (config_->dataflow) {
                    case Dataflow::ReuseA: return streamTile.shape.rows;
                    case Dataflow::ReuseB: return streamTile.shape.cols;
                    case Dataflow::ReuseC: return streamTile.shape.cols;
                }
                break;

            // streamed = A = (tileK, tileN) -> coveredN = cols, numPorts = totalVectorPeCol.
            case ConfigKind::ToroidalDual: {
                const long ports = config_->totalVectorPeCol;
                const long need  = (streamTile.shape.cols + ports - 1) / ports;   // ceilDiv
                return std::min<long>(need, panel);
            }
        }
        return 1;
    }

    // Cost of the drain (pipeTail) that flushes completed C tiles out of the array.
    // Identical to drainTail/pipeTail of the HDL test driver (DataflowDriver).
    //
    // Note (timeline vs energy): the new Scala hides the drain of a finished panel
    // behind the next panel's compute using the ping-pong of BigAccSramDoubleBuffer.
    // So the pipeTail of intermediate panels is not exposed on wall-clock time (cycle_);
    // only the pipeTail of the last panel, which has no following panel, is exposed on
    // the timeline (drainOutputBuffer). Energy is different: the array really toggles
    // during a time-hidden drain, so the pipeTail of intermediate panels is still charged
    // to dynamic energy via addBusyCycles (see Simulator's flushReadOutputBuffer).
    // The return value below is still the per-drain pipeTail cost.
    //
    //   ReuseA/ReuseB (input/weight-stationary):
    //     pipeTail = pipeline flush. Partial-sum accumulation is done by RMW on the
    //     output SRAM, so an array drain costs only the per-drain pipeTail.
    //       Grouped : outerPeRow + outerPeCol + log2Up(multPerPe)
    //              + log2Up(max(localPeRow, localPeCol)) + 2   (passes through both dimensions)
    //       Toroidal ReuseA : outerPeCol + log2Up(multPerPe) + log2Up(localPeCol) + 2
    //       Toroidal ReuseB : outerPeRow + log2Up(multPerPe) + log2Up(localPeRow) + 2
    //
    //   ReuseC (output-stationary): the (output-stationary) array flushes in one pass,
    //     same for Grouped/Toroidal. Fixed cost only, no k-reduction drain:
    //       3*(totalVectorPeRow + totalVectorPeCol) + 2
    //         + log2Up(multPerPe) + log2Up(max(localPeRow, localPeCol))
    //     (Same as drainTail in Scala runGroupedReuseC:562 /
    //      runToroidalReuseC:1070 / runReuseABNoAcc:1185. The old formula 2*(outerPe*)-2
    //      was wrong in two places: (a) where it said outerPe* it should be
    //      totalVectorPe*, and (b) the coefficient is 3*(...)+2, not 2*(...)-2 - the Scala
    //      comment explains that a 2x2 minimum array is one cycle short and the last row
    //      is cut off. The 07-23 sync verified only the ReuseA/B pipeTail and left ReuseC
    //      unchecked.)
    //
    //   ToroidalDual: same formula as Toroidal ReuseB (see the case comment below).

    //TODO fix this not using this model
    long cyclesForOutput(){
        const long oRow = config_->outerPeRow;
        const long oCol = config_->outerPeCol;
        const long tRow = config_->totalVectorPeRow;
        const long tCol = config_->totalVectorPeCol;

        const int  lMul    = detail::log2Up(config_->multPerPe);
        const int  lMaxLoc = detail::log2Up(std::max(config_->localPeRow, config_->localPeCol));

        // The ReuseC drainTail is shared by Grouped and Toroidal.
        const long reuseCDrainTail = 3 * (tRow + tCol) + 2 + lMul + lMaxLoc;

        switch (config_->kind()) {
            case ConfigKind::Grouped:
                switch (config_->dataflow) {
                    case Dataflow::ReuseA:
                    case Dataflow::ReuseB: {
                        const long pipeTail = oRow + oCol + lMul + lMaxLoc + 2;
                        return pipeTail;
                    }
                    case Dataflow::ReuseC:
                        return reuseCDrainTail;
                }
                break;

            case ConfigKind::Toroidal:
                switch (config_->dataflow) {
                    case Dataflow::ReuseA: {
                        const long pipeTail =
                            oCol + lMul + detail::log2Up(config_->localPeCol) + 2;
                        return pipeTail;
                    }
                    case Dataflow::ReuseB: {
                        const long pipeTail =
                            oRow + lMul + detail::log2Up(config_->localPeRow) + 2;
                        return pipeTail;
                    }
                    case Dataflow::ReuseC:
                        return reuseCDrainTail;
                }
                break;
            // ToroidalDual: identical to Scala ToroidalDualModeDriver.Geometry.latency.
            //   1 (dualStreamBuffer entry stage) + outerPeRow + 1 (multiplier output register)
            //   + log2Up(multPerPe) + log2Up(localVectorPeRow) (in-PE adder trees)
            // This ends up being the same formula as the Toroidal ReuseB pipeTail. Older code
            // lacked the log2Up(localPeRow) term, which is actually wrong now that
            // localPeRow > 1 is allowed.
            //
            // Semantic change after the adder-tree reorg (phase 1): the dual result is now
            // reduced not by the C-chain but by the array-level DualModeAdderTree (depth
            // log2(outerPeRow*localPeRow)), so the real tree path is shallower than the
            // formula above. However, phase 1 inserted a compensating Delay at the
            // collection point so the dual output arrives exactly at the stock chain depth
            // in Pulse, so the formula above (including
            // the outerPeRow term) is still cycle-exact. When phase 2 (remove compensation +
            // accControl retime) lands, the outerPeRow term should shrink to the form
            // log2Up(outerPeRow*localPeRow) - log2Up(localPeRow); update this formula then.
            case ConfigKind::ToroidalDual: {
                const long pipeTail =
                    oRow + lMul + detail::log2Up(config_->localPeRow) + 2;
                return pipeTail;
            }
            }
        return 1;
    }

private:
    int arrayActiveCount = 0;       // streaming tile ops executed
    long arrayBusyCycles = 0;       // fill + stream + drain cycles (timed or hidden)
    long usefulStreamCycles = 0;    // stream cycles with real data (subset of busy)
    int stationaryLoadCount = 0;    // stationary operand loads
    double switchPowerMw;
    double internalPowerMw;
    double leakagePowerNw;
    double arrayArea;               // um^2
    // Fraction of total streaming power that is clock-tree/register internal power
    // burned "unconditionally on every clocked cycle". 1.0 = the old flat charging
    // (all busy cycles at full power). Comes from arrayClockPowerFraction in the .cfg
    // (see ArrayConfigParser).
    double clockPowerFraction_ = 1.0;

    std::shared_ptr<const SystolicArrayConfig> config_;

    // Validates the power/area parameters (assert-based).
    //   Array PE power is always required regardless of dataflow.
    //   (Accumulation is done by RMW on the output SRAM, so there is no separate
    //   accumulator validation.)
    void validate() const {
        assert(switchPowerMw   > 0.0 && "array switch power must be positive");
        assert(internalPowerMw > 0.0 && "array internal power must be positive");
        assert(leakagePowerNw  > 0.0 && "array leakage power must be positive");
        assert(arrayArea       > 0.0 && "array area must be positive");
    }

};