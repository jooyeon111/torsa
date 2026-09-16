// Simulator.hpp
//
// Cycle-level execution engine for TORSA. Given a SchedulePlan (tiles A/B/C
// and the ordered list of partial-sum ops), it walks every op through the
// memory hierarchy DRAM <-> double-buffered SRAM (input A, weight B, output C)
// <-> systolic PE array, and accounts for:
//   - wall-clock cycles (cycle_), where DRAM/SRAM data-movement cost is hidden
//     behind concurrent array-busy cycles whenever possible,
//   - array busy cycles (dynamic energy of the PE array),
//   - SRAM read/write volume (bytes) per buffer and DRAM read/write volume,
//   - an optional DRAM transaction trace for DRAMsim3 energy estimation.
// At the end of run() the totals are collected into SimulationResults.
#pragma once
#include <utility>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <unordered_map>
#include "Dataflow.hpp"
#include "SchedulePlan.hpp"
#include "Array.hpp"
#include "Dram.hpp"
#include "DoubleBuffer.hpp"
#include "SimulationResults.hpp"
#include "DramTrace.hpp"

// Simulates one layer (one SchedulePlan) on a given array configuration.
// The Array, Dram and the three DoubleBuffers are owned by the caller and are
// mutated in place (statistics accumulate in them); the plan is copied because
// tile locations/progress are updated during simulation.
class Simulator {
public:
    // dramMode selects how DRAM energy is obtained:
    //   DramSim3 : write a DRAM trace file for an external DRAMsim3 run,
    //   Analytic : no trace; DRAM dynamic energy is included in the results directly.
    Simulator(SchedulePlan plan,
              Array& array,
              Dram& dram,
              DoubleBuffer& inputBuffer,
              DoubleBuffer& outputBuffer,
              DoubleBuffer& weightBuffer,
              DramEnergyMode dramMode = DramEnergyMode::DramSim3)
        : plan_(std::move(plan)),
          array_(array),
          dram_(dram),
          inputBuffer_(inputBuffer),
          outputBuffer_(outputBuffer),
          weightBuffer_(weightBuffer),
          dramMode_(dramMode)
    {
        // Initially a C tile does not exist anywhere yet (accumulation happens in
        // the PE array/accumulator or in output SRAM). When it starts being
        // produced it is placed on the write side (Ping/Pong) of the output buffer,
        // and when that side fills up it is flushed to DRAM:  None -> Ping/Pong -> DRAM.
        // (The Scheduler sets it to PingBuffer purely cosmetically, so normalize to
        //  None here; otherwise tiles not yet placed would be wrongly picked up by
        //  the drain loops.)
        for (TileC& c : plan_.tilesC)
            c.location = TileLocation::None;
    }

    // Sanity-check (via assert) that each SRAM single side is large enough for
    // the minimum unit the chosen dataflow must hold on-chip at once.
    void checkSramSize() {

        const SystolicArrayConfig& cfg = array_.config();

        assert (inputBuffer_.singleBufferSize >= plan_.tilesA.front().shape.sizeBytes() &&
         "Input Buffer A cannot contain single tile A");
        assert (weightBuffer_.singleBufferSize >= plan_.tilesB.front().shape.sizeBytes() &&
         "Weight buffer B cannot cnotain single tiel B");

        switch(cfg.dataflow) {
            case Dataflow::ReuseA:
            case Dataflow::ReuseB: {
                // RMW accumulation (ReuseA/ReuseB, including ToroidalDual): previously
                // an entire accumulation group (ReuseB = all m tiles for a fixed n,
                // ReuseA = all n tiles for a fixed m) had to reside on a single output
                // SRAM side at once. For layers with huge M such as prefill attention
                // this floor grew to hundreds of MB, which no realistic SRAM could meet.
                //
                // Now the Scheduler derives G (= maxResidentPsumTiles) from the output
                // side capacity and emits the accumulation axis in blocks of G (see op
                // ordering), so at most G partial-sum tiles ever reside on one side.
                // Hence only the minimum condition for blocking is checked here: one
                // partial-sum tile must fit on one side (G >= 1). The front tile is the
                // largest, which matches exactly how the Scheduler derives G.
                const long psumTileBytes = plan_.tilesC.front().partialSumSizeBytes();
                assert(outputBuffer_.singleBufferSize >= psumTileBytes &&
                    "Output SRAM single buffer cannot hold even one partial-sum tile "
                    "(G < 1); provide a larger output SRAM");
                break;
            }
            case Dataflow::ReuseC:
                assert (outputBuffer_.singleBufferSize >= plan_.tilesC.front().shape.sizeBytes() &&
                 "Output SRAM single buffer cannot contain single tile C");
                break;
        }
    }

    // Number of C tiles along the m (row) / n (column) direction, used to size
    // accumulation groups.
    int countTilesM() const {
        int m = 0;
        for (const TileC& c : plan_.tilesC) m = std::max(m, c.coord.row + 1);
        return m;
    }
    int countTilesN() const {
        int n = 0;
        for (const TileC& c : plan_.tilesC) n = std::max(n, c.coord.col + 1);
        return n;
    }

    // Run the whole plan: initial SRAM fill, every op, final output drain, then
    // collect/print/write results (and close the DRAM trace if tracing).
    void run() {

        if (tracing())
            // Pass the DRAM bandwidth (bytes per newSim cycle) so the trace derives
            // transaction spans with the same formula as the cycle_ accounting
            // (do not duplicate the constant).
            dramTrace_.open(traceFileName(), dram_.bandwidth);

        firstFillUp();

        //   inAccCycle  : array-busy cycles accumulated since the last input load (reset in loadTiles)
        //   outAccCycle : array-busy cycles accumulated since the last output drain (reset in flushReadOutputBuffer)
        long inAccCycle  = 0;
        long outAccCycle = 0;

        // Per-op pipeline. The accumulation/output model depends on the dataflow:
        //   ReuseC (output-stationary): k-reduction is accumulated inside the array,
        //     and storeOutput() writes the completed TileC to output SRAM once.
        //   ReuseA/ReuseB (RMW): the array only produces partial sums, which are
        //     accumulated directly in output SRAM (RMW) inside executePartialSum
        //     -> storeOutput() is a no-op.
        for (const PartialSumId& op : plan_.ops) {
            prepareInputs(op, inAccCycle);                    // 1) input SRAM: make A/B operands ready
            executePartialSum(op, inAccCycle, outAccCycle);    // 2) PE array: stationary load + MAC + accumulation
            storeOutput(outAccCycle);                         // 3) output SRAM: write completed TileC (ReuseC only)
            // 4) RMW (ReuseA/ReuseB): when an accumulation block ends (= all tiles on this
            //    side are complete), swap the side out if the next block would no longer
            //    fit on the current write side. Draining "only at a clean boundary, and
            //    only when the next block does not fit" guarantees that:
            //      (a) completed tiles never pile up beyond the side's capacity, and
            //      (b) a new block can never start on a full side that still holds
            //          incomplete tiles (the cause of ensureOutputWriteResidency
            //          aborting at small cd).
            //    Draining happens only at block boundaries (no incomplete tiles), so the
            //    swap is always legal, and several blocks are collected until the side is
            //    full before draining at once, preserving the original drain cadence.
            if (op.endsBlock && usesRmwOutput() &&
                !outputBuffer_.canFit(plan_.maxResidentBlockBytes,
                                      outputBuffer_.currentWriteLocation()))
                drainCompletedBlock(outAccCycle);
        }

        drainOutputBuffer();

        // printState();

        const bool isFinished = finished();
        if (isFinished) {
            std::cout << "Simulation finished. cycles=" << cycle_ << "\n";
        } else {
            std::cerr << "Error: all operations executed but "
                         "some C tiles are not generated\n";
        }

        // Aggregate, store and print the final results at the end of simulation.
        //   In Analytic mode the DRAM dynamic energy is included in the result totals.
        results_.dramEnergyIncluded = (dramMode_ == DramEnergyMode::Analytic);
        results_.collect(cycle_, isFinished,
                         array_, dram_,
                         inputBuffer_, weightBuffer_, outputBuffer_,
                         plan_.dimM, plan_.dimN, plan_.dimK,
                         plan_.layerName);
        results_.print(std::cout);
        results_.writeToFile(resultFileName());

        if (tracing()) {
            // Explicitly flush/close the trace. Optimizer child processes end with
            // _exit(), so the ofstream destructor never runs; without closing, the
            // last write-back transactions stay in the buffer and are missing from
            // the file (which made DRAMsim3 report write_energy = 0).
            dramTrace_.close();

            std::cout << "DRAM trace written: " << dramTrace_.path()
                      << " (" << dramTrace_.transactionCount() << " transactions)\n";
        }
    }

    // [Stage 1] Input-side double buffering: bring the A/B tiles needed by this op
    // onto the read side of the input/weight SRAM so the op becomes ready. If they
    // are not on the read side, load them from DRAM (loadTiles) only if they are not
    // already on the write (prefetch) side either, then swap write -> read for each
    // buffer whose tile is not yet on its read side.
    // (If both are already on the read side, nothing is done.)
    void prepareInputs(const PartialSumId& op, long& inAccCycle) {
        invalidateStaleDualPanel(op);
        if (isPartialSumReady(op)) return;
        if (!isInInputWriteBuffer())
            loadTiles(inAccCycle);
        swapInputBuffer();
    }

    // ToroidalDual (GEMV) only. tilesA is indexed by (m,k), but its physical content
    // is a K x N panel of B (reshapeToroidalDual: rowA=tileK, colA=tileN), i.e. the
    // content depends on the op's n. When the same (m,k) slot is reused for a
    // different n block and the previous n's panel is treated as "already on-chip",
    // (nTiles-1)/nTiles of B is never loaded from DRAM, undercounting DRAM traffic,
    // SRAM writes and cycles. So track the n held by each slot, and if a different n
    // is needed, evict the slot so ensureResident reloads it.
    // Tiles preloaded by firstFillUp are treated as holding the first n block (n=0;
    // missing entry = 0). On non-dual paths this is a no-op, so existing results are
    // preserved bit-for-bit.
    void invalidateStaleDualPanel(const PartialSumId& op) {
        if (array_.config().kind() != ConfigKind::ToroidalDual) return;
        Tile* a = findMutableTile(plan_.tilesA, plan_.kTiles, op.indexM, op.indexK);
        if (a == nullptr) return;
        if (a->isInDram()) {                    // will be freshly loaded anyway -> that load carries this n
            tileAPanelN_[a] = op.indexN;
            return;
        }
        const auto it = tileAPanelN_.find(a);
        const int residentN = (it != tileAPanelN_.end()) ? it->second : 0;
        if (residentN == op.indexN) return;     // required panel is already on-chip

        inputBuffer_.freeBytes(a->shape.sizeBytes(), a->location);
        a->writeBackToDram();
        lastUsedA_.erase(a);
        eraseResident(residentA_, a);
        tileAPanelN_[a] = op.indexN;
    }

    // [Stage 3] Output side (ReuseC only): write the completed TileC, whose
    // accumulation finished in the array, once onto the write side of output SRAM.
    // If the write side has no room, empty the read side (flushing to DRAM if
    // needed), swap, then write.
    // ReuseA/ReuseB (RMW) accumulate directly in output SRAM inside
    // executePartialSum, so isOutputGenerated() is false and this stage is skipped.
    void storeOutput(long& outAccCycle) {
        if (!isOutputGenerated()) return;
        if (!canSendOutput()) {
            if (!canSwapOutputBuffer())
                flushReadOutputBuffer(outAccCycle);
            swapOutputBuffer();
        }
        writeOutput();
    }

    // [Stage 4] Output side (ReuseA/ReuseB RMW only): an accumulation block
    // (ReuseB: a group of m for a fixed n; ReuseA: a group of n for a fixed m) has
    // just completed. Hand the write side, filled with completed tiles, over to the
    // read (drain) side so the next block accumulates on the opposite, empty side.
    // The read side is flushed to DRAM first if still occupied; the swapped-out side
    // is flushed on the next swap or by drainOutputBuffer(). At a block boundary the
    // write side holds no incomplete tiles, so the swap is always legal (same drain
    // flow as ReuseC storeOutput, keeping output double buffering).
    //   Without this drain, completed tiles would pile up on the side and, at small
    //   cd, the side would fill up mixed with the next block's incomplete tiles,
    //   making ensureOutputWriteResidency abort.
    void drainCompletedBlock(long& outAccCycle) {
        if (!canSwapOutputBuffer())            // opposite side not yet drained -> flush it first
            flushReadOutputBuffer(outAccCycle);
        swapOutputBuffer();                    // completed block -> drain side; write side is now empty
    }

    // Total simulated wall-clock cycles so far.
    long cycle() const { return cycle_; }

    // Results collected at the end of run().
    const SimulationResults& results() const { return results_; }

    // Path of the DRAM trace written by run() (same base name as the result file,
    // with .trace), or "-" when not tracing (Analytic mode).
    // The Optimizer runs this trace through DRAMsim3 to add the actual DRAM energy.
    std::string traceFilePath() const {
        return tracing() ? traceFileName() : std::string("-");
    }

    // Debug dump of the whole schedule plan.
    void printState() const {
        std::cout << "===== Simulator State (cycle=" << cycle_ << ") =====\n";
        plan_.print(std::cout);
        std::cout << "==========================================\n";
    }

    // Print the partial-sum accumulation status of all C(m,n) tiles at once:
    //   shows generatedCount/partialSumCount and whether each tile is done.
    void printPartialSums() const {
        std::cout << "===== Partial Sums (cycle=" << cycle_ << ") =====\n";
        for (const TileC& c : plan_.tilesC)
            std::cout << "  " << c << "\n";
        std::cout << "==========================================\n";
    }

private:

    // Build the result file path from the array configuration, encoding all dimensions:
    //   results/sim_<kind>_<dataflow>_<oR>x<oC>_<lR>x<lC>_<mult>_cd<contracting>.txt
    //     order: outer PE (row x col), local PE (row x col), mult per PE
    std::string resultFileName() const {
        const SystolicArrayConfig& cfg = array_.config();

        std::string kind;
        switch (cfg.kind()) {
            case ConfigKind::Grouped:      kind = "Grouped";      break;
            case ConfigKind::Toroidal:     kind = "Toroidal";     break;
            case ConfigKind::ToroidalDual: kind = "ToroidalDual"; break;
        }

        std::string df;
        switch (cfg.dataflow) {
            case Dataflow::ReuseA: df = "ReuseA"; break;
            case Dataflow::ReuseB: df = "ReuseB"; break;
            case Dataflow::ReuseC: df = "ReuseC"; break;
        }

        return "results/sim_" + kind + "_" + df
             + "_" + std::to_string(cfg.outerPeRow) + "x" + std::to_string(cfg.outerPeCol)
             + "_" + std::to_string(cfg.localPeRow) + "x" + std::to_string(cfg.localPeCol)
             + "_" + std::to_string(cfg.multPerPe)
             + "_cd" + std::to_string(cfg.contractingDimensionSize)
             + ".txt";
    }

    // Trace file path: same base name as the result file with a .trace extension.
    std::string traceFileName() const {
        std::string r = resultFileName();
        const std::string suffix = ".txt";
        if (r.size() >= suffix.size())
            r.erase(r.size() - suffix.size());
        return r + ".trace";
    }

    SchedulePlan plan_;             // private copy; tile locations/progress are mutated
    Array&        array_;
    Dram&         dram_;
    DoubleBuffer& inputBuffer_;     // input activations (tiles A)
    DoubleBuffer& outputBuffer_;    // outputs / partial-sum accumulators (tiles C)
    DoubleBuffer& weightBuffer_;    // weights (tiles B)

    SimulationResults results_;
    DramTrace         dramTrace_;
    DramEnergyMode    dramMode_ = DramEnergyMode::DramSim3;

    long cycle_ = 0;                // simulated wall-clock cycles

    // No trace is written in Analytic mode (DRAMsim3 is not run).
    bool tracing() const { return dramMode_ == DramEnergyMode::DramSim3; }

    // The argument-less helpers used by run() (loadTiles/swap/flush etc.) need to
    // know what is "currently being processed", so it is kept in members.
    //   currentOp_ : updated in isPartialSumReady() (used by input-side helpers)
    //   currentC_  : updated in accumulateOutput(), called from executePartialSum() (used by output-side helpers)
    const PartialSumId* currentOp_ = nullptr;
    TileC*             currentC_  = nullptr;

    // Stationary dataflow: the operand tile currently held (stationary) in the PE array
    // (ReuseA->A, ReuseB->B, ReuseC->C accumulator). If the next op's stationary operand
    // is the same, the array is not refilled and only the other operand is streamed.
    const Tile*        loadedStationary_ = nullptr;

    // Last-use timestamps for managing the input/weight buffers as LRU caches.
    long lruTick_ = 0;
    std::unordered_map<const Tile*, long> lastUsedA_;
    std::unordered_map<const Tile*, long> lastUsedB_;

    // ToroidalDual only: the n block index of the B panel currently held by each tilesA slot.
    // Missing entry = 0 (initial content loaded by firstFillUp). See invalidateStaleDualPanel.
    std::unordered_map<const Tile*, int> tileAPanelN_;

    // Tiles currently residing on-chip (Ping/Pong). Previously each eviction/drain
    // linearly scanned the entire tile vector (mTiles*nTiles scale), which was
    // quadratic, O(events x tiles) (days for huge-M layers). location only changes on
    // load and on evict/drain; swap does not touch it (DoubleBuffer::swap), so these
    // lists only need updating at those two points. Their size is bounded by one SRAM
    // side's capacity, so eviction/drain cost drops to O(resident). They visit exactly
    // the same tiles as the old full scan, so results are bit-for-bit identical.
    std::vector<Tile*>  residentA_;
    std::vector<Tile*>  residentB_;
    std::vector<TileC*> residentC_;

    // Remove t from v by swap-with-last and pop (order is not preserved).
    template <class T>
    static void eraseResident(std::vector<T*>& v, T* t) {
        for (size_t i = 0; i < v.size(); ++i)
            if (v[i] == t) { v[i] = v.back(); v.pop_back(); return; }
    }

    // Scheduler::emitTiles lays tiles out densely, so the vector index is computed
    // directly from the coordinates (O(1)). Previously every op linearly scanned the
    // whole tile vector, making total cost quadratic, O(ops x tiles) (days for huge-M
    // layers). Indexing visits exactly the same tiles, so results are bit-for-bit
    // identical.
    //   tilesA[m*kTiles + k], tilesB[k*nTiles + n] (ToroidalDual: tilesB[k]), tilesC[m*nTiles + n]
    // Out-of-range coordinates (never produced by a valid op) return nullptr, just like
    // the old linear search's not-found, preserving the defensive contract (the null
    // check in isPartialSumReady).
    const Tile* findTileA(int m, int k) const {
        if (m < 0 || k < 0 || m >= plan_.mTiles || k >= plan_.kTiles) return nullptr;
        return &plan_.tilesA[static_cast<size_t>(m) * plan_.kTiles + k];
    }

    const Tile* findTileB(int k, int n) const {
        if (k < 0 || n < 0 || k >= plan_.kTiles || n >= plan_.nTiles) return nullptr;
        // ToroidalDual: tilesB is the k-th slice of the input vector (content is
        // independent of n), so there is only one canonical tile per k
        // (see Scheduler::emitTiles).
        if (array_.config().kind() == ConfigKind::ToroidalDual)
            return &plan_.tilesB[static_cast<size_t>(k)];
        return &plan_.tilesB[static_cast<size_t>(k) * plan_.nTiles + n];
    }

    TileC* findTileC(int m, int n) {
        if (m < 0 || n < 0 || m >= plan_.mTiles || n >= plan_.nTiles) return nullptr;
        return &plan_.tilesC[static_cast<size_t>(m) * plan_.nTiles + n];
    }

    // Operand tile that stays stationary in the PE array under the current dataflow.
    //   ReuseA(input-stationary) : A(m,k)
    //   ReuseB(weight-stationary): B(k,n)
    //   ReuseC(output-stationary): C(m,n) accumulator
    const Tile* stationaryTile(const PartialSumId& op) const {
        switch (array_.dataflow()) {
            case Dataflow::ReuseA: return findTileA(op.indexM, op.indexK);
            case Dataflow::ReuseB: return findTileB(op.indexK, op.indexN);
            case Dataflow::ReuseC: {
                const int m = op.indexM, n = op.indexN;
                if (m < 0 || n < 0 || m >= plan_.mTiles || n >= plan_.nTiles) return nullptr;
                return &plan_.tilesC[static_cast<size_t>(m) * plan_.nTiles + n];
            }
        }
        return nullptr;
    }

    // Operand tile streamed through the PE array under the current dataflow,
    // i.e. the "other side" of the stationary operand.
    //   ReuseA(input-stationary) : stream B(k,n)
    //   ReuseB(weight-stationary): stream A(m,k)
    //   ReuseC(output-stationary): both A and B are streamed -> cost measured with A(m,k)
    // With physical mapping A.cols and B.rows may differ, so cyclesForTileOp must be
    // computed from the shape of the tile actually streamed, not the abstract tileK.
    const Tile* streamingTile(const PartialSumId& op) const {
        switch (array_.dataflow()) {
            case Dataflow::ReuseA: return findTileB(op.indexK, op.indexN);
            case Dataflow::ReuseB: return findTileA(op.indexM, op.indexK);
            case Dataflow::ReuseC: return findTileA(op.indexM, op.indexK);
        }
        return nullptr;
    }

    // SRAM buffer holding the stationary operand (nullptr if none).
    //   ReuseA(input-stationary) : A -> inputBuffer_
    //   ReuseB(weight-stationary): B -> weightBuffer_
    //   ReuseC(output-stationary): C accumulator -> not a buffer read (nullptr)
    DoubleBuffer* stationaryBuffer() {
        switch (array_.dataflow()) {
            case Dataflow::ReuseA: return &inputBuffer_;
            case Dataflow::ReuseB: return &weightBuffer_;
            case Dataflow::ReuseC: return nullptr;
        }
        return nullptr;
    }

    // SRAM buffer the streaming operand flows from (same mapping as streamingTile).
    //   ReuseA : B -> weightBuffer_
    //   ReuseB : A -> inputBuffer_
    //   ReuseC : A -> inputBuffer_ (both A and B flow, but A is used, matching the cost model)
    DoubleBuffer& streamingBuffer() {
        switch (array_.dataflow()) {
            case Dataflow::ReuseA: return weightBuffer_;
            case Dataflow::ReuseB: return inputBuffer_;
            case Dataflow::ReuseC: return inputBuffer_;
        }
        return inputBuffer_;
    }

    // Integer ceil(a / b) for non-negative a and positive b.
    static long ceilDiv(long a, long b) { return (a + b - 1) / b; }

    // Load tiles from DRAM onto the given buffer side in vector order, stopping at
    // the first tile that does not fit. Records DRAM reads, SRAM writes and trace
    // transactions, and registers loaded tiles as resident. Returns bytes loaded.
    long loadFirstTiles(std::vector<Tile>& tiles, DoubleBuffer& buffer, TileLocation side,
                        std::vector<Tile*>& resident) {

        long loaded = 0;

        for (Tile& t : tiles) {
            if (!t.isInDram()) continue;

            const int need = t.shape.sizeBytes();
            if (!buffer.canFit(need, side)) break;

            dram_.addRead(need);
            if (tracing()) dramTrace_.record(t, /*isWrite=*/false, cycle_);
            buffer.addWrite(need);   // whole tile loaded into SRAM -> proportional to bytes
            buffer.occupy(need, side);
            (side == TileLocation::PingBuffer) ? t.loadToPing() : t.loadToPong();
            resident.push_back(&t);   // add to on-chip resident list
            loaded += need;
        }
        return loaded;

    }

    // Initial fill: load as many A and B tiles as fit onto the read side of the
    // input and weight buffers. Nothing runs in the array yet, so the full transfer
    // time is charged: max over the DRAM bound (A+B share the DRAM bandwidth) and
    // each SRAM's single-side bandwidth bound.
    void firstFillUp() {

        const long inputLoaded  =
            loadFirstTiles(plan_.tilesA, inputBuffer_,  inputBuffer_.currentReadLocation(), residentA_);
        const long weightLoaded =
            loadFirstTiles(plan_.tilesB, weightBuffer_, weightBuffer_.currentReadLocation(), residentB_);

        const long dramBound   = ceilDiv(inputLoaded + weightLoaded, dram_.bandwidth);
        const long inputBound  = ceilDiv(inputLoaded,  inputBuffer_.singleBufferBandwidth);
        const long weightBound = ceilDiv(weightLoaded, weightBuffer_.singleBufferBandwidth);

        cycle_ += std::max({dramBound, inputBound, weightBound});

    }

    // True if the op's A tile and B tile are both on the read side of their buffers.
    // Side effect: sets currentOp_ = &op for the input-side helpers.
    bool isPartialSumReady(const PartialSumId& op) {

        currentOp_ = &op;

        const Tile* a = findTileA(op.indexM, op.indexK);
        const Tile* b = findTileB(op.indexK, op.indexN);
        if (a == nullptr || b == nullptr) return false;

        return a->location == inputBuffer_.currentReadLocation()
            && b->location == weightBuffer_.currentReadLocation();

    }

    // Split execution of one op into three steps:
    //   1) (re)load the stationary operand into the PE array   -> loadStationary
    //   2) stream the other operand and run the MAC             -> streamExecution
    //   3) accumulate the partial sum into the C tile (+RMW)    -> accumulateOutput
    void executePartialSum(const PartialSumId& op, long& inAccCycle, long& outAccCycle) {
        loadStationary(op, inAccCycle, outAccCycle);
        streamExecution(op, inAccCycle, outAccCycle);
        accumulateOutput(op, outAccCycle);
    }

    // 1) Stationary reuse: when consecutive ops use the same stationary operand tile,
    // the PE array is already filled with it, so the load cost is skipped and only
    // the other operand is streamed. The array is refilled only when the operand changes.
    //   ReuseB(weight-stationary): B(k,n) stays, A(m,k) input is streamed
    //   ReuseA(input-stationary) : A(m,k) stays, B(k,n) weight is streamed
    //   ReuseC(output-stationary): C(m,n) accumulator stays, A and B are streamed
    //                              (the partial sum resides in the array during k-reduction)
    // (Array fill is also array-busy time, so it is added to both the input and output
    //  accCycle so that DMA on both sides can overlap with and hide behind it.)
    // With physical mapping A.cols and B.rows may differ, so the cost is computed from
    // the actual stationary tile's shape (not collapsed to abstract tileM/N/K).
    void loadStationary(const PartialSumId& op, long& inAccCycle, long& outAccCycle) {
        const Tile* stationary = stationaryTile(op);
        if (stationary == loadedStationary_) return;

        const long stationaryLoadCycle = array_.cyclesForStationaryLoad(*stationary);
        cycle_      += stationaryLoadCycle;
        inAccCycle  += stationaryLoadCycle;
        outAccCycle += stationaryLoadCycle;
        array_.addBusyCycles(stationaryLoadCycle);   // fill = array-busy -> dynamic energy
        loadedStationary_ = stationary;

        // (Re)loading the stationary operand into the PE array = one read from its SRAM buffer.
        // It is not re-read while it stays in the array, so it is counted only on load.
        // The whole stationary tile is read, so the count is proportional to bytes.
        // (ReuseC's stationary operand is the C accumulator, not an input/weight buffer read -> nullptr.)
        if (DoubleBuffer* sb = stationaryBuffer())
            sb->addRead(stationary->shape.sizeBytes());
    }

    // 2) Stream the streaming operand and run the MAC. The cost is computed from the
    // shape of the tile actually streamed (stationary and streaming shapes may differ).
    void streamExecution(const PartialSumId& op, long& inAccCycle, long& outAccCycle) {
        const Tile* streaming = streamingTile(op);
        const long executionCycle = array_.cyclesForTileOp(*streaming);
        cycle_      += executionCycle;
        inAccCycle  += executionCycle;
        outAccCycle += executionCycle;
        array_.addBusyCycles(executionCycle);   // stream(MAC) = array-busy -> dynamic energy
        // (Useful stream cycles are accumulated alongside inside cyclesForTileOp as
        //  usefulStreamCycles: an energy-only statistic excluding the zero-pad tail.
        //  Fill/drain contribute 0 useful cycles.)

        // The streaming operand is streamed out of its SRAM buffer on every op. Each
        // streamed element is an SRAM read, so this is counted proportional to the
        // streamed tile's bytes rather than once per op. (Counting once per op would
        // create a modeling artifact where changing cd (= tile width) changes energy
        // even though the same total data is moved.)
        streamingBuffer().addRead(streaming->shape.sizeBytes());

        // Update the last-use time of the A/B tiles consumed by this op (for the LRU cache).
        lastUsedA_[findTileA(op.indexM, op.indexK)] = ++lruTick_;
        lastUsedB_[findTileB(op.indexK, op.indexN)] = ++lruTick_;
    }

    // 3) Accumulate this op's partial sum into tile C(m,n) (progress counters for all
    // dataflows).
    // ReuseA/ReuseB (and ToroidalDual): the array only produces partial sums and the
    // accumulation is done by RMW on output SRAM (no separate accumulator). Every
    // partial sum RMWs output SRAM, and the TileC is complete after the last partial
    // sum, requantized on the SRAM->DRAM path (until then it is just a partial sum,
    // not a TileC). ReuseC accumulates and requantizes inside the array (drained via
    // cyclesForOutput in writeOutput), so it skips the RMW step here.
    void accumulateOutput(const PartialSumId& op, long& outAccCycle) {
        TileC* c = findTileC(op.indexM, op.indexN);
        const bool firstPartial = (c->generatedCount == 0);
        c->addPartialSum();
        currentC_ = c;   // the C tile "just updated", referenced by output-side helpers

        if (usesRmwOutput())
            accumulatePartialSumRmw(c, firstPartial, outAccCycle);

        printPartialSum(op, *c);
    }

    // Print which partial sum (k index) this op added to which C(m,n) tile, and the
    // accumulation progress so far (generatedCount/partialSumCount).
    void printPartialSum(const PartialSumId& op, const TileC& c) const {
        // Per-op logging becomes an I/O bottleneck in large sweeps, so it is gated by
        // an environment variable: printed only when SIM_VERBOSE is set (default: quiet).
        static const bool verbose = (std::getenv("SIM_VERBOSE") != nullptr);
        if (!verbose) return;
        std::cout << "[cycle=" << cycle_ << "] "
                  << "C(m=" << op.indexM << ", n=" << op.indexN << ") "
                  << "+= A(m=" << op.indexM << ", k=" << op.indexK << ")"
                  << " * B(k=" << op.indexK << ", n=" << op.indexN << ")"
                  << "   partial sum " << c.generatedCount << "/" << c.partialSumCount
                  << (c.isGenerated ? "  -> done" : "")
                  << "\n";
    }

    // ===================== Output (C) side =====================

    // Does the current dataflow accumulate via RMW on output SRAM?
    //   ReuseA/ReuseB (and ToroidalDual = ReuseB): the array only produces partial sums
    //     and accumulation is RMW on output SRAM (no separate accumulator) -> true.
    //   ReuseC (output-stationary): accumulates inside the array -> false (original path).
    bool usesRmwOutput() const {
        return array_.dataflow() != Dataflow::ReuseC;
    }

    // Bytes actually read from output SRAM when draining one C tile.
    // RMW (ReuseA/ReuseB): requantization happens on the SRAM->DRAM path, so the wide
    //   partial sum still resides in SRAM -> read partialSumSizeBytes.
    // ReuseC: writeOutput already placed the requantized, completed TileC -> shape.sizeBytes.
    long outputResidencyBytes(const TileC& c) const {
        return usesRmwOutput() ? c.partialSumSizeBytes() : c.shape.sizeBytes();
    }


    // ReuseA/ReuseB: accumulate one partial sum directly in output SRAM.
    // What resides in output SRAM during accumulation is not a "completed TileC" but
    // the in-progress partial sum (accumulator). It only becomes a TileC once the last
    // partial sum has been added (requantization is modelled on the drain to DRAM,
    // see outputResidencyBytes / flushReadOutputBuffer).
    //   - first partial sum : place the accumulator on the write side (occupy), write-only
    //                         (no previous value to read)
    //   - later partial sums: read (previous accumulation) + write (update) in place = RMW
    // RMW accesses are hidden behind array streaming (MAC), so no cycles are added
    // (only access counts/energy are recorded).
    void accumulatePartialSumRmw(TileC* c, bool firstPartial, long& outAccCycle) {
        if (firstPartial) {
            // What resides in output SRAM during accumulation is the pre-requantization
            // partial sum, so space must be reserved at partial-sum width
            // (partialSumSizeBytes), not the smaller requantized TileC size (shape.sizeBytes).
            ensureOutputWriteResidency(c->partialSumSizeBytes(), outAccCycle);
            const TileLocation w = outputBuffer_.currentWriteLocation();
            outputBuffer_.occupy(c->partialSumSizeBytes(), w);
            (w == TileLocation::PingBuffer) ? c->loadToPing() : c->loadToPong();
            residentC_.push_back(c);   // add to on-chip resident list (accumulator placed)
            // First partial sum: write-only. The whole partial-sum tile is written -> proportional to bytes.
            outputBuffer_.addWrite(c->partialSumSizeBytes());
        } else {
            // RMW: read the whole partial-sum tile, accumulate, write back -> both proportional to bytes.
            outputBuffer_.addRead(c->partialSumSizeBytes());    // read previous accumulation
            outputBuffer_.addWrite(c->partialSumSizeBytes());   // write updated value
        }

    }

    // Make room on the write side for a new RMW accumulator tile.
    // Every tile of the current accumulation block must reside on one side until it is
    // complete. With block-boundary draining in run() this is normally a no-op; if the
    // write side is still full, it must hold only completed tiles from earlier blocks,
    // so that side is swapped to the read side for draining to DRAM (same flow as
    // the regular output double buffering).
    void ensureOutputWriteResidency(int need, long& outAccCycle) {
        const TileLocation writeSide = outputBuffer_.currentWriteLocation();
        if (outputBuffer_.canFit(need, writeSide)) return;

        // If the write side still holds tiles that are mid-accumulation (incomplete),
        // it cannot be drained (partial sums would be lost). This means an accumulation
        // block is larger than one output SRAM side (an infeasible configuration).
        assert(!writeSideHasIncompleteTile(writeSide) &&
               "output SRAM side cannot hold a full C-tile accumulation group");

        if (!canSwapOutputBuffer())
            flushReadOutputBuffer(outAccCycle);
        swapOutputBuffer();

        assert(outputBuffer_.canFit(need, outputBuffer_.currentWriteLocation()) &&
               "output SRAM side cannot hold a single C tile");
    }

    // Does side still hold a C tile whose accumulation is not finished (partial-sum state)?
    // Incomplete tiles are by definition on-chip accumulators, so checking residentC_ is
    // enough (same result as the old full tilesC scan, O(resident)).
    bool writeSideHasIncompleteTile(TileLocation side) const {
        for (const TileC* c : residentC_)
            if (c->location == side && !c->isGenerated) return true;
        return false;
    }

    bool isOutputGenerated() const {
        // ReuseC (output-stationary) only. There is a new completed TileC to emit only
        // if the C tile handled by the op just executed has finished all accumulation
        // (isGenerated) and has not yet been placed in the output buffer (None).
        // ReuseA/ReuseB (RMW) place the accumulator on the output SRAM write side at the
        // first partial sum, so location becomes Ping/Pong (naturally filtered by the
        // location==None check below), but they are also excluded explicitly by dataflow
        // for clarity. On the RMW path the completed TileC is produced by the last
        // partial sum (+requantization), not by writeOutput.
        return !usesRmwOutput()
            && currentC_ != nullptr
            && currentC_->isGenerated
            && currentC_->location == TileLocation::None;
    }

    bool canSendOutput() {
        // Is there room on the current write side for this C tile?
        return outputBuffer_.canFit(currentC_->shape.sizeBytes(),
                                    outputBuffer_.currentWriteLocation());
    }

    // ReuseC: drain the completed TileC out of the array (cyclesForOutput, charged to
    // wall-clock and busy cycles) and write it onto the output write side.
    void writeOutput() {
        const TileLocation writeSide = outputBuffer_.currentWriteLocation();
        const int need = currentC_->shape.sizeBytes();

        const long drainCycle = array_.cyclesForOutput();
        cycle_ += drainCycle;
        array_.addBusyCycles(drainCycle);   // drain(pipeTail) = array-busy -> dynamic energy

        outputBuffer_.occupy(need, writeSide);
        outputBuffer_.addWrite(need);   // whole completed TileC written -> proportional to bytes
        (writeSide == TileLocation::PingBuffer) ? currentC_->loadToPing()
                                                : currentC_->loadToPong();
        residentC_.push_back(currentC_);   // add to on-chip resident list
    }

    bool canSwapOutputBuffer() {
        // Swap is safe only when the read side is empty (already fully drained to DRAM).
        return outputBuffer_.usedOf(outputBuffer_.currentReadLocation()) == 0;
    }

    void swapOutputBuffer() {
        outputBuffer_.swap();
    }

    void flushReadOutputBuffer(long& outAccCycle) {
        // Write the completed C tiles accumulated on the read side out to DRAM.
        // DRAM receives the requantized size (shape.sizeBytes); the SRAM drain read uses
        // the residency size (outputResidencyBytes).
        const TileLocation readSide = outputBuffer_.currentReadLocation();

        long bytes = 0;
        // Drain only completed C tiles residing on readSide and remove them from the
        // resident list by swap-pop. (Same visited set as the old full tilesC scan ->
        // same totals, O(resident).)
        size_t i = 0;
        while (i < residentC_.size()) {
            TileC* c = residentC_[i];
            if (c->location != readSide) { ++i; continue; }
            c->writeBackToDram();
            dram_.addWrite(c->shape.sizeBytes());
            if (tracing()) dramTrace_.record(*c, /*isWrite=*/true, cycle_);
            outputBuffer_.addRead(outputResidencyBytes(*c));   // SRAM drain read: proportional to bytes
            bytes += c->shape.sizeBytes();
            residentC_[i] = residentC_.back();
            residentC_.pop_back();
        }
        outputBuffer_.release(readSide);

        // The drain cost is hidden behind the compute cycles accumulated meanwhile
        // (outAccCycle); only the excess is added to wall-clock.
        const long dramBound = ceilDiv(bytes, dram_.bandwidth);
        const long bufBound  = ceilDiv(bytes, outputBuffer_.singleBufferBandwidth);
        const long cost = std::max(dramBound, bufBound);
        if (cost > outAccCycle) cycle_ += (cost - outAccCycle);

        // Draining an intermediate panel no longer exposes the pipeTail bubble on the
        // timeline: the new Scala design (DataflowDriverSram + the ping-pong of
        // BigAccSramDoubleBuffer) overlaps a finished panel's drain with the next panel's
        // weight load + stream. Therefore an intermediate flush that has a following
        // panel does not pay pipeTail in cycle_. The depth-long accumulator readout is
        // already represented by the SRAM/DRAM bandwidth cost above (+ double-buffer
        // overlap). Only the final drain with no following panel exposes pipeTail on
        // wall-clock, in drainOutputBuffer().
        //
        // Energy accounting differs, however: even when the drain is hidden in time, the
        // array really does flush its output pipeline and toggle at that moment, so it
        // consumes dynamic energy. Hence nothing is added to cycle_, but the activity
        // cycles are counted via addBusyCycles.
        // (RMW (ReuseA/ReuseB) only; ReuseC already handles this in writeOutput.)
        if (bytes > 0 && usesRmwOutput()) {
            array_.addBusyCycles(array_.cyclesForOutput());
        }

        resetAccCycle(outAccCycle);
    }

    void drainOutputBuffer() {
        // At the end, write all C tiles remaining on both sides down to DRAM.
        // residentC_ is exactly the set of on-chip (Ping/Pong) C tiles (same as the old isOnChip scan).
        // Nothing runs concurrently, so the full transfer cost is added to wall-clock.
        long bytes = 0;
        for (TileC* c : residentC_) {
            c->writeBackToDram();
            dram_.addWrite(c->shape.sizeBytes());
            if (tracing()) dramTrace_.record(*c, /*isWrite=*/true, cycle_);
            outputBuffer_.addRead(outputResidencyBytes(*c));   // SRAM drain read: proportional to bytes
            bytes += c->shape.sizeBytes();
        }
        residentC_.clear();
        outputBuffer_.release(TileLocation::PingBuffer);
        outputBuffer_.release(TileLocation::PongBuffer);

        const long dramBound = ceilDiv(bytes, dram_.bandwidth);
        const long bufBound  = ceilDiv(bytes, outputBuffer_.singleBufferBandwidth);
        cycle_ += std::max(dramBound, bufBound);

        // Array readout pipeTail flush bubble of the last remaining panel, charged to
        // both wall-clock and busy cycles (see flushReadOutputBuffer). RMW (ReuseA/ReuseB) only.
        if (bytes > 0 && usesRmwOutput()) {
            const long drainCycle = array_.cyclesForOutput();
            cycle_ += drainCycle;
            array_.addBusyCycles(drainCycle);
        }
    }

    // ===================== Input (A) / weight (B) side =====================

    bool isInInputWriteBuffer() {
        // Are the A/B tiles needed by the current op already on-chip (on either the
        // read or the write side), so that a swap alone can make the op ready?
        const Tile* a = findTileA(currentOp_->indexM, currentOp_->indexK);
        const Tile* b = findTileB(currentOp_->indexK, currentOp_->indexN);
        if (a == nullptr || b == nullptr) return false;

        const bool aOk = a->location == inputBuffer_.currentReadLocation()
                      || a->location == inputBuffer_.currentWriteLocation();
        const bool bOk = b->location == weightBuffer_.currentReadLocation()
                      || b->location == weightBuffer_.currentWriteLocation();
        return aOk && bOk;
    }

    void swapInputBuffer() {
        // Swap only the buffer(s) whose tile is not on the read side (= prefetched on the write side).
        const Tile* a = findTileA(currentOp_->indexM, currentOp_->indexK);
        const Tile* b = findTileB(currentOp_->indexK, currentOp_->indexN);
        if (a->location != inputBuffer_.currentReadLocation())  swapInputBufferA();
        if (b->location != weightBuffer_.currentReadLocation()) swapInputbufferB();
    }

    void swapInputBufferA() { inputBuffer_.swap(); }

    void swapInputbufferB() { weightBuffer_.swap(); }

    void loadTiles(long& inAccCycle){
        // Double-buffering model: load the required A/B tiles onto the write (prefetch) side.
        // Right after this, prepareInputs() calls swapInputBuffer() to flip the write side
        // to the read side, at which point the op becomes ready. (If the tiles are already
        // on-chip, isInInputWriteBuffer() is true, this function is not reached, and only
        // the swap is performed.)
        // The load cost is hidden behind array-busy cycles accumulated since the last
        // load (inAccCycle); only the excess is added to wall-clock.
        const long inputBytes  = ensureResident(plan_.tilesA, inputBuffer_,  lastUsedA_, residentA_,
                                                plan_.kTiles, currentOp_->indexM, currentOp_->indexK);
        // ToroidalDual: tilesB has one canonical tile per k (compact) -> look up with (inner=1, col=0).
        const bool dualB = array_.config().kind() == ConfigKind::ToroidalDual;
        const long weightBytes = ensureResident(plan_.tilesB, weightBuffer_, lastUsedB_, residentB_,
                                                dualB ? 1 : plan_.nTiles,
                                                currentOp_->indexK,
                                                dualB ? 0 : currentOp_->indexN);

        const long dramBound   = ceilDiv(inputBytes + weightBytes, dram_.bandwidth);
        const long inputBound  = ceilDiv(inputBytes,  inputBuffer_.singleBufferBandwidth);
        const long weightBound = ceilDiv(weightBytes, weightBuffer_.singleBufferBandwidth);
        const long cost = std::max({dramBound, inputBound, weightBound});
        if (cost > inAccCycle) cycle_ += (cost - inAccCycle);

        resetAccCycle(inAccCycle);
    }

    // Load tile (row,col) onto the buffer's current write (prefetch) side, evicting
    // LRU tiles from that side as needed.
    // Returns 0 if it is already on the read or write side (no load needed), or the
    // number of bytes loaded if it was freshly read from DRAM.
    long ensureResident(std::vector<Tile>& tiles, DoubleBuffer& buffer,
                        std::unordered_map<const Tile*, long>& lastUsed,
                        std::vector<Tile*>& resident,
                        int inner, int row, int col) {
        Tile* target = findMutableTile(tiles, inner, row, col);
        assert(target != nullptr && "requested tile does not exist in plan");

        const TileLocation readSide  = buffer.currentReadLocation();
        const TileLocation writeSide = buffer.currentWriteLocation();

        // On the read side: use as-is (swapInputBuffer will not flip this buffer);
        // on the write side: already prefetched -> either way no DRAM load is needed.
        if (target->location == readSide || target->location == writeSide) return 0;

        const int need = target->shape.sizeBytes();
        assert(need <= buffer.singleBufferSize && "tile larger than a single buffer");

        // Evict LRU tiles until the write side has room.
        while (!buffer.canFit(need, writeSide)) {
            Tile* victim = lruVictim(tiles, resident, lastUsed, writeSide, target);
            assert(victim != nullptr && "no victim but cannot fit");
            buffer.freeBytes(victim->shape.sizeBytes(), writeSide);
            victim->writeBackToDram();   // input tiles are read-only -> just mark non-resident (no DRAM write)
            lastUsed.erase(victim);
            eraseResident(resident, victim);   // remove from on-chip resident list
        }

        dram_.addRead(need);
        if (tracing()) dramTrace_.record(*target, /*isWrite=*/false, cycle_);
        buffer.addWrite(need);   // whole tile loaded into SRAM -> proportional to bytes
        buffer.occupy(need, writeSide);
        target->location = writeSide;
        resident.push_back(target);   // add to on-chip resident list
        return need;
    }

    // Find an element of tilesA/tilesB by coordinates in O(1) (dense indexing, same
    // convention as findTile*). inner = number of tiles along the inner dimension
    // (tilesA -> kTiles, tilesB -> nTiles, or 1 for ToroidalDual tilesB).
    // idx = row*inner + col.
    static Tile* findMutableTile(std::vector<Tile>& tiles, int inner, int row, int col) {
        if (row < 0 || col < 0 || inner <= 0 || col >= inner) return nullptr;
        const size_t idx = static_cast<size_t>(row) * inner + col;
        if (idx >= tiles.size()) return nullptr;
        return &tiles[idx];
    }

    // Pick the LRU victim among resident (currently on-chip) tiles on the given side
    // (excluding keep). Selection is exactly the same as the old "linear scan over all
    // tiles": lexicographic minimum of (lastUsed, original index). Never-used tiles have
    // tick 0 (evicted first); ties go to the smaller original vector index.
    Tile* lruVictim(const std::vector<Tile>& tiles,
                    const std::vector<Tile*>& resident,
                    std::unordered_map<const Tile*, long>& lastUsed,
                    TileLocation side, const Tile* keep) {
        Tile* victim = nullptr;
        long bestU = LONG_MAX;
        long bestIdx = LONG_MAX;
        const Tile* base = tiles.data();
        for (Tile* t : resident) {
            if (t == keep) continue;
            if (t->location != side) continue;
            long u = 0;
            auto it = lastUsed.find(t);
            if (it != lastUsed.end()) u = it->second;   // never-used tiles get 0 (evicted first)
            const long idx = static_cast<long>(t - base);
            if (u < bestU || (u == bestU && idx < bestIdx)) {
                bestU = u; bestIdx = idx; victim = t;
            }
        }
        return victim;
    }

    void resetAccCycle(long& accCycle) {
        accCycle = 0;
    }


    // True once every C tile has received all of its partial sums.
    bool finished() const {
        for (const TileC& c : plan_.tilesC) {
            if (!c.isGenerated) return false;
        }
        return true;
    }
};