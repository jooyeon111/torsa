#pragma once
// Systolic-array geometry configuration (SystolicArrayConfig and its
// Grouped / Toroidal / ToroidalDual variants): PE hierarchy, derived array
// dimensions, SRAM port bandwidths and AccSram free-dimension (depth) helpers.
#include <algorithm>
#include <cassert>
#include "Dataflow.hpp"

// Array topology. Grouped = grouped PE array, Toroidal = toroidal systolic
// array, ToroidalDual = toroidal array with the dual (burst/GEMV) mode.
enum class ConfigKind{ Grouped, Toroidal, ToroidalDual};

namespace detail {
    inline bool isPowerOfTwo(int x) {
        return x > 0 && (x & (x - 1)) == 0;
    }

    // ceil(log2(x)); returns 0 for x <= 1.
    inline int log2Up(int x) {
        if (x <= 1) return 0;
        int bits = 0;
        int v = x - 1;
        while (v > 0) { v >>= 1; ++bits; }
        return bits;
    }
}

// Abstract base for all array configurations. PE hierarchy:
//   outer PE grid (outerPeRow x outerPeCol)
//     -> local vector PEs per outer PE (localPeRow x localPeCol)
//       -> multPerPe multipliers per vector PE.
class SystolicArrayConfig {
public:
    const Dataflow dataflow;
    const int inputOperandBitWidth = 8;       // bits per input/weight element
    const int multOutputBitWidth = 16;        // bits of one multiplier output
    const int outerPeRow;
    const int outerPeCol;
    const int localPeRow;
    const int localPeCol;
    const int multPerPe;                      // multipliers per vector PE (power of two)
    const int totalVectorPeRow;               // outerPeRow * localPeRow
    const int totalVectorPeCol;               // outerPeCol * localPeCol
    const int systolicArrayRowDimension;      // totalVectorPeRow * multPerPe
    const int systolicArrayColDimension;      // totalVectorPeCol * multPerPe
    const int numVectorPePerGroupPe;          // localPeRow * localPeCol
    // Whether the .cfg kind was ToroidalDual. For GEMM (M>1) kind() is demoted
    // to Toroidal, so kind() alone cannot tell (ArrayConfigSpec::resolveBurstForLayerM).
    // SRAM source selection (SramSelector::cactiSourceFor) reads this value, because
    // there is physically only one weight buffer and both modes must use the same macro.
    // Set by ArrayConfigSpec::makeConfig. Has no effect on geometry/timing.
    bool declaredBurst = false;
    // Same value as Scala's freeDimensionSize: AccSram depth (number of addresses)
    // = number of cycles psumValid is held during one K-pass. This is not the GEMM
    // M/N but a hardware provisioning value (SRAM area), and it is the free
    // variable swept by the Optimizer.
    // (The member still has its old name.)
    const int contractingDimensionSize;
    // Width over which the Grouped output de-skew chain spreads.
    //   = systolicOutputDimC / numAccTapCellPort (Scala GoldenMatrix.scala:250)
    //   ReuseA -> totalVectorPeRow/localPeRow = outerPeRow
    //   ReuseB -> totalVectorPeCol/localPeCol = outerPeCol
    //   ReuseC -> 1
    // Computed for every kind, but only meaningful when hasDeskewFlush().
    const int accDeskewSpan;
    // The AccSram address counter is free-running and wraps at depth, so the slot
    // length must divide depth. Grouped picks the *smallest* slot that covers the
    // de-skew span (Scala GoldenMatrix.freeDimensionCap).
    const int deskewSlotStride;
    // Maximum GEMM K this array is provisioned for. 0 = unspecified.
    // For ReuseA/ReuseB it only sets the accumulator data width and does not affect
    // tile shapes; it is used only as the tileK upper bound of Grouped ReuseC.
    const int maxContractionSize;
    // Duplicate of multOutputBitWidth (different capitalisation); this one is
    // what systolicOutputPortBitWidth() uses.
    const int multOutputBitwidth = 16;


    virtual ~SystolicArrayConfig() = default;
    virtual ConfigKind kind() const = 0;

    // ---- Values derived from the free dimension (AccSram depth) -------------
    // Alias using the Scala name. Same value as contractingDimensionSize.
    int freeDimensionSize() const { return contractingDimensionSize; }

    // Only Grouped ReuseA/ReuseB reserve the last slot for the de-skew zero-flush
    // (tilesPerPanel = numSlots - 1). For every other kind/dataflow there is no
    // reservation, so the derived values below reduce to the identity (depth).
    bool hasDeskewFlush() const {
        return kind() == ConfigKind::Grouped &&
               (dataflow == Dataflow::ReuseA || dataflow == Dataflow::ReuseB);
    }

    // Number of free-axis rows a single panel actually processes = depth - slotStride
    // (clamped to >= 1), or depth when there is no de-skew flush.
    // This is the Scheduler's tileM (ReuseB) / tileN (ReuseA).
    int freeAxisTileExtent() const {
        if (!hasDeskewFlush()) return contractingDimensionSize;
        return std::max(1, contractingDimensionSize - deskewSlotStride);
    }

    // Streaming cycles of one panel. The last slot is a zero-flush and short slots
    // are zero-padded, so even a short panel always takes depth cycles.
    int panelStreamCycles() const { return contractingDimensionSize; }

    // Whether depth satisfies the Grouped slot constraint:
    //   depth >= 1, and if hasDeskewFlush() and accDeskewSpan > 1:
    //   depth % accDeskewSpan == 0 and depth / accDeskewSpan >= 2.
    bool isValidFreeDimensionSize(int depth) const {
        if (depth < 1) return false;
        if (!hasDeskewFlush() || accDeskewSpan <= 1) return true;
        return depth % accDeskewSpan == 0 && depth / accDeskewSpan >= 2;
    }

    // Largest valid depth <= depth. Returns 0 if none exists.
    int roundDownFreeDimensionSize(int depth) const {
        if (depth < 1) return 0;
        if (!hasDeskewFlush() || accDeskewSpan <= 1) return depth;
        const int d = (depth / accDeskewSpan) * accDeskewSpan;
        return d >= 2 * accDeskewSpan ? d : 0;
    }

    // If unspecified, fall back to the historical value (systolicArrayColDimension)
    // so that existing results are preserved.
    int effectiveMaxContractionSize() const {
        return maxContractionSize > 0 ? maxContractionSize : systolicArrayColDimension;
    }

    // Input-SRAM port bandwidth in bits (streamed operand port width).
    int systolicArrayInputBandwidthBtis() const {
        // input = port of the streaming operand. Following Pulse:
        //   ReuseA: A stationary -> B is streamed (inputB, systolicInputDimB),
        //   ReuseB: B stationary -> A is streamed (inputA, systolicInputDimA),
        //   ReuseC: both streamed -> input SRAM serves A (weight SRAM serves B).
        switch (kind()) {
            case ConfigKind::Grouped:
                // GroupedTopology: systolicInputDimA = rowDimension, systolicInputDimB = colDimension.
                switch (dataflow) {
                    case Dataflow::ReuseA: return systolicArrayColDimension * inputOperandBitWidth;
                    case Dataflow::ReuseB: return systolicArrayRowDimension * inputOperandBitWidth;
                    case Dataflow::ReuseC: return systolicArrayRowDimension * inputOperandBitWidth;
                }
                break;
            case ConfigKind::Toroidal:
                // ToroidalTopology: ReuseA's inputB = outerPeRow*multPerPe (= colDimension,
                // given the constraint oCol*lvC == oRow); ReuseB's inputA = outerPeCol*multPerPe
                // (= rowDimension, given oRow*lvR == oCol). Broadcasting (lvR/lvC > 1) does
                // not widen the port.
                switch (dataflow) {
                    case Dataflow::ReuseA: return systolicArrayColDimension * inputOperandBitWidth;
                    case Dataflow::ReuseB: return systolicArrayRowDimension * inputOperandBitWidth;
                    case Dataflow::ReuseC: return systolicArrayRowDimension * inputOperandBitWidth;
                }
                break;
            case ConfigKind::ToroidalDual:
                // Burst input: rowDimension elements for each of the totalVectorPeCol ports.
                return systolicArrayRowDimension * outerPeCol * localPeCol * inputOperandBitWidth;
        }
        return 0;
    }

    // Weight-SRAM port bandwidth in bits (stationary operand load port width).
    int systolicArrayWeightBandwidthBtis() const {
        switch (kind()) {
            case ConfigKind::Grouped:
                // weight = load port of the stationary operand. Pulse GroupedTopology:
                //   ReuseA: A stationary -> inputA port (systolicInputDimA = rowDimension),
                //   ReuseB: B stationary -> inputB port (systolicInputDimB = colDimension),
                //   ReuseC: output-stationary -> weight SRAM feeds the B stream (colDimension).
                switch (dataflow) {
                    case Dataflow::ReuseA: return systolicArrayRowDimension * inputOperandBitWidth;
                    case Dataflow::ReuseB: return systolicArrayColDimension * inputOperandBitWidth;
                    case Dataflow::ReuseC: return systolicArrayColDimension * inputOperandBitWidth;
                }
                break;
            case ConfigKind::Toroidal:
                // Same port assignment as Grouped.
                switch (dataflow) {
                    case Dataflow::ReuseA: return systolicArrayRowDimension * inputOperandBitWidth;
                    case Dataflow::ReuseB: return systolicArrayColDimension * inputOperandBitWidth;
                    case Dataflow::ReuseC: return systolicArrayColDimension * inputOperandBitWidth;
                }
                break;
            case ConfigKind::ToroidalDual:
                // GEMV weight port: localPeCol * multPerPe elements (minimum of the burst series).
                return localPeCol * multPerPe * inputOperandBitWidth;
        }
        return 0;
    }

    // Output-SRAM port bandwidth in bits = output element width * number of output ports.
    int systolicArrayOutputBandwidthBtis() const {
        switch (kind()) {
            case ConfigKind::Grouped:
                switch (dataflow) {
                    case Dataflow::ReuseA: //return totalVectorPeRow;
                        return systolicOutputPortBitWidth() * totalVectorPeRow;

                    case Dataflow::ReuseB: //return totalVectorPeCol;
                        return systolicOutputPortBitWidth() * totalVectorPeCol;

                    case Dataflow::ReuseC:
                        return systolicOutputPortBitWidth() * outerPeCol * numVectorPePerGroupPe;
                }
                break;
            case ConfigKind::Toroidal:
                switch (dataflow) {
                    case Dataflow::ReuseA:
                        return systolicOutputPortBitWidth() * totalVectorPeRow;

                    case Dataflow::ReuseB:
                        return systolicOutputPortBitWidth() * totalVectorPeCol;

                    case Dataflow::ReuseC:
                        return systolicOutputPortBitWidth() * totalVectorPeCol;
                }
                break;
            case ConfigKind::ToroidalDual:
                return systolicOutputPortBitWidth() * totalVectorPeCol;
        }
        return 0;
    }

protected:
    // oRow/oCol = outer PE grid, lRow/lCol = local vector PEs per outer PE,
    // mult = multipliers per vector PE, cd = contractingDimensionSize (AccSram depth),
    // maxContraction = maxContractionSize (0 = unspecified).
    SystolicArrayConfig(int oRow, int oCol, int lRow, int lCol,
                        int mult, Dataflow df, int cd, int maxContraction = 0)
        : dataflow(df),
          outerPeRow(oRow), outerPeCol(oCol),
          localPeRow(lRow), localPeCol(lCol),
          multPerPe(mult),
          totalVectorPeRow(oRow * lRow),
          totalVectorPeCol(oCol * lCol),
          systolicArrayRowDimension(oRow*lRow*mult),
          systolicArrayColDimension(oCol*lCol*mult),
          numVectorPePerGroupPe(lRow * lCol),
          contractingDimensionSize(cd),
          accDeskewSpan(computeAccDeskewSpan(df, oRow, oCol)),
          deskewSlotStride(computeDeskewSlotStride(cd, computeAccDeskewSpan(df, oRow, oCol))),
          maxContractionSize(maxContraction)
    {}

    // De-skew chain span. kind() cannot be called from the constructor, so this is
    // computed from dataflow/geometry only; hasDeskewFlush() decides at call time
    // whether it applies (Grouped only).
    static int computeAccDeskewSpan(Dataflow df, int oRow, int oCol) {
        switch (df) {
            case Dataflow::ReuseA: return oRow;
            case Dataflow::ReuseB: return oCol;
            case Dataflow::ReuseC: return 1;
        }
        return 1;
    }

    // Same choice as Scala GoldenMatrix.freeDimensionCap:
    //   the smallest s in [1, depth] that divides depth, is >= span, and leaves at
    //   least 2 slots. If none exists, fall back to depth/2 (= 50% utilisation).
    //   depth < 2 yields max(1, depth).
    static int computeDeskewSlotStride(int depth, int span) {
        if (depth < 2) return std::max(1, depth);
        for (int s = 1; s <= depth; ++s)
            if (depth % s == 0 && s >= span && depth / s >= 2) return s;
        return std::max(1, depth / 2);
    }

    // Geometry checks shared by all kinds (assert-based; compiled out with NDEBUG).
    void validateCommon() const {
        assert(outerPeRow > 0 && "outerPeRow must be positive");
        assert(outerPeCol > 0 && "outerPeCol must be positive");
        assert(localPeRow > 0 && "localPeRow must be positive");
        assert(localPeCol > 0 && "localPeCol must be positive");
        assert(multPerPe  > 0 && "multPerPe must be positive");
        assert(detail::isPowerOfTwo(multPerPe) && "multPerPe must be power of two");
        // Same as the require in Scala SystolicArrayConfig (Grouped ReuseC's tileK
        // must be divisible by the mult packing).
        assert((maxContractionSize == 0 || maxContractionSize % multPerPe == 0) &&
               "maxContractionSize must be a multiple of multPerPe");

        // The local dimension along the adder tree must be a power of two.
        switch (dataflow) {
            case Dataflow::ReuseA:
                assert(detail::isPowerOfTwo(localPeCol) &&
                       "localPeCol must be power of two for ReuseA");
                break;
            case Dataflow::ReuseB:
                assert(detail::isPowerOfTwo(localPeRow) &&
                       "localPeRow must be power of two for ReuseB");
                break;
            default:
                break;
        }
    }

private:

    //ReuseA and ReuseB are assumed as partial sum generation
    //ReuseC is assumed as re-quantized output
    // Bit width of one output element:
    //   ReuseA: 16 + ceil(log2(multPerPe * totalVectorPeCol))
    //   ReuseB: 16 + ceil(log2(multPerPe * totalVectorPeRow))  (adder-tree growth)
    //   ReuseC: inputOperandBitWidth (requantized)
    int systolicOutputPortBitWidth() const {
        switch(dataflow) {
            case Dataflow::ReuseA: return (multOutputBitwidth + detail::log2Up(multPerPe * outerPeCol * localPeCol));
            case Dataflow::ReuseB: return (multOutputBitwidth + detail::log2Up(multPerPe * outerPeRow * localPeRow));
            case Dataflow::ReuseC: return inputOperandBitWidth;
        }

        return 0;
    }

};

// Grouped PE array. Only the common geometry checks apply.
class GroupedConfig final : public SystolicArrayConfig {
public:
    GroupedConfig(int oRow, int oCol, int lRow, int lCol, int mult, Dataflow df, int cd,
             int maxContraction = 0)
        : SystolicArrayConfig(oRow, oCol, lRow, lCol, mult, df, cd, maxContraction) {
        validate();
    }
    ConfigKind kind() const override { return ConfigKind::Grouped; }

private:
    void validate() const {
        validateCommon();
    }
};

// Toroidal systolic array.
class ToroidalConfig final : public SystolicArrayConfig {
public:
    ToroidalConfig(int oRow, int oCol, int lRow, int lCol, int mult, Dataflow df, int cd,
             int maxContraction = 0)
        : SystolicArrayConfig(oRow, oCol, lRow, lCol, mult, df, cd, maxContraction) {
        validate();
    }
    ConfigKind kind() const override { return ConfigKind::Toroidal; }

private:
    void validate() const {
        validateCommon();
        // Toroidal no longer requires a square vector-PE array (broadcast feed).
        switch(dataflow){
            case Dataflow::ReuseA:
                assert(detail::isPowerOfTwo(localPeCol) &&
                       "Toroidal ReuseA localPeCol must be power of two (adder-tree dim)");
                assert(outerPeCol * localPeCol == outerPeRow &&
                       "Toroidal ReuseA: outerPeRow must equal outerPeCol * localPeCol");
                break;

            case Dataflow::ReuseB:
                assert(detail::isPowerOfTwo(localPeRow) &&
                       "Toroidal ReuseB localPeRow must be power of two (adder-tree dim)");
                assert(outerPeRow * localPeRow == outerPeCol &&
                       "Toroidal ReuseB: outerPeCol must equal outerPeRow * localPeRow");
                break;

            case Dataflow::ReuseC:
                // ReuseC still requires a square (outer PE) systolic array.
                assert(outerPeRow == outerPeCol && "Toroidal ReuseC requires square systolic array");
                break;

            default:
                break;

        }
    }
};

// ToroidalDual shares its basic components with Toroidal ReuseB, so its dataflow is
// fixed to ReuseB; GEMM on this array is also computed as ReuseB.
class ToroidalDualConfig final: public SystolicArrayConfig {
public:
    ToroidalDualConfig(int oRow, int oCol, int lRow, int lCol, int mult, int cd,
                  int maxContraction = 0)
        : SystolicArrayConfig(oRow, oCol, lRow, lCol, mult, Dataflow::ReuseB, cd,
                              maxContraction) {
        validate();
    }
    ConfigKind kind() const override { return ConfigKind::ToroidalDual;}

    private:
    void validate() const {
        validateCommon();
        assert(detail::isPowerOfTwo(localPeRow) &&
               "ToroidalDual ReuseB localPeRow must be power of two");
        assert(outerPeRow * localPeRow == outerPeCol &&
               "ToroidalDual ReuseB: outerPeCol must equal outerPeRow * localPeRow");
    }

};