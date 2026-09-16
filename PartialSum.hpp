#pragma once

// PartialSumId: identifier of one unit GEMM operation C[m][n][k], i.e. the
// partial sum contributed by input tile A(m,k) x weight tile B(k,n) into the
// output tile C(m,n). The Scheduler emits these in execution order and the
// Simulator processes them one by one.

#include <ostream>

struct PartialSumId{
    int indexM;   // output-row tile index (A and C)
    int indexN;   // output-column tile index (B and C)
    int indexK;   // contraction tile index (A and B)
    // Marks the last op of an RMW accumulation block (currently only emitted by
    // Scheduler for ReuseB). Once this op is processed every C tile of that block
    // is complete, so it is a safe point for the Simulator to drain the write face.
    // (ReuseA, ReuseC and non-blocked paths keep the default false -> no effect.)
    bool endsBlock = false;
};

inline std::ostream& operator<<(std::ostream& os, const PartialSumId& op) {
    os << "op(m=" << op.indexM
       << ", n=" << op.indexN
       << ", k=" << op.indexK << ")";
    return os;
}