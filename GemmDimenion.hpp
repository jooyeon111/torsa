#pragma once

// GEMM problem dimensions of a layer: C (M x N) = A (M x K) * B (K x N),
// where K is the contracted (summed-over) axis.
// (The file name keeps its historical misspelling "Dimenion".)
class GemmDimension {
public:
    const int dimensionM;
    const int dimensionN;
    const int dimensionK;

    GemmDimension( int m, int n, int k)
        : dimensionM(m),
        dimensionN(n),
        dimensionK(k)
    {}

    

};