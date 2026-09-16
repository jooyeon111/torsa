#pragma once
#include "GemmDimenion.hpp"
#include <string>

// A workload layer: a name plus the GEMM dimensions it maps to.
// Built from a layer/*.cfg file by parseLayer() (LayerParser.hpp).
class Layer {

public:
    const std::string layerName;
    const GemmDimension gemmDimenion;   // member name keeps the historical misspelling
    
    Layer( std::string lName, GemmDimension gDimension)
        : layerName(lName),
        gemmDimenion(gDimension)
    {}
};