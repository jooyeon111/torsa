#pragma once
#include <string>
#include <fstream>
#include <stdexcept>
#include "Layer.hpp"
#include "GemmDimenion.hpp"

// Parser for layer/*.cfg files (layer = GEMM configuration).
// Replaces the Layer("Test", GemmDimension(16,16,16)) that used to be hard-coded in main.cpp.
//
// File format: one "key : value" (or "key = value") per line.
//   - Blank lines and lines starting with '#' are ignored.
//   - Leading/trailing whitespace around key and value is stripped.
//   - The line is split at the first ':' or '='.
//
// Required keys (unknown keys are an error):
//   name        layer name (string)
//   dimensionM  GEMM M
//   dimensionN  GEMM N
//   dimensionK  GEMM K
namespace layer_cfg_detail {

// Strip leading/trailing spaces, tabs, CR and LF.

inline std::string trim(const std::string& s) {
    std::size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    std::size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

} // namespace layer_cfg_detail

// Parse a layer configuration file. Throws std::runtime_error if the file cannot be opened,
// a line is malformed, a key is unknown or a required field is missing (and propagates
// std::stoi exceptions for non-numeric dimensions).
inline Layer parseLayer(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("layer config: cannot open file \"" + path + "\"");

    std::string name;
    int m = 0, n = 0, k = 0;
    bool haveName = false, haveM = false, haveN = false, haveK = false;

    std::string line;
    while (std::getline(in, line)) {
        std::string trimmed = layer_cfg_detail::trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        std::size_t sep = trimmed.find_first_of(":=");
        if (sep == std::string::npos)
            throw std::runtime_error(
                "layer config: ':' or '=' expected in \"" + line + "\"");

        std::string key = layer_cfg_detail::trim(trimmed.substr(0, sep));
        std::string val = layer_cfg_detail::trim(trimmed.substr(sep + 1));
        if (val.empty())
            throw std::runtime_error(
                "layer config: empty value for \"" + key + "\" in \"" + path + "\"");

        if (key == "name") {
            name = val; haveName = true;
        } else if (key == "dimensionM") {
            m = std::stoi(val); haveM = true;
        } else if (key == "dimensionN") {
            n = std::stoi(val); haveN = true;
        } else if (key == "dimensionK") {
            k = std::stoi(val); haveK = true;
        } else {
            throw std::runtime_error(
                "layer config: unknown key \"" + key + "\" in \"" + path + "\"");
        }
    }

    if (!haveName || !haveM || !haveN || !haveK)
        throw std::runtime_error(
            "layer config: missing required field(s) in \"" + path + "\"");

    return Layer(name, GemmDimension(m, n, k));
}
