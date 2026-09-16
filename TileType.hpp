#pragma once

// Kind of tile in the tiled GEMM C = A x B:
//   TypeA - tile of operand A (m x k), TypeB - tile of operand B (k x n),
//   TypeC - tile of the output/partial-sum matrix C (m x n).
enum class TileType {TypeA, TypeB, TypeC};