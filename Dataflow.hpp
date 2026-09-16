#pragma once

// Systolic-array dataflow: which operand is held stationary inside the array
// while the other operand(s) are streamed through it.
enum class Dataflow {
    ReuseA, //input stationary (A held, B streamed)
    ReuseB, //weight stationary (B held, A streamed)
    ReuseC //output stationary (A and B both streamed, C accumulated in place)
};