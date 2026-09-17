#pragma once

#include "cutlass/barrier.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts

enum class NamedBarriers {
    SReady = 1,
    SoftmaxReady = 2,
    // Barriers private to the softmax warp group (kNThreadsS = 128 threads). They must never be
    // implemented with __syncthreads(): the producer warp group runs a different code path.
    SoftmaxMaxReduce = 3,   // cross-warp row-max exchange through shared memory (softmax.h)
    PReady = 4,             // sP fully written by all 4 softmax warps before the PV wgmma reads it
};

} // flash
