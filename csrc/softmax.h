// Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/softmax.h
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "utils.h"
#include "named_barrier.h"

#include <cstdio>
#include <cooperative_groups.h>
#include <cmath>
#include <cooperative_groups/reduce.h>

#include <cutlass/cutlass.h>


namespace cg = cooperative_groups;

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void warp_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<32>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

__device__ __forceinline__ float compute_group_max(float val) {
    const unsigned int lane_id = threadIdx.x % 32;
    const int group_id = lane_id % 4;
    const unsigned mask_group = 0x11111111 << group_id; 

    val = fmaxf(val, __shfl_xor_sync(mask_group, val, 16));
    val = fmaxf(val, __shfl_xor_sync(mask_group, val, 8));
    val = fmaxf(val, __shfl_xor_sync(mask_group, val, 4));

    return val;
}

__device__ __forceinline__ float compute_group_sum(float val) {
    const unsigned int lane_id = threadIdx.x % 32;
    const int group_id = lane_id % 4;
    const unsigned mask_group = 0x11111111 << group_id;

    val += __shfl_xor_sync(mask_group, val, 16);
    val += __shfl_xor_sync(mask_group, val, 8);
    val += __shfl_xor_sync(mask_group, val, 4);

    return val;
}

// ---------------------------------------------------------------------------------------------------
// Cross-thread reductions for the *transposed* score tile S^T = K * Q^T (kBlockN x kBlockM = 64 x 16).
//
// In the transposed wgmma accumulator layout every thread owns 4 query columns (kNRows == 4) and a few
// key rows, so a softmax statistic (max / sum over the keys) has to be reduced
//   (1) inside the thread                                  -> thread_reduce_()
//   (2) across the 8 lanes of a warp that share lane % 4     -> compute_group_{max,sum}() (xor 16/8/4)
//       (those lanes hold the same query columns)
//   (3) across the warps of a warp group                     -> shared-memory exchange below
// ---------------------------------------------------------------------------------------------------

// Step (3) for the running row max.
//
// This is executed inside the main loop by the kNThreadsSync (= 128) threads of the softmax warp group
// ONLY, while the producer warp group is on a completely different code path (cp.async loads / its own
// PV gemm). The barrier therefore has to be a *named* barrier covering exactly those threads.
// A __syncthreads() here would have to be "matched" by an unrelated __syncthreads() somewhere in the
// producer loop, which is undefined behaviour (the two warp groups are not converged) and deadlocks as
// soon as the number of barriers on the two paths stops being identical.
template<int kNThreadsSync, typename Engine1, typename Layout1>
__device__ __forceinline__ void group_4x8_4_max(Tensor<Engine1, Layout1> &max) {
    constexpr int query_size = decltype(size(max))::value;          // == kNRows (4 query columns / thread)
    constexpr int kNWarpsSync = kNThreadsSync / 32;
    static_assert(kNThreadsSync % 32 == 0, "named barriers count whole warps");
    __shared__ float shared_max[kNWarpsSync][4 * query_size];

    const int lane = threadIdx.x % 32;
    const int col = lane % 4;
    const int row = lane / 4;
    const int warp_id = (threadIdx.x % kNThreadsSync) >> 5;         // warp index inside the warp group

    #pragma unroll
    for (int mi = 0; mi < query_size; mi++) {
        max(mi) = compute_group_max(max(mi));
    }

    if (row == 0) {
        #pragma unroll
        for (int mi = 0; mi < query_size; mi++) {
            shared_max[warp_id][col * query_size + mi] = max(mi);
        }
    }
    // Only the softmax warp group takes part -> 128-thread named barrier, NOT __syncthreads().
    cutlass::arch::NamedBarrier::sync(kNThreadsSync, static_cast<int>(NamedBarriers::SoftmaxMaxReduce));

    #pragma unroll
    for (int mi = 0; mi < query_size; mi++) {
        #pragma unroll
        for (int r = 0; r < kNWarpsSync; r++) {
            max(mi) = fmaxf(max(mi), shared_max[r][col * query_size + mi]);
        }
    }
    // The write-after-read hazard on shared_max between two consecutive calls is covered by the
    // __syncthreads() at the top of the main loop, which every thread of the CTA executes each iteration.
}

// Step (3) for the row sum.
//
// Executed once, in the epilogue, by ALL kNThreads threads of the CTA (both warp groups hold identical
// row_sum values at that point and both need the reduced value to normalise their half of O), so a
// CTA-wide __syncthreads() is the right barrier here. Each warp group reduces across its own warps.
template<int kNThreads, int kNThreadsPerGroup, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_group_4x8_4_sum(Tensor<Engine1, Layout1> &sum) {
    constexpr int query_size = decltype(size(sum))::value;
    constexpr int kNWarps = kNThreads / 32;
    constexpr int kNWarpsPerGroup = kNThreadsPerGroup / 32;
    static_assert(kNThreads % kNThreadsPerGroup == 0 && kNThreadsPerGroup % 32 == 0);
    // Sized for every warp of the CTA: the old [4][...] array overflowed for warps 4..7 (warp group 1).
    __shared__ float shared_sum[kNWarps][4 * query_size];

    const int lane = threadIdx.x % 32;
    const int col = lane % 4;
    const int row = lane / 4;
    const int warp_id = threadIdx.x >> 5;
    const int group_base = (warp_id / kNWarpsPerGroup) * kNWarpsPerGroup;

    #pragma unroll
    for (int mi = 0; mi < query_size; mi++) {
        sum(mi) = compute_group_sum(sum(mi));
    }

    if (row == 0) {
        #pragma unroll
        for (int mi = 0; mi < query_size; mi++) {
            shared_sum[warp_id][col * query_size + mi] = sum(mi);
        }
    }
    __syncthreads();   // all kNThreads threads of the CTA call this function

    #pragma unroll
    for (int mi = 0; mi < query_size; mi++) {
        float all_warp_sum = 0.f;
        #pragma unroll
        for (int r = 0; r < kNWarpsPerGroup; r++) {
            all_warp_sum += shared_sum[group_base + r][col * query_size + mi];
        }
        sum(mi) = all_warp_sum;
    }
}

template<bool zero_init=true, int kNThreadsSync, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_group_4x8_4_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    thread_reduce_<zero_init>(tensor, max, max_op);

    group_4x8_4_max<kNThreadsSync>(max);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ auto scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            // The following macro will disable the use of fma.
            // See: https://github.com/pytorch/pytorch/issues/121558 for more details
            // This macro is set in PyTorch and not FlashAttention
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            #endif
        }
    }
    return tensor;
}

// Apply the exp to all the elements.
template <bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        MaxOp<float> max_op;
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        max(mi) = Allreduce<4>::run(max(mi), max_op);
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
        sum(mi) = 0;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            sum(mi) += tensor(mi, ni);
        }
        SumOp<float> sum_op;
        sum(mi) = Allreduce<4>::run(sum(mi), sum_op);
    }
}

template<typename Tensor0, typename Tensor1>
__forceinline__ __device__ void rescale_o(Tensor0 &acc_o, Tensor1 &scale_o) {
    // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol<true>(acc_o.layout()));
    #pragma unroll
    for (int mi = 0; mi < size(scale_o); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale_o(mi); }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// kNThreads  : threads per CTA (both warp groups) -> used by the epilogue row-sum reduction
// kNThreadsS : threads of the softmax warp group   -> used by the in-loop row-max reduction
template <int kNRows, int kNThreads = 256, int kNThreadsS = 128>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT softmax(Tensor0 &acc_s, float softmax_scale_log2) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol<true>(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        TensorT scale_o;
        clear(scale_o);

        if (Is_first) {
            // blockn = 64, row_max
            // flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::template reduce_group_4x8_4_max</*zero_init=*/true, kNThreadsS>(scores, row_max);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);

            flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            // flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            flash::template reduce_group_4x8_4_max</*zero_init=*/false, kNThreadsS>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                scale_o(mi) = scores_scale;
                row_sum(mi) *= scores_scale;
            }
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
        return scale_o;
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        // quad_allreduce_(row_sum, row_sum, sum_op);
        reduce_group_4x8_4_sum<kNThreads, kNThreadsS>(row_sum);

        TensorT lse = make_fragment_like(row_sum);
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol<true>(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
};

}  // namespace flash
