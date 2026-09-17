#include "flash_fwd_mla_kernel.h"

// (head_dim, head_dim_v) = (kv_lora_rank + 64, kv_lora_rank)
template void run_mha_fwd_splitkv_mla<cutlass::half_t, 576, 512>(Flash_fwd_mla_params &params, cudaStream_t stream);
template void run_mha_fwd_splitkv_mla<cutlass::half_t, 320, 256>(Flash_fwd_mla_params &params, cudaStream_t stream);
