# FlashMLA-ETAP

**Efficient Transpose Attention Pipeline for Multi-Head Latent Attention (MLA) decoding on NVIDIA Hopper GPUs**

📄 Paper: [*FlashMLA-ETAP: Efficient Transpose Attention Pipeline for Accelerating MLA Inference on NVIDIA H20 GPUs*](https://arxiv.org/abs/2506.01969) (arXiv:2506.01969) · [PDF](https://arxiv.org/pdf/2506.01969) · Upstream: [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)

FlashMLA-ETAP is a CUDA decoding kernel for Multi-Head Latent Attention that extends DeepSeek's FlashMLA with the **Efficient Transpose Attention Pipeline (ETAP)**. It is designed for the *single-instance* deployment of large MLA models such as DeepSeek-R1 (671B) on a single 8-GPU NVIDIA H20 server, where each GPU serves only 16 attention heads and the attention kernel therefore processes a handful of query rows against key-value contexts of tens of thousands of tokens.

On the NVIDIA H20, FlashMLA-ETAP reaches **89 TFLOPS at a 64K context (batch size 16), a 2.78× speedup over FlashMLA**, and **5.24× / 4.94× over FlashAttention-3 / FlashInfer**, while its FP16 output error (RMSE 1.25×10⁻⁵ against an FP64 reference) is 15.2× lower than that of FlashAttention-3. Full details are given in the paper and summarized in [Performance](#performance).

---

## Contents

- [Motivation](#motivation)
- [Method: the Efficient Transpose Attention Pipeline](#method-the-efficient-transpose-attention-pipeline)
- [Performance](#performance)
- [Supported configurations](#supported-configurations)
- [Installation](#installation)
- [Usage](#usage)
- [Testing and benchmarking](#testing-and-benchmarking)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Motivation

Hopper's WarpGroup Matrix-Multiply-Accumulate (WGMMA) instructions require an $`M`$ dimension of at least 64. In the conventional attention formulation the $`M`$ dimension of both GEMMs is the number of query rows, i.e. `seq_len_q × num_heads_q / num_heads_kv`. During autoregressive decoding of DeepSeek-R1 on an 8-GPU H20 server, the 128 attention heads are split across the GPUs (16 heads per GPU) and each step decodes one token, so a GPU issues attention over **16 query rows**. The remaining 48 rows of every WGMMA tile are padding, and the paper reports compute utilization of **below 25%** for FlashMLA in this regime. The problem is most acute on the H20, whose FP16 tensor-core throughput (148 TFLOPS) is a small fraction of that of the H100/H800 (1979 TFLOPS), while its KV contexts are just as long. In this setting MLA accounts for roughly 30% of a decoding forward pass of DeepSeek-V3 (batch size 16, 16K context).

## Method: the Efficient Transpose Attention Pipeline

Standard attention computes, for one head,

$$S = Q K^{\top},\qquad P = \mathrm{softmax}(S),\qquad O = P V .$$

ETAP evaluates the **transposed** pipeline instead:

$$S^{\top} = K\,Q^{\top} \in \mathbb{R}^{N \times N_q},\qquad
P^{\top} = \mathrm{softmax}(S^{\top}),\qquad
O^{\top} = V^{\top} P^{\top} \in \mathbb{R}^{d \times N_q},\qquad
O = (O^{\top})^{\top},$$

where $`N`$ is the KV context length and $`N_q`$ the number of query rows. The KV context length now occupies the WGMMA $`M`$ dimension and the query rows occupy the $`N`$ dimension, which has no minimum-size constraint of 64. Padding on the query dimension is eliminated; the single transpose of the output is performed once per tile, whereas the two GEMMs benefit at every KV block. The gain grows with the ratio of context length to query length, which is exactly the decoding regime.

| GEMM | Conventional layout (FlashMLA) | ETAP layout (this repository) |
|---|---|---|
| Scores | $`M`$ = query rows (padded to 64), $`N`$ = keys, $`K`$ = head_dim | $`M`$ = keys (64 per block), $`N`$ = query rows (16), $`K`$ = head_dim |
| Output | $`M`$ = query rows (padded to 64), $`N`$ = head_dim_v, $`K`$ = keys | $`M`$ = head_dim_v, $`N`$ = query rows (16), $`K`$ = keys |

### Kernel structure

FlashMLA-ETAP keeps FlashMLA's overall design (paged KV cache, split-KV scheduling with a combine kernel, variable-length batches) and re-implements the inner loop in the transposed form. Each CTA processes one tile of **16 query rows × 64 keys** per iteration with 256 threads organised as two warp groups:

- **Consumer warp group 0** computes $`S^{\top}_j = K_j Q^{\top}`$ with a shared-memory WGMMA, runs the online softmax along the key axis (which now spans threads and warps: the running maximum is reduced across lanes and warps in every iteration, the row sums once in the epilogue), writes $`P^{\top}_j`$ and the rescale factors to shared memory, and accumulates the first half of $`O^{\top}`$ ($`V_{j,0}^{\top} P^{\top}_j`$).
- **Producer warp group 1** streams the K/V pages from HBM into a double-buffered shared-memory ring with `cp.async`, waits for $`P^{\top}_j`$ via a named barrier, and accumulates the second half of $`O^{\top}`$ ($`V_{j,1}^{\top} P^{\top}_j`$).
- **Epilogue**: both halves are normalised by the softmax denominator, $`O^{\top}`$ is transposed to $`O`$ in shared memory, and $`O`$ together with the log-sum-exp is written to HBM (or to the split-KV accumulators, which the combine kernel reduces).

All producer/consumer hand-offs through shared memory are ordered with named barriers and `fence.proxy.async`, so results are deterministic across runs.

## Performance

All numbers below are taken from the paper (Section 4). Setup: NVIDIA H20 (96 GB HBM3, 4.0 TB/s, 148 TFLOPS FP16), DeepSeek-R1 configuration with 16 heads per GPU and head dimension 576, FP16, one decoded token per forward pass, each measurement averaged over five runs. Baselines: FlashAttention-3, FlashInfer and FlashMLA.

**Decoding throughput, batch size 16 (TFLOPS)**

| Sequence length | 512 | 1K | 2K | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|---|---|
| FlashAttention-3 | 10 | 15 | 19 | 16 | 17 | 17 | 17 | 17 |
| FlashInfer | 8 | 16 | 20 | 23 | 18 | 19 | 18 | 18 |
| FlashMLA | 9 | 13 | 19 | 23 | 27 | 30 | 31 | 32 |
| **FlashMLA-ETAP** | **13** | **21** | **34** | **46** | **61** | **75** | **85** | **89** |

**Decoding throughput, batch size 32 (TFLOPS)**

| Sequence length | 512 | 1K | 2K | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|---|---|
| FlashAttention-3 | 15 | 20 | 18 | 19 | 20 | 21 | 21 | 21 |
| FlashInfer | 16 | 20 | 23 | 22 | 23 | 23 | 23 | 23 |
| FlashMLA | 12 | 18 | 22 | 26 | 29 | 31 | 32 | 32 |
| **FlashMLA-ETAP** | **20** | **32** | **43** | **59** | **73** | **83** | **87** | **87** |

The speedup over FlashMLA increases with context length, from 1.44× at 512 tokens to 2.78× at 64K tokens (batch size 16); at 64K tokens FlashMLA-ETAP is 2.72× faster than FlashMLA for batch size 32, and 5.24× / 4.14× faster than FlashAttention-3 and 4.94× / 3.78× faster than FlashInfer for batch sizes 16 / 32.

**Numerical accuracy** (FP16 outputs against an FP64 reference, following a methodology similar to that of the FlashAttention-3 paper)

| Framework | RMSE |
|---|---|
| FlashAttention-3 | 1.9 × 10⁻⁴ |
| **FlashMLA-ETAP** | **1.25 × 10⁻⁵** |

*Scope of the evaluation.* The paper evaluates a single GPU model (H20), autoregressive decoding with one token per step, 16 heads with head dimension 576 and contexts up to 64K tokens. Performance on other Hopper GPUs or for other shapes has not been characterised and may differ. The scripts listed under [Testing and benchmarking](#testing-and-benchmarking) run the FlashMLA-ETAP configuration of the paper and the two baseline libraries; the numbers above were obtained on an H20 and will not be reproduced on other GPUs.

## Supported configurations

| | |
|---|---|
| GPU | NVIDIA Hopper (`sm_90a`); tuned for the H20, runs on any Hopper GPU |
| Data types | BF16, FP16 (FP16 can be excluded at build time, see below) |
| KV cache | Paged, page (block) size 64; K and V share one cache tensor (MLA latent + RoPE part) |
| Head dimensions `(head_dim, head_dim_v)` | `(576, 512)` — DeepSeek-V2/V3/R1, `kv_lora_rank = 512`; `(320, 256)` — `kv_lora_rank = 256`. In both cases `head_dim_v = head_dim − 64`, the 64 being `qk_rope_head_dim`. |
| Batching | Variable sequence lengths per batch element (`cache_seqlens`), any `num_heads_q` that is a multiple of `num_heads_kv` |
| Query length | Any `seq_len_q`; causal masking for multi-token queries (e.g. speculative / MTP decoding) |
| Scheduling | Split-KV across SMs with the FlashMLA tile scheduler and combine kernel |

Adding a further head-dimension pair requires one explicit instantiation per `.cu` file and, in `csrc/flash_api.cpp`, extending the `head_size` check and the dispatch in both data-type branches; the kernel constraints (`head_dim % 32 == 0`, `head_dim_v % 128 == 0`, `head_dim_v ≤ head_dim`) are enforced by `static_assert`s.

## Installation

Requirements:

- NVIDIA Hopper GPU
- CUDA 12.3 or newer (12.8 or newer is strongly recommended for best performance)
- PyTorch 2.0 or newer

```bash
git clone https://github.com/pengcuo/FlashMLA-ETAP.git
cd FlashMLA-ETAP
python setup.py install
```

CUTLASS is vendored under `csrc/cutlass`; no additional dependencies are required. To build the BF16 kernels only:

```bash
FLASH_MLA_DISABLE_FP16=TRUE python setup.py install
```

## Usage

The Python API is identical to FlashMLA's.

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

# Once per decoding step (or whenever cache_seqlens changes):
tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens,              # (batch_size,), int32: current KV length of every sequence
    s_q * h_q // h_kv,          # query rows per KV head
    h_kv,                       # number of KV heads
)

for layer in range(num_layers):
    ...
    o, lse = flash_mla_with_kvcache(
        q,                      # (batch_size, s_q, h_q, head_dim)
        kv_cache,               # (num_blocks, 64, h_kv, head_dim), K and V share this tensor
        block_table,            # (batch_size, max_blocks_per_seq), int32
        cache_seqlens,          # (batch_size,), int32
        head_dim_v,             # 512 for head_dim 576, 256 for head_dim 320
        tile_scheduler_metadata,
        num_splits,
        softmax_scale=None,     # defaults to head_dim ** -0.5
        causal=True,
    )
    # o:   (batch_size, s_q, h_q, head_dim_v), same dtype as q
    # lse: (batch_size, h_q, s_q), float32 log-sum-exp of the attention logits
```

`q` and `kv_cache` must be BF16 or FP16 with a contiguous last dimension, and the KV cache page size must be 64. Unsupported head dimensions, data types and strides are rejected with a descriptive error.

## Testing and benchmarking

**Correctness.** `tests/test_flash_mla.py` compares the kernel with an FP32 PyTorch reference at batch size 128 over context lengths of 4K and 8K tokens, 16 to 128 heads and query lengths of 1 and 2 (fixed- and variable-length, causal), and reports throughput for every configuration:

```bash
python tests/test_flash_mla.py                      # head_dim 576 / head_dim_v 512, BF16
python tests/test_flash_mla.py --dtype fp16         # FP16
python tests/test_flash_mla.py --kv-lora-rank 256   # head_dim 320 / head_dim_v 256
```

**Paper benchmark.** `tests/pengcuo_test_flash_mla.py` runs the configuration used in the paper (batch size 16, 16 heads, `s_q = 1`, contexts from 512 to 64K tokens) and prints latency, TFLOPS and bandwidth for each length:

```bash
python tests/pengcuo_test_flash_mla.py [--dtype bf16|fp16]
```

**Baselines.** `benchmark/pengcuo_test_fa3_mla.py` (requires FlashAttention-3, `flash_attn_interface`) and `benchmark/pengcuo_test_flashinfer_mla.py` (requires FlashInfer) time the paper's head count, head dimensions and context lengths with the two baseline libraries; both are hard-coded to batch size 32 and should be edited to match other configurations. `benchmark/bench_flash_mla.py` and `benchmark/visualize.py`, inherited from FlashMLA, provide a broader comparison against a PyTorch/Triton/FlashInfer baseline and plot the resulting CSV files.

## Limitations

- The kernel is specialised for decoding-style workloads with few query rows per KV head (16 rows are processed per tile). Prefill or other workloads with many query rows should use a kernel with a query-major tiling such as upstream FlashMLA or FlashAttention-3.
- Only the two head-dimension pairs listed above and a page size of 64 are supported; forward pass only; no FP8.
- Reported performance is specific to the NVIDIA H20; the benefit of ETAP on GPUs with substantially higher tensor-core throughput has not been evaluated.
- Integration of ETAP into FlashAttention-3 and FlashInfer is analysed theoretically in the paper but not implemented here.

## Acknowledgements

FlashMLA-ETAP is built on [FlashMLA](https://github.com/deepseek-ai/FlashMLA) by DeepSeek, which in turn draws on [FlashAttention-2/3](https://github.com/Dao-AILab/flash-attention) and [CUTLASS](https://github.com/NVIDIA/cutlass). The paper is a collaboration between Tencent, Shenzhen University and Shenzhen Polytechnic University.

## Citation

If you use FlashMLA-ETAP in your research, please cite:

```bibtex
@misc{dege2025flashmlaetap,
      title         = {FlashMLA-ETAP: Efficient Transpose Attention Pipeline for Accelerating MLA Inference on NVIDIA H20 GPUs},
      author        = {Pengcuo Dege and Qiuming Luo and Rui Mao and Chang Kong},
      year          = {2025},
      eprint        = {2506.01969},
      archivePrefix = {arXiv},
      primaryClass  = {cs.DC},
      url           = {https://arxiv.org/abs/2506.01969},
}
```

## License

This project is released under the MIT License (see [LICENSE](LICENSE)); the FlashMLA components retain their original DeepSeek copyright.
