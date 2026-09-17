# FlashMLA-ETAP


## Introduction

FlashMLA-ETAP is an efficient MLA decoding kernel for Hopper H20 GPUs, optimized for the single-instance deployment scenario.

FlashMLA is an efficient MLA decoding kernel for Hopper GPUs, optimized for variable-length sequences serving.

Currently released:
- BF16, FP16
- Paged kvcache with block size of 64
- Head dims `(head_dim, head_dim_v)` = `(576, 512)` (DeepSeek-V2/V3, `kv_lora_rank=512`) and `(320, 256)` (`kv_lora_rank=256`); `head_dim_v = head_dim - 64` (the 64 is `qk_rope_head_dim`)

## Requirements

- Hopper GPUs
- CUDA 12.3 and above
    - **But we highly recommend 12.8 or above for the best performance**
- PyTorch 2.0 and above

## Quick start

### Install

```bash
python setup.py install
```

### Benchmark

```bash
python3 tests/pengcuo_test_flash_mla.py 
```

### Test

```bash
python tests/test_flash_mla.py                     # head_dim 576 / head_dim_v 512 (kv_lora_rank=512)
python tests/test_flash_mla.py --kv-lora-rank 256  # head_dim 320 / head_dim_v 256 (kv_lora_rank=256)
```


### Usage

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True,
    )
    ...
```

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.



## Citation
```bibtex
@misc{flashmla2025,
      title={FlashMLA-ETAP: Efficient Transpose Attention Pipeline for Accelerating MLA Inference on NVIDIA H20 GPUs},
      author={Pengcuo Dege, Chang Kong},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/pengcuo/FlashMLA-ETAP}},
}
```
