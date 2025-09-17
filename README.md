# Multi-node LLM Training Playground

This repository packages a ready-to-run training entrypoint and data-loading
utilities for large language model pretraining across **4 nodes / 32 A100
GPUs**.  The codebase adapts a single-node script into a `torchrun`
compatible workflow using **Fully Sharded Data Parallel (FSDP)** and
streaming datasets.

## Repository layout

```
.
├── train.py                # Torch/FSDP training driver for multi-node jobs
├── datas/
│   ├── __init__.py         # Dataset package exports
│   └── SmolLM/
│       └── load_data.py    # Mosaic MDS streaming loader with packing support
└── README.md
```

## Key features

* End-to-end distributed training setup with automatic rank/world-size
  detection for elastic launches.
* FSDP wrapping, optional activation checkpointing, flash attention toggle,
  and gradient accumulation for large effective batch sizes.
* Streaming dataloader for MosaicML MDS corpora (e.g., FineWeb-Edu) with
  on-the-fly sequence packing and per-rank cache directories.
* Configurable checkpoint formats (full, sharded, or local) with resume
  support, TensorBoard logging, and optional C4 perplexity evaluation.

## Requirements

* Python 3.10+
* [PyTorch](https://pytorch.org/get-started/locally/) with CUDA and
  distributed support (tested with A100 GPUs).
* [Transformers](https://huggingface.co/docs/transformers/index) 4.39+.
* `tensorboard` (only if logging is enabled).
* Optional: [`mosaicml-streaming`](https://github.com/mosaicml/streaming)
  when using the FineWeb-Edu loader.
* Project-specific helper packages referenced by the script (`utils`,
  `slt`) must be available on `PYTHONPATH`.

Install dependencies via pip (adjust CUDA wheels as needed):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers tensorboard mosaicml-streaming
```

## Dataset loaders

The repository ships a streaming loader tuned for MosaicML's MDS format:

```python
from datas.SmolLM import load_data
loader = load_data.build_train_loader(
    mds_root="/path/to/fineweb-edu",
    strategy="pack",              # or "overflow" / "trunc"
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    seq_len=2048,
    micro_blocks=64,
    local_cache="/tmp/mds_cache"
)
```

The loader automatically infers `rank`/`world_size` from environment
variables set by `torchrun`, isolates cache directories per rank, and
supports packing text streams into fixed-length blocks for maximum
throughput.

## Launching training

A typical 4-node (8 GPUs per node) launch looks like:

```bash
torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train.py \
    --data_dir=/datasets/fineweb-edu \
    --data_name=FineWeb-Edu \
    --local_cache_dir=/local/mds_cache \
    --config=/path/to/llama/config.json \
    --tokenizer_dir=/path/to/tokenizer \
    --output_dir=/checkpoints/run1 \
    --tensorboard_dir=/logs/tensorboard \
    --global_batch=256 \
    --local_batch=32 \
    --train_steps=200000 \
    --enable_tensorboard \
    --bf16
```

`train.py` derives the distributed topology from `torchrun` environment
variables, configures gradient accumulation based on `global_batch`, and
wraps the model with FSDP (`FULL_SHARD` strategy by default).  Optional
flags enable activation checkpointing, flash attention, sparse linear
transform (SLT) modules, and evaluation on the C4 validation set.

## Checkpoints and resume

Checkpoints are stored under `output_dir/<run_name>/` using the naming
pattern `step_<step_id>/`.  Choose one of three formats via
`--ckpt_type`:

* `full` – rank 0 saves a full model state_dict.
* `shard` *(default)* – each rank persists its FSDP shard.
* `local` – local state per rank.

To resume, pass `--resume_dir` pointing to a saved step directory.  The
script restores optimizer, scheduler, and gradient scaler states unless
`--skip_optimizer` is provided.

## Monitoring and evaluation

* When `--enable_tensorboard` is set, rank 0 writes metrics to
  `tensorboard_dir/<run_name>/` and broadcasts the log directory name to
  all ranks.
* Set `--eval_every` to a positive integer to periodically measure C4
  perplexity via `datas.C4.c4_eval.evaluate_c4_ppl` (the helper module
  must be importable during evaluation).

## Customisation tips

* Adjust `micro_blocks` / `local_batch` to balance throughput vs. GPU
  memory usage.
* Override tokenizer or dataset paths to point at your local copies.
* Extend `datas/` with additional loaders and import them in `train.py`
  via the `data_name` argument.

Happy training!
