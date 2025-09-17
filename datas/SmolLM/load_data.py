#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distributed-aware DataLoader utilities for Mosaic MDS corpora."""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

try:  # pragma: no cover - optional dependency at runtime
    from streaming import StreamingDataset  # type: ignore
except ImportError as exc:  # pragma: no cover - deferred failure
    StreamingDataset = None  # type: ignore
    _STREAMING_IMPORT_ERROR = exc
else:  # pragma: no cover - module is available
    _STREAMING_IMPORT_ERROR = None


def _require_streaming() -> None:
    """Ensure :mod:`streaming` is installed before constructing datasets."""

    if StreamingDataset is None:  # pragma: no cover - executed only on failure
        raise ImportError(
            "The 'streaming' package is required to build the FineWeb-Edu loader. "
            "Install it via 'pip install mosaicml-streaming'."
        ) from _STREAMING_IMPORT_ERROR


def _infer_distributed_env(world_size: Optional[int], rank: Optional[int]) -> Tuple[int, int]:
    """Infer distributed world size and rank from explicit values or env vars."""

    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    return world_size, rank


def _rank_local_cache(local_cache: Optional[str], rank: int) -> Optional[str]:
    """Derive a rank-specific cache directory to avoid collisions across ranks."""

    if not local_cache:
        return local_cache
    expanded = os.path.expanduser(local_cache)
    if rank == 0:
        return expanded
    return os.path.join(expanded, f"rank{rank:05d}")


# ------------------------- utils -------------------------
def _extract_text(sample) -> str:
    """从样本字典中抽取文本; 兼容 {'text': str} 或其他字符串字段."""

    if isinstance(sample, dict):
        value = sample.get("text")
        if isinstance(value, str):
            return value
        for candidate in sample.values():
            if isinstance(candidate, str):
                return candidate
    if isinstance(sample, str):
        return sample
    return ""


# ------------------------- collates -------------------------
class TruncCollate:
    def __init__(self, tok, seq_len: int = 2048, pad_to: int = 8):
        self.tok = tok
        self.seq_len = seq_len
        self.pad_to = pad_to

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        texts = [_extract_text(x) for x in batch]
        enc = self.tok(
            texts,
            truncation=True,
            max_length=self.seq_len,
            padding="longest",
            pad_to_multiple_of=self.pad_to,
            return_tensors="pt",
        )
        enc["attention_mask"] = enc["attention_mask"].to(torch.bool)
        labels = enc["input_ids"].masked_fill(~enc["attention_mask"], -100)
        enc["labels"] = labels
        return enc


class OverflowCollate:
    def __init__(self, tok, seq_len: int = 2048, stride: int = 128, pad_to: int = 8):
        self.tok = tok
        self.seq_len = seq_len
        self.stride = stride
        self.pad_to = pad_to

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        texts = [_extract_text(x) for x in batch]
        enc = self.tok(
            texts,
            truncation=True,
            max_length=self.seq_len,
            return_overflowing_tokens=True,
            stride=self.stride,
            padding="longest",
            pad_to_multiple_of=self.pad_to,
            return_tensors="pt",
        )
        flat: Dict[str, List[torch.Tensor]] = {k: [] for k in enc if k != "overflow_to_sample_mapping"}
        for key, tensor in enc.items():
            if key == "overflow_to_sample_mapping":
                continue
            flat[key].append(tensor)
        flat = {k: torch.cat(v, dim=0) for k, v in flat.items()}

        flat["attention_mask"] = flat["attention_mask"].to(torch.bool)
        labels = flat["input_ids"].masked_fill(~flat["attention_mask"], -100)
        flat["labels"] = labels
        return flat


# ------------------------- pack as IterableDataset -------------------------
class PackedBatchingDataset(IterableDataset):
    """Wrap :class:`StreamingDataset` to perform on-the-fly packing."""

    def __init__(
        self,
        base: "StreamingDataset",
        tok: AutoTokenizer,
        seq_len: int = 2048,
        blocks_per_batch: int = 64,
        sep_with_eos: bool = True,
        sep_id: Optional[int] = None,
        pack_batch_texts: int = 512,
    ):
        super().__init__()
        self.base = base
        self.tok = tok
        self.seq_len = seq_len
        self.blocks_per_batch = blocks_per_batch
        self.sep_with_eos = sep_with_eos
        self.sep_id = sep_id if sep_id is not None else getattr(tok, "eos_token_id", None)
        if self.sep_with_eos and self.sep_id is None:
            raise ValueError("sep_with_eos=True，但 tokenizer 没有 eos_token_id；可传 sep_id 覆盖。")
        self.pack_batch_texts = int(max(1, pack_batch_texts))
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def _batch_tokenize(self, texts: List[str]) -> List[List[int]]:
        out = self.tok(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        ids_list: List[List[int]] = out["input_ids"]
        if self.sep_with_eos and self.sep_id is not None:
            for ids in ids_list:
                if not ids or ids[-1] != self.sep_id:
                    ids.append(self.sep_id)
        return ids_list

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buf: List[int] = []
        pending: deque[str] = deque()
        base_it = iter(self.base)
        need = self.blocks_per_batch * self.seq_len

        while True:
            while len(buf) < need:
                while len(pending) < self.pack_batch_texts:
                    try:
                        sample = next(base_it)
                    except StopIteration:
                        break
                    text = _extract_text(sample)
                    if text:
                        pending.append(text)
                if not pending:
                    break

                take_n = min(self.pack_batch_texts, len(pending))
                batch_texts = [pending.popleft() for _ in range(take_n)]
                for ids in self._batch_tokenize(batch_texts):
                    buf.extend(ids)

            if len(buf) < need:
                break

            take, buf = buf[:need], buf[need:]
            block = torch.tensor(take, dtype=torch.long).view(self.blocks_per_batch, self.seq_len)
            attn = torch.ones(self.blocks_per_batch, self.seq_len, dtype=torch.bool)
            yield {"input_ids": block, "attention_mask": attn, "labels": block}


def _build_streaming_dataset(
    *,
    mds_root: str,
    local_cache: Optional[str],
    cache_limit: str,
    shuffle: bool,
    shuffle_algo: str,
    shuffle_seed: int,
    ds_batch: int,
    predownload: Optional[int],
    world_size: int,
    rank: int,
) -> "StreamingDataset":
    _require_streaming()
    cache_dir = _rank_local_cache(local_cache, rank)
    return StreamingDataset(  # type: ignore[call-arg]
        remote=mds_root,
        local=cache_dir,
        cache_limit=cache_limit,
        shuffle=shuffle,
        shuffle_algo=shuffle_algo,
        shuffle_seed=shuffle_seed,
        batch_size=ds_batch,
        predownload=predownload,
        num_canonical_nodes=max(world_size, 1),
        num_replicas=max(world_size, 1),
        rank=rank,
    )


# ------------------------- public API -------------------------
def build_train_loader(
    *,
    mds_root: str = "/mnt/datasets/fineweb-edu-mds/0-1-0/",
    strategy: str = "pack",
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    seq_len: int = 2048,
    micro_blocks: int = 64,
    per_device_batch: int = 8,
    num_workers: int = 8,
    persistent_workers: bool = True,
    local_cache: Optional[str] = "/tmp/streaming_cache",
    cache_limit: str = "250GB",
    shuffle_seed: int = 42,
    predownload: Optional[int] = 1000,
    pin_memory: bool = False,
    prefetch_factor: int = 1,
    pack_batch_texts: int = 512,
    world_size: Optional[int] = None,
    global_rank: Optional[int] = None,
) -> DataLoader:
    """Construct a distributed-ready :class:`DataLoader` for MDS corpora."""

    world_size, global_rank = _infer_distributed_env(world_size, global_rank)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = int(1e12)

    ds_batch = 1 if strategy == "pack" else per_device_batch
    base_ds = _build_streaming_dataset(
        mds_root=mds_root,
        local_cache=local_cache,
        cache_limit=cache_limit,
        shuffle=True,
        shuffle_algo="py1e",
        shuffle_seed=shuffle_seed,
        ds_batch=ds_batch,
        predownload=predownload,
        world_size=world_size,
        rank=global_rank,
    )

    if strategy == "pack":
        dataset = PackedBatchingDataset(
            base_ds,
            tok,
            seq_len=seq_len,
            blocks_per_batch=micro_blocks,
            sep_with_eos=True,
            sep_id=getattr(tok, "eos_token_id", None),
            pack_batch_texts=pack_batch_texts,
        )

        def _identity(batch_list):
            return batch_list[0]

        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            collate_fn=_identity,
        )
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(0)
        return loader

    if strategy == "overflow":
        collate = OverflowCollate(tok, seq_len=seq_len, stride=128)
    elif strategy == "trunc":
        collate = TruncCollate(tok, seq_len=seq_len, pad_to=8)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    loader = DataLoader(
        base_ds,
        batch_size=per_device_batch,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        collate_fn=collate,
    )
    if hasattr(base_ds, "set_epoch"):
        base_ds.set_epoch(0)
    return loader


__all__ = ["build_train_loader"]

