#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distributed training entrypoint for 4-node / 32-GPU A100 clusters.

This script converts the single-node training loop provided by the user into
an elastic, multi-node aware launcher that can be started with ``torchrun``.
The code retains the original model / optimizer logic but augments it with
distributed-safe setup, checkpointing, and logging utilities so that it can be
executed on a 4-node cluster (8 GPUs per node) without additional changes.

Typical launch command (replace the node specific arguments accordingly)::

    torchrun \
        --nproc_per_node=8 \
        --nnodes=4 \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        train.py --config ...

The script automatically derives ``rank``/``world_size``/``local_rank`` from
the distributed environment variables set by ``torchrun``.
"""

import argparse
import functools
import json
import os
import pathlib
import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup

# ----------------------------------------------------------------------
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard

# 允许 ShardedTensor 及其 Shard 部分都可被 pickle
torch.serialization.add_safe_globals([ShardedTensor, Shard])

# ----------------------------------------------------------------------
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# ----------------------------------------------------------------------
# tools from utils
from utils import *  # noqa: F401, F403

# SLT wrapper
from slt.slt_wrapper import *  # noqa: F401, F403


# -------------------- CLI --------------------
def get_args():
    parser = argparse.ArgumentParser()
    # 主要参数
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--data_name",
        required=False,
        default="Dolma",
        choices=["Dolma", "RedPajama", "FineWeb-Edu"],
    )
    parser.add_argument("--local_cache_dir", required=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tensorboard_dir", required=True)
    parser.add_argument("--slt_config", required=False, help="slt对应的mask和sparsity方法")
    # 训练超参
    parser.add_argument("--global_batch", type=int, default=256)
    parser.add_argument("--local_batch", type=int, default=32)
    parser.add_argument("--train_steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_mid", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    # 开关
    parser.add_argument("--enable_ckpt", action="store_true")
    parser.add_argument("--enable_flash", action="store_true")
    parser.add_argument("--enable_prefetch", action="store_true")
    parser.add_argument("--enable_slt", action="store_true")
    parser.add_argument("--enable_slt_adasup_raw", action="store_true")
    parser.add_argument("--enable_slt_adasup_raw_2", action="store_true")
    parser.add_argument("--enable_bitnet", action="store_true")
    parser.add_argument("--enable_tensorboard", action="store_true")
    parser.add_argument("--disable_dropout", action="store_true")
    # checkpoint
    parser.add_argument("--save_every", type=int, default=2_000)
    parser.add_argument(
        "--ckpt_type",
        choices=["full", "shard", "local"],
        default="shard",
    )
    parser.add_argument("--resume_dir", type=str, help="要恢复的检查点目录")
    parser.add_argument(
        "--skip_optimizer",
        action="store_true",
        help="只加载权重，优化器重新初始化",
    )
    parser.add_argument("--log_every", type=int, default=1)
    # C4验证集评测
    parser.add_argument(
        "--c4_arrow_dir",
        type=str,
        default="/lpai/volumes/so-volume-bd-ga/anqi/datasets/C4/arrows",
        help="C4 validation 的 Arrow 目录 (save_to_disk 输出目录)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="每多少个 global_step 评一次（0 表示不评）",
    )
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    """Initialise the distributed process group.

    Returns
    -------
    rank : int
        Global rank of the current process.
    world_size : int
        Total number of distributed processes.
    local_rank : int
        Rank of the process within the current node.
    device : torch.device
        CUDA device assigned to the current process.
    """

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if rank == 0:
        node_rank = int(os.environ.get("NODE_RANK", 0))
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "N/A")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        nnodes = max(1, world_size // max(local_world_size, 1))
        print(
            f"Initialized distributed environment | nnodes={nnodes} "
            f"node_rank={node_rank} master={master_addr}:{master_port} "
            f"global_rank={rank} local_rank={local_rank} world_size={world_size}"
        )

    return rank, world_size, local_rank, device


def barrier_safe():
    """Synchronise processes when the distributed backend is initialised."""

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def gmem(tag, device, rank):
    if rank:
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(device) // 2**20
    reserved = torch.cuda.memory_reserved(device) // 2**20
    print(f"[{tag}] alloc {allocated:>6} MiB | rsv {reserved:>6} MiB")


def get_tensorboard_name(args, world_size, rank):
    # 1) 仅 rank 0 生成
    if rank == 0:
        if args.enable_slt:
            if args.enable_bitnet:
                slt_str = ".bitnet_1.58b"
            else:
                slt_config_json = load_json_file(pathlib.Path(args.slt_config))
                init_P_str = slt_config_json.get("init_P_method", "None")
                abs_method_str = slt_config_json.get("abs_method", "None")
                ste_method_str = slt_config_json.get("ste_method", "None")
                ste_coat_num_str = slt_config_json.get("ste_coat_num", "None")
                sparsity_method_str = slt_config_json.get("method", "None")
                sparsity_str = slt_config_json.get("final_sparsity", "None")

                raw = "_raw" if args.enable_slt_adasup_raw else ""
                raw = "_raw_2" if args.enable_slt_adasup_raw_2 else raw
                slt_str = (
                    f".init_P_{init_P_str}.abs_method_{abs_method_str}."
                    f"ste_method_{ste_method_str}{raw}.ste_coat_num_{ste_coat_num_str}."
                    f"sparsity_method_{sparsity_method_str}.sparsity_{sparsity_str}"
                )

        date = time.strftime("%Y_%m_%d-%H_%M_%S")
        model_name = args.config.split("/")[-1].split(".")[0]
        name = (
            f"{date}.{model_name}.{args.data_name}.{'SLT' if args.enable_slt else 'DENSE'}"
            f".mbs_{args.local_batch}.global_{args.global_batch}.gpu_{world_size}"
            f".lr_{args.lr}_{args.lr_mid}.warmup_{args.warmup}."
            f"weight_decay_{args.weight_decay}.bf16_{args.bf16}"
            f"{slt_str if args.enable_slt else ''}"
        )
    else:
        name = None

    # 2) 广播到所有 rank
    names = [name]
    dist.broadcast_object_list(names, src=0)
    return names[0]


# -------------------- CHECKPOINT I/O --------------------
def save_ckpt(
    model,
    opt,
    sched,
    scaler,
    step,
    path: pathlib.Path,
    ckpt_type: str,
    rank: int,
):
    path.mkdir(parents=True, exist_ok=True)

    # 1) 模型权重
    if ckpt_type == "full" and rank == 0:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            torch.save(model.state_dict(), path / "pytorch_model.bin")
    elif ckpt_type == "shard":
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            torch.save(model.state_dict(), path / f"shard_rank{rank:02d}.pt")
    else:
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            torch.save(model.state_dict(), path / f"rank{rank:02d}.pt")

    # 2) 优化器状态（每个 rank 的 shard）
    optim_sd = FSDP.optim_state_dict(model, opt, optim_state_dict=opt.state_dict())
    torch.save(optim_sd, path / f"optim_state_rank{rank:02d}.pt")

    # 3) rank0 保存 scheduler & scaler & meta
    if rank == 0:
        torch.save(
            {
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
            },
            path / "trainer_states.pt",
        )
        (path / "meta.json").write_text(json.dumps({"step": step}))


def load_ckpt(
    model,
    opt,
    sched,
    scaler,
    resume_dir: pathlib.Path,
    ckpt_type: str,
    rank: int,
    skip_opt: bool = False,
) -> int:
    # 1) 模型权重
    if ckpt_type == "full":
        assert rank == 0
        state = torch.load(resume_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(state)
    elif ckpt_type == "shard":
        shard = torch.load(resume_dir / f"shard_rank{rank:02d}.pt", map_location="cpu", weights_only=False)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model.load_state_dict(shard)
    else:
        shard = torch.load(resume_dir / f"rank{rank:02d}.pt", map_location="cpu", weights_only=False)
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            model.load_state_dict(shard)

    # 解析 step
    meta = resume_dir / "meta.json"
    if meta.exists():
        step = json.loads(meta.read_text())["step"]
    else:
        step = int(resume_dir.name.split("_")[-1])

    # 2) 如果不跳过，并且存在，加载 optimizer & scheduler & scaler
    if not skip_opt and (resume_dir / f"optim_state_rank{rank:02d}.pt").exists():
        optim_sd = torch.load(
            resume_dir / f"optim_state_rank{rank:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        optim_sd = FSDP.optim_state_dict_to_load(
            optim_state_dict=optim_sd,
            model=model,
            optim=opt,
        )
        opt.load_state_dict(optim_sd)

        states = torch.load(resume_dir / "trainer_states.pt", map_location="cpu", weights_only=False)
        sched.load_state_dict(states["scheduler"])
        if scaler and states["scaler"]:
            scaler.load_state_dict(states["scaler"])

    # Broadcast 保证所有 rank 用同一个 step
    step_list = [step]
    dist.broadcast_object_list(step_list, src=0)
    return step_list[0]


def get_grad_accumulation_steps(args, world_size: int) -> int:
    """Compute gradient accumulation steps based on global batch size."""

    per_step = args.local_batch * max(world_size, 1)
    if per_step == 0:
        raise ValueError("local_batch must be > 0")
    grad_acc = max(1, args.global_batch // per_step)
    if args.global_batch % per_step != 0 and dist.get_rank() == 0:
        print(
            f"[WARN] global_batch ({args.global_batch}) is not divisible by "
            f"local_batch * world_size ({per_step}); using grad_acc={grad_acc}"
        )
    return grad_acc


# -------------------- MAIN --------------------
def main():
    args = get_args()
    rank, world_size, local_rank, device = init_distributed()

    if not args.enable_flash:
        torch.backends.cuda.enable_flash_sdp(False)

    # ---------------- Tensorboard ----------------
    tensorboard_name = get_tensorboard_name(args, world_size, rank)
    if rank == 0 and args.enable_tensorboard:
        writer = SummaryWriter(log_dir=pathlib.Path(args.tensorboard_dir) / tensorboard_name)
    else:
        writer = None

    # ---------------- DATA (streaming, shard & shuffle) ----------------
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        use_fast=True,
        padding_side="right",
        truncation_side="right",
    )
    tok.pad_token = tok.eos_token

    if args.data_name == "Dolma":
        # Dolma dataloader
        from datas.Dolma import load_data

        loader = load_data.build_train_loader(
            data_dir=args.data_dir,
            batch_size=args.local_batch,
            is_distributed=(world_size > 1),
            rank=rank,
            world_size=world_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            pad_token_id=tok.pad_token_id,
        )
    elif args.data_name == "RedPajama":
        # RedPajama dataloader
        from datas.RedPajama import load_data

        loader = load_data.build_train_loader(
            mds_root=args.data_dir,
            strategy="pack",
            tokenizer_name=args.tokenizer_dir,
            seq_len=2048,
            micro_blocks=args.local_batch,
            per_device_batch=args.local_batch,
            num_workers=16,
            persistent_workers=True,
            local_cache=args.local_cache_dir,
            cache_limit="25GB",
            pin_memory=False,
            prefetch_factor=4,
            predownload=2000,
            pack_batch_texts=2048,
        )
    elif args.data_name == "FineWeb-Edu":
        from datas.SmolLM import load_data

        loader = load_data.build_train_loader(
            mds_root=args.data_dir,
            strategy="pack",
            tokenizer_name=args.tokenizer_dir,
            seq_len=2048,
            micro_blocks=args.local_batch,
            per_device_batch=args.local_batch,
            num_workers=16,
            persistent_workers=True,
            local_cache=args.local_cache_dir,
            cache_limit="25GB",
            pin_memory=False,
            prefetch_factor=4,
            predownload=2000,
            pack_batch_texts=2048,
            world_size=world_size,
            global_rank=rank,
        )
    else:
        raise ValueError(f"Unsupported data source: {args.data_name}")

    # ----------- MODEL -----------
    if args.disable_dropout:
        cfg = LlamaConfig.from_pretrained(args.config, attention_dropout=0.0, hidden_dropout=0.0)
    else:
        cfg = LlamaConfig.from_pretrained(args.config)
    base = LlamaForCausalLM(cfg)
    base.resize_token_embeddings(len(tok))
    # 替换所有的nn.Linear层为SLTLinear层
    if args.enable_slt:
        if args.enable_slt_adasup_raw:
            patch_llama_with_slt(base, slt_config=pathlib.Path(args.slt_config))
        elif args.enable_slt_adasup_raw_2:
            patch_llama_with_ada(base)
        elif args.enable_bitnet:
            patch_llama_with_bitnet(base)
        else:
            replace_linear_with_slt(base, slt_config=pathlib.Path(args.slt_config))

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    mp = MixedPrecision(dtype, dtype, torch.float32)
    wrap = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)

    model = FSDP(
        base.to(device),
        auto_wrap_policy=wrap,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        forward_prefetch=args.enable_prefetch,
        # 避免再聚合到单个 flat 参数
        use_orig_params=True,
    )

    # 应用FSDP友好的梯度检查点
    if args.enable_ckpt:
        # 使用非重入式检查点
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        # 仅对Transformer层应用检查点
        check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn,
        )

    # ----------- OPTIMIZER / SCHEDULER -----------
    grad_acc = get_grad_accumulation_steps(args, world_size)
    if args.enable_slt_adasup_raw_2:

        def make_param_groups(model, wd_main: float = 0.1):
            no_wd, wd = [], []
            for _, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if getattr(param, "_no_weight_decay", False):
                    no_wd.append(param)
                else:
                    wd.append(param)
            return [
                {"params": no_wd, "weight_decay": 0.0},
                {"params": wd, "weight_decay": wd_main},
            ]

        param_groups = make_param_groups(model, wd_main=args.weight_decay)
        opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.95, 0.99), eps=1e-8)
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    if args.enable_slt and args.lr_mid > 0:
        # 两步线形学习率
        # ref: The Era of 1-bit LLMs: Training Tips, Code and FAQ (Figure 1(c) )
        # 配置参数，注意，下面的都是global_step，因为sched是在一个global_step后才更新。
        warmup_steps = args.warmup
        total_steps = args.train_steps // grad_acc
        first_stage_steps = total_steps // 2
        # 1) warm-up：0 -> lr
        warmup_scheduler = LinearLR(
            optimizer=opt,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # 2) 第一阶段线性衰减：lr -> 0.5 * lr
        decay1_scheduler = LinearLR(
            optimizer=opt,
            start_factor=1.0,
            end_factor=0.5,
            total_iters=max(first_stage_steps - warmup_steps, 1),
        )

        # 3) 第二阶段线性衰减：lr_mid -> 0
        decay2_scheduler = LinearLR(
            optimizer=opt,
            start_factor=min(0.5, args.lr_mid / args.lr),
            end_factor=0.0,
            total_iters=max(total_steps - first_stage_steps, 1),
        )

        # 4) 串联在一起
        sched = SequentialLR(
            optimizer=opt,
            schedulers=[warmup_scheduler, decay1_scheduler, decay2_scheduler],
            milestones=[warmup_steps, first_stage_steps],
        )
    else:
        sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.train_steps)
    scaler_cls = (
        torch.cuda.amp.GradScaler
        if hasattr(torch.cuda.amp, "GradScaler")
        else torch.amp.GradScaler
    )
    scaler = scaler_cls(enabled=not args.bf16)

    # ----------- RESUME -----------
    start_step = 1
    if args.resume_dir:
        start_step = (
            load_ckpt(
                model,
                opt,
                sched,
                scaler,
                pathlib.Path(args.resume_dir),
                args.ckpt_type,
                rank,
                args.skip_optimizer,
            )
            + 1
        )
        if rank == 0:
            print(f"✔ 继续训练：从 step {start_step} 开始")

    # ----------- TRAIN LOOP -----------
    # 记录global loss
    if rank == 0:
        accum_raw_loss = 0.0

    model.train()
    tic = time.time()
    for step, batch in enumerate(loader, start_step):
        if step > args.train_steps:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.autocast("cuda", dtype=dtype):
            raw_loss = model(**batch, use_cache=False).loss
            scaled_loss = raw_loss / grad_acc

        # 记录原始loss
        if rank == 0:
            accum_raw_loss += raw_loss.item()

        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (step % grad_acc) == 0:
            global_step = step // grad_acc
            cur_sparsity = 0  # 仅限slt使用，记录当前的稀疏率
            if rank == 0:
                # 计算global_loss
                global_loss = accum_raw_loss / grad_acc
                accum_raw_loss = 0.0

            # SLT训练更新策略
            if args.enable_slt:
                # 可变稀疏率
                for module in model.modules():
                    if isinstance(module, SLTLinear):
                        module.update_sparsity(global_step)
                        if cur_sparsity == 0:
                            cur_sparsity = 1 - module.density
                        # tmp set up tau
                        module.update_tau_tmp(global_step)
                # 两步weight decay
                # ref: The Era of 1-bit LLMs: Training Tips, Code and FAQ (Figure 1(d) )
                if global_step == (args.train_steps // grad_acc // 2) + 1:
                    for pg in opt.param_groups:
                        pg["weight_decay"] = 0.0

                if args.enable_slt_adasup_raw_2:
                    FSDP.clip_grad_norm_(model, max_norm=1.0)

            # 优化器更新
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            # —— 周期性评测 C4 PPL ——
            if args.eval_every > 0 and (global_step % args.eval_every == 0):
                from datas.C4.c4_eval import evaluate_c4_ppl  # 延迟导入避免无用依赖

                ppl, ntok = evaluate_c4_ppl(
                    model,
                    tok,
                    args.c4_arrow_dir,
                    device=device,
                    batch_size=32,
                    num_workers=4,
                    enable_tqdm=True,  # 开启进度条（rank0）
                    use_autocast=True,
                    amp_dtype=(torch.bfloat16 if args.bf16 else torch.float16),
                    sampler_drop_last=True,  # ← 评测端适配 world size
                    dataloader_drop_last=False,
                    filter_dummy=True,  # ← 有 is_dummy 列就过滤掉
                    max_windows_total=102400,
                    eval_seed=1234,
                    shuffle_before_select=True,
                )
                if rank == 0:
                    print(
                        f"[eval @ global_step={global_step:,}] "
                        f"C4 validation PPL={ppl:.2f} on {ntok} tokens (global)"
                    )
                    if args.enable_tensorboard and writer is not None:
                        writer.add_scalar("eval/c4_ppl", ppl, global_step)
                        writer.add_scalar("eval/c4_tokens", ntok, global_step)

            # log，这回只报告global_step
            if rank == 0 and global_step % args.log_every == 0:
                lr = opt.param_groups[0]["lr"]
                wd = opt.param_groups[0]["weight_decay"]
                dt = time.time() - tic
                tic = time.time()
                toks = (
                    args.local_batch
                    * world_size
                    * args.log_every
                    * cfg.max_position_embeddings
                    * grad_acc
                )
                sparsity_str = f"sparsity={cur_sparsity:.4f}" if args.enable_slt else ""
                print(
                    f"[global_step={global_step:,}] loss={global_loss:.4f} | "
                    f"{toks / dt / 1e6:.2f} M tok/s | elapse={dt / args.log_every:.4f} s | "
                    f"lr={lr:.2e} | weight_decay={wd}  | {sparsity_str}"
                )
                if global_step // args.log_every % 10 == 0:
                    gmem(f"global_step={global_step}", device, rank)

                # write tensorboard
                if args.enable_tensorboard:
                    writer.add_scalar("train/loss", global_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/weight_decay", wd, global_step)
                    writer.add_scalar("train/tokens_per_second", toks / dt, global_step)
                    writer.add_scalar("train/time_per_step", dt / args.log_every, global_step)
                    # 记录稀疏率
                    if args.enable_slt:
                        writer.add_scalar("train/sparsity", cur_sparsity, global_step)

        # ckpt
        if args.save_every and step % args.save_every == 0:
            save_ckpt(
                model,
                opt,
                sched,
                scaler,
                step,
                pathlib.Path(args.output_dir) / tensorboard_name / f"step_{step:06d}",
                args.ckpt_type,
                rank,
            )
            if rank == 0:
                print(f"[CKPT] 已保存 step_{step:06d}")

    # ----------- FINAL -----------
    save_ckpt(
        model,
        opt,
        sched,
        scaler,
        step,
        pathlib.Path(args.output_dir) / tensorboard_name / "final_full",
        args.ckpt_type,
        rank,
    )
    if rank == 0:
        tok.save_pretrained(pathlib.Path(args.output_dir) / tensorboard_name)
        if args.enable_tensorboard:
            writer.close()
        print("训练完成")

    barrier_safe()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
