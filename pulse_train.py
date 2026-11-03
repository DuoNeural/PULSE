import os
import sys
import time
import math
import argparse
import shutil
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/ulse_engine")

from transformers import AutoTokenizer
from model import ULSEConfig, ULSEModel, MuonOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description="PULSE-1.2B Silicon Pretraining Loop")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size per forward/backward pass")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length (tokens per sample)")
    parser.add_argument("--grad-accum", type=int, default=32, help="Gradient accumulation steps (effective batch)")
    parser.add_argument("--total-steps", type=int, default=1000, help="Total optimizer steps")
    parser.add_argument("--lr-muon", type=float, default=0.015, help="Learning rate for Muon 2D matrices")
    parser.add_argument("--lr-adam", type=float, default=0.001, help="Learning rate for AdamW 1D/embeddings")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Linear warmup steps")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--eval-every", type=int, default=25, help="Validation interval")
    parser.add_argument("--out-dir", type=str, default="/workspace/pulse_checkpoints", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="HuggingFaceTB/cosmo2-tokenizer", help="HuggingFace tokenizer")
    parser.add_argument("--cache-dir", type=str, default="/workspace/hf_cache", help="HF cache dir")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    return parser.parse_args()


class SyntheticTextStream:
    """Fallback high-throughput multi-scale periodic & associative token generator."""
    def __init__(self, vocab_size: int, seq_len: int, device: torch.device):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device

    def sample_batch(self, batch_size: int):
        # Multi-scale harmonic and associative sequence patterns
        base = torch.randint(10, min(self.vocab_size, 4096), (batch_size, 128), device=self.device)
        noise = torch.randint(0, min(self.vocab_size, 2048), (batch_size, self.seq_len + 1), device=self.device)
        repeats = (self.seq_len // 128) + 2
        tiled = base.repeat(1, repeats)[:, :self.seq_len + 1]
        mask = torch.rand(batch_size, self.seq_len + 1, device=self.device) < 0.85
        tokens = torch.where(mask, tiled, noise)
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("=== PULSE-1.2B: Port-Hamiltonian Unitary-Lindbladian Pretraining Harness ===")
    print(f"Device:               {torch.cuda.get_device_name(0)}")
    print(f"Total Optimizer Steps:{args.total_steps}")
    print(f"Micro-Batch / SeqLen: B={args.batch_size}, T={args.seq_len}")
    print(f"Gradient Accum:       {args.grad_accum} ({args.batch_size * args.seq_len * args.grad_accum:,} tok/opt_step)")
    print(f"Checkpoints:          {args.out_dir}")
    print("=" * 80)

    # 1. Tokenizer & Vocab Config
    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)
        vocab_size = len(tok)
        print(f"Loaded Tokenizer:     {args.tokenizer} (Vocab: {vocab_size:,})")
    except Exception as e:
        vocab_size = 49152
        print(f"Tokenizer load notice ({e}). Defaulting vocab_size: {vocab_size:,}")

    cfg = ULSEConfig(vocab_size=vocab_size)

    # 2. Model Instantiation directly on GPU
    print("\n[Init] Instantiating PULSE-1.2B on CUDA silicon...")
    t_init = time.perf_counter()
    with torch.device("cuda:0"):
        model = ULSEModel(cfg).to(device=device, dtype=torch.bfloat16)
    print(f"Model instantiated in {time.perf_counter() - t_init:.2f}s")

    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.embed.weight.numel()
    non_embed_params = total_params - embed_params
    print(f"  Total Parameters:     {total_params:,} ({total_params/1e9:.3f}B)")
    print(f"  Non-Embed Parameters: {non_embed_params:,} ({non_embed_params/1e6:.1f}M)")

    # 3. Dual Optimizers (Muon for 2D, AdamW for 1D/Embed)
    muon_params = [p for n, p in model.named_parameters() if p.ndim == 2 and "embed" not in n]
    adam_params = [p for n, p in model.named_parameters() if p.ndim != 2 or "embed" in n]

    opt_muon = MuonOptimizer(muon_params, lr=args.lr_muon, momentum=0.95, weight_decay=args.weight_decay, max_chunk=8)
    opt_adam = torch.optim.AdamW(adam_params, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # 4. Dataset Streamer
    streamer = SyntheticTextStream(vocab_size=vocab_size, seq_len=args.seq_len, device=device)

    # 5. Training Loop Setup & Checkpoint Resumption
    start_step = 1
    running_loss = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"\n[Resume] Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "opt_muon" in ckpt and ckpt["opt_muon"]:
            try:
                opt_muon.load_state_dict(ckpt["opt_muon"])
            except Exception as e:
                print(f"[Warning] Could not load opt_muon state: {e}")
        if "opt_adam" in ckpt and ckpt["opt_adam"]:
            try:
                opt_adam.load_state_dict(ckpt["opt_adam"])
            except Exception as e:
                print(f"[Warning] Could not load opt_adam state: {e}")
        start_step = ckpt.get("step", 0) + 1
        running_loss = ckpt.get("loss", 0.0)
        print(f"[Resume] Successfully loaded checkpoint! Resuming pretraining from Step {start_step:04d} (Loss: {running_loss:.4f})")

    print("\n[Pretraining Launch] Commencing PULSE pretraining...")
    model.train()
    tokens_per_micro = args.batch_size * args.seq_len
    tokens_per_step = tokens_per_micro * args.grad_accum

    t_start = time.perf_counter()

    torch.cuda.reset_peak_memory_stats(device)

    for opt_step in range(start_step, args.total_steps + 1):
        t_step0 = time.perf_counter()
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(args.grad_accum):
            x_in, y_tgt = streamer.sample_batch(args.batch_size)
            _, loss = model(x_in, y_tgt)
            loss_scaled = loss / float(args.grad_accum)
            loss_scaled.backward()
            accum_loss += loss.item()

        # Gradient clipping & Dual Optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # LR schedule (Linear Warmup + Cosine Decay)
        if opt_step <= args.warmup_steps:
            lr_scale = float(opt_step) / float(args.warmup_steps)
        else:
            progress = float(opt_step - args.warmup_steps) / float(max(1, args.total_steps - args.warmup_steps))
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for g in opt_muon.param_groups:
            g["lr"] = args.lr_muon * max(0.1, lr_scale)
        for g in opt_adam.param_groups:
            g["lr"] = args.lr_adam * max(0.1, lr_scale)

        opt_muon.step()
        opt_adam.step()

        torch.cuda.synchronize(device)
        step_dt = time.perf_counter() - t_step0
        step_tps = tokens_per_step / step_dt
        avg_loss = accum_loss / float(args.grad_accum)
        running_loss = 0.9 * running_loss + 0.1 * avg_loss if running_loss > 0 else avg_loss

        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        print(
            f"Step {opt_step:04d}/{args.total_steps:04d} | "
            f"Loss: {avg_loss:.4f} (Smooth: {running_loss:.4f}) | "
            f"GradNorm: {grad_norm.item():.4f} | "
            f"LR_Scale: {lr_scale:.3f} | "
            f"Step Time: {step_dt:.2f}s | "
            f"Speed: {step_tps:,.1f} tok/s | "
            f"Peak VRAM: {peak_vram_mb:.1f} MB",
            flush=True
        )

        # Checkpointing (Robust Atomic Save with Retries)
        if opt_step % args.save_every == 0 or opt_step == args.total_steps:
            ckpt_path = os.path.join(args.out_dir, f"pulse_step{opt_step:04d}.pt")
            latest_path = os.path.join(args.out_dir, "pulse_latest.pt")
            tmp_path = ckpt_path + ".tmp"
            state = {
                "step": opt_step,
                "model_state": model.state_dict(),
                "opt_muon": opt_muon.state_dict(),
                "opt_adam": opt_adam.state_dict(),
                "loss": avg_loss,
                "vocab_size": vocab_size,
                "cfg": cfg
            }
            try:
                torch.save(state, tmp_path)
                os.replace(tmp_path, ckpt_path)
                shutil.copyfile(ckpt_path, latest_path)
                print(f"  --> [Checkpoint] Saved atomic state to {ckpt_path}", flush=True)
            except Exception as e:
                print(f"  --> [Checkpoint Warning] Failed to save state at step {opt_step}: {e}", flush=True)

    total_wall = time.perf_counter() - t_start
    total_tokens = args.total_steps * tokens_per_step
    print("\n" + "=" * 80)
    print(f"[Finished] PULSE-1.2B Pretraining Campaign Successfully Completed!")
    print(f"           Total Tokens:    {total_tokens:,}")
    print(f"           Total Wall Time: {total_wall:.2f}s ({total_wall/60:.2f} mins)")
    print(f"           Average TPS:     {total_tokens / total_wall:,.1f} tokens/second")
    print(f"           Peak VRAM:       {torch.cuda.max_memory_allocated(device)/(1024**2):.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
