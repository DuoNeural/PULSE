import time
import torch
import torch.nn.functional as F
from model import ULSEConfig, ULSEModel, MuonOptimizer


def run_convergence_verification():
    device = torch.device("cuda:0")
    cfg = ULSEConfig()
    
    print("=" * 80)
    print("=== ULSE-1.2B Multi-Step Convergence & Quantum State Telemetry (SM120) ===")
    print("=" * 80)
    
    with torch.device("cuda:0"):
        model = ULSEModel(cfg).to(device=device, dtype=torch.bfloat16)

    muon_params = [p for n, p in model.named_parameters() if p.ndim == 2 and "embed" not in n]
    adam_params = [p for n, p in model.named_parameters() if p.ndim != 2 or "embed" in n]

    opt_muon = MuonOptimizer(muon_params, lr=0.015, momentum=0.95, weight_decay=0.01, max_chunk=8)
    opt_adam = torch.optim.AdamW(adam_params, lr=0.001, betas=(0.9, 0.95), weight_decay=0.01)

    # Structured multi-scale periodic + associative pattern batch (B=2, T=4096 = 8,192 tokens/step)
    torch.manual_seed(42)
    B, T = 2, 4096
    base_pattern = torch.randint(10, 4096, (B, 128), device=device)
    seq = base_pattern.repeat(1, (T // 128) + 1)[:, :T + 1]
    input_ids = seq[:, :-1].contiguous()
    targets = seq[:, 1:].contiguous()

    torch.cuda.reset_peak_memory_stats(device)
    initial_vram = torch.cuda.memory_allocated() / (1024**2)
    print(f"Batch Shape: B={B}, T={T} ({B*T:,} tokens/step) | Initial VRAM: {initial_vram:.1f} MB\n")

    losses = []
    step_times = []

    for step in range(1, 11):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        _, loss = model.forward_reversible(input_ids, targets)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_muon.step()
        opt_adam.step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        losses.append(loss.item())
        step_times.append(dt)

        tps = (B * T) / dt
        print(f"Step {step:02d}/10 | Loss: {loss.item():.4f} | GradNorm: {grad_norm.item():.4f} | Step Time: {dt:.3f}s | TPS: {tps:,.1f} tok/s")

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    print("\n" + "=" * 80)
    print(f"Initial Loss (Step 1): {losses[0]:.4f} ---> Final Loss (Step 10): {losses[-1]:.4f} (Delta: {losses[-1] - losses[0]:.4f})")
    print(f"Mean Steady-State Step Time (Steps 3-10): {sum(step_times[2:])/len(step_times[2:]):.3f}s")
    print(f"Peak Allocated VRAM Across 10 Steps:      {peak_vram:.1f} MB / 15,838.8 MB ({peak_vram/158.388:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_convergence_verification()
