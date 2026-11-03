"""
Unitary-Lindbladian Sequence Engine (ULSE) - Core PyTorch Model (Phase 4)
Hardware Architecture: NVIDIA GeForce RTX 5070 Ti (SM 12.0 Blackwell)
Sub-Layer Formulations:
  Channel F: Coherent Unitary-Lindbladian State-Space Evolution (Zero-Copy Strided Triton Associative Scan)
  Channel G: Thermodynamic Squared-ReLU Gated Mixing (Gibbs-Boltzmann Channel with Fused Gate)
Carrier Streams: Bit-Exact FP32 Boundary Reversible Trunk with BF16 Sub-Layers
Readout: Coalesced Real-Stacked BF16 Tensor-Core Born POVM Analytical Autograd
Optimizer: Dual Muon (Chunked 3D Tensor Core bmm) + AdamW
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_lindblad_scan import triton_lindblad_scan


@dataclass
class ULSEConfig:
    vocab_size: int = 65536
    d_model: int = 2048
    d_rev: int = 1024  # d_model // 2
    n_blocks: int = 24  # Reversible blocks
    n_heads: int = 24
    d_head: int = 128
    d_ffn: int = 11264  # SwiGLU / sqReLU dimension
    born_eps: float = 1e-6
    chunk_size: int = 64
    max_seq_len: int = 4096

    @property
    def d_state(self) -> int:
        return self.n_heads * self.d_head


def fused_sqrelu_gate(up_val: torch.Tensor, gate_val: torch.Tensor) -> torch.Tensor:
    return F.relu(up_val).square() * gate_val


def fused_debiased_energy_norm(
    sr: torch.Tensor,
    si: torch.Tensor,
    cr: torch.Tensor,
    ci: torch.Tensor,
    tau0_sq: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    obs_r = sr * cr - si * ci
    obs_i = sr * ci + si * cr
    energy = (obs_r.square() + obs_i.square()).view(sr.size(0), sr.size(1), -1)
    mean = energy.mean(dim=-1, keepdim=True)
    var = energy.var(dim=-1, unbiased=False, keepdim=True)
    return ((energy - mean) * torch.rsqrt(var + tau0_sq)).to(out_dtype)


class UnitaryLindbladChannel(nn.Module):
    """Channel F: Coherent Unitary-Lindbladian State-Space Evolution."""
    def __init__(self, cfg: ULSEConfig):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.n_heads
        self.dh_half = cfg.d_head // 2
        d_rev = cfg.d_rev

        self.norm = nn.RMSNorm(d_rev)
        # Horizontal GEMM Fusion: Combine proj_theta, proj_gamma, proj_B into a single coalesced projection
        # Cuts 72 GEMM launches per pass (216 across Fwd + Recomp + Bwd) to 24 coalesced launches
        self.d_theta = self.H * self.dh_half
        self.d_gamma = self.H * self.dh_half
        self.d_B = cfg.d_state
        self.proj_all = nn.Linear(d_rev, self.d_theta + self.d_gamma + self.d_B, bias=False)
        self.proj_dt = nn.Linear(d_rev, self.H, bias=True)
        
        self.register_parameter(
            "proj_C_r",
            nn.Parameter(torch.randn(self.H, self.dh_half, dtype=torch.float32) / math.sqrt(self.dh_half))
        )
        self.register_parameter(
            "proj_C_i",
            nn.Parameter(torch.randn(self.H, self.dh_half, dtype=torch.float32) / math.sqrt(self.dh_half))
        )
        self.proj_out = nn.Linear(self.H * self.dh_half, d_rev, bias=False)
        self.tau0_sq = 1e-4

        self._init_weights()

    def _init_weights(self):
        # Initialize sections of proj_all weight
        nn.init.normal_(self.proj_all.weight[:self.d_theta], mean=0.0, std=0.01)
        nn.init.orthogonal_(self.proj_all.weight[self.d_theta : self.d_theta + self.d_gamma])
        nn.init.orthogonal_(self.proj_all.weight[self.d_theta + self.d_gamma:])
        nn.init.zeros_(self.proj_dt.bias)
        nn.init.orthogonal_(self.proj_out.weight)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, T, _ = u.shape
        u_norm = self.norm(u)

        # Single coalesced GEMM replaces 3 separate projections
        all_proj = self.proj_all(u_norm)
        theta_raw, gamma_raw, B_raw = torch.split(
            all_proj, [self.d_theta, self.d_gamma, self.d_B], dim=-1
        )

        theta = theta_raw.float().view(B, T, self.H, self.dh_half)
        gamma = F.softplus(gamma_raw.float()).view(B, T, self.H, self.dh_half) + 1e-4
        dt = F.softplus(self.proj_dt(u_norm).float()).view(B, T, self.H, 1) + 1e-3

        phi = 2.0 * torch.atan(0.5 * dt * theta)
        log_decay = -dt * gamma

        b_real_imag = B_raw.float().view(B, T, self.H, self.dh_half, 2)
        b_real = b_real_imag[..., 0].contiguous()
        b_imag = b_real_imag[..., 1].contiguous()

        # Zero-Copy Native Strided Triton Scan
        sr, si = triton_lindblad_scan(log_decay, phi, b_real, b_imag)

        # Fused debiased spherical energy norm
        Cr = self.proj_C_r.unsqueeze(0).unsqueeze(0)
        Ci = self.proj_C_i.unsqueeze(0).unsqueeze(0)
        energy_normed = fused_debiased_energy_norm(sr, si, Cr, Ci, self.tau0_sq, u.dtype)
        return self.proj_out(energy_normed)


class GibbsBoltzmannChannel(nn.Module):
    """Channel G: Thermodynamic Squared-ReLU Gated Mixing with Fused Projections."""
    def __init__(self, cfg: ULSEConfig):
        super().__init__()
        d_rev = cfg.d_rev
        self.norm = nn.RMSNorm(d_rev)
        # Fuse w_up and w_gate into a single linear projection for 2x GEMM launch reduction
        self.w_up_gate = nn.Linear(d_rev, 2 * cfg.d_ffn, bias=False)
        self.w_down = nn.Linear(cfg.d_ffn, d_rev, bias=False)
        self.d_ffn = cfg.d_ffn

        nn.init.orthogonal_(self.w_up_gate.weight)
        nn.init.orthogonal_(self.w_down.weight)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y_norm = self.norm(y)
        up_gate = self.w_up_gate(y_norm)
        up, gate = torch.split(up_gate, self.d_ffn, dim=-1)
        hidden = fused_sqrelu_gate(up, gate)
        return self.w_down(hidden)


class ReversibleULSEFunction(torch.autograd.Function):
    """
    Exact algebraic reversible residual execution with bit-exact FP32 accumulation.
    """
    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor, f_block: nn.Module, g_block: nn.Module):
        ctx.f_block = f_block
        ctx.g_block = g_block
        with torch.no_grad():
            f_val = f_block(x2.bfloat16()).float()
            y1 = x1 + f_val
            g_val = g_block(y1.bfloat16()).float()
            y2 = x2 + g_val
        ctx.save_for_backward(y1.detach(), y2.detach())
        return y1, y2

    @staticmethod
    def backward(ctx, dy1: torch.Tensor, dy2: torch.Tensor):
        y1, y2 = ctx.saved_tensors
        f_block = ctx.f_block
        g_block = ctx.g_block

        # 1. Reconstruct x2 = y2 - G(y1) exactly in FP32
        with torch.enable_grad():
            y1_bf16 = y1.detach().bfloat16().requires_grad_(True)
            g_val = g_block(y1_bf16).float()
            torch.autograd.backward(g_val, dy2)

        with torch.no_grad():
            x2_rec = y2 - g_val
            dy1_total = dy1 + y1_bf16.grad.float()

        # 2. Reconstruct x1 = y1 - F(x2) exactly in FP32
        with torch.enable_grad():
            x2_bf16 = x2_rec.detach().bfloat16().requires_grad_(True)
            f_val = f_block(x2_bf16).float()
            torch.autograd.backward(f_val, dy1_total)

        with torch.no_grad():
            dx2_total = dy2 + x2_bf16.grad.float()
            dx1_total = dy1_total

        return dx1_total, dx2_total, None, None


class ReversibleULSEBlock(nn.Module):
    def __init__(self, cfg: ULSEConfig):
        super().__init__()
        self.F = UnitaryLindbladChannel(cfg)
        self.G = GibbsBoltzmannChannel(cfg)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return ReversibleULSEFunction.apply(x1, x2, self.F, self.G)


class FusedBornPOVMLossFn(torch.autograd.Function):
    """
    Coalesced Real-Stacked Analytical Autograd for Born Rule POVM Cross-Entropy:
    Uses [2*c, D] @ [D, V] Tensor Core GEMMs with zero intermediate graph retention.
    """
    @staticmethod
    def forward(ctx, psi: torch.Tensor, embed_weight: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
        B, T, D = psi.shape
        N = B * T
        V = embed_weight.size(0)
        half_D = D // 2

        psi_2d = psi.detach().view(N, 2, half_D)
        psi_r = psi_2d[:, 0, :].bfloat16()
        psi_i = psi_2d[:, 1, :].bfloat16()

        p_cat = torch.cat([psi_r, psi_i], dim=-1)
        p_rot = torch.cat([psi_i, -psi_r], dim=-1)

        E_norm = F.normalize(embed_weight.detach().float(), p=2, dim=-1).bfloat16()

        d_p_cat = torch.zeros_like(p_cat, dtype=torch.float32)
        d_p_rot = torch.zeros_like(p_rot, dtype=torch.float32)
        d_E = torch.zeros_like(E_norm, dtype=torch.float32)

        targets_flat = targets.view(-1)
        total_nll = torch.zeros((), device=psi.device, dtype=torch.float32)
        # 48.0 MB L2 Cache Residency Sweet Spot:
        # chunk_size=128 produces [256, 65536] BF16 tensor = 33.55 MB, fitting 100% inside the 48 MB L2 cache!
        chunk_size = 128

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            c_len = end - start
            sub_p_cat = p_cat[start:end]
            sub_p_rot = p_rot[start:end]
            tgt = targets_flat[start:end]

            p_pair = torch.cat([sub_p_cat, sub_p_rot], dim=0)
            re_im = (p_pair @ E_norm.t()).float()
            re, im = re_im[:c_len], re_im[c_len:]

            mod_sq = re.square() + im.square() + eps
            Z = mod_sq.sum(dim=-1, keepdim=True)
            tgt_mod = mod_sq.gather(dim=-1, index=tgt.unsqueeze(-1))

            total_nll += -(torch.log(tgt_mod) - torch.log(Z)).sum()

            grad_mod = torch.full_like(mod_sq, 1.0 / float(N)) / Z
            grad_tgt = -1.0 / (float(N) * tgt_mod)
            grad_mod.scatter_add_(dim=-1, index=tgt.unsqueeze(-1), src=grad_tgt)

            g_re = (2.0 * grad_mod * re).bfloat16()
            g_im = (2.0 * grad_mod * im).bfloat16()

            g_pair = torch.cat([g_re, g_im], dim=0)
            d_E.add_((g_pair.t() @ p_pair).float())

            d_p_cat[start:end] = (g_re @ E_norm).float()
            d_p_rot[start:end] = (g_im @ E_norm).float()

        d_psi_r = d_p_cat[:, :half_D] - d_p_rot[:, half_D:]
        d_psi_i = d_p_cat[:, half_D:] + d_p_rot[:, :half_D]

        E_raw_f32 = embed_weight.detach().float()
        E_norms = E_raw_f32.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        E_norm_f32 = E_norm.float()
        dot = (d_E * E_norm_f32).sum(dim=-1, keepdim=True)
        d_embed = ((d_E - E_norm_f32 * dot) / E_norms).to(embed_weight.dtype)

        d_psi = torch.stack([d_psi_r, d_psi_i], dim=1).view(B, T, D).to(psi.dtype)
        ctx.save_for_backward(d_psi, d_embed)
        return total_nll / float(N)

    @staticmethod
    def backward(ctx, grad_output):
        d_psi, d_embed = ctx.saved_tensors
        scale = grad_output.to(d_psi.dtype)
        return d_psi * scale, d_embed * grad_output.to(d_embed.dtype), None, None


class ULSEModel(nn.Module):
    def __init__(self, cfg: ULSEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(cfg.d_model))

        self.blocks = nn.ModuleList([ReversibleULSEBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = nn.RMSNorm(cfg.d_model)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed(input_ids)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x1, x2 = x1.float(), x2.float()

        for block in self.blocks:
            x1, x2 = block(x1, x2)

        psi_real = self.final_norm(torch.cat([x1, x2], dim=-1).to(self.embed.weight.dtype))
        if targets is None:
            return psi_real, None

        loss = FusedBornPOVMLossFn.apply(psi_real, self.embed.weight, targets, self.cfg.born_eps)
        return psi_real, loss

    # Method alias for reversible training loop compatibility
    forward_reversible = forward


def batched_newtonschulz5(G_batch: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G_batch.bfloat16()
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    norms = X.flatten(1).norm(p=2, dim=1).view(-1, 1, 1) + eps
    X = X / norms
    for _ in range(steps):
        A = torch.bmm(X, X.transpose(1, 2))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)
    return X.transpose(1, 2) if transposed else X


class MuonOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, max_chunk: int = 8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay, max_chunk=max_chunk)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            max_chunk = group.get("max_chunk", 8)

            shape_groups = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                shape_groups.setdefault(p.shape, []).append(p)

            for shape, p_list in shape_groups.items():
                # Process in mini-chunks of max_chunk=8 to prevent 1.03 GB scratchpad spike
                for c_start in range(0, len(p_list), max_chunk):
                    p_chunk = p_list[c_start : c_start + max_chunk]
                    grad_updates = []
                    for p in p_chunk:
                        g = p.grad
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            buf = state["momentum_buffer"] = torch.zeros_like(p.data)
                        else:
                            buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(g)
                        gu = g.add(buf, alpha=momentum) if nesterov else buf
                        grad_updates.append(gu)

                    G_stack = torch.stack(grad_updates, dim=0)
                    updates_stack = batched_newtonschulz5(G_stack, steps=ns_steps)

                    scale = max(1.0, shape[0] / shape[1]) ** 0.5
                    for i, p in enumerate(p_chunk):
                        if wd > 0:
                            p.data.mul_(1.0 - lr * wd)
                        p.data.add_(updates_stack[i], alpha=-lr * scale)

        return loss
