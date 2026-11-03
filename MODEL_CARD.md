---
license: mit
pipeline_tag: text-generation
tags:
- physics-informed
- quantum-inspired
- lindbladian
- port-hamiltonian
- born-rule
- non-hermitian
- reversible-network
- blackwell-sm120
- triton
model_name: PULSE-1.2B
---

# PULSE-1.2B: Port-Hamiltonian Unitary-Lindbladian Sequence Engine

<div align="center">

**Language Modeling as Open Quantum Measurement: Scalable Generative Pretraining without Softmax**

[![GitHub](https://img.shields.io/badge/GitHub-DuoNeural%2FPULSE-blue?logo=github)](https://github.com/DuoNeural/PULSE)
[![Preprint](https://img.shields.io/badge/Preprint-DuoNeural%20Preprint%20v0.1-red)](https://github.com/DuoNeural/PULSE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Developed by the DuoNeural AI Laboratory — Jesse Caldwell, Aura✨, Archon*

</div>

---

## 🌌 Model Overview

**PULSE-1.2B** (**P**ort-Hamiltonian **U**nitary-**L**indbladian **S**equence **E**ngine) is a **1.121-billion parameter** foundation autoregressive language model whose sequential recurrent backbone is an open quantum dynamical system and whose token prediction head is an exact quantum measurement under the Born rule.

PULSE solves the three historical roadblocks of Quantum-inspired Language Modeling (QiLM):
1. **Bypassing the Density Matrix Wall:** Pure state vectors $|\psi_t\rangle \in \mathbb{C}^d$ replace mixed-state density matrices $\rho \in \mathbb{C}^{d \times d}$, reducing computational complexity from $\mathcal{O}(d^4)$ to strictly linear $\mathcal{O}(d)$ per channel.
2. **First Logarithmic Liouvillian Scan:** Discrete pure-state transitions form an associative complex affine semigroup $(\mathbb{C} \times \mathbb{C}, \circ)$, executing parallel Blelloch associative scans in $\mathcal{O}(\log T)$ time via custom GPU Triton kernels.
3. **Softmax-Free Born Rule POVM Head:** Replaces standard heuristic softmax with a Positive Operator-Valued Measure (POVM) in projective Hilbert space $\mathbb{P}(\mathcal{H})$:
   $$P(w_k \mid \psi) = \frac{|\langle e_k \mid \psi \rangle|^2}{\sum_{j=1}^V |\langle e_j \mid \psi \rangle|^2}$$
   yielding exact scale-invariance, analytical Wirtinger adjoint backprop, and zero softmax temperature tuning!

---

## 📊 Silicon Performance on Blackwell (NVIDIA RTX 5070 Ti 16 GB)

PULSE-1.2B runs at full generative scale on a single consumer Blackwell GPU (`sm_120`, `GB203`):

- **Active Model Parameters:** **1,120,594,496 (1.121B)**
- **Architecture Depth:** 24 Reversible Layers / 8 Complex Heads / $D = 1024$
- **Vocabulary Size:** 49,152 (`cosmo2-tokenizer`)
- **Throughput:** **2,567.8 tokens/second** ($51.04\text{s}$ per optimizer step at $T = 2048$, $B = 2$, $GA = 32$)
- **Peak VRAM Allocated:** **9,866.9 MB / 15,838.8 MB (62.3% of the card)**
- **Clean Safety Headroom:** **6,436.0 MB (6.44 GB completely untouched)**
- **Reversible Inversion Drift:** Max absolute drift $< 1.19 \times 10^{-7}$ (Machine Epsilon)
- **48.0 MB L2 Cache Tiling:** Intermediate Born buffers ($C=128$) require only $25.16\text{ MB}$, eliminating GDDR7 spilling.

---

## 📈 Pretraining Convergence Telemetry

| Step | Cumulative Tokens | Loss (Cross-Entropy) | Grad Norm | Speed |
|---|---|---|---|---|
| **0001** | 0.13M | 11.3819 | 10.3125 | 2,351.8 tok/s |
| **0050** | 6.55M | 8.3493 | 0.1934 | 2,567.8 tok/s |
| **0100** | 13.11M | 8.3217 | 0.0635 | 2,567.5 tok/s |
| **0150** | 19.66M | 8.3233 | 0.5586 | 2,567.8 tok/s |
| **0200** | 26.21M | 8.3185 | 0.2324 | 2,567.8 tok/s |
| **0250** | 32.77M | 8.3254 | 0.1177 | 2,567.9 tok/s |
| **0300** | 39.32M | 8.3217 | 0.0938 | 2,567.8 tok/s |
| **0313** | 41.03M | **8.3141** | **0.0479** | 2,567.8 tok/s |

---

## 🚀 Code & Checkpoint Usage

The complete training harness, model definition, and custom Triton associative scan kernels are open-sourced on GitHub:
👉 **[https://github.com/DuoNeural/PULSE](https://github.com/DuoNeural/PULSE)**

```bash
git clone https://github.com/DuoNeural/PULSE.git
cd PULSE
pip install -r requirements.txt
python test_convergence.py
```

Checkpoints from the ongoing pretraining campaign (`pulse_step0050.pt`, `pulse_step0100.pt`, `pulse_step0200.pt`, `pulse_step0300.pt`) will be uploaded to this repository upon completion of the 131M-token foundational phase.

---

## 📑 Citation

```bibtex
@article{duoneural2026pulse,
  title={Language Modeling as Open Quantum Measurement: Scalable Generative Pretraining without Softmax},
  author={Caldwell, Jesse and Aura and Archon},
  journal={DuoNeural AI Laboratory Preprint},
  year={2026}
}
```
