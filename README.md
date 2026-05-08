# DRF-MWU — Dynamic Recursive Field Model with Modulated Weights Update

**A geometrically self-aware neural architecture for abstract reasoning**  
*Derived from Meta-Ontological Psychodynamics (MOPD) theory*  
*Institute for Meta-Sciences · SNA · 2026*

---

## What This Is

DRF-MWU is a novel neural architecture built on a theoretical framework called **Meta-Ontological Psychodynamics (MOPD)**. It is a proof of concept submitted to the ARC-AGI 2026 competition demonstrating a fundamentally different approach to abstract reasoning — one that is geometrically self-aware and terminates based on internal field resolution rather than a fixed computational budget.

The architecture is not a fine-tuned language model. It is a purpose-built recursive field system under 1 million parameters that monitors the shape of its own reasoning at every step.

---

## The Core Idea

Every conventional neural network processes input through a fixed number of layers and produces an output. It has no mechanism to know when it is done — it always runs exactly N steps regardless of whether the answer is ready.

DRF-MWU works differently. Instead of fixed-depth computation, it runs a **mirror loop** between two independent reasoning tracks (y and z) until their geometric relationship signals genuine convergence. The system knows when it is done because the field tells it.

```
Input grid
    ↓
Two independent encoders → y₀ and z₀ (start from different perspectives)
    ↓
Mirror loop:
    y_τ = Block_y(x_encoded, z_{τ-1})   ← y attends to z through the field
    z_τ = Block_z(x_encoded, y_{τ-1})   ← z attends to y through the field
    
    At each step, compute θ(τ) = angular separation between y and z
    
    Terminate when θ < threshold (the field has resolved)
    ↓
Output prediction
```

The key insight: **x_encoded is fixed throughout the loop** — it is the unchanging ground truth both tracks converge toward. y and z start from different readings of the input and converge toward agreement. When they agree, the reasoning is complete.

---

## Why Two Tracks?

In MOPD theory, reasoning requires two bias operators that take different initial perspectives on the same field and converge through mutual conditioning. This is not an ensemble — the tracks are not independent reasoners whose outputs are averaged. They are a single reasoning process expressed as a geometric convergence.

The angular separation θ between y and z at each step is the primary diagnostic of reasoning quality:
- θ high → the two tracks are reading the field very differently → more reasoning needed
- θ low → the tracks have converged on the same reading → reasoning can terminate
- θ below threshold → genuine resolution, output is trusted

---

## The Field Variables — How the System Watches Itself

At every refinement step, the FieldMonitor computes a set of geometric variables that describe the state of the reasoning field:

| Variable | Formula | What It Measures |
|---|---|---|
| **θ(τ)** | `1 − cos(y, z)` | Angular separation between tracks. Primary convergence signal. |
| **A(τ)** | `θ(τ) − θ̄` | Deviation from baseline. Positive = harder than average. |
| **P(τ)** | `P₀·e^(−α·n) + β·Σ\|A\|·e^(−λk)` | Field permeability. Memory of prior shocks. |
| **K(τ)** | `K₀ · exp(A·P)` | Dynamic curvature. Always positive. |
| **γ(τ)** | `K₀ / K(τ)` | Recursion rate. Drives dynamic weight transition. |
| **w_y, w_z** | `w_z = 1/(1+γ)`, `w_y = 1 − w_z` | Track weights. Shift from z-dominant to y-dominant as field resolves. |
| **EWS** | fires when: calm AND P rising AND P > threshold | Early Warning Signal. Detects false consensus. |

These are not diagnostic tools bolted on after the fact. They are the mechanism of self-awareness — the field reads itself at every step and adapts.

---

## The Spatial Encoder — Why Cells, Not Vectors

The most consequential architectural decision is encoding each grid cell as an **independent vector** rather than compressing the entire grid to a single vector.

```python
# Wrong approach — destroys spatial structure
x_encoded = Linear(H * W * 32, d_model)(flatten(grid))  # [d_model]

# Correct approach — preserves spatial structure
cells = color_embed(grid.flatten()) + pos_embed(positions)  # [n_cells, d_model]
cells = self_attention(cells)                                 # cells see each other
x_encoded = cells                                            # [n_cells, d_model]
```

When the field encoding is a single vector, the model learns the most statistically common colors across training puzzles — it achieves ~48% cell accuracy with zero exact matches by predicting background colors. When each cell has its own vector that has attended to all other cells, the model can learn spatial relationships and transformation patterns.

**The diagnostic signature of a single-vector bottleneck:** high cell accuracy, zero exact matches, all cells in the same puzzle getting identical predictions.

---

## Hypothesis Selection — Trying Multiple Readings

The current architecture adds a **HypothesisGenerator** that produces K=3 candidate readings of the field encoding before the mirror loop begins. The mirror loop runs on each candidate independently. The geometrically healthiest result is selected.

```
x_encoded (base field encoding)
    ↓
HypothesisGenerator
    ├── H₀: projector_0(x_encoded)  → mirror loop → score_0
    ├── H₁: projector_1(x_encoded)  → mirror loop → score_1
    └── H₂: projector_2(x_encoded)  → mirror loop → score_2
    ↓
Select candidate with lowest geometric health score
    ↓
Output
```

The geometric health score rewards lower final θ, calmer P, no EWS events, y-track dominance at termination, and smooth convergence path. The hypothesis with the lowest score — the most geometrically resolved reading — becomes the output.

This implements MOPD's attentional commitment principle: before convergence begins, the system tries different ways of attending to the input and commits to the one that the field geometry indicates is most worth pursuing.

---

## Key Empirical Results

All results on ARC-AGI benchmark. Architecture: 881,805 parameters, tau_max=12, K=3 hypotheses.

### Training Strategy Comparison

| Strategy | Training Data | AGI-1 Eval Acc | AGI-1 Eval Exact | Generalization Gap |
|---|---|---|---|---|
| AGI-1 only | 300 puzzles | 54.96% | 2/334 | 24.72pp |
| **AGI-2 full curriculum** | **800 puzzles** | **70.58%** | **8/334** | **8.02pp** |
| AGI-2 then AGI-1 fine-tune | 800 + 300 puzzles | 64.06% | 6/334 | 13.10pp |

Training on the harder AGI-2 distribution produces dramatically better generalization to AGI-1 held-out puzzles than training on AGI-1 directly. The generalization gap dropped from 24.72pp to 8.02pp — a 68% reduction.

### Transfer Ratio vs Reference

| Model | Parameters | Transfer Ratio | Notes |
|---|---|---|---|
| TRM (reference) | 7M | 17.8% | Conventional fixed-depth approach |
| **DRF-MWU** | **881K** | **108%+** | Field-generic adaptation |

The transfer ratio measures how much performance is retained when moving from one ARC distribution to another with zero retuning. DRF-MWU achieves above 100% transfer — meaning it performs better on the target distribution than on the source — with 8x fewer parameters than the reference model.

### Validated Theoretical Predictions

Every testable prediction made by MOPD theory before implementation was confirmed:

- ✅ **Spherical bound**: max observed θ = 2R exactly (validated on real ARC data)
- ✅ **A(τ) discrimination**: mean θ when A>0 is 2.22x mean θ when A<0 (threshold: 2.0x)
- ✅ **Mirror effect prior**: θ decreases on ~80% of untrained inference passes
- ✅ **Gamma crossover**: w_z gives way to w_y during healthy convergence, without explicit training
- ✅ **Field-generic transfer**: geometric adaptation transfers across distributions without retuning

---

## The Omega Kernel — Why Architectural Continuity Matters

Every architectural addition to this system must preserve what we call the **Omega Kernel** — the geometric continuity of the reasoning field. The field's convergence dynamics operate on a spherical manifold with constant positive curvature. Any component that introduces a flat, linear, or non-recursive step before the geometric structure is established breaks the convergence properties.

This is why the architecture looks the way it does:
- Two **independent** encoders (never shared weights — sharing makes θ(0) = 0 and the mirror effect has nothing to resolve)
- x_encoded is **detached** from the computation graph (the field is fixed — if it moves, convergence has no stable target)
- d_ff = d_model, not 4x (spatial reasoning does not need knowledge storage capacity)
- Geometric termination, not fixed depth (the field decides when it is done)
- Hypothesis selection **before** the mirror loop (attentional commitment precedes convergence)

Each of these choices follows from the requirement that the architecture's geometric spiral must be continuous. A previous experiment (multi-example encoding via mean-difference vectors) violated this principle and dropped performance from 67% to 58% cell accuracy and collapsed transfer ratio from 95.8% to 86.4%. The lesson was precise: you cannot attach a straight pipe to a nautilus shell.

---

## What Is Not Included Here

This repository contains the base architecture demonstrating the proof of concept — the mirror loop, field monitoring, spatial encoding, and hypothesis selection that achieve the transfer ratio results above.

The following components are under active development and not included:

- **AIDA** (Autonomous Intervention and Decision Architecture) — the field daemon that reads geometric state and decides whether to accept, extend, or perturb the field's convergence. AIDA's internal architecture and training methodology are patent-pending.
- **Sequential Example Refinement** — the geometrically correct approach to multi-example encoding, currently in development. Rather than preprocessing all examples before the mirror loop, the loop runs across examples in sequence — each example's terminal states initialize the next example's loop, allowing the field to accumulate cross-example rule knowledge through geometric convergence rather than flat aggregation.

---

## Architecture at a Glance

```
Config: d_model=128, n_heads=2, d_ff=128, tau_max=12, K=3
Total parameters: 881,805

Components:
├── SpatialEncoder × 2 (independent)     ~320K params
│   ├── Color embedding [n_colors+1, 128]
│   ├── Position embedding [max_cells, 128]
│   └── Single-head self-attention
├── HypothesisGenerator (K=3)            ~100K params
│   ├── Projector_0: Linear(128,128) → GELU → Linear(128,128) → LayerNorm
│   ├── Projector_1: (same)
│   └── Projector_2: (same)
├── MirrorTransformerBlock × 2           ~400K params
│   ├── Cross-attention (other_state → x_encoded)
│   ├── Self-attention
│   └── Feedforward (d_ff = d_model)
├── Cell head                            ~50K params
│   └── Linear(128,128) → GELU → Linear(128,10)
└── FieldMonitor3                        0 params
    └── Pure computation: θ, A, P, K, γ, w_y, w_z, EWS
```

---

## Theoretical Background

DRF-MWU is the first computational implementation of **Meta-Ontological Psychodynamics (MOPD)** — a theoretical framework for recursive reasoning developed at the Institute for Meta-Sciences. MOPD models reasoning as a geometric convergence process on a spherical manifold. Two bias operators move along geodesics, conditioned on each other, until they converge on the same reading of the input field.

The theoretical documents produced by this programme include:
- CRIS Foundational Knowledge — complete practitioner reference for recursive field systems
- The Omega Kernel — geometric continuity as the first architectural principle of recursive intelligence
- CRIS Technical Architecture — detailed explanation of every code component
- DRF-MWU Peak Results — complete empirical benchmark record

---

## Citation

```
@misc{drfmwu2026,
  title   = {DRF-MWU: Dynamic Recursive Field Model with Modulated Weights Update},
  author  = {SNA},
  year    = {2026},
  note    = {Institute for Meta-Sciences. ARC-AGI 2026 Competition Submission.}
}
```

---

*Institute for Meta-Sciences · 2026 · ARC-AGI Competition Submission*
