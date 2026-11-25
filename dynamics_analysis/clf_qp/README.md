# CLF-QP controller — cooperative pinch-lift-move

Companion to `../feedback_linearization`. It builds a **Control-Lyapunov-Function
Quadratic-Program (CLF-QP)** controller on the same 26-DOF constrained model and
compares it head-to-head with the exact feedback-linearizing (FL) law.

Motivation (from the report): exact FL cancels all nonlinearities but pins the
input via the pseudo-inverse, leaving only a 2-D null space to shape the
40-dimensional (unstable) zero dynamics. The CLF-QP instead enforces a **single
scalar Lyapunov-decrease constraint**, freeing far more input authority to be
directed toward internal stabilization.

## Method
With the constraint-consistent relative-degree-2 output model
`ÿ = L_f²y_eff(x) + Ĝ(x) u` (from the KKT-reduced dynamics `q̈ = α₀ + M_u u`) and
`η = (y, ẏ)`, the chain-of-integrators form is `η̇ = Fη + G ÿ`. We take the CLF
`V(η) = ηᵀPη`, with `P` from the **CARE** of the double integrator (LQR weights
chosen so the optimal closed loop is critically damped at `s = -ω_o`, but `P` is
well conditioned, unlike the Lyapunov `P` of a fast `A_cl`). Then

```
V̇ = L_fV + L_gV·u,   L_fV = ηᵀ(FᵀP+PF)η + 2ηᵀPG·L_f²y_eff,   L_gV = 2ηᵀPG·Ĝ .
```

The controller solves, at each state,

```
u = argmin_u ‖u − u_des‖²   s.t.   L_gV·u ≤ −r·V ,
```

a QP with a single inequality (closed-form half-space projection — no external
solver). `u_des` is an optional arm-posture PD torque
(`u_des = −k_{p,n}(θ_arm−θ_nom) − k_{d,n}θ̇_arm`); the QP honours it wherever the
output constraint leaves room. The decrease rate `r` is a design parameter (the
QP can enforce any feasible rate since `Ĝ` is full row rank).

> **Implementation note.** Unlike FL — which cancels `L_f²y` with its *own*
> estimate and is therefore immune to errors in it — the CLF-QP constraint feeds
> `L_f²y_eff` directly into `V̇`, so it must be accurate. We compute it as a
> directional derivative of `ẏ` along the `u=0` flow (`clf_qp.py`), not the nested
> spatial finite difference used by the FL drift term.

## Files
| file | contents |
|------|----------|
| `clf_qp.py` | CLF construction (`clf_from_care`), CLF-QP control law (`clf_qp_control`). |
| `run_clf.py` | head-to-head FL vs CLF-QP simulation + `fig_clf_comparison.png`. |

Run (uses the model/controller/ic modules from `../feedback_linearization`):
```bash
conda activate isaaclab
python run_clf.py            # writes fig_clf_comparison.png
python run_clf.py --quick     # faster, coarser
```

## Findings (see `fig_clf_comparison.png`; `ω_o=40`, `r=30`, `Δt=5e-4`)
Two initial conditions: a **zero-dynamics IC** (`y₀=0`, internal arm DOF
perturbed) and an **output IC** (payload perturbed in the controllable x–z plane).

| metric (zero-dynamics IC) | FL | CLF-QP (min) | CLF-QP (posture) |
|---|---|---|---|
| arm-shape drift ‖Δθ_arm‖(T) [rad] | 2.92 | 0.60 | **0.27** |
| control effort ∫‖u‖dt [N·m·s] | 27.6 | 16.2 | **13.7** |
| output excursion ‖y(T)‖ | 1.3e-3 | 9.1e-2 | 9.8e-2 |

- **Null-space / zero-dynamics drift (the point of the comparison).** Holding the
  payload, FL lets the internal arm-shape drift grow ~10× larger than CLF-QP; the
  posture-augmented CLF-QP keeps it smallest — the freed input authority is spent
  directly on internal regulation.
- **Control effort.** `∫‖u‖dt` is ~2× lower for CLF-QP (FL spends effort enforcing
  exact decoupling in all six output directions).
- **Output convergence (output IC).** FL gives the fastest, tightest convergence
  (exact `ÿ=v`, `‖y‖→1e-3` by `t≈0.1`). CLF-QP converges the output partially
  (`‖y‖≈0.032→0.012` by `t≈0.1`) and is then pulled back up by the unstable zero
  dynamics — neither controller beats them long-term. CLF-QP also induces stiffer
  closed-loop dynamics (the single-constraint QP), so its output convergence is
  somewhat under-resolved at `Δt=5e-4` (finer steps converge faster).

In short: **CLF-QP slows the zero-dynamics drift and roughly halves control effort
by better directing the available authority, at the cost of looser/slower output
tracking** — exactly the trade-off anticipated in the report. It does **not** make
the zero dynamics asymptotically stable: with arm torques alone the authority is
still insufficient (only a 2-D redundancy for 8 unstable internal DOF), consistent
with the structural result in `../feedback_linearization` — base regulation by the
locomotion layer is still required for a bounded formation.
