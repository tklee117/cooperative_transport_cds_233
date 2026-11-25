# Feedback linearization & zero-dynamics study — cooperative pinch-lift-move

Numerical companion to `../system_model_notes.txt`. It builds the 26-DOF
control-affine model of two Go2 floating bases + 4-DOF D1 arms pinching a rigid
payload, constructs the feedback-linearizing controller for the 6-DOF payload
output, and assesses **output convergence** and **zero-dynamics stability** by
simulation. Pure NumPy/SciPy/Matplotlib — no Isaac Lab needed.

## Files
| file | contents |
|------|----------|
| `model.py` | generalized coordinates `q∈R^26`, forward kinematics, analytic body Jacobians, block inertia `D(q)` (SPD), gravity `G(q)`, Coriolis `C(q,q̇)q̇` (block Christoffel), actuation `B` (8 arm torques), holonomic contact constraints `κ` and `J_κ`, payload-pose output `y` and `J_y`. |
| `controller.py` | constraint-consistent feedback linearization, the KKT-reduced constrained dynamics, the damped pseudo-inverse law, optional null-space posture term, and the passive/locomotion-inner-loop forces. |
| `ic.py` | nominal pinch posture and projection onto the constraint manifold `{κ=0}` and its tangent `{J_κ q̇=0}`. |
| `analysis.py` | rigorous zero-dynamics tooling: equilibrium feasibility on `Z` + closed-loop linearization eigenvalues. |
| `study.py` | Experiments A & B (rollout) + figures `fig_A_output_convergence.png`, `fig_B_zero_dynamics.png`. |
| `spectrum.py` | Experiment C: no-equilibrium check + variational eigenvalue spectrum → `fig_C_eigenvalue_spectrum.png`. |
| `smoke_test.py` | sanity checks (SPD `D`, `κ→0`, `ÿ=v`, decoupling rank). |

Run:
```bash
conda activate isaaclab          # provides numpy/scipy/matplotlib
python study.py                  # Experiments A & B (single-threaded; a few minutes)
python study.py --quick          # faster, coarser
python spectrum.py               # Experiment C: zero-dynamics eigenvalue spectrum
python smoke_test.py             # quick validation
```

## Model (faithful to the report)
- `q = (q₁ᵇ, θ₁ᵃ, q₂ᵇ, θ₂ᵃ, p) ∈ R²⁶`: two robots (6-DOF ZYX-Euler base + 4 arm
  joints each) and the 6-DOF payload pose.
- Block-diagonal inertia `D = diag(D₁₁, D₂₂, D_oo)` exactly as in the report;
  the `Wᵣ` Euler-rate map is used for the rotational blocks (singular at
  `θ=±π/2`, the gimbal-lock domain restriction).
- Robots couple to the payload **only** through the holonomic pinch constraints
  `κᵣ(qᵣ,p)=0` (each end-effector pose = a contact frame on the payload),
  enforced by Lagrange multipliers `F` (the contact wrenches).
- First-order control-affine form `ẋ = f(x) + g_u(x)u + g_F(x)F`, `x=(q,q̇)∈R⁵²`.

## Controller — a needed correction to the report's decoupling matrix
The report writes the decoupling matrix `G = J_y D⁻¹ B`. **For this system that
matrix is identically zero**: `D` is block diagonal, `J_y` is nonzero only on the
payload columns, and `B` only on the arm-torque rows, so `J_y D⁻¹ B ≡ 0`
(verified numerically in `smoke_test.py`). The payload is unactuated — it is
driven purely by the contact wrenches `F`. The correct relative-degree-2 map
comes from the **constrained** dynamics, where `F = F(q,q̇,u)`:

```
[ D   -J_κᵀ ][ q̈ ]   [ Bu - h ]
[ J_κ   0   ][ F  ] = [  -a_κ  ]      ⇒   q̈ = α₀(x) + M_u(x) u
```

so `ÿ = (drift_y + J_y α₀) + (J_y M_u) u =: L_f²y_eff + Ĝ u`. The
**constraint-consistent decoupling matrix** `Ĝ = J_y M_u ∈ R^{6×8}` is full row
rank 6 away from arm/gimbal singularities (exactly the report's condition), and

```
u* = Ĝ⁺ (-L_f²y_eff + v),   v = -K_P y - K_D ẏ
```

gives `ÿ = v` (verified to ~1e-7). A damped (singularity-robust) pseudo-inverse
is used for `Ĝ⁺`.

## Results

**(A) Output convergence — verified.** From several perturbed initial
conditions the output error decays exponentially along the designed Hurwitz
dynamics `η̇=A_clη`; the feedback-linearization residual `‖ÿ-v‖` stays ~1e-6 and
the decoupling matrix stays well-conditioned. The 6-DOF payload output is
feedback-linearizable with vector relative degree `(2,2,2,2,2,2)`, as derived in
the report.

**(B) Zero dynamics — UNSTABLE (rollout, `fig_B`).** Holding the payload on its
trajectory (`y≡0`) and integrating the 40-D internal dynamics, the internal
arm/base self-motion **diverges** while the output stays held; the open-loop
(`u=0`) internal *shape* of the same plant stays bounded. So it is the
payload-pose output itself that makes the cooperative system non-minimum phase —
the simulation answer to the report's open question ("…whether arm joint angles
or base poses drift to singularities": they do).

**(C) Zero dynamics — linearization (rigorous, `fig_C`, `spectrum.py`).** Two
findings pin down the mechanism:
- **No equilibrium on `Z`.** The smallest internal acceleration achievable over
  the 8-DOF contact-consistent self-motion manifold is bounded away from zero
  (`min‖q̈‖ ≈ 20 rad/s²`, and `≈ 149 rad/s²` at the nominal posture) — *even with
  full gravity compensation*. Holding the unactuated payload requires arm torques
  that necessarily induce internal acceleration, and the 2-D control redundancy
  cannot null it. The zero dynamics therefore have **no rest point** and *drift*.
- **Unstable variational modes.** Linearizing the closed loop about the nominal
  point on `Z`, the 12 output-error modes sit at the placed location `s = -ω_o`
  (with a Baumgarte constraint cluster nearby), but **5 internal modes have
  `Re(λ) > 0`** (max ≈ 3.5/s). So the zero dynamics are both drifting and locally
  expansive.

**Why, and what it implies.** This is *structural*: 8 arm torques control 6
outputs, leaving only a 2-D redundancy for the 8 unstable internal configuration
DOF. Null-space posture control (only delays divergence), arm joint damping, and
base velocity damping were all verified insufficient. Maintaining a bounded
formation therefore requires the **locomotion inner loop to actively regulate the
base pose** — precisely the report's modelling assumption that the base is a
velocity-controlled body governed by the frozen low-level policy. With arm
torques alone the payload tracks perfectly but the robots cannot hold formation.

## Key parameters (`model.py`)
Go2 trunk 13 kg; D1-scale 4R arm (links ~0.1–0.16 m, 0.45 kg, pad 0.3 kg);
box payload 3 kg, 0.30×0.40×0.25 m; pads on the ±x faces; `g=9.81`. Values are
representative — the structural conclusions do not depend on the exact numbers.
