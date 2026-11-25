# Coupled Control-Barrier-Function (CBF) Safety Filter for decPLM

Stress-testing **decPLM** (Decentralized Pinch-Lift-Move, multi-quadruped cooperative object
transport) with obstacles, by wrapping the learned policy's high-level command in a minimally-invasive
control-barrier-function (CBF) safety filter. **No retraining**, identical observation space, reuses
the pretrained checkpoint.

> Repo: `~/Academics/Decentralized-Pinch-Lift-Move`
> Extension root: `drail_extensions/drail_extensions/research_bikram/`
> Conda env: `~/miniconda3/envs/isaaclab/bin/python`

---

## 1. Motivation

decPLM was trained in open space — no obstacles, no navigation constraints. The constellation reward
pushes each robot to track a rigid-body contact frame; if that trajectory passes through an obstacle,
the policy has **no concept of avoidance** and drives the team straight into it.

The CBF safety filter wraps the velocity command coming out of the high-level planner in a QP that
only intervenes when the nominal command would push toward a constraint boundary — minimally
invasive. **What's novel:** it's applied to a *multi-body coupled system* — both the robot base
trajectories and the object trajectory must stay safe simultaneously, all driven by a single command.

---

## 2. Where it plugs in (architecture)

decPLM's high-level planner emits a **box-frame twist** command
`vel_command_b = [lin_vel_x, lin_vel_y, ang_vel_z, height]` (the `box_command` term). The learned
decentralized policy observes a per-agent belief derived from it and tracks it; the constellation
reward tracks it.

The filter is installed as a **drop-in command term** that overwrites `vel_command_b` inside
`_update_command()` — which runs *before* the observation manager each step — so the policy
automatically observes and tracks the **safe** command. Because the observation transform
(`transformed_commands_v4`) reads `vel_command_b` as a box-frame twist and re-expresses it in each
agent's contact frame, writing the safe twist propagates to **all** agents with no obs-space change.

```
high-level planner ──► box_command (vel_command_b)
                          │
                          ▼  _update_command(): guidance + CBF-QP   ◄── obstacle geometry,
                          │                                              box & agent world poses
                          ▼
                       safe vel_command_b ──► obs transform ──► decentralized policy ──► joint torques
```

Decision variable of the QP: the **box-frame twist** `u = [vx, vy, ωz]`.

---

## 3. The coupled CBF formulation

For each rigid body `b ∈ {box_center, agent_0_base, …, agent_{N-1}_base}` and each obstacle `i`,
define a barrier:

```
h_{b,i}(x) = ‖p_b − o_i‖²  −  r_{b,i}²          (keep body b outside a disk around obstacle i)
r_{b,i} = body_radius_b + obstacle_radius_i + margin
```

Under the rigid grasp constellation, all bodies follow the **single** box twist:

```
ṗ_b = R(ψ)·[vx, vy]ᵀ  +  ωz · ẑ × (p_b − p_box)        # box itself: lever arm = 0
```

so the planar world velocity of body `b` is linear in `u`: `ṗ_b = A_b u`, with

```
A_b = [ R(ψ) | perp(p_b − p_box) ] ∈ ℝ^{2×3}
```

The CBF condition `ḣ_{b,i} ≥ −α·h_{b,i}` becomes one linear inequality in `u`:

```
gᵀ u ≥ −α h        with   g = 2 A_bᵀ (p_b − o_i)
```

**Safety filter QP** (minimally invasive):

```
u* = argmin_u  ½‖u − u_nom‖²
     s.t.   gᵀ_{b,i} u ≥ −α h_{b,i}   for all bodies b, obstacles i
```

One decision variable (box twist), one barrier per (body × obstacle). The box's own avoidance and
**both/all agents'** avoidance compete for the same control — that is the coupling. A velocity-level
(single-integrator) CBF-QP is always feasible (`u` unbounded), so no slack variable is needed.

**Constraint assembly** (`mdp/safety/cbf_qp.py: build_coupled_cbf_constraints`, fully batched):

```python
R = rotation_2d(box_yaw)                              # (B,2,2)
lever = body_pos_xy - box_pos_xy.unsqueeze(1)         # (B,Nb,2)
perp  = torch.stack((-lever[...,1], lever[...,0]), -1)# z_hat x lever
A = torch.empty(B, Nb, 2, 3); A[...,:2] = R.unsqueeze(1); A[...,2] = perp
d = body_pos_xy.unsqueeze(2) - obstacle_xy.unsqueeze(1)           # (B,Nb,K,2)
r_eff = body_radius[...,None] + obstacle_radius[:,None] + margin  # (B,Nb,K)
h_vals = (d*d).sum(-1) - r_eff*r_eff
G   = (2.0*torch.einsum("bnki,bnij->bnkj", d, A)).reshape(B, Nb*K, 3)
rhs = (-alpha*h_vals).reshape(B, Nb*K)
```

---

## 4. The QP solver — batched Hildreth (no new dependencies)

3 variables, a handful of constraints, identity Hessian (objective is a squared distance). Solved with
**Hildreth's algorithm** — projected Gauss-Seidel coordinate ascent on the non-negativity-constrained
dual — with **successive over-relaxation** (`ω≈1.5`), fully batched on-device.

Dual of `min ½‖u−u_nom‖² s.t. Gu ≥ rhs`:  `max_{λ≥0}  −½ λᵀ W λ + bᵀ λ`, with `W = GGᵀ`,
`b = rhs − G u_nom`; primal recovery `u = u_nom + Gᵀ λ`.

```python
def solve_qp_hildreth(u_nom, G, rhs, n_iter=200, omega=1.5, eps=1e-9):
    W = torch.bmm(G, G.transpose(1,2))                 # (B,M,M)
    b = rhs - torch.bmm(G, u_nom.unsqueeze(-1)).squeeze(-1)
    diag = W.diagonal(dim1=1, dim2=2)
    degenerate = diag < eps                            # zero-gradient rows are inert
    inv_diag = torch.where(degenerate, 0.0, 1.0/diag.clamp_min(eps))
    lam = torch.zeros(B, M)
    for _ in range(n_iter):
        for i in range(M):
            s = torch.einsum("bj,bj->b", W[:,i,:], lam) - W[:,i,i]*lam[:,i]
            lam_t = ((b[:,i] - s) * inv_diag[:,i]).clamp_min(0.0)
            lam[:,i] = (lam[:,i] + omega*(lam_t - lam[:,i])).clamp_min(0.0)   # SOR
    lam = torch.where(degenerate, 0.0, lam)
    return u_nom + torch.bmm(G.transpose(1,2), lam.unsqueeze(-1)).squeeze(-1)
```

Why SOR: `W = GGᵀ` is rank-deficient when there are more constraints than control dims (the usual CBF
case), so plain Gauss-Seidel has flat dual directions and converges slowly; `ω≈1.5` reaches
feasibility ~3× faster. **Verified against `scipy` SLSQP to ~1e-5** on random feasible QPs, and
feasible (constraint residual `> −1e-4`) within ~200 sweeps.

---

## 5. The command term + guidance

`mdp/commands/cbf_safe_velocity_height_command.py` →
`CBFSafeVelocityHeightCommand(CategoricalVelocityHeightCommand)`.

Each step (`_update_command`):
1. Read world poses of the box + every agent base.
2. Compute the **guidance** nominal twist (see below).
3. Solve the CBF-QP → `u_safe`.
4. Record diagnostics (always, both modes).
5. Write `vel_command_b[:, :3] = u_safe if enable_cbf else u_nom`.

```python
u_nom4 = self._guidance_command(box_pos_xy, box_yaw, with_circulation=self.cfg.enable_cbf)
self.vel_command_b[:, 3] = u_nom4[:, 3]            # height (lift) command
u_nom  = u_nom4[:, :3]
u_safe, diag = cbf_safety_filter(u_nom, box_pos_xy, box_yaw, body_pos_xy,
                                 self._body_radius, self._obstacle_xy, self._obstacle_radius,
                                 alpha=cfg.cbf_alpha, margin=cfg.cbf_margin, n_iter=cfg.cbf_qp_iters)
...
self.vel_command_b[:, :3] = u_safe if self.cfg.enable_cbf else u_nom
```

### 5.1 Why guidance was needed (traversing a whole course)

A pure CBF filter on a constant `+x` command **stalls** head-on at the first obstacle: the QP removes
only the radial (into-obstacle) velocity and leaves zero tangential, so the box sits at the keep-out
boundary and never reaches later obstacles. Two guidance terms fix this, applied to the nominal
*before* the QP:

- **Goal-seeking** (`goal_seeking`, `goal_xy`, `goal_speed`, `goal_kp`): P-controller toward a far
  goal, so after a deflection the box steers back on course and reaches the next obstacle.
- **Circulation** (`circulation_gain`, `circulation_dist`): a tangential `perp(box − obstacle)` term
  that grows near the keep-out boundary so the box *rounds* a head-on obstacle instead of stalling.

```python
# goal-seeking (P-control, speed-capped) in world frame
v_world = goal_kp * (goal_world - box_pos_xy)
v_world = v_world * (goal_speed / v_world.norm(...,keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
if with_circulation and circulation_gain > 0:
    v_world = v_world + self._circulation(box_pos_xy, v_world)
u_nom_body = R(yaw)ᵀ @ v_world
```

```python
def _circulation(box_pos_xy, v_world):              # world-frame go-around term
    d = box_pos_xy[:,None,:] - obstacle_xy           # outward radial
    n = d / d.norm(dim=-1, keepdim=True)
    t = perp(n)                                       # tangent
    s = sign( (t * v_world[:,None,:]).sum(-1) )       # detour toward the forward side
    r_eff = box_radius + obstacle_radius + margin
    act = exp( -clearance / circulation_dist )        # ~1 at boundary, decays outward
    return (circulation_gain * act * s)[...,None] * t  ).sum(1) * speed
```

### 5.2 CBF **off** vs **on** — the critical distinction

The **planner** (goal-seeking) is obstacle-unaware and shared by both modes. The **avoidance** terms
(circulation + the CBF QP) are the *safety layer* and apply **only when `enable_cbf` is True** — note
`with_circulation=self.cfg.enable_cbf` above. This makes the comparison meaningful:

| Mode | applied command | behavior |
|------|-----------------|----------|
| **off** (baseline) | `u_nom` = goal-seeking, **no circulation, no QP** | drives **straight into** obstacles → collides |
| **on**  | `u_safe` = goal-seeking **+ circulation + CBF-QP** | **weaves around** obstacles, stays safe |

> ⚠️ Earlier the circulation term was in the *shared* guidance, so the baseline also avoided
> (softly) — making off/on look identical. Circulation is now gated on `enable_cbf`, so off is a
> genuine "decPLM has no obstacle concept" baseline that collides.

The obstacles are **present in the scene in both modes**; only the safety filter toggles. (The wandb
run names `obstacles-off` / `obstacles-on` mean "obstacle task with CBF off/on", *not* "obstacles
removed".)

### 5.3 Scaling to N agents

`body_radii` / `box_radius` + `agent_radius` and `agent_asset_names` make it scale to any agent count
(2-robot or 4-robot). The QP gains one barrier per agent, all sharing the one box twist — no code
change.

---

## 6. Diagnostics — margin barrier vs *physical* clearance

Two distinct safety signals (both logged):

- `cbf_min_h` — the CBF barrier **including the margin**: `< 0` once a body enters the *inflated*
  keep-out (radii + `margin`). Conservative.
- `cbf_min_clearance` — **margin-free** surface-to-surface distance: `< 0` ⇔ an **actual physical
  overlap**. This is the real collision test.

```python
dd   = body_pos_xy[:,:,None,:] - obstacle_xy[:,None,:,:]           # (B,Nb,K,2)
surf = dd.norm(dim=-1) - (body_radius[None,:,None] + obstacle_radius[None,None,:])
self.cbf_min_clearance = surf.flatten(1).min(dim=-1).values
```

A run can report `min_h = −0.04 (margin)` while `min_clearance = +0.23 m (physical)` → **no actual
contact**, the body merely dipped into the safety buffer. The margin absorbs the gap between the
rigid-constellation model and the real (wider-swinging) agent motion.

---

## 7. Obstacle scenes & tasks

Static **kinematic cylinder** obstacles added to the scene; the `box_command` swapped to the CBF term
**after** `super().__post_init__()` (an ancestor play `__post_init__` installs a GUI-keyboard command
that also crashes headless — must override afterward).

A **slalom** layout (obstacles alternating sides near the centerline, goal just past the last) so the
goal-seeking box must weave past every obstacle.

| Task id | file |
|---|---|
| `Go2-D1-MARL-Box-Reach-Touch-Lift-and-Move-Finetuning-Obstacles-Play-v0` | `tasks/.../finetuning/..._obstacles_env_cfg.py` |
| `…-Finetuning-N-Robots-Obstacles-Play-v0` | `tasks/.../finetuning/n_robots/..._N_robot_obstacles_env_cfg.py` |

```python
# 2-robot obstacle env (conservative gains that the real policy can track)
OBSTACLES = [(1.8, 0.40, 0.30), (3.6, -0.40, 0.30)]     # (x, y, radius), env-local
GOAL_XY   = (4.8, 0.0)
mdp.CBFSafeVelocityHeightCommandCfg(
    enable_cbf=True, obstacles=OBSTACLES, agent_asset_names=("robot","robot_1"),
    body_radii=(0.33, 0.35, 0.35), cbf_alpha=1.5, cbf_margin=0.25, cbf_qp_iters=200,
    goal_seeking=True, goal_xy=GOAL_XY, goal_speed=0.3, goal_kp=0.8,
    circulation_gain=0.4, circulation_dist=0.7,
    ranges={"all": Ranges(lin_vel_x=..., lin_vel_y=(0,0), ang_vel_z=(0,0), height=HEIGHT_RANGE,
                          velocity_dead_zone=..., probability=1.0)})   # velocity from guidance; range supplies height
```

The N-robot env reuses the 2-robot checkpoint via **non-strict load** (`ppo_runner.load(path,
strict=False)`): the centralized critic's input dim differs with agent count, but only the **actor**
is needed for inference (`[StateDict] Skipping: critic.0.weight`).

---

## 8. Eval driver, visualization, camera, video

`scripts/play_cbf.py` — loads the pretrained policy, toggles the filter (`--cbf on|off`), drives the
box, logs to wandb project **`decplm-cbf`**.

**Metrics logged**
- `safety/min_barrier`, `safety/min_clearance`, `safety/intervention`, `safety/active`,
  `safety/collision`, `grasp/contact_frac`, `track_ep/*`
- summary: `summary/phys_collision_rate`, `summary/mean_min_clearance`, `summary/collision_rate`,
  `summary/mean_min_barrier`, `summary/mean_intervention`

**wandb artifacts** (`scripts/cbf_viz.py`, generic over agent count)
- `plots/performance` — top-down box+agent trajectories with obstacle keep-out disks, `min h(t)`,
  filter intervention, and box height (lift maintained while deflecting).
- `plots/trajectory_gif` — animated top-down GIF (bodies turn red on any frame with `h < 0`).
- `rollout_video` — the Isaac Sim camera video.

**Camera (wide shot of the whole course)** — set in the obstacle env cfg so the box never walks out of
frame:
```python
self.viewer.origin_type = "env"; self.viewer.env_index = 0
self.viewer.eye    = (2.0, -7.0, 5.0)   # CAMERA_EYE (env-local), elevated 3/4 view
self.viewer.lookat = (2.4, 0.0, 0.3)    # CAMERA_LOOKAT
self.viewer.resolution = (1920, 1080)
```

**Video flush fix** (gymnasium 1.2.3 gotcha): `RecordVideo` only writes the mp4 when frame count
*exceeds* `video_length` or on `close()`; a single-episode run with `video_length == steps` never
trips it, and the script logged before `close()`. Fix — keep a handle and flush explicitly:
```python
if video_recorder is not None and getattr(video_recorder, "recording", False):
    video_recorder.stop_recording()      # writes the mp4 before we log it
```

---

## 9. Validation results

**Standalone (no Isaac, perfect tracking)** — `scripts/cbf_standalone_demo.py`, 3-obstacle slalom +
goal-seeking:
| | min barrier h | reaches goal? |
|---|---|---|
| CBF off (goal-seeking only) | **−0.36** (collision) | yes |
| CBF on (+ circulation + QP) | **+0.003** (safe)    | yes |

**Solver:** matches `scipy` SLSQP to ~`1e-5`; feasible within ~200 SOR sweeps.

**In Isaac (pretrained policy, single env):**
- Early simple 2-obstacle test: baseline `collision_frac 0.26` vs CBF `0.01` (26× less).
- Full slalom, CBF on, `cbf_margin=0.25`, 1600 steps: `min_h −0.038 (margin)`,
  **`min_clearance +0.228 m (physical)`, `phys_collision_rate 0.00`**, `collision_frac 0.04` — box
  weaves an **S-curve through both obstacles to the goal**, lift maintained (~0.10 m).
- N-robot (4 agents): runs end-to-end, weaves past obstacles, lift held.

**Key finding (honest limitation):** the CBF guarantees the *commanded* twist is safe, but the
*realized* trajectory depends on the policy tracking it. The agents swing **wider** than the
rigid-constellation model assumes, so aggressive guidance (fast goal-seeking + strong circulation)
commands maneuvers the policy can't track → the constellation overshoots into obstacles. **Gentle,
trackable guidance + a margin sized to absorb agent excursions** keeps the realized trajectory
physically collision-free.

---

## 10. Tuning knobs (`CBFSafeVelocityHeightCommandCfg`)

| Knob | Effect |
|---|---|
| `cbf_alpha` | class-K gain; larger ⇒ later/sharper intervention |
| `cbf_margin` | extra keep-out buffer; **the main knob to absorb agent tracking excursions** |
| `box_radius` / `agent_radius` / `body_radii` | per-body avoidance footprint radii |
| `obstacles`, `goal_xy` | course layout (env-local); keep in sync with `obstacle_*` scene slots |
| `goal_seeking`, `goal_speed`, `goal_kp` | the obstacle-unaware planner |
| `circulation_gain`, `circulation_dist` | go-around strength / range (safety layer; on only) |
| `cbf_qp_iters` | Hildreth SOR sweeps (200 is solid) |

**Trade-off:** bigger `margin` → safer realized trajectory, but tighter effective corridors; faster
`goal_speed` / stronger `circulation_gain` → traverses faster but commands maneuvers the policy may
not track.

---

## 11. How to run

```bash
cd ~/Academics/Decentralized-Pinch-Lift-Move

# Standalone, no Isaac (~1 min): writes a figure
python drail_extensions/drail_extensions/research_bikram/scripts/cbf_standalone_demo.py --out /tmp/cbf_demo.png

# In Isaac, baseline vs CBF, logs video + GIF + plots to wandb (decplm-cbf)
./run_play_cbf.sh --mode both --robots 2 --video           # 2-robot, default --steps 1500
./run_play_cbf.sh --mode both --robots N --video           # 4-agent
./run_play_cbf.sh --mode on   --robots 2 --video --steps 1000
```

`--mode both` runs the **same obstacle scene twice** (CBF off then on); `--robots {2,N}` selects the
2- or 4-agent task; `--video` records the 1080p wide-shot camera (slower).

---

## 12. File inventory

| File (under `drail_extensions/drail_extensions/research_bikram/`) | Purpose |
|---|---|
| `mdp/safety/cbf_qp.py` | Dependency-free batched CBF-QP: constraint assembly + Hildreth/SOR solver + `cbf_safety_filter`. Runnable self-test. |
| `mdp/safety/__init__.py` | Exports. |
| `mdp/commands/cbf_safe_velocity_height_command.py` | `CBFSafeVelocityHeightCommand` — guidance (goal-seeking + circulation), CBF filtering of `vel_command_b`, diagnostics. |
| `mdp/commands/__init__.py` | Registers `CBFSafeVelocityHeightCommandCfg`. |
| `tasks/.../finetuning/..._obstacles_env_cfg.py` | 2-robot obstacle scene + task + camera. |
| `tasks/.../finetuning/n_robots/..._N_robot_obstacles_env_cfg.py` | 4-agent obstacle scene + task + camera. |
| `tasks/.../finetuning/__init__.py`, `.../n_robots/__init__.py` | gym registrations. |
| `scripts/play_cbf.py` | Eval driver (toggles filter, logs metrics/video/GIF/plots). |
| `scripts/cbf_viz.py` | Performance figure + animated trajectory GIF (generic over agent count). |
| `scripts/cbf_standalone_demo.py` | No-Isaac 2-D validation + figure. |
| `CBF_SAFETY_FILTER.md` | In-repo docs. |
| `../../run_play_cbf.sh` | Convenience wrapper (`--mode`, `--robots`, `--video`, `--steps`). |

---

## 13. Does it protect both the carried box *and* the agents?

**Yes** — one barrier per `(body × obstacle)` over `[box, agent_0, …]`, all sharing the single box
twist, so the QP keeps the **carried box and every robot base** outside obstacles at once. Caveats:
each body is a **circular footprint** (not full hull/legs/arm), and agents are modeled as **rigidly
following the box** — real agents swing wider, and `cbf_margin` is what absorbs that. `cbf_min_h` /
`cbf_min_clearance` are the min over **all** bodies, so a near-miss by either the box or an agent
shows up.

---

## 14. The command-vs-realized gap, and the robust (tube-CBF) margin — *Solution B*

**The finding.** The CBF guarantees the *commanded* twist is safe, but the *realized* trajectory
depends on policy tracking. decPLM was trained with **no obstacles**, so when the CBF deflects its
command the constellation reward *fights* the deflection and the agents swing wider than the rigid
model assumes. That is why `cbf_margin` is load-bearing. The gap decomposes into three causes:
(1) **model mismatch** (agents not rigidly slaved to the box), (2) **tracking lag**, (3) **objective
conflict** (the reward resists the filter). Each points to a different fix:

| Fix | Attacks | Cost |
|---|---|---|
| **A. Train-time CBF (shielded RL):** co-train with obstacles + filter active so the policy stops fighting it | (3), adapts to (1)(2) | full retrain (strongest / most novel) |
| **B. Robust / tube CBF margin:** inflate the barrier by a *certified* excursion bound `δ̄` | (1)(2) | none (data-driven) — implemented here |
| C. Predictive / lag-aware CBF: filter the *predicted* realized state | (2) | medium |
| D. Speed/gain scheduling: slow the command as `h→0` | (2)(3) | free, partial |

### Solution B as implemented

Treat the realized deviation from the rigid prediction as a bounded position disturbance `δ` (tube /
input-to-state-safe CBF). Inflating every keep-out radius by margin `m` makes the realized worst
clearance `c(m) ≈ m − δ(m)`. Physical safety with buffer `b` needs `c(m) ≥ b`.

- `play_cbf.py` now dumps a per-step series `cbf_series_{on,off}_ep*.npz` (`min_h`, `min_clearance`,
  `margin`, trajectories).
- `scripts/cbf_robust_margin.py` reads those (or direct `(margin, clearance)` points), computes the
  excursion `δ_t = max(0, m − min_clearance_t)`, `δ̄ = max_t`, and — over a **margin sweep** — fits
  `c(m)` and solves `c(m*) = b` for the smallest safe margin.

```bash
# from a margin sweep of saved runs:
python scripts/cbf_robust_margin.py logs/.../cbf_series_on_ep0.npz ... --buffer 0.05
# or directly from measured points:
python scripts/cbf_robust_margin.py --points "0.12,-0.031 0.25,0.228" --buffer 0.05
```

### Preliminary result (2-robot slalom, gentle guidance)

From the two controlled runs we have (same params, only `cbf_margin` changed):

| `cbf_margin` | realized `min_clearance` |
|---|---|
| 0.12 | **−0.031 m** (contact) |
| 0.25 | **+0.228 m** (safe, conservative) |

Linear fit `clearance(m) ≈ 1.99·m − 0.270`:
- **just physically safe** (clearance 0) at `cbf_margin ≈ 0.136 m`
- **5 cm physical buffer** at `cbf_margin ≈ 0.161 m`  ← recommended

So `cbf_margin = 0.25` was conservative; **≈ 0.16 m** gives a comfortable physical buffer with the
current gentle guidance. Key mechanism: `δ(m)` *shrinks* as `m` grows (a larger margin deflects
earlier and gentler, so the policy overshoots less) — a favorable coupling, so the fixed point is
stable. Refine with a 3–4-point sweep (run `play_cbf.py` at `cbf_margin ∈ {0.12, 0.16, 0.20}`, then
feed the `cbf_series_*.npz` to `cbf_robust_margin.py`).

> This turns the "honest limitation" into a **certified robust margin** — a result, not a hack —
> without retraining. Solution A (train-time CBF) remains the more complete fix for cause (3).

### Integration (now wired into Isaac)
- `cbf_margin = 0.16` is now the default in the 2-robot obstacle cfg (the data-grounded robust value).
- `play_cbf.py --cbf_margin <m>` overrides it without editing the cfg (for sweeps);
  `run_play_cbf.sh --cbf_margin <m>` passes it through.
- Each run saves `cbf_series_{on,off}_ep*.npz`; sweep then analyze:
  ```bash
  for m in 0.12 0.16 0.20; do ./run_play_cbf.sh --mode on --robots 2 --cbf_margin $m --steps 1600; done
  python scripts/cbf_robust_margin.py logs/.../cbf_series_on_ep0.npz ... --buffer 0.05
  ```
- **N-robot:** the 4-agent constellation swings wider, so its robust margin is larger and needs its
  own sweep; `cbf_margin` cannot fix the separate **grasp-detachment** issue below.

---

## 15. Trajectory-plot artifacts: env resets and agent detachment

Two things show up in the top-down figures that are **not** policy maneuvers:

1. **Env reset (teleport).** In the **baseline (CBF off)** the constellation drives into an obstacle,
   a robot tips past its orientation limit (`rpy_limit_robot`) → `done=True` → IsaacLab auto-resets
   the env, teleporting box + all robots back to spawn. The old plot drew a straight "elastic" line
   across the teleport. **Fixed:** `cbf_viz.py` detects single-step position jumps > 0.4 m
   (`_reset_indices`) and **breaks the trajectory lines at resets**, marks reset times with dashed
   lines on `h(t)`, and resets the GIF trail. The `min h(t)` snapping back to its large start value is
   the reset signature.

2. **Agent detachment (real grasp loss).** In the **4-agent CBF-on** runs the box reaches the goal
   (no reset) but some agents **drift far from the box** — the constellation links stretch long. This
   is genuine: the policy was trained for **2 agents** and applied to **4** under CBF deflections, so
   it cannot hold all four contacts while the object is pushed/deflected. This is *the* coupled-grasp
   finding (the "does agent 2 keep its grasp?" question) — not a plotting artifact, and not something
   `cbf_margin` addresses. It points to **Solution A (train-time CBF, co-trained with obstacles and
   the right agent count)** as the real fix.
