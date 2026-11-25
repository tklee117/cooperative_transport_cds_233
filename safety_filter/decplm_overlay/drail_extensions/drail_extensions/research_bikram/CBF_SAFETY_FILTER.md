# Stress-testing decPLM with obstacles: a coupled-CBF safety filter

decPLM was trained in open space — no obstacles, no navigation constraints. The constellation reward
pushes each robot to track a rigid-body contact frame, so if the commanded box trajectory passes
through an obstacle, the learned policy has no notion of avoidance and drives the team straight into
it. This module wraps decPLM's high-level box command in a **control-barrier-function (CBF) safety
filter** that keeps the box *and both robot bases* outside obstacles, without retraining and without
touching the learned policy.

## Idea

decPLM's high-level planner emits a box-frame twist command
`u_nom = [v_x, v_y, ω_z]` (the `box_command` term) that the decentralized policy tracks. We replace
that command with the solution of a minimally-invasive QP:

```
u* = argmin ‖u − u_nom‖²
     s.t.   ∇h_{b,i}(x)·ṗ_b(u) ≥ −α · h_{b,i}(x)   for every body b, obstacle i
```

with one barrier per (rigid body `b`, obstacle `i`):

```
h_{b,i}(x) = ‖p_b − o_i‖² − r_{b,i}²       (keep body b outside a disk around obstacle i)
```

The filter only intervenes when the nominal command would push a body toward a constraint boundary —
otherwise `u* = u_nom`.

### Why this is a *coupled* (multi-body) CBF

A single decision variable — the **box twist** — moves three rigid bodies at once: the box center
(body 0) and the two robot bases (bodies 1, 2), which under the rigid grasp constellation follow

```
ṗ_b = R(ψ)·[v_x, v_y] + ω_z · ẑ × (p_b − p_box).
```

So the box's own avoidance constraint and *both agents'* avoidance constraints share, and compete
for, the same control input. An obstacle on one agent's path bends the whole team's commanded
trajectory. This is the non-trivial coupling: the object's CBF is tied to both agents' states.

### The open research question this is built to probe

When agent 1's path is deflected around an obstacle, agent 2's decPLM policy — which only observes the
object state and its own proprioception, with no obstacle awareness and no inter-robot communication —
must *infer the deflection from the shared command and the object's motion* and keep its grasp. Does
it adapt and maintain contact, or does the object get torqued into an unstable configuration? The
eval logs `grasp/contact_frac` alongside the safety metrics precisely to measure this.

## Solver

The QP has 3 variables and a handful of constraints, with a strictly convex identity Hessian (the
objective is a squared distance). It is solved with **Hildreth's algorithm** — over-relaxed projected
Gauss-Seidel on the non-negativity-constrained dual — which needs no matrix inversion and runs fully
batched across parallel environments on-device, adding no new dependencies. A velocity-level (single
integrator) CBF-QP is always feasible (`u` is unbounded), so no slack variable is needed. The solver
is unit-tested for feasibility and matches a `scipy` SLSQP reference to ~1e-5 on random feasible QPs.

## Files

| File | Purpose |
|------|---------|
| `mdp/safety/cbf_qp.py` | Dependency-free batched CBF-QP core: constraint assembly, Hildreth solver, `cbf_safety_filter`. Has a runnable self-test (`python cbf_qp.py`). |
| `mdp/commands/cbf_safe_velocity_height_command.py` | `CBFSafeVelocityHeightCommand`: a drop-in for `box_command` that filters `vel_command_b` each step (before the obs manager), so the policy observes the *safe* command. `enable_cbf=False` runs the baseline while still logging diagnostics. |
| `tasks/.../go2_d1_marl_box_reach_touch_lift_and_move_finetuning_obstacles_env_cfg.py` | Obstacle scene + task cfg. Registers `…-Finetuning-Obstacles-Play-v0`. |
| `scripts/play_cbf.py` | Eval driver: loads the pretrained policy, toggles the filter, logs safety + grasp metrics, a performance figure and an animated trajectory GIF to wandb `decplm-cbf`. Works for 2 and N agents. |
| `scripts/cbf_viz.py` | Builds the wandb performance figure + top-down trajectory GIF from a rollout record. Generic over agent count. |
| `scripts/cbf_standalone_demo.py` | No-Isaac 2-D validation of the coupled filter; writes a trajectory/barrier/intervention figure. |
| `tasks/.../n_robots/...N_robot_obstacles_env_cfg.py` | N-robot (4-agent) obstacle env. Registers `…-N-Robots-Obstacles-Play-v0`. |
| `../../run_play_cbf.sh` | Convenience wrapper: baseline vs filtered, `--robots {2,N}`, `--video`. |

## wandb artifacts

Each `play_cbf.py` run logs, in addition to the scalar metrics:
- `plots/performance` — a static figure: top-down box+agents trajectories with obstacle keep-out
  disks, `min h(t)`, filter intervention, and box height (lift maintained while deflecting).
- `plots/trajectory_gif` — an animated top-down GIF; bodies turn red on any frame with `h < 0`, so
  the baseline-vs-filtered difference is obvious at a glance.
- `rollout_video` — the Isaac camera video (when launched with `--video`).

## Running

**Standalone (no Isaac), produces a figure:**
```bash
python drail_extensions/drail_extensions/research_bikram/scripts/cbf_standalone_demo.py --out /tmp/cbf_demo.png
```

**In Isaac with the pretrained policy (baseline vs filtered):**
```bash
./run_play_cbf.sh --mode both --vel_x 0.4               # 2-robot, logs to wandb project decplm-cbf
./run_play_cbf.sh --mode both --robots N --vel_x 0.4    # N-robot (4-agent) obstacle task
./run_play_cbf.sh --mode on --video                     # also record the Isaac camera video
```

The N-robot env reuses the 2-robot checkpoint (loaded non-strict: only the actor is needed; the
centralized critic's input differs with agent count). The CBF QP gains one barrier per agent, all
sharing the single box twist — so it scales to any agent count with no code change.

## Metrics logged (`decplm-cbf`)

- `safety/min_barrier`, `safety/min_barrier_ep` — smallest barrier `h` over {box, agent 0, agent 1} ×
  obstacles; `< 0` means a body entered a keep-out disk (collision).
- `safety/collision_frac`, `summary/collision_rate` — how often / whether the team collided.
- `safety/intervention`, `safety/active_frac` — magnitude/frequency of the filter's correction
  (minimally invasive: ~0 away from obstacles).
- `grasp/contact_frac` — fraction of steps both contact pads stay loaded while the trajectory is
  deflected (the agent-2 grasp-maintenance question).

## Traversing a whole obstacle course (goal-seeking + circulation)

A pure CBF filter on a constant `+x` command **stalls** at the first head-on obstacle: the QP removes
only the radial (into-obstacle) velocity and leaves zero tangential, so the box sits at the keep-out
boundary and never reaches later obstacles. Two guidance terms fix this (both in
`CBFSafeVelocityHeightCommand`, applied to the nominal *before* the QP):

- **Goal-seeking** (`goal_seeking=True`, `goal_xy`, `goal_speed`, `goal_kp`): the nominal command is a
  P-controller toward a far goal, so after a deflection the box steers back on course and reaches the
  next obstacle. The goal-seeking nominal is used in **both** baseline and CBF runs, so the only
  difference is the filter — an apples-to-apples comparison (baseline plows through, CBF weaves).
- **Circulation** (`circulation_gain`, `circulation_dist`): a tangential `perp(box − obstacle)` term
  that grows near the keep-out boundary, so the box *rounds* a head-on obstacle instead of stalling.

The obstacle envs ship a **slalom** layout (obstacles alternating sides near the centerline, goal just
past the last one) so the box must weave past every obstacle. Because the box moves at ~0.15–0.2 m/s,
the episode needs to be long enough to traverse the course — `run_play_cbf.sh` defaults to
`--steps 1500` (~5 m). Validate the geometry quickly with `cbf_standalone_demo.py` (no Isaac).

## Tuning

`CBFSafeVelocityHeightCommandCfg` exposes: `obstacles` (env-local `(x, y, radius)`), `box_radius` /
`agent_radius` / `body_radii` (avoidance radii), `cbf_alpha` (larger ⇒ later/sharper intervention),
`cbf_margin` (extra buffer), `cbf_qp_iters`, and the guidance knobs above. Obstacle prims are declared
as `obstacle_0..3` slots in `ObstacleSceneCfg`; keep that layout in sync with the `OBSTACLES` list and
`GOAL_XY` used by the command.

## Extending to the consensus task

The current term subclasses `CategoricalVelocityHeightCommand` (the finetuning task). For the
divergent-belief consensus task, subclass `ConsensusVelocityHeightCommand` the same way: filter the
shared mean `vel_command_b` (the per-agent beliefs `mean ± divergence` then ride on top of the safe
mean automatically). The CBF core in `cbf_qp.py` is unchanged.
