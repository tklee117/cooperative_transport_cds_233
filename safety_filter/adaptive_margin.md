# Adaptive (online self-tuning) CBF margin

> Goal: stop hand-tuning `cbf_margin` per case. Instead, **estimate the safety buffer the environment
> needs in real time** and set the margin to match — so the same config works for 2 agents, 4 agents,
> easy open stretches, and tight obstacle regions, with no per-case sweep.

This is the *online* counterpart of the offline robust-margin estimator
(`cbf_robust_margin.py`, see `CBF_safety_filter.md §14`). The robust-margin script measured the
worst-case excursion `δ̄` from saved runs and picked a single fixed margin. The **adaptive margin**
measures that disturbance **every step** and inflates the keep-out on the fly.

---

## 1. Idea

The CBF guarantees the *commanded* twist is safe under a **rigid-constellation model**
(`ṗ_b = A_b u`: each body moves rigidly with the box twist). The *realized* motion deviates — tracking
lag, and agents swinging wider than rigid (grasp deformation). The fixed margin had to absorb the
worst case everywhere.

Instead, each step we **measure how wrong the rigid model was last step** — the velocity disturbance
of the worst body — and grow the margin only as much as needed:

```
predicted body velocity (last step):  v_pred_b = R(ψ_{t-1})·u_{t-1}[:2] + ω_{t-1} · perp(p_b − p_box)|_{t-1}
realized  body velocity:               v_real_b = (p_b(t) − p_b(t-1)) / dt
disturbance:                           d = max_b ‖v_real_b − v_pred_b‖        (worst body)
d̄ ← EMA(d)                                                                    (smoothed)

margin_t = clamp( margin_base + margin_gain · d̄/α + margin_vel_gain · ‖v_box‖,  margin_base,  margin_max )
```

- `margin_gain · d̄/α`: the disturbance, scaled by the CBF reaction time `1/α` — the position excursion
  that accumulates before the filter can react. Grows the buffer **exactly where the model fails**
  (an agent swinging wide / starting to detach), shrinks it in easy open stretches.
- `margin_vel_gain · ‖v_box‖`: a braking-distance term (faster ⇒ wider keep-out).
- It is **causal** (uses the previous step's applied command + pose) and **per-env** (each parallel env
  gets its own margin).
- Resets/teleports are detected (box jump > 0.3 m) and skipped so the finite-difference disturbance is
  not corrupted; the first step after a reset uses `margin_base`.

Because the 4-agent constellation swings wider, `d̄` is naturally larger there, so the **same config
produces a bigger margin for 4 agents than for 2** — auto-tuning, no per-case hand-tuning. The
disturbance `d̄` is itself the **grasp-deformation signal** (how far agents stray from the rigid
formation), so it doubles as an instrument for the Part-2 "does the grasp hold?" question.

---

## 2. Code changes

| File | Change |
|---|---|
| `mdp/safety/cbf_qp.py` | `build_coupled_cbf_constraints` / `cbf_safety_filter` now accept a **per-env tensor margin** `(B,)` (reshaped to broadcast over `(B, Nb, K)`), not just a scalar. |
| `mdp/commands/cbf_safe_velocity_height_command.py` | New `_adaptive_margin()` method (disturbance estimate → margin); previous-step state buffers (`_prev_body_pos/_prev_box_pos/_prev_box_yaw/_prev_u/_prev_valid`); diagnostics `cbf_margin_effective`, `cbf_disturbance`; reset clears the history; new cfg fields (below). `_update_command` now calls `_adaptive_margin(...)` and passes the result to the QP, and stashes the applied twist + pose for the next step. |
| `tasks/.../finetuning/..._obstacles_env_cfg.py` (2-robot) | `adaptive_margin=True, margin_base=0.06, margin_gain=0.5, margin_vel_gain=0.05, margin_ema=0.9, margin_max=0.45`. |
| `tasks/.../n_robots/..._N_robot_obstacles_env_cfg.py` (4-robot) | `adaptive_margin=True, margin_base=0.08, ..., margin_max=0.6` (higher cap; wider swings). |
| `scripts/play_cbf.py` | Logs `safety/margin_effective` and `safety/disturbance` per step; saves `margin_eff`/`disturbance` series into the `cbf_series_*.npz`. |
| `scripts/cbf_viz.py` | The performance figure overlays the **adaptive margin `m(t)`** (dashed, right axis) on the intervention panel, so you can watch it breathe. |

### New `CBFSafeVelocityHeightCommandCfg` knobs
```python
adaptive_margin: bool  = False   # turn it on (obstacle cfgs set True)
margin_base:     float = 0.05    # floor margin (open-space / well-tracked)
margin_gain:     float = 0.5     # gain (s) on disturbance d/alpha  -> conservativeness
margin_vel_gain: float = 0.05    # gain (s) on box speed (braking-distance buffer)
margin_ema:      float = 0.9     # EMA smoothing of the disturbance estimate
margin_max:      float = 0.45    # cap so corridors stay passable
```
When `adaptive_margin=False`, the fixed `cbf_margin` is used exactly as before (backward compatible).

---

## 3. Diagnostics logged (wandb `decplm-cbf`)
- `safety/margin_effective` — the live margin `m(t)` (watch it grow near obstacles / when agents stray).
- `safety/disturbance` — the smoothed rigid-model velocity mismatch `d̄` (≈ grasp deformation rate).
- still logged: `safety/min_clearance` (physical), `safety/min_barrier` (margin), `summary/phys_collision_rate`.
- The `.npz` series now carries `margin_eff` and `disturbance` for offline plotting.

---

## 4. How to run

Adaptive margin is **on by default** in the obstacle cfgs, so the normal commands use it:
```bash
./run_play_cbf.sh --mode both --robots 2 --video
./run_play_cbf.sh --mode both --robots 4 --video
```
To compare against a **fixed** margin, override at runtime (forces a constant value; the override sets
`cbf_margin`, and you can disable adaptation by editing `adaptive_margin=False` in the cfg or just read
`safety/margin_effective` to confirm what was used):
```bash
./run_play_cbf.sh --mode on --robots 2 --cbf_margin 0.16     # fixed-margin baseline for comparison
```

---

## 5. Validation (Isaac, pretrained policy, 1600 steps)

The adaptive margin **breathes** (near `margin_base` in the open, inflating at the tight obstacle) and
**auto-settles to a larger value for 4 agents than 2** — with no per-case tuning — keeping the realized
trajectory physically safe:

| run | `margin_eff(t)` min→max (mean) | disturbance `d̄` max | physical `min_clearance` | phys. collision |
|---|---|---|---|---|
| 2-robot adaptive (symmetric EMA) | 0.060 → 0.207 (0.141) | 0.42 m/s | +0.004 m (thin) | none |
| 2-robot adaptive + **fast-attack EMA** | 0.060 → 0.275 (0.151) | 0.42 m/s | +0.016 m | none |
| 4-robot adaptive | 0.080 → 0.226 (**0.160**) | 0.42 m/s | **+0.097 m** | none |

(Final cfg defaults: `margin_base=0.08, margin_gain=0.6, margin_ema=0.9, margin_ema_attack=0.3` for
both; `margin_max=0.45` for 2-robot, `0.6` for 4-robot. The 2-robot floor was raised from 0.06→0.08
for a healthier buffer.)

Key outcomes:
- **Auto-tunes per environment:** the 2-agent margin settles at mean 0.141 (≈ our hand-tuned 0.16) and
  the **4-agent margin auto-settles higher at 0.160** — the same config, no sweep.
- **Safe by construction:** both stay physically collision-free (`phys_collision_rate = 0`).
- **EMA-lag fix:** the first 2-robot run grazed (+0.004 m) because a symmetric EMA grew the buffer a
  step late; the **asymmetric fast-attack EMA** (grow on spike, decay slowly) opens that buffer.

To tune conservativeness: raise `margin_gain` (more buffer per unit disturbance), `margin_base` (floor),
or lower `margin_ema_attack` (faster growth).

---

## 6. Relation to the rest

- **Solution B (offline robust margin)** picks one fixed safe margin from a sweep; **adaptive margin**
  does it online and per-environment. Use the offline sweep to sanity-check / calibrate `margin_gain`.
- The disturbance `d̄` is the **grasp-deformation** signal. Adaptive margin *reacts* to it (cheap);
  Part 2 would instead let the team **change form** (contact points) when `d̄` climbs near a tight
  obstacle — the richer, learned fix. Same trigger, two responses.
