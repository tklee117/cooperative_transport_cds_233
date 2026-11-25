# Grasp-morphing prototype #1 — "tuck the form near tight obstacles" (Idea 2a)

> Question: instead of holding a fixed grasp form the whole episode, can the team **change form**
> (contact configuration) when it meets a tight obstacle region — and does the frozen, 2-agent-trained
> decPLM policy *re-grasp* the new form, or does it lose the grasp?

This is the cheapest, no-retrain probe of Part 2's "adaptive form" idea. It reuses the existing
contact-frame machinery: the policy's grasp setpoint is a runtime-writable frame offset, so we move it.

---

## 1. Why this is even possible without retraining

Two facts in decPLM make a runtime morph plausible:

1. **The grasp setpoint is a runtime buffer.** Each agent's contact target is
   `FrameTransformer._target_frame_offset_pos` (the `box_contact_frame` / `box_operation_frame` offsets
   on the box). An existing event (`randomize_frame_offsets`) already writes it at reset; we write it
   every step.
2. **The policy observes that frame.** The actor obs `box_pos_contact_rel` is the box pose **relative
   to `box_contact_frame`** (via `robot_base_frame{suffix}`). So moving the offset changes what the
   policy sees → it is *asked* to re-grasp the moved setpoint. And training **randomizes the contact
   position per episode**, so decPLM already generalizes across a distribution of *static* forms.

**Caveat discovered (important):** there is a temporal-mask variant of this obs
(`pos_rel_temporal_mask_v5`) that **stops updating `box_pos_contact_rel` after ~2 s** (it freezes the
grasp perception once acquired). If the policy group uses that variant, a mid-episode morph may **not**
reach the policy — it would keep tracking the *old* grasp. Whether the morph propagates is therefore an
empirical question, not a given. (Also: the operating *distance* is fixed in training while the contact
*position* is randomized — so a pure radial "tuck" is partly out-of-distribution.)

---

## 2. The morph (v1: radial tuck)

As obstacles get close, scale the box-frame contact/operation offsets' **XY toward the box center**
(keep the grasp height), shrinking the constellation footprint:

```
frac  = clamp((min_clearance − morph_clearance_lo) / (morph_clearance_hi − morph_clearance_lo), 0, 1)
scale = morph_min_scale + (1 − morph_min_scale)·frac        # 1 when clear, morph_min_scale when tight
offset_xy ← scale · offset_xy(default)                       # per frame, per env; Z unchanged
```

`min_clearance` is the margin-free physical clearance (reused from the CBF). The CBF already reads the
agents' **actual** positions, so as they tuck in, the keep-out automatically reflects the smaller
footprint — no CBF change needed.

---

## 3. Implementation (files changed)

| File | Change |
|---|---|
| `mdp/commands/cbf_safe_velocity_height_command.py` | `_morph_constellation(min_clearance)`: lazily captures the default contact/operation offsets per episode, then each step writes `scale·offset_xy` into `robot_base_frame{suffix}` and `arm_contactpad_frame{suffix}` `_target_frame_offset_pos`. Reset restores the offsets + re-captures. New diagnostic `cbf_morph_scale`. New cfg knobs: `morph_constellation, morph_min_scale, morph_clearance_lo, morph_clearance_hi`. |
| `scripts/play_cbf.py` | `--morph` flag (sets `morph_constellation=True`); logs `safety/morph_scale`; tags morph runs `..._morph`. |

It is **off by default** (the validated obstacle envs are unchanged); enable per-run with `--morph`.

### Run it
```bash
# compare same scene with / without the morph:
python .../scripts/play_cbf.py --task ...Obstacles-Play-v0 --cbf on        --headless --steps_per_episode 1600 ...   # no morph
python .../scripts/play_cbf.py --task ...Obstacles-Play-v0 --cbf on --morph --headless --steps_per_episode 1600 ...   # morph
```

### Knobs
`morph_min_scale` (how hard to tuck; 0.7 = 30%), `morph_clearance_lo/hi` (proximity band that triggers
it). `safety/morph_scale` in wandb shows when/how much it tucks (1.0 = full form, → `morph_min_scale`
near obstacles).

---

## 4. Results — 2-robot, `--morph` vs no-morph (1600 steps, CBF on + adaptive margin)

| run | reaches goal? | reward | agent→box dist (footprint) | physical `min_clearance` |
|---|---|---|---|---|
| no morph | yes | 343 | ~0.69 m | +0.016 m |
| **morph** | **yes** (box → 4.75, 0.51) | 327 | 0.69 → **0.60 m** (tucked ~12%) | **+0.112 m** |

**It works — positive result.** Three things happened:
1. **decPLM re-grasped the tucked form.** Reward held (327 vs 343), the box still reached the goal, and
   the agents stayed on the box (agent→box distance ~0.6 m, no drift). The temporal-mask freeze did
   *not* block it, and the (partly-OOD) tuck did not destabilize the grasp.
2. **The footprint actually shrank.** Agent-to-box distance fell 0.69 → 0.60 m as the morph engaged
   near the obstacles — the tuck is real and causal (the policy followed the moved setpoint inward).
3. **Safety improved ~7×.** `min_clearance` went +0.016 → **+0.112 m**: the smaller form threads the
   tight gap with far more room, at a small reward cost.

Caveats / notes:
- One stochastic run each (morph vs no-morph); the effect (footprint shrink + clearance gain) is
  consistent and causal, but a few-seed repeat would firm up the numbers.
- The radial tuck scales the *operating standoff* too, which is fixed in training (mildly OOD); decPLM
  tolerated it here, but a more in-distribution morph (moving the contact *position* around the box)
  is the natural v2.
- **Implementation cost:** writing `_target_frame_offset_pos` every step is slow (it re-triggers the
  FrameTransformer). Frame objects are now cached in `_morph_constellation` to reduce per-step
  overhead; still, the morph run is noticeably slower than a plain run.
- `morph_scale` is now also saved into the `cbf_series_*.npz` (it was wandb-only in the first run).

---

## 5. Interpretation & next steps

decPLM **can re-form mid-episode without retraining** — at least for a gentle radial tuck — and doing so
**improves obstacle clearance**. So form-morphing is viable as a cheap add-on to the CBF, and is worth
pushing further:

1. **v2 morph — drive *which* form by geometry.** Instead of a symmetric tuck, shift/rotate the contact
   *positions* (in-distribution) so agents move to the side *away* from the obstacle, not just inward.
2. **N-robot.** Repeat for 4 agents (where detachment was the bigger problem) — the tuck should help the
   footprint most there.
3. **Couple to the disturbance signal.** Trigger the morph on the same `d̄` the adaptive margin uses
   ("when the grasp deforms / we're about to need a big deflection, change form") — unifying #1 (margin)
   and #2 (form) under one signal.
4. **Part 2 proper.** This frozen-policy result is the floor; a learned high-level that *chooses* the
   form (and a non-frozen grasp obs) is the ceiling. The prototype shows the floor is already useful.

---

## 6. v2 (directional) and v3 (topological re-grasp) — and where the frozen policy gives out

After v1, two richer morphs were built and run on the obstacle course (CBF on + adaptive margin,
1500-step episodes, logged to wandb `decplm-cbf`, with the Isaac video): **v2 "directional"** (each agent
tucks by *its own* clearance, not the worst) and **v3 "regrasp"** (a true *topological* re-grasp — line the
agents up fore/aft along the passage axis, with a speed-gated creep). The headline: **v2/v3 do not help,
and v3 exposes the real ceiling of the frozen policy.**

### 6.1 Results matrix

| run | box traversal | lateral envelope ‖Δy‖ (half) | min_clearance | reward | outcome |
|---|---|---|---|---|---|
| **2ag** no-morph | reaches goal | — | +0.016 | 343 | ✅ |
| **2ag tuck (v1)** | x→**5.03** (goal 4.8) | 0.26 | +0.107 | 299 | ✅ |
| **2ag directional (v2)** | x→**0.77** (1st obs at 1.8) | 0.26 | n/a | 294 | ❌ **deadlocked at start** |
| **4ag tuck (v1)** | x→**3.41** | 0.69 | +0.138 | 649 | ⚠️ jams at obs 2 (4.6) |
| **4ag directional (v2)** | x→**3.38** | **0.73** (worse) | +0.130 | 747 | ⚠️ jams at obs 2 |
| **4ag regrasp (v3, late)** | x→**3.5** | **0.76** | +0.107 | 566 | ⚠️ jams at obs 2 |
| **4ag regrasp (v3, early/strong)** | x→**3.05**, end 2.54 | **0.81** (worst) | +0.112 | 576 | ❌ crawls, stalls between |

The radial morph (v1/v2) only **scales** the contacts — it preserves the grasp *topology* (every agent
stays on its face). For 4 agents the lateral envelope is therefore stuck at ~0.7 m either way; v2's
per-agent tuck is actually a wash-to-worse (it widened the envelope to 0.73). Both jam at obstacle 2.

### 6.2 v3 design — topological re-grasp with speed-gated creep

v3 (`morph_mode="regrasp"`) does what v1/v2 cannot: it re-assigns the contacts to a **fore/aft line**.
Agents are ranked by their current lateral position and slewed onto along-axis stations (so the order is
monotone → no two agents swap sides / collide), with the cross-passage component compressed toward the
centerline. It is **not a snap**: each step the written offset creeps toward the target at a bounded rate,
and the **box speed is throttled** by the reconfiguration residual + the grasp disturbance `d̄` (so the box
crawls/pauses while agents migrate, resumes once the grasp settles — "stopping" emerges from the gate, not
a hard FSM). Done *before* the QP, so the CBF still filters the throttled nominal (provably safe). New
diagnostics: `safety/regrasp_gate`, and `morph_scale` reads the commanded lateral compression. Knobs:
`regrasp_{engage_dist,full_dist,cross_scale,spacing,slew_rate,gate_recon,gate_dist,gate_min}`.

### 6.3 The freeze investigation (a corrected hypothesis)

First read of the v3 failure blamed the **temporal-mask** grasp obs: in *training*, `box_pos/quat_contact_rel`
freeze after 2 s (`update_rate_time_window` → `prob 0`), which would make a mid-episode re-grasp invisible.
A `--unfreeze_grasp` override was added to test this. But **dumping the live `env_cfg.observations` at play
time showed the play config does NOT use the mask** — it uses plain live `pos_rel` / `quat_rel`. So the
policy perceives the moving contacts every step; `--unfreeze` is a no-op here (it correctly reports "0 terms,
contacts perceived live"). **Perception was never the blocker.** The v3 negative result is real.

### 6.4 Verdict — the frozen policy will not re-grasp

With live perception confirmed, v3 was retuned to reconfigure **early, in open space, before the pinch**
(engage 1.6→2.8 m, full by 1.5 m, stronger lateral pull-in, faster creep) — removing the "box deadlocked
before the creep finished" confound. The agents **still did not line up**:

- lateral envelope **grew** (0.76 → **0.81 m**, vs tuck 0.69) instead of shrinking;
- the constellation aspect `alongX/latY = 0.91` stayed **lateral-dominant** (a lined-up form would be > 1);
- the agents partially followed the fore/aft *stagger* (along-extent grew, peak 1.16 m) but **refused the
  lateral pull-in** — they held their trained ~0.8 m side-grasp standoff;
- net the box crawled (speed gate) into an even **earlier** stall (x→3.05).

**Interpretation.** The frozen, 2-agent-trained decPLM treats each contact as a *held grasp it tracks
continuously*; it will slide a contact a little along the box but **will not pull laterally across the box
toward the centerline**, because doing so conflicts with maintaining its side grasp. A *topological*
re-grasp requires **breaking and re-making contact** (a discrete release → reposition → re-acquire), and a
continuous-tracking policy will not cross that discontinuity on its own. This confirms the original
intuition: you **cannot** morph through the topology change smoothly with the frozen policy — you must
**stop, disengage, reconfigure, and continue**, or retrain.

This is the clean boundary of the frozen-policy probe:
- **radial form change (v1)** — *within* topology, small, in-distribution → **works** (footprint −12%, clearance ↑).
- **topological re-grasp (v3)** — *crosses* topology, requires release/re-acquire → **frozen policy refuses.**

### 6.5 Revised next steps

1. **Option B — the re-grasp maneuver (stop-disengage-reconfigure-continue).** Implement an explicit FSM:
   halt the box at the gap → command a release (lower/retract the arm contact) → reposition bases to the
   narrow config → re-acquire (the same reach→touch→lift the policy *is* trained for, just mid-course) →
   resume. Each phase is in-distribution; the discontinuity is handled explicitly rather than asked of a
   continuous morph. This is the natural test of whether decPLM can re-acquire a *new* topology.
2. **Part 2 (retrain).** Make the form a learned action: a high-level that *chooses* the grasp topology and
   a policy trained to release/re-acquire on command (and to hold fore/aft grasps). v3's machinery (target
   topology + speed gate) becomes the action interface; the floor here says the continuous morph alone is
   not enough — the policy must be trained to cross topologies.
3. **Run commands (all log to wandb `decplm-cbf` with the Isaac video):**
   ```bash
   conda activate isaaclab && cd ~/Academics/Decentralized-Pinch-Lift-Move
   ./run_play_cbf.sh --mode on --robots 2 --morph --morph_mode tuck        --video --device cuda:0
   ./run_play_cbf.sh --mode on --robots 2 --morph --morph_mode directional --video --device cuda:0
   ./run_play_cbf.sh --mode on --robots 4 --morph --morph_mode tuck        --video --device cuda:1
   ./run_play_cbf.sh --mode on --robots 4 --morph --morph_mode directional --video --device cuda:1
   ./run_play_cbf.sh --mode on --robots 4 --morph --morph_mode regrasp     --video --device cuda:1   # v3
   ```
   (`--unfreeze` exists but is a no-op at play; the play obs is already live.)

---

## 7. v4 — the explicit re-grasp maneuver (stop-disengage-reconfigure-continue) — and the real ceiling

v3 showed the frozen policy won't slide a *held* grasp laterally. The hypothesis for v4: break the grasp
first. `morph_mode="regrasp_fsm"` runs a timed state machine in the box command term:
**CRUISE → HALT** (box speed gate → 0) **→ RELEASE** (raise the contact-frame Z to retract the arm off the
box) **→ REPOSITION** (slew the released contact setpoints to the fore/aft line) **→ REACQUIRE** (lower them
back to re-grasp at the new stations) **→ RESUME** (hold the narrow form, drive through the corridor). It
arms only after the box has lifted (`fsm_arm_after_steps`) and fires once on entry to the obstacle field
(both obstacles share one passage, so the narrow form is held, not restored). New diagnostics:
`safety/fsm_state`, and the phase index is saved to the npz.

**It runs correctly** — the phases fire in order with the right durations (verified: CRUISE→HALT@314 →
RELEASE@338 → REPOS@378 → REACQ@468 → RESUME@528). **But the policy does not reconfigure.** Tracking the
agents' positions *relative to the box* through the whole maneuver:

| phase | agent rel-XY (4 agents) | lateral env ‖Δy‖ |
|---|---|---|
| pre-trigger | (−0.75,−0.09) (0.13,−0.74) (0.77,0.10) (−0.26,0.77) | 0.82 |
| end REPOSITION | (−0.77,−0.16) (0.24,−0.74) (0.73,0.21) (−0.38,0.70) | — |
| end (RESUME) | (−0.69,−0.32) (0.20,−0.75) (0.73,0.28) (−0.45,0.63) | 0.77 |

The agent **bases moved only 6–13 cm** over the 90-step REPOSITION; the "surround" formation (one agent per
box face) is intact start to finish. latY went 0.82 → 0.77 (noise). Aspect stays 0.97 (not lined up).

**Root cause — the frozen policy's base formation is locked after grasp.** decPLM was trained
reach→touch→lift→move-and-**hold**; it has **no release-and-re-walk behavior**. Moving the contact setpoint
steers the *arm's* fine grasp (the arm did respond — the box rose ~+0.04 m when we raised the contact Z,
which incidentally means the Z-lift *lifts* rather than cleanly *releases* — a mechanism caveat), but the
**base** does not relocate around the box once it is holding. So:

- **v3 (continuous creep):** base won't slide laterally while holding → no reconfiguration.
- **v4 (explicit halt + release + reposition):** base still won't walk to a new station → no reconfiguration.

Both routes hit the same wall. **The answer to "smooth re-grasp vs stop-disengage-reconfigure" is: neither
works with the frozen policy** — the base formation cannot change mid-episode at all, because re-forming
requires a release-and-re-walk skill decPLM never learned. (Caveat: the v4 release was imperfect — raising
contact-Z lifts rather than ungrasps; a cleaner release that retracts the base outward might disengage
better. But the dominant, consistent signal across v3 and v4 is that the *base* formation is frozen.)

**This is the definitive ceiling of the frozen-policy probe**, and it makes the Part-2 case concrete:
re-grasp must be a **trained** capability — the policy needs (a) a release / re-acquire skill and (b) the
base formation (grasp topology) as a controllable action. v3/v4's machinery (fore/aft target geometry,
speed gate, phase FSM) is the ready-made *action interface* for that retraining; what's missing is a policy
trained to use it.

### 7.1 Run v4
```bash
conda activate isaaclab && cd ~/Academics/Decentralized-Pinch-Lift-Move
./run_play_cbf.sh --mode on --robots 4 --morph --morph_mode regrasp_fsm --video --device cuda:1
```
Knobs (cfg): `fsm_trigger_dist, fsm_arm_after_steps, fsm_{halt,release,repos,reacq}_steps,
fsm_release_lift, fsm_hold_gate` (+ the `regrasp_{cross_scale,spacing,slew_rate}` target geometry).

