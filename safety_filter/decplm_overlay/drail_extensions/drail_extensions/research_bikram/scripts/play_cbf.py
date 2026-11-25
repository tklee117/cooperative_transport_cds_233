# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Play decPLM in the obstacle stress-test environment, with the coupled-CBF safety filter.

This is the eval driver for Part 1 (stress-testing decPLM with obstacles). It loads the pretrained
finetuning policy unchanged and runs it in the obstacle scene, optionally wrapping the box command in
the CBF safety filter (``--cbf on``) or letting it pass through unfiltered (``--cbf off``, the
baseline that exposes the open-space policy's failure mode).

Beyond the usual reward/tracking metrics it logs **safety diagnostics** read straight from the CBF
command term:

* ``safety/min_barrier``      -- smallest CBF value over {box, agent 0, agent 1} x obstacles this step
                                 (``< 0`` => a body is inside a keep-out disk, i.e. collision).
* ``safety/collision_frac``   -- fraction of steps with any penetration.
* ``safety/intervention``     -- ``||u_safe - u_nom||``, how hard the filter is correcting.
* ``safety/active_frac``      -- fraction of steps the filter measurably intervened.
* ``grasp/contact_frac``      -- fraction of steps both contact pads stay loaded (grasp maintained
                                 while the trajectory is deflected -- the agent-2 question).

Example::

    # baseline: open-space policy drives straight into the obstacles
    python play_cbf.py --cbf off --cmd_vel_x 0.4 --num_envs 1 --headless --video \
        --task Go2-D1-MARL-Box-Reach-Touch-Lift-and-Move-Finetuning-Obstacles-Play-v0 \
        agent.raw_checkpoint=<path-to>/model_latest.pt

    # safety-filtered: the QP deflects the constellation around the obstacles
    python play_cbf.py --cbf on  --cmd_vel_x 0.4 ...   # same as above with the filter on
"""

import argparse
import importlib
import os
import sys
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch._dynamo  # pre-import before Isaac Sim's typing_extensions patch breaks it
from isaaclab.app import AppLauncher

import wandb

# local imports
import cbf_viz  # isort: skip
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play decPLM in obstacles with a CBF safety filter.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric / use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Go2-D1-MARL-Box-Reach-Touch-Lift-and-Move-Finetuning-Obstacles-Play-v0",
                    help="Name of the task (defaults to the obstacle stress-test play env).")
parser.add_argument("--use_pretrained_checkpoint", action="store_true",
                    help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--eval_name", type=str, default="cbf-eval", help="WandB run name.")
parser.add_argument("--cbf", type=str, default="on", choices=["on", "off"],
                    help="Enable ('on') or disable ('off') the CBF safety filter. 'off' is the baseline.")
parser.add_argument("--cbf_margin", type=float, default=None,
                    help="Override the CBF keep-out margin (m) without editing the cfg. For margin sweeps.")
parser.add_argument("--morph", action="store_true",
                    help="Enable the constellation-morph prototype (Idea 2a): change the grasp form near obstacles.")
parser.add_argument("--morph_mode", type=str, default="tuck",
                    choices=["tuck", "directional", "regrasp", "regrasp_fsm"],
                    help="Morph mode: 'tuck' (v1, symmetric), 'directional' (v2, per-agent), "
                         "'regrasp' (v3, fore/aft creep), or 'regrasp_fsm' (v4, stop-disengage-"
                         "reconfigure-continue maneuver).")
parser.add_argument("--unfreeze_grasp", action="store_true",
                    help="Override the temporal-mask freeze so the policy keeps perceiving the contact "
                         "frames past 2 s (required for a mid-episode re-grasp to reach the policy).")
parser.add_argument("--num_episodes", type=int, default=5, help="Episodes to roll out.")
parser.add_argument("--steps_per_episode", type=int, default=600, help="Steps per episode.")
parser.add_argument("--viz_episodes", type=int, default=1,
                    help="Number of (first) episodes to render trajectory figures + GIFs for.")
parser.add_argument("--cmd_vel_x", type=float, default=0.4, help="Fix box lin_vel_x command (m/s). Omit for random.")
parser.add_argument("--cmd_vel_y", type=float, default=None, help="Fix box lin_vel_y command (m/s). Omit for random.")
parser.add_argument("--cmd_ang_vel_z", type=float, default=None, help="Fix box ang_vel_z command (rad/s).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import omni.log  # noqa: E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import drail_extensions  # noqa: F401, E402
import drail_extensions.research_bikram  # noqa: F401, E402
from drail_extensions.core.isaaclab_rsl_rl.utils import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # --- toggle the CBF safety filter (off => baseline open-space policy, diagnostics still logged) ---
    enable_cbf = args_cli.cbf == "on"
    print(f"[INFO] box_command cfg type: {type(env_cfg.commands.box_command).__name__}")
    if hasattr(env_cfg.commands.box_command, "enable_cbf"):
        env_cfg.commands.box_command.enable_cbf = enable_cbf
        print(f"[INFO] CBF safety filter: {'ENABLED' if enable_cbf else 'DISABLED (baseline)'}")
        if args_cli.cbf_margin is not None:
            env_cfg.commands.box_command.cbf_margin = args_cli.cbf_margin
            print(f"[INFO] CBF margin override: {args_cli.cbf_margin} m")
        if args_cli.morph:
            env_cfg.commands.box_command.morph_constellation = True
            env_cfg.commands.box_command.morph_mode = args_cli.morph_mode
            print(f"[INFO] Constellation morph (Idea 2a): ENABLED (mode={args_cli.morph_mode})")
    else:
        print("[WARN] box_command has no 'enable_cbf' field; is this the obstacles task? Continuing.")

    # Unfreeze the grasp perception. The trained policy freezes box_pos/quat_contact_rel after ~2 s
    # (update_rate_time_window prob 0 past 2 s), so a mid-episode morph/re-grasp is INVISIBLE to it.
    # For the re-grasp probe we override every such window to always-update so the policy can actually
    # perceive the moved contact setpoint (note: this is itself OOD -- the net was trained with the
    # freeze -- but it is the only way to test whether a *perceivable* re-grasp is trackable).
    if args_cli.unfreeze_grasp:
        n_unfrozen = 0
        hit = []
        always = [{"start": 0.0, "end": float("inf"), "prob": 1.0}]

        def _terms_of(grp):
            # ObsGroup terms are set via setattr -> live in __dict__; fall back to dir() just in case.
            seen = {}
            for k, v in list(vars(grp).items()):
                seen[k] = v
            for k in dir(grp):
                if not k.startswith("__") and k not in seen:
                    try:
                        seen[k] = getattr(grp, k)
                    except Exception:
                        pass
            return seen

        groups = _terms_of(env_cfg.observations)  # group name -> ObsGroup
        for grp_name, grp in groups.items():
            if not hasattr(grp, "__dict__") and not hasattr(type(grp), "__annotations__"):
                continue
            for term_name, term in _terms_of(grp).items():
                params = getattr(term, "params", None)
                if isinstance(params, dict) and "update_rate_time_window" in params:
                    params["update_rate_time_window"] = always
                    n_unfrozen += 1
                    hit.append(f"{grp_name}.{term_name}")
        print(f"[INFO] Grasp perception UNFROZEN: {n_unfrozen} obs term(s) set to always-update.")
        print(f"[INFO] unfrozen terms: {hit[:12]}{' ...' if len(hit) > 12 else ''}")
        if n_unfrozen == 0:
            # The *play* obs config uses live pos_rel/quat_rel (the temporal-mask freeze is a TRAINING
            # config); so there is nothing to unfreeze and the policy already perceives the morph live.
            print("[INFO] no temporal-mask windows in the play obs (contacts are perceived live); --unfreeze is a no-op here.")
    # effective margin + agent count, used to make wandb run names / saved filenames unique per sweep.
    # adaptive runs tag as "adaptive" (the margin varies) so they don't collide with fixed-margin runs.
    cbf_margin_val = getattr(env_cfg.commands.box_command, "cbf_margin", None)
    is_adaptive = bool(getattr(env_cfg.commands.box_command, "adaptive_margin", False))
    n_agents = len(getattr(env_cfg.commands.box_command, "agent_asset_names", []) or [])
    if is_adaptive:
        margin_tag = "adaptive"
    elif cbf_margin_val is not None:
        margin_tag = f"m{cbf_margin_val:g}"
    else:
        margin_tag = "mNA"
    if args_cli.morph:
        margin_tag += f"_morph-{args_cli.morph_mode}"
    if args_cli.unfreeze_grasp:
        margin_tag += "_unfrozen"

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    elif agent_cfg.raw_checkpoint:
        resume_path = retrieve_file_path(agent_cfg.raw_checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # fix command direction if requested (drives the box toward the obstacles)
    if any(v is not None for v in [args_cli.cmd_vel_x, args_cli.cmd_vel_y, args_cli.cmd_ang_vel_z]):
        for key, r in env_cfg.commands.box_command.ranges.items():
            if key == "clip_range":
                continue
            if args_cli.cmd_vel_x is not None:
                r.lin_vel_x = (args_cli.cmd_vel_x, args_cli.cmd_vel_x)
            if args_cli.cmd_vel_y is not None:
                r.lin_vel_y = (args_cli.cmd_vel_y, args_cli.cmd_vel_y)
            if args_cli.cmd_ang_vel_z is not None:
                r.ang_vel_z = (args_cli.cmd_ang_vel_z, args_cli.cmd_ang_vel_z)
        print(f"[INFO] Fixed command: vel_x={args_cli.cmd_vel_x}, vel_y={args_cli.cmd_vel_y}, ang_z={args_cli.cmd_ang_vel_z}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    video_recorder = None
    video_dir = None
    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", f"play_cbf_{args_cli.cbf}_{n_agents}ag_{margin_tag}")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        # keep a direct handle: gymnasium's RecordVideo only flushes the mp4 when the frame count
        # *exceeds* video_length or on close(). A single 600-step episode with video_length=600 never
        # trips that, so we stop_recording() explicitly after the rollout to flush before logging.
        video_recorder = env

    if hasattr(env_cfg, "wrapper_class"):
        env = env_cfg.wrapper_class(env, debug=False)
    else:
        env = RslRlVecEnvWrapper(env)

    runner_class_path = getattr(agent_cfg, "class_name", "rsl_rl.runners.OnPolicyRunner")
    module_path, class_name = runner_class_path.rsplit(".", 1)
    runner_class = getattr(importlib.import_module(module_path), class_name)
    if not hasattr(agent_cfg, "class_name"):
        agent_cfg.class_name = "rsl_rl.runners.OnPolicyRunner"

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    try:
        # strict=False: load all matching (actor) weights and skip any mismatch. The N-robot env uses
        # a centralized critic whose input scales with agent count, so its critic shape differs from
        # the 2-robot checkpoint -- but only the actor is needed for inference.
        try:
            ppo_runner.load(resume_path, strict=False)
        except TypeError:
            ppo_runner.load(resume_path)  # runner.load without a strict kwarg
        print(f"[INFO] Loaded checkpoint. Learning iteration: {ppo_runner.current_learning_iteration}")
    except Exception as e:
        omni.log.error(f"Failed to load checkpoint from: {resume_path}. Error: {e}")
        omni.log.error(traceback.format_exc())

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # --- handles for tracking + safety diagnostics ---
    cmd_term = env.unwrapped.command_manager.get_term("box_command")
    has_cbf = hasattr(cmd_term, "cbf_min_h")

    def tracking_errors():
        box = cmd_term.robot
        cmd = cmd_term.vel_command_b
        return {
            "error_vel_xy": torch.norm(cmd[:, :2] - box.data.root_lin_vel_b[:, :2], dim=-1).mean().item(),
            "error_vel_yaw": torch.abs(cmd[:, 2] - box.data.root_ang_vel_b[:, 2]).mean().item(),
            "error_height": torch.abs(cmd[:, 3] - box.data.root_pos_w[:, 2]).mean().item(),
        }

    # include cbf + margin + agent count in the run name so a margin sweep produces distinct,
    # filterable wandb runs (e.g. "obstacles-on-4ag-m0.20").
    run_name = f"{args_cli.eval_name}-{args_cli.cbf}-{n_agents}ag-{margin_tag}"
    wandb.init(project="decplm-cbf", name=run_name, config={
        "cbf": args_cli.cbf,
        "cbf_margin": cbf_margin_val,
        "n_agents": n_agents,
        "checkpoint": resume_path,
        "cmd_vel_x": args_cli.cmd_vel_x,
        "task": args_cli.task,
        "num_episodes": args_cli.num_episodes,
        "steps_per_episode": args_cli.steps_per_episode,
    })

    # geometry/timing for the trajectory plots + GIF (env 0, env-local frame).
    env_origins = env.unwrapped.scene.env_origins  # (B, 3)
    dt = env.unwrapped.step_dt
    static_geom = None
    if has_cbf and cmd_term._num_obstacles > 0:
        static_geom = {
            "obstacles": (cmd_term._obstacle_xy[0] - env_origins[0, :2]).cpu().numpy(),
            "obstacle_r": cmd_term._obstacle_radius.cpu().numpy(),
            "body_r": cmd_term._body_radius.cpu().numpy(),
            "margin": float(cmd_term.cfg.cbf_margin),
            "dt": float(dt),
            "cbf": args_cli.cbf,
        }
        agent_assets = cmd_term._agent_assets

    ep_summaries = []
    obs, infos = env.get_observations()

    for episode in range(args_cli.num_episodes):
        ep_reward = 0.0
        contact_steps = 0
        collision_steps = 0
        active_steps = 0
        min_barrier_ep = float("inf")
        min_clearance_ep = float("inf")
        phys_collision_steps = 0
        interv_sum = 0.0
        track_sums = {"error_vel_xy": 0.0, "error_vel_yaw": 0.0, "error_height": 0.0}
        # per-step trajectory capture (env 0) for the figure/GIF
        capture = static_geom is not None and episode < args_cli.viz_episodes
        traj = {"box": [], "agents": [], "height": [], "min_h": [], "min_clearance": [],
                "intervention": [], "margin_eff": [], "disturbance": [], "morph_scale": [],
                "regrasp_gate": [], "fsm_state": []}

        for step in range(args_cli.steps_per_episode):
            with torch.inference_mode():
                actions = policy(obs)
                obs_, rewards, dones, infos = env.step(actions)
                ppo_runner.alg.policy.reset(dones)
                obs = deepcopy(obs_)

            ep_reward += rewards.sum().item()

            # grasp maintenance from the extra contact obs group
            extra = infos.get("observations", {}).get("extra", None)
            if extra is not None:
                contact_steps += int(extra.abs().sum().item() > 0.1)

            inst = tracking_errors()
            for k in track_sums:
                track_sums[k] += inst[k]

            log = {
                "episode": episode,
                "global_step": episode * args_cli.steps_per_episode + step,
                "step_reward": rewards.sum().item(),
                **{f"track_step/{k}": v for k, v in inst.items()},
            }
            if has_cbf:
                min_h = cmd_term.cbf_min_h.mean().item()
                interv = cmd_term.cbf_intervention.mean().item()
                active = cmd_term.cbf_active.float().mean().item()
                collided = (cmd_term.cbf_min_h < 0.0).float().mean().item()
                min_barrier_ep = min(min_barrier_ep, cmd_term.cbf_min_h.min().item())
                clearance = cmd_term.cbf_min_clearance.min().item()
                min_clearance_ep = min(min_clearance_ep, clearance)
                phys_collision_steps += int((cmd_term.cbf_min_clearance < 0.0).any().item())
                interv_sum += interv
                active_steps += int(active > 0.5)
                collision_steps += int(collided > 0.5)
                log.update({
                    "safety/min_barrier": min_h,         # with margin: < 0 => inside inflated keep-out
                    "safety/min_clearance": clearance,   # margin-free: < 0 => actual physical overlap
                    "safety/margin_effective": cmd_term.cbf_margin_effective.mean().item(),
                    "safety/disturbance": cmd_term.cbf_disturbance.mean().item(),
                    "safety/morph_scale": cmd_term.cbf_morph_scale.mean().item(),
                    "safety/regrasp_gate": cmd_term.cbf_regrasp_gate.mean().item(),
                    "safety/fsm_state": cmd_term.cbf_fsm_state.mean().item(),
                    "safety/intervention": interv,
                    "safety/active": active,
                    "safety/collision": collided,
                })
            wandb.log(log)

            if capture:
                o = env_origins[0, :2]
                traj["box"].append((cmd_term.robot.data.root_pos_w[0, :2] - o).cpu().numpy())
                traj["agents"].append(
                    np.stack([(a.data.root_pos_w[0, :2] - o).cpu().numpy() for a in agent_assets])
                )
                traj["height"].append(cmd_term.robot.data.root_pos_w[0, 2].item())
                traj["min_h"].append(cmd_term.cbf_min_h[0].item())
                traj["min_clearance"].append(cmd_term.cbf_min_clearance[0].item())
                traj["intervention"].append(cmd_term.cbf_intervention[0].item())
                traj["margin_eff"].append(cmd_term.cbf_margin_effective[0].item())
                traj["disturbance"].append(cmd_term.cbf_disturbance[0].item())
                traj["morph_scale"].append(cmd_term.cbf_morph_scale[0].item())
                traj["regrasp_gate"].append(cmd_term.cbf_regrasp_gate[0].item())
                traj["fsm_state"].append(cmd_term.cbf_fsm_state[0].item())

        contact_frac = contact_steps / args_cli.steps_per_episode
        summary = {
            "episode": episode,
            "episode_reward": ep_reward,
            "grasp/contact_frac": contact_frac,
            **{f"track_ep/{k}": v / args_cli.steps_per_episode for k, v in track_sums.items()},
        }
        if has_cbf:
            summary.update({
                "safety/min_barrier_ep": min_barrier_ep,
                "safety/min_clearance_ep": min_clearance_ep,
                "safety/collision_frac": collision_steps / args_cli.steps_per_episode,
                "safety/phys_collision_frac": phys_collision_steps / args_cli.steps_per_episode,
                "safety/active_frac": active_steps / args_cli.steps_per_episode,
                "safety/mean_intervention": interv_sum / args_cli.steps_per_episode,
                "safety/episode_collided": float(min_barrier_ep < 0.0),
                "safety/episode_phys_collided": float(min_clearance_ep < 0.0),
            })
        wandb.log(summary)
        ep_summaries.append(summary)

        # render + log the trajectory figure and animated GIF for this episode
        if capture and len(traj["box"]) > 0:
            rec = {
                "box": np.asarray(traj["box"]),
                "agents": np.asarray(traj["agents"]),
                "height": np.asarray(traj["height"]),
                "min_h": np.asarray(traj["min_h"]),
                "min_clearance": np.asarray(traj["min_clearance"]),
                "intervention": np.asarray(traj["intervention"]),
                "margin_eff": np.asarray(traj["margin_eff"]),
                "disturbance": np.asarray(traj["disturbance"]),
                "morph_scale": np.asarray(traj["morph_scale"]),
                "regrasp_gate": np.asarray(traj["regrasp_gate"]),
                "fsm_state": np.asarray(traj["fsm_state"]),
                **static_geom,
            }
            tag = f"{args_cli.cbf}_{n_agents}ag_{margin_tag}_ep{episode}"
            # Dump the per-step safety series for offline robust-margin analysis (cbf_robust_margin.py).
            # rec already carries `margin` (= cbf_margin) and `cbf` via static_geom; the realized
            # excursion is delta = max(0, margin - min_clearance).
            npz_path = os.path.join(log_dir, f"cbf_series_{tag}.npz")
            np.savez(npz_path, **rec)
            print(f"[INFO] saved per-step safety series -> {npz_path}")
            fig_path = os.path.join(log_dir, f"cbf_perf_{tag}.png")
            if cbf_viz.make_performance_figure(rec, fig_path):
                wandb.log({"plots/performance": wandb.Image(fig_path), "episode": episode})
            gif_path = os.path.join(log_dir, f"cbf_traj_{tag}.gif")
            if cbf_viz.make_trajectory_gif(rec, gif_path):
                wandb.log({"plots/trajectory_gif": wandb.Video(gif_path, fps=20, format="gif"),
                           "episode": episode})
            print(f"[INFO] logged performance figure + trajectory GIF for episode {episode}")

        msg = (f"[Ep {episode+1}/{args_cli.num_episodes}] reward={ep_reward:.2f} contact={contact_frac:.2f} "
               f"err_vel_xy={summary['track_ep/error_vel_xy']:.3f}")
        if has_cbf:
            msg += (f"  min_h={min_barrier_ep:+.3f}(margin) "
                    f"min_clearance={min_clearance_ep:+.3f}(physical) "
                    f"phys_collision={'Y' if min_clearance_ep < 0 else 'n'} "
                    f"collision_frac={summary['safety/collision_frac']:.2f} "
                    f"mean_interv={summary['safety/mean_intervention']:.3f}")
        print(msg)

        with torch.inference_mode():
            env.unwrapped.reset()
        obs, infos = env.get_observations()

    # cross-episode summary
    def _mean(key):
        return float(np.mean([s[key] for s in ep_summaries if key in s]))

    final = {
        "mean_episode_reward": _mean("episode_reward"),
        "mean_contact_frac": _mean("grasp/contact_frac"),
    }
    if has_cbf:
        final.update({
            "summary/collision_rate": _mean("safety/episode_collided"),            # vs inflated keep-out
            "summary/phys_collision_rate": _mean("safety/episode_phys_collided"),  # actual contact
            "summary/mean_collision_frac": _mean("safety/collision_frac"),
            "summary/mean_min_barrier": _mean("safety/min_barrier_ep"),
            "summary/mean_min_clearance": _mean("safety/min_clearance_ep"),
            "summary/mean_intervention": _mean("safety/mean_intervention"),
        })
    wandb.log(final)
    print(f"\n=== DONE ({args_cli.cbf.upper()}) === reward={final['mean_episode_reward']:.2f} "
          f"contact={final['mean_contact_frac']:.2f}"
          + (f"  phys_collision_rate={final['summary/phys_collision_rate']:.2f} "
             f"min_clearance={final['summary/mean_min_clearance']:+.3f}m "
             f"min_barrier={final['summary/mean_min_barrier']:+.3f}" if has_cbf else ""))

    if args_cli.video:
        # flush the in-progress recording to disk before logging (see note where the recorder is set up)
        if video_recorder is not None and getattr(video_recorder, "recording", False):
            video_recorder.stop_recording()
        if video_dir is not None and os.path.isdir(video_dir):
            mp4s = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
            if not mp4s:
                print(f"[WARN] no mp4 found in {video_dir}; the camera video was not recorded.")
            for vf in mp4s:
                print(f"[INFO] logging rollout video: {vf}")
                wandb.log({"rollout_video": wandb.Video(os.path.join(video_dir, vf), fps=20, format="mp4")})

    wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
