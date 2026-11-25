#!/bin/bash
# Obstacle stress-test for decPLM with the coupled-CBF safety filter.
#
# Runs the pretrained finetuning policy in the obstacle scene twice -- once WITHOUT the filter
# (baseline: the open-space policy drives into the obstacles) and once WITH it (the CBF-QP deflects
# the box + ALL agents around the obstacles) -- and logs safety diagnostics, a performance figure and
# an animated trajectory GIF to wandb (project decplm-cbf). Also records the sim video to wandb.
#
# Usage: ./run_play_cbf.sh [options]
#   --mode {both|on|off}  Which run(s) to do (default: both).
#   --robots {2|4}        2-robot or 4-robot obstacle task (4 == N; default: 2).
#   --vel_x FLOAT         Fixed box lin_vel_x driving it at the obstacles (default: 0.4).
#   --steps N             Steps per episode (default: 600).
#   --video               Also record the Isaac camera video to wandb.
#   --checkpoint PATH     Override checkpoint path.
#   --num_envs N          Parallel envs (default: 1).

set -e

CKPT="drail_extensions/drail_extensions/research_bikram/resources/pretrained_policy/Go2-D1-MARL-Box-Reach-Touch-Lift-and-Move-Finetuning-Play-v0/team-osu/isaaclab/ectjv07i/model_latest.pt"
TASK="Go2-D1-MARL-Box-Reach-Touch-Lift-and-Move-Finetuning-Obstacles-Play-v0"
MODE="both"
ROBOTS="2"
VEL_X="0.4"
NUM_ENVS=1
STEPS=1500   # the goal-seeking box must traverse the ~5 m slalom; needs a long episode
VIDEO=""
CBF_MARGIN=""   # optional override (m); empty = use the cfg's cbf_margin
DEVICE="cuda:0"
MORPH=""        # "--morph" to enable the grasp-morph prototype
MORPH_MODE="tuck"
UNFREEZE=""     # "--unfreeze_grasp" to keep the policy perceiving the contacts past 2s (re-grasp probe)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)         MODE="$2";         shift 2 ;;
        --robots)       ROBOTS="$2";       shift 2 ;;
        --vel_x)        VEL_X="$2";        shift 2 ;;
        --steps)        STEPS="$2";        shift 2 ;;
        --cbf_margin)   CBF_MARGIN="$2";   shift 2 ;;
        --device)       DEVICE="$2";       shift 2 ;;
        --morph)        MORPH="--morph";   shift 1 ;;
        --morph_mode)   MORPH_MODE="$2";   shift 2 ;;
        --unfreeze)     UNFREEZE="--unfreeze_grasp"; shift 1 ;;
        --video)        VIDEO="--video";   shift 1 ;;
        --checkpoint)   CKPT="$2";         shift 2 ;;
        --num_envs)     NUM_ENVS="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Select the task by agent count. The 4-agent task is the "N-Robots" env (ma_config = 4 agents);
# the 2-agent task is the finetuning obstacle env. Only 2 and 4 (alias N/n) are available.
case "$ROBOTS" in
    2)
        TASK="Go2-D1-MARL-Box-Reach-Touch-Lift-and-Move-Finetuning-Obstacles-Play-v0" ;;
    4|N|n)
        TASK="Go2-D1-MARL-Box-Reach-Touch-Lift-and-Move-Finetuning-N-Robots-Obstacles-Play-v0" ;;
    *)
        echo "Unsupported --robots '$ROBOTS'. Available: 2 or 4 (alias N)."; exit 1 ;;
esac
echo "[run_play_cbf] robots=${ROBOTS} -> task ${TASK}"

cd ~/Academics/Decentralized-Pinch-Lift-Move

VIDEO_ARGS=""
[ -n "$VIDEO" ] && VIDEO_ARGS="--video --video_length ${STEPS} --enable_cameras"
MARGIN_ARGS=""
[ -n "$CBF_MARGIN" ] && MARGIN_ARGS="--cbf_margin ${CBF_MARGIN}"
MORPH_ARGS=""
[ -n "$MORPH" ] && MORPH_ARGS="--morph --morph_mode ${MORPH_MODE}"
[ -n "$UNFREEZE" ] && MORPH_ARGS="${MORPH_ARGS} ${UNFREEZE}"

run_one () {
    local cbf="$1"
    echo "=================== CBF: ${cbf}  (task: ${TASK}) ==================="
    WANDB_DISABLE_GIT=true python \
        drail_extensions/drail_extensions/research_bikram/scripts/play_cbf.py \
        --task "${TASK}" \
        --cbf "${cbf}" \
        --headless \
        ${VIDEO_ARGS} \
        ${MARGIN_ARGS} \
        ${MORPH_ARGS} \
        --device "${DEVICE}" \
        --num_envs "${NUM_ENVS}" \
        --cmd_vel_x "${VEL_X}" \
        --steps_per_episode "${STEPS}" \
        --num_episodes 1 \
        --viz_episodes 1 \
        --eval_name "obstacles" \
        "agent.raw_checkpoint=${CKPT}" \
        2>&1 | tee "/tmp/play_cbf_${cbf}.log"
}

case "$MODE" in
    both) run_one off; run_one on ;;
    on)   run_one on ;;
    off)  run_one off ;;
    *) echo "Unknown mode: $MODE"; exit 1 ;;
esac
