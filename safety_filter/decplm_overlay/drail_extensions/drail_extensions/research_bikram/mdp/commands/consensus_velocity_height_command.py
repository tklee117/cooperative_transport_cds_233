from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

from .categorical_velocity_height_command import (
    CategoricalVelocityHeightCommand,
    CategoricalVelocityHeightCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ConsensusVelocityHeightCommand(CategoricalVelocityHeightCommand):
    r"""A categorical velocity + height command where the stored command is the *team mean*,
    augmented with an independently-resampled per-env divergence used to form divergent agent
    beliefs for decentralized consensus manipulation.

    Motivation: when two agents move an object together they do not share their motion policies.
    Each agent only has its own (possibly different) "picture" of the desired object trajectory and
    must adjust to its partner by feeling forces / looking at poses. We model this by keeping a
    single shared *mean* command ``c`` (what the team should actually achieve, used by the reward)
    and a per-env divergence ``d``. Each agent observes a belief ``c + sign * d`` (``sign`` is +1 for
    agent 0 and -1 for agent 1) so that the mean of the two agents' beliefs is exactly ``c``. Because
    the reward tracks ``c`` and ``c`` is the mean of the beliefs, the optimum is a physical compromise
    (consensus-to-mean).

    The divergence resamples on its *own* clock (``divergence_resampling_time_range``), independent of
    the mean. A mid-episode divergence resample makes the partner's belief shift abruptly, acting as a
    disturbance the other agent must feel and re-converge to -- this is the disturbance-tolerance the
    setup is meant to study.

    Note: ``command`` / ``vel_command_b`` (read by the reward terms via
    ``command_manager.get_command``) remains the mean ``c``; beliefs are exposed via :meth:`belief` and
    consumed by the ``transformed_belief_commands`` observation term.
    """

    cfg: ConsensusVelocityHeightCommandCfg

    def __init__(self, cfg: ConsensusVelocityHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # per-env divergence on (lin_vel_x, lin_vel_y, ang_vel_z, height)
        self.divergence_b = torch.zeros(self.num_envs, 4, device=self.device)
        self._divergence_range = torch.tensor(cfg.divergence_range, device=self.device)
        # independent resampling clock for the divergence
        self._divergence_time_left = torch.zeros(self.num_envs, device=self.device)
        self.metrics["divergence_norm"] = torch.zeros(self.num_envs, device=self.device)

    """
    Properties.
    """

    def belief(self, sign: float) -> torch.Tensor:
        """The per-agent belief command ``mean + sign * divergence``. Shape is (num_envs, 4)."""
        return self.vel_command_b + sign * self.divergence_b

    """
    Implementation specific functions.
    """

    def _resample_divergence(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        # uniform in [-range, +range] per dimension
        self.divergence_b[env_ids] = (
            (torch.rand(n, 4, device=self.device) * 2.0 - 1.0) * self._divergence_range
        )
        # reset the (independent) divergence clock for these envs
        lo, hi = self.cfg.divergence_resampling_time_range
        self._divergence_time_left[env_ids] = torch.rand(n, device=self.device) * (hi - lo) + lo

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # also (re)sample divergence on mean-resample and on env reset.
        # guard against base-class calls that may precede our buffer creation.
        if hasattr(self, "divergence_b"):
            self._resample_divergence(env_ids)

    def _update_command(self):
        super()._update_command()
        if not hasattr(self, "divergence_b"):
            return
        # independent divergence clock: resample for envs whose timer elapsed
        self._divergence_time_left -= self._env.step_dt
        env_ids = (self._divergence_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_divergence(env_ids)

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["divergence_norm"] = torch.norm(self.divergence_b, dim=-1)


@configclass
class ConsensusVelocityHeightCommandCfg(CategoricalVelocityHeightCommandCfg):
    """Configuration for the consensus (divergent-belief) velocity and height command generator."""

    class_type: type = ConsensusVelocityHeightCommand

    divergence_range: tuple[float, float, float, float] = (0.15, 0.15, 0.1, 0.05)
    """Max absolute per-agent divergence on (lin_vel_x [m/s], lin_vel_y [m/s], ang_vel_z [rad/s],
    height [m]). The divergence is sampled uniformly in [-range, +range]. Agent 0 sees mean+d,
    agent 1 sees mean-d. Set to (0, 0, 0, 0) to recover the shared-goal task."""

    divergence_resampling_time_range: tuple[float, float] = (2.0, 4.0)
    """Time range (s) over which the divergence is independently resampled, so a partner's belief
    shift acts as a disturbance mid-episode."""
