"""Offline sanity check for a velocity hover checkpoint — no drone, no Isaac Lab.

Loads the skrl checkpoint exactly like collect_hover_vel.py / exec_vel.py and
feeds it a few canonical observations to confirm the weights + observation
preprocessor loaded correctly and the policy reacts sensibly. Run this FIRST on
the laptop you will deploy from (e.g. a MacBook) before ever arming the drone.

    python execution/sim2real/check_policy.py --checkpoint path/to/best_agent.pt

Observation layout (matches vel_hovering.py): [lin_vel_b(3), desired_pos_b(3)].
Action layout: [vx, vy, vz] in [-1, 1], scaled by max_velocity=1.0 m/s.

Expected behaviour of a healthy hover policy:
  * at target, zero velocity   -> command ~ 0
  * target above (err_b z > 0) -> positive vz
  * target ahead (err_b x > 0) -> positive vx
  * moving toward target already -> damped command
"""

import os
import argparse

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler

MAX_VELOCITY = 1.0


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 32), nn.ELU(),
            nn.Linear(32, 32), nn.ELU())
        self.policy_layer = nn.Linear(32, self.num_actions)
        self.value_layer = nn.Linear(32, 1)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

    def compute(self, inputs, role):
        x = self.net_container(inputs["observations"])
        mean = self.policy_layer(x) if role == "policy" else self.value_layer(x)
        return mean, {"log_std": self.log_std_parameter}


def load_agent(checkpoint_path, device):
    obs_space, act_space = 6, 3
    policy = Policy(observation_space=obs_space, action_space=act_space, device=device)
    cfg = PPO_CFG(observation_preprocessor=RunningStandardScaler,
                  observation_preprocessor_kwargs={"size": obs_space, "device": device})
    agent = PPO(models={"policy": policy}, memory=None, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)
    assert checkpoint_path and os.path.exists(checkpoint_path), "checkpoint not found"
    agent.load(checkpoint_path)              # skrl maps to `device` internally
    agent.enable_training_mode(False)
    return agent


def act(agent, obs_list, device):
    obs = torch.tensor([obs_list], dtype=torch.float32, device=device)
    with torch.no_grad():
        _, outputs = agent.act(obs, None, timestep=0, timesteps=1)
        a = outputs["mean_actions"].squeeze(0).clamp(-1.0, 1.0)
    return a.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Offline sanity check for a velocity hover checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps. Default: auto.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}, torch={torch.__version__}")

    agent = load_agent(args.checkpoint, device)
    print(f"[INFO] loaded {args.checkpoint}\n")

    # [lin_vel_b(x,y,z), desired_pos_b(x,y,z)]
    cases = [
        ("at target, still",          [0, 0, 0,   0, 0, 0]),
        ("target 0.5 m ahead (+x)",   [0, 0, 0,   0.5, 0, 0]),
        ("target 0.5 m left (+y)",    [0, 0, 0,   0, 0.5, 0]),
        ("target 0.5 m above (+z)",   [0, 0, 0,   0, 0, 0.5]),
        ("target 0.5 m below (-z)",   [0, 0, 0,   0, 0, -0.5]),
        ("already moving +x to tgt",  [0.5, 0, 0, 0.5, 0, 0]),
    ]
    print(f"{'case':30s} {'action (vx,vy,vz)':>34s}   cmd [m/s]")
    ok = True
    for name, obs in cases:
        a = act(agent, obs, device)
        cmd = [round(x * MAX_VELOCITY, 3) for x in a]
        print(f"{name:30s} {str([round(x,3) for x in a]):>34s}   {cmd}")
        if name.startswith("at target") and max(abs(x) for x in a) > 0.3:
            ok = False
    print()
    # directional sanity: command should point toward the target
    ax = act(agent, [0, 0, 0, 0.5, 0, 0], device)[0]
    az = act(agent, [0, 0, 0, 0, 0, 0.5], device)[2]
    signs_ok = ax > 0 and az > 0
    print(f"[CHECK] +x target -> vx>0: {ax:+.3f}   |   +z target -> vz>0: {az:+.3f}")
    if ok and signs_ok:
        print("[PASS] checkpoint loads and reacts sensibly. Safe to proceed to hardware pre-checks.")
    else:
        print("[WARN] policy output looks off (drifts at target or wrong direction). "
              "Re-check the checkpoint / training before flying.")


if __name__ == "__main__":
    main()
