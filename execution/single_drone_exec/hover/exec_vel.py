import os
import sys
import time
import threading
import argparse
import logging
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn

import cflib.crtp
from cflib.crazyflie import Crazyflie

# Make the `common` package importable; appended so it cannot shadow installed modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.flight_recorder import CrazyflieStateBase
from common.utils import (emergency_land, quat_apply, quat_inv, wait_until, add_run_args,
                          positive_int, nonnegative_int)

# skrl imports
from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format="{asctime} [{levelname}] {message}",
                        style="{",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
logger = logging.getLogger("CrazyflieRL")
target_pos = None  # initialized in control_loop after takeoff

# ── Safety thresholds ────────────────────────────────────────────────────────
POS_STALE_TIMEOUT_S    = 0.5   # max seconds without a position callback before emergency land
POS_VARIANCE_THRESHOLD = 0.5   # kalman position variance [m²] above which tracking is unreliable


# ============================================================
#                      MODEL DEFINITION
# ============================================================


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0,
                 initial_log_std=0.0):

        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)

        # === Must match checkpoint architecture ===
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU()
        )

        # Policy head
        self.policy_layer = nn.Linear(32, self.num_actions)

        # Value head (SKRL requires it even for policy model)
        self.value_layer = nn.Linear(32, 1)

        # Shared log std
        self.log_std_parameter = nn.Parameter(
            torch.ones(self.num_actions) * initial_log_std
        )

    def compute(self, inputs, role):
        x = self.net_container(inputs["observations"])

        if role == "policy":
            mean = self.policy_layer(x)
        else:  # value function
            mean = self.value_layer(x)

        return mean, {"log_std": self.log_std_parameter}

# ============================================================
#                    CRAZYFLIE CONTROLLER
# ============================================================

class CrazyflieController(CrazyflieStateBase):
    """Crazyflie controller for velocity-based RL agents.

    Uses send_velocity_world_setpoint to command velocity in world frame.
    Sim equivalent: vel_hovering.py (command_level="velocity", max_velocity=1.0 m/s).
    """
    def __init__(self, uri: str, agent: PPO, run_args, initial_target=None,
                 obs_input_hist: int = 1, action_input_hist: int = 0):
        super().__init__(device)
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.agent = agent
        self.initial_target = initial_target
        self.k, self.m = obs_input_hist, action_input_hist  # see resolve_obs_hist

        self.make_recorder(run_args,
                           obs_fields=["lin_vel_b_x", "lin_vel_b_y", "lin_vel_b_z",
                                       "des_pos_b_x", "des_pos_b_y", "des_pos_b_z"],
                           action_fields=["action_vx", "action_vy", "action_vz"],
                           cmd_fields=["vx", "vy", "vz"])

        self._setup_callbacks()

    # ---------- Crazyflie callbacks ----------

    def _setup_callbacks(self):
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

    def _connected(self, uri: str):
        logger.info(f"Connected to {uri}, taking off...")
        self.cf.high_level_commander.takeoff(0.5, 1)
        time.sleep(1.5)
        self.setup_state_logging()
        self.recorder.start()
        threading.Thread(target=self.control_loop, daemon=True).start()

    def _disconnected(self, uri: str):
        pass

    def _connection_failed(self, uri: str, msg: str):
        logger.error(f"Connection to {uri} failed: {msg}")
        self.running = False

    def _connection_lost(self, uri: str, msg: str):
        logger.warning(f"Connection to {uri} lost: {msg} — triggering safe landing")
        self.running = False

    # ---------- Control loop ----------

    def control_loop(self):
        """Main control loop: send world-frame velocity commands.

        Must match vel_hovering.py: action in [-1, 1] scaled by max_velocity=1.0 m/s.
        """
        INTERVAL = 0.01  # 100 Hz — matches sim (dt=1/500, decimation=5)
        SPIN_MARGIN = 0.002  # sleep to 2 ms before each deadline, busy-wait the rest (see wait_until)
        MAX_VEL = 1.0    # m/s — must match vel_hovering.py QuadcopterEnvCfg.max_velocity

        logger.info("Waiting for first position estimate...")
        while not self.position_received and self.running:
            time.sleep(0.05)
        if not self.running:
            return
        logger.info(f"Position received: {self.current_pos}")

        # Initialize target
        global target_pos
        if self.initial_target is not None:
            target_pos = torch.tensor(self.initial_target, dtype=torch.float32, device=device)
        else:
            target_pos = self.current_pos.clone()
            target_pos[2] = max(0.5, min(1.5, target_pos[2].item()))
        logger.info(f"Init target pos={target_pos}")

        # Policy input as vel_hovering._get_observations builds it: the last K frames, oldest first,
        # primed with K copies of the first one, then the last M commanded actions, zeros at first.
        window = None
        act_hist = torch.zeros(self.m, 3, device=device)

        nn_start_time = time.time()
        GRACE_PERIOD = 3.0  # seconds before enforcing z lower bound
        deadline = time.perf_counter()
        step = 0
        while self.cf.is_connected() and self.running:

            # ── Safety watchdog ──────────────────────────────────────────────
            elapsed_since_nn = time.time() - nn_start_time
            z = self.current_pos[2].item()
            if (elapsed_since_nn > GRACE_PERIOD and z < 0.1) or z > 2.5:
                logger.error(f"Position out of bounds z={z:.2f} — emergency landing")
                emergency_land(self)
                break
            if self._last_pos_time > 0 and time.time() - self._last_pos_time > POS_STALE_TIMEOUT_S:
                logger.error(
                    f"Position data stale ({time.time() - self._last_pos_time:.2f} s) — emergency landing"
                )
                emergency_land(self)
                break
            with self.lock:
                var = self._pos_variance.clone()
            if var.max().item() > POS_VARIANCE_THRESHOLD:
                logger.error(f"Position variance too high {var.tolist()} — emergency landing")
                emergency_land(self)
                break

            state = self.snapshot()  # one consistent read, used for the obs and its recorded row
            obs = retrieve_and_create_observation(state.vel, state.pos, state.quat)
            if obs is None:
                logger.warning("No observation received, hovering...")
                step += 1
                deadline, _, _ = wait_until(deadline + INTERVAL, SPIN_MARGIN)
                continue

            window = obs.expand(self.k, -1) if window is None else torch.cat([window[1:], obs[None]])
            policy_input = torch.cat([window.flatten(), act_hist.flatten()])

            with torch.no_grad():
                _, outputs = self.agent.act(policy_input.unsqueeze(0), None, timestep=0, timesteps=1)
                action = outputs["mean_actions"].squeeze(0)  # deterministic mean
                action = action.clamp(-1.0, 1.0)
            if self.m:  # cat([act_hist[1:], a]) would grow an empty (M=0) history to one row
                act_hist = torch.cat([act_hist[1:], action[None]])

            # Scale action to velocity in m/s (matching sim: actions * max_velocity)
            velocity_cmd = action * MAX_VEL

            control_time = time.time()
            self.cf.commander.send_velocity_world_setpoint(
                velocity_cmd[0].item(),
                velocity_cmd[1].item(),
                velocity_cmd[2].item(),
                0.0  # yaw_rate = 0
            )

            self.record_step(step, state, control_time, target_pos, obs, action, velocity_cmd)
            step += 1

            deadline, _, _ = wait_until(deadline + INTERVAL, SPIN_MARGIN)

        self.cf.commander.send_stop_setpoint()
        logger.info("Control loop stopped")

    # ---------- Connection management ----------

    def start(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(self.uri)

    def stop(self):
        logger.info("Stopping controller...")
        self.running = False
        try:
            time.sleep(0.6)
            logger.info("Landing...")
            self.cf.high_level_commander.land(0.0, 2.0)
            time.sleep(2.5)
            self.cf.close_link()
            logger.info("Link closed")
        finally:
            self.recorder.close()  # always write the Parquet footer, or the file is unreadable


# ============================================================
#                    OBSERVATION CREATION
# ============================================================

def retrieve_and_create_observation(current_vel, current_pos, current_quat) -> Optional[torch.Tensor]:
    """Build obs tensor matching vel_hovering.py: [lin_vel_b(3), desired_pos_b(3)]."""
    if target_pos is None:
        return None

    # Rotate world-frame Kalman velocity into body frame
    linear_vel_b = quat_apply(quat_inv(current_quat), current_vel)
    desired_pos_b = quat_apply(quat_inv(current_quat), target_pos - current_pos)

    obs = torch.cat([linear_vel_b, desired_pos_b], dim=-1)
    return obs


# ============================================================
#                    MODEL LOADING / MAIN
# ============================================================

def read_obs_hist(checkpoint_path: Optional[str]) -> Optional[Tuple[int, int]]:
    """(obs_input_hist, action_input_hist) from the training run's params/env.yaml, or None.

    train.py writes <run>/params/env.yaml next to <run>/checkpoints/. Its `observation_space`
    is stale (dumped before the env recomputes it), so read the two knobs instead, with a
    regex because yaml.safe_load rejects the file's !!python/tuple tags.
    """
    if not checkpoint_path:
        return None
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    path = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    k = re.search(r"^obs_input_hist:\s*(\d+)\s*$", text, re.MULTILINE)
    m = re.search(r"^action_input_hist:\s*(\d+)\s*$", text, re.MULTILINE)
    return (int(k.group(1)), int(m.group(1))) if k and m else None


def resolve_obs_hist(checkpoint_path: Optional[str], flag_k: Optional[int],
                     flag_m: Optional[int]) -> Tuple[int, int]:
    """Frames K and past actions M the policy was trained with.

    The checkpoint only fixes the input width 6K + 3M, and equal widths load silently
    (K=1, M=2 and K=2, M=0 both give 12), so env.yaml wins and a flag that disagrees
    with it is an error. Without env.yaml (copied .pt, run before WP1) the flags decide,
    then the plain K=1, M=0.
    """
    trained = read_obs_hist(checkpoint_path)
    if trained is None:
        k, m = (1 if flag_k is None else flag_k), (0 if flag_m is None else flag_m)
        logger.info(f"No obs_input_hist/action_input_hist in the run's env.yaml: K={k}, M={m} from the flags")
        return k, m
    for flag, name, value in ((flag_k, "--obs-input-hist", trained[0]),
                              (flag_m, "--action-input-hist", trained[1])):
        if flag is not None and flag != value:
            raise ValueError(f"{name} {flag} disagrees with the training run's env.yaml ({value})")
    logger.info(f"K={trained[0]}, M={trained[1]} from the training run's env.yaml")
    return trained


def load_agent(checkpoint_path: Optional[str], device: torch.device, obs_space: int = 6) -> PPO:
    act_space = 3

    policy = Policy(observation_space=obs_space, action_space=act_space, device=device)
    models = {"policy": policy}

    cfg = PPO_CFG(
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": obs_space, "device": device},
    )
    agent = PPO(models=models, memory=None, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)

    assert checkpoint_path and os.path.exists(checkpoint_path), "No valid checkpoint provided. Please give a path for weights."

    agent.load(checkpoint_path)
    agent.enable_training_mode(False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return agent


def main():
    parser = argparse.ArgumentParser(description="Run a trained velocity RL agent on a Crazyflie drone.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E8", help="URI of the Crazyflie")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        help="Initial target [x, y, z] in world frame. If not set, hovers above takeoff pos.")
    parser.add_argument("--obs-input-hist", type=positive_int, default=None,
                        help="Observation frames K the policy was trained with. Read from the run's "
                             "params/env.yaml when present; only needed for a checkpoint outside its run.")
    parser.add_argument("--action-input-hist", type=nonnegative_int, default=None,
                        help="Past commanded actions M the policy was trained with. Same rule as K.")
    add_run_args(parser)
    args = parser.parse_args()

    try:
        k, m = resolve_obs_hist(args.checkpoint, args.obs_input_hist, args.action_input_hist)
    except ValueError as e:
        parser.error(str(e))
    agent = load_agent(args.checkpoint, device, 6 * k + 3 * m)

    controller = CrazyflieController(uri=args.uri, agent=agent, run_args=args, initial_target=args.target,
                                     obs_input_hist=k, action_input_hist=m)

    try:
        controller.start()
        timeout = 10
        elapsed = 0
        while not controller.cf.is_connected() and controller.running:
            time.sleep(1)
            elapsed += 1
            if elapsed >= timeout:
                logger.error(f"Connection timeout after {timeout}s — make sure cfclient is closed and drone is on")
                return
        if not controller.running:
            logger.error("Connection failed — check radio URI and that cfclient is closed")
            return
        logger.info("Cf is connected !")
        while controller.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    finally:
        controller.stop()
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
