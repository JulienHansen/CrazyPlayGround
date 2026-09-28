import os
import sys
import time
import threading
import argparse
import logging
from typing import Optional

import torch
import torch.nn as nn

import cflib.crtp
from cflib.crazyflie import Crazyflie

# Make the `common` package importable; appended so it cannot shadow installed modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.flight_recorder import CrazyflieStateBase
from common.utils import (emergency_land, quat_apply, quat_inv, wait_until, WaypointGate,
                          add_run_args, apply_run_args)

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
waypoint_gate = WaypointGate()  # replaced in main() from --waypoint-radius / --waypoint-hold

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
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 32), nn.ELU(),
            nn.Linear(32, 32), nn.ELU()
        )
        self.policy_layer = nn.Linear(32, self.num_actions)
        self.value_layer = nn.Linear(32, 1)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

    def compute(self, inputs, role):
        x = self.net_container(inputs["observations"])
        if role == "policy":
            mean = self.policy_layer(x)
        else:
            mean = self.value_layer(x)
        return mean, {"log_std": self.log_std_parameter}

# ============================================================
#                    CRAZYFLIE CONTROLLER
# ============================================================

class CrazyflieController(CrazyflieStateBase):
    """Crazyflie controller for position-based RL agents.

    Uses send_position_setpoint to command position in world frame.
    Sim equivalent: pos_hovering.py (command_level="position", action clamp ±0.1 displacement).
    """
    def __init__(self, uri: str, agent: PPO, run_args, initial_target=None):
        super().__init__(device)
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.agent = agent
        self.initial_target = initial_target

        self.make_recorder(run_args,
                           obs_fields=["lin_vel_b_x", "lin_vel_b_y", "lin_vel_b_z",
                                       "des_pos_b_x", "des_pos_b_y", "des_pos_b_z"],
                           action_fields=["action_dx", "action_dy", "action_dz"],
                           cmd_fields=["desired_pos_x", "desired_pos_y", "desired_pos_z"])

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
        """Main control loop: send position setpoints.

        Must match pos_hovering.py: action in [-0.1, 0.1] as position displacement
        added to current position each step. The CF firmware's position controller
        (pos → vel → att → rate → mixer) handles tracking.
        """
        INTERVAL = 0.01  # 100 Hz — matches sim (dt=1/500, decimation=5)
        SPIN_MARGIN = 0.002  # sleep to 2 ms before each deadline, busy-wait the rest (see wait_until)
        MAX_DISPLACEMENT = 0.1  # m — must match pos_hovering.py clamp(-0.1, 0.1)

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

            with torch.no_grad():
                _, outputs = self.agent.act(obs.unsqueeze(0), None, timestep=0, timesteps=1)
                action = outputs["mean_actions"].squeeze(0)  # deterministic mean
                action = action.clamp(-1.0, 1.0)

            # Position displacement: scale action by MAX_DISPLACEMENT
            # In sim: actions clamped to [-0.1, 0.1], target_pos = root_pos + actions
            displacement = action * MAX_DISPLACEMENT

            desired_pos = state.pos + displacement
            control_time = time.time()

            # send_position_setpoint uses the firmware's full cascade PID
            # (position → velocity → attitude → rate → mixer), matching
            # sim's command_level="position"
            self.cf.commander.send_position_setpoint(
                desired_pos[0].item(),
                desired_pos[1].item(),
                desired_pos[2].item(),
                0.0  # yaw = 0
            )
            self.record_step(step, state, control_time, target_pos, obs, action, desired_pos)
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
    """Build obs tensor matching pos_hovering.py: [lin_vel_b(3), desired_pos_b(3)]."""
    global target_pos
    if target_pos is None:
        return None
    target_pos = waypoint_gate.step(current_pos, target_pos)

    # Rotate world-frame Kalman velocity into body frame
    linear_vel_b = quat_apply(quat_inv(current_quat), current_vel)
    desired_pos_b = quat_apply(quat_inv(current_quat), target_pos - current_pos)

    obs = torch.cat([linear_vel_b, desired_pos_b], dim=-1)
    return obs


# ============================================================
#                    MODEL LOADING / MAIN
# ============================================================

def load_agent(checkpoint_path: Optional[str], device: torch.device) -> PPO:
    obs_space = 6
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
    parser = argparse.ArgumentParser(description="Run a trained position RL agent on a Crazyflie drone.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E8", help="URI of the Crazyflie")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        help="Initial target [x, y, z] in world frame. If not set, hovers above takeoff pos.")
    add_run_args(parser)
    args = parser.parse_args()

    global waypoint_gate
    waypoint_gate = apply_run_args(args)

    agent = load_agent(args.checkpoint, device)

    controller = CrazyflieController(uri=args.uri, agent=agent, run_args=args, initial_target=args.target)

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
