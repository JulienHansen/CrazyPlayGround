import logging
import time
from typing import Any, Dict

import torch
from cflib.crazyflie.log import LogConfig


def setup_state_logging(controller, log_frequency_ms: int = 10) -> None:
    """Register the shared telemetry LogConfig blocks (position/velocity,
    quaternion, Kalman variance, motor PWM) and their callbacks on `controller`.

    Writes into the same attributes each script's CrazyflieController already
    declares in __init__: current_pos, current_vel, _last_pos_time,
    position_received, current_quat, _pos_variance, current_motor_pwm.
    """
    device = controller.current_pos.device

    def _log_posvel_callback(timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with controller.lock:
            controller.current_pos = torch.tensor([
                data["stateEstimate.x"],
                data["stateEstimate.y"],
                data["stateEstimate.z"]
            ], dtype=torch.float32, device=device)
            controller.current_vel = torch.tensor([
                data["stateEstimate.vx"],
                data["stateEstimate.vy"],
                data["stateEstimate.vz"]
            ], dtype=torch.float32, device=device)
            controller._last_pos_time = time.time()
            controller.position_received = True

    def _log_data_quat_callback(timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with controller.lock:
            controller.current_quat = torch.tensor([
                data['stateEstimate.qw'],
                data['stateEstimate.qx'],
                data['stateEstimate.qy'],
                data['stateEstimate.qz'],
            ], dtype=torch.float32, device=device)

    def _log_variance_callback(timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with controller.lock:
            controller._pos_variance = torch.tensor([
                data["kalman.varPX"],
                data["kalman.varPY"],
                data["kalman.varPZ"],
            ], dtype=torch.float32, device=device)

    def _log_motor_callback(timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        with controller.lock:
            controller.current_motor_pwm = [
                data["motor.m1"], data["motor.m2"], data["motor.m3"], data["motor.m4"],
            ]

    # Block 1: position + velocity (6 floats = 24 bytes, within 26-byte limit)
    log_posvel = LogConfig(name="posvel", period_in_ms=log_frequency_ms)
    log_posvel.add_variable("stateEstimate.x", "float")
    log_posvel.add_variable("stateEstimate.y", "float")
    log_posvel.add_variable("stateEstimate.z", "float")
    log_posvel.add_variable("stateEstimate.vx", "float")
    log_posvel.add_variable("stateEstimate.vy", "float")
    log_posvel.add_variable("stateEstimate.vz", "float")
    controller.cf.log.add_config(log_posvel)
    log_posvel.data_received_cb.add_callback(_log_posvel_callback)
    log_posvel.start()

    # Block 2: quaternion (4 floats = 16 bytes)
    log_quat = LogConfig(name="quat", period_in_ms=log_frequency_ms)
    log_quat.add_variable("stateEstimate.qx", "float")
    log_quat.add_variable("stateEstimate.qy", "float")
    log_quat.add_variable("stateEstimate.qz", "float")
    log_quat.add_variable("stateEstimate.qw", "float")
    controller.cf.log.add_config(log_quat)
    log_quat.data_received_cb.add_callback(_log_data_quat_callback)
    log_quat.start()

    # Block 3: Kalman variance (for safety watchdog)
    log_var = LogConfig(name="quality", period_in_ms=200)
    log_var.add_variable("kalman.varPX", "float")
    log_var.add_variable("kalman.varPY", "float")
    log_var.add_variable("kalman.varPZ", "float")
    controller.cf.log.add_config(log_var)
    log_var.data_received_cb.add_callback(_log_variance_callback)
    log_var.start()

    # Block 4: motor PWM (4 uint16 = 8 bytes, within 26-byte limit)
    log_motor = LogConfig(name="motor", period_in_ms=log_frequency_ms)
    log_motor.add_variable("motor.m1", "uint16_t")
    log_motor.add_variable("motor.m2", "uint16_t")
    log_motor.add_variable("motor.m3", "uint16_t")
    log_motor.add_variable("motor.m4", "uint16_t")
    controller.cf.log.add_config(log_motor)
    log_motor.data_received_cb.add_callback(_log_motor_callback)
    log_motor.start()


def emergency_land(controller) -> None:
    logger = logging.getLogger("CrazyflieRL")
    logger.warning("EMERGENCY LANDING triggered")
    controller.running = False
    try:
        controller.cf.high_level_commander.land(0.0, 2.0)
    except Exception as e:
        logger.error(f"Emergency land command failed: {e}")
        try:
            controller.cf.commander.send_stop_setpoint()
        except Exception:
            pass


def quat_apply(quat, vec):
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_conjugate(q):
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


def quat_inv(q, eps=1e-9):
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)
