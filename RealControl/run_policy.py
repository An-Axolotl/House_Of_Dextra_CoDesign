#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, tty, termios, select, time, math
import argparse
from collections import deque
import numpy as np
import torch
from dynamixel_sdk import *

# Import utilities
from utils.config_loader import Config
from utils.transform import reorder_isaaclab_to_mujoco, reorder_mujoco_to_isaaclab, get_default_joint_positions_sim_mj
from utils.hardware_converter import HardwareConverter
from utils.input import setup_input, getch

# =========================
# Argument parsing (before loading config)
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Run robotic hand control with policy')
    parser.add_argument('--config', '-c',
                       type=str,
                       default="config/config.yaml",
                       help='Path to configuration file (default: config/config.yaml)')
    parser.add_argument('--policy-path', '-p', 
                       type=str,
                       default=None,  # Will be set from config if not provided
                       help='Path to the policy file (overrides config default)')
    return parser.parse_args()

# Parse arguments first
args = parse_args()

# =========================
# Load configuration
# =========================
config = Config(args.config)

# Set policy path: command line overrides config default
if args.policy_path is None:
    TS_POLICY_PATH = config.policy_default_path
else:
    TS_POLICY_PATH = args.policy_path

print(f"Using policy: {TS_POLICY_PATH}")
print(f"Using config: {args.config}")

# =========================
# Config (policy + timing)
# =========================
ACT_MOVING_AVG = config.act_moving_avg
CONTROL_HZ = config.control_hz

# =========================
# Set up 
# =========================
hardware_converter = HardwareConverter(config)
fd, old_settings = setup_input()

# =========================
# Dynamixel constants from config
# =========================
ADDR_TORQUE_ENABLE = config.get_dynamixel_address('torque_enable')
ADDR_GOAL_POSITION = config.get_dynamixel_address('goal_position')
LEN_GOAL_POSITION = config.get_data_length('goal_position')
ADDR_PRESENT_POSITION = config.get_dynamixel_address('present_position')
LEN_PRESENT_POSITION = config.get_data_length('present_position')
ADDR_MIN_POSITION_LIMIT = config.get_dynamixel_address('min_position_limit')
ADDR_MAX_POSITION_LIMIT = config.get_dynamixel_address('max_position_limit')
ADDR_PROFILE_VELOCITY = config.get_dynamixel_address('profile_velocity')
ADDR_PROFILE_ACCELERATION = config.get_dynamixel_address('profile_acceleration')
ADDR_POSITION_P_GAIN = config.get_dynamixel_address('position_p_gain')
ADDR_POSITION_I_GAIN = config.get_dynamixel_address('position_i_gain')
ADDR_POSITION_D_GAIN = config.get_dynamixel_address('position_d_gain')
ADDR_FIRMWARE_VERSION = config.get_dynamixel_address('firmware_version')
ADDR_RETURN_DELAY_TIME = config.get_dynamixel_address('return_delay_time')

BAUDRATE = config.dynamixel_baudrate
PROTOCOL_VERSION = config.dynamixel_protocol_version

DXL_ID_ALL = config.motor_ids
DXL_ID_MOVE = DXL_ID_ALL[:]  # control all motors

# Palms are IDs from config
PALM_IDS = config.palm_ids

LIMIT_OFFSETS = config.limit_offsets
DXL_PID_GAINS = config.pid_gains

DEVICENAME = config.dynamixel_device_name
TORQUE_ENABLE = config.get('dynamixel.torque_enable_value')
TORQUE_DISABLE = config.get('dynamixel.torque_disable_value')
INVALID_U32 = config.get('dynamixel.invalid_u32')

# =========================
# Init comms
# =========================
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)

if not portHandler.openPort(): print("Failed to open the port"); sys.exit(1)
print("Succeeded to open the port")
if not portHandler.setBaudRate(BAUDRATE): print("Failed to set the baudrate"); sys.exit(1)
print("Succeeded to change the baudrate")

def all_fw_ok(ids):
    for i in ids:
        fw, _, _ = packetHandler.read1ByteTxRx(portHandler, i, ADDR_FIRMWARE_VERSION)
        if fw is None or fw < 46:
            return False
    return True
FAST_API = hasattr(groupSyncRead, 'fastSyncRead')
FAST_OK = FAST_API and all_fw_ok(DXL_ID_ALL)
print("Fast Sync Read available:", FAST_OK)

# =========================
# Configure motors + limits
# =========================
motor_limits = {}
profile_velocity = config.get('dynamixel.profile_velocity')
profile_acceleration = config.get('dynamixel.profile_acceleration')

for i, motor_id in enumerate(DXL_ID_ALL):
    packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    min_limit, _, _ = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_MIN_POSITION_LIMIT)
    max_limit, _, _ = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_MAX_POSITION_LIMIT)
    tight_lo = max(0, min(min_limit + LIMIT_OFFSETS[i][0], 4095))
    tight_hi = max(0, min(max_limit + LIMIT_OFFSETS[i][1], 4095))
    if tight_lo >= tight_hi:
        center = (min_limit + max_limit) // 2
        tight_lo, tight_hi = center - 512, center + 512

    motor_limits[motor_id] = {
        'lo': min_limit, 'hi': max_limit,
        'tight_lo': tight_lo, 'tight_hi': tight_hi,
        'mid': (tight_lo + tight_hi) // 2
    }

    packetHandler.write2ByteTxRx(portHandler, motor_id, ADDR_POSITION_P_GAIN, DXL_PID_GAINS[i][0])
    packetHandler.write2ByteTxRx(portHandler, motor_id, ADDR_POSITION_I_GAIN, DXL_PID_GAINS[i][1])
    packetHandler.write2ByteTxRx(portHandler, motor_id, ADDR_POSITION_D_GAIN, DXL_PID_GAINS[i][2])
    packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PROFILE_VELOCITY, profile_velocity)
    packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PROFILE_ACCELERATION, profile_acceleration)

    print(f"Configured motor {motor_id}: tight [{tight_lo}, {tight_hi}]")

# Enable torque
for motor_id in DXL_ID_MOVE:
    packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

# SyncRead: read ALL for observations
for motor_id in DXL_ID_ALL:
    if not groupSyncRead.addParam(motor_id):
        print(f"[ID:{motor_id:03d}] groupSyncRead addparam failed"); sys.exit(1)

# =========================
# Policy + observation state
# =========================
hist_buf = np.zeros((config.history_frames, 40), dtype=np.float32)
object_one_hot = config.object_one_hot
policy = torch.jit.load(TS_POLICY_PATH, map_location="cpu"); policy.eval()

# last-known ticks cache for robust reads
last_known_ticks = {mid: motor_limits[mid]['mid'] for mid in DXL_ID_ALL}

def read_all_positions_sim_mj() -> np.ndarray:
    _ = groupSyncRead.fastSyncRead() if FAST_OK else groupSyncRead.txRxPacket()
    qpos_sim = np.zeros(20, dtype=np.float32)
    for i, motor_id in enumerate(DXL_ID_ALL):
        val = INVALID_U32
        if groupSyncRead.isAvailable(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
            val = groupSyncRead.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        ticks = last_known_ticks.get(motor_id, motor_limits[motor_id]['mid'])
        if val != INVALID_U32:
            ticks = val
        # clamp and cache
        lo = motor_limits[motor_id]['tight_lo']; hi = motor_limits[motor_id]['tight_hi']
        ticks = min(max(int(ticks), lo), hi)
        last_known_ticks[motor_id] = ticks
        # to sim-rad with per-joint range
        qpos_sim[i] = hardware_converter.ticks_to_sim_rad(motor_limits, motor_id, ticks)
    return qpos_sim

def build_obs_mj(qpos_sim_mj: np.ndarray, cur_targets_sim_mj: np.ndarray) -> torch.Tensor:
    # Normalize per joint: [-1,1] using arrays
    sim_min = hardware_converter.sim_min
    sim_max = hardware_converter.sim_max
    sim_span = hardware_converter.sim_span
    
    q_norm_mj = (2.0 * qpos_sim_mj - (sim_max + sim_min)) / (sim_span + 1e-8)
    q_norm_isaac = reorder_mujoco_to_isaaclab(q_norm_mj)
    cur_targets_isaac = reorder_mujoco_to_isaaclab(cur_targets_sim_mj)  # raw sim-rad (unnormalized)
    frame = np.concatenate([q_norm_isaac, cur_targets_isaac], axis=0)  # (40,)
    global hist_buf
    hist_buf[:-1] = hist_buf[1:]
    hist_buf[-1] = frame
    flat = hist_buf.reshape(-1)  # (120,)
    obs = np.concatenate([flat, object_one_hot], axis=0).astype(np.float32)
    obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.from_numpy(obs).unsqueeze(0)

def policy_step(cur_targets_sim_mj: np.ndarray, qpos_sim_mj: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs = build_obs_mj(qpos_sim_mj, cur_targets_sim_mj)
        acts_isaac = policy(obs)  # (1,20), [-1,1], relative deltas
    acts_mj = np.clip(reorder_isaaclab_to_mujoco(acts_isaac.squeeze(0).cpu().numpy()), -1.0, 1.0)
    new_targets_sim = cur_targets_sim_mj + ACT_MOVING_AVG * acts_mj
    # Per-joint clamp in sim space
    sim_min = hardware_converter.sim_min
    sim_max = hardware_converter.sim_max
    new_targets_sim = np.minimum(np.maximum(new_targets_sim, sim_min), sim_max)
    return new_targets_sim.astype(np.float32)

def send_goals_ticks_mj(targets_sim_mj: np.ndarray):
    for idx, motor_id in enumerate(DXL_ID_ALL):
        if motor_id not in DXL_ID_MOVE: continue
        goal_ticks = hardware_converter.sim_rad_to_ticks(motor_limits, motor_id, float(targets_sim_mj[idx]))
        param = [
            DXL_LOBYTE(DXL_LOWORD(goal_ticks)), DXL_HIBYTE(DXL_LOWORD(goal_ticks)),
            DXL_LOBYTE(DXL_HIWORD(goal_ticks)), DXL_HIBYTE(DXL_HIWORD(goal_ticks))
        ]
        if not groupSyncWrite.addParam(motor_id, param):
            print(f"[ID:{motor_id:03d}] groupSyncWrite addparam failed"); sys.exit(1)
    dxl_comm_result = groupSyncWrite.txPacket()
    groupSyncWrite.clearParam()
    if dxl_comm_result != COMM_SUCCESS:
        print(PacketHandler(PROTOCOL_VERSION).getTxRxResult(dxl_comm_result))

PERIOD = 1.0 / CONTROL_HZ
next_tick = time.perf_counter()

cmd_periods = deque(maxlen=120)
read_periods = deque(maxlen=120)
bus_rtts    = deque(maxlen=120)
t_prev_tx = None
read_t_prev = None
last_print = time.perf_counter()
missed_ticks = 0

# Start from MuJoCo default (sim-space) and send once
cur_targets_sim = get_default_joint_positions_sim_mj(config).astype(np.float32)
send_goals_ticks_mj(cur_targets_sim)

print("\n[READY] Sent default pose. Press ANY key to start policy. Press 'q' or ESC to quit.\n")

# Block once for activation
ch_block = getch(block=True, timeout=None, flush=True)
if ch_block in ('\x1b', 'q'):
    # Clean shutdown (skip starting loop)
    groupSyncRead.clearParam()
    for motor_id in DXL_ID_MOVE:
        packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    portHandler.closePort()
    sys.exit(0)

policy_active = True
hist_buf[:] = 0.0
print("[START] Policy control activated.")

try:
    while True:
        ch = getch(block=False, timeout=0)
        if ch in ('\x1b', 'q'):
            break

        t_before_read = time.perf_counter()
        qpos_sim = read_all_positions_sim_mj()
        t_after_read = time.perf_counter()
        if read_t_prev is not None:
            read_periods.append(t_after_read - read_t_prev)
        read_t_prev = t_after_read

        t_before_policy = time.perf_counter()
        new_targets_sim = policy_step(cur_targets_sim, qpos_sim) if policy_active else cur_targets_sim
        t_after_policy  = time.perf_counter()

        now = time.perf_counter()
        if t_prev_tx is not None:
            cmd_periods.append(now - t_prev_tx)
        t_prev_tx = now
        send_goals_ticks_mj(new_targets_sim)
        t_after_write = time.perf_counter()

        bus_rtts.append(t_after_read - t_after_write)
        cur_targets_sim = new_targets_sim

        if cmd_periods and (time.perf_counter() - last_print) >= 1.0:
            avg_cmd  = 1.0 / (sum(cmd_periods) / len(cmd_periods))
            avg_read = 1.0 / (sum(read_periods) / len(read_periods)) if read_periods else 0.0
            avg_rtt_ms = 1000.0 * (sum(bus_rtts) / len(bus_rtts)) if bus_rtts else 0.0
            print(f"[PROFILE] cmd≈{avg_cmd:.1f} Hz (target {CONTROL_HZ:.1f}), "
                  f"read≈{avg_read:.1f} Hz, bus_rtt≈{avg_rtt_ms:.2f} ms, "
                  f"policy_step={(t_after_policy - t_before_policy)*1000:.2f} ms, "
                  f"missed_ticks={missed_ticks}, active={policy_active}")
            print(f"q0={qpos_sim[0]:+.3f} rad  tgt0={cur_targets_sim[0]:+.3f} rad")
            last_print = time.perf_counter()

        next_tick += PERIOD
        sleep_dt = next_tick - time.perf_counter()
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        else:
            missed_ticks += 1
            next_tick = time.perf_counter()

finally:
    groupSyncRead.clearParam()
    for motor_id in DXL_ID_MOVE:
        packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    portHandler.closePort()

