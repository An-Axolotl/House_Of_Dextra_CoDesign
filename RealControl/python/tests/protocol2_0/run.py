#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, tty, termios, select, time, math
from collections import deque
import numpy as np
import torch
from dynamixel_sdk import *

# =========================
# Config (policy + timing)
# =========================
# TS_POLICY_PATH = "agents/just_cubes_and_cyli_2048e_policy.pt"
# TS_POLICY_PATH = "agents/trained_on_one_cube_best.pt"
# TS_POLICY_PATH = "agents/trained_on_one_cube_1000e.pt"
# TS_POLICY_PATH = "agents/diverse_train_no_adr.pt"
# TS_POLICY_PATH = "agents/cd_2048e_mod_init.pt"
# TS_POLICY_PATH = "agents/cd_2048e_no_index_small_cube.pt"
TS_POLICY_PATH = "agents/2048e_mod_init_small_cube.pt"
ACT_MOVING_AVG = 1.0 / 24.0
CONTROL_HZ     = 30.0

# =========================
# Isaac/MuJoCo ordering helpers
# =========================
def reorder_isaaclab_to_mujoco(a):
    a = np.asarray(a).reshape(-1)
    if a.size != 20: raise ValueError(f"Expected 20-dim, got {a.size}")
    out = np.empty_like(a)
    for i in range(5):
        b = 4*i
        out[b+0] = a[0*5 + i]  # palm
        out[b+1] = a[1*5 + i]  # base
        out[b+2] = a[2*5 + i]  # middle
        out[b+3] = a[3*5 + i]  # end
    return out

def reorder_mujoco_to_isaaclab(a):
    a = np.asarray(a).reshape(-1)
    if a.size != 20: raise ValueError(f"Expected 20-dim, got {a.size}")
    out = np.empty_like(a)
    for i in range(5):
        b = 4*i
        out[0*5 + i] = a[b+0]
        out[1*5 + i] = a[b+1]
        out[2*5 + i] = a[b+2]
        out[3*5 + i] = a[b+3]
    return out

def get_default_joint_positions_sim_mj():
    """MuJoCo default (values in sim-rad, MUJOCO order)."""
    # isaac_order = np.array([
    #     0.0, 0.0, 0.0, 0.0, 0.0,       # palms f1..f5
    #     1.0, 0.0, 0.0, 0.0, 0.4,       # bases f1..f5
    #     1.57, 1.57, 1.57, 1.57, 1.57,  # middles f1..f5
    #     0.0, 0.0, 0.0, 0.0, 0.0        # ends f1..f5
    # ], dtype=np.float32)
    
    # no index
    isaac_order = np.array([
        0.3, 0.0, 0.663, 0.0, 0.0,      # palm joints
        1.0, 0.0, 0.3, 0.0, 0.487,      # base joints
        1.57, 0.0, 1.57, 1.57, 1.46,  # middle joints
        0.5, 0.0, 0.64, 0.5, 1.16        # end joints
    ], dtype=np.float32)
    
    # ALL CLOSE
    # isaac_order = np.array([
    #     0.3, 0.477, 0.0, 0.0, 0.0,      # palm joints
    #     1.0, 0.748, 0.0, 0.0, 0.487,      # base joints
    #     1.57, 1.32, 1.57, 1.57, 1.46,  # middle joints
    #     0.5, 0.726, 0.5, 0.5, 1.16        # end joints
    # ], dtype=np.float32)
    
    isaac_order = np.array([
        0.3, 0.477, 0.0, 0.0, 0.0,      # palm joints
        1.0, 0.487, 0.0, 0.0, 0.487,      # base joints
        1.57, 1.32, 1.57, 1.57, 1.46,  # middle joints
        0.5, 0.726, 0.0, 0.0, 1.16        # end joints
    ], dtype=np.float32)
    
    return reorder_isaaclab_to_mujoco(isaac_order)

# =========================
# Non-blocking / blocking key input (works in a real TTY)
# =========================
if os.name == 'nt':
    import msvcrt
    def getch(block=False, timeout=None, flush=False):
        # blocking gate: wait for a key
        if block and timeout is None:
            return msvcrt.getch().decode(errors="ignore")
        # timed / non-blocking
        if timeout is not None:
            # emulate a timed wait
            start = time.time()
            while time.time() - start < timeout:
                if msvcrt.kbhit():
                    return msvcrt.getch().decode(errors="ignore")
                time.sleep(0.01)
            return None
        return msvcrt.getch().decode(errors="ignore") if msvcrt.kbhit() else None
else:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch(block=False, timeout=None, flush=False):
        """Return one char or None. Blocks if block=True and timeout is None."""
        old = termios.tcgetattr(fd)
        try:
            if flush:
                termios.tcflush(fd, termios.TCIFLUSH)
            tty.setcbreak(fd)  # raw-ish mode, no enter needed
            sys.stdout.flush()
            if block and timeout is None:
                # truly block for one char
                return sys.stdin.read(1)
            # timed / non-blocking
            wait = 0 if timeout is None else timeout
            r, _, _ = select.select([fd], [], [], wait)
            return sys.stdin.read(1) if r else None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

# =========================
# Dynamixel constants
# =========================
ADDR_TORQUE_ENABLE        = 64
ADDR_GOAL_POSITION        = 116; LEN_GOAL_POSITION = 4
ADDR_PRESENT_POSITION     = 132; LEN_PRESENT_POSITION = 4
ADDR_MIN_POSITION_LIMIT   = 52
ADDR_MAX_POSITION_LIMIT   = 48
ADDR_PROFILE_VELOCITY     = 112
ADDR_PROFILE_ACCELERATION = 108
ADDR_POSITION_P_GAIN      = 84
ADDR_POSITION_I_GAIN      = 82
ADDR_POSITION_D_GAIN      = 80
ADDR_FIRMWARE_VERSION     = 6
ADDR_RETURN_DELAY_TIME    = 9

BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

DXL_ID_ALL  = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
DXL_ID_MOVE = DXL_ID_ALL[:]  # control all 20

# Palms are IDs 1,5,9,13,17 (MUJOCO order index = id-1)
PALM_IDS = {1,5,9,13,17}

LIMIT_OFFSETS = [
    [+548, -641],[+58, -298],[+58, -248],[+58, -268],
    [+568, -568], [+74, -252],[+68, -248],[+48, -278],
    [+568, -569],[+84, -272], [+58, -248],[+52, -270],
    [+569, -569],[+64, -252],[+58, -248], [0, -266],
    [+374, -569],[72, 0],[0, -261],[98, -328]
]

DXL_PID_GAINS = [
    [1200, 100, 1500], [1200, 0, 1500], [2500, 0, 3000], [2000, 0, 2000],
    [900, 180, 1500],  [1200, 0, 2000], [2500, 0, 6000], [2000, 0, 2000],
    [900, 0, 1500],    [2500, 100, 6000], [2000, 10, 2500], [2000, 0, 2000],
    [1500, 0, 1500],   [2500, 100, 6000], [2000, 10, 2500], [2000, 0, 2000],
    [1500, 0, 1500],   [2500, 100, 6000], [2000, 0, 2500], [2000, 0, 2000]
]

DEVICENAME = '/dev/cu.usbserial-FT8ISFXP'
TORQUE_ENABLE  = 1
TORQUE_DISABLE = 0
INVALID_U32    = 0xFFFFFFFF

# =========================
# Per-joint sim ranges
#   - Palms: [-0.785, +0.785]
#   - Others: [0, 1.57]
# =========================
SIM_MIN_MJ = np.full(20, 0.0, dtype=np.float32)
SIM_MAX_MJ = np.full(20, 1.57, dtype=np.float32)
for mid in PALM_IDS:
    SIM_MIN_MJ[mid-1] = -0.785
    SIM_MAX_MJ[mid-1] =  0.785
SIM_SPAN_MJ = SIM_MAX_MJ - SIM_MIN_MJ

# =========================
# Per-joint tick <-> sim-rad mapping
# =========================
def ticks_to_sim_rad(motor_limits, motor_id: int, ticks: int) -> float:
    lo = motor_limits[motor_id]['tight_lo']; hi = motor_limits[motor_id]['tight_hi']
    idx = motor_id - 1
    t = min(max(int(ticks), lo), hi)
    frac = (t - lo) / max(1, (hi - lo))
    return float(SIM_MIN_MJ[idx] + frac * SIM_SPAN_MJ[idx])

def sim_rad_to_ticks(motor_limits, motor_id: int, sim_r: float) -> int:
    lo = motor_limits[motor_id]['tight_lo']; hi = motor_limits[motor_id]['tight_hi']
    idx = motor_id - 1
    s = min(max(float(sim_r), SIM_MIN_MJ[idx]), SIM_MAX_MJ[idx])
    frac = (s - SIM_MIN_MJ[idx]) / max(1e-8, SIM_SPAN_MJ[idx])
    return int(round(lo + frac * (hi - lo)))

# =========================
# Init comms
# =========================
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
groupSyncRead  = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)

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
FAST_OK  = FAST_API and all_fw_ok(DXL_ID_ALL)
print("Fast Sync Read available:", FAST_OK)

# =========================
# Configure motors + limits
# =========================
motor_limits = {}
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
    packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PROFILE_VELOCITY, 300)
    packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PROFILE_ACCELERATION, 100)

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
hist_buf = np.zeros((3, 40), dtype=np.float32)
object_one_hot = np.array([0,0,1,0], dtype=np.float32)
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
        qpos_sim[i] = ticks_to_sim_rad(motor_limits, motor_id, ticks)
    return qpos_sim

def build_obs_mj(qpos_sim_mj: np.ndarray, cur_targets_sim_mj: np.ndarray) -> torch.Tensor:
    # Normalize per joint: [-1,1] using arrays
    q_norm_mj = (2.0 * qpos_sim_mj - (SIM_MAX_MJ + SIM_MIN_MJ)) / (SIM_SPAN_MJ + 1e-8)
    q_norm_isaac      = reorder_mujoco_to_isaaclab(q_norm_mj)
    cur_targets_isaac = reorder_mujoco_to_isaaclab(cur_targets_sim_mj)  # raw sim-rad (unnormalized)
    frame = np.concatenate([q_norm_isaac, cur_targets_isaac], axis=0)  # (40,)
    global hist_buf
    hist_buf[:-1] = hist_buf[1:]
    hist_buf[-1]  = frame
    flat = hist_buf.reshape(-1)  # (120,)
    obs  = np.concatenate([flat, object_one_hot], axis=0).astype(np.float32)
    obs  = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.from_numpy(obs).unsqueeze(0)

def policy_step(cur_targets_sim_mj: np.ndarray, qpos_sim_mj: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs = build_obs_mj(qpos_sim_mj, cur_targets_sim_mj)
        acts_isaac = policy(obs)  # (1,20), [-1,1], relative deltas
    acts_mj = np.clip(reorder_isaaclab_to_mujoco(acts_isaac.squeeze(0).cpu().numpy()), -1.0, 1.0)
    new_targets_sim = cur_targets_sim_mj + ACT_MOVING_AVG * acts_mj
    # Per-joint clamp in sim space
    new_targets_sim = np.minimum(np.maximum(new_targets_sim, SIM_MIN_MJ), SIM_MAX_MJ)
    return new_targets_sim.astype(np.float32)

def send_goals_ticks_mj(targets_sim_mj: np.ndarray):
    for idx, motor_id in enumerate(DXL_ID_ALL):
        if motor_id not in DXL_ID_MOVE: continue
        goal_ticks = sim_rad_to_ticks(motor_limits, motor_id, float(targets_sim_mj[idx]))
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
cur_targets_sim = get_default_joint_positions_sim_mj().astype(np.float32)
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

