#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright 2017 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################


#*******************************************************************************
#***********************     SyncRead and SyncWrite Example      ***********************
#  Required Environment to run this example :
#    - Protocol 2.0 supported DYNAMIXEL(X, P, PRO/PRO(A), MX 2.0 series)
#    - DYNAMIXEL Starter Set (U2D2, U2D2 PHB, 12V SMPS)
#  How to use the example :
#    - Select the DYNAMIXEL in use at the MY_DXL in the example code. 
#    - Build and Run from proper architecture subdirectory.
#    - For ARM based SBCs such as Raspberry Pi, use linux_sbc subdirectory to build and run.
#    - https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/
#  Author: Ryu Woon Jung (Leon)
#  Maintainer : Zerom, Will Son
# *******************************************************************************

import os
import sys, tty, termios, select
import time
from collections import deque
import math

# ---- HELPERS TO READ JOINT POSITION ----
TICKS_PER_REV = 4096
TICK2RAD = 2.0 * math.pi / TICKS_PER_REV

def ticks_to_rad(ticks: int) -> float:
    """Convert Dynamixel ticks [0..4095] to radians [0..2π)."""
    return (ticks % TICKS_PER_REV) * TICK2RAD

def tick_delta(a: int, b: int) -> int:
    """Shortest signed tick difference a-b wrapped to [-2048, 2048)."""
    d = (a - b) % TICKS_PER_REV
    if d > TICKS_PER_REV // 2:
        d -= TICKS_PER_REV
    return d

def error_rad(goal_ticks: int, present_ticks: int) -> float:
    """Shortest angular error (goal - present) in radians, wrapped to [-π, π)."""
    return tick_delta(goal_ticks, present_ticks) * TICK2RAD

# ---- PROFILING STATE ----
PROFILE_WINDOW = 120  # number of samples in the moving window
cmd_periods = deque(maxlen=PROFILE_WINDOW)   # time between SyncWrites
read_periods = deque(maxlen=PROFILE_WINDOW)  # time between SyncReads
bus_rtts    = deque(maxlen=PROFILE_WINDOW)   # write -> first successful read

t_prev_tx = None   # timestamp of previous SyncWrite

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch(block=True, timeout=None, flush=False):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            sys.stdout.flush()
            if flush:
                termios.tcflush(fd, termios.TCIFLUSH)  # only when blocking prompt
            tty.setcbreak(fd)
            if block and timeout is None:
                return sys.stdin.read(1)
            r, _, _ = select.select([fd], [], [], 0 if timeout is None else timeout)
            return sys.stdin.read(1) if r else None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


from dynamixel_sdk import *                    # Uses Dynamixel SDK library

# Control table address
ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
LEN_GOAL_POSITION           = 4         # Data Byte Length
ADDR_PRESENT_POSITION       = 132
LEN_PRESENT_POSITION        = 4         # Data Byte Length
ADDR_MIN_POSITION_LIMIT     = 52
ADDR_MAX_POSITION_LIMIT     = 48
ADDR_PROFILE_VELOCITY       = 112
ADDR_PROFILE_ACCELERATION   = 108
ADDR_POSITION_P_GAIN        = 84
ADDR_POSITION_I_GAIN        = 82
ADDR_POSITION_D_GAIN        = 80
ADDR_FIRMWARE_VERSION       = 6
ADDR_RETURN_DELAY_TIME      = 9

DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
BAUDRATE                    = 1000000

# DYNAMIXEL Protocol Version (1.0 / 2.0)
# https://emanual.robotis.com/docs/en/dxl/protocol2/
PROTOCOL_VERSION            = 2.0

# Motor configuration
DXL_ID_ALL_CNT = 20
DXL_ID_ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Per-ID limit offsets for tighter limits: {tight_lo_offset, tight_hi_offset}
LIMIT_OFFSETS = [
    [+618, -458],  # ID 1
    [+58, -298],   # ID 2
    [+58, -248],   # ID 3
    [+58, -268],   # ID 4
    [+682, -682],  # ID 5
    [+74, -252],   # ID 6
    [+68, -248],   # ID 7
    [+48, -278],   # ID 8
    [+682, -682],  # ID 9
    [+84, -272],   # ID 10
    [+58, -248],   # ID 11
    [+52, -270],   # ID 12
    [+172, -171],  # ID 13
    [+64, -252],   # ID 14
    [+58, -248],   # ID 15
    [0, -266],    # ID 16
    [+322, -489],   # ID 17
    [72,  0],   # ID 18
    [0,  -261],   # ID 19
    [98,  -328]    # ID 20
]

# PID Gains [P, I, D]
DXL_PID_GAINS = [
    [1200, 100, 1500], # ID 0 - Thumb Motor 0
    [1200, 0, 1500],   # ID 1 - Thumb Motor 1 
    [2500, 0, 3000],   # ID 2 - Thumb Motor 2
    [2000, 0, 2000],   # ID 3 - Thumb Motor 3
    [900, 180, 1500],  # ID 4 - Index Motor 4
    [1200, 0, 2000],   # ID 5 - Index Motor 5
    [2500, 0, 6000],   # ID 6 - Index Motor 6
    [2000, 0, 2000],   # ID 7 - Index Motor 7
    [900, 0, 1500],    # ID 8
    [2500, 100, 6000], # ID 9
    [2000, 10, 2500],  # ID 10
    [2000, 0, 2000],   # ID 11
    [1500, 0, 1500],   # ID 12
    [2500, 100, 6000], # ID 13
    [2000, 10, 2500],  # ID 14
    [2000, 0, 2000],   # ID 15
    [1500, 0, 1500],   # ID 16
    [2500, 100, 6000], # ID 17
    [2000, 0, 2500],   # ID 18
    [2000, 0, 2000]    # ID 19
]

# Motors to actually move (only 2 and 3)
# DXL_ID_MOVE = [15]
DXL_ID_MOVE = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

# DXL_ID_MIDDLE = []
DXL_ID_MIDDLE = [1, 5, 9, 13, 17]



# Use the actual port assigned to the U2D2.
# ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"
DEVICENAME                  = '/dev/cu.usbserial-FT8ISFXP'

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
DXL_MOVING_STATUS_THRESHOLD = 20                # Dynamixel moving status threshold

# Store limits for each motor
motor_limits = {}

index = 0

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Initialize GroupSyncWrite instance
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

# Initialize GroupSyncRead instace for Present Position
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

# Check SDK support + firmware gate for Fast Sync Read (X/XL-330 need fw >= 46)
def all_fw_ok(ids):
    for i in ids:
        fw, _, _ = packetHandler.read1ByteTxRx(portHandler, i, ADDR_FIRMWARE_VERSION)
        if fw is None or fw < 46:
            return False
    return True

FAST_API = hasattr(groupSyncRead, 'fastSyncRead')  # available in recent SDKs
FAST_OK  = FAST_API and all_fw_ok(DXL_ID_MOVE)
print("Fast Sync Read available:", FAST_OK)

# Configure all motors
for i, motor_id in enumerate(DXL_ID_ALL):
    # Disable torque first
    packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    
    # Read current limits
    min_limit, _, _ = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_MIN_POSITION_LIMIT)
    max_limit, _, _ = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_MAX_POSITION_LIMIT)
    
    # Calculate tight limits
    tight_lo = min_limit + LIMIT_OFFSETS[i][0]
    tight_hi = max_limit + LIMIT_OFFSETS[i][1]
    
    motor_limits[motor_id] = {
        'lo': min_limit,
        'hi': max_limit,
        'tight_lo': tight_lo,
        'tight_hi': tight_hi,
        'mid': (tight_lo + tight_hi) // 2
    }
    
    # Set PID gains
    packetHandler.write2ByteTxRx(portHandler, motor_id, ADDR_POSITION_P_GAIN, DXL_PID_GAINS[i][0])
    packetHandler.write2ByteTxRx(portHandler, motor_id, ADDR_POSITION_I_GAIN, DXL_PID_GAINS[i][1])
    packetHandler.write2ByteTxRx(portHandler, motor_id, ADDR_POSITION_D_GAIN, DXL_PID_GAINS[i][2])
    
    # Set profile velocity and acceleration
    packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PROFILE_VELOCITY, 300)
    packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PROFILE_ACCELERATION, 100)
    
    print(f"Configured motor {motor_id}: limits({tight_lo}-{tight_hi}), PID({DXL_PID_GAINS[i]})")

# Enable torque for motors we want to move
for motor_id in DXL_ID_MOVE:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel#%d has been successfully connected" % motor_id)
        
# Enable torque and set middle position for stationary motors
for motor_id in DXL_ID_MIDDLE:
    # Enable torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel#%d enabled for middle position" % motor_id)
    
    # Set to middle position
    mid_pos = motor_limits[motor_id]['mid']
    param_middle_position = [DXL_LOBYTE(DXL_LOWORD(mid_pos)), DXL_HIBYTE(DXL_LOWORD(mid_pos)), 
                            DXL_LOBYTE(DXL_HIWORD(mid_pos)), DXL_HIBYTE(DXL_HIWORD(mid_pos))]
    
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_GOAL_POSITION, mid_pos)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel#%d set to middle position: %d" % (motor_id, mid_pos))

# Add parameter storage for motors we want to read
for motor_id in DXL_ID_MOVE:
    dxl_addparam_result = groupSyncRead.addParam(motor_id)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncRead addparam failed" % motor_id)
        quit()

CONTROL_HZ = 30.0
PERIOD = 1.0 / CONTROL_HZ

# Ramping toward target (middle) to keep motion smooth
GOAL_TOL_TICKS = 6                # stop ramping when this close to target

next_tick = time.perf_counter()

cmd_periods.clear()
read_periods.clear()
bus_rtts.clear()
t_prev_tx = None
index = 0  # keep your demo toggle

last_print = time.perf_counter()
missed_ticks = 0
read_t_prev = None

last_goals = {}       # motor_id -> last goal sent this tick
last_positions = {}   # motor_id -> last position read this tick
goal_state     = {}  # motor_id -> current ramped goal (ticks)

# Oscillation state per motor
DWELL_CYCLES = 8   # stay at a target this many ticks (~5/30 ≈ 0.17 s) before flipping
osc_target = {}    # motor_id -> current target endpoint (mid or tight_lo)
osc_dwell  = {}    # motor_id -> how long we've been within tolerance at that target

for m in DXL_ID_MOVE:
    # start by heading to the default endpoint (tight_lo). Change to 'mid' to start in the other direction.
    osc_target[m] = motor_limits[m]['tight_lo']
    osc_dwell[m]  = 0

_ = groupSyncRead.fastSyncRead() if FAST_OK else groupSyncRead.txRxPacket()
for m in DXL_ID_MOVE:
    if groupSyncRead.isAvailable(m, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
        goal_state[m] = groupSyncRead.getData(m, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    else:
        goal_state[m] = motor_limits[m]['mid']  # fallback

try:
    while True:
        # Quit on ESC/q if you want (optional)
        ch = getch(block=False, timeout=0)
        if ch in ('\x1b','q'):
            break

        # --- build ramped goals for this tick (present-based oscillation) ---
        for motor_id in DXL_ID_MOVE:
            # endpoints
            mid     = motor_limits[motor_id]['mid']
            default = motor_limits[motor_id]['tight_lo']

            target  = osc_target[motor_id]
            present = last_positions.get(motor_id, goal_state.get(motor_id, target))

            g = int(target)   # <-- let the servo profile do the smoothing
            last_goals[motor_id] = g
            
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(g)), DXL_HIBYTE(DXL_LOWORD(g)),
                DXL_LOBYTE(DXL_HIWORD(g)), DXL_HIBYTE(DXL_HIWORD(g))
            ]
            if not groupSyncWrite.addParam(motor_id, param_goal_position):
                print(f"[ID:{motor_id:03d}] groupSyncWrite addparam failed"); sys.exit(1)

            # Flip after you’ve arrived (and dwelled) at the endpoint
            if abs(target - present) <= GOAL_TOL_TICKS:
                osc_dwell[motor_id] += 1
                if osc_dwell[motor_id] >= DWELL_CYCLES:
                    osc_target[motor_id] = (mid if target == default else default)
                    osc_dwell[motor_id] = 0
            else:
                osc_dwell[motor_id] = 0


        # --- measure + write once per tick (this defines control frequency) ---
        now = time.perf_counter()
        if t_prev_tx is not None:
            cmd_periods.append(now - t_prev_tx)
        t_prev_tx = now

        dxl_comm_result = groupSyncWrite.txPacket()
        groupSyncWrite.clearParam()
        if dxl_comm_result != COMM_SUCCESS:
            print(packetHandler.getTxRxResult(dxl_comm_result))

        # --- read once per tick and profile ---
        t_after_write = now
        dxl_comm_result = groupSyncRead.fastSyncRead() if FAST_OK else groupSyncRead.txRxPacket()
        t_after_read = time.perf_counter()

        if read_t_prev is not None:
            read_periods.append(t_after_read - read_t_prev)
        read_t_prev = t_after_read

        bus_rtts.append(t_after_read - t_after_write)

        # cache present positions for this tick
        for motor_id in DXL_ID_MOVE:
            if groupSyncRead.isAvailable(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                last_positions[motor_id] = groupSyncRead.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)

        # once per ~1s: profile + status
        if cmd_periods and (time.perf_counter() - last_print) >= 1.0:
            avg_cmd = 1.0 / (sum(cmd_periods) / len(cmd_periods))
            avg_read = 1.0 / (sum(read_periods) / len(read_periods)) if read_periods else 0.0
            avg_rtt_ms = 1000.0 * (sum(bus_rtts) / len(bus_rtts)) if bus_rtts else 0.0
            print(f"[PROFILE] cmd≈{avg_cmd:.1f} Hz (target {CONTROL_HZ:.1f}), "
                  f"read≈{avg_read:.1f} Hz, bus_rtt≈{avg_rtt_ms:.2f} ms, "
                  f"missed_ticks={missed_ticks}")

            print("=== ALL MOTOR STATUS ===")
            for motor_id in sorted(DXL_ID_MOVE):
                goal_t  = int(last_goals.get(motor_id, 0))
                pres_t  = int(last_positions.get(motor_id, 0))
                diff    = abs(goal_t - pres_t)
                print(f"[ID:{motor_id:02d}] MOVING | Goal:{goal_t:4d} Present:{pres_t:4d} Diff:{diff:3d}")

            # middle motors (single read)
            middle_positions = {}
            for motor_id in DXL_ID_MIDDLE:
                pos, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
                    portHandler, motor_id, ADDR_PRESENT_POSITION
                )
                if dxl_comm_result == COMM_SUCCESS:
                    middle_positions[motor_id] = pos

            for motor_id in sorted(DXL_ID_MIDDLE):
                goal_t  = motor_limits[motor_id]['mid']
                pres_t  = middle_positions.get(motor_id, 0)
                diff    = abs(goal_t - pres_t)
                print(f"[ID:{motor_id:02d}] MIDDLE | Goal:{goal_t:4d} Present:{pres_t:4d} Diff:{diff:3d}")
            print("=" * 50)

            last_print = time.perf_counter()

        # --- hold the 30 Hz period ---
        next_tick += PERIOD
        sleep_dt = next_tick - time.perf_counter()
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        else:
            missed_ticks += 1
            next_tick = time.perf_counter()
finally:
    # Clear syncread parameter storage
    groupSyncRead.clearParam()

    # Disable torque for moved motors
    for motor_id in DXL_ID_MOVE + DXL_ID_MIDDLE:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))

    # Close port
    portHandler.closePort()
