#!/usr/bin/env python3
# Selectively update XL330 settings (only what you pass via CLI).
# For --set-baud, the script: (1) reads & preserves each motor's Torque Enable,
# (2) Torque Off only those that need EEPROM writes, (3) writes Baud Rate,
# (4) switches the host port to the new baud, (5) restores original torque states.

# example usage:
# python configure.py --port /dev/cu.usbserial-FT8ISFXP --current-baud 1000000 --ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --set-baud 3000000 --verify

import argparse, sys, time
from typing import List, Tuple
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite

PROTOCOL_VERSION = 2.0

# Control table (XL330-M288-T)
ADDR_TORQUE_ENABLE = 64   # 1 byte (RAM)
ADDR_BAUD_RATE     = 8    # 1 byte (EEPROM)

def bps_to_reg(bps: int) -> int:
    mapping = {9600:0, 57600:1, 115200:2, 1_000_000:3, 2_000_000:4, 3_000_000:5, 4_000_000:6}
    if bps not in mapping:
        raise ValueError(f"Unsupported baud {bps}. Choose one of: {sorted(mapping)}")
    return mapping[bps]

def gs_write(port: PortHandler, ph: PacketHandler, addr: int, data_len: int, pairs: List[Tuple[int, bytes]]):
    gsw = GroupSyncWrite(port, ph, addr, data_len)
    for (dxl_id, payload) in pairs:
        if not gsw.addParam(dxl_id, payload):
            print(f"[ID:{dxl_id:03d}] addParam failed at addr {addr}")
            sys.exit(1)
    rc = gsw.txPacket()
    gsw.clearParam()
    if rc != 0:  # COMM_SUCCESS=0
        print("GroupSyncWrite error:", ph.getTxRxResult(rc))
        sys.exit(1)

def read_torque(ph: PacketHandler, port: PortHandler, dxl_id: int) -> int:
    val, rc, ec = ph.read1ByteTxRx(port, dxl_id, ADDR_TORQUE_ENABLE)
    if rc != 0 or ec != 0x00:
        # Default to 0 if unreadable so we don't accidentally torque back on
        return 0
    return int(val)

def write_torque(ph: PacketHandler, port: PortHandler, ids: List[int], value: int):
    pairs = [(i, bytes([value & 0xFF])) for i in ids]
    if pairs:
        gs_write(port, ph, ADDR_TORQUE_ENABLE, 1, pairs)

def main():
    ap = argparse.ArgumentParser(description="Selective XL330 configurator (only updates flags you pass).")
    ap.add_argument("--port", required=True, help="Serial device, e.g., /dev/ttyUSB0 or COM3")
    ap.add_argument("--current-baud", type=int, default=1_000_000, help="Current bus baud (bps)")
    ap.add_argument("--ids", required=True, help="Comma-separated motor IDs, e.g., 1,2,3,...,20")

    # Optional updates (only applied if provided)
    ap.add_argument("--set-baud", type=int, help="Change Baud Rate (bps): 9600, 57600, 115200, 1000000, 2000000, 3000000, 4000000")
    ap.add_argument("--verify", action="store_true", help="After changes, verify by pinging each ID at the new baud")
    args = ap.parse_args()

    ids = [int(s) for s in args.ids.split(",") if s.strip()]

    # Nothing to do?
    if args.set_baud is None:
        print("No updates requested (e.g., --set-baud). Exiting without changes.")
        return

    port = PortHandler(args.port)
    if not port.openPort():
        print("Failed to open port"); sys.exit(1)
    if not port.setBaudRate(args.current_baud):
        print(f"Failed to set current baud to {args.current_baud}"); sys.exit(1)

    ph = PacketHandler(PROTOCOL_VERSION)

    # --- Preserve current torque state per ID
    torque_before = {}
    for i in ids:
        torque_before[i] = read_torque(ph, port, i)

    # --- For EEPROM writes (baud), torque must be OFF
    to_torque_off = [i for i in ids if torque_before[i] != 0]
    write_torque(ph, port, to_torque_off, 0)

    # --- Apply Baud Rate if requested
    if args.set_baud is not None:
        baud_reg = bps_to_reg(args.set_baud)
        gs_write(port, ph, ADDR_BAUD_RATE, 1, [(i, bytes([baud_reg])) for i in ids])
        # Switch host to the new baud immediately after write
        time.sleep(0.05)
        if not port.setBaudRate(args.set_baud):
            print("WARNING: Host failed to switch to new baud. Power-cycle or set manually may be required.")

    # --- Restore original torque states
    to_torque_on = [i for i in ids if torque_before[i] == 1]
    write_torque(ph, port, to_torque_on, 1)

    # --- Optional verification (simple ping)
    if args.verify and args.set_baud is not None:
        ok = True
        for i in ids:
            _, rc, ec = ph.ping(port, i)
            if rc != 0 or ec != 0x00:
                print(f"[VERIFY] ID {i}: ping FAILED at {args.set_baud} bps")
                ok = False
        if ok:
            print(f"[VERIFY] All IDs responded at {args.set_baud} bps")

    print("Done.")
    port.closePort()

if __name__ == "__main__":
    main()
