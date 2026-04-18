# Hardware Setup

Item List:
- Dynamixel XL-330-M288-T Motor (20x)
- Dynamixel U2D2 Power Hub
- 5V 2A Barrel Jack

## Motor Position Resolution
Each position tick represents 360°/4096 ≈ 0.088 degrees of rotation.

<details>
<summary><strong>Detailed Motor Configuration Settings</strong></summary>

### Motor Settings Used in Research

**Common Parameters for All Motors:**
- Moving Threshold: 10 (2.29 rev/min, unit is 0.22888 rev/min)
- Temperature Limit: 70°C
- Voltage Range: 3.5V - 7V
- PWM Limit: 100%
- Current Limit: 1750 mA
- Velocity Limit: 445 (101.85 rev/min, unit is 0.22888 rev/min)

---

#### Thumb Motors

**Motor 1:**
- Homing Offset: 0
- Min Limit: 1404 (actual: 856)
- Max Limit: 2315 (actual: 2956)
- Drive Mode: 1
- PID Gains: (1200, 100, 1500)
- Limit Offset: (+548, -641)

**Motor 2:**
- Homing Offset: 0
- Min Limit: 2048 (actual: 1990)
- Max Limit: 3072 (actual: 3370)
- Drive Mode: 1
- PID Gains: (1200, 0, 1500)
- Limit Offset: (+58, -298)

**Motor 3:**
- Homing Offset: 0
- Min Limit: 2048 (actual: 1990)
- Max Limit: 3072 (actual: 3320)
- Drive Mode: 1
- PID Gains: (2500, 0, 3000)
- Limit Offset: (+58, -248)

**Motor 4:**
- Homing Offset: 0
- Min Limit: 2048 (actual: 1990)
- Max Limit: 3072 (actual: 3340)
- Drive Mode: 1
- PID Gains: (2000, 0, 2000)
- Limit Offset: (+58, -268)

---

#### Index Finger Motors

**Motor 5:**
- Homing Offset: 0
- Min Limit: 2616 (actual: 2048)
- Max Limit: 3527 (actual: 4095)
- Drive Mode: 0
- PID Gains: (900, 180, 1500)
- Limit Offset: (+568, -568)

**Motor 6:**
- Homing Offset: 0
- Min Limit: 1024 (actual: 950)
- Max Limit: 2048 (actual: 2300)
- Drive Mode: 1
- PID Gains: (1200, 0, 2000)
- Limit Offset: (+74, -252)

**Motor 7:**
- Homing Offset: 0
- Min Limit: 2048 (actual: 1980)
- Max Limit: 3072 (actual: 3320)
- Drive Mode: 1
- PID Gains: (2500, 0, 6000)
- Limit Offset: (+68, -248)

**Motor 8:**
- Homing Offset: 0
- Min Limit: 2048 (actual: 2000)
- Max Limit: 3072 (actual: 3350)
- Drive Mode: 1
- PID Gains: (2000, 0, 2000)
- Limit Offset: (+48, -278)

---

#### Middle Finger Motors

**Motor 9:**
- Homing Offset: 0
- Min Limit: 1592 (actual: 1024)
- Max Limit: 2503 (actual: 3072)
- Drive Mode: 0
- PID Gains: (900, 0, 1500)
- Limit Offset: (+568, -569)

**Motor 10:**
- Homing Offset: 0
- Min Limit: 1024 (actual: 940)
- Max Limit: 2048 (actual: 2320)
- Drive Mode: 1
- PID Gains: (2500, 100, 6000)
- Limit Offset: (+84, -272)

**Motor 11:**
- Homing Offset: 0
- Min Limit: 2048 (actual: 1990)
- Max Limit: 3072 (actual: 3320)
- Drive Mode: 1
- PID Gains: (2000, 10, 2500)
- Limit Offset: (+58, -248)

**Motor 12:**
- Homing Offset: 0
- Min Limit: 2048 (actual: 1996)
- Max Limit: 3072 (actual: 3342)
- Drive Mode: 1
- PID Gains: (2000, 0, 2000)
- Limit Offset: (+52, -270)

---

#### Ring Finger Motors

**Motor 13:**
- Homing Offset: 0
- Min Limit: 1820 (actual: 1251)
- Max Limit: 2730 (actual: 3299)
- Drive Mode: 1
- PID Gains: (1500, 0, 1500)
- Limit Offset: (+569, -569)

**Motor 14:**
- Homing Offset: 0
- Min Limit: 1024 (actual: 960)
- Max Limit: 2048 (actual: 2300)
- Drive Mode: 1
- PID Gains: (2500, 100, 6000)
- Limit Offset: (+64, -252)

**Motor 15:**
- Homing Offset: 0
- Min Limit: 2048 (actual: 1990)
- Max Limit: 3072 (actual: 3320)
- Drive Mode: 1
- PID Gains: (2000, 10, 2500)
- Limit Offset: (+58, -248)

**Motor 16:**
- Homing Offset: 0
- Min Limit: 0 (actual: 0)
- Max Limit: 1024 (actual: 1300)
- Drive Mode: 1
- PID Gains: (2000, 0, 2000)
- Limit Offset: (0, -266)

---

#### Pinky Finger Motors

**Motor 17:**
- Homing Offset: 512
- Min Limit: 374 (actual: 0)
- Max Limit: 1285 (actual: 1854)
- Drive Mode: 0
- PID Gains: (1500, 0, 1500)
- Limit Offset: (+374, -569)

**Motor 18:**
- Homing Offset: 0
- Min Limit: 3072 (actual: 3000)
- Max Limit: 4095 (actual: 4095)
- Drive Mode: 1
- PID Gains: (2500, 100, 6000)
- Limit Offset: (+72, 0)

**Motor 19:**
- Homing Offset: 0
- Min Limit: 1024 (actual: 1285)
- Max Limit: 2048 (actual: 1787)
- Drive Mode: 1
- PID Gains: (2000, 0, 2500)
- Limit Offset: (0, -261)

**Motor 20:**
- Homing Offset: 0
- Min Limit: 2048 (actual: 1990)
- Max Limit: 3072 (actual: 3350)
- Drive Mode: 1
- PID Gains: (2000, 0, 2000)
- Limit Offset: (+98, -328)

</details>


# Installation
Run ```sudo python python/setup.py install```

# Usage
Verify control movement with ```python sync_midpoint_control.py```

Run policy with ```python run.py```

Specify a policy with ```python run.py --policy-path agents/<.pt>``` or with short form ```python run.py -p agents/<.pt>```

See more options with ```python run.py --help```

# Configuration
All configs can be found in config/config.yaml

Device name can be figured out by running:
```
pip install arduino-cli
arduino-cli core update-index
arduino-cli core install arduino:avr
arduino-cli core list
arduino-cli board list
```

# Note On Robustness
Running the policy on the real hand requires a specific default joint position that the policy was trained under. Number of observations also matters and for now these are the only observations:
1. 3 frames of <joint position normalized between [-1, 1], joint target position in radian space> 
2. One hot encoded vector for the object type between [Sphere, Cross, Cube, Cylinder]
3. Optional: Object state

Current code will need to be modified to allow for robust observation/action space based on hand design.

Also the LEAP Hand Authors chose to use the first part of the frame in [-1, 1] space while the second part is in radians space, so this may have to be changed soon too.


# References
SDK Python Scripts: 
https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-protocol-20

SDK API REFERENCE:
https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/api_reference/python/

Python SDK Quick Start Guide:
https://www.youtube.com/watch?v=LAizFTTdL8o

Python SDK Walkthrough:
https://www.youtube.com/watch?v=uHnrLVZEGi4
https://www.youtube.com/watch?v=pZmueNctY0s


# Dynamixel SDK
<img src="http://emanual.robotis.com/assets/images/sw/sdk/dynamixel_sdk/overview/dynamixel_sdk_concept_logo.jpg">

The ROBOTIS Dynamixel SDK is a software development kit that provides Dynamixel control functions using packet communication. The API is designed for Dynamixel actuators and Dynamixel-based platforms. For more information on Dynamixel SDK, please refer to the e-manual below.
- [ROBOTIS e-Manual for Dynamixel SDK](http://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/)


