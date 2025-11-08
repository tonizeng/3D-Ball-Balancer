# 3D-Ball-Balancer

## Checkpoint Requirements

### 1. Camera tracking:
The cameras should be operational and able to track the ball.

### 2. Axis balancing:
You should be able to balance the ball along each servo axis. The balance does not need to be perfect, but it should converge.

### 3. Platform balancing:
You should have started working on balancing the ball on the platform.

## Migrating the System to 3D

**`ball_detection.py`**  
- Update it so it reports the ball position in both X and Y (normalized and in meters).  
- The controller will read these values to decide how to move the platform.

**New 3-axis controller (based on `basic_controller.py`)**  
- Read the `(x, y)` position from `ball_detection.py`.  
- Replace the single-axis PID with logic that drives all three axes of the platform.  
- Send the target angles/leg lengths to the hardware over serial.

**`SP_Motor_Controller.ino`**  
- Arduino code that listens for commands and moves the platform motors accordingly.  
- Keep the serial command format in sync with the new controller.

notes/things to think about: 
- we should finetune to make our starting position more flat
- we will probably need script similar to `simple_cal.py` to fetch hsv bounds, frame size, and pixel-to-meter mapping, and store it in a `config.json` file. we will probably also need servo and geometry info for other scripts (probably similar to what was in the old cal file)
- will we do live tuning for PID? is the multi threading necessary or can we just hardcode it so we don't have to deal with the gui popping up?
- where will the inverse kinematics equations be used? -> for now, since we only care about single-axis movement, this is not necessary yet.