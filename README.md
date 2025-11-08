# 3D-Ball-Balancer

## Checkpoint Requirements

### Camera tracking:
The cameras should be operational and able to track the ball.

### Axis balancing:
You should be able to balance the ball along each servo axis. The balance does not need to be perfect, but it should converge.

### Platform balancing:
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

notes: we should finetune to make our starting position more flat