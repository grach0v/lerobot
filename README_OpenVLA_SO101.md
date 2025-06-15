# OpenVLA SO-101 Robot Arm Integration

This integration allows you to control your SO-101 robot arm using OpenVLA (Open Vision-Language-Action) model predictions. The system captures images from a camera, sends them to your OpenVLA endpoint along with text instructions, and executes the predicted actions on the robot.

## 🚀 Quick Start

### Prerequisites

1. **SO-101 Robot Arm**: Assembled and calibrated
2. **OpenVLA Endpoint**: Deployed on Modal (as per your setup)
3. **Camera**: USB camera or built-in webcam
4. **Python Environment**: Python 3.8+ with LeRobot installed

### Installation

```bash
# Install required packages
pip install requests opencv-python pillow numpy

# If you haven't installed LeRobot with feetech support:
pip install -e ".[feetech]"
```

### Basic Usage

1. **Simple Demo** (easiest to start with):
   ```bash
   python simple_openvla_demo.py
   ```
   Edit the configuration variables at the top of the file to match your setup.

2. **Full-Featured Script** (more control):
   ```bash
   python openvla_so101_control.py \
       --endpoint "https://andreemic--openvla-robot-get-next-gripper-position.modal.run" \
       --port "/dev/tty.usbmodem5A680114471" \
       --instruction "pick up the red cube" \
       --camera_index 0 \
       --fps 2
   ```

## 📁 Files

- **`simple_openvla_demo.py`**: Simple demonstration script with minimal configuration
- **`openvla_so101_control.py`**: Full-featured script with command-line options
- **`README_OpenVLA_SO101.md`**: This documentation file

## ⚙️ Configuration

### Robot Setup

Your robot port is already configured in your `notes.md` as `/dev/tty.usbmodem5A680114471`. Make sure:

1. **Robot is calibrated**: Run calibration if you haven't already:
   ```bash
   python -m lerobot.calibrate \
       --robot.type=so101_follower \
       --robot.port=/dev/tty.usbmodem5A680114471 \
       --robot.id=my_so101_arm
   ```

2. **Motors are set up**: If needed, run motor setup:
   ```bash
   python -m lerobot.setup_motors \
       --teleop.type=so101_leader \
       --teleop.port=/dev/tty.usbmodem5A680114471
   ```

### Camera Setup

- **Camera Index**: Usually `0` for built-in camera, `1` for external USB camera
- **Resolution**: Default is 640x480, but you can adjust in the scripts
- **Test your camera**: Use `python -m lerobot.find_cameras` to detect available cameras

### OpenVLA Endpoint

Your endpoint URL: `https://andreemic--openvla-robot-get-next-gripper-position.modal.run`

The script expects the endpoint to:
- Accept POST requests with JSON payload: `{"image": "base64_string", "instruction": "text"}`
- Return JSON response with: `{"action": [float, float, float, float, float, float]}`

## 🎮 Usage Examples

### Example 1: Pick and Place
```bash
python openvla_so101_control.py \
    --endpoint "https://andreemic--openvla-robot-get-next-gripper-position.modal.run" \
    --port "/dev/tty.usbmodem5A680114471" \
    --instruction "pick up the red cube and place it in the box" \
    --fps 2 \
    --duration 60
```

### Example 2: Continuous Operation
```bash
python openvla_so101_control.py \
    --endpoint "https://andreemic--openvla-robot-get-next-gripper-position.modal.run" \
    --port "/dev/tty.usbmodem5A680114471" \
    --instruction "organize the objects on the table" \
    --fps 1 \
    --verbose
```

### Example 3: With Safety Limits
```bash
python openvla_so101_control.py \
    --endpoint "https://andreemic--openvla-robot-get-next-gripper-position.modal.run" \
    --port "/dev/tty.usbmodem5A680114471" \
    --instruction "carefully move the fragile object" \
    --fps 1 \
    --max_relative_target 10.0
```

## 🔧 Command Line Options

### Required Arguments
- `-e, --endpoint`: OpenVLA endpoint URL
- `-p, --port`: Robot serial port
- `-i, --instruction`: Text instruction for the robot

### Optional Arguments
- `-c, --camera_index`: Camera index (default: 0)
- `-f, --fps`: Control loop frequency in Hz (default: 5.0)
- `-d, --duration`: Maximum duration in seconds (default: unlimited)
- `--camera_width`: Camera width (default: 640)
- `--camera_height`: Camera height (default: 480)
- `--max_relative_target`: Maximum relative movement for safety (default: None)
- `-v, --verbose`: Enable verbose logging

## 🛡️ Safety Features

### Built-in Safety
- **Max Relative Target**: Limits how far joints can move in a single step
- **Connection Monitoring**: Automatically disconnects if connection is lost
- **Error Handling**: Graceful handling of network errors and robot failures
- **Interrupt Handling**: Clean shutdown with Ctrl+C

### Recommended Safety Settings
- Start with `--fps 1` or `--fps 2` for slow, controlled movements
- Use `--max_relative_target 20.0` to limit joint movement per step
- Always test with `--duration 30` first to limit operation time
- Keep the emergency stop (if available) within reach

## 🔍 Troubleshooting

### Common Issues

1. **"Robot not connected"**
   - Check that the robot port is correct: `/dev/tty.usbmodem5A680114471`
   - Make sure no other programs are using the robot
   - Try unplugging and reconnecting the robot

2. **"No camera image available"**
   - Check camera index with `python -m lerobot.find_cameras`
   - Try different camera indices (0, 1, 2...)
   - Make sure camera is not being used by other applications

3. **"OpenVLA API error"**
   - Verify your endpoint URL is correct
   - Check your internet connection
   - Test the endpoint manually with curl or browser

4. **Robot moves erratically**
   - Lower the FPS: `--fps 1`
   - Add safety limits: `--max_relative_target 10.0`
   - Check robot calibration

### Debug Mode
Run with verbose logging to see detailed information:
```bash
python openvla_so101_control.py [...] --verbose
```

### Test Components Individually

1. **Test Robot Connection**:
   ```bash
   python -m lerobot.teleoperate \
       --robot.type=so101_follower \
       --robot.port=/dev/tty.usbmodem5A680114471 \
       --teleop.type=keyboard
   ```

2. **Test Camera**:
   ```bash
   python -m lerobot.find_cameras
   ```

3. **Test OpenVLA Endpoint**:
   Use the example from your original documentation to test the endpoint manually.

## 🎯 Action Format

The SO-101 robot expects actions for 6 motors:
1. **shoulder_pan**: Base rotation
2. **shoulder_lift**: Shoulder up/down
3. **elbow_flex**: Elbow bend
4. **wrist_flex**: Wrist up/down
5. **wrist_roll**: Wrist rotation
6. **gripper**: Open/close gripper

OpenVLA should return a 6-element action vector corresponding to these joints.

## 📊 Performance Tips

- **Lower FPS** for more stable control: `--fps 1` or `--fps 2`
- **Shorter instructions** tend to work better than complex multi-step commands
- **Good lighting** improves vision model performance
- **Clear camera view** of the workspace is important
- **Consistent setup** helps the model learn your environment

## 🆘 Emergency Stop

Always be ready to:
1. Press **Ctrl+C** to stop the script
2. Physically disconnect the robot if needed
3. Use any emergency stop button on your robot setup

## 📝 Customization

### Modify Action Conversion
Edit the `convert_openvla_action()` function in either script to change how OpenVLA actions are mapped to robot joints.

### Add Preprocessing
You can add image preprocessing before sending to OpenVLA by modifying the `predict()` method in the `OpenVLAClient` class.

### Change Safety Limits
Adjust the safety parameters in the robot configuration or command line arguments.

## 🤝 Integration with LeRobot

This integration uses the standard LeRobot interfaces:
- `SO101Follower` for robot control
- `OpenCVCameraConfig` for camera setup
- Standard LeRobot logging and utilities

You can easily extend this to work with other LeRobot robots by changing the robot class and configuration.

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your robot and camera setup with LeRobot's built-in tools
3. Test your OpenVLA endpoint independently
4. Review the logs with `--verbose` flag enabled

Happy robot controlling! 🤖✨ 