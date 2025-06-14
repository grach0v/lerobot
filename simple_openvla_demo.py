#!/usr/bin/env python3
"""
Simple OpenVLA SO-101 Demo Script

This is a minimal example showing how to integrate OpenVLA with your SO-101 robot arm.
Modify the configuration variables below to match your setup.

Requirements:
- pip install lerobot requests opencv-python pillow numpy
"""

import base64
import json
import time
from io import BytesIO
from typing import Dict, Any

import requests
from PIL import Image

# Import LeRobot components
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ===== CONFIGURATION =====
# Update these variables to match your setup:

OPENVLA_ENDPOINT = "https://andreemic--openvla-robot-get-next-gripper-position.modal.run"
ROBOT_PORT = "/dev/tty.usbmodem5A460857481"  # From your notes.md
INSTRUCTION = "pick up the red cube"
CAMERA_INDEX = 0  # Usually 0 for built-in camera, 1 for external

# Control settings
FPS = 2.0  # Lower FPS to avoid overwhelming the robot
MAX_STEPS = 50  # Stop after this many steps
MAX_RELATIVE_TARGET = 20.0  # Safety limit for joint movements

# ===== MAIN CODE =====

def send_to_openvla(image_array, instruction: str) -> Dict[str, Any]:
    """Send image and instruction to OpenVLA endpoint."""
    # Convert numpy array to PIL image, then to base64
    pil_image = Image.fromarray(image_array)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Create payload
    payload = {
        "image": img_str,
        "instruction": instruction
    }
    
    # Send request
    response = requests.post(
        OPENVLA_ENDPOINT,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def convert_action_to_robot_format(action_vector):
    """Convert OpenVLA action vector to SO-101 robot format."""
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    # Ensure we have the right number of values
    if len(action_vector) != len(motor_names):
        print(f"Warning: Expected {len(motor_names)} actions, got {len(action_vector)}")
        # Pad with zeros if too short, truncate if too long
        while len(action_vector) < len(motor_names):
            action_vector.append(0.0)
        action_vector = action_vector[:len(motor_names)]
    
    # Create action dictionary
    robot_action = {}
    for i, motor_name in enumerate(motor_names):
        robot_action[f"{motor_name}.pos"] = float(action_vector[i])
    
    return robot_action


def main():
    print("🤖 Starting OpenVLA SO-101 Demo")
    print(f"📡 Endpoint: {OPENVLA_ENDPOINT}")
    print(f"🔌 Robot Port: {ROBOT_PORT}")
    print(f"📝 Instruction: {INSTRUCTION}")
    print()
    
    # Configure and connect to robot
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id="second_follower",
        max_relative_target=MAX_RELATIVE_TARGET,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=CAMERA_INDEX,
                width=640,
                height=480,
                fps=30
            )
        }
    )
    
    robot = SO101Follower(robot_config)
    
    try:
        print("🔗 Connecting to robot...")
        robot.connect()
        print("✅ Robot connected!")
        
        print(f"🚀 Starting control loop for {MAX_STEPS} steps...")
        
        for step in range(MAX_STEPS):
            print(f"\n--- Step {step + 1}/{MAX_STEPS} ---")
            
            # Get observation (includes camera image and joint positions)
            obs = robot.get_observation()
            
            # Get camera image
            camera_image = obs.get("front")
            if camera_image is None:
                print("❌ No camera image available!")
                time.sleep(1.0)
                continue
            
            print("📷 Got camera image")
            
            # Get current joint positions for reference
            joint_positions = {k: v for k, v in obs.items() if k.endswith('.pos')}
            print(f"🦾 Current joints: {joint_positions}")
            
            # Send to OpenVLA
            # print("🧠 Sending to OpenVLA...")
            # prediction = send_to_openvla(camera_image, INSTRUCTION)
            
            # if prediction is None:
            #     print("❌ Failed to get prediction!")
            #     time.sleep(1.0)
            #     continue
            
            # # Extract action vector
            # action_vector = prediction.get("action", [])
            # if not action_vector:
            #     print("❌ No action vector in response!")
            #     time.sleep(1.0)
            #     continue
            
            # print(f"🎯 OpenVLA action: {action_vector}")
            
            # Convert to robot format
            # Apply the homing position (middle of range for each joint)
            robot_action = {
                "shoulder_pan.pos": -1688,
                "shoulder_lift.pos": -1130,
                "elbow_flex.pos": 1596,
                "wrist_flex.pos": 1881,
                "wrist_roll.pos": -570,
                "gripper.pos": 1159,  # Homing position for gripper
            }
            print(f"🤖 Robot action: {robot_action}")
            
            # Execute action
            executed_action = robot.send_action(robot_action)
            print(f"✅ Executed: {executed_action}")
            
            # Wait for next step
            time.sleep(1.0 / FPS)
        
        print(f"\n🎉 Completed {MAX_STEPS} steps!")
        
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        print("🔌 Disconnecting robot...")
        robot.disconnect()
        print("✅ Done!")


if __name__ == "__main__":
    main() 