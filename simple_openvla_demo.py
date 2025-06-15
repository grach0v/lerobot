#!/usr/bin/env python3
"""
Simple OpenVLA SO-101 Demo Script with SmolVLA Support

This is a minimal example showing how to integrate OpenVLA and SmolVLA with your SO-101 robot arm.
Use CLI arguments to configure the system.

Usage Examples:
    # Manual mode (default) - enter motor values manually:
    python simple_openvla_demo.py --method manual

    # Manual mode with live camera streaming:
    python simple_openvla_demo.py --method manual --stream-camera

    # SmolVLA local inference:
    python simple_openvla_demo.py --method smolvla --smolvla-model-path lerobot/smolvla_base

    # OpenVLA API mode:
    python simple_openvla_demo.py --method openvla --openvla-endpoint https://your-endpoint.com

    # Both models (SmolVLA first, fallback to OpenVLA):
    python simple_openvla_demo.py --method both

    # Custom robot port and instruction with camera streaming:
    python simple_openvla_demo.py --port /dev/ttyUSB0 --instruction "pick up the blue block" --stream-camera

    # Help:
    python simple_openvla_demo.py --help

Requirements:
- pip install lerobot requests opencv-python pillow numpy torch
- pip install -e ".[smolvla]"  # For SmolVLA support
"""

import argparse
import base64
import json
import logging
import threading
import time
from io import BytesIO
from typing import Dict, Any, Optional

import cv2
import numpy as np
import requests
import torch
from PIL import Image

# Import LeRobot components
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig

# SmolVLA imports
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Set up logging
formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# ===== MAIN CODE =====

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenVLA + SmolVLA SO-101 Robot Demo")
    
    # Inference method
    parser.add_argument(
        "-m", "--method", 
        choices=["openvla", "smolvla", "both", "manual"],
        default="manual",
        help="Inference method to use"
    )
    
    # OpenVLA API settings
    parser.add_argument(
        "--openvla-endpoint",
        default="https://andreemic--openvla-robot-get-next-gripper-position.modal.run",
        help="OpenVLA API endpoint URL"
    )
    
    # SmolVLA model settings
    parser.add_argument(
        "--smolvla-model-path",
        default="lerobot/smolvla_base",
        help="SmolVLA pretrained model path"
    )
    
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    
    # Robot settings
    parser.add_argument(
        "-p", "--port",
        default="/dev/tty.usbmodem5A460857481",
        help="Robot serial port"
    )
    
    parser.add_argument(
        "-i", "--instruction",
        default="pick up the red cube",
        help="Natural language instruction for the robot"
    )
    
    parser.add_argument(
        "-c", "--camera-index",
        type=int,
        default=0,
        help="Camera index (usually 0 for built-in, 1 for external)"
    )
    
    # Camera streaming settings
    parser.add_argument(
        "--stream-camera",
        action="store_true",
        help="Enable live camera streaming in a separate window"
    )
    
    parser.add_argument(
        "--stream-fps",
        type=int,
        default=30,
        help="Camera streaming FPS (default: 30)"
    )
    
    parser.add_argument(
        "--stream-window-size",
        nargs=2,
        type=int,
        default=[640, 480],
        help="Camera streaming window size [width height] (default: 640 480)"
    )
    
    # Control settings
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Control loop frequency in Hz"
    )
    
    parser.add_argument(
        "-s", "--max-steps",
        type=int,
        default=50,
        help="Maximum number of control steps"
    )
    
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=20.0,
        help="Safety limit for joint movements"
    )
    
    return parser.parse_args()


def send_to_openvla(image_array: np.ndarray, instruction: str, endpoint: str) -> Optional[Dict[str, Any]]:
    """Send image and instruction to OpenVLA endpoint."""
    try:
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
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"OpenVLA API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"OpenVLA API Exception: {e}")
        return None


class SmolVLAInference:
    """Local SmolVLA inference wrapper."""
    
    def __init__(self, model_path: str, use_cuda: bool = True):
        """Initialize SmolVLA model for inference."""
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        logger.info(f"Loading SmolVLA model from {model_path} on {self.device}")
        
        try:
            # Load pretrained SmolVLA model
            self.policy = SmolVLAPolicy.from_pretrained(model_path)
            self.policy.to(self.device)
            self.policy.eval()
            
            # Reset policy state
            self.policy.reset()
            
            logger.info("✅ SmolVLA model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load SmolVLA model: {e}")
            raise
    
    def infer_action(self, image_array: np.ndarray, joint_positions: Dict[str, float], 
                    instruction: str) -> Optional[np.ndarray]:
        """Run SmolVLA inference to get next action."""
        try:
            with torch.no_grad():
                # Prepare image tensor (convert from RGB to expected format)
                # SmolVLA expects images in range [0, 1] with shape (B, C, H, W)
                if image_array.dtype == np.uint8:
                    image_tensor = torch.from_numpy(image_array).float() / 255.0
                else:
                    image_tensor = torch.from_numpy(image_array).float()
                
                # Add batch dimension and ensure correct shape (B, C, H, W)
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
                
                image_tensor = image_tensor.to(self.device)
                
                # Prepare state tensor from joint positions
                # Extract joint values in expected order
                motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                state_values = []
                for motor_name in motor_names:
                    key = f"{motor_name}.pos"
                    if key in joint_positions:
                        state_values.append(joint_positions[key])
                    else:
                        state_values.append(0.0)  # Default value if not found
                
                state_tensor = torch.tensor(state_values, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Prepare batch dictionary
                batch = {
                    "front": image_tensor,  # Camera image
                    "observation.state": state_tensor,  # Joint positions
                    "task": instruction  # Natural language instruction
                }
                
                # Run inference
                action_tensor = self.policy.select_action(batch)
                
                # Convert to numpy array
                action_array = action_tensor.cpu().numpy()
                
                return action_array
                
        except Exception as e:
            logger.error(f"SmolVLA inference error: {e}")
            return None


def get_manual_motor_values(current_positions: Dict[str, float]) -> Optional[np.ndarray]:
    """Get motor values through manual terminal input."""
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    print("\n🎮 Manual Motor Control")
    print("Current positions:")
    for motor_name in motor_names:
        key = f"{motor_name}.pos"
        current_val = current_positions.get(key, 0.0)
        print(f"  {motor_name}: {current_val:.3f}")
    
    print("\nEnter new motor values (press Enter to keep current value, 'q' to quit):")
    
    new_values = []
    for motor_name in motor_names:
        key = f"{motor_name}.pos"
        current_val = current_positions.get(key, 0.0)
        
        while True:
            try:
                user_input = input(f"  {motor_name} [{current_val:.3f}]: ").strip()
                
                if user_input.lower() == 'q':
                    print("❌ Manual control cancelled")
                    return None
                elif user_input == "":
                    # Keep current value
                    new_values.append(current_val)
                    break
                else:
                    # Parse new value
                    new_val = float(user_input)
                    new_values.append(new_val)
                    break
            except ValueError:
                print("    ❌ Invalid number, please try again")
            except KeyboardInterrupt:
                print("\n❌ Manual control cancelled")
                return None
    
    return np.array(new_values)


def convert_action_to_robot_format(action_vector: np.ndarray) -> Dict[str, float]:
    """Convert action vector to SO-101 robot format."""
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    # Ensure we have the right number of values
    if len(action_vector) != len(motor_names):
        logger.warning(f"Expected {len(motor_names)} actions, got {len(action_vector)}")
        # Pad with zeros if too short, truncate if too long
        if len(action_vector) < len(motor_names):
            action_vector = np.pad(action_vector, (0, len(motor_names) - len(action_vector)))
        else:
            action_vector = action_vector[:len(motor_names)]
    
    # Create action dictionary
    robot_action = {}
    for i, motor_name in enumerate(motor_names):
        robot_action[f"{motor_name}.pos"] = float(action_vector[i])
    
    return robot_action


class CameraStreamer:
    """Handle live camera streaming in a separate thread."""
    
    def __init__(self, camera_index: int, fps: int = 30, window_size: tuple = (640, 480)):
        """Initialize camera streamer."""
        self.camera_index = camera_index
        self.fps = fps
        self.window_size = window_size
        self.running = False
        self.thread = None
        self.cap = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
    def start(self):
        """Start the camera streaming thread."""
        if self.running:
            return
            
        logger.info(f"🎥 Starting camera stream from index {self.camera_index}")
        
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"❌ Failed to open camera {self.camera_index}")
                return False
                
            # Configure camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_size[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.running = True
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            
            logger.info("✅ Camera streaming started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start camera streaming: {e}")
            return False
    
    def stop(self):
        """Stop the camera streaming thread."""
        if not self.running:
            return
            
        logger.info("🛑 Stopping camera stream...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
        logger.info("✅ Camera streaming stopped")
    
    def _stream_loop(self):
        """Main streaming loop (runs in separate thread)."""
        window_name = "Robot Camera Feed"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        frame_time = 1.0 / self.fps
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("❌ Failed to read camera frame")
                    time.sleep(0.1)
                    continue
                
                # Store latest frame for other uses
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Add timestamp and info overlay
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"Time: {timestamp}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Camera: {self.camera_index}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to close", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("🛑 Camera stream quit requested")
                    break
                    
                time.sleep(frame_time)
                
            except Exception as e:
                logger.error(f"❌ Camera streaming error: {e}")
                break
        
        self.running = False
        cv2.destroyWindow(window_name)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest camera frame."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None


def main():
    args = parse_args()
    
    print("🤖 Starting OpenVLA + SmolVLA SO-101 Demo")
    print(f"🧠 Inference Method: {args.method}")
    if args.method in ["openvla", "both"]:
        print(f"📡 OpenVLA Endpoint: {args.openvla_endpoint}")
    if args.method in ["smolvla", "both"]:
        print(f"🏠 SmolVLA Model: {args.smolvla_model_path}")
        print(f"🔧 Device: {'CUDA' if not args.no_cuda and torch.cuda.is_available() else 'CPU'}")
    if args.method == "manual":
        print("🎮 Manual motor control mode - no models will be loaded")
    print(f"🔌 Robot Port: {args.port}")
    print(f"📝 Instruction: {args.instruction}")
    if args.stream_camera:
        print(f"🎥 Camera streaming enabled (camera {args.camera_index}, {args.stream_fps} FPS)")
    print()
    
    # Initialize SmolVLA if needed (skip for manual mode)
    smolvla_inference = None
    if args.method in ["smolvla", "both"]:
        try:
            use_cuda = not args.no_cuda
            smolvla_inference = SmolVLAInference(args.smolvla_model_path, use_cuda)
        except Exception as e:
            logger.error(f"Failed to initialize SmolVLA: {e}")
            if args.method == "smolvla":
                return  # Exit if SmolVLA-only mode fails
            logger.info("Continuing with OpenVLA API only")
    
    # Configure and connect to robot
    robot_config = SO101FollowerConfig(
        port=args.port,
        id="second_follower",
        max_relative_target=args.max_relative_target,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=args.camera_index,
                width=640,
                height=480,
                fps=30
            )
        }
    )
    
    robot = SO101Follower(robot_config)
    
    # Initialize camera streamer if requested
    camera_streamer = None
    if args.stream_camera:
        camera_streamer = CameraStreamer(
            camera_index=args.camera_index,
            fps=args.stream_fps,
            window_size=tuple(args.stream_window_size)
        )
        if not camera_streamer.start():
            print("❌ Failed to start camera streaming, continuing without it")
            camera_streamer = None
    
    try:
        print("🔗 Connecting to robot...")
        robot.connect()
        print("✅ Robot connected!")
        
        print("🚀 Starting control mode...")
        print(f"📝 Task: {args.instruction}")
        print(f"🎯 Max steps: {args.max_steps}")
        if args.method == "manual":
            print("🎮 Manual mode: You'll be prompted to enter motor values for each step")
        if args.stream_camera:
            print("🎥 Live camera feed opened in separate window")
        print("Press Ctrl+C to exit")
        
        step_count = 0
        
        while step_count < args.max_steps:
            print(f"\n--- Step {step_count + 1}/{args.max_steps} ---")
            
            # Get current observation
            obs = robot.get_observation()
            
            # Get camera image (not needed for manual mode but good for debugging)
            camera_image = obs.get("front")
            if camera_image is None and args.method != "manual":
                print("❌ No camera image available!")
                break
            elif camera_image is not None:
                print("📷 Got camera image")
            
            # Get current joint positions for reference
            joint_positions = {k: v for k, v in obs.items() if k.endswith('.pos')}
            print(f"🦾 Current joints: {joint_positions}")
            
            action_vector = None
            
            # Manual mode - get input from user
            if args.method == "manual":
                print("🎮 Getting manual motor values...")
                action_vector = get_manual_motor_values(joint_positions)
                
                if action_vector is not None:
                    print(f"🎯 Manual action: {action_vector}")
                else:
                    print("❌ Manual input cancelled")
                    break
            
            # Try SmolVLA inference first if enabled
            elif args.method in ["smolvla", "both"] and smolvla_inference is not None:
                print("🧠 Running SmolVLA inference...")
                action_vector = smolvla_inference.infer_action(camera_image, joint_positions, args.instruction)
                
                if action_vector is not None:
                    print(f"🎯 SmolVLA action: {action_vector}")
                else:
                    print("❌ SmolVLA inference failed")
            
            # Fall back to OpenVLA API if SmolVLA failed or if using both/API-only
            if action_vector is None and args.method in ["openvla", "both"]:
                print("🧠 Calling OpenVLA API...")
                openvla_response = send_to_openvla(camera_image, args.instruction, args.openvla_endpoint)
                
                if openvla_response is not None and "action" in openvla_response:
                    action_vector = np.array(openvla_response["action"])
                    print(f"🎯 OpenVLA action: {action_vector}")
                else:
                    print("❌ OpenVLA API request failed")
            
            # Execute action if we got one
            if action_vector is not None:
                # Convert to robot action format
                robot_action = convert_action_to_robot_format(action_vector)
                print(f"🤖 Robot action: {robot_action}")
                
                # Execute action
                executed_action = robot.send_action(robot_action)
                print(f"✅ Executed: {executed_action}")
            else:
                print("❌ No valid action obtained, stopping")
                break
            
            # Wait before next step
            time.sleep(1.0 / args.fps)
            step_count += 1
        
        print(f"\n🏁 Completed {step_count} steps")
        
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        raise
    finally:
        # Stop camera streaming
        if camera_streamer:
            camera_streamer.stop()
        
        print("🔌 Disconnecting robot...")
        robot.disconnect()
        print("✅ Done!")


if __name__ == "__main__":
    main() 