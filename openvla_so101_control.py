#!/usr/bin/env python3
"""
OpenVLA SO-101 Robot Control Script

This script connects to your SO-101 robot arm and uses OpenVLA predictions
to control the robot based on camera images and text instructions.

Example usage:
python openvla_so101_control.py \
    --endpoint "https://andreemic--openvla-robot-get-next-gripper-position.modal.run" \
    --port "/dev/tty.usbmodem5A680114471" \
    --instruction "pick up the red cube" \
    --camera_index 0 \
    --fps 5

Requirements:
- pip install lerobot requests opencv-python pillow numpy
"""

import argparse
import base64
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import requests
from PIL import Image

# Import LeRobot components
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.common.utils.utils import init_logging


# Configure logging
formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class OpenVLAClient:
    """Client for interacting with OpenVLA endpoint."""
    
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def predict(self, image: np.ndarray, instruction: str) -> Dict[str, Any]:
        """
        Send image and instruction to OpenVLA and get action prediction.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            instruction: Text instruction for the robot
            
        Returns:
            Dictionary containing action prediction and metadata
        """
        # Convert image to base64
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Create request payload
        payload = {
            "image": img_str,
            "instruction": instruction
        }
        
        # Time the request
        start_time = time.time()
        
        try:
            response = self.session.post(
                self.endpoint_url,
                data=json.dumps(payload),
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                result["response_time_ms"] = (end_time - start_time) * 1000
                return result
            else:
                logger.error(f"OpenVLA API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None


class SO101OpenVLAController:
    """Controller that integrates OpenVLA with SO-101 robot arm."""
    
    def __init__(
        self,
        robot_port: str,
        openvla_endpoint: str,
        camera_index: int = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        max_relative_target: Optional[float] = None
    ):
        self.openvla_client = OpenVLAClient(openvla_endpoint)
        
        # Configure robot
        robot_config = SO101FollowerConfig(
            port=robot_port,
            id="openvla_so101",
            max_relative_target=max_relative_target,
            cameras={
                "front": OpenCVCameraConfig(
                    index_or_path=camera_index,
                    width=camera_width,
                    height=camera_height,
                    fps=30
                )
            }
        )
        
        self.robot = SO101Follower(robot_config)
        self.is_connected = False
        
        # Action mapping - convert OpenVLA output to SO-101 format
        self.motor_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex", 
            "wrist_flex", "wrist_roll", "gripper"
        ]
        
    def connect(self):
        """Connect to robot and camera."""
        if not self.is_connected:
            logger.info("Connecting to SO-101 robot...")
            self.robot.connect()
            self.is_connected = True
            logger.info("Robot connected successfully!")
        
    def disconnect(self):
        """Disconnect from robot."""
        if self.is_connected:
            logger.info("Disconnecting from robot...")
            self.robot.disconnect()
            self.is_connected = False
            logger.info("Robot disconnected.")
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current robot observation including joint positions and camera image."""
        if not self.is_connected:
            raise RuntimeError("Robot not connected!")
            
        obs = self.robot.get_observation()
        return obs
    
    def convert_openvla_action(self, action_vector: list) -> Dict[str, float]:
        """
        Convert OpenVLA action vector to SO-101 joint positions.
        
        Args:
            action_vector: List of action values from OpenVLA
            
        Returns:
            Dictionary with motor names and target positions
        """
        if len(action_vector) != len(self.motor_names):
            logger.warning(f"Action vector length {len(action_vector)} doesn't match expected {len(self.motor_names)}")
            # Pad or truncate as needed
            if len(action_vector) < len(self.motor_names):
                action_vector.extend([0.0] * (len(self.motor_names) - len(action_vector)))
            else:
                action_vector = action_vector[:len(self.motor_names)]
        
        # Create action dictionary
        action = {}
        for i, motor_name in enumerate(self.motor_names):
            action[f"{motor_name}.pos"] = float(action_vector[i])
            
        return action
    
    def execute_action(self, action: Dict[str, float]) -> Dict[str, float]:
        """Execute action on robot."""
        if not self.is_connected:
            raise RuntimeError("Robot not connected!")
            
        return self.robot.send_action(action)
    
    def run_control_loop(
        self,
        instruction: str,
        fps: float = 5.0,
        duration: Optional[float] = None,
        verbose: bool = True
    ):
        """
        Main control loop that captures images, gets predictions, and controls robot.
        
        Args:
            instruction: Text instruction for the robot
            fps: Control loop frequency
            duration: Maximum duration to run (None for unlimited)
            verbose: Whether to print detailed logs
        """
        if not self.is_connected:
            self.connect()
            
        logger.info(f"Starting control loop with instruction: '{instruction}'")
        logger.info(f"Running at {fps} FPS")
        
        start_time = time.time()
        step_count = 0
        
        try:
            while True:
                loop_start = time.time()
                
                # Get observation (includes camera image and joint positions)
                obs = self.get_observation()
                
                # Extract camera image
                camera_image = obs.get("front")
                if camera_image is None:
                    logger.error("No camera image available!")
                    time.sleep(1/fps)
                    continue
                
                # Get prediction from OpenVLA
                if verbose:
                    logger.info(f"Step {step_count}: Getting prediction from OpenVLA...")
                
                prediction = self.openvla_client.predict(camera_image, instruction)
                
                if prediction is None:
                    logger.error("Failed to get prediction from OpenVLA!")
                    time.sleep(1/fps)
                    continue
                
                # Extract action vector
                action_vector = prediction.get("action", [])
                if not action_vector:
                    logger.error("No action vector in prediction!")
                    time.sleep(1/fps)
                    continue
                
                # Convert to robot action format
                robot_action = self.convert_openvla_action(action_vector)
                
                # Execute action
                executed_action = self.execute_action(robot_action)
                
                if verbose:
                    logger.info(f"Step {step_count}: Executed action in {prediction.get('response_time_ms', 0):.1f}ms")
                    logger.info(f"Action: {robot_action}")
                
                step_count += 1
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    logger.info(f"Reached duration limit of {duration}s")
                    break
                
                # Maintain FPS
                loop_time = time.time() - loop_start
                sleep_time = (1.0 / fps) - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("Control loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in control loop: {e}")
            raise
        finally:
            logger.info(f"Control loop finished after {step_count} steps")


def main():
    parser = argparse.ArgumentParser(description="Control SO-101 robot arm with OpenVLA")
    
    # Required arguments
    parser.add_argument(
        "-e", "--endpoint", 
        required=True,
        help="OpenVLA endpoint URL"
    )
    parser.add_argument(
        "-p", "--port",
        required=True,
        help="Robot serial port (e.g., /dev/tty.usbmodem5A680114471)"
    )
    parser.add_argument(
        "-i", "--instruction",
        required=True,
        help="Text instruction for the robot (e.g., 'pick up the red cube')"
    )
    
    # Optional arguments
    parser.add_argument(
        "-c", "--camera_index",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "-f", "--fps",
        type=float,
        default=5.0,
        help="Control loop frequency in Hz (default: 5.0)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        help="Maximum duration to run in seconds (default: unlimited)"
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=640,
        help="Camera width (default: 640)"
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=480,
        help="Camera height (default: 480)"
    )
    parser.add_argument(
        "--max_relative_target",
        type=float,
        help="Maximum relative target for safety (default: None)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    init_logging()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create controller
    controller = SO101OpenVLAController(
        robot_port=args.port,
        openvla_endpoint=args.endpoint,
        camera_index=args.camera_index,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        max_relative_target=args.max_relative_target
    )
    
    try:
        # Connect to robot
        controller.connect()
        
        # Run control loop
        controller.run_control_loop(
            instruction=args.instruction,
            fps=args.fps,
            duration=args.duration,
            verbose=args.verbose
        )
        
    finally:
        # Ensure robot is disconnected
        controller.disconnect()


if __name__ == "__main__":
    main()