#!/usr/bin/env python3
"""
Gripper Controller

Real-time gripper control using arrow keys:
- Right Arrow: Close gripper (increase percentage)
- Left Arrow: Open gripper (decrease percentage)
- Up/Down Arrow: Large steps
- Space: Go to middle position (50%)
- Q: Quit

Gripper position is controlled as percentage (0-100%):
- 0% = Fully open
- 100% = Fully closed

Requirements:
- pip install lerobot keyboard
"""

import time
import os
import sys
from typing import Dict, Any

try:
    import keyboard
except ImportError:
    print("❌ Please install keyboard library: pip install keyboard")
    sys.exit(1)

from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ===== CONFIGURATION =====
ROBOT_PORT = "/dev/tty.usbmodem5A460857481"  # Update this to match your robot

# Gripper configuration from your JSON
GRIPPER_MIN = 2033  # Fully open position
GRIPPER_MAX = 3534  # Fully closed position
GRIPPER_HOMING = 1159  # Home/reference position

# Control settings
SMALL_STEP = 2  # Small step size (%)
LARGE_STEP = 10  # Large step size (%)
UPDATE_RATE = 0.1  # How often to check for key presses (seconds)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def percentage_to_servo_position(percentage: float) -> float:
    """Convert 0-100% to actual servo position."""
    if percentage < 0:
        percentage = 0
    elif percentage > 100:
        percentage = 100
    
    # Linear interpolation between min and max
    servo_pos = GRIPPER_MIN + (percentage / 100.0) * (GRIPPER_MAX - GRIPPER_MIN)
    return servo_pos

def servo_position_to_percentage(servo_pos: float) -> float:
    """Convert servo position to 0-100%."""
    if servo_pos <= GRIPPER_MIN:
        return 0.0
    elif servo_pos >= GRIPPER_MAX:
        return 100.0
    
    percentage = ((servo_pos - GRIPPER_MIN) / (GRIPPER_MAX - GRIPPER_MIN)) * 100.0
    return percentage

def format_status(current_percentage: float, target_percentage: float, servo_pos: float) -> str:
    """Format the current status display."""
    lines = []
    
    lines.append("=" * 60)
    lines.append("🤖 SO-101 Gripper Controller")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Current Position: {current_percentage:6.1f}% (Servo: {servo_pos:6.0f})")
    lines.append(f"Target Position:  {target_percentage:6.1f}%")
    lines.append("")
    
    # Visual bar
    bar_length = 40
    filled_length = int(bar_length * current_percentage / 100)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    lines.append(f"[{bar}] {current_percentage:5.1f}%")
    lines.append("")
    
    lines.append("Controls:")
    lines.append("  ◀️  Left Arrow  - Open gripper (decrease %)")
    lines.append("  ▶️  Right Arrow - Close gripper (increase %)")
    lines.append("  🔺 Up Arrow    - Large step close")
    lines.append("  🔻 Down Arrow  - Large step open")
    lines.append("  ⏺️  Space       - Go to middle (50%)")
    lines.append("  ❌ Q           - Quit")
    lines.append("")
    lines.append(f"Timestamp: {time.strftime('%H:%M:%S')}")
    lines.append("-" * 60)
    
    return "\n".join(lines)

def main():
    print("🤖 Starting SO-101 Gripper Controller")
    print(f"🔌 Robot Port: {ROBOT_PORT}")
    print("📋 Initializing...")
    
    # Configure robot (no cameras needed for gripper control)
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id="second_follower",
        max_relative_target=20.0
    )
    
    robot = SO101Follower(robot_config)
    target_percentage = 50.0  # Start at middle position
    
    try:
        # Connect to robot
        robot.connect()
        print("✅ Robot connected!")
        print("🎮 Starting gripper control...")
        print("   Press any arrow key to begin, 'q' to quit")
        time.sleep(1)
        
        while True:
            # Get current robot state
            obs = robot.get_observation()
            current_servo_pos = obs.get("gripper.pos", GRIPPER_HOMING)
            current_percentage = servo_position_to_percentage(current_servo_pos)
            
            # Clear screen and show status
            clear_screen()
            status = format_status(current_percentage, target_percentage, current_servo_pos)
            print(status)
            
            # Check for keyboard input
            if keyboard.is_pressed('q'):
                print("Quitting...")
                break
            elif keyboard.is_pressed('right'):
                target_percentage = min(100.0, target_percentage + SMALL_STEP)
                print(f"→ Closing to {target_percentage:.1f}%")
            elif keyboard.is_pressed('left'):
                target_percentage = max(0.0, target_percentage - SMALL_STEP)
                print(f"← Opening to {target_percentage:.1f}%")
            elif keyboard.is_pressed('up'):
                target_percentage = min(100.0, target_percentage + LARGE_STEP)
                print(f"⬆ Large close to {target_percentage:.1f}%")
            elif keyboard.is_pressed('down'):
                target_percentage = max(0.0, target_percentage - LARGE_STEP)
                print(f"⬇ Large open to {target_percentage:.1f}%")
            elif keyboard.is_pressed('space'):
                target_percentage = 50.0
                print("⏺ Going to middle position (50%)")
            
            # Send command to gripper
            target_servo_pos = percentage_to_servo_position(target_percentage)
            robot_action = {"gripper.pos": target_servo_pos}
            
            try:
                robot.send_action(robot_action)
            except Exception as e:
                print(f"⚠️ Action failed: {e}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(UPDATE_RATE)
            
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
    # Check if running with proper permissions (keyboard library needs it)
    if os.name != 'nt':  # Not Windows
        if os.geteuid() != 0:
            print("⚠️  Note: You might need to run with sudo for keyboard access")
            print("   Try: sudo python gripper_controller.py")
            print("   Or install without sudo: pip install --user keyboard")
    
    main() 