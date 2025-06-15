#!/usr/bin/env python3
"""
Robot Position Monitor

Continuously reads and displays SO-101 robot joint positions in a table format.
Updates every second. Press Ctrl+C to stop.

Requirements:
- pip install lerobot
"""

import time
import os
from typing import Dict, Any

from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ===== CONFIGURATION =====
ROBOT_PORT = "/dev/tty.usbmodem5A460857481"  # Update this to match your robot
UPDATE_INTERVAL = 1.0  # Update every second

# Joint names in order
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift", 
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_position_table(joint_positions: Dict[str, float], step_count: int) -> str:
    """Format joint positions into a nice table."""
    table_lines = []
    
    # Header
    table_lines.append("=" * 60)
    table_lines.append(f"🤖 SO-101 Robot Position Monitor (Step {step_count})")
    table_lines.append("=" * 60)
    table_lines.append(f"{'Joint Name':<15} {'Position':<12} {'Raw Value':<12}")
    table_lines.append("-" * 60)
    
    # Joint positions
    for joint_name in JOINT_NAMES:
        position_key = f"{joint_name}.pos"
        if position_key in joint_positions:
            position = joint_positions[position_key]
            table_lines.append(f"{joint_name:<15} {position:>8.1f}    {int(position):>8}")
        else:
            table_lines.append(f"{joint_name:<15} {'N/A':<12} {'N/A':<12}")
    
    table_lines.append("-" * 60)
    table_lines.append(f"Timestamp: {time.strftime('%H:%M:%S')}")
    table_lines.append("Press Ctrl+C to stop monitoring")
    table_lines.append("")
    
    return "\n".join(table_lines)

def main():
    print("🤖 Starting SO-101 Position Monitor")
    print(f"🔌 Robot Port: {ROBOT_PORT}")
    print(f"⏱️  Update Interval: {UPDATE_INTERVAL}s")
    print("\nConnecting to robot...")
    
    # Configure robot
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id="second_follower",
        max_relative_target=20.0
    )
    
    robot = SO101Follower(robot_config)
    step_count = 0
    
    try:
        # Connect to robot
        robot.connect()
        print("✅ Robot connected! Starting monitoring...\n")
        time.sleep(1)
        
        while True:
            step_count += 1
            
            # Get current robot observation
            obs = robot.get_observation()
            
            # Extract joint positions
            joint_positions = {k: v for k, v in obs.items() if k.endswith('.pos')}
            
            # Clear screen and display table
            clear_screen()
            table = format_position_table(joint_positions, step_count)
            print(table)
            
            # Wait for next update
            time.sleep(UPDATE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n⛔ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        print("🔌 Disconnecting robot...")
        robot.disconnect()
        print("✅ Done!")

if __name__ == "__main__":
    main() 