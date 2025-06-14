#!/usr/bin/env python3
"""
Simple script to test SO-101 robot connection and show status.
"""

from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig

def test_robot_connection():
    print("🤖 Testing SO-101 Robot Connection")
    print("=" * 40)
    
    # Configure robot
    config = SO101FollowerConfig(
        port="/dev/tty.usbmodem5A460857481",  # Your robot port
        id="connection_test"
    )
    
    robot = SO101Follower(config)
    
    try:
        print("🔌 Attempting to connect...")
        robot.connect()
        print("✅ Robot connected successfully!")
        
        print(f"🔗 Connection status: {robot.is_connected}")
        print(f"📏 Calibration status: {robot.is_calibrated}")
        
        # Try to read current position
        print("\n📊 Reading current joint positions...")
        observation = robot.get_observation()
        
        # Display joint positions
        joint_positions = {k: v for k, v in observation.items() if k.endswith('.pos')}
        print("🦾 Current joint positions:")
        for joint, position in joint_positions.items():
            print(f"   {joint}: {position:.2f}")
        
        # Check motors
        print(f"\n⚙️  Motors configured: {list(robot.bus.motors.keys())}")
        print(f"🎯 Action features: {list(robot.action_features.keys())}")
        
        print("\n✅ All tests passed! Robot is working correctly.")
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check that robot is powered on")
        print("2. Verify USB cable is connected")
        print("3. Make sure no other programs are using the robot")
        print("4. Try unplugging and reconnecting the robot")
        return False
    
    finally:
        if robot.is_connected:
            print("\n🔌 Disconnecting robot...")
            robot.disconnect()
            print("✅ Robot disconnected safely.")
    
    return True

if __name__ == "__main__":
    success = test_robot_connection()
    exit(0 if success else 1) 