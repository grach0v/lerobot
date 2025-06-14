#!/usr/bin/env python3
"""
Simple script to verify LeRobot setup is working correctly.
"""

import sys
import os

def check_setup():
    print("🔍 Checking LeRobot setup...")
    print(f"📍 Current directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    print(f"🐍 Python executable: {sys.executable}")
    
    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        print(f"📁 Virtual env path: {sys.prefix}")
    else:
        print("❌ Virtual environment is NOT active")
        return False
    
    # Check LeRobot import
    try:
        import lerobot
        print("✅ LeRobot import successful")
    except ImportError as e:
        print(f"❌ LeRobot import failed: {e}")
        return False
    
    # Check key dependencies
    dependencies = ['torch', 'gymnasium', 'datasets', 'huggingface_hub']
    for dep in dependencies:
        try:
            module = __import__(dep)
            if hasattr(module, '__version__'):
                print(f"✅ {dep}: {module.__version__}")
            else:
                print(f"✅ {dep}: imported successfully")
        except ImportError:
            print(f"❌ {dep}: not found")
            return False
    
    print(f"\n🎉 Setup verification complete!")
    print(f"📂 Run examples from: {os.path.join(os.getcwd(), 'examples')}")
    return True

if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1) 