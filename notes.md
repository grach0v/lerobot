USB Port:/dev/tty.usbmodem5A680114471


Motor calibration leader:
(.venv) mikhailandreev:lerobot/ (main✗) $ python -m lerobot.setup_motors \                                                                          [13:52:32]
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A680114471 # paste here the port found at previous step
Connect the controller board to the 'gripper' motor only and press enter.
'gripper' motor id set to 6
Connect the controller board to the 'wrist_roll' motor only and press enter.
'wrist_roll' motor id set to 5
Connect the controller board to the 'wrist_flex' motor only and press enter.
'wrist_flex' motor id set to 4
Connect the controller board to the 'elbow_flex' motor only and press enter.
'elbow_flex' motor id set to 3
Connect the controller board to the 'shoulder_lift' motor only and press enter.
'shoulder_lift' motor id set to 2
Connect the controller board to the 'shoulder_pan' motor only and press enter.
'shoulder_pan' motor id set to 1


python -m lerobot.calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A680114471 \ # <- The port of your robot
    --robot.id=my_awesome_follower_arm # <- Give the robot a unique name

python -m lerobot.calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A680114471 \
    --teleop.id=manual_override_leader_arm