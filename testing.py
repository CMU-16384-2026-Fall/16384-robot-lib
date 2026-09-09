import sys

from xarm7_lib import RealXArm7
IP = "192.168.1.?"
if "?" in IP:
    sys.exit(
        f"Robot IP is not set (got '{IP}' from testing).\n"
        "Edit the IP variable in testing.py and replace the '?' "
        "with your arm's address, displayed behind the robot control box."
    )

robot = RealXArm7(ip=IP)
robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0])
