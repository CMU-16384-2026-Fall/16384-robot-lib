import os

from .api import RobotInterface
from .safety import DEFAULT_BOX, DEFAULT_MARGIN
from .xarm7_mujoco import SimulatedXArm7
from .xarm7_real import RealXArm7

class Robot(RobotInterface):
    def __init__(self, sim=False, safety_box=DEFAULT_BOX,
                 safety_margin=DEFAULT_MARGIN, guard=True):
        super().__init__()
        safety = dict(safety_box=safety_box, safety_margin=safety_margin, guard=guard)
        ip = os.environ.get('ROBOT_IP', '').strip()
        if ip and not sim:
            self.robot = RealXArm7(ip, **safety)
        else:
            self.robot = SimulatedXArm7(visualize=True, **safety)

    @property
    def joint_values(self):
        return self.robot.joint_values

    @property
    def joint_velocities(self):
        return self.robot.joint_velocities

    @property
    def joint_efforts(self):
        return self.robot.joint_efforts

    @property
    def joint_torques(self):
        return self.robot.joint_torques

    def set_joint_targets(self, joints, speed=None, wait=True, timeout=None):
        return self.robot.set_joint_targets(joints, speed=speed, wait=wait, timeout=timeout)

    def set_velocity(self, speeds, duration=0, wait=True):
        return self.robot.set_velocity(speeds, duration=duration, wait=wait)

    def servo_joints(self, joints, velocities=None):
        return self.robot.servo_joints(joints, velocities=velocities)

    def stop(self, wait=True, timeout=None):
        return self.robot.stop(wait=wait, timeout=timeout)

    def check_safety(self, joints):
        return self.robot.check_safety(joints)

if __name__ == '__main__':
    import numpy as np
    import time

    robot = Robot()
    robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0], wait=True)

    t = 0
    try:
        while True:
            joints = [
                .1 * np.sin(t),
                .1 * (np.cos(t) - 1),
                -.1 * np.sin(t),
                .1 * (1 - np.cos(t)),
                .1 * np.sin(t),
                .1 * np.sin(t),
                .1 * np.sin(t),
            ]
            robot.servo_joints(joints)

            time.sleep(0.05)
            t += 0.05
    except KeyboardInterrupt:
        # The sleep above is the caller's, not the arm's, so the interrupt
        # lands here rather than inside a motion call.
        robot.stop()
