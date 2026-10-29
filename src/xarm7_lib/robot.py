import os

from .api import RobotInterface
from .gripper import GRIPPER_CLOSED
from .safety import DEFAULT_BOX, DEFAULT_MARGIN
from .xarm7_mujoco import SimulatedXArm7
from .xarm7_real import RealXArm7

class Robot(RobotInterface):
    """The arm, real when ROBOT_IP is set and simulated otherwise."""
    def __init__(self, sim=False, safety_box=DEFAULT_BOX,
                 safety_margin=DEFAULT_MARGIN, guard=True, q0=None,
                 gripper=True):
        super().__init__()
        safety = dict(safety_box=safety_box, safety_margin=safety_margin, guard=guard)
        ip = os.environ.get('ROBOT_IP', '').strip()
        if ip and not sim:
            self.robot = RealXArm7(ip, gripper=gripper, **safety)
        else:
            self.robot = SimulatedXArm7(
                visualize=True, q0=q0, gripper=gripper, **safety
            )

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

    @property
    def gripper_opening(self):
        return self.robot.gripper_opening

    def grip(self, opening=GRIPPER_CLOSED, speed=None, wait=True, timeout=None):
        return self.robot.grip(opening, speed=speed, wait=wait, timeout=timeout)

    def stop(self, wait=True, timeout=None):
        return self.robot.stop(wait=wait, timeout=timeout)

    def check_safety(self, joints):
        return self.robot.check_safety(joints)

    def close(self):
        """Release the backend: the connection, or the simulation and viewer.

        The real arm is left stopped with its motors still holding — see
        `RealXArm7.close` for why it is not switched off.
        """
        return self.robot.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def free_drive(self, duration=30.0, **options):
        """Hand-guide the arm and record where it went.

        Every joint is free unless `watch=` says otherwise, and `duration=None`
        runs until something stops it. Real arm only, and deliberately so:
        there is nothing to push in simulation. See `RealXArm7.free_drive` for
        the options.
        """
        if not hasattr(self.robot, "free_drive"):
            raise NotImplementedError(
                "free drive needs the real arm — there is nothing to push in "
                "simulation. Set ROBOT_IP and construct Robot() without sim=True."
            )
        return self.robot.free_drive(duration, **options)


if __name__ == '__main__':
    import numpy as np
    import time

    # `with` closes the backend on the way out — the connection for the real
    # arm, the simulation thread and viewer for the sim — including when the
    # loop below is interrupted.
    with Robot() as robot:
        robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0], wait=True)

        robot.grip(1.0)  # jaws open
        robot.grip(0.0)  # and shut again

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
