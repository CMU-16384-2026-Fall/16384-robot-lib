class RobotInterface:
    def __init__(self):
        pass

    @property
    def joint_values(self):
        # Get current joint values from the robot
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def joint_velocities(self):
        # Get current joint velocities from the robot
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def joint_efforts(self):
        # Get current joint efforts from the robot
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def joint_torques(self):
        # Get current joint torques from the robot
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_joint_targets(self, joints, speed=None, wait=True, timeout=None):
        """
        Set joint targets for the robot.

        Args:
            joints: A list or array of joint positions.
            speed: The speed at which to move the joints.
            wait: If True, block until the robot reaches the target.
            timeout: The maximum time to wait for the robot to reach the target.

        Returns:
            True if the robot reached the target, False otherwise.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_velocity(self, speeds, duration=0, wait=True):
        """
        Set joint velocity commands for the robot.

        Args:
            speeds: A list or array of joint velocities.
            duration: The duration of the velocity command in seconds. If 0, the command is indefinite.
            wait: If True and duration is finite and non-zero, block until the command expires. If False, return immediately.

        Returns:
            None if wait=False, or a boolean indicating whether the command completed successfully.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def check_safety(self, joints):
        """
        Check whether the arm is allowed to be at the given configuration.

        The same check every motion command applies, exposed so a target can be
        tested before committing to it.

        Args:
            joints: A list or array of joint positions.

        Returns:
            A Violation saying what is wrong, or None if the configuration is allowed.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def servo_joints(self, joints, velocities=None):
        """
        Servo joints to the specified positions with optional feedforward velocities.
        Returns immediately and applies the setpoint as given. Pass velocities to
        feed the intended joint velocity forward; otherwise it is estimated from
        successive commands.

        Args:
            joints: A list or array of joint positions.
            velocities: A list or array of joint velocities to use as feedforward.

        Returns:
            None
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def stop(self, wait=True, timeout=None):
        """
        Stop moving and hold the current position.

        Cancels whatever command is running. A keyboard interrupt during a
        blocking motion call does the same thing before it propagates.

        Args:
            wait: If True, block until the arm has come to rest.
            timeout: The maximum time to wait for the arm to come to rest.

        Returns:
            True if the arm came to rest, False if wait was False or if
            timeout elapsed first.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
