import itertools
import os
import traceback

from .api import RobotInterface
from .gripper import GRIPPER_CLOSED
from .safety import DEFAULT_BOX, DEFAULT_MARGIN
from .visualize import RobotMirror, VisualizerHandle
from .xarm7_mujoco import SimulatedXArm7
from .xarm7_real import RealXArm7

class Robot(RobotInterface):
    """The arm, real when ROBOT_IP is set and simulated otherwise.

    Either way it comes with a meshcat scene showing where the arm is. The
    simulator draws itself off the model it is stepping; the real arm has a
    `RobotMirror` built for it, which reads the measured joint angles and runs
    forward kinematics on a copy of the model. Pass `visualize=False` for a
    run with nobody watching.

    `register_visualizer` adds other things to that scene — see `visualize.py`.
    Those only arrive after construction, so the browser tab can be held back
    until there is something more than the arm in it: the first
    `register_visualizer` opens it, or `show()` does. The scene is published
    and its URL printed either way.

    Which is the default depends on which arm you got. The simulator is the
    picture — a run with no tab is a run with nothing to see — so it opens one
    straight away. The real arm is in the room, and a browser tab appearing
    over whatever is on screen is an interruption, so it waits for the thing
    the scene was opened for. `open_browser=True`/`False` says outright.
    """

    def __init__(self, sim=False, safety_box=DEFAULT_BOX,
                 safety_margin=DEFAULT_MARGIN, guard=True, q0=None,
                 visualize=True, open_browser=None, zmq_url=None, gripper=True):
        super().__init__()
        safety = dict(safety_box=safety_box, safety_margin=safety_margin, guard=guard)
        ip = os.environ.get('ROBOT_IP', '').strip()
        self._mirror = None
        self._visualizers = []
        real = bool(ip) and not sim
        if open_browser is None:
            open_browser = not real
        # Only a deferral: with no scene at all there is nothing to open later.
        self._deferred_open = bool(visualize) and not open_browser
        if real:
            self.robot = RealXArm7(ip, gripper=gripper, **safety)
            if visualize:
                self._mirror = RobotMirror(
                    self.robot, guard=self.robot.guard, zmq_url=zmq_url,
                    open_browser=open_browser
                )
        else:
            self.robot = SimulatedXArm7(
                visualize=visualize, open_browser=open_browser, q0=q0,
                gripper=gripper, zmq_url=zmq_url, **safety
            )

    # ------------------------------------------------------------------
    # the picture
    # ------------------------------------------------------------------

    @property
    def viewer(self):
        """The `MeshcatVisualizer` drawing this arm, or None.

        Read it again every time you need it rather than keeping it: a viewer
        whose browser has gone away is replaced by None, and a reference held
        across that keeps writing to a socket nobody is listening to.
        """
        owner = self._mirror if self._mirror is not None else self.robot
        return getattr(owner, "viz", None)

    @property
    def base_frame(self):
        """The arm's base frame in the world the picture is drawn in, as a 4x4.

        Joint angles, the safety box and `HandEye.points_to_base` all speak the
        base frame; an MJCF is free to put it anywhere, and this one puts it
        120 mm up. Anything drawn in base coordinates goes through here.
        """
        owner = self._mirror if self._mirror is not None else self.robot
        return owner.base_frame

    @property
    def guard(self):
        """The `SafetyGuard` checking this arm's setpoints, or None."""
        return self.robot.guard

    @property
    def visualizers(self):
        """The visualizers currently registered, as their handles."""
        return list(self._visualizers)

    def show(self):
        """Open the scene in a browser tab, and stop holding the tab back.

        Only needed after `open_browser=False`, and only for a scene nothing
        was registered into — the first `register_visualizer` opens the tab by
        itself. Idempotent, and returns the URL.
        """
        self._deferred_open = False
        viewer = self.viewer
        if viewer is None:
            print(
                "[xarm7_lib] nothing to show: this Robot was built with "
                "visualize=False, so no scene was published"
            )
            return None
        opener = getattr(viewer, "open", None)
        if opener is not None:
            opener()
        return getattr(viewer, "url", None)

    def register_visualizer(self, visualizer, rate=None, name=None):
        """Draw `visualizer`'s idea of the scene beside the arm, on its own thread.

        The object needs a `poll()` — do the slow work there, it is called with
        no lock held — and a `geometry()` returning `{name: (geometry, material,
        transform)}` for what should be on screen now. A `frame` attribute of
        "camera" has its coordinates carried through the bench's hand-eye
        calibration, so a detector hands over the numbers it produced and
        nothing else. `visualize.Visualizer` is the full contract.

        A visualizer drawing in camera coordinates should say which camera it
        is looking through, as a `serial` or a `camera` exposing one: that is
        how the calibration is found. Asking the camera instead means opening
        it, and the detector already has it open.

        A tab held back by `open_browser=False` opens here, on the first
        registration: this is the thing it was waiting for.

        Returns a handle; `stop()` it to take the visualizer off the scene.
        `close()` stops every one of them.
        """
        label = name or type(visualizer).__name__.lower()
        taken = {handle.name for handle in self._visualizers}
        if label in taken:
            label = next(
                f"{label}-{n}" for n in itertools.count(2) if f"{label}-{n}" not in taken
            )

        if self.viewer is None:
            # Still started, deliberately. poll() is what drives a detector, so
            # a registration that quietly did nothing would stop it producing
            # readings at all — and the detector is the last place anyone would
            # look for a missing browser.
            print(
                f"[xarm7_lib] no viewer, so {label} will not be drawn; "
                "it is still running and still readable"
            )

        handle = VisualizerHandle(self, visualizer, label, rate=rate)
        self._visualizers.append(handle)
        if self._deferred_open:
            self.show()
        return handle

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

        Registered visualizers are stopped first, because one may be reading
        the arm, and the viewer goes last, so the final thing on screen is the
        arm where it stopped. Every step runs even if an earlier one fails:
        this is reached from `__exit__`, including on a Ctrl-C, and must not
        raise over the top of whatever got us here. Safe to call twice.
        """
        for handle in reversed(self._visualizers):
            try:
                handle.stop()
            except Exception:
                print(f"[xarm7_lib] stopping {handle.name} raised:\n{traceback.format_exc()}")
        self._visualizers.clear()

        try:
            return self.robot.close()
        finally:
            if self._mirror is not None:
                mirror, self._mirror = self._mirror, None
                try:
                    mirror.close()
                except Exception:
                    print(f"[xarm7_lib] closing the mirror raised:\n{traceback.format_exc()}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def free_drive(self, duration=None, **options):
        """Release the arm so it can be pushed by hand, and start recording.

        Returns straight away; `end_free_drive` locks the arm again and hands
        back the recording, so the waiting in between is yours:

            robot.free_drive()
            print("push the arm around; press Enter when you are done")
            input()
            traj = robot.end_free_drive()

        Every joint is free unless `watch=` says otherwise, and by default
        there is no time limit. Real arm only, and deliberately so: there is
        nothing to push in simulation. See `RealXArm7.free_drive` for the
        options.
        """
        self._free_drive_backend()
        return self.robot.free_drive(duration, **options)

    def end_free_drive(self):
        """End the run `free_drive` started and return its `Trajectory`.

        The arm is locked where it stands on the way out. If a watched joint
        was pushed off its lock, or the arm ended up outside the safety guard,
        this is what asks for hands off the robot and puts it right — see
        `RealXArm7.end_free_drive`.
        """
        return self._free_drive_backend().end_free_drive()

    def _free_drive_backend(self):
        """The backend, if it can be hand-guided at all."""
        if not hasattr(self.robot, "free_drive"):
            raise NotImplementedError(
                "free drive needs the real arm — there is nothing to push in "
                "simulation. Set ROBOT_IP and construct Robot() without sim=True."
            )
        return self.robot


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
