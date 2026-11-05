"""Real xArm7 over the UFACTORY controller, implementing the RobotInterface API.

This is the hardware twin of `xarm7_mujoco.SimulatedXArm7`: same methods, same
units (radians, radians/second, newton-metres), same joint ordering. Swapping
one for the other should be a one-line change in a script.

Everything goes through `xarm.wrapper.XArmAPI`. The controller runs the servo
loop itself, so unlike the simulator there is no reference shaping or gravity
feedforward here — the three motion commands are just the controller's three
motion modes, and the work is in switching between them cleanly:

    set_joint_targets  mode 0, position control. `set_servo_angle` plans a
                       time-synchronized joint move at the requested speed and
                       the controller's acceleration.
    set_velocity       mode 4, joint velocity control. `vc_set_joint_velocity`
                       holds the commanded speeds until they are replaced or
                       until `duration` expires (firmware >= 1.8.0 expires them
                       on the controller; older firmware is timed here).
    servo_joints       mode 1, servo streaming. `set_servo_angle_j` applies the
                       setpoint at the next control tick with no planning.

`free_drive` adds a fourth: mode 2, joint teaching, in which the controller
gravity-compensates the arm and lets a person push it around. Nothing is
commanded in that mode, so there is nothing to check before sending: the guard
is run against what the arm reports instead, and a sample it rejects ends the
run. It is the one call here that leaves something running behind it — a thread
sampling and watching the arm until `end_free_drive` locks it again and hands
back the recording. See `free_drive.py` for the rest of what it does.

A mode change is only accepted while the arm is stopped, so `_ensure_mode`
first cancels whatever is running — zero velocities in mode 4, a state-4 abort
of the queued trajectory in mode 0 — and waits for the joints to come to rest.
That makes a new command supersede the previous one, which is what the
simulator does too.

State comes off the controller's report stream, so the readings are up to one
report period stale — and that period is a good deal longer than it takes to
issue a command, so nothing here decides that a move is over from a fixed
number of consecutive polls. `joint_efforts` is the exception: it is a live
request/response round trip, which makes it far slower than the others and a
poor thing to poll in a tight loop.

The mode the SDK reports comes off that same stream, and `set_mode` doesn't
update it, so a mode change isn't finished until a report confirms it: while
the SDK still believes the arm is in the previous mode, its own blocking wait
in `set_servo_angle` returns immediately instead of waiting.

Servo mode has no watchdog on this end. The controller rejects setpoints that
jump too far in one tick, so a stream that stalls and resumes will fault rather
than lunge, but pacing the stream is still the caller's job.

Every mode is checked for self-collision and against the safety box before
anything is sent (see `safety.py`), and a command that would break either is
refused with `SafetyError` rather than started and stopped part-way. The
simulator, by contrast, runs such a command and halts at the boundary — so
anything it merely warns about is something this class will reject outright.

The gripper on the flange is a separate device again: the controller drives it
over its own modbus link, outside the motion modes entirely, so `grip` doesn't
touch the arm's mode, doesn't supersede the running command, and doesn't wait
for it. It is set up lazily — mode and enable go out with the first grip — so
an arm with nothing bolted on still connects. Its *geometry* is not lazy: the
guard carries the jaws from construction, so every arm command here is checked
with the gripper on the end of it.

The controller has its own self-collision detection, which is switched on here
as a backstop. It only covers planned moves in mode 0 though: the setpoints in
mode 1 and the velocities in mode 4 aren't planned, so for those the check on
this end is the only one there is.
"""

import atexit
import contextlib
import logging
import math
import threading
import time
import traceback

import numpy as np

from xarm.core.config.x_config import XCONF
from xarm.wrapper import XArmAPI

from .api import RobotInterface
from .free_drive import (
    FREE_DRIVE_RATE,
    JOIN_TIMEOUT,
    LOCKED_TOLERANCE,
    LOCKED_WARN_FRACTION,
    RATE_LIMITS,
    RECOVER_SPEED,
    STALL_ABORT,
    WARN_PERIOD,
    Trajectory,
    hands_off_prompt,
    watch_mask,
)
from .gripper import (
    GRIPPER_CLOSED,
    RAW_TRAVEL,
    raw_position,
    raw_speed,
)
from .safety import DEFAULT_BOX, DEFAULT_MARGIN, SafetyError, SafetyGuard

# The SDK narrates through this logger at INFO: "[motion_enable], xArm is ready
# to move" and "[set_state], ..." on every connect. Its errors and warnings are
# what a person needs to see, and those stay on.
logging.getLogger("origin.print").setLevel(logging.WARNING)

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
VELOCITY_LIMIT = 3.14  # rad/s; the controller itself allows up to 4.0

DEFAULT_SPEED = 0.3  # rad/s

# Loose enough for the servos' own steady-state error (~0.05 deg), tight enough
# that a move that stopped short still reads as a miss.
_POSITION_TOLERANCE = 5e-3  # rad
_VELOCITY_TOLERANCE = 1e-2  # rad/s

_POLL_PERIOD = 0.02  # s between report reads while blocking
# Long enough to span a report period, so a settle test can't pass by reading
# one pre-move sample several times over.
_SETTLE_HOLD = 0.3  # s the arm must hold still before a wait returns
_MODE_SETTLE = 0.05  # s for a mode change to take effect on the controller
_MODE_REPORT_TIMEOUT = 2.0  # s for the report to confirm a mode change
_STOP_TIMEOUT = 2.0  # s to let the arm come to rest before changing mode

MODE_POSITION = 0
MODE_SERVO = 1
MODE_TEACH = 2  # joint teaching: every joint free, gravity compensated
MODE_VELOCITY = 4

_STATE_MOVING = 1  # the controller is executing its motion queue
_STATE_PAUSED = 3  # ... and has been paused part-way through it

_WAIT_FINISH_TIMEOUT = 100  # APIState.WAIT_FINISH_TIMEOUT
_STOPPED = -9  # APIState.EMERGENCY_STOP: the move was cancelled mid-flight
_GRIPPER_FAULT = 102  # APIState.END_EFFECTOR_HAS_FAULT
_GRIPPER_NOT_ENABLED = 103  # APIState.END_EFFECTOR_NOT_ENABLED
_DURATION_FIRMWARE = (1, 8, 0)  # vc_set_joint_velocity honours `duration` from here

_TOOL_NONE = 0  # set_collision_tool_model: bare flange
_TOOL_XARM_GRIPPER = 1  # ... and with the xArm Gripper on it
# How far ahead an open-ended velocity command is checked. It runs until it is
# replaced, so there is no natural end to check up to; this is far enough that
# anything closer is the caller aiming at a wall.
_VELOCITY_HORIZON = 5.0  # s


class XArmError(RuntimeError):
    """The controller refused a command, or isn't in a state to serve one.

    `code` is the SDK's API code when the failure came from a call; `error_code`
    is whatever error the controller itself is latching, if any.
    """

    def __init__(self, what, code=0, error_code=0):
        detail = what
        if code:
            detail += f" (code={code})"
        if error_code:
            detail += f" (controller error={error_code})"
        super().__init__(detail)
        self.code = code
        self.error_code = error_code


class RealXArm7(RobotInterface):
    """A real xArm7.

    Parameters
    ----------
    ip : controller address, e.g. "192.168.1.185".
    acceleration : joint acceleration for `set_joint_targets`, rad/s^2.
        Defaults to the controller's own default (500 deg/s^2 ~ 8.7 rad/s^2).
    velocity_limit : cap applied to every commanded joint speed, rad/s. The
        controller's own reported limit is used instead when it is lower.
    position_limit : joint ranges (7, 2), rad. Normally read from the SDK's
        table for the connected model; pass it when the arm reports a model
        that isn't in that table, or to work inside a tighter envelope.
    clear_errors : clear a latched controller error/warning on connect. With
        this off, connecting to an arm that is in an error state raises.
    timeout : seconds to wait for the report stream to come up on connect.
    arm : an already-connected `XArmAPI` to adopt instead of opening one.
        `close()` then leaves it connected — whoever opened it owns it.
    safety_box : ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in metres,
        which the whole arm must stay inside. None allows it anywhere it can
        reach; self-collision is still refused.
    safety_margin : clearance in metres between link pairs, at which the
        self-collision check trips. The box is not margined — its faces stop
        the arm where they are drawn.
    guard : refuse commands that would self-collide or leave the box. With this
        off you are left with the controller's own detection, which does not
        cover servo or velocity commands.
    gripper : there is an xArm Gripper on the flange. On means `grip` works, the
        safety guard carries the gripper, and the controller's own collision
        detection is told about it too; off means `grip` raises and the flange
        is modelled bare everywhere. The gripper itself is only talked to when
        it is first used, so this costs nothing on an arm that never grips.
    controller_boundary : also hand the box to the controller as its reduced-mode
        TCP boundary. Off by default: it constrains only the tool where the box
        here constrains the whole arm, and switching reduced mode on also caps
        the arm's speed, which is a surprising thing to inherit from setting a
        box.

    The arm is left energized and holding position on `close()`; dropping the
    motors would let it fall, and the gripper is left holding whatever it has
    for the same reason.
    """

    def __init__(
        self,
        ip=None,
        acceleration=None,
        velocity_limit=VELOCITY_LIMIT,
        position_limit=None,
        clear_errors=True,
        timeout=10.0,
        arm=None,
        safety_box=DEFAULT_BOX,
        safety_margin=DEFAULT_MARGIN,
        guard=True,
        gripper=True,
        controller_boundary=False,
    ):
        super().__init__()

        if arm is None:
            if ip is None:
                raise ValueError("pass the controller's ip (or an XArmAPI as `arm`)")
            self.arm = XArmAPI(ip, is_radian=True)
            self._owns_arm = True
        else:
            self.arm = arm
            self._owns_arm = False
        if not self.arm.connected:
            self.arm.connect()

        # An adopted XArmAPI may have been built in degrees; every call below
        # passes is_radian explicitly, but the report *properties* follow the
        # instance's own default, so convert those.
        self._to_rad = 1.0 if self.arm.default_is_radian else math.radians(1.0)

        self.joint_names = list(ARM_JOINTS)
        self.nq = self.nv = len(ARM_JOINTS)

        self._lock = threading.RLock()
        self._generation = 0  # bumped by every command, so waiters can tell
        self._timer = None
        self._mode = None
        self._acc = None if acceleration is None else float(acceleration)
        self._has_gripper = bool(gripper)
        self._gripper_ready = False  # set up on first use, not on connect
        self._auto_clear_errors = bool(clear_errors)
        self._free_drive_run = None  # the run `end_free_drive` is waiting on

        self._wait_for_report(timeout)

        axis = self.arm.axis
        if axis is not None and axis != self.nq:
            raise ValueError(f"expected a 7-axis arm, controller reports {axis}")

        self.position_limit = (
            self._model_joint_limits()
            if position_limit is None
            else np.asarray(position_limit, dtype=float).reshape(self.nq, 2)
        )
        self.velocity_limit = np.full(self.nv, self._speed_cap(velocity_limit))
        self._duration_in_firmware = self.arm.version_number >= _DURATION_FIRMWARE

        if clear_errors and self.arm.has_err_warn:
            self.arm.clean_warn()
            self.arm.clean_error()
        elif self.arm.error_code:
            raise XArmError(
                "controller is in an error state; clear it or pass clear_errors=True",
                error_code=self.arm.error_code,
            )

        self._guard = (
            SafetyGuard(safety_box, safety_margin, gripper=self._has_gripper)
            if guard
            else None
        )

        self._check(self.arm.motion_enable(True), "motion_enable")
        self._apply_mode(MODE_POSITION)
        self._arm_controller_guards(controller_boundary)

    # ------------------------------------------------------------------
    # RobotInterface
    # ------------------------------------------------------------------

    @property
    def joint_values(self):
        """Measured joint positions, rad (7,). From the report stream."""
        return np.asarray(self.arm.angles[: self.nq], dtype=float) * self._to_rad

    @property
    def joint_velocities(self):
        """Measured joint velocities, rad/s (7,).

        The controller's real-time joint speeds, which need firmware > 1.2.11;
        without them this falls back to a `get_joint_states` round trip.
        """
        speeds = self.arm.realtime_joint_speeds
        if speeds is not None and len(speeds) >= self.nv:
            return np.asarray(speeds[: self.nv], dtype=float) * self._to_rad
        return self._joint_states()[1]

    @property
    def joint_efforts(self):
        """Joint efforts as the servos report them, N*m (7,).

        A live `get_joint_states` request (firmware >= 1.9.0), not a read of the
        report stream — a millisecond-scale round trip per call.
        """
        return self._joint_states()[2]

    @property
    def joint_torques(self):
        """Joint torques from the report stream, N*m (7,).

        The controller's `joints_torque`, which needs a 'rich' report. This is
        the sensed joint torque, so unlike `joint_efforts` it includes what the
        environment pushes back with.
        """
        torque = self.arm.joints_torque
        if torque is None or len(torque) < self.nq:
            raise XArmError("joints_torque unavailable (needs a 'rich' report)")
        return np.asarray(torque[: self.nq], dtype=float)

    def set_joint_targets(self, joints, speed=None, wait=True, timeout=None):
        """Move to a joint configuration.

        The controller plans a time-synchronized joint move, so the joints
        arrive together; `speed` (rad/s) applies to the fastest-moving joint
        and is clipped to `velocity_limit`. Acceleration comes from the
        `acceleration` given at construction, not from this call.

        Raises ValueError if any target is outside the model's joint ranges,
        or `SafetyError` if the path there would self-collide or leave the
        safety box — a planned move to a pose the arm can't legally hold is a
        bug in the caller, and starting it anyway would just mean stopping
        part-way somewhere it was never asked to be.

        If `timeout` elapses first, the arm is stopped where it is: a move
        nobody is waiting for any more should not carry on unattended. Waiting
        for it to come to rest can take the call slightly past `timeout`.

        Returns True if the arm settled on the target, False if `wait` was
        False, if `timeout` elapsed first, or if the move was cancelled while
        waiting — by `stop()`, by another command, or by the emergency stop.
        """
        goal = self._check_position(self._as_config(joints))
        # The controller time-synchronizes the joints, so the whole move is the
        # straight line from here to there — check all of it, not just the end.
        self._check_safe_path(self.joint_values, goal)
        return self._move_joints(goal, speed, wait=wait, timeout=timeout)

    def _move_joints(self, goal, speed=None, wait=True, timeout=None):
        """`set_joint_targets` with the checks already done.

        Split out for free drive's retreat, which is the one move in the library
        that starts somewhere the guard would refuse: the arm is already past
        the boundary by then, and `check_path` would reject the way back out for
        exactly the reason it is being made.
        """
        speed = DEFAULT_SPEED if speed is None else float(speed)
        if speed <= 0.0:
            raise ValueError("speed must be positive")
        speed = min(speed, float(np.min(self.velocity_limit)))

        self._ensure_mode(MODE_POSITION)
        with self._lock:
            generation = self._begin_command()
        # The SDK's own blocking wait knows when the controller's motion queue
        # has drained, so it is the first thing to ask — but it is not the last
        # word: it gives up the moment the *reported* mode isn't 0, which can
        # still be the mode this call just switched away from. Take it as a
        # head start on `wait_for_motion` rather than as proof of arrival.
        started = time.monotonic()
        with self._stop_on_interrupt():
            code = self.arm.set_servo_angle(
                angle=goal.tolist(),
                speed=speed,
                mvacc=self._acc,
                is_radian=True,
                wait=wait,
                timeout=timeout,
            )
        if code == _STOPPED:
            return False
        if code == _WAIT_FINISH_TIMEOUT:
            self._stop_timed_out(generation)
            return False
        self._check(code, "set_servo_angle")

        if not wait:
            return False
        remaining = None
        if timeout is not None:
            remaining = float(timeout) - (time.monotonic() - started)
            if remaining <= 0.0:
                self._stop_timed_out(generation)
                return False
        if not self.wait_for_motion(timeout=remaining):
            self._stop_timed_out(generation)
            return False
        return bool(np.max(np.abs(self.joint_values - goal)) <= _POSITION_TOLERANCE)

    def _stop_timed_out(self, generation):
        """Stop a move whose wait ran out, and wait for the arm to come to rest.

        Leaves the arm alone if a newer command has replaced the move, since
        that command is what the arm should be doing now, or if the connection
        has gone, since there is nothing left to tell it.
        """
        with self._lock:
            if self._generation != generation or not self.arm.connected:
                return
            self.stop(wait=False)
        self.wait_for_motion(timeout=_STOP_TIMEOUT)

    def set_velocity(self, speeds, duration=0, wait=True):
        """Command joint velocities, rad/s (7,).

        The command is held until it is replaced, until `stop()`, or until
        `duration` seconds have elapsed (`duration=0` — or infinite — holds it
        indefinitely). The controller ramps to the commanded speeds and stops
        the joints at their limits; commands are clipped to `velocity_limit`.

        With `wait=True` and a finite, non-zero `duration` this blocks until
        the command expires; an indefinite command has nothing to wait for, so
        it returns immediately whatever `wait` says.

        Raises `SafetyError` if holding these speeds would run the arm into
        itself or out of the safety box. An open-ended command is checked
        `_VELOCITY_HORIZON` seconds ahead; there is no watchdog behind that,
        because the report stream is far too slow to stop the arm on the
        strength of what it says.

        Returns True if it blocked for the whole duration, False otherwise —
        including when another command replaced this one while waiting.
        """
        target = np.asarray(speeds, dtype=float).reshape(-1)
        if target.size != self.nv:
            raise ValueError(f"expected {self.nv} velocities, got {target.size}")
        target = np.clip(target, -self.velocity_limit, self.velocity_limit)

        timed = duration > 0 and np.isfinite(duration)
        duration = float(duration) if timed else 0.0
        self._check_safe_ray(target, duration if timed else _VELOCITY_HORIZON)

        self._ensure_mode(MODE_VELOCITY)
        with self._lock:
            generation = self._begin_command()
            code = self.arm.vc_set_joint_velocity(
                target.tolist(),
                is_radian=True,
                duration=duration if self._duration_in_firmware else 0,
            )
            if code == 0 and timed and not self._duration_in_firmware:
                # Pre-1.8.0 firmware ignores `duration`; expire it from here
                # instead, so an unattended command still comes to a stop.
                self._timer = threading.Timer(
                    duration, self._expire_velocity, args=(generation,)
                )
                self._timer.daemon = True
                self._timer.start()
            deadline = time.monotonic() + duration
        self._check(code, "vc_set_joint_velocity")

        if not wait or not timed:
            return None
        return self._wait_until(deadline, generation)

    def servo_joints(self, joints, velocities=None):
        """Stream one joint setpoint from a high-frequency control loop.

        Unlike `set_joint_targets` this plans nothing: the setpoint goes to the
        controller's servo mode and is applied at the next control tick. Call
        it in a loop at your control rate (100 Hz - 250 Hz) and the arm moves
        continuously through the stream.

        `velocities` is accepted for parity with the simulator and ignored:
        the controller's servo interface takes positions only, and there is no
        channel to feed an intended velocity forward. Feedforward that matters
        belongs in the setpoints themselves.

        Setpoints are clamped to the joint ranges, and refused with
        `SafetyError` if they would self-collide or leave the safety box —
        the controller plans nothing here, so this check is the only one
        standing between a streamed setpoint and the arm. Nothing else is
        limited: smoothness and pacing are the caller's to own, which is the
        point of servo mode.
        """
        target = self._clamp_position(self._as_config(joints))
        self._check_safe(target)
        if velocities is not None:
            # Checked for shape and then dropped — see above.
            size = np.asarray(velocities, dtype=float).size
            if size != self.nv:
                raise ValueError(f"expected {self.nv} velocities, got {size}")

        self._ensure_mode(MODE_SERVO)
        with self._lock:
            self._begin_command()
            code = self.arm.set_servo_angle_j(target.tolist(), is_radian=True)
        self._check(code, "set_servo_angle_j")

    @property
    def gripper_opening(self):
        """How far open the gripper is, 0.0 shut to 1.0 wide open.

        A live modbus round trip to the gripper, like `joint_efforts` and
        unlike the joint readings: the gripper is not in the arm's report
        stream, so there is nothing cached here to read. Poll it sparingly.
        """
        self._require_gripper()
        with self._lock:
            code, pulses = self.arm.get_gripper_position()
        self._check(code, "get_gripper_position")
        if pulses is None:
            raise XArmError("gripper reported no position")
        return float(np.clip(pulses / RAW_TRAVEL, 0.0, 1.0))

    def grip(self, opening=GRIPPER_CLOSED, speed=None, wait=True, timeout=None):
        """Open or close the gripper, 0.0 shut to 1.0 wide open.

        The gripper is its own device on its own link, so this is the one
        command here that leaves the arm alone: it doesn't change the arm's
        mode, doesn't supersede the command the arm is running, and doesn't
        wait for it — the jaws can close while the arm is still moving, which
        is what makes grasping on the fly possible. Nothing is checked against
        the safety guard either: the guard carries the jaws at every opening,
        so moving them between two openings it already allows cannot take the
        arm anywhere new.

        The gripper is put into position mode and energized on the first grip,
        not on connect, and a fault it is latching is cleared then too if the
        arm was constructed with `clear_errors=True`.

        `speed` is a fraction of the gripper's full speed; `timeout` defaults
        to the SDK's own 10 s. A keyboard interrupt out of the wait leaves the
        jaws doing as they were told — the gripper's protocol has nothing that
        cancels a move part-way, only a new position to go to.

        Returns True if the jaws came to a stop, which is also what closing on
        an object looks like: the gripper stalls where the object is and
        reports itself stopped. `gripper_opening` is what tells the two apart —
        short of the target means something is in the way. Returns False if
        `wait` was False or if `timeout` elapsed with the jaws still moving.

        Raises ValueError if `opening` or `speed` is out of range, and
        `XArmError` if the gripper faults (it stops being enabled when it
        does; `clear_errors()` clears it).
        """
        pulses = raw_position(opening)
        pulse_speed = raw_speed(speed)
        self._enable_gripper()
        code = self.arm.set_gripper_position(
            pulses,
            speed=pulse_speed,
            wait=bool(wait),
            timeout=timeout,
            # The gripper isn't on the arm's motion queue, but the SDK holds
            # this command back until the arm's own move has finished unless
            # told not to — a coupling nothing about the hardware asks for.
            wait_motion=False,
        )
        if code == _WAIT_FINISH_TIMEOUT:
            return False
        if code in (_GRIPPER_FAULT, _GRIPPER_NOT_ENABLED):
            with self._lock:
                self._gripper_ready = False  # a fault drops the enable
            raise XArmError(
                "gripper would not move (its own error code is "
                f"{self.arm.gripper_error_code}); clear_errors() clears it",
                code,
            )
        self._check(code, "set_gripper_position")
        return bool(wait)

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    def stop(self, wait=True, timeout=None):
        """Stop moving and hold the current position.

        Cancels whatever is running: a queued position move is aborted, a
        velocity command is zeroed. A servo stream has nothing queued to
        cancel — the arm holds the last setpoint it was sent, and a later
        `servo_joints` call starts a fresh stream. A keyboard interrupt out of
        a blocking motion call runs this too.

        In teaching mode there is likewise nothing queued, and the arm does
        hold itself up — but it yields to anything that pushes it, which is the
        point of the mode. Stopping there means leaving it, which puts the arm
        back under position control, where a push is resisted instead.
        """
        with self._lock:
            self._begin_command()
            if self._mode == MODE_TEACH:
                self._apply_mode(MODE_POSITION)
                code = 0
            else:
                code = self._halt()
        self._check(code, "stop")
        if not wait:
            return False
        return self.wait_for_motion(timeout=timeout)

    def check_safety(self, joints):
        """Return why `joints` would be refused, or None if it's allowed.

        The same check every command runs, exposed so a target can be tested
        before it is committed to.
        """
        if self._guard is None:
            return None
        return self._guard.check(self._as_config(joints))

    @contextlib.contextmanager
    def _stop_on_interrupt(self):
        """Stop the arm if the wait inside is interrupted.

        Ctrl-C out of a blocking call should leave the arm standing still: the
        command it was waiting on is nobody's any more, and the controller
        would otherwise keep running it while the traceback prints.

        Must not be entered holding the lock around a wait the halt has to get
        past — `_wait_stopped` runs inside `_ensure_mode`, which has already
        halted the arm itself.
        """
        try:
            yield
        except KeyboardInterrupt:
            self.stop(wait=False)
            raise

    def wait_for_motion(self, timeout=None):
        """Block until the arm settles. Returns False on timeout, or if the
        connection to the controller has gone away.

        In position mode this also asks the controller whether its motion
        queue is still running, so a move that has been accepted but hasn't
        started yet doesn't read as already finished.

        A keyboard interrupt stops the arm before it propagates.
        """
        with self._stop_on_interrupt():
            return self._wait_settled(timeout, check_queue=self._mode == MODE_POSITION)

    @property
    def has_error(self):
        """True if the controller is holding an error or warning code."""
        return bool(self.arm.has_err_warn)

    def clear_errors(self):
        """Clear a latched error/warning and re-arm the arm for motion.

        A fault on the gripper is cleared too — it keeps its own error code,
        which the arm's own clearing doesn't reach — and the gripper is
        re-enabled by the next grip, since faulting drops its enable.
        """
        with self._lock:
            self.arm.clean_warn()
            self.arm.clean_error()
            self._check(self.arm.motion_enable(True), "motion_enable")
            self._apply_mode(MODE_POSITION)
            if self._has_gripper:
                self._clear_gripper_error()
                self._gripper_ready = False

    def close(self):
        """Stop the arm and drop the connection.

        The motors stay enabled and holding — disabling them would let the arm
        fall under its own weight — and the gripper is left exactly as it is,
        holding whatever is in it, for the same reason.

        A free drive run still going is stopped first, and its recording is
        dropped: `close()` is the way out of a script, not a way to collect a
        run. What matters here is that the watch thread is not left sampling an
        arm whose connection is about to go, and that the arm is locked rather
        than left in teaching mode where anything can push it.
        """
        try:
            self._free_drive_halt()
        except Exception:  # closing must not depend on a healthy link
            pass

        with self._lock:
            self._begin_command()
            try:
                # Teaching mode yields to anything that touches the arm.
                # Disconnecting from it would leave the arm free to be pushed
                # anywhere with nothing left watching where it ended up.
                if self._mode == MODE_TEACH:
                    self._apply_mode(MODE_POSITION)
                else:
                    self._halt()
            except Exception:  # closing must not depend on a healthy link
                pass
        if self._owns_arm and self.arm.connected:
            self.arm.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _wait_for_report(self, timeout):
        """Block until the controller is streaming state, or raise."""
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if not self.arm.connected:
                raise XArmError("controller not reachable")
            code, _ = self.arm.get_servo_angle(is_radian=True)
            if code == 0 and self.arm.state is not None:
                return
            time.sleep(_POLL_PERIOD)
        raise XArmError(f"no report from the controller within {timeout}s")

    def _model_joint_limits(self):
        """Joint ranges for the connected model, straight out of the SDK's own
        table — the same one it checks commands against."""
        axis = self.arm.axis or self.nq
        device_type = self.arm.device_type
        sn = self.arm.sn
        try:  # the SDK's own rule for the 1305-series variants
            if sn and 1305 <= int(sn[2:6]) < 8500:
                device_type = int(f"{axis}1305")
        except ValueError:
            pass

        limits = XCONF.Robot.JOINT_LIMITS.get(axis, {}).get(device_type)
        if not limits or len(limits) < self.nq:
            # Joint 4's range is flipped between xArm7 variants, so guessing a
            # default here could invert a limit check. Make the caller say.
            raise ValueError(
                f"no joint limits known for axis={axis} device_type={device_type};"
                " pass position_limit= to say what they are"
            )
        return np.asarray(limits[: self.nq], dtype=float)

    def _speed_cap(self, requested):
        """The tighter of the requested cap and the controller's own."""
        cap = float(requested)
        limit = self.arm.joint_speed_limit
        if limit is not None and len(limit) >= 2 and limit[1] > 0:
            cap = min(cap, float(limit[1]) * self._to_rad)
        return cap

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _as_config(self, values):
        q = np.asarray(values, dtype=float).reshape(-1)
        if q.size != self.nq:
            raise ValueError(f"expected {self.nq} joint values, got {q.size}")
        return q

    def _clamp_position(self, q):
        return np.clip(q, self.position_limit[:, 0], self.position_limit[:, 1])

    def _check_position(self, q):
        """Return `q`, or raise if any joint is outside its range."""
        low, high = self.position_limit[:, 0], self.position_limit[:, 1]
        bad = np.flatnonzero((q < low) | (q > high))
        if bad.size:
            detail = ", ".join(
                f"{self.joint_names[i]}={q[i]:.4f} not in "
                f"[{low[i]:.4f}, {high[i]:.4f}]"
                for i in bad
            )
            raise ValueError(f"joint target outside limits: {detail}")
        return q

    @property
    def guard(self):
        """The `SafetyGuard` checking this arm's setpoints, or None."""
        return self._guard

    def _check_safe(self, q):
        """Raise if the arm may not be at `q`."""
        if self._guard is None:
            return
        violation = self._guard.check(q)
        if violation is not None:
            raise SafetyError(violation)

    def _check_safe_path(self, start, goal):
        """Raise if the straight joint-space line from `start` to `goal` isn't
        clear the whole way."""
        if self._guard is None:
            return
        violation, _ = self._guard.check_path(start, goal)
        if violation is not None:
            raise SafetyError(violation)

    def _check_safe_ray(self, velocity, horizon):
        """Raise if holding `velocity` runs into something within `horizon`."""
        if self._guard is None:
            return
        violation, _ = self._guard.check_ray(self.joint_values, velocity, horizon)
        if violation is not None:
            raise SafetyError(violation)

    def _arm_controller_guards(self, controller_boundary):
        """Switch on the checks the controller can make for itself.

        These cover less than the guard on this end — self-collision detection
        only applies to planned moves, and the boundary only to the tool — but
        they run on the controller in real time, which nothing here does.
        """
        self._check(self.arm.set_self_collision_detection(True), "self-collision")
        tool = _TOOL_XARM_GRIPPER if self._has_gripper else _TOOL_NONE
        self._check(self.arm.set_collision_tool_model(tool), "tool model")

        if not controller_boundary:
            return
        box = self._guard.box if self._guard is not None else None
        if box is None:
            raise ValueError("controller_boundary needs a safety box")
        # The controller wants millimetres, and each axis max-then-min.
        boundary = []
        for low, high in box:
            boundary += [high * 1000.0, low * 1000.0]
        self._check(self.arm.set_reduced_tcp_boundary(boundary), "tcp boundary")
        self._check(self.arm.set_reduced_mode(True), "reduced mode")

    def _check(self, code, what):
        if code != 0:
            raise XArmError(f"{what} failed", code, self.arm.error_code)
        return code

    def _joint_states(self):
        code, states = self.arm.get_joint_states(is_radian=True, num=3)
        self._check(code, "get_joint_states")
        return [np.asarray(s[: self.nq], dtype=float) for s in states]

    def _require_gripper(self):
        """Raise unless this arm was told it has a gripper on it."""
        if not self._has_gripper:
            raise XArmError(
                "this arm was constructed with gripper=False; pass "
                "gripper=True if there is an xArm Gripper on the flange"
            )

    def _enable_gripper(self):
        """Put the gripper in position mode and energize it, once.

        Left until the first grip rather than done on connect: these are
        round trips to a device that may not be plugged in, and an arm with a
        bare flange has to be able to connect and move without them.
        """
        self._require_gripper()
        with self._lock:
            if self._gripper_ready:
                return
            if self._auto_clear_errors:
                self._clear_gripper_error()
            self._check(self.arm.set_gripper_mode(0), "set_gripper_mode")
            self._check(self.arm.set_gripper_enable(True), "set_gripper_enable")
            self._gripper_ready = True

    def _clear_gripper_error(self):
        """Clear a fault the gripper is latching, if it has one.

        The gripper keeps its own error code and refuses to move while it is
        set; the arm's `clean_error` doesn't reach it. Caller must hold the
        lock.
        """
        code, err = self.arm.get_gripper_err_code()
        if code == 0 and not err:
            return
        self._check(self.arm.clean_gripper_error(), "clean_gripper_error")

    def _begin_command(self):
        """Supersede whatever is running. Caller must hold the lock."""
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        return self._generation

    def _expire_velocity(self, generation):
        """Zero a timed velocity command on firmware that can't expire it."""
        with self._lock:
            if self._generation != generation or self._mode != MODE_VELOCITY:
                return  # superseded; whatever replaced it owns the arm now
            self.arm.vc_set_joint_velocity(
                [0.0] * self.nv, is_radian=True, duration=0
            )

    def _halt(self):
        """Cancel the running command. Caller must hold the lock."""
        if self._mode == MODE_VELOCITY:
            return self.arm.vc_set_joint_velocity(
                [0.0] * self.nv, is_radian=True, duration=0
            )
        if self._mode == MODE_POSITION:
            # State 4 drops the controller's motion queue; state 0 re-arms it.
            code = self.arm.set_state(4)
            return code if code != 0 else self.arm.set_state(0)
        # Servo mode: nothing queued, the last setpoint holds. Teaching mode:
        # nothing queued either, and what is moving the arm is a person, so
        # there is nothing here that could stop it. `stop()` handles that one
        # by leaving the mode.
        return 0

    def _apply_mode(self, mode):
        """Switch the controller to `mode` and wait for the report to say so.

        `set_mode` doesn't touch the mode the SDK reports — only a report
        packet does — and until one arrives the SDK still believes the arm is
        in the mode it was in before. That belief is load-bearing: the
        blocking wait inside `set_servo_angle` returns immediately whenever
        the mode it can see isn't 0, so a move issued in that window would not
        wait at all. Caller must hold the lock.
        """
        self._check(self.arm.set_mode(mode), f"set_mode({mode})")
        self._check(self.arm.set_state(0), "set_state(0)")
        self._mode = mode
        time.sleep(_MODE_SETTLE)
        deadline = time.monotonic() + _MODE_REPORT_TIMEOUT
        while self.arm.mode != mode:
            if time.monotonic() >= deadline:
                raise XArmError(
                    f"set_mode({mode}) not confirmed by the report within "
                    f"{_MODE_REPORT_TIMEOUT}s (still reads {self.arm.mode})"
                )
            time.sleep(_POLL_PERIOD)

    def _ensure_mode(self, mode):
        """Put the controller in `mode`, stopping the arm first if needed.

        The controller only accepts a mode change from a stopped state, so the
        running command is cancelled — which is also the semantics we want: a
        new command supersedes the old one.

        Leaving teaching mode skips both steps. Nothing is queued there to
        cancel, and the only thing that moves the arm is a hand on it: waiting
        for the arm to be still would mean blocking until the person happens to
        let go, when switching mode is exactly how the arm is made to hold
        still in the first place.
        """
        with self._lock:
            if self._mode == mode:
                return
            if self._mode == MODE_TEACH:
                self._apply_mode(mode)
                return
            self._check(self._halt(), "stop")
            if not self._wait_stopped(_STOP_TIMEOUT):
                raise XArmError(f"arm still moving {_STOP_TIMEOUT}s after stop")
            self._apply_mode(mode)

    def _queue_is_running(self):
        """Whether the controller is still working through its motion queue.

        A live `get_state` round trip, not `arm.state`: the reported state is
        a report period old, so a move issued moments ago still shows the idle
        state the arm had before it was commanded.
        """
        code, state = self.arm.get_state()
        return code == 0 and state in (_STATE_MOVING, _STATE_PAUSED)

    def _wait_settled(self, timeout, check_queue):
        """Block until the arm holds still for `_SETTLE_HOLD`.

        Both readings this leans on — velocities and positions — come off the
        report stream, and polls come round faster than reports do, so the
        test is held across wall-clock time rather than over a count of polls:
        one pre-move sample, read several times over, must not pass for a
        finished move. Positions are compared to where the arm was when it first
        looked settled, which catches a joint creeping along too slowly to
        clear the velocity tolerance.
        """
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        anchor = None
        since = 0.0
        while True:
            if not self.arm.connected:
                return False
            moving = check_queue and self._queue_is_running()
            moving = moving or bool(
                np.max(np.abs(self.joint_velocities)) > _VELOCITY_TOLERANCE
            )
            now = time.monotonic()
            values = self.joint_values
            if moving or anchor is None or (
                np.max(np.abs(values - anchor)) > _POSITION_TOLERANCE
            ):
                anchor, since = values, now
            elif now - since >= _SETTLE_HOLD:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(_POLL_PERIOD)

    def _wait_stopped(self, timeout):
        """Block until the joints are at rest, before a mode change.

        Unlike `wait_for_motion` this asks only whether the arm is physically
        still: it runs straight after `_halt`, when the controller's state is
        still on its way back to idle.
        """
        return self._wait_settled(timeout, check_queue=False)

    def _wait_until(self, deadline, generation):
        """Block until the velocity command expiring at `deadline` is done.
        Returns False if it was replaced, or if the link dropped. A keyboard
        interrupt stops the arm before it propagates."""
        with self._stop_on_interrupt():
            while True:
                with self._lock:
                    if self._generation != generation:
                        return False  # superseded by another command, or stopped
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return True
                if not self.arm.connected:
                    return False
                time.sleep(min(_POLL_PERIOD, remaining))

    # ------------------------------------------------------------------
    # Free drive
    # ------------------------------------------------------------------

    def free_drive(
        self,
        duration=None,
        *,
        watch=(),
        tolerance=LOCKED_TOLERANCE,
        rate=FREE_DRIVE_RATE,
        recover_speed=RECOVER_SPEED,
        teach_sensitivity=None,
        confirm=None,
        on_sample=None,
        verbose=False,
    ):
        """Release the arm so it can be pushed by hand, and start recording.

        This returns as soon as the arm is free. The run carries on in a thread
        behind you, and `end_free_drive` is what ends it and hands back the
        recording — so the waiting is yours to do, however suits you:

            robot.free_drive()
            print("push the arm around; press Enter when you are done")
            input()
            traj = robot.end_free_drive()

        The arm is sampled, guarded and watched the whole time, including while
        you sit in `input()`.

        The run uses the controller's own joint teaching mode, which frees
        **every** joint and gravity-compensates the arm. By default that is the
        whole of it: the arm goes wherever it is pushed, the loop samples it,
        and the recording is what you get back. The safety guard is still
        checked every tick — see below.

        `watch` is for joints that are meant to stay put. Teaching mode has no
        per-joint version, so the joints named there are not held — they are
        watched. Their positions are latched before the run starts, and if one
        of them drifts more than `tolerance` off its latched value the run ends
        itself: teaching mode is dropped for position control, which stiffens
        the arm where it stands. Nothing moves then. The next `end_free_drive`
        asks you to take your hands off and then puts the watched joints back on
        their latched values, leaving the free ones where you left them.

        That the watch loop stops the arm rather than moving it is what makes it
        safe to wait for a run at a prompt: whoever started it is sitting in
        `input()`, where they cannot be asked to let go first. Every move a
        recovery needs happens inside `end_free_drive`, which runs on your own
        thread and can ask.

        Args:
            duration: seconds of teaching after which the run ends itself, or
                None — the default — for no limit, leaving `end_free_drive`,
                the guard or a fault to end it. A run that ends on its own still
                has to be collected with `end_free_drive`, which is what hands
                over the recording.
            watch: which joints to watch, as **0-indexed** joint numbers —
                `[1, 2]` is joint2 and joint3 — or a length-7 boolean mask in
                which True means watched. Defaults to watching nothing: every
                joint is free and the arm is simply recorded. `WATCHED_JOINTS`
                is the four the lab's `HOME_POSE` is defined by. Keyword-only,
                so that a run holding part of the arm has to say so.
            tolerance: rad a watched joint may drift before the run ends and
                waits to be put back. Below about 0.02 rad ordinary handling of
                the free joints trips it through the arm's own structure.
                Unused if nothing is watched.
            rate: how often the arm is sampled, Hz, 20 to 250. Nothing is
                commanded at this rate — it is a sampling rate, and the report
                stream underneath it is what really limits how soon a drift or
                a guard violation is seen.
            recover_speed: rad/s for the move `end_free_drive` makes to put a
                watched joint back, and for the step back inside the guard.
            teach_sensitivity: 1-5 to apply for the run and restore afterwards,
                or None to leave the controller's own setting alone. Higher is
                easier to push. UFACTORY's advice is not to touch it unless the
                arm is hard to move.
            confirm: called by `end_free_drive` with the hands-off warning
                before the arm moves on its own, and must not return until it
                is safe for it to. Return True to let it move, False to leave
                the arm where it stands. Defaults to a terminal prompt.
            on_sample: called as `on_sample(t, q)` after every recorded tick,
                for live plotting or a stop condition of your own. Return
                something truthy to end the run. It runs on the watch thread,
                so it must return quickly — time spent in it is time nothing is
                watching, and past `STALL_ABORT` the run ends — and whatever it
                touches, it touches from that thread.
            verbose: narrate the run: the banner, why it ended, and a warning
                as a watched joint nears its limit. Off by default, since
                `Trajectory.reason` says why a run ended. The guard is checked
                either way, and a guard stop and faults print a warning either
                way.

        Returns:
            None. `end_free_drive` returns the `Trajectory`.

        Raises ValueError for a malformed `watch` or `duration`, `SafetyError`
        if the arm is somewhere the guard won't allow before the run even
        starts, and RuntimeError if a run is already going.

        Ctrl-C is no longer what ends a run: it lands in whatever you are
        waiting in rather than in the loop, and the loop goes on watching. What
        hands the arm back is `end_free_drive`, or `close()` — so a
        `with Robot() as robot:` block locks the arm on the way out whether it
        ends normally or not.

        The guard cannot *refuse* anything during the run — the person's hand
        is not a command — so a self-collision or a step outside the safety box
        is caught after the fact rather than before it. What it can do is stop:
        a sample the guard rejects ends the run there and then and the arm
        stiffens where it stands. `end_free_drive` then steps it back to the
        last configuration the guard allowed, after a hands-off prompt — a few
        milliseconds of the hand's movement undone, which is all the further
        past the line it can have got at the sampling rate. That matters for
        more than tidiness: a guarded command checks the whole path from where
        the arm *is*, so an arm left outside the box would refuse every move
        afterwards. `reason` says what was hit; everything recorded up to it is
        returned as usual.
        """
        if self._free_drive_run is not None:
            raise RuntimeError(
                "free drive is already running — end_free_drive() ends that "
                "run and hands back its recording, and then another can start"
            )

        mask = watch_mask(watch, self.nq)
        try:
            duration = np.inf if duration is None else float(duration)
        except TypeError:
            # The old signature took the joints first. Someone porting a script
            # gets this rather than a run that quietly means something else.
            raise ValueError(
                f"duration must be a number of seconds or None, not "
                f"{duration!r} — joints go in `watch=`, and they are the ones "
                "to *hold*, not the ones to free"
            ) from None
        # `not > 0` rather than `<= 0`, so a NaN duration is caught here rather
        # than becoming a run that never reaches its own deadline.
        if not duration > 0.0:
            raise ValueError("duration must be positive, or None for no limit")
        rate = float(rate)
        if not RATE_LIMITS[0] <= rate <= RATE_LIMITS[1]:
            raise ValueError(
                f"rate must be between {RATE_LIMITS[0]} and {RATE_LIMITS[1]} Hz"
            )
        tolerance = float(tolerance)
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if confirm is not None and not callable(confirm):
            raise ValueError(
                "confirm must be callable: it is called with the hands-off "
                "warning before the arm is moved on its own"
            )

        # A run that starts from a configuration the guard rejects has nowhere
        # legal to put the arm back to, and the recovery move would be refused
        # the first time it was needed. Say so now rather than then.
        if self._guard is not None:
            blocked = self._guard.check(self.joint_values)
            if blocked is not None:
                raise SafetyError(blocked)

        state = {
            "run": {
                "t": [], "q": [], "qd": [],
                "overruns": 0, "interruptions": 0, "ticks": 0,
                "teaching": 0.0, "source": "state", "warned_at": -np.inf,
            },
            "mask": mask,
            "duration": duration,
            "period": 1.0 / rate,
            "rate": rate,
            "tolerance": tolerance,
            "recover_speed": recover_speed,
            "confirm": hands_off_prompt if confirm is None else confirm,
            "on_sample": on_sample,
            "verbose": bool(verbose),
            # Latched while the arm is still in position control and holding
            # itself, so the lock is where the caller put the arm rather than
            # wherever it had drifted to a moment after being released.
            "q_latched": self.joint_values.copy(),
            "stop": threading.Event(),
            "thread": None,
            "sensitivity": None,
            "reason": None,
            "recovery": None,  # what `end_free_drive` has to put right
            "announced": False,  # whether the thread has printed the reason
        }

        if teach_sensitivity is not None:
            state["sensitivity"] = self.arm.teach_sensitivity
            self._check(
                self.arm.set_teach_sensitivity(int(teach_sensitivity)),
                "set_teach_sensitivity",
            )

        if verbose:
            self._free_drive_banner(mask, duration, tolerance, rate)

        try:
            self._ensure_mode(MODE_TEACH)
        except BaseException:
            # No run started, so nothing else will put the setting back.
            self._restore_teach_sensitivity(state)
            raise

        state["thread"] = threading.Thread(
            target=self._free_drive_thread,
            args=(state,),
            name="xarm7-free-drive",
            daemon=True,
        )
        # Registered before the thread runs and dropped by `_free_drive_halt`:
        # an arm left in teaching mode is pushable by anything, so it is worth
        # locking even on the way out of a script that crashed.
        self._free_drive_run = state
        atexit.register(self._free_drive_halt)
        state["thread"].start()

    def end_free_drive(self):
        """End the run `free_drive` started, lock the arm, return the recording.

        The arm is back under position control by the time this returns: out of
        teaching mode, holding where it stands, resisting a push instead of
        yielding to it.

        If the run ended with a watched joint off its lock, or with the arm
        outside the safety guard, this is where that is put right — `confirm`
        asks for hands off the robot, and then the arm moves on its own, slowly.
        Decline and the arm is left where it is, which `Trajectory.reason`
        records.

        Returns the `Trajectory`: `t`, `q` and `qd` sampled once per tick, with
        `reason` saying why the run ended. Raises RuntimeError if no run is
        going, which includes a second call for the same run — the recording is
        handed over once.
        """
        state = self._free_drive_halt()
        if state is None:
            raise RuntimeError(
                "there is no free drive run to end: free_drive() starts one, "
                "and end_free_drive() collects it exactly once"
            )

        verbose = state["verbose"]
        reason = state["reason"] or "stopped by end_free_drive()"
        if verbose and not state["announced"]:
            print(f"[xarm7] free drive ended: {reason}")
        if state["recovery"] is not None:
            reason = self._free_drive_put_back(state, reason)
        return self._free_drive_trajectory(
            state["run"], state["mask"], state["rate"], reason, verbose
        )

    def _free_drive_halt(self):
        """Stop the watch loop and see that the arm is locked. Returns the run.

        `end_free_drive` is the usual way in. `close()` and the interpreter
        shutting down come through here too, which is what stops a forgotten
        run from leaving an arm that anything can push. Returns None when there
        is no run to stop, and does not raise: it is called from paths that are
        already on their way out.
        """
        state = self._free_drive_run
        if state is None:
            return None
        self._free_drive_run = None
        atexit.unregister(self._free_drive_halt)

        state["stop"].set()
        thread = state["thread"]
        if thread.ident is None:
            # `free_drive` was interrupted between freeing the arm and starting
            # the thread: nobody to wait for, and nobody else to lock the arm.
            ours = True
        else:
            thread.join(JOIN_TIMEOUT)
            ours = thread.is_alive()
            if ours:
                # Stuck in a call to the controller. The thread is a daemon so
                # it cannot hold the interpreter open, but the arm is still
                # free, and locking it is this thread's problem now.
                print(
                    "[xarm7] the free drive watch loop has not stopped after "
                    f"{JOIN_TIMEOUT:g} s; locking the arm from here instead. "
                    "The recording is whatever had been sampled by now."
                )
                if state["reason"] is None:
                    state["reason"] = "the watch loop stopped responding"
        if ours:
            self._free_drive_release()
            self._restore_teach_sensitivity(state)
        return state

    def _restore_teach_sensitivity(self, state):
        """Put back the teaching sensitivity a run borrowed, at most once."""
        previous, state["sensitivity"] = state["sensitivity"], None
        if not previous:
            return
        try:
            self.arm.set_teach_sensitivity(int(previous))
        except Exception as err:  # a run ending must not fail over a setting
            print(f"[xarm7] could not restore the teach sensitivity: {err}")

    def _free_drive_thread(self, state):
        """Run the watch loop, then hand the arm back. The thread's whole life.

        Nothing is allowed to escape from here. This is the top of a thread, so
        an exception would be printed and then lost, leaving the arm in
        teaching mode with nothing watching it. Whatever happens, the arm goes
        back under position control and the reason is kept for `end_free_drive`
        to report.
        """
        try:
            state["reason"] = self._free_drive_loop(state)
        except BaseException as err:  # noqa: BLE001 — see above
            state["reason"] = f"the watch loop failed: {err!r}"
            print(
                "[xarm7] the free drive watch loop failed\n"
                f"{traceback.format_exc()}"
            )
        finally:
            self._free_drive_release()
            self._restore_teach_sensitivity(state)
            recovery = state["recovery"]
            if recovery is not None and recovery["kind"] == "guard":
                # Not narration: the arm has just gone stiff in somebody's
                # hands, and they need to know why whatever `verbose` says.
                state["announced"] = True
                print(f"[xarm7] WARNING: free drive stopped: {state['reason']}")
                print(
                    "        the arm is locked where it stands; end_free_drive() "
                    "will ask you to stand clear, step it back inside, and "
                    "return the recording"
                )
            elif state["verbose"] and not state["stop"].is_set():
                # The run ended on its own and whoever started it is off doing
                # something else: say so, and say what is waiting for them.
                state["announced"] = True
                print(f"[xarm7] free drive ended: {state['reason']}")
                if state["recovery"] is None:
                    print(
                        "        the arm is locked where it stands; "
                        "end_free_drive() returns the recording"
                    )
                else:
                    print(
                        "        the arm is locked where it stands; "
                        "end_free_drive() will ask you to stand clear, put "
                        "that right, and return the recording"
                    )

    def _report_tick(self):
        """Something that changes once per report packet, and what it is.

        `arm.count` is the controller's own cycle counter, carried in every rich
        report long enough to hold it. It advances whether or not the arm is
        moving, which is what makes it a measure of the report *rate*.

        Falling back to the joint positions measures how often the readings
        *change*, which is a lower bound on the report rate and can be zero — a
        motionless arm reports the same numbers every packet. In free drive the
        arm is usually being moved, so it is a fair bound here.
        """
        count = getattr(self.arm, "count", -1)
        if count is not None and count != -1:
            return "count", count
        return "state", tuple(self.joint_values)

    def _free_drive_banner(self, mask, duration, tolerance, rate):
        """Say what the run is about to do. Cheap to print, expensive to
        discover halfway through a lab."""
        watched = np.flatnonzero(~mask).tolist()
        print("[xarm7] free drive (joint teaching mode): every joint is free")
        print("        push the arm around; end_free_drive() locks it and "
              "returns the recording")
        if watched:
            names = ", ".join(self.joint_names[i] for i in watched)
            print(
                f"        watching {names} — pushed more than "
                f"{np.degrees(tolerance):.1f} deg off and the run stops there, "
                "for end_free_drive() to put them back"
            )
        ends = "no time limit" if np.isinf(duration) else f"{duration:g} s"
        print(f"        {ends}, sampling at {rate:.0f} Hz")

    def _free_drive_loop(self, state):
        """Sample the arm and watch the locked joints. Returns why it ended.

        This runs on the watch thread, and it reads the arm and stops the arm —
        it never moves it. Whatever needs undoing afterwards is left in
        `state["recovery"]` for `end_free_drive` to do, where there is somebody
        to ask first.
        """
        run = state["run"]
        mask, q_latched = state["mask"], state["q_latched"]
        duration, period = state["duration"], state["period"]
        tolerance, stop = state["tolerance"], state["stop"]
        on_sample, verbose = state["on_sample"], state["verbose"]

        warn_at = tolerance * LOCKED_WARN_FRACTION
        # Nothing watched is the default, and then the drift check has nothing
        # to compare: the loop is there to record and to check the guard.
        watching = bool((~mask).any())
        started = time.monotonic()
        next_tick = started + period
        last = started
        last_tick = None
        # `free_drive` refuses to start from a pose the guard rejects, so the
        # latched configuration is a known-good place to retreat to until a
        # sample replaces it.
        q_safe = q_latched

        while True:
            if stop.is_set():
                return "stopped by end_free_drive()"
            now = time.monotonic()
            run["teaching"] = now - started
            if run["teaching"] >= duration:
                return "duration elapsed"
            if now - last > STALL_ABORT:
                # The watch is the lock, and it has stopped running. Ending
                # the run is what puts the joints back under position control,
                # where they are held rather than watched.
                return f"the loop stalled for {now - last:.2f} s"
            last = now
            if not self.arm.connected:
                return "the connection to the controller dropped"
            if self.arm.has_err_warn:
                return f"the controller latched an error (code={self.arm.error_code})"
            if self.arm.mode != MODE_TEACH:
                # `_ensure_mode` trusts our own cache, so this is the only thing
                # that notices the controller leaving teaching mode on its own.
                return f"the controller left teaching mode (now {self.arm.mode})"

            q = self.joint_values
            source, tick = self._report_tick()
            if last_tick is not None and tick != last_tick:
                run["ticks"] += 1
            last_tick, run["source"] = tick, source

            run["t"].append(now - started)
            run["q"].append(q)
            run["qd"].append(self.joint_velocities)
            if on_sample is not None and on_sample(now - started, q.copy()):
                return "stopped by the caller"

            # Checked whether or not anyone is watching the terminal: this is
            # the one thing in the run that ends it rather than reporting it.
            violation = None if self._guard is None else self._guard.check(q)
            if violation is not None:
                # A violation always ends the run: teaching would resume one
                # tick from the boundary and trip again immediately.
                state["recovery"] = {
                    "kind": "guard", "violation": violation, "q_safe": q_safe,
                }
                return f"unsafe pose: {violation}"
            q_safe = q

            if not watching:
                next_tick += period
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    run["overruns"] += 1
                    next_tick = time.monotonic()
                continue

            drift = np.abs(q - q_latched)
            drift[mask] = 0.0
            worst = int(np.argmax(drift))

            if drift[worst] > tolerance:
                run["interruptions"] += 1
                state["recovery"] = {
                    "kind": "watch", "joint": worst, "drift": float(drift[worst]),
                }
                return (
                    f"{self.joint_names[worst]} was pushed "
                    f"{np.degrees(drift[worst]):.1f} deg off its lock"
                )

            if verbose and drift[worst] > warn_at:
                self._free_drive_warn(
                    run,
                    f"{self.joint_names[worst]} is locked and has been "
                    f"pushed {np.degrees(drift[worst]):.1f} deg off it — "
                    "ease off, or the run will stop",
                )

            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                run["overruns"] += 1
                next_tick = time.monotonic()

    def _free_drive_put_back(self, state, reason):
        """Undo whatever ended the run: a watched joint off its lock, or a pose
        the guard won't allow. Returns `reason`, said in full.

        Called from `end_free_drive`, on the caller's own thread, which is the
        whole point of leaving it until now: the arm is about to move on its
        own, and here there is somebody at the terminal to tell first. It is in
        position control and stiff where it stands by then, which is what makes
        "take your hands off" a safe instruction rather than an alarming one.

        The step back inside the guard is not a checked path: it starts where
        the guard says the arm may not be, so `check_path` would refuse the
        move for the very reason it is being made. What makes it safe is that
        the arm has just come along it, a sampling period ago.
        """
        recovery, state["recovery"] = state["recovery"], None
        confirm = state["confirm"]
        speed = state["recover_speed"]
        verbose = state["verbose"]

        if recovery["kind"] == "watch":
            name = self.joint_names[recovery["joint"]]
            if not confirm(
                f"\n[xarm7] {name} is locked, and it was pushed "
                f"{np.degrees(recovery['drift']):.1f} deg off its lock.\n"
                f"        TAKE YOUR HANDS OFF THE ROBOT — it is about to move "
                f"on its own to put {name} back."
            ):
                return f"{reason}; the put-back was declined, so it is still off it"

            target = self.joint_values
            target[~state["mask"]] = state["q_latched"][~state["mask"]]
            try:
                reached = self.set_joint_targets(target, speed=speed)
            except (ValueError, SafetyError, XArmError) as err:
                # Most likely the free joints have been driven somewhere the
                # guard won't allow, which this move cannot undo — it only
                # moves the watched ones.
                return f"{reason}; {name} could not be put back on its lock: {err}"
            if not reached:
                return f"{reason}; the move putting {name} back did not finish"
            if verbose:
                print(f"[xarm7] {name} is back on its lock")
            return reason

        violation = recovery["violation"]
        if not confirm(
            "\n[xarm7] the arm reached the edge of what the guard allows:"
            f"\n        {violation}"
            "\n        Free drive stopped there and the arm is holding still."
            "\n        TAKE YOUR HANDS OFF THE ROBOT — it is about to move on "
            "its own to step back inside."
        ):
            # Left where it stopped, which is a pose every guarded command will
            # refuse, so say so rather than leave it to be discovered.
            return (
                f"{reason}; the step back inside was declined, so the arm is "
                "left outside the guard and later commands will be refused"
            )
        try:
            reached = self._move_joints(recovery["q_safe"], speed=speed)
        except (ValueError, XArmError) as err:
            return f"{reason}; the arm could not be stepped back inside: {err}"
        if not reached:
            return f"{reason}; the move back inside did not finish"
        if verbose:
            print("[xarm7] the arm is back inside the guard")
        return reason

    def _free_drive_warn(self, run, what):
        """Print at most once every `WARN_PERIOD`, so a hand held against the
        boundary doesn't bury everything else on the terminal."""
        now = time.monotonic()
        if now - run["warned_at"] < WARN_PERIOD:
            return
        run["warned_at"] = now
        print(f"[xarm7] {what}")

    def _free_drive_release(self):
        """Hand the arm back: out of teaching mode, into position control.

        Teaching mode holds the arm up but yields to anything that pushes it,
        so leaving a run in it would leave the arm movable by hand with nothing
        watching where it went. This runs on every path out of a run: the watch
        thread's own way out, and `_free_drive_halt` giving up on a thread that
        has stopped answering. Neither fault below is narration, so both print
        whatever `verbose` says.
        """
        if self.arm.has_err_warn:
            print(
                "[xarm7] the controller has latched an error; call "
                f"clear_errors() before moving again (code={self.arm.error_code})"
            )
        try:
            self._ensure_mode(MODE_POSITION)
        except XArmError as err:
            print(f"[xarm7] could not return to position control: {err}")

    def _free_drive_trajectory(self, run, mask, rate, reason, verbose):
        """Assemble the recording, and work out whether `qd` is real.

        `realtime_joint_speeds` is a pre-initialized list of zeros until a long
        enough rich report writes it, so an arm that cannot report speeds
        reports standing still instead of reporting nothing. A Jacobian checked
        against that would be checked against a zero vector with nothing to say
        it went wrong, so if the joints demonstrably moved while the speeds
        never did, differentiate the positions and say so.
        """
        t = np.asarray(run["t"], dtype=float)
        q = np.asarray(run["q"], dtype=float).reshape(-1, self.nq)
        qd = np.asarray(run["qd"], dtype=float).reshape(-1, self.nv)
        source = "report"

        moved = q.size and float(np.max(np.ptp(q, axis=0))) > _POSITION_TOLERANCE
        if moved and not np.any(qd):
            source = "difference"
            qd = np.gradient(q, t, axis=0) if t.size > 1 else np.zeros_like(q)
            if verbose:
                print(
                    "[xarm7] the controller reported no joint speeds, so qd is "
                    "a finite difference of q — a Jacobian checked against it "
                    "is being compared with another finite difference."
                )

        report_rate = float("nan")
        if run["ticks"] and run["teaching"] > 0.0:
            report_rate = run["ticks"] / run["teaching"]

        return Trajectory(
            t=t, q=q, qd=qd, joint_names=list(self.joint_names), free=mask,
            rate=float(rate), report_rate=report_rate, qd_source=source,
            reason=reason, overruns=int(run["overruns"]),
            interruptions=int(run["interruptions"]),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ip", help="controller address, e.g. 192.168.1.185")
    parser.add_argument("-y", "--yes", action="store_true", help="don't ask first")
    args = parser.parse_args()

    np.set_printoptions(precision=3, suppress=True)

    if not args.yes:
        print("This moves joint 1 by +/-0.3 rad. Clear the workspace.")
        if input("continue? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit(0)

    with RealXArm7(args.ip) as arm:
        print("joints:", arm.joint_names)
        start = arm.joint_values
        print("  q   =", start)
        print("  tau =", arm.joint_torques)

        target = start.copy()
        target[0] += 0.3
        print("moving to", target)
        print("reached:", arm.set_joint_targets(target, speed=0.5))
        print("  q   =", arm.joint_values)

        print("jogging joint 1 back for 1 s")
        arm.set_velocity([-0.3, 0, 0, 0, 0, 0, 0], duration=1.0)
        print("  v   =", arm.joint_velocities)
        arm.wait_for_motion()
        print("  q   =", arm.joint_values)

        print("returning to start")
        arm.set_joint_targets(start, speed=0.5)
        print("  q   =", arm.joint_values)
        print("  tau =", arm.joint_torques, "(holding against gravity)")
