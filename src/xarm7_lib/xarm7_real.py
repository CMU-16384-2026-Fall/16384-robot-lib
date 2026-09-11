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
commanded in that mode — see `free_drive.py` for what it does instead.

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

The controller has its own self-collision detection, which is switched on here
as a backstop. It only covers planned moves in mode 0 though: the setpoints in
mode 1 and the velocities in mode 4 aren't planned, so for those the check on
this end is the only one there is.
"""

import contextlib
import math
import threading
import time

import numpy as np

from xarm.core.config.x_config import XCONF
from xarm.wrapper import XArmAPI

from .api import RobotInterface
from .free_drive import (
    FREE_DRIVE_RATE,
    FREE_JOINTS,
    LOCKED_TOLERANCE,
    LOCKED_WARN_FRACTION,
    RATE_LIMITS,
    RECOVER_SPEED,
    REPEAT_LIMIT,
    REPEAT_WINDOW,
    STALL_ABORT,
    WARN_PERIOD,
    Trajectory,
    free_mask,
    hands_off_prompt,
)
from .safety import DEFAULT_BOX, DEFAULT_MARGIN, SafetyError, SafetyGuard

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
_DURATION_FIRMWARE = (1, 8, 0)  # vc_set_joint_velocity honours `duration` from here

_TOOL_NONE = 0  # set_collision_tool_model: bare flange
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
    safety_margin : clearance in metres at which the check trips.
    guard : refuse commands that would self-collide or leave the box. With this
        off you are left with the controller's own detection, which does not
        cover servo or velocity commands.
    controller_boundary : also hand the box to the controller as its reduced-mode
        TCP boundary. Off by default: it constrains only the tool where the box
        here constrains the whole arm, and switching reduced mode on also caps
        the arm's speed, which is a surprising thing to inherit from setting a
        box.

    The arm is left energized and holding position on `close()`; dropping the
    motors would let it fall.
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

        self._guard = SafetyGuard(safety_box, safety_margin) if guard else None

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

        Returns True if the arm settled on the target, False if `wait` was
        False, if `timeout` elapsed first, or if the move was cancelled while
        waiting — by `stop()`, by another command, or by the emergency stop.
        """
        goal = self._check_position(self._as_config(joints))
        # The controller time-synchronizes the joints, so the whole move is the
        # straight line from here to there — check all of it, not just the end.
        self._check_safe_path(self.joint_values, goal)
        speed = DEFAULT_SPEED if speed is None else float(speed)
        if speed <= 0.0:
            raise ValueError("speed must be positive")
        speed = min(speed, float(np.min(self.velocity_limit)))

        self._ensure_mode(MODE_POSITION)
        with self._lock:
            self._begin_command()
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
        if code in (_WAIT_FINISH_TIMEOUT, _STOPPED):
            return False
        self._check(code, "set_servo_angle")

        if not wait:
            return False
        remaining = None
        if timeout is not None:
            remaining = float(timeout) - (time.monotonic() - started)
            if remaining <= 0.0:
                return False
        if not self.wait_for_motion(timeout=remaining):
            return False
        return bool(np.max(np.abs(self.joint_values - goal)) <= _POSITION_TOLERANCE)

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
        """Clear a latched error/warning and re-arm the arm for motion."""
        with self._lock:
            self.arm.clean_warn()
            self.arm.clean_error()
            self._check(self.arm.motion_enable(True), "motion_enable")
            self._apply_mode(MODE_POSITION)

    def close(self):
        """Stop the arm and drop the connection.

        The motors stay enabled and holding — disabling them would let the arm
        fall under its own weight.
        """
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
        self._check(self.arm.set_collision_tool_model(_TOOL_NONE), "tool model")

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
        free=FREE_JOINTS,
        duration=30.0,
        *,
        tolerance=LOCKED_TOLERANCE,
        rate=FREE_DRIVE_RATE,
        recover_speed=RECOVER_SPEED,
        teach_sensitivity=None,
        confirm=None,
        on_sample=None,
        verbose=True,
    ):
        """Let a person push the arm by hand, and record where they took it.

        This runs the controller's own joint teaching mode, which frees **every**
        joint and gravity-compensates the arm. There is no per-joint version of
        it, so the joints outside `free` are not held — they are watched. Their
        positions are latched before the run starts, and if one of them drifts
        more than `tolerance` off its latched value the run interrupts itself:

            1. back to position control, which stiffens the arm where it stands;
            2. `confirm` asks the person to take their hands off, and waits;
            3. a slow move puts the watched joints back on their latched values,
               leaving the free ones where the person left them;
            4. back to teaching mode, and the recording carries on.

        So a watched joint is one the arm returns to, not one it refuses to
        leave, and it is free for as long as it takes the loop to notice.
        `tolerance` is how far wrong the arm may be in the meantime. Every
        interruption leaves a gap in the returned `t` — see `Trajectory`.

        Args:
            free: which joints to release *and mean it*, as **0-indexed** joint
                numbers — `[0, 3, 6]` is joint1, joint4 and joint7 — or a
                length-7 boolean mask. The rest are watched. Defaults to
                `FREE_JOINTS`, the three the lab's `HOME_POSE` is built around.
            duration: seconds of teaching to run for. Time spent stopped for an
                interruption doesn't count against it. Ctrl-C ends the run early
                and still returns everything recorded up to that point.
            tolerance: rad a watched joint may drift before the run puts it
                back. Below about 0.02 rad ordinary handling of the free joints
                trips it through the arm's own structure.
            rate: how often the watched joints are checked, Hz, 20 to 250.
                Nothing is commanded at this rate — it is a sampling rate, and
                the report stream underneath it is what really limits how soon
                a drift is seen.
            recover_speed: rad/s for the move that puts a watched joint back.
            teach_sensitivity: 1-5 to apply for the duration and restore
                afterwards, or None to leave the controller's own setting alone.
                Higher is easier to push. UFACTORY's advice is not to touch it
                unless the arm is hard to move.
            confirm: called with the hands-off warning when a watched joint has
                to be put back, and must not return until it is safe for the arm
                to move. Return True to carry on, False to end the run. Defaults
                to a terminal prompt.
            on_sample: called as `on_sample(t, q)` after every recorded tick,
                for live plotting or a caller's own stop condition. Return
                something truthy to end the run. It runs inside the watch loop,
                so it must return quickly — time spent in it is time nothing is
                watching the locked joints, and past `STALL_ABORT` the run ends.
            verbose: print the banner and the running warnings.

        Returns:
            A `Trajectory` with `t`, `q` and `qd` sampled once per tick, plus
            `reason` saying why the run ended and `interruptions` saying how many
            gaps are in it.

        Raises ValueError for a malformed `free`, and `SafetyError` if the arm is
        somewhere the guard won't allow before the run even starts.

        The guard cannot refuse anything during the run — the person's hand is
        not a command — so where the free joints go is theirs to get right, and
        a self-collision or a step outside the safety box is warned about rather
        than prevented.
        """
        mask = free_mask(free, self.nq)
        duration = float(duration)
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        rate = float(rate)
        if not RATE_LIMITS[0] <= rate <= RATE_LIMITS[1]:
            raise ValueError(
                f"rate must be between {RATE_LIMITS[0]} and {RATE_LIMITS[1]} Hz"
            )
        tolerance = float(tolerance)
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if mask.all():
            raise ValueError(
                "every joint is free, so there is nothing to watch — that is "
                "the controller's own drag-teach, and `arm.set_mode(2)` is all "
                "it takes"
            )
        confirm = hands_off_prompt if confirm is None else confirm

        # A run that starts from a configuration the guard rejects has nowhere
        # legal to put the arm back to, and the recovery move would be refused
        # the first time it was needed. Say so now rather than then.
        if self._guard is not None:
            blocked = self._guard.check(self.joint_values)
            if blocked is not None:
                raise SafetyError(blocked)

        period = 1.0 / rate
        run = {
            "t": [], "q": [], "qd": [],
            "overruns": 0, "interruptions": 0, "ticks": 0, "paused": 0.0,
            "teaching": 0.0, "source": "state", "warned_at": -np.inf,
        }
        previous_sensitivity = None
        reason = "duration elapsed"

        # Latched while the arm is still in position control and holding itself,
        # so the lock is where the caller put the arm rather than wherever it
        # had drifted to a moment after being released.
        q_latched = self.joint_values.copy()

        if teach_sensitivity is not None:
            previous_sensitivity = self.arm.teach_sensitivity
            self._check(
                self.arm.set_teach_sensitivity(int(teach_sensitivity)),
                "set_teach_sensitivity",
            )

        if verbose:
            self._free_drive_banner(mask, tolerance, rate, teach_sensitivity)

        try:
            self._ensure_mode(MODE_TEACH)
            reason = self._free_drive_loop(
                mask=mask, duration=duration, period=period, tolerance=tolerance,
                recover_speed=recover_speed, q_latched=q_latched, confirm=confirm,
                on_sample=on_sample, verbose=verbose, run=run,
            )
        except KeyboardInterrupt:
            # Deliberately swallowed rather than re-raised: Ctrl-C is how a run
            # is meant to end, and the recording is the whole point of it.
            reason = "interrupted"
        finally:
            # A second Ctrl-C — and students will press it — must not cost the
            # data by escaping from the cleanup, and must not leave the arm in
            # a mode where a push moves it and nothing is watching.
            try:
                if verbose:
                    print(f"[xarm7] free drive ended: {reason}")
                self._free_drive_release(verbose)
                if previous_sensitivity:
                    self.arm.set_teach_sensitivity(int(previous_sensitivity))
            except KeyboardInterrupt:
                self._free_drive_release(verbose)

        return self._free_drive_trajectory(run, mask, rate, reason, verbose)

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

    def _free_drive_banner(self, mask, tolerance, rate, teach_sensitivity):
        """Say what the run is about to do. Cheap to print, expensive to
        discover halfway through a lab."""
        free = np.flatnonzero(mask).tolist()
        locked = np.flatnonzero(~mask).tolist()
        sensitivity = (
            self.arm.teach_sensitivity if teach_sensitivity is None
            else teach_sensitivity
        )
        print(
            f"[xarm7] free drive (joint teaching mode)\n"
            f"  free      {free}  ({', '.join(self.joint_names[i] for i in free)})\n"
            f"  watched   {locked}, put back past "
            f"{np.degrees(tolerance):.1f} deg of drift\n"
            f"  sampling  {rate:.0f} Hz   teach sensitivity {sensitivity}"
            f"{'' if teach_sensitivity is None else ' (restored afterwards)'}\n"
            "  every joint is released, so the watched ones are free until the "
            "drift is\n  noticed. Push the free joints; Ctrl-C to stop early."
        )

    def _free_drive_loop(self, *, mask, duration, period, tolerance,
                         recover_speed, q_latched, confirm, on_sample,
                         verbose, run):
        """Sample the arm and watch the locked joints. Returns why it ended."""
        warn_at = tolerance * LOCKED_WARN_FRACTION
        started = time.monotonic()
        next_tick = started + period
        last = started
        last_tick = None
        resumed_at = -np.inf  # the first trip is nobody's repeat
        repeats = 0

        while True:
            now = time.monotonic()
            run["teaching"] = now - started - run["paused"]
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

            drift = np.abs(q - q_latched)
            drift[mask] = 0.0
            worst = int(np.argmax(drift))

            if drift[worst] > tolerance:
                if now - resumed_at < REPEAT_WINDOW:
                    repeats += 1
                else:
                    repeats = 0
                if repeats >= REPEAT_LIMIT:
                    # Putting it back and having it come straight off again is
                    # not someone leaning on the arm, it is a joint this pose
                    # cannot hold — carry on and the person only ever sees
                    # prompts. The arm is left stiff, off the lock, and said so.
                    return (
                        f"{self.joint_names[worst]} came off its lock again "
                        f"within {REPEAT_WINDOW:.0f} s, {REPEAT_LIMIT} times "
                        "over: it is not holding at this pose"
                    )
                paused_from = time.monotonic()
                ended = self._free_drive_recover(
                    worst=worst, drift=float(drift[worst]), mask=mask,
                    q_latched=q_latched, recover_speed=recover_speed,
                    confirm=confirm, verbose=verbose,
                )
                run["paused"] += time.monotonic() - paused_from
                run["interruptions"] += 1
                if ended is not None:
                    return ended
                # The clocks all restart: the pause was not a stalled loop, and
                # the samples either side of it are seconds apart.
                resumed_at = last = time.monotonic()
                next_tick = resumed_at + period
                continue

            if verbose:
                if drift[worst] > warn_at:
                    self._free_drive_warn(
                        run,
                        f"{self.joint_names[worst]} is locked and has been "
                        f"pushed {np.degrees(drift[worst]):.1f} deg off it — "
                        "ease off, or the run will stop to put it back",
                    )
                violation = None if self._guard is None else self._guard.check(q)
                if violation is not None:
                    # Nothing here can refuse it. Every other mode checks a
                    # configuration before committing to it, but in free drive
                    # there is no command to refuse — a hand went somewhere, and
                    # all this can do is say so.
                    self._free_drive_warn(run, f"unsafe pose: {violation}")

            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                run["overruns"] += 1
                next_tick = time.monotonic()

    def _free_drive_recover(self, *, worst, drift, mask, q_latched,
                            recover_speed, confirm, verbose):
        """Stiffen, get the hands clear, put the watched joints back.

        Returns None to carry on teaching, or why the run is over.
        """
        name = self.joint_names[worst]
        # Position control first, and before anyone is asked anything: it
        # stiffens the arm exactly where it stands, which is what makes "take
        # your hands off" a safe instruction rather than an alarming one.
        self._ensure_mode(MODE_POSITION)
        if not confirm(
            f"\n[xarm7] {name} is locked, and it has been pushed "
            f"{np.degrees(drift):.1f} deg off its lock.\n"
            f"        TAKE YOUR HANDS OFF THE ROBOT — it is about to move on "
            f"its own to put {name} back."
        ):
            return f"{name} drifted off its lock and the recovery was declined"

        target = self.joint_values
        target[~mask] = q_latched[~mask]
        try:
            reached = self.set_joint_targets(target, speed=recover_speed)
        except (ValueError, SafetyError, XArmError) as err:
            # Most likely the free joints have been driven somewhere the guard
            # won't allow, which the recovery move cannot undo — it only moves
            # the watched ones.
            return f"{name} could not be put back on its lock: {err}"
        if not reached:
            return f"the move putting {name} back on its lock did not finish"

        if verbose:
            print(f"[xarm7] {name} is back on its lock; free drive resuming\n")
        self._ensure_mode(MODE_TEACH)
        return None

    def _free_drive_warn(self, run, what):
        """Print at most once every `WARN_PERIOD`, so a hand held against the
        boundary doesn't bury everything else on the terminal."""
        now = time.monotonic()
        if now - run["warned_at"] < WARN_PERIOD:
            return
        run["warned_at"] = now
        print(f"[xarm7] {what}")

    def _free_drive_release(self, verbose):
        """Hand the arm back: out of teaching mode, into position control.

        Teaching mode holds the arm up but yields to anything that pushes it,
        so leaving a run in it would leave the arm movable by hand with nothing
        watching where it went. This runs on every path out of a run, including
        a second Ctrl-C.
        """
        if verbose and self.arm.has_err_warn:
            print(
                "[xarm7] the controller has latched an error; call "
                f"clear_errors() before moving again (code={self.arm.error_code})"
            )
        try:
            self._ensure_mode(MODE_POSITION)
        except XArmError as err:
            if verbose:
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
        if verbose and run["interruptions"]:
            print(
                f"[xarm7] {run['interruptions']} interruption(s), so t has that "
                "many gaps in it; np.diff(t) finds them."
            )

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
