"""Simulated xArm7 on MuJoCo, implementing the RobotInterface API.

Physics is MuJoCo stepping the Menagerie xarm7 MJCF (arm + gripper);
visualization is meshcat, driven by `mujoco_meshcat.MeshcatVisualizer`. The
simulation runs on a background thread paced to wall time, and the scene shows
up in a browser tab.

Control is MuJoCo's own position actuators — the MJCF already declares one per
joint, so `data.ctrl[i] = angle` is the whole interface. Two thin layers sit on
top of that, per tick:

    reference   the setpoint ramps toward the goal at `speed` rather than
                snapping to it; a snapped setpoint whips the arm to ~9 rad/s
                (3x the joint velocity limit) with the motors saturated
    feedforward the setpoint is offset by the torque the servo would otherwise
                have to develop an error to produce:
                    ctrl = q_ref + (bias + friction + kv v_ref) / kp
                without it the arm droops ~0.9 deg under gravity and never
                reaches its target

The compensation goes through the servo setpoint rather than around it, so
`data.actuator_force` still reports the true motor torque (gravity holding
torque included) and MuJoCo still clamps it to the model's force range.

`qfrc_bias` is read from the previous tick's forward pass, which is one
timestep stale — worth ~0.1% of the gravity term at full speed, and it saves a
redundant `mj_forward` per step.

Every mode is guarded against self-collision and against leaving the safety box
(see `safety.py`), and every mode is guarded in the same place: `_update_reference`
is the one funnel all three run through, so the check goes on the setpoint just
before it is committed. When the next setpoint would be unsafe the old one is
kept and the arm holds — it coasts to a stop at the boundary rather than being
refused up front, which is the behaviour worth having in a simulator: the motion
happens, is visible, and stops where it should. A warning says what was hit.

The real arm refuses these commands outright instead. Anything the simulator
merely warns about is something the hardware will reject.

The MJCF carries a gripper the real arm doesn't. `gripper=False` takes it out
of both the picture and the contact set; its mass stays either way, standing in
for whatever is actually bolted on. Turning it back on gives you a gripper the
guard knows nothing about, so the simulator can then jam on a part no check will
warn you about — which is why the flange is bare by default.
"""

import contextlib
import threading
import time

import mujoco
import numpy as np

from .api import RobotInterface
from .mujoco_meshcat import MeshcatVisualizer
from .safety import DEFAULT_BOX, DEFAULT_MARGIN, SafetyGuard

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
VELOCITY_LIMIT = 3.14  # rad/s; the MJCF carries no joint velocity limits

DEFAULT_SPEED = 0.3  # rad/s

_POSITION_TOLERANCE = 2e-3  # rad; above the servos' stiction deadband
_VELOCITY_TOLERANCE = 1e-2  # rad/s
_STRIBECK_VELOCITY = 1e-2  # rad/s, smooths the Coulomb friction sign()

SERVO_TIMEOUT = 0.1  # s without a new servo command before the arm coasts down

# The guard costs about as much as a physics tick, so it is not run on every
# one. Re-checking once the setpoint has used up this fraction of the clearance
# margin keeps a fast move honest and a slow one cheap.
_GUARD_TRAVEL = 0.25
_WARN_PERIOD = 1.0  # s between repeats of the same warning
_REACH = 0.8  # m, roughly how far the far end of the arm sits from any joint

# How near a face of the safety box the arm has to be for that face to be
# drawn. Boxed in on all six sides at once the scene is mostly wall, and none
# of it is telling you anything; a face earns its place once it is close enough
# to be what stops you.
_FACE_PROXIMITY = 0.15  # m

# ...except the box's z_min face, which the viewer draws as the surface the arm
# is standing on. Hiding the ground leaves the arm floating in an empty scene —
# the MJCF has no floor of its own — so it stays whatever the arm is doing.
_FLOOR_FACE = "zmin"

_GRIPPER_BODIES = ("gripper", "finger", "knuckle")
_HIDDEN_GROUP = 3  # outside the groups the visualizer draws

_MODE_HOLD = "hold"
_MODE_POSITION = "position"
_MODE_VELOCITY = "velocity"
_MODE_SERVO = "servo"


class SimulatedXArm7(RobotInterface):
    """A simulated xArm7.

    Parameters
    ----------
    mjcf_path : MJCF to load. Defaults to the Menagerie xarm7 description.
    dt : physics timestep, seconds. Defaults to the model's own timestep.
    q0 : initial arm configuration (7,). Defaults to all zeros; pass
        `keyframe="home"` instead to start from the MJCF's home pose.
    keyframe : name or index of an MJCF keyframe to reset into.
    visualize : publish the scene to meshcat and open it in a browser tab.
    real_time : run the sim in a background thread paced to wall time. When
        False the sim only advances inside `step()` (or while a blocking call
        waits on it), which makes runs deterministic and as fast as the CPU
        allows.
    gravity : include gravity in the dynamics.
    feedforward : compensate gravity/Coriolis/friction through the servo
        setpoint. With this off you get the MJCF's raw position servos, which
        droop under gravity and lag during motion — useful for testing how a
        controller copes, but then a move may never reach its target within
        tolerance.
    safety_box : ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in metres,
        which the whole arm must stay inside. None allows the arm anywhere it
        can reach.
    safety_margin : clearance in metres at which the guard trips.
    guard : check every setpoint for self-collision and box violations. Turning
        this off lets the arm drive into itself, which the real one won't do.
    gripper : keep the MJCF's gripper. Off by default, matching the bare flange
        the guard and the real arm assume; on, it is drawn and collides, but
        the guard still doesn't model it.
    """

    def __init__(
        self,
        mjcf_path=None,
        dt=None,
        q0=None,
        keyframe=None,
        visualize=False,
        real_time=True,
        gravity=True,
        feedforward=True,
        viz_rate=30.0,
        safety_box=DEFAULT_BOX,
        safety_margin=DEFAULT_MARGIN,
        guard=True,
        gripper=False,
    ):
        super().__init__()

        if mjcf_path is None:
            from robot_descriptions.loaders.mujoco import load_robot_description

            self.model = load_robot_description("xarm7_mj_description")
        else:
            self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))

        if dt is not None:
            self.model.opt.timestep = float(dt)
        if not gravity:
            self.model.opt.gravity[:] = 0.0
        self.dt = float(self.model.opt.timestep)
        self.data = mujoco.MjData(self.model)

        self.joint_names = list(ARM_JOINTS)
        self.nq = len(ARM_JOINTS)
        self.nv = len(ARM_JOINTS)
        self._jid, self._qadr, self._vadr, self._act = self._index_arm()

        # Servo gains, straight out of the MJCF: force = kp*(ctrl - q) - kv*v.
        self._kp = self.model.actuator_gainprm[self._act, 0].copy()
        self._kv = -self.model.actuator_biasprm[self._act, 2].copy()
        self._feedforward = bool(feedforward) and bool(np.all(self._kp > 0.0))

        self.position_limit = self.model.jnt_range[self._jid].copy()  # (7, 2)
        self.velocity_limit = np.full(self.nv, VELOCITY_LIMIT)
        self.effort_limit = self.model.actuator_forcerange[self._act, 1].copy()

        self._damping = self.model.dof_damping[self._vadr].copy()
        self._frictionloss = self.model.dof_frictionloss[self._vadr].copy()

        self._gripper_geoms = self._find_gripper_geoms()
        if not gripper:
            self._remove_gripper()

        self._guard = SafetyGuard(safety_box, safety_margin) if guard else None
        self.last_violation = None
        self._warned_at = -np.inf
        # A joint turning by dq sweeps the far end of the arm through about
        # `_REACH * dq`, so this is the setpoint travel that spends
        # `_GUARD_TRAVEL` of the clearance margin.
        self._guard_step = (
            _GUARD_TRAVEL * safety_margin / _REACH if guard else np.inf
        )

        self._reset_state(q0, keyframe)

        self._mode = _MODE_HOLD
        self._vel_target = np.zeros(self.nv)
        self._vel_deadline = None

        self._lock = threading.RLock()
        self._thread = None
        self._stop_event = threading.Event()

        self.viz = None
        self._viz_period = 1.0 / viz_rate if viz_rate else None
        self._last_viz = -np.inf
        if visualize:
            self.viz = MeshcatVisualizer(self.model, self.data)
            self._draw_safety_box()
            print(f"Meshcat: {self.viz.url}")

        if real_time:
            self._thread = threading.Thread(
                target=self._run, name="xarm7-mujoco", daemon=True
            )
            self._thread.start()

    # ------------------------------------------------------------------
    # RobotInterface
    # ------------------------------------------------------------------

    @property
    def joint_values(self):
        """Measured joint positions, rad (7,)."""
        with self._lock:
            return self.data.qpos[self._qadr].copy()

    @property
    def joint_velocities(self):
        """Measured joint velocities, rad/s (7,)."""
        with self._lock:
            return self.data.qvel[self._vadr].copy()

    @property
    def joint_efforts(self):
        """Actuator torque delivered by the servos, N*m (7,).

        MuJoCo's `actuator_force`, i.e. the motor torque after clamping to the
        model's force range.
        """
        with self._lock:
            return self.data.actuator_force[self._act].copy()

    @property
    def joint_torques(self):
        """Net torque seen at the joint, N*m (7,).

        Actuator torque plus the passive (damping, friction) and constraint
        (joint-limit, contact) torques acting on the same DOF, so this is what
        a joint-side torque sensor would read.
        """
        with self._lock:
            return (
                self.data.qfrc_actuator[self._vadr]
                + self.data.qfrc_passive[self._vadr]
                + self.data.qfrc_constraint[self._vadr]
            )

    def set_joint_targets(self, joints, speed=None, wait=True, timeout=None):
        """Move to a joint configuration at constant speed.

        The setpoint travels in a straight line through joint space, so the
        joints are time-synchronized and arrive together after
        `max|goal - start| / speed` seconds. `speed` (rad/s) applies to the
        fastest-moving joint and is clipped to `velocity_limit`.

        Raises ValueError if any target is outside the model's joint ranges —
        a planned move to an unreachable pose is a bug in the caller, and
        silently clamping it would have the arm end up somewhere it was never
        asked to go.

        Returns True if the arm settled on the target, False if `wait` was
        False, if `timeout` elapsed first, or if the guard stopped the arm at
        the safety boundary short of the target — `last_violation` says which.
        """
        goal = self._check_position(self._as_config(joints))
        speed = DEFAULT_SPEED if speed is None else float(speed)
        if speed <= 0.0:
            raise ValueError("speed must be positive")

        with self._lock:
            start = self.data.qpos[self._qadr].copy()
            delta = goal - start
            span = float(np.max(np.abs(delta)))

            if span < _POSITION_TOLERANCE:  # already there
                self._q_ref = goal
                self._v_ref = np.zeros(self.nv)
                self._mode = _MODE_HOLD
            else:
                direction = delta / span  # unit along the fastest joint
                moving = np.abs(direction) > 0.0
                speed = min(
                    speed,
                    float(np.min(self.velocity_limit[moving] / np.abs(direction[moving]))),
                )
                self._goal = goal
                self._start = start
                self._direction = direction
                self._span = span
                self._travelled = 0.0
                self._speed = speed
                self._mode = _MODE_POSITION

            self._vel_deadline = None
            self._vel_target = np.zeros(self.nv)

        if not wait:
            return False
        if not self.wait_for_motion(timeout=timeout):
            return False
        # Settling isn't arriving: the guard parks the arm at the boundary and
        # it settles there perfectly happily.
        with self._lock:
            reached = self.data.qpos[self._qadr]
        return bool(np.max(np.abs(reached - goal)) <= _POSITION_TOLERANCE)

    def set_velocity(self, speeds, duration=0, wait=True):
        """Command joint velocities, rad/s (7,).

        The command is held until it is replaced, until `stop()`, or until
        `duration` seconds have elapsed (`duration=0` — or infinite — holds it
        indefinitely). Joints stop at their limits; commands are clipped to
        `velocity_limit`.

        With `wait=True` and a finite, non-zero `duration` this blocks until
        the command expires; an indefinite command has nothing to wait for, so
        it returns immediately whatever `wait` says.

        Returns True if it blocked for the whole duration, False otherwise —
        including when another command replaced this one while waiting.
        """
        target = np.asarray(speeds, dtype=float).reshape(-1)
        if target.size != self.nv:
            raise ValueError(f"expected {self.nv} velocities, got {target.size}")
        target = np.clip(target, -self.velocity_limit, self.velocity_limit)

        timed = duration > 0 and np.isfinite(duration)
        with self._lock:
            self._mode = _MODE_VELOCITY
            self._vel_target = target
            self._vel_deadline = self._t + float(duration) if timed else None
            # Integrate the setpoint from where the arm actually is, so a stale
            # reference can't cause a jump.
            self._q_ref = self.data.qpos[self._qadr].copy()
            deadline = self._vel_deadline

        if not wait or not timed:
            return None
        return self._wait_until(deadline)

    def servo_joints(self, joints, velocities=None):
        """Stream one joint setpoint from a high-frequency control loop.

        Unlike `set_joint_targets` this plans nothing: the setpoint is applied
        as given, and neither the start nor the end of the command is assumed
        to be at rest. Call it in a loop at your control rate (100 Hz - 1 kHz)
        and the arm moves continuously through the stream.

        The commanded velocity is fed forward so the servo doesn't have to
        develop a tracking error to move. Pass `velocities` if you know it —
        from the trajectory you're sampling, say — otherwise it is estimated
        by differencing successive commands.

        Between commands the setpoint keeps moving at that velocity, so a
        control loop slower than the physics rate still produces smooth motion
        instead of a staircase. If commands stop arriving for `SERVO_TIMEOUT`,
        the setpoint stops extrapolating and holds — without that watchdog a
        dropped stream would coast the arm into its limits.

        Setpoints are clamped to the joint ranges. Nothing else is limited:
        smoothness is the caller's to own, which is the point of servo mode.
        """
        target = self._clamp_position(self._as_config(joints))
        if velocities is not None:
            velocity = np.asarray(velocities, dtype=float).reshape(-1)
            if velocity.size != self.nv:
                raise ValueError(
                    f"expected {self.nv} velocities, got {velocity.size}"
                )
        else:
            velocity = None

        with self._lock:
            if velocity is None:
                # Difference against the previous command, not against q_ref —
                # the first-order hold has already extrapolated q_ref forward,
                # so differencing that would cancel the very motion we're
                # trying to measure. The first command of a stream has nothing
                # to difference against, so it starts from the arm's measured
                # velocity rather than assuming rest.
                if self._mode == _MODE_SERVO and self._t > self._servo_t:
                    velocity = (target - self._servo_cmd) / (self._t - self._servo_t)
                else:
                    velocity = self.data.qvel[self._vadr].copy()

            self._mode = _MODE_SERVO
            self._q_ref = target
            self._servo_cmd = target
            self._v_ref = np.clip(velocity, -self.velocity_limit, self.velocity_limit)
            self._servo_t = self._t
            self._vel_deadline = None
            self._vel_target = np.zeros(self.nv)

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    def stop(self, wait=True, timeout=None):
        """Stop moving and hold the current position.

        Cancels whatever is running, including a servo stream — a later
        `servo_joints` call starts a fresh one. A keyboard interrupt out of a
        blocking motion call runs this too.
        """
        with self._lock:
            self._vel_deadline = None
            self._vel_target = np.zeros(self.nv)
            self._v_ref = np.zeros(self.nv)
            self._q_ref = self.data.qpos[self._qadr].copy()
            self._servo_t = -np.inf
            self._mode = _MODE_HOLD
        if not wait:
            return False
        return self.wait_for_motion(timeout=timeout)

    def check_safety(self, joints):
        """Return why `joints` would be refused, or None if it's allowed.

        Useful for testing a target before committing to it — the arm applies
        this same check to every setpoint it is given.
        """
        if self._guard is None:
            return None
        return self._guard.check(self._as_config(joints))

    @contextlib.contextmanager
    def _stop_on_interrupt(self):
        """Stop the arm if the wait inside is interrupted.

        Ctrl-C out of a blocking call should leave the arm standing still: the
        command it was waiting on is nobody's any more, and the sim thread
        would otherwise keep running it while the traceback prints.
        """
        try:
            yield
        except KeyboardInterrupt:
            self.stop(wait=False)
            raise

    def wait_for_motion(self, timeout=None):
        """Block until the arm settles. Returns False on timeout, or if the
        simulation has been shut down and can no longer make progress.

        A keyboard interrupt stops the arm before it propagates."""
        deadline = None if timeout is None else time.perf_counter() + float(timeout)
        with self._stop_on_interrupt():
            while True:
                with self._lock:
                    if self._is_settled():
                        return True
                if deadline is not None and time.perf_counter() >= deadline:
                    return False
                if self._thread is not None and not self._thread.is_alive():
                    return False
                if self._thread is None:
                    self.step(self.dt)  # no sim thread: drive it ourselves
                else:
                    time.sleep(self.dt)

    def _wait_until(self, deadline):
        """Block until the velocity command expiring at sim time `deadline` is
        done. Returns False if it was replaced, or if the sim can no longer
        make progress. A keyboard interrupt stops the arm before it
        propagates."""
        with self._stop_on_interrupt():
            while True:
                with self._lock:
                    if self._vel_deadline != deadline:
                        return False  # superseded by another command, or stopped
                    if self._t >= deadline:
                        return True
                if self._thread is None:
                    self.step(self.dt)  # no sim thread: drive it ourselves
                elif not self._thread.is_alive():
                    return False
                else:
                    time.sleep(self.dt)

    def step(self, duration=None):
        """Advance the simulation by `duration` seconds (default: one dt).

        Only useful when constructed with `real_time=False`; with the sim
        thread running it would double-step the physics, so it is a no-op.
        """
        if self._thread is not None:
            return
        n = 1 if duration is None else max(1, int(round(float(duration) / self.dt)))
        for _ in range(n):
            with self._lock:
                self._step_once()
        self._maybe_sync()

    def close(self):
        """Stop the simulation thread and close the viewer."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.viz is not None:
            self.viz.close()
            self.viz = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _index_arm(self):
        """Locate the arm's joints, DOFs and actuators (the MJCF also carries
        a gripper, which this API doesn't expose)."""
        jid, qadr, vadr, act = [], [], [], []
        for name in ARM_JOINTS:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError(f"joint {name!r} not found in the model")
            jid.append(j)
            qadr.append(self.model.jnt_qposadr[j])
            vadr.append(self.model.jnt_dofadr[j])

            driving = [
                a
                for a in range(self.model.nu)
                if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                and self.model.actuator_trnid[a, 0] == j
            ]
            if len(driving) != 1:
                raise ValueError(f"expected exactly one actuator for {name!r}")
            act.append(driving[0])
        return tuple(np.asarray(v) for v in (jid, qadr, vadr, act))

    def _base_frame(self):
        """The arm's base frame in the world, as a 4x4.

        The frame every joint configuration is written in — and so the frame
        the safety box is written in — is the one joint1 turns in, which an
        MJCF is free to put anywhere in its world. Menagerie's xarm7 mounts
        `link_base` at z = 120 mm, so the two frames are not the same one and
        anything given in base coordinates has to be carried across.
        """
        body = self.model.body_parentid[self.model.jnt_bodyid[self._jid[0]]]
        frame = np.eye(4)
        frame[:3, :3] = self.data.xmat[body].reshape(3, 3)
        frame[:3, 3] = self.data.xpos[body]
        return frame

    def _draw_safety_box(self):
        """Show the box in the viewer, if there is a viewer and a box."""
        if self.viz is None:
            return
        box = self._guard.box if self._guard is not None else None
        if box is None:
            self.viz.clear_safety_box()
        else:
            # In base coordinates, like the guard's own box — the picture would
            # otherwise sit 120 mm under the arm and libel safe poses.
            self.viz.draw_safety_box(box, frame=self._base_frame())
            self._update_safety_faces()

    def _update_safety_faces(self):
        """Leave drawn only the floor and the faces the arm is getting close to.

        Run off the joint angles MuJoCo is actually at rather than off the
        setpoint the guard checks, because this is about the arm in the
        picture: a face should appear as the arm you can see approaches it.

        Measuring the six gaps costs around 150 us, a tenth of a physics tick,
        which is why this goes on the viewer's clock and not the simulation's.
        """
        if self.viz is None or self._guard is None:
            return
        clearances = self._guard.box_clearances(self.data.qpos[self._qadr])
        near = [face for face, gap in clearances.items() if gap <= _FACE_PROXIMITY]
        self.viz.show_safety_faces(near + [_FLOOR_FACE])

    def _find_gripper_geoms(self):
        """Geoms belonging to the MJCF's gripper, by the bodies they hang off."""
        geoms = []
        for geom in range(self.model.ngeom):
            body = self.model.geom_bodyid[geom]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if any(part in name for part in _GRIPPER_BODIES):
                geoms.append(geom)
        return geoms

    def _remove_gripper(self):
        """Take the gripper out of the contact set and out of the picture.

        The real arm this mirrors has a bare flange, and the guard's model has
        no gripper either. Leaving the MJCF's one collidable would have the
        simulator jam on a part the hardware doesn't carry — and jam silently,
        since the guard would have nothing to say about it. Only the geometry
        goes; the mass stays, standing in for whatever is actually bolted on.
        """
        for geom in self._gripper_geoms:
            self.model.geom_contype[geom] = 0
            self.model.geom_conaffinity[geom] = 0
            # MuJoCo's own way of saying "don't draw this", which the meshcat
            # visualizer follows too.
            self.model.geom_group[geom] = _HIDDEN_GROUP

    def _reset_state(self, q0, keyframe):
        if keyframe is not None:
            key = keyframe
            if isinstance(key, str):
                key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key)
                if key < 0:
                    raise ValueError(f"keyframe {keyframe!r} not found")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            mujoco.mj_resetData(self.model, self.data)

        if q0 is not None:
            self.data.qpos[self._qadr] = self._clamp_position(self._as_config(q0))
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        q = self.data.qpos[self._qadr].copy()
        self._q_ref = q
        self._v_ref = np.zeros(self.nv)
        self._goal = q.copy()
        self._start = q.copy()
        self._direction = np.zeros(self.nv)
        self._span = 0.0
        self._travelled = 0.0
        self._speed = DEFAULT_SPEED
        self._servo_cmd = q.copy()
        self._servo_t = -np.inf
        self._guard_checked = q.copy()
        self._t = 0.0
        # Park every actuator we don't drive (the gripper) at its own setpoint.
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self._act] = q

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

    def _friction_torque(self, v):
        """Torque opposing motion (viscous + smoothed Coulomb)."""
        return self._damping * v + self._frictionloss * np.tanh(v / _STRIBECK_VELOCITY)

    def _is_settled(self):
        """Caller must hold the lock."""
        v = self.data.qvel[self._vadr]

        if self._mode == _MODE_POSITION:
            return False  # setpoint still travelling

        if self._mode == _MODE_SERVO:
            # A live stream never settles; a stale one settles once the arm
            # has caught up with the last setpoint it was given.
            if self._t - self._servo_t < SERVO_TIMEOUT:
                return False
            error = self.data.qpos[self._qadr] - self._q_ref
            return bool(
                np.max(np.abs(error)) <= _POSITION_TOLERANCE
                and np.max(np.abs(v)) <= _VELOCITY_TOLERANCE
            )

        if self._mode == _MODE_VELOCITY:
            commanded = self._vel_target
            if self._vel_deadline is not None and self._t >= self._vel_deadline:
                commanded = np.zeros(self.nv)
            if np.any(np.abs(commanded) > _VELOCITY_TOLERANCE):
                return False  # a live velocity command never "settles"
            return bool(np.max(np.abs(v)) <= _VELOCITY_TOLERANCE)

        error = self.data.qpos[self._qadr] - self._q_ref
        return bool(
            np.max(np.abs(error)) <= _POSITION_TOLERANCE
            and np.max(np.abs(v)) <= _VELOCITY_TOLERANCE
        )

    def _update_reference(self):
        """Advance the setpoint by one dt, and refuse to commit it if it would
        put the arm somewhere it may not be. Caller must hold the lock.

        Every mode funnels through here, so this is the one place that has to
        know about the guard. On a violation the last setpoint that checked out
        is put back and the arm holds, which brings it to rest at the boundary
        instead of driving through it. It has to be that one rather than simply
        the previous one: `servo_joints` writes its setpoint straight into the
        reference, so by the time the violation is seen the previous value may
        be the unsafe setpoint itself.
        """
        self._advance_reference()

        # Checking every tick would cost about as much as the physics; the arm
        # only needs re-checking once the setpoint has moved far enough to eat
        # into the margin.
        if np.max(np.abs(self._q_ref - self._guard_checked)) < self._guard_step:
            return

        violation = self._guard.check(self._q_ref)
        if violation is None:
            self._guard_checked = self._q_ref.copy()
            return

        self._q_ref = self._guard_checked.copy()
        self._v_ref = np.zeros(self.nv)
        self._vel_target = np.zeros(self.nv)
        self._vel_deadline = None
        self._servo_t = -np.inf
        self._mode = _MODE_HOLD
        self.last_violation = violation
        self._warn(violation)

    def _warn(self, violation):
        """Report a violation, at most once every `_WARN_PERIOD`.

        A servo stream pushed into the boundary would otherwise reprint this at
        the physics rate, burying whatever the caller was printing.
        """
        if self._t - self._warned_at < _WARN_PERIOD:
            return
        self._warned_at = self._t
        print(f"[xarm7] stopped at the safety boundary — {violation}")

    def _advance_reference(self):
        """Advance the setpoint by one dt. Caller must hold the lock."""
        if self._mode == _MODE_POSITION:
            self._travelled += self._speed * self.dt
            if self._travelled >= self._span:
                self._q_ref = self._goal
                self._v_ref = np.zeros(self.nv)
                self._mode = _MODE_HOLD
            else:
                self._q_ref = self._start + self._travelled * self._direction
                self._v_ref = self._speed * self._direction
            return

        if self._mode == _MODE_SERVO:
            # First-order hold: keep the setpoint moving at the commanded
            # velocity until the next command lands, then stop extrapolating
            # if the stream goes quiet.
            if self._t - self._servo_t >= SERVO_TIMEOUT:
                self._v_ref = np.zeros(self.nv)
            self._q_ref = self._clamp_position(self._q_ref + self._v_ref * self.dt)
            return

        if self._mode == _MODE_VELOCITY:
            target = self._vel_target
            if self._vel_deadline is not None and self._t >= self._vel_deadline:
                target = np.zeros(self.nv)
            self._v_ref = target
            self._q_ref = self._q_ref + self._v_ref * self.dt

            # Don't drive the setpoint past a joint limit.
            clamped = self._clamp_position(self._q_ref)
            at_limit = clamped != self._q_ref
            if np.any(at_limit):
                self._q_ref = clamped
                self._v_ref = np.where(at_limit, 0.0, self._v_ref)

            if not np.any(self._v_ref):
                self._mode = _MODE_HOLD
            return

        self._v_ref = np.zeros(self.nv)

    def _apply_control(self):
        """Write the servo setpoints. Caller must hold the lock."""
        ctrl = self._q_ref
        if self._feedforward:
            # Torque the servo would otherwise need a tracking error to make.
            torque = (
                self.data.qfrc_bias[self._vadr]
                + self._friction_torque(self._v_ref)
                + self._kv * self._v_ref  # cancel the servo's own damping term
            )
            ctrl = ctrl + torque / self._kp

        self.data.ctrl[self._act] = np.clip(
            ctrl,
            self.model.actuator_ctrlrange[self._act, 0],
            self.model.actuator_ctrlrange[self._act, 1],
        )

    def _step_once(self):
        """One physics tick. Caller must hold the lock."""
        self._update_reference()
        self._apply_control()
        mujoco.mj_step(self.model, self.data)
        self._t = self.data.time

    def _maybe_sync(self):
        """Push state to the viewer. Called from the thread that steps."""
        if self.viz is None or self._viz_period is None:
            return
        now = time.perf_counter()
        if now - self._last_viz < self._viz_period:
            return
        self._last_viz = now
        try:
            self._update_safety_faces()
            self.viz.sync()
        except Exception as err:  # a dead viewer must not take the sim with it
            self.viz = None
            print(f"[xarm7_mujoco] visualization disabled: {err!r}")

    def _run(self):
        """Background loop pacing the physics to wall time."""
        tick = max(self.dt, 2e-3)
        max_steps = max(1, int(0.05 / self.dt))
        wall0 = time.perf_counter()
        sim0 = self._t
        while not self._stop_event.is_set():
            target_t = sim0 + (time.perf_counter() - wall0)
            with self._lock:
                steps = 0
                while self._t < target_t and steps < max_steps:
                    self._step_once()
                    steps += 1
                behind = target_t - self._t
                self._maybe_sync()
            if behind > 0.05:  # can't keep up; drop the backlog rather than spiral
                wall0 = time.perf_counter()
                sim0 = self._t
            time.sleep(tick)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    with SimulatedXArm7(visualize=True) as arm:
        print("joints:", arm.joint_names)

        target = np.array([0.0, -0.6, 0.0, 0.9, 0.0, 1.5, 0.0])
        print("moving to", target)
        reached = arm.set_joint_targets(target, speed=1.0)
        print("reached:", reached)
        print("  q   =", arm.joint_values)
        print("  tau =", arm.joint_efforts)

        print("jogging joint 1 for 2 s")
        arm.set_velocity([0.5, 0, 0, 0, 0, 0, 0], duration=2.0)
        time.sleep(1.0)
        print("  v   =", arm.joint_velocities)
        arm.wait_for_motion()
        print("  q   =", arm.joint_values)

        print("returning home")
        arm.set_joint_targets(np.zeros(7))
        print("  q   =", arm.joint_values)
        print("  tau =", arm.joint_efforts, "(holding against gravity)")

        time.sleep(2.0)
