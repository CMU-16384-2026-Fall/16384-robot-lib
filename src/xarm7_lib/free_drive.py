"""Tunables and data types for `RealXArm7.free_drive`.

Free drive lets a person push the arm around by hand and records where they
took it. Every joint is free unless you ask for otherwise, which is also all
the controller can do: its one hand-guiding mode — joint teaching,
`set_mode(2)` — frees all seven joints at once and gravity-compensates the arm,
and there is no way to tell it to hold some of them.

A run is two calls. `free_drive()` releases the arm and returns at once,
leaving a thread sampling and watching it; `end_free_drive()` ends the run,
locks the arm and hands back the recording. So the waiting is the caller's, and
they can wait however they like:

    robot.free_drive()
    input("push the arm around, then press Enter: ")
    traj = robot.end_free_drive()

`watch=` is for the joints that are meant to stay put anyway. They are not
locked, they are *watched*. The arm still sits in joint teaching mode with
everything free, and the loop in `xarm7_real.py` reads the joint positions and
compares the watched ones against the values latched when the run started. Past
`tolerance` on any of them the run ends itself: back to position control, which
stiffens the arm where it stands. Nothing moves then — the watch loop never
drives the arm, because the person waiting at `input()` has no way to be asked
to let go of it first. The put-back is `end_free_drive`'s job, where there is
somebody to prompt.

That is the whole trick, and it is worth being clear about what it buys and
what it does not. It buys a lock that is *reported* and then restored, rather
than held: between the moment a joint starts to move and the moment the loop
notices, that joint is free, and it will have moved by up to `tolerance` plus
whatever a report period of lag allows. What it avoids is fighting the
controller's own servo — the previous attempt at this was an admittance loop
streaming setpoints in servo mode, which needed a gravity model, a torque sign
convention, per-joint damping above a stability floor set by servo gains the
controller does not publish, and tuning per arm. Joint teaching needs none of
that: the controller does the gravity compensation with its own identified
model of itself.

Nothing here needs a robot — the constants you tune, the mask arithmetic, the
trajectory the loop hands back — so it can be read and changed without one.

Running this module (`python -m xarm7_lib.free_drive`) does none of it. That is
the plain drag-teach underneath: mode 2, nothing recorded and the guard off, for
checking the link to the arm and for pushing one back inside the safety box
after it has been parked outside it.
"""

import dataclasses
import os

import numpy as np

FREE_DRIVE_RATE = 100.0  # Hz the watched joints are sampled at
RATE_LIMITS = (20.0, 250.0)  # Hz; the report stream is the real limit on what
# a higher rate can tell you, but oversampling it costs nothing

# rad a watched joint may drift before the run interrupts itself, ~2.9 deg.
# This is a distance, not a speed, so what sets it is how far the arm may be
# wrong before the pose stops meaning what the lab says it means — not how fast
# the loop can react. Tighter than about 0.02 rad and ordinary handling of the
# free joints will trip it through the arm's own structure, and the person
# spends the session answering prompts.
LOCKED_TOLERANCE = 0.05
# Warn at a fraction of it, so there is a chance to back off before the
# interrupt rather than only being told afterwards.
LOCKED_WARN_FRACTION = 0.5

RECOVER_SPEED = 0.2  # rad/s for the move that puts a watched joint back
WARN_PERIOD = 1.0  # s between repeats of a warning, as in the simulator
STALL_ABORT = 0.5  # s gap in the loop after which the run ends
# A stall matters more here than it would in a controller-side loop: the watch
# on the locked joints *is* the lock, and while the loop is not running there
# is nothing enforcing it. Ending the run is what puts the arm back into
# position control, where the joints are held rather than watched.

# How long `end_free_drive` waits for the watch loop to notice it is over
# before giving up on it and locking the arm itself. A tick is milliseconds, so
# anything near this means the thread is stuck in a call to the controller.
JOIN_TIMEOUT = 5.0  # s

# The 16-384 lab setup. Joints 1, 4 and 7 are the free ones, and the four
# between them are posed so that all three sit on axes gravity has no moment
# about — the free joints carry nothing, wherever they are driven.
#
# The locked values are what defines the pose; the free ones are only a
# starting point. From here the guard allows joint1 about +/-88 deg, joint4
# about 99 deg, and joint7 all the way round.
HOME_POSE = (
    -np.pi * 7 / 18,  # joint1, -70 deg   free
    np.pi / 2,        # joint2,  90 deg   locked
    np.pi / 2,        # joint3,  90 deg   locked
    np.pi * 7 / 18,   # joint4, +70 deg   free
    -np.pi / 2,       # joint5, -90 deg   locked
    np.pi / 2,        # joint6,  90 deg   locked
    0.0,              # joint7,   0 deg   free
)
FREE_JOINTS = (0, 3, 6)  # 0-indexed: joint1, joint4, joint7
WATCHED_JOINTS = (1, 2, 4, 5)  # the other four: what `HOME_POSE` is defined by,
# and so what `free_drive(watch=)` wants to be given to hold that pose


def watch_mask(watch, nq=7):
    """Turn the joints to watch into the free mask the loop uses, (nq,).

    Returns a boolean array in which True means *free* — the sense the watch
    loop wants — so `watch_mask(())` is all True and nothing is watched.

    `watch` is a sequence of **0-indexed** joint numbers to watch — `[1, 2]`
    watches joint2 and joint3 and frees the other five — or a length-`nq`
    boolean array in which True means watched. None and `()` watch nothing.
    Joint names are deliberately not accepted: the SDK's own `servo_id` is
    1-indexed, and the one thing worth being loud about here is which
    convention applies.

    A duplicate index is an error rather than something to quietly drop, since
    it usually means the caller miscounted.
    """
    if watch is None:
        return np.ones(nq, dtype=bool)

    array = np.asarray(watch)
    if array.dtype == bool:
        if array.shape != (nq,):
            raise ValueError(
                f"a boolean mask must have {nq} entries, got shape {array.shape}"
            )
        watched = array.copy()
    else:
        indices = []
        for value in np.atleast_1d(array).tolist():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"joint indices must be integers, got {value!r} "
                    f"({type(value).__name__})"
                )
            index = int(value)
            if not 0 <= index < nq:
                raise ValueError(
                    f"joint index {index} is out of range for a {nq}-joint arm. "
                    f"Joints are 0-indexed here, so joint{nq} is index {nq - 1}."
                )
            if index in indices:
                raise ValueError(
                    f"duplicate joint index {index} in watch={list(watch)!r}"
                )
            indices.append(index)

        watched = np.zeros(nq, dtype=bool)
        watched[indices] = True

    if watched.all():
        raise ValueError(
            "every joint is watched, so there is nothing to push — free drive "
            "would only hold position"
        )
    return ~watched


def hands_off_prompt(message):
    """Ask at the terminal, and wait. The default for `free_drive(confirm=)`.

    Only ever called from `end_free_drive`, on the caller's own thread, so this
    is the only thing reading stdin and a plain `input` is enough.

    Returns True to carry on, False to leave the arm where it is — which is
    what a closed stdin means, since a prompt nobody can answer must not be
    taken for a person with their hands clear.
    """
    print(message)
    try:
        input("        press Enter when your hands are clear: ")
    except EOFError:
        print("        (nothing on stdin to answer with; not moving the arm)")
        return False
    return True


@dataclasses.dataclass
class Trajectory:
    """What a hand-guided run recorded.

    `t` is seconds from the start of the run, `q` joint positions in radians and
    `qd` joint velocities in radians per second, sampled once per loop tick —
    so `q[k]` and `qd[k]` are the arm at `t[k]`, and `q` is what a forward
    kinematics or Jacobian check should be run against.

    `interruptions` is how many times a watched joint drifted past tolerance —
    at most one, since that is what ends the run, and always zero for a run
    that watched nothing, which is the default. The sampling itself is uniform
    either way: the arm is never moved under its own power while a run is
    recording, so there are no gaps in `t`. The put-back happens in
    `end_free_drive`, after the last sample.

    `qd_source` says where the velocities came from. "report" is the
    controller's own realtime joint speeds. "difference" means those weren't
    available and `q` was differentiated instead, which matters if you are
    checking a Jacobian: you would then be comparing one finite difference
    against another rather than against an independent measurement.

    `reason` says why the run ended, which is how a 30-second recording that
    actually stopped after four seconds explains itself.
    """

    t: np.ndarray  # (N,) s, starting at 0.0
    q: np.ndarray  # (N, 7) rad
    qd: np.ndarray  # (N, 7) rad/s
    joint_names: list
    free: np.ndarray  # (7,) bool
    rate: float  # loop rate the run asked for, Hz
    report_rate: float  # Hz the state was seen to change at, a lower bound:
    # it cannot exceed `rate`, since the loop only looks that often
    qd_source: str  # "report" or "difference"
    reason: str
    overruns: int  # ticks the loop failed to keep up with
    interruptions: int  # times a watched joint had to be put back

    def __len__(self):
        return int(self.t.size)

    def __repr__(self):
        free = np.flatnonzero(self.free).tolist()
        span = float(self.t[-1]) if self.t.size else 0.0
        return (
            f"Trajectory({len(self)} samples, {span:.1f} s, free={free}, "
            f"qd={self.qd_source}, {self.interruptions} interruptions, "
            f"{self.reason!r})"
        )

    @property
    def free_indices(self):
        """The joints that were free, as 0-indexed numbers."""
        return np.flatnonzero(self.free).tolist()

    @property
    def watched_indices(self):
        """The joints that were watched, as 0-indexed numbers. Empty unless
        the run was given `watch=`."""
        return np.flatnonzero(~self.free).tolist()

    def as_dict(self):
        """Just the three arrays, for `np.savez` or a DataFrame."""
        return {"t": self.t, "q": self.q, "qd": self.qd}

    def save(self, path):
        """Write the whole recording, metadata included, as a .npz."""
        np.savez(
            path,
            t=self.t,
            q=self.q,
            qd=self.qd,
            joint_names=np.asarray(self.joint_names),
            free=self.free,
            rate=self.rate,
            report_rate=self.report_rate,
            qd_source=self.qd_source,
            reason=self.reason,
            overruns=self.overruns,
            interruptions=self.interruptions,
        )

    @classmethod
    def load(cls, path):
        """Read back what `save` wrote."""
        with np.load(path, allow_pickle=False) as data:
            return cls(
                t=data["t"],
                q=data["q"],
                qd=data["qd"],
                joint_names=[str(name) for name in data["joint_names"]],
                free=data["free"],
                rate=float(data["rate"]),
                report_rate=float(data["report_rate"]),
                qd_source=str(data["qd_source"]),
                reason=str(data["reason"]),
                overruns=int(data["overruns"]),
                interruptions=int(data["interruptions"]),
            )


if __name__ == "__main__":
    import argparse

    from .xarm7_real import MODE_POSITION, MODE_TEACH, RealXArm7

    parser = argparse.ArgumentParser(
        prog="python -m xarm7_lib.free_drive",
        description=(
            "Release every joint so the arm can be pushed by hand, then lock "
            "it wherever you left it. Nothing is recorded and the guard is "
            "off — which is what makes this the way to move an arm that "
            "started somewhere the guard would not allow. For a run that is "
            "recorded, guarded, and can hold some joints, call "
            "`Robot.free_drive` from your own script."
        )
    )
    parser.add_argument(
        "ip",
        nargs="?",
        default=os.environ.get("ROBOT_IP", "").strip(),
        help="controller address, e.g. 192.168.1.185. Defaults to $ROBOT_IP.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="don't ask first")
    args = parser.parse_args()
    if not args.ip:
        parser.error("no controller address: pass one, or set ROBOT_IP")

    np.set_printoptions(precision=3, suppress=True)

    def show(arm, when):
        q = arm.joint_values
        print(f"  joints {when}  {np.degrees(q)} deg")
        return q

    # `guard=False` is the whole point: the guard refuses to let the arm start
    # from a pose it would not allow, and an arm parked out of the box is
    # exactly what someone reaches for this to fix. It also skips loading the
    # collision meshes, so this connects in about a second — which is what
    # makes it a decent test of whether anything can talk to the arm at all.
    with RealXArm7(args.ip, guard=False) as arm:
        print(f"connected to {args.ip}")
        show(arm, "now ")

        if not args.yes:
            print(
                "\nEvery joint is about to be released. The controller holds "
                "the arm up against\ngravity, but nothing else is watching it: "
                "it will go wherever you push it,\nincluding into itself or out "
                "of the safety box. Keep the e-stop within reach."
            )
            if input("continue? [y/N] ").strip().lower() not in ("y", "yes"):
                raise SystemExit(0)

        arm._ensure_mode(MODE_TEACH)
        print("\nthe arm is free — push it where you want it")
        try:
            input("press Enter to lock it there (Ctrl-C does the same): ")
        except EOFError:
            # Nothing on stdin to wait for, so there was never a chance to push
            # the arm. Still a complete round trip: connected, released, locked.
            print("(nothing on stdin to wait for)")
        except KeyboardInterrupt:
            print()

        # Explicit rather than left to `close()`, which would do the same on the
        # way out: it locks the arm before the pose below is read, so what gets
        # printed is where the arm is holding, not where a hand still on it had
        # it a moment earlier.
        arm._ensure_mode(MODE_POSITION)
        print("locked.")
        q = show(arm, "left")
        print(f"  as radians   {q.tolist()}")
