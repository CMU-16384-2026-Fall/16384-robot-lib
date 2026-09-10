"""Tunables and data types for `RealXArm7.free_drive`.

Free drive lets a person push the arm around by hand while a chosen subset of
joints is supposed to stay put. The controller has one mode for hand-guiding —
joint teaching, `set_mode(2)` — and it frees all seven joints at once. There is
no way to tell it to hold four of them.

So the four are not locked, they are *watched*. The arm sits in joint teaching
mode with everything free and gravity-compensated by the controller, and the
loop in `xarm7_real.py` reads the joint positions and compares the watched ones
against the values latched when the run started. Past `tolerance` on any of
them the run interrupts itself: back to position control, which stiffens the
arm where it stands, a prompt asking the person to let go, then a slow move
that puts the watched joints back on their latched values, then back to
teaching. The recording continues afterwards with a gap in it.

That is the whole trick, and it is worth being clear about what it buys and
what it does not. It buys a lock that is *restored* rather than *held*: between
the moment a joint starts to move and the moment the loop notices, that joint is
free, and it will have moved by up to `tolerance` plus whatever a report period
of lag allows. What it avoids is fighting the controller's own servo — the
previous attempt at this was an admittance loop streaming setpoints in servo
mode, which needed a gravity model, a torque sign convention, per-joint damping
above a stability floor set by servo gains the controller does not publish, and
tuning per arm. Joint teaching needs none of that: the controller does the
gravity compensation with its own identified model of itself.

Nothing here needs a robot — the constants you tune, the mask arithmetic, the
trajectory the loop hands back — so it can be read and changed without one.
"""

import dataclasses

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

# A recovery that is undone again within this long is not the person pushing,
# it is the pose failing to hold — a joint the controller's own compensation
# leaves drifting under a load the model doesn't know about, say. Repeat it
# this many times in a row and the run ends and says so, rather than prompting
# forever.
REPEAT_WINDOW = 1.0  # s of teaching after a recovery
REPEAT_LIMIT = 3

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


def free_mask(free, nq=7):
    """Turn a set of joint indices into a boolean mask, (nq,).

    `free` is a sequence of **0-indexed** joint numbers — `[0, 1, 3]` frees
    joint1, joint2 and joint4 — or a length-`nq` boolean array. Joint names are
    deliberately not accepted: the SDK's own `servo_id` is 1-indexed, and the
    one thing worth being loud about here is which convention applies.

    A duplicate index is an error rather than something to quietly drop, since
    it usually means the caller miscounted.
    """
    array = np.asarray(free)
    if array.dtype == bool:
        if array.shape != (nq,):
            raise ValueError(
                f"a boolean mask must have {nq} entries, got shape {array.shape}"
            )
        mask = array.copy()
        if not mask.any():
            raise ValueError("no joints are free; free drive would only hold position")
        return mask

    indices = []
    for value in np.atleast_1d(array).tolist():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"joint indices must be integers, got {value!r} ({type(value).__name__})"
            )
        index = int(value)
        if not 0 <= index < nq:
            raise ValueError(
                f"joint index {index} is out of range for a {nq}-joint arm. "
                f"Joints are 0-indexed here, so joint{nq} is index {nq - 1}."
            )
        if index in indices:
            raise ValueError(f"duplicate joint index {index} in free={list(free)!r}")
        indices.append(index)

    if not indices:
        raise ValueError("free is empty; free drive would only hold position")

    mask = np.zeros(nq, dtype=bool)
    mask[indices] = True
    return mask


def hands_off_prompt(message):
    """Ask at the terminal, and wait. The default for `free_drive(confirm=)`.

    Returns True to carry on, False to end the run — which is what a closed
    stdin means, since a run that cannot ask anyone to let go of the arm must
    not start moving it.
    """
    print(message)
    try:
        input("        press Enter when your hands are clear: ")
    except EOFError:
        print("        (nothing on stdin to answer with; ending the run)")
        return False
    return True


@dataclasses.dataclass
class Trajectory:
    """What a hand-guided run recorded.

    `t` is seconds from the start of the run, `q` joint positions in radians and
    `qd` joint velocities in radians per second, sampled once per loop tick —
    so `q[k]` and `qd[k]` are the arm at `t[k]`, and `q` is what a forward
    kinematics or Jacobian check should be run against.

    `interruptions` is how many times a watched joint drifted past tolerance and
    the run stopped to put it back. Nothing is sampled while that happens, so
    each one leaves a **gap in `t`** — consecutive samples either side of a gap
    are seconds apart and the arm moved under its own power in between. If
    `interruptions` is zero the sampling is uniform and there is nothing to
    worry about; if it isn't, `np.diff(t)` finds the seams.

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

    from .xarm7_real import RealXArm7

    parser = argparse.ArgumentParser(
        description="Run one hand-guided session from the lab's home pose."
    )
    parser.add_argument("ip", help="controller address, e.g. 192.168.1.185")
    parser.add_argument("-d", "--duration", type=float, default=30.0, help="seconds")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=LOCKED_TOLERANCE,
        help="rad a locked joint may drift before the run puts it back",
    )
    parser.add_argument("-o", "--save", help="write the recording to this .npz")
    parser.add_argument("-y", "--yes", action="store_true", help="don't ask first")
    args = parser.parse_args()

    np.set_printoptions(precision=3, suppress=True)

    if not args.yes:
        print(
            "This moves the arm to the home pose, then releases every joint so "
            "you can push it by hand.\nClear the workspace and keep the e-stop "
            "within reach."
        )
        if input("continue? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit(0)

    with RealXArm7(args.ip) as arm:
        print("moving to the home pose")
        arm.set_joint_targets(HOME_POSE, speed=0.4)

        traj = arm.free_drive(duration=args.duration, tolerance=args.tolerance)
        print(traj)
        print(f"  q first = {traj.q[0]}" if len(traj) else "  nothing recorded")
        print(f"  q last  = {traj.q[-1]}" if len(traj) else "")
        if args.save:
            traj.save(args.save)
            print(f"  written to {args.save}")
