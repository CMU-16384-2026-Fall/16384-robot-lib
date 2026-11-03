"""Tunables and data types for `RealXArm7.free_drive`.

Free drive lets a person push the arm around by hand and records where they
took it. Every joint is free unless you ask for otherwise, which is also all
the controller can do: its one hand-guiding mode — joint teaching,
`set_mode(2)` — frees all seven joints at once and gravity-compensates the arm,
and there is no way to tell it to hold some of them.

`watch=` is for the joints that are meant to stay put anyway. They are not
locked, they are *watched*. The arm still sits in joint teaching mode with
everything free, and the loop in `xarm7_real.py` reads the joint positions and
compares the watched ones against the values latched when the run started. Past
`tolerance` on any of them the run interrupts itself: back to position control, which stiffens the
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

Running this module (`python -m xarm7_lib.free_drive`) does none of it. That is
the plain drag-teach underneath: mode 2, nothing recorded and the guard off, for
checking the link to the arm and for pushing one back inside the safety box
after it has been parked outside it.
"""

import dataclasses
import io
import os
import select
import sys

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


# Watching the terminal without owning it. The condition `stop_on_enter`
# hands back is asked once a tick by the watch loop, and it *peeks* at stdin
# with `select` rather than reading it: nothing is waiting on stdin between
# ticks, and nothing at all is once the run is over.
#
# A reader thread is the obvious way to do this and the wrong one. A thread
# blocked in `readline` goes on owning stdin after the run that started it has
# ended, there is no portable way to call it back, and in an interactive
# session the next thing typed at the prompt disappears into it instead of
# reaching Python.
_STDIN_EOF = False
_STDIN_BUFFER = bytearray()


def _stdin_fd():
    """stdin's file descriptor, or None if it hasn't got one.

    A notebook kernel or an IDE console replaces `sys.stdin` with an object
    that has no descriptor to select on. Worth saying plainly rather than
    crashing over.
    """
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        return None
    return fd if fd >= 0 else None


def _stdin_waiting(fd):
    """Whether stdin has anything to read this instant.

    Raises if `select` cannot answer for this kind of descriptor, which on
    Windows is most of them.
    """
    return bool(select.select([fd], [], [], 0.0)[0])


def _stdin_take_line(fd):
    """True if a whole line has been typed since this was last asked.

    Consumes that line, so one Enter cannot both stop a run and answer a
    prompt. Reads the descriptor rather than `sys.stdin`, which would block
    on a partial line and would hide what it had read in a buffer of its own
    where a later `input()` could not reach it.
    """
    global _STDIN_EOF
    while not _STDIN_EOF:
        try:
            if not _stdin_waiting(fd):
                break
            chunk = os.read(fd, 4096)
        except (OSError, ValueError):
            _STDIN_EOF = True
            break
        if not chunk:
            _STDIN_EOF = True
            break
        _STDIN_BUFFER.extend(chunk)

    if b"\n" not in _STDIN_BUFFER:
        return False
    del _STDIN_BUFFER[: _STDIN_BUFFER.index(b"\n") + 1]
    return True


def drain_terminal():
    """Throw away anything typed but not yet read.

    For a prompt that must not be answered by a newline from earlier — "take
    your hands off the robot" above all.
    """
    fd = _stdin_fd()
    if fd is None:
        return
    while _stdin_take_line(fd):
        pass
    _STDIN_BUFFER.clear()


def read_line(prompt=""):
    """Prompt for a line at the terminal, ignoring anything typed before now."""
    drain_terminal()
    return input(prompt)


def stop_on_enter(announce=True):
    """A `free_drive(until=)` condition that fires when Enter is pressed.

        robot.free_drive(None, until=stop_on_enter())

    Returns a callable the watch loop asks once a tick. It peeks at the
    terminal instead of reading it, so the arm goes on being sampled and
    watched while it waits — and so the run leaves nothing behind holding
    stdin, which in an interactive session would swallow the next thing typed
    at the prompt. Ctrl-C still ends a run too; this is for the student who
    shouldn't have to reach for it.

    Enter pressed at a prompt the run puts up — the hands-off warning — does
    not also stop the run: that prompt ignores anything typed before it.

    Anything typed before the condition is made is dropped, so a run started
    by pressing Enter is not stopped by the same keystroke.

    `announce` prints the one-line instruction. Pass False if your own banner
    already says it. `reason` on the returned object is what
    `Trajectory.reason` will say.
    """
    fd = _stdin_fd()
    if fd is None:
        unusable = "stdin has no file descriptor here"
    else:
        try:
            _stdin_waiting(fd)
        except (OSError, ValueError):
            unusable = "stdin cannot be polled on this platform"
        else:
            _stdin_take_line(fd)
            unusable = "stdin is closed" if _STDIN_EOF else None

    if unusable is not None:
        # Say so now: with `duration=None` as well, nothing would be left to
        # end the run.
        print(
            f"[xarm7] {unusable}, so Enter will never be seen — this run needs "
            "its duration, another stop condition, or Ctrl-C to end"
        )

        class _NeverStops:
            reason = "stopped by Enter at the terminal"

            def __call__(self):
                return False

            def __repr__(self):
                return "stop_on_enter() [the terminal cannot be watched]"

        return _NeverStops()

    if announce:
        print("[xarm7] press Enter to stop the run")

    class _EnterStop:
        reason = "stopped by Enter at the terminal"

        def __call__(self):
            return _stdin_take_line(fd)

        def __repr__(self):
            return "stop_on_enter()"

    return _EnterStop()


def hands_off_prompt(message):
    """Ask at the terminal, and wait. The default for `free_drive(confirm=)`.

    Returns True to carry on, False to end the run — which is what a closed
    stdin means, since a run that cannot ask anyone to let go of the arm must
    not start moving it.
    """
    print(message)
    try:
        read_line("        press Enter when your hands are clear: ")
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
    the run stopped to put it back — always zero for a run that watched nothing,
    which is the default. Nothing is sampled while that happens, so
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
