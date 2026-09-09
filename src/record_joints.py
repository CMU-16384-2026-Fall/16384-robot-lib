"""Record the arm's joint angles to a CSV. Reads only — commands nothing.

    python record_joints.py --real                  # until ctrl-c
    python record_joints.py --real --duration 30    # 30 seconds
    python record_joints.py --sim --out /tmp/x.csv  # no hardware needed

Meant to run in a second terminal beside `goto_pose.py --real --guided`, while
a student swings the arm by hand, so the motion can be reconstructed in a later
assignment.

That is why this opens its own connection rather than borrowing the library's
`RealXArm7`: that class's constructor calls `motion_enable(True)` and puts the
controller into position mode, which would re-lock the very joints the guided
hold just released. A bare `XArmAPI` is passive — `connect()` sends the protocol
identifier, a debug flag and a timeout, then opens the report socket, and
nothing else — so a second process can watch the arm without touching it.

It listens on report port 30003 (`report_type='real'`), which carries joint
angles far more often than the rich report on 30002 that `RealXArm7` itself
uses. The two connections read different sockets and never compete.

A hardware run here loads only the xarm SDK — no mujoco, no pinocchio — because
the one thing it borrows from `goto_pose.py` (the `ip.txt` lookup) sits above
that module's own imports.

The file is one header row and then a row per sample, in the library's own
units — **seconds and radians**, not degrees:

    t,joint1,joint4,joint7

Those are the three joints the guided hold releases, the ones a student can
actually turn; the other four are clamped at the pose that makes the arm
planar and would only write four constant columns. Pass `--all-joints` to
record all seven anyway.

The angles are **absolute** — each is the joint's own position as the
controller reports it, not an offset from wherever the recording started. Only
`t` is relative, starting at zero on the first sample. To read one back:

    import numpy as np
    data = np.loadtxt("recordings/joints-....csv", delimiter=",", skiprows=1)
    t, q = data[:, 0], data[:, 1:]      # (n,) and (n, 3)
"""

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from goto_pose import FREE_SERVOS, controller_ip

# The joints worth recording, as the controller numbers them: exactly the ones
# `goto_pose.py --guided` releases, so the two scripts cannot disagree about
# which joints are the free ones.
DEFAULT_JOINTS = FREE_SERVOS
ALL_JOINTS = tuple(range(1, 8))

DEFAULT_RATE = 50.0  # Hz
_STATUS_PERIOD = 0.5  # s between status lines on a terminal
_LOGGED_STATUS_PERIOD = 2.0  # s between them when stdout isn't a terminal
_REPORT_TIMEOUT = 10.0  # s to wait for the first reading


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--real", action="store_true",
        help="record the real arm. Without this it records its own MuJoCo "
        "simulation, which is useful for checking the file format but will sit "
        "perfectly still.",
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="record the simulation (the default; say it explicitly if you like)",
    )
    parser.add_argument(
        "--ip",
        help="controller address. Defaults to the contents of ip.txt, in the "
        "working directory or next to this script.",
    )
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_RATE,
        help="samples per second (default: %(default)s)",
    )
    parser.add_argument(
        "--duration", type=float,
        help="seconds to record for. Without this it runs until ctrl-c.",
    )
    parser.add_argument(
        "--all-joints", action="store_true",
        help="record all seven joints instead of just the three the guided hold "
        f"releases ({', '.join(f'joint{n}' for n in DEFAULT_JOINTS)})",
    )
    parser.add_argument(
        "--out",
        help="file to write (default: recordings/joints-<timestamp>.csv)",
    )
    args = parser.parse_args(argv)
    if args.real and args.sim:
        raise SystemExit("--real and --sim are opposites; pass one or neither.")
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")
    return args


def default_path():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("recordings") / f"joints-{stamp}.csv"


class RealSource:
    """A read-only view of the arm's joint angles, on its own connection."""

    def __init__(self, ip):
        from xarm.wrapper import XArmAPI

        # is_radian so the `angles` property comes back in the library's units;
        # report_type='real' for port 30003, the fast one.
        self.api = XArmAPI(ip, is_radian=True, report_type="real")
        if not self.api.connected:
            self.api.connect()

        # One round-trip read, both to prove the link works and to seed the
        # cached angles before the first report lands. This is a query, not a
        # command — it changes nothing on the controller.
        deadline = time.monotonic() + _REPORT_TIMEOUT
        while True:
            code, _ = self.api.get_servo_angle(is_radian=True)
            if code == 0:
                break
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"no reading from the controller within {_REPORT_TIMEOUT}s "
                    f"(last code {code})"
                )
            time.sleep(0.05)

    def read(self):
        return np.asarray(self.api.angles[:7], dtype=float)

    def close(self):
        self.api.disconnect()


class SimSource:
    """The same view of a simulated arm, for testing without hardware."""

    def __init__(self):
        from xarm7_lib import SimulatedXArm7

        self.arm = SimulatedXArm7(visualize=False)

    def read(self):
        return self.arm.joint_values

    def close(self):
        self.arm.close()


def open_source(args):
    if args.real:
        ip = controller_ip(args.ip, prefix="rec")
        print(f"[rec] connecting to {ip}, read-only")
        return RealSource(ip)
    print("[rec] recording the simulation; it will sit still")
    return SimSource()


def main(argv=None):
    args = parse_args(argv)
    path = Path(args.out) if args.out else default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    period = 1.0 / args.rate

    joints = ALL_JOINTS if args.all_joints else DEFAULT_JOINTS
    columns = [n - 1 for n in joints]  # the controller counts from 1, numpy from 0

    source = open_source(args)
    rows = 0
    fresh = 0  # samples that differed from the one before
    started = None
    previous = None
    low = np.full(len(columns), np.inf)
    high = np.full(len(columns), -np.inf)

    try:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t"] + [f"joint{n}" for n in joints])
            print(f"[rec] recording {', '.join(f'joint{n}' for n in joints)} — "
                  "absolute angles, in radians")

            print(f"[rec] writing {path} at {args.rate:g} Hz; ctrl-c to stop.")
            # In place on a terminal, one line at a time into a file — the same
            # split `goto_pose.py`'s monitor makes.
            live = sys.stdout.isatty()
            status_period = _STATUS_PERIOD if live else _LOGGED_STATUS_PERIOD
            next_tick = time.perf_counter()
            next_status = next_tick
            while True:
                now = time.perf_counter()
                q = source.read()[columns]
                if started is None:
                    started = now
                writer.writerow(
                    [f"{now - started:.4f}"] + [f"{v:.6f}" for v in q]
                )
                rows += 1
                if previous is None or not np.array_equal(q, previous):
                    fresh += 1
                previous = q
                low = np.minimum(low, q)
                high = np.maximum(high, q)

                if now >= next_status:
                    next_status = now + status_period
                    shown = " ".join(f"{math.degrees(v):7.2f}" for v in q)
                    line = (f"[rec] {rows:6d} rows  {now - started:6.1f}s  "
                            f"[{shown}] deg")
                    if live:
                        print(f"\r{line}", end="", flush=True)
                    else:
                        print(line, flush=True)

                if args.duration is not None and now - started >= args.duration:
                    break

                # Absolute deadlines rather than sleeping a period each time,
                # so the sample times don't drift with the loop's own cost.
                next_tick += period
                remaining = next_tick - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:  # fell behind; give up on the backlog rather than spiral
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print()
        return report(path, rows, fresh, started, args.rate,
                      joints, low, high) or 130
    finally:
        source.close()

    print()
    return report(path, rows, fresh, started, args.rate, joints, low, high)


def report(path, rows, fresh, started, asked, joints, low, high):
    """Say what was written. Returns 0, or 1 if nothing was."""
    if not rows or started is None:
        print(f"[rec] nothing recorded; {path} is empty.")
        return 1

    elapsed = time.perf_counter() - started
    achieved = rows / elapsed if elapsed > 0 else 0.0
    print(f"[rec] wrote {rows} rows to {path}")
    print(f"[rec] {elapsed:.1f}s at {achieved:.1f} Hz (asked for {asked:g})")

    # Every sample is written, but if the report stream updates more slowly than
    # we poll, consecutive rows repeat. That is worth knowing before someone
    # differentiates the file and wonders why the velocity is a staircase.
    share = fresh / rows
    print(f"[rec] {fresh} of {rows} samples were new readings ({share * 100:.0f}%)")
    if share < 0.5:
        print(f"[rec] the report stream is updating well below {asked:g} Hz — "
              "either the arm\n      was still, or a lower --rate would record "
              "the same information.")

    # How far each joint actually travelled. A column that never moved is the
    # thing to know about: it says the joint was not released, was not turned,
    # or is not being reported — and which of those it is, is a question for
    # the arm rather than for this file.
    travel = np.degrees(high - low)
    print("[rec] travel: " + "  ".join(
        f"joint{n} {t:.2f}" for n, t in zip(joints, travel)
    ) + " deg")
    still = [n for n, t in zip(joints, travel) if t < 0.05]
    if still and len(still) < len(joints):
        names = ", ".join(f"joint{n}" for n in still)
        subject, verb = ("it", "was") if len(still) == 1 else ("they", "were")
        print(f"[rec] {names} never moved while the others did. If {subject} "
              f"{verb} released\n      and turned by hand, the controller is "
              "not reporting that joint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
