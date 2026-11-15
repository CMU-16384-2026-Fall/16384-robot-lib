"""Set the gripper's opening from a terminal.

    python -m xarm7_lib.grip 1.0     # wide open
    python -m xarm7_lib.grip 0       # shut
    python -m xarm7_lib.grip 0.4 --speed 0.2

The opening is the same fraction `Robot.grip` takes, 0.0 shut to 1.0 wide open.
It talks to the real arm at `$ROBOT_IP`, or the address given after the
opening, and moves nothing but the jaws.

The arm is connected with the safety guard off. A grip is never checked against
the guard anyway (see `gripper.py`), and leaving it off skips loading the
collision meshes, so this connects in about a second and works on an arm parked
outside the safety box.
"""

import argparse
import os

from .gripper import check_opening, check_speed


def _fraction(check):
    """An argparse type that runs one of `gripper.py`'s range checks."""

    def parse(text):
        try:
            return check(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from None

    return parse


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m xarm7_lib.grip",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "opening",
        type=_fraction(check_opening),
        help="0.0 shut to 1.0 wide open",
    )
    parser.add_argument(
        "ip",
        nargs="?",
        default=os.environ.get("ROBOT_IP", "").strip(),
        help="controller address, e.g. 192.168.1.185. Defaults to $ROBOT_IP.",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=_fraction(check_speed),
        default=None,
        help="fraction of full speed in (0, 1]; defaults to half",
    )
    args = parser.parse_args(argv)
    if not args.ip:
        parser.error("no controller address: pass one, or set ROBOT_IP")

    # Imported here so `--help` and a bad opening answer without the SDK.
    from .xarm7_real import RealXArm7

    with RealXArm7(args.ip, guard=False) as arm:
        stopped = arm.grip(args.opening, speed=args.speed)
        reached = arm.gripper_opening

    if not stopped:
        print(f"jaws still moving after the timeout, at {reached:.2f}")
        return 1
    # A shut gripper reads a few hundredths above 0.0, so "short" needs slack.
    if abs(reached - args.opening) > 0.05:
        print(f'stopped at {reached:.2f} (asked for {args.opening:.2f})')
    else:
        print(f"gripper at {reached:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
