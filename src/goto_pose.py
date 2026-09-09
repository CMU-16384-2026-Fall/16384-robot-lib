"""Send the arm to a fixed joint pose — by default the planar 3R, aimed forward.

    python goto_pose.py --check            # say what would happen, move nothing
    python goto_pose.py                    # the move, in meshcat
    python goto_pose.py --real             # the move, on the real arm
    python goto_pose.py --real --guided    # move, hand it to a student, park it

Simulation is the default; hardware needs `--real`.

The default target holds joints 2, 3, 5 and 6 at +90, +90, -90 and +90 degrees.
That is the "90 90 -90 90" locked set, and it is what leaves joints 1, 4 and 7
with their axes all parallel to world z, so what is left is an exact planar 3R
turning in a horizontal plane: shoulder at the base, elbow 298 mm out, wrist
426 mm past that.

Those three free joints are what aims it, and they are free — moving them
cannot break the planarity, only point it somewhere else. At -52, +90 and +38
degrees the chain reaches out along +x and stops with the flange on the robot's
centre line, 599 mm in front of the base at z = 290 mm; the 100 mm tool the
exercise draws off the flange then ends at x = 699 mm, level with and pointed
at whoever is standing in front of the arm. The elbow swings 199 mm to the -y
side to get there, which is what keeps the forearm clear of the shoulder, and
the straight joint-space line from the zero pose is clear the whole way.

Leaving the three free joints at zero instead — the bare "90 90 -90 90", every
other joint at 0 — does not work, and not because the guard is fussy. joint4 = 0
is the folded end of the elbow's travel (its range is -11 to +225 degrees, not
the +-360 the other free joints get), so the forearm already doubles back under
the upper arm at zero, and swinging joint2 to +90 then carries it down through
the base: link5, link6 and link7 finish behind the shoulder, at x = -49 to
-146 mm, with link2 and link6 overlapping by 34 mm. UFACTORY's own collision
meshes and the Menagerie MJCF agree on those depths to a tenth of a millimetre,
so it is the geometry, not one model's opinion. Run

    python goto_pose.py --check --joints 0,90,90,0,-90,90,0

to see it refused, and drop --check to watch the simulator jam there with
joint4 pushed 10 degrees off its setpoint by the links it is resting on.

`--guided` adds two more phases to the run:

    1. drive to the target pose;
    2. release joints 1, 4 and 7 so the arm can be swung by hand about z, with
       2, 3, 5 and 6 left powered so it cannot be pushed out of its plane;
    3. re-lock those three and drive back to the zero pose.

Phase 2 is UFACTORY's own per-joint unlock — the lock icons beside the joint
sliders in Studio — reached through `set_servo_detach(servo_id=n)`. The arm
never leaves position mode and no stiffness is being emulated: joints 2, 3, 5
and 6 hold with their full servo torque, because nothing has asked them not to.

Releasing 1, 4 and 7 is only safe *in this pose*. Their axes are vertical here,
and gravity exerts no torque about a vertical axis, so a released joint swings
instead of falling. Joint 1 is vertical always; 4 and 7 are vertical only while
2, 3, 5 and 6 sit at +-90, so phase 2 checks that before it releases anything.

So this script checks before it commands, and prints what it found either way.
`--force` turns the library's guard off for whoever is sure the model is wrong
about their arm; it does not turn off the controller's own detection, which
will abort the move on its own if it agrees with the model.

The pose is passed and printed in degrees, because that is how the locked set
is written; everything below the CLI is radians, like the rest of the library.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# joints 1..7, degrees. Joints 2, 3, 5 and 6 are the "90 90 -90 90" that
# makes the arm planar; 1, 4 and 7 are the free ones, here aiming it down +x.
DEFAULT_TARGET_DEG = (-52.0, 90.0, 90.0, 90.0, -90.0, 90.0, 38.0)

# Where phase 3 parks the arm: the folded, upright rest pose.
HOME = np.zeros(7)

# The joints the guided hold releases, as the controller numbers them (1-based),
# and the ones it leaves powered, as this library indexes them (0-based).
FREE_SERVOS = (1, 4, 7)
LOCKED_INDICES = (1, 2, 4, 5)

DEFAULT_SPEED = 0.3  # rad/s, the library's own default
RETURN_SPEED_CAP = 0.2  # rad/s; phase 3 moves further, with people closer
_SETTLE_GRACE = 15.0  # s added to a move's travel time before the wait gives up

# How far a locked joint may sit from +-90 and still leave the free three
# vertical enough to release. 3 degrees tilts a 426 mm forearm by 22 mm.
_PLANAR_TOLERANCE = math.radians(3.0)
# Movement across the re-attach that counts as the servo snapping back.
_SNAP_TOLERANCE = math.radians(2.0)
_MONITOR_PERIOD = 0.1  # s between readings while the arm is hand-guided
_LOGGED_STATUS_PERIOD = 1.0  # s between status lines when stdout isn't a terminal
_ATTACH_SETTLE = 0.3  # s to let each joint take hold before reading it back


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--real", action="store_true",
        help="drive the real arm. Without this the run goes to the MuJoCo "
        "simulation in meshcat.",
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="drive the simulation (the default; say it explicitly if you like)",
    )
    parser.add_argument(
        "--ip",
        help="controller address. Defaults to the contents of ip.txt, in the "
        "working directory or next to this script.",
    )
    parser.add_argument(
        "--joints",
        default=",".join(f"{v:g}" for v in DEFAULT_TARGET_DEG),
        help="target as 7 comma-separated joint angles in degrees "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--speed", type=float, default=DEFAULT_SPEED,
        help="joint speed for the move, rad/s (default: %(default)s)",
    )
    parser.add_argument(
        "--guided", action="store_true",
        help="after arriving, release joints 1, 4 and 7 so the arm can be swung "
        "by hand about z — 2, 3, 5 and 6 stay powered and hold the plane — then "
        "re-lock them and return to the zero pose",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="report whether the pose and the path to it are allowed, then stop",
    )
    parser.add_argument(
        "--no-box", action="store_true",
        help="switch off the workspace box; self-collision is still checked",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="command the pose even if the check refuses it. This disables the "
        "library's guard only — the controller keeps its own self-collision "
        "detection, and will abort the move if it agrees with the model.",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="don't ask before moving",
    )
    args = parser.parse_args(argv)
    if args.real and args.sim:
        raise SystemExit("--real and --sim are opposites; pass one or neither.")
    return args


def target_from(text):
    """The 7-vector, in radians, that `--joints` names."""
    values = [float(v) for v in text.replace(" ", "").split(",") if v]
    if len(values) != 7:
        raise SystemExit(f"--joints wants 7 angles in degrees, got {len(values)}")
    return np.radians(values)


def controller_ip(given, prefix="goto"):
    """The address to connect to: what was asked for, or what ip.txt says.

    `prefix` only tags the one line this prints, so `record_joints.py` can
    borrow this without announcing itself as goto_pose.
    """
    if given:
        return given
    for path in (Path.cwd() / "ip.txt", Path(__file__).resolve().parent / "ip.txt"):
        if path.exists():
            ip = path.read_text().strip()
            if ip:
                print(f"[{prefix}] using {ip} from {path}")
                return ip
    raise SystemExit("no controller address: pass --ip or put one in ip.txt.")


def degrees(q):
    return "[" + ", ".join(f"{math.degrees(v):7.2f}" for v in q) + "]"


def preflight(start, goal, box):
    """Report on the pose and on the straight line to it. True if both are clear.

    Its own `SafetyGuard` rather than the arm's, so the verdict is the same
    under `--force`, where the arm no longer has one — a check that goes quiet
    exactly when it is being overridden is worth nothing.
    """
    from xarm7_lib.safety import DEFAULT_MARGIN, SafetyGuard

    print("[goto] loading the collision model...")
    guard = SafetyGuard(box=box, margin=DEFAULT_MARGIN)

    pose = guard.check(goal)
    print(f"[goto] target pose: {'allowed' if pose is None else pose}")

    path, reached = guard.check_path(start, goal)
    if path is None:
        print("[goto] path from here: clear the whole way")
    else:
        # `reached` is the fraction of the line that is safe, so this is the
        # furthest the arm could legally get before something touches.
        stopped = start + reached * (goal - start)
        print(f"[goto] path from here: {path}")
        print(f"[goto]   clear for {reached * 100:.0f}% of the way, to {degrees(stopped)}")
    return pose is None and path is None


def connect(args, box):
    """The arm this run drives, and the errors its controller raises.

    Imported here rather than at the top so that importing this module costs
    nothing but the standard library and numpy — which is what lets
    `record_joints.py` borrow `controller_ip` and still run a hardware session
    without loading mujoco or pinocchio at all. (Importing `xarm7_lib` itself
    pulls in every backend either way; its `__init__` imports both.)

    The second half of the pair is what `except` should treat as "the
    controller refused it". The simulator has no controller, so its tuple is
    empty, and an empty tuple never matches.
    """
    guard = not args.force
    if args.real:
        from xarm7_lib import RealXArm7, XArmError

        arm = RealXArm7(controller_ip(args.ip), safety_box=box, guard=guard)
        return arm, (XArmError,)

    from xarm7_lib import SimulatedXArm7

    return SimulatedXArm7(visualize=True, safety_box=box, guard=guard), ()


def confirm(question):
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:  # not a terminal: treat silence as no
        return False


def wait_for_enter(message):
    """Hold until someone presses enter. False if nobody is at the terminal.

    The guided phases gate on this rather than on `confirm`, and they gate even
    under `-y`: it is a beat to get hands clear, not a question. Nobody at the
    terminal means nobody watching the arm, so the caller skips instead.
    """
    try:
        input(message)
        return True
    except EOFError:
        print("[goto] no terminal to pause at; skipping.")
        return False


def travel_timeout(start, goal, speed):
    span = float(np.max(np.abs(np.asarray(goal) - np.asarray(start))))
    return span / max(speed, 1e-6) + _SETTLE_GRACE


def move_to(arm, goal, speed, controller_errors):
    """Drive to `goal`. Returns (reached, exit code) — the code is None if the
    move was allowed to happen at all, whatever came of it."""
    from xarm7_lib.safety import SafetyError

    try:
        reached = arm.set_joint_targets(
            goal, speed=speed, wait=True,
            timeout=travel_timeout(arm.joint_values, goal, speed),
        )
    except SafetyError as err:
        print(f"[goto] refused by the guard: {err}")
        return False, 2
    except controller_errors as err:
        print(f"[goto] the controller refused it: {err}")
        return False, 3
    except KeyboardInterrupt:
        arm.stop()
        print("\n[goto] interrupted; the arm is stopped and holding.")
        return False, 130

    final = arm.joint_values
    residual = math.degrees(float(np.max(np.abs(final - goal))))
    print(f"[goto] {'arrived' if reached else 'did NOT arrive'}")
    print(f"[goto] now at  {degrees(final)} deg  (worst joint off by "
          f"{residual:.2f} deg)")
    return reached, None


# ----------------------------------------------------------------------
# Phase 2: the guided hold
# ----------------------------------------------------------------------


def out_of_plane(q):
    """The locked joints that are too far from +-90 to release the free three.

    At +-90 the axes of joints 4 and 7 are vertical, and gravity has no moment
    about a vertical axis — which is the whole reason those joints can have
    their brakes taken off without the arm dropping.
    """
    return [
        index for index in LOCKED_INDICES
        if abs(math.cos(q[index])) > math.sin(_PLANAR_TOLERANCE)
    ]


def brake_states(arm):
    """(brakes, enables) per joint, as the controller reports them."""
    brakes = arm.arm.motor_brake_states
    enables = arm.arm.motor_enable_states
    if brakes is None or enables is None:
        return None, None
    return list(brakes[:7]), list(enables[:7])


def show_states(label, brakes, enables):
    if brakes is None:
        print(f"[goto] {label}: the controller isn't reporting brake states")
        return
    print(f"[goto] {label}: brakes {brakes}  enables {enables}")


def release_joints(arm, servos):
    """Take the brakes off `servos`.

    Returns (released, failure): the joints that actually went free, and why it
    stopped if it did. The list matters more than the error — whatever was
    released has to be put back, including on the way out of a failure.
    """
    released = []
    for servo in servos:
        code = arm.arm.set_servo_detach(servo_id=servo)
        if code != 0:
            return released, (f"set_servo_detach(servo_id={servo}) failed with "
                              f"code {code}")
        released.append(servo)
        print(f"[goto] joint{servo} released")
    return released, None


def monitor(arm, real):
    """Print what the hand-guided arm is doing, until ctrl-c.

    Nothing is guarding the arm here — the library's guard only runs inside a
    motion command, and this phase issues none — so the safety check is a
    running commentary rather than a limit. It is worth having anyway: the arm
    can be walked into itself by hand, and the student should hear about it.
    """
    print("[goto] swing the arm by hand; ctrl-c here to re-lock it.")
    # On a terminal the status line rewrites itself in place; piped to a file it
    # has to be one line per sample, so it is slowed right down instead.
    live = sys.stdout.isatty()
    status_period = _MONITOR_PERIOD if live else _LOGGED_STATUS_PERIOD

    spoken = "nothing said yet"
    next_tick = time.perf_counter()
    next_status = next_tick
    while True:
        q = arm.joint_values
        line = "  ".join(
            f"joint{i + 1}={math.degrees(q[i]):7.2f}" for i in (0, 3, 6)
        )
        if real:
            flange = arm.arm.position
            if flange is not None and len(flange) >= 3:
                line += (f"   flange=({flange[0]:6.1f},{flange[1]:6.1f},"
                         f"{flange[2]:6.1f}) mm")

        violation = arm.check_safety(q)
        said = None if violation is None else str(violation)
        if said != spoken:
            if said is not None:
                print(f"\n[goto] warning: {said}")
            elif spoken != "nothing said yet":
                print("\n[goto] back inside the safe envelope")
            spoken = said

        now = time.perf_counter()
        if now >= next_status:
            next_status = now + status_period
            if live:
                print(f"\r[goto] {line}   ", end="", flush=True)
            else:
                print(f"[goto] {line}", flush=True)

        next_tick += _MONITOR_PERIOD
        remaining = next_tick - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:  # fell behind; don't try to catch up on a display loop
            next_tick = time.perf_counter()


def relock(arm, servos, q_hand, before):
    """Put the brakes back on. True if the arm didn't move doing it.

    The controller has been holding a setpoint for these joints since the move
    that got here, while the student turned the physical joints somewhere else.
    If re-enabling resumes that setpoint the arm snaps back through the arc in
    between, so the movement across the attach is measured, not assumed.

    Measured, and not asked about: there is no way to read the controller's
    commanded angles back and compare them with the real ones. Both spellings
    of `get_servo_angle` — `is_real=True` and `is_real=False` — end up at the
    same `GET_JOINT_POS` register (`uxbus_cmd.py:630-634`), so both return the
    measured position and their difference is always exactly zero. The arm
    moving, or not moving, across the re-attach is the only evidence there is.
    """
    for servo in servos:
        code = arm.arm.set_servo_attach(servo_id=servo)
        if code != 0:
            print(f"[goto] set_servo_attach(servo_id={servo}) failed with code "
                  f"{code} — joint{servo} may still be free.")
        else:
            print(f"[goto] joint{servo} re-locked")
        time.sleep(_ATTACH_SETTLE)

    moved = float(np.max(np.abs(arm.joint_values - q_hand)))
    print(f"[goto] moved {math.degrees(moved):.2f} deg across the re-attach")

    brakes, enables = brake_states(arm)
    show_states("after re-locking", brakes, enables)
    if before[0] is not None and brakes is not None and (
        brakes != before[0] or enables != before[1]
    ):
        print("[goto] warning: brake/enable states did not return to what they "
              "were before the release.")

    if arm.has_error:
        print("[goto] the controller latched an error while the arm was "
              "hand-guided; clearing it.")
        arm.clear_errors()

    return moved <= _SNAP_TOLERANCE


def guided_hold(arm, real):
    """Phase 2. Returns (ok, pose the student left the arm in)."""
    q = arm.joint_values
    offenders = out_of_plane(q)
    if offenders:
        names = ", ".join(f"joint{i + 1}" for i in offenders)
        verb = "is" if len(offenders) == 1 else "are"
        print(
            f"[goto] refusing to release anything: {names} {verb} not within "
            f"{math.degrees(_PLANAR_TOLERANCE):.0f} deg of +-90, so the axes of "
            "joints 4 and 7 are not vertical here.\n"
            "       Releasing them in this pose would let the arm fall rather "
            "than swing. Run without --joints to get the planar pose first."
        )
        return False, q

    print("[goto] about to release joint1, joint4 and joint7. Joints 2, 3, 5 "
          "and 6 stay\n       powered, so the arm can turn about z but cannot "
          "leave its plane.")
    if not real:
        print("[goto] (simulated: MuJoCo has no joint brakes, so nothing is "
              "actually released)")

    if not wait_for_enter("[goto] stand clear, then press enter to release. "):
        return False, q

    if not real:
        try:
            monitor(arm, real)
        except KeyboardInterrupt:
            print()  # the status line is unterminated
        q_hand = arm.joint_values
        print(f"[goto] left at {degrees(q_hand)} deg")
        return True, q_hand

    before = brake_states(arm)
    show_states("before releasing", *before)

    # Everything from here to the `finally` runs with joints off their brakes,
    # so every way out of it goes through the re-attach — a failure part-way
    # through the release included, since the joints before it are already free.
    released, failure = [], None
    try:
        released, failure = release_joints(arm, FREE_SERVOS)
        if failure is None:
            show_states("after releasing", *brake_states(arm))
            monitor(arm, real)
    except KeyboardInterrupt:
        print()  # the status line is unterminated
    finally:
        q_hand = arm.joint_values
        print(f"[goto] left at {degrees(q_hand)} deg")
        steady = relock(arm, released, q_hand, before) if released else True

    if failure is not None:
        print(f"[goto] {failure}")
        return False, q_hand
    return steady, q_hand


# ----------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    from xarm7_lib.safety import DEFAULT_BOX

    goal = target_from(args.joints)
    box = None if args.no_box else DEFAULT_BOX
    np.set_printoptions(precision=3, suppress=True)

    arm, controller_errors = connect(args, box)
    with arm:
        start = arm.joint_values
        print(f"[goto] now at  {degrees(start)} deg")
        print(f"[goto] going to {degrees(goal)} deg at {args.speed} rad/s")

        clear = preflight(start, goal, box)
        if args.check:
            return 0 if clear else 2
        if not clear and not args.force:
            print(
                "[goto] refusing to command a pose the collision model rejects.\n"
                "       Pass --force if you are sure the model is wrong about\n"
                "       your arm; the controller keeps its own detection either way."
            )
            return 2
        if not clear:
            print("[goto] --force given: the library's guard is off for this move.")
            if not args.yes and not confirm(
                "This may drive the arm into itself. Continue?"
            ):
                return 1

        if not args.yes and not confirm("Clear the workspace. Move now?"):
            return 1

        reached, code = move_to(arm, goal, args.speed, controller_errors)
        if code is not None:
            return code

        if not args.guided:
            if not args.real:
                # The viewer dies with the process, so hold it open to be
                # looked at.
                try:
                    input("[goto] meshcat is live; press enter to close. ")
                except EOFError:
                    pass
            return 0 if reached else 1

        if not reached:
            print("[goto] not releasing anything: the arm never reached the pose.")
            return 1

        # ---- phase 2 -------------------------------------------------
        ok, q_hand = guided_hold(arm, args.real)
        if not ok:
            print("[goto] stopping here rather than commanding a move. The arm "
                  "is where it\n       is; re-run this script to reset it.")
            return 1

        # ---- phase 3 -------------------------------------------------
        print(f"[goto] returning to {degrees(HOME)} deg")
        if not preflight(arm.joint_values, HOME, box):
            print("[goto] the way back to zero isn't clear from where the arm "
                  "was left.\n       Leaving it as it is — move it clear by "
                  "hand and re-run.")
            return 2

        speed = min(args.speed, RETURN_SPEED_CAP)
        if not args.yes and not wait_for_enter(
            f"[goto] hands clear — press enter to drive home at {speed} rad/s. "
        ):
            return 1

        reached, code = move_to(arm, HOME, speed, controller_errors)
        if code is not None:
            return code
        return 0 if reached else 1


if __name__ == "__main__":
    sys.exit(main())
