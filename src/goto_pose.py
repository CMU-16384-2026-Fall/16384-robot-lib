"""Send the arm to a fixed joint pose — by default the planar 3R, aimed forward.

    python goto_pose.py --check            # say what would happen, move nothing
    python goto_pose.py                    # the move, in meshcat
    python goto_pose.py --real             # the move, on the real arm
    python goto_pose.py --real --guided               # capture on each enter
    python goto_pose.py --real --guided --capture 1  # capture once a second

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

    1. drive to the planar pose;
    2. take the brakes off joints 1, 4 and 7 so the arm can be swung by hand
       about z, while 2, 3, 5 and 6 stay powered and hold the plane, capturing
       waypoints as you go;
    3. re-lock those three and drive back to the pose phase 1 reached.

Phase 2 is UFACTORY's own per-joint unlock — the lock icons beside the joint
sliders in Studio — through `set_servo_detach(servo_id=n)`. The arm never
leaves position mode and nothing emulates a stiffness: joints 2, 3, 5 and 6
hold with their full servo torque because nothing has asked them not to.

The catch, measured rather than assumed, is that **a released joint is not
reported at all**. `probe_feedback.py` turned one 96 degrees by hand and watched
it read as perfectly still the whole way — through the report stream and a live
query alike — until the servo re-armed, whereupon the reading jumped straight
to the truth. So there is no continuous trace to record: the angles are only
knowable while the joints are held. `--capture` works with that rather than
against it, locking the three joints for a fraction of a second to read them
and letting go again, either when you press enter or on a fixed interval.

Doing it the other way round — leaving the servos on and making three joints
compliant in software — was tried and does not work. There is no joint-torque
interface to use (`set_servot` is gone from the SDK), so the only lever is
velocity commanded into the very servo being compensated, sampled over TCP at a
fraction of the controller's own rate. The arm pushes back and then latches an
error. Teach mode manages it because the compensation runs inside the servo
loop, on identified joint friction (`iden_joint_friction`), the payload
(`set_tcp_load`) and the mounting (`set_gravity_direction`) — none of which is
reachable per joint.

Releasing 1, 4 and 7 is only safe *in this pose*. Their axes are vertical here,
and gravity exerts no moment about a vertical axis, so a released joint swings
instead of falling. Joint 1 is vertical always; 4 and 7 are vertical only while
2, 3, 5 and 6 sit at +-90, so phase 2 checks that before releasing anything.

Waypoints are written as seconds and radians, matching `record_joints.py`:

    t,joint1,joint4,joint7

So this script checks before it commands, and prints what it found either way.
`--force` turns the library's guard off for whoever is sure the model is wrong
about their arm; it does not turn off the controller's own detection, which
will abort the move on its own if it agrees with the model.

The pose is passed and printed in degrees, because that is how the locked set
is written; everything below the CLI is radians, like the rest of the library.
"""

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# joints 1..7, degrees. Joints 2, 3, 5 and 6 are the "90 90 -90 90" that
# makes the arm planar; 1, 4 and 7 are the free ones, here aiming it down +x.
DEFAULT_TARGET_DEG = (-52.0, 90.0, 90.0, 90.0, -90.0, 90.0, 38.0)

# The joints the guided hold releases, as the controller numbers them (1-based),
# and the ones it leaves powered, as this library indexes them (0-based).
FREE_SERVOS = (1, 4, 7)
FREE_INDICES = tuple(n - 1 for n in FREE_SERVOS)
LOCKED_INDICES = (1, 2, 4, 5)

DEFAULT_SPEED = 0.3  # rad/s, the library's own default
RETURN_SPEED_CAP = 0.2  # rad/s; phase 3 moves with people close to the arm
_SETTLE_GRACE = 15.0  # s added to a move's travel time before the wait gives up

# How far a locked joint may sit from +-90 and still leave the free three
# vertical enough to release. 3 degrees tilts a 426 mm forearm by 22 mm.
_PLANAR_TOLERANCE = math.radians(3.0)
# Movement across the re-attach worth naming a joint over.
_SNAP_TOLERANCE = math.radians(2.0)
# How long to let the arm settle after re-locking before reading its pose.
_RELOCK_SETTLE = 5.0  # s
_ATTACH_SETTLE = 0.3  # s to let each joint take hold before reading it back

# Capturing a waypoint means locking the joints, reading them, and letting go
# again. The reading only catches up once the servos are back on, so the wait
# is until it stops changing rather than a fixed sleep.
_CAPTURE_TIMEOUT = 1.0  # s to give the reading to settle before taking it anyway
_CAPTURE_POLL = 0.02  # s between reads while waiting for it
_CAPTURE_STABLE = math.radians(0.05)  # two reads this close count as settled
_CAPTURE_MIN = 0.1  # s, the fastest automatic capture allowed
_CAPTURE_MAX = 5.0  # s, the slowest


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
        help="after arriving, make joints 1, 4 and 7 compliant so the arm can be "
        "swung by hand about z — 2, 3, 5 and 6 stay rigid and hold the plane — "
        "then stiffen up again and return to the zero pose",
    )
    parser.add_argument(
        "--capture", default="manual",
        help="when to record a waypoint during the hold: \"manual\" to capture "
        f"each time you press enter, or an interval in seconds between "
        f"{_CAPTURE_MIN:g} and {_CAPTURE_MAX:g} to capture automatically "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        help="where to write the captured waypoints "
        "(default: recordings/waypoints-<timestamp>.csv)",
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
    args.capture = capture_from(args.capture)
    return args


def capture_from(text):
    """"manual", or an interval in seconds.

    The range is a real limit, not a formality: every automatic capture locks
    and releases three brakes, so a short interval means cycling them
    thousands of times in a session.
    """
    if text.strip().lower() in ("manual", "enter"):
        return "manual"
    try:
        interval = float(text)
    except ValueError:
        raise SystemExit(f"--capture wants \"manual\" or a number, got {text!r}")
    if not _CAPTURE_MIN <= interval <= _CAPTURE_MAX:
        raise SystemExit(
            f"--capture interval must be between {_CAPTURE_MIN:g} and "
            f"{_CAPTURE_MAX:g} seconds, got {interval:g}"
        )
    return interval


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


def default_capture_path():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("recordings") / f"waypoints-{stamp}.csv"


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


def describe_catch_up(q_locked, q_frozen):
    """Say how far the arm was actually turned while its brakes were off.

    This looks like a measurement of the re-attach and isn't. The controller
    does not report a detached joint — measured directly with
    `probe_feedback.py`: a joint turned 96 deg by hand read as perfectly still
    the whole time, through both the report cache and a live `get_servo_angle`,
    and only jumped to its true angle once the servo re-armed. So `q_frozen` is
    not where the arm was a moment ago, it is where it was when the brakes came
    off, and the difference is the hand motion catching up in the readout.

    Which makes this the only measurement of that motion there is, and worth
    printing for it — but it is not evidence the arm moved as it re-locked. It
    did not: if re-arming had driven the joints back to their old setpoint, the
    reading afterwards would match `q_frozen` instead of differing from it.
    """
    delta = q_locked - q_frozen
    movers = np.flatnonzero(np.abs(delta) > _SNAP_TOLERANCE)
    if not movers.size:
        print("[goto] the arm came back to the pose it was released in")
        return

    named = ", ".join(
        f"joint{i + 1} {math.degrees(delta[i]):+.2f}" for i in movers
    )
    print(f"[goto] turned by hand: {named} deg")
    print("[goto] (the readout catches up here — a released joint's angle is "
          "not reported\n       while its brake is off, so this is the whole "
          "hand motion at once.)")


def relock(arm, servos, q_hand, before):
    """Put the brakes back on, and say what the arm did as they took hold.

    Re-arming does not drag the arm back to the pose phase 1 commanded, which
    was the worry: `probe_feedback.py` turned a released joint 96 deg by hand,
    and after re-locking it read 96 deg from where it started rather than
    snapping home. The controller takes the new position as the truth. The
    motion queue is still dropped first, below, because `set_servo_attach` ends
    with `set_state(0)` and a live trajectory sitting in that queue would be
    another matter entirely.
    """
    # Drop whatever is still in the controller's motion queue before re-arming
    # the servos. `set_servo_attach` ends with `set_state(0)`, which puts the
    # controller back into motion state — and if phase 1's trajectory is still
    # live there, that is the cue for it to resume and drive the arm back to
    # the pose it was commanded to, out from under the student's hands.
    print("[goto] stand clear — re-locking, and the arm may move.")
    try:
        arm.stop(wait=True, timeout=_RELOCK_SETTLE)
    except Exception as err:  # never let tidying up be what breaks the exit
        print(f"[goto] couldn't clear the motion queue ({err}); re-locking anyway.")

    ok = True
    for servo in servos:
        code = arm.arm.set_servo_attach(servo_id=servo)
        if code != 0:
            ok = False
            print(f"[goto] set_servo_attach(servo_id={servo}) failed with code "
                  f"{code} — joint{servo} may still be free.")
        else:
            print(f"[goto] joint{servo} re-locked")
        time.sleep(_ATTACH_SETTLE)

    # Whatever the re-enable set off, let it finish before reading a position:
    # a pose sampled mid-swing describes nowhere the arm actually is.
    if not arm.wait_for_motion(timeout=_RELOCK_SETTLE):
        print(f"[goto] the arm was still moving {_RELOCK_SETTLE:.0f}s after "
              "re-locking; hands off it?")

    describe_catch_up(arm.joint_values, q_hand)

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

    return ok


def settled_reading(arm):
    """The arm's pose, once the readout has caught up with it.

    Called straight after the brakes go back on, when the reported angles are
    still the ones the joints held when they were released. Polling until two
    reads agree gets the true pose as soon as the report delivers it, instead
    of guessing at a sleep long enough to cover it.
    """
    deadline = time.perf_counter() + _CAPTURE_TIMEOUT
    previous = arm.joint_values
    while time.perf_counter() < deadline:
        time.sleep(_CAPTURE_POLL)
        current = arm.joint_values
        if np.max(np.abs(current - previous)) <= _CAPTURE_STABLE:
            return current
        previous = current
    return previous  # took too long to settle; the last read is the best we have


def capture_pose(arm, real, servos):
    """One waypoint: lock the free joints, read them, let them go again.

    The locking is not ceremony. A joint with its brake off is not reported at
    all, so the only moment its angle can be known is while it is held — which
    is why this is a capture rather than a sample of something continuous.
    """
    if not real:
        return arm.joint_values  # nothing is released in simulation

    for servo in servos:
        arm.arm.set_servo_attach(servo_id=servo)
    pose = settled_reading(arm)
    for servo in servos:
        arm.arm.set_servo_detach(servo_id=servo)
    return pose


def capture_loop(arm, real, servos, capture, writer, started):
    """Take waypoints until ctrl-c, writing each one as it is taken."""
    if capture == "manual":
        print("[goto] pose the arm, then press enter to capture it. "
              "ctrl-c when you're done.")
    else:
        print(f"[goto] capturing every {capture:g}s — pose the arm and hold "
              "still between\n       captures. ctrl-c when you're done.")

    taken = 0
    while True:
        if capture == "manual":
            try:
                input(f"[goto] enter to capture #{taken + 1} ")
            except EOFError:  # nobody left at the terminal to press it
                print()
                return
        else:
            time.sleep(capture)

        pose = capture_pose(arm, real, servos)
        elapsed = time.perf_counter() - started
        writer.writerow(
            [f"{elapsed:.4f}"] + [f"{pose[i]:.6f}" for i in FREE_INDICES]
        )
        taken += 1
        shown = "  ".join(
            f"joint{n}={math.degrees(pose[i]):7.2f}"
            for n, i in zip(FREE_SERVOS, FREE_INDICES)
        )
        print(f"[goto] #{taken:3d} at {elapsed:6.1f}s   {shown} deg")


def guided_hold(arm, real, capture, out):
    """Phase 2. Returns (ok, pose the arm is left in)."""
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
    print("[goto] a released joint isn't reported at all, so the angles freeze "
          "while you\n       move it. Each capture locks the three joints for "
          "a moment to read them\n       properly, then lets go again.")
    if capture != "manual":
        print(f"[goto] every capture cycles three brakes, and at {capture:g}s "
              "that adds up over a\n       session — use the slowest interval "
              "that suits you.")
    if not real:
        print("[goto] (simulated: MuJoCo has no joint brakes, so nothing is "
              "released and every\n       capture reads the same still pose)")

    if not wait_for_enter("[goto] stand clear, then press enter to release. "):
        return False, q

    out.parent.mkdir(parents=True, exist_ok=True)
    before = brake_states(arm) if real else (None, None)
    if real:
        show_states("before releasing", *before)

    released, failure = [], None
    taken = 0
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t"] + [f"joint{n}" for n in FREE_SERVOS])

        # Past this point joints are off their brakes, so every way out goes
        # through the re-attach — a failure part-way through the release
        # included, since the joints before it are already free.
        try:
            if real:
                released, failure = release_joints(arm, FREE_SERVOS)
                if failure is None:
                    show_states("after releasing", *brake_states(arm))
            if failure is None:
                started = time.perf_counter()
                capture_loop(arm, real, FREE_SERVOS, capture, writer, started)
        except KeyboardInterrupt:
            print()
        finally:
            q_hand = arm.joint_values
            print(f"[goto] left at {degrees(q_hand)} deg")
            ok = relock(arm, released, q_hand, before) if released else True

    print(f"[goto] waypoints written to {out}")
    if failure is not None:
        print(f"[goto] {failure}")
        return False, q_hand
    return ok, q_hand


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
        out = Path(args.out) if args.out else default_capture_path()
        ok, q_hand = guided_hold(arm, args.real, args.capture, out)
        if not ok:
            print("[goto] a joint could not be released or re-locked, so the "
                  "arm is not in a\n       state to be driven. Leaving it as "
                  "it is; check it before re-running.")
            return 1

        # ---- phase 3 -------------------------------------------------
        # Back to the planar pose, not to all-zeros: this is the configuration
        # the exercise is about, the one that leaves joints 1, 4 and 7 turning
        # about z, and parking here leaves the arm ready for the next run.
        # Only the three free joints have to travel — the other four never left.
        print(f"[goto] returning to {degrees(goal)} deg")
        if not preflight(arm.joint_values, goal, box):
            print("[goto] the way back isn't clear from where the arm was "
                  "left.\n       Leaving it as it is — move it clear by hand "
                  "and re-run.")
            return 2

        speed = min(args.speed, RETURN_SPEED_CAP)
        if not args.yes and not wait_for_enter(
            f"[goto] hands clear — press enter to drive back at {speed} rad/s. "
        ):
            return 1

        reached, code = move_to(arm, goal, speed, controller_errors)
        if code is not None:
            return code
        return 0 if reached else 1


if __name__ == "__main__":
    sys.exit(main())
