"""Does the controller still report a joint's angle while its brake is off?

    python probe_feedback.py --ip 192.168.1.185

Everything about recording a hand-guided arm depends on the answer, and it is
not in the SDK — it is firmware behaviour, so it has to be measured.

`goto_pose.py --guided` releases joints 1, 4 and 7 with `set_servo_detach`,
which takes the brake off by disabling that servo. If the controller's position
feedback comes from the servo loop, disabling it may stop the feedback too, and
then the joint's reported angle simply stops changing however far you turn it.
Two things we have already seen would both follow from that:

  - `record_joints.py` writes one repeated pose while the arm is being moved;
  - `relock` measures several degrees of "movement" across the re-attach, which
    would not be the arm snapping back at all, but the reading catching up the
    moment the servo re-arms.

This releases ONE joint — joint 1 by default, whose axis is vertical in every
pose, so it can only swing, never fall — reads its angle through both channels
for a few seconds while you turn it by hand, then re-locks and reads again.

    report cache   `api.angles`, whatever the report socket last delivered
    round trip     `get_servo_angle`, a live question to the controller

If neither moves while you are turning the joint, but the reading jumps once it
re-locks, the controller does not report a detached joint and no client-side
change can record one. If either moves, that is the channel to record from.
"""

import argparse
import math
import sys
import time

DEFAULT_SECONDS = 20.0
_TICK = 0.5  # s between readings
_MOVED = 0.5  # deg; more than the noise on a still joint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ip", help="controller address (default: ip.txt)")
    parser.add_argument(
        "--joint", type=int, default=1,
        help="which joint to release, 1-7 (default: %(default)s — the base, "
        "whose axis is vertical in every pose)",
    )
    parser.add_argument(
        "--seconds", type=float, default=DEFAULT_SECONDS,
        help="how long to watch it for (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.joint <= 7:
        raise SystemExit("--joint must be 1-7")
    return args


def main(argv=None):
    args = parse_args(argv)
    from goto_pose import controller_ip
    from xarm.wrapper import XArmAPI

    ip = controller_ip(args.ip, prefix="probe")
    api = XArmAPI(ip, is_radian=True)
    if not api.connected:
        api.connect()
    index = args.joint - 1

    def sample():
        """(report cache, round trip) for the joint under test, in degrees.

        The cache is read first on purpose: `get_servo_angle` writes it as a
        side effect, so asking the other way round would compare the round trip
        with itself.
        """
        cached = api.angles
        cached = math.degrees(cached[index]) if cached else float("nan")
        code, live = api.get_servo_angle(is_radian=True)
        live = math.degrees(live[index]) if code == 0 else float("nan")
        return cached, live

    print(f"[probe] firmware {api.version_number}, joint{args.joint}")
    if args.joint != 1:
        print(f"[probe] joint{args.joint} is not the base: make sure its axis "
              "is vertical, or\n        support the arm — releasing it under "
              "load will let the arm drop.")

    start = sample()
    print(f"[probe] before releasing: cache {start[0]:.2f}  live {start[1]:.2f} deg")
    try:
        if input("[probe] release it? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1
    except EOFError:
        return 1

    code = api.set_servo_detach(servo_id=args.joint)
    if code != 0:
        print(f"[probe] set_servo_detach failed with code {code}")
        return 3
    print(f"[probe] joint{args.joint} released — TURN IT BY HAND, as far as you "
          "can, for the\n        next few seconds.")

    lo = [float("inf"), float("inf")]
    hi = [-float("inf"), -float("inf")]
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            cached, live = sample()
            for i, v in enumerate((cached, live)):
                if v == v:  # not NaN
                    lo[i], hi[i] = min(lo[i], v), max(hi[i], v)
            print(f"\r[probe] cache {cached:8.2f}   live {live:8.2f} deg   ",
                  end="", flush=True)
            time.sleep(_TICK)
    except KeyboardInterrupt:
        pass
    print()

    print("[probe] re-locking; let go.")
    code = api.set_servo_attach(servo_id=args.joint)
    if code != 0:
        print(f"[probe] set_servo_attach failed with code {code} — joint"
              f"{args.joint} may still be free!")
    time.sleep(1.0)
    after = sample()
    print(f"[probe] after re-locking:  cache {after[0]:.2f}  live {after[1]:.2f} deg")

    print()
    for i, name in enumerate(("report cache", "round trip ")):
        span = hi[i] - lo[i] if hi[i] > -float("inf") else 0.0
        jump = abs(after[i] - start[i])
        moved = "moved" if span > _MOVED else "DID NOT move"
        print(f"[probe] {name}: {moved} while released "
              f"(span {span:.2f} deg), and reads {jump:.2f} deg "
              "from where it started, now it is locked")

    span_live = hi[1] - lo[1] if hi[1] > -float("inf") else 0.0
    if span_live <= _MOVED and abs(after[1] - start[1]) > _MOVED:
        print("\n[probe] verdict: the controller does NOT report a detached "
              "joint. The angle\n        only caught up once the servo "
              "re-armed, so a hand-guided motion\n        cannot be recorded "
              "while the brakes are off — and `relock`'s\n        "
              "\"movement across the re-attach\" is that catch-up, not the arm "
              "moving.")
    elif span_live > _MOVED:
        print("\n[probe] verdict: the round trip tracks a detached joint, so "
              "record from it.")
    else:
        print("\n[probe] verdict: nothing moved at all — was the joint actually "
              "turned?")
    api.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
