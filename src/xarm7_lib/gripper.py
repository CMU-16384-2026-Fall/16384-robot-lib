"""The UFACTORY xArm Gripper, shared by both backends.

The gripper is its own device on the end of the arm. The controller drives it
over a separate modbus link, and nothing about it goes through the arm's motion
modes — so a grip is not an arm command: it doesn't wait for whatever the arm
is doing, doesn't cancel it, and doesn't stop the arm from being commanded
while the jaws are still closing. `grip` is the one command in this library
that leaves the arm alone entirely.

Openings are a fraction of the jaws' travel, `GRIPPER_CLOSED` (0.0, shut) to
`GRIPPER_OPEN` (1.0, wide), and speeds a fraction of the gripper's full speed.
The hardware counts that travel in pulses and that speed in pulses per second;
the simulator drives a tendon whose units are neither. A fraction is the one
thing that means the same in both places, and it leaves each backend to say
what a fraction is in its own terms — which is what makes a script holding a
part in simulation the same script that holds it for real.

The safety guard in `safety.py` carries the gripper: it is in the guard's model
as one convex envelope on the flange, covering the jaws at every opening and the
space between them, so the safety box and the self-collision check both account
for the jaws and for a part held in them. What the guard does *not* check is the
grip itself — opening or closing the jaws moves nothing the guard models, so a
`grip` is never refused, and a part wider than the envelope is a part the guard
cannot see.
"""

GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 1.0

# The gripper's own units: the controller positions the jaws in pulses across
# this travel, 0 shut, and takes a speed in pulses per second.
RAW_TRAVEL = 850
RAW_SPEED_MIN = 1
RAW_SPEED_MAX = 5000

# Roughly how far apart the fingers are at `GRIPPER_OPEN`, metres. Nothing in
# the library uses it — the gripper is in neither the guard's model nor the
# arm's kinematics — but sizing a part to the jaws needs the number, so it is
# written down here rather than in a comment in somebody's script.
GRIPPER_STROKE = 0.085

# Seconds for the jaws to cross their whole travel at full speed, out of the
# two numbers above. The controller paces the real gripper itself; this is what
# lets the simulator pace its jaws to the same clock.
FULL_TRAVEL_TIME = RAW_TRAVEL / RAW_SPEED_MAX

# Half speed: fast enough not to be a wait, slow enough that closing on
# something is a grip rather than a knock.
DEFAULT_GRIP_SPEED = 0.5


def check_opening(opening):
    """Return `opening` as a float, or raise if it isn't a jaw position.

    Out of range is refused rather than clamped, for the reason a joint target
    is: asking for jaws that open wider than they do is a bug in the caller,
    and quietly moving somewhere else would hide it.
    """
    value = float(opening)
    if not GRIPPER_CLOSED <= value <= GRIPPER_OPEN:
        raise ValueError(
            f"gripper opening must be between {GRIPPER_CLOSED} (shut) and "
            f"{GRIPPER_OPEN} (wide open), got {opening!r}"
        )
    return value


def check_speed(speed):
    """Return `speed` as a fraction of full speed, or raise. None is the default.

    Zero is refused rather than read as "don't move": a grip that cannot finish
    is a command with no end, and `wait=True` on it would never return.
    """
    value = DEFAULT_GRIP_SPEED if speed is None else float(speed)
    if not 0.0 < value <= 1.0:
        raise ValueError(
            f"gripper speed must be a fraction of full speed in (0, 1], "
            f"got {speed!r}"
        )
    return value


def raw_position(opening):
    """`opening` as the controller's own pulse count."""
    return int(round(check_opening(opening) * RAW_TRAVEL))


def raw_speed(speed):
    """`speed` as the controller's own pulses per second."""
    pulses = round(check_speed(speed) * RAW_SPEED_MAX)
    return int(min(max(pulses, RAW_SPEED_MIN), RAW_SPEED_MAX))
