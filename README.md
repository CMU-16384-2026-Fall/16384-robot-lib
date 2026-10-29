# xarm7_lib

Simulated and real UFACTORY xArm7 control behind one interface.

# Install instructions
```
conda env create -f environment.yml
```

Or, into an existing environment:
```
conda activate 16384
conda env update -f environment.yml
```

# Testing
```python
from xarm7_lib import RealXArm7
robot = RealXArm7(ip='192.168.1.?')
robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0])
```

`Robot` picks the backend for you — the real arm when the `ROBOT_IP`
environment variable is set, the MuJoCo simulation otherwise:
```python
from xarm7_lib import Robot
robot = Robot()
robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0])
```

`with` releases the backend when you are done — the connection for the real
arm, the simulation thread and viewer for the sim — including when the block
is interrupted or raises:
```python
with Robot() as robot:
    robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0])
```
The real arm is left stopped with its motors still holding: switching them off
would let it fall under its own weight.

# Gripper

There is a UFACTORY xArm Gripper on the flange, driven with `grip`. Openings
are a fraction of the jaws' travel — `0.0` shut, `1.0` wide open (about 85 mm
between the fingers) — and speeds a fraction of the gripper's full speed:

```python
from xarm7_lib import Robot

with Robot() as robot:
    robot.grip(1.0)             # jaws wide open
    robot.grip(0.0, speed=0.2)  # close slowly onto a part
    robot.gripper_opening       # 0.31, say: as far as the part let them shut
```

`grip` blocks until the jaws stop moving and returns True when they have.
Stopping is also what a successful grasp looks like — the jaws stall on the
object instead of reaching the target — so `gripper_opening` is what tells the
two apart: short of what you asked for means something is in there.

The gripper is a separate device on a link of its own, so a grip doesn't touch
the arm: it neither waits for the arm's current move nor cancels it, and the
arm can be commanded while the jaws are still closing. `grip(..., wait=False)`
returns immediately if you want both at once.

The gripper itself is only talked to on the first grip, not on connect, so an
arm with nothing bolted on still works; pass `gripper=False` to say so
properly, which makes `grip` raise and takes the gripper back out of the safety
guard as well.

In simulation the jaws are the MJCF's own, and they can hold things. They are
soft and heavily damped, though, so they settle onto the setpoint rather than
snapping to it — a shut gripper reads a few hundredths above `0.0` rather than
exactly `0.0`. Compare `gripper_opening` with a tolerance, not with `==`.

## The gripper and the safety guard

The guard in `safety.py` carries the gripper. It is in the guard's model as one
convex envelope on the flange — the hull of the gripper over the jaws' whole
travel — so the safety box and the self-collision check both stop the
fingertips rather than `link7`, whatever the jaws happen to be doing. The hull
closes over the gap between the jaws as well, which is where a held part sits:
a part that fits in the jaws can't be taken anywhere the jaws couldn't go.

The envelope is 175 mm of arm that wasn't being checked before, so **poses that
used to pass can now be refused**. With the default box in particular:

- the all-zeros pose puts the jaws 54 mm below `zmin`,
- `HOME_POSE` puts them 5 mm below it, which `free_drive` refuses to start from.

Both are the box's `z = 0` floor being drawn at the arm's own mounting surface.
Re-measure the box for where the arms actually stand and they go away;
`SafetyGuard(box=...)` and the backends' `safety_box=` are where it is set.

What the guard still doesn't check:

- **the grip itself** — opening or closing the jaws moves nothing the guard
  models, so `grip` is never refused,
- **a part wider than the jaws** — the envelope covers the jaws and the space
  between them, and nothing that sticks out past it.

`gripper=False` on either backend goes back to the bare flange: `grip` stops
working and the guard stops at `link7`, as it did before there was a gripper.

# Free drive

Push the arm around by hand with only some of the joints meant to move, and
record where it went. Real arm only — there is nothing to push in simulation.

Put the arm at `HOME_POSE` first. The four locked joints are posed there so that
gravity has no moment about joint1, joint4 or joint7 — those three carry
nothing, wherever you drive them.

```python
from xarm7_lib import Robot, HOME_POSE
robot = Robot()
robot.set_joint_targets(HOME_POSE)

# Frees joint1, joint4 and joint7; watches the other four.
traj = robot.free_drive(duration=30)

traj.q     # (N, 7) joint positions, rad
traj.qd    # (N, 7) joint velocities, rad/s
traj.t     # (N,)   seconds from the start
traj.save("run.npz")
```

## Pushing the arm around from a terminal

```
python -m xarm7_lib.free_drive
```

Releases every joint, waits for you to push the arm where you want it, and
locks it there when you press Enter. It reads `$ROBOT_IP`, or takes an address
as its argument.

This is not `free_drive` above: nothing is watched, nothing is recorded, and
the safety guard is off. That last part is the point of it — the guard refuses
to start a run from a pose it would not allow, so if the arm is parked out of
the safety box this is what gets it back in. It is also the quickest check that
your machine can talk to the controller at all: it connects, prints the joint
angles, and doesn't move the arm anywhere on its own.

## What "locked" means here

The arm runs in the controller's joint teaching mode, which releases **all
seven** joints and holds the arm up against gravity. There is no way to release
only three, so the other four are not held — they are watched. Their positions
are latched before the run starts, and if one drifts more than `tolerance`
(0.05 rad, about 3°) the run stops to put it back:

1. the arm stiffens where it stands,
2. you are asked to take your hands off it, and it waits for you to say you have,
3. it moves slowly to put the watched joints back, leaving the free ones alone,
4. teaching resumes.

So a locked joint is one the arm *returns to*, not one it refuses to leave. You
will feel it move if you push it, and if you keep pushing you will be
interrupted. Ease off when the warning appears and nothing stops.

Each interruption leaves a gap in `traj.t` — the samples either side of it are
seconds apart and the arm moved on its own in between. `traj.interruptions`
counts them, and `np.diff(traj.t)` finds them. Time spent stopped doesn't count
against `duration`.

Ctrl-C ends a run early and still returns everything recorded up to that point;
`traj.reason` says why it ended. However a run ends, the arm is put back into
position control — it stays exactly where it was left, but it now resists being
pushed instead of yielding, and the locked joints are held rather than watched.

Nothing can be refused while a person is pushing the arm, so the safety guard
only warns: from the home pose it is happy with about ±88° on joint1, 99° on
joint4 and all the way round on joint7, and past that you get a message rather
than a wall. If the arm is hard to move, `teach_sensitivity=` (1-5) is the
controller's own knob for that.
