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
