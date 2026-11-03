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

## Camera install

OpenCV is an optional extra, so a machine that only runs the simulator does not
need it:

```bash
pip install -e ".[camera]"
```

Then follow the instructions in `native/` to install zed-open-capture and build
the shim. Both are needed on a machine with a ZED attached; neither is needed
for the simulator.

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

Push the arm around by hand and record where it went. Real arm only — there is
nothing to push in simulation.

```python
from xarm7_lib import Robot
robot = Robot()

traj = robot.free_drive(30)   # 30 s of pushing, every joint free

traj.q     # (N, 7) joint positions, rad
traj.qd    # (N, 7) joint velocities, rad/s
traj.t     # (N,)   seconds from the start
traj.save("run.npz")
```

Every joint is free unless you say otherwise. The arm runs in the controller's
joint teaching mode, which releases all seven and holds the arm up against
gravity, so it goes wherever it is pushed and stays where it is left.

Ctrl-C ends a run early and still returns everything recorded up to that point;
`traj.reason` says why it ended. However a run ends, the arm is put back into
position control — it stays exactly where it was left, but it now resists being
pushed instead of yielding.

If the arm is hard to move, `teach_sensitivity=` (1-5) is the controller's own
knob for that.

## Running until something says stop

`duration=None` takes the time limit off. Then it is up to you to end the run —
Ctrl-C, or a condition:

```python
from xarm7_lib import stop_on_enter

traj = robot.free_drive(None, until=stop_on_enter())
```

`until=` is asked once a tick and ends the run as soon as it returns something
truthy. `stop_on_enter()` peeks at the terminal rather than reading it, so the
arm goes on being sampled and watched while it waits — which `input()` would
not — and so the run hands stdin back when it ends, instead of leaving a reader
to swallow the next thing you type at a `>>>` prompt.
`on_sample=lambda t, q: ...` is the same idea with the sample in hand, for
stopping when joint4 passes some angle or when a recording is long enough.

## Holding some of the joints

`watch=` names the joints that are meant to stay put, 0-indexed:

```python
from xarm7_lib import Robot, HOME_POSE, WATCHED_JOINTS, stop_on_enter
robot = Robot()
robot.set_joint_targets(HOME_POSE)

# Frees joint1, joint4 and joint7; watches the other four.
traj = robot.free_drive(None, watch=WATCHED_JOINTS, until=stop_on_enter())
```

`HOME_POSE` and `WATCHED_JOINTS` go together: the four watched joints are posed
there so that gravity has no moment about joint1, joint4 or joint7 — those
three carry nothing, wherever you drive them. Put the arm at `HOME_POSE` first.

### What "watched" means here

There is no way to release only three joints, so the other four are not held —
they are watched. Their positions are latched before the run starts, and if one
drifts more than `tolerance` (0.05 rad, about 3°) the run stops to put it back:

1. the arm stiffens where it stands,
2. you are asked to take your hands off it, and it waits for you to say you have,
3. it moves slowly to put the watched joints back, leaving the free ones alone,
4. teaching resumes.

So a watched joint is one the arm *returns to*, not one it refuses to leave. You
will feel it move if you push it, and if you keep pushing you will be
interrupted. Ease off when the warning appears and nothing stops.

Each interruption leaves a gap in `traj.t` — the samples either side of it are
seconds apart and the arm moved on its own in between. `traj.interruptions`
counts them, and `np.diff(traj.t)` finds them. Time spent stopped doesn't count
against `duration`.

## The safety guard during a run

Nothing can be *refused* while a person is pushing the arm — a hand is not a
command — so the guard is checked after each sample rather than before a move.
What it can do is stop the run: a pose it rejects ends the run there and then,
the arm stiffens, and after a hands-off prompt it steps back to the last
configuration the guard allowed. That last part matters beyond tidiness,
because a guarded command checks the path from wherever the arm *is*: an arm
left outside the box would refuse every move afterwards.

From `HOME_POSE` the guard is happy with about ±88° on joint1, 99° on joint4
and all the way round on joint7. `traj.reason` says what was hit.

## Pushing the arm around from a terminal

```
python -m xarm7_lib.free_drive
```

Releases every joint, waits for you to push the arm where you want it, and
locks it there when you press Enter. It reads `$ROBOT_IP`, or takes an address
as its argument.

This is not `free_drive` above: nothing is recorded and the safety guard is
off. That last part is the point of it — `free_drive` refuses to start a run
from a pose the guard would not allow, so if the arm is parked out of the
safety box this is what gets it back in. It is also the quickest check that
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

# Hand-eye calibration

The ZED is bolted down relative to the arm's base, so one transform carries
whatever the camera measures into robot coordinates. `StereoCalibration` stops
short of it: the factory calibration describes the camera's own geometry, so
triangulating a point gives metres in the *camera's* frame, and nothing about
the camera knows where the camera is.

That transform is measured once per bench by a TA and filed inside this
installation. Students read it; they do not run it.

## Using it

```python
from xarm7_lib import ZedOpenCapture, load_handeye

handeye = load_handeye()

with ZedOpenCapture() as camera:
    left, right = camera.read()
    # ... find the same feature in both eyes, then:
    camera_points = camera.calibration.triangulate(left_points, right_points)
    base_points = handeye.points_to_base(camera_points)
```

`base_points` is metres in the robot's base frame — the same frame
`set_joint_targets` and the safety box live in, so it is somewhere the arm can
be asked to go.

There is no path to pass and nothing to configure: `load_handeye()` reads the
attached camera's serial and finds that camera's calibration. The other
direction and the camera's own pose are there too:

```python
handeye.camera_position        # where the camera is, in base coordinates
handeye.points_to_camera(pts)  # base frame -> camera frame
handeye.project(pts)           # base frame -> rectified pixels, for drawing
```

If it raises `HandEyeError`, the station has not been calibrated — that is a
TA's job, not something to work around.

## Calibrating a bench (TAs)

A ChArUco board goes on the gripper and the arm takes it through a spread of
poses while the camera watches. Both unknowns come out together: where the
camera is, and where the board ended up on the flange.

**1. Print the board.**

```bash
python -m xarm7_lib.calibrate_handeye board
```

That writes `charuco_board.pdf`: the board at exact scale, centred on a sheet
of the smallest standard paper it fits. `--square-length MM` resizes it, and
the markers follow at three quarters of the square unless `--marker-length`
overrides them. Pass `-o board.png` for an image instead — it carries its own
print size now, but a print dialog has more ways to rescale an image than a
page.

Print at 100% ("actual size") — no "fit to page" — then **caliper a square**.
The board has its own spec printed along the bottom (`5x7  square 10.0 mm
marker 7.5 mm  DICT_4X4_50`) so you can check the board in your hand against
the numbers the routine is being given. A board scaled by the print dialog
gives a calibration wrong by that same scale, with residuals that look
perfect.

Mount it flat on the gripper. It may not move relative to the flange after
this point.

Bigger is better, and so is `--resolution 2K`. What decides the accuracy is how
many pixels a square covers at working distance. The 10 mm default is sized to
fit a gripper rather than to measure well: at 0.85 m it is about 8 px per
square in HD, so it wants 2K and good light, and it wants the camera closer. A
30 mm board at 0.85 m is 25 px per square in HD and 50 px at 2K, which on a
synthetic bench is the difference between 0.6 mm and 0.16 mm of camera position
error. Orientation suffers most from a small board — the board's own pose comes
back about three times less precisely at 10 mm than at 30 mm — so print the
largest board the gripper will carry flat. 2K runs at 15 fps, which costs
nothing here — the arm stops at every pose anyway.

**2. Put the arm where the camera can see the board.** The sweep starts from
wherever the arm already is and stays near it, so this pose decides what gets
calibrated.

**3. Dry run first.**

```bash
python -m xarm7_lib.calibrate_handeye run --dry-run
```

The arm visits ~18 generated poses, each one checked against the safety box and
for self-collision before it is commanded, and the result is reported without
anything being written. A pose where the board is not properly in view is
skipped with a reason, which is ordinary — but if most of them are skipped, the
board is not where the camera can see it and step 2 needs redoing.

Two numbers decide whether to keep it:

- **reprojection RMS** under about 0.5 px — the corners fit.
- **leave-one-out wander** of a few mm — the answer does not depend on any one
  pose. This is the one that catches a pose set that did not rotate enough, and
  it can be bad while the RMS looks excellent.

A window shows the left eye throughout, with the corners the solver is going
to use drawn in green and the ones only one eye could see in amber. Watch it
for the first few poses: a board that is edge-on, half out of frame, washed out
or still ringing from the move is obvious there and invisible in a corner
count. `--no-window` turns it off for a session with no display, and
`--display-scale` sizes it; if OpenCV has no GUI to open, it says so once and
carries on without one.

`--manual` instead releases the arm between poses so you can push it by hand,
for a bench where the generated sweep puts the board out of view. Turn the
wrist a long way between poses: rotation is what solves this. With the window
up, space captures the pose you have posed and `q` finishes — the live corner
count is what you are aiming at while you move the arm.

**4. File it.**

```bash
python -m xarm7_lib.calibrate_handeye run
```

Same thing, then written to `<installation>/xarm7_lib/data/handeye/SN<serial>.json`
— on a lab machine, under `/opt/16384`, which students can read and cannot
write. Every student on the station now gets it from `load_handeye()`. Writing
there needs a TA's access; `--user` files it under `~/.config/xarm7_lib/handeye/`
for this machine's accounts instead, and `--output` puts it anywhere. Those two
are also where a TA tries a candidate without disturbing the class's copy —
they are searched first, in that order.

**5. Check it.**

```bash
python -m xarm7_lib.calibrate_handeye verify
```

Compares the board pose the camera measures against the one forward kinematics
predicts, live, while you move the arm by hand. The window draws both: magenta
rings where the calibration says the corners are, green dots where they
actually are. Rings sitting on dots is a good calibration, and the gap between
them is the error the table is reporting. Do this at poses the fit never
saw — it is the only check that catches a calibration that is self-consistent
and wrong. A tape measure to the camera is worth one more minute: the reported
position should be where the camera physically is, to within a few centimetres.

## What goes wrong quietly

Three things, none of which announces itself:

- **The board printed at the wrong scale.** Nothing downstream can tell. Hence
  the spec on the printout and the caliper.
- **A TCP offset configured on the controller.** Forward kinematics here is of
  the bare flange, so an offset makes every pose wrong by a constant and the
  calibration absorbs it. `run` compares the model against the controller
  before it moves anything and refuses to continue if they disagree.
- **Poses that did not rotate enough.** Hand-eye is solved by rotation; poses
  differing only in position leave the camera's orientation unconstrained. The
  generated sweep turns each wrist joint in turn for this reason, and the
  leave-one-out number is what exposes it.

The solver itself can be checked without a camera or an arm:

```bash
python -m xarm7_lib.calibrate_handeye self-test
```

A synthetic bench with a known answer, real kinematics and real board geometry.
It recovers the camera to about 0.03° and 0.5 mm, and fails loudly if any
transform is composed the wrong way round.

# The picture

`Robot()` comes with a meshcat scene showing where the arm is, and prints its
URL on startup. In simulation that is the arm MuJoCo is stepping. On the real
arm it is a `RobotMirror`: a copy of the same model with the measured joint
angles written into it and forward kinematics run over them, thirty times a
second, on its own thread. No physics — a mirror that stepped physics would
droop away from the arm it is supposed to be showing.

Either way the safety box is drawn around it, and faces light up as the arm
gets near them. Pass `visualize=False` for a run with nobody watching.

```python
with Robot() as robot:          # prints e.g. Meshcat: http://127.0.0.1:7000/static/
    robot.set_joint_targets(HOME_POSE)
```

## Drawing other things beside the arm

`register_visualizer` adds something to that scene and keeps it up to date on a
thread of its own:

```python
det = LightsaberDetector()
with Robot() as robot:
    robot.register_visualizer(det)
    ...                              # det.blade_in_base() while the picture keeps up
```

A visualizer is any object with these. Only `poll` is required:

| | when | may it block? | what it does |
|---|---|---|---|
| `poll()` | every iteration | **yes** | the slow work: read a camera, find something, remember it |
| `geometry()` | straight after | no | say what should be in the scene *now* |
| `draw(node)` | straight after | no | the escape hatch: its own meshcat node, for what `geometry` cannot say |
| `close()` | once, at the end | — | let go of whatever it owns |
| `frame` | — | — | `"base"` (the default), `"camera"` or `"world"` |
| `rate` | — | — | Hz, or `None` to run as fast as `poll` returns |
| `serial` | — | — | which camera, when `frame` is `"camera"` — see [Frames](#frames) |

The split between `poll` and `geometry` is the reason this is worth having.
Reading a camera and finding something in the picture takes 60–140 ms, and
meshcat speaks over a socket that has to be held exclusively while it is
written to. `poll` runs with nothing held, so it can take as long as it likes;
only the handful of messages `geometry` implies are sent under the lock. The
arm's own thirty frames a second never wait for the camera.

`geometry()` returns `{name: (geometry, material, transform)}` — exactly
meshcat's `set_object` arguments plus a pose, so a point cloud or a line set
goes through unchanged:

```python
import meshcat.geometry as g
from xarm7_lib import segment_transform

class Stick:
    frame = "camera"            # the numbers the detector produced, untouched

    def __init__(self):
        # Once, here — not in geometry(). The scene is diffed, so a geometry
        # object it has seen before is only moved. Rebuilding it every frame
        # pushes a new mesh over the socket thirty times a second instead.
        self.shaft = g.Cylinder(1.0, radiusBottom=1.0, radiusTop=0.6)
        self.red = g.MeshLambertMaterial(color=0xFF2020)

    def poll(self):
        self.ends = my_detector.read()          # slow; nothing is held

    def geometry(self):
        if self.ends is None:
            return {}                           # nothing named, nothing on screen
        return {"blade": (self.shaft, self.red,
                          segment_transform(*self.ends, radius=0.0125))}
```

Two things follow from `geometry()` being a statement about *now* rather than a
list of changes. A name that stops appearing is hidden, so a detection going
stale leaves the screen without anyone having to remember to remove it — which
matters, because a stick left on screen from four seconds ago is the worst
outcome available here: it looks right. And a name that comes back is shown
again without the mesh being re-sent.

`segment_transform(start, end, radius)` places a unit cylinder as the segment
between two points. The length and radius live in the transform rather than in
the geometry, which is what makes a moving, re-lengthening stick cost one
message a frame.

### Frames

`frame` says what the numbers in those transforms mean, and the subtree is
placed accordingly:

| `frame` | the transform applied to the whole subtree |
|---|---|
| `"world"` | nothing |
| `"base"` | the arm's base frame — what joint angles, the safety box and `points_to_base` all speak |
| `"camera"` | the base frame composed with the bench's hand-eye calibration |

So a detector declares `frame = "camera"` and hands over the metres its
stereo gave it. It never loads a calibration, never composes a transform, and
never has to know that the MJCF stands the arm's base 120 mm above the world
origin. Re-filing the calibration and calling `handle.refresh_frame()` moves
everything it ever drew.

A visualizer in camera coordinates should also say *which* camera, as a
`serial` or as a `camera` exposing one — `LightsaberDetector` has both, so
nothing is needed for the usual case:

```python
class Stick:
    frame = "camera"

    @property
    def serial(self):
        return self.camera.serial      # whatever the camera reported
```

That is what the calibration is filed under. It has to come from the
visualizer because a ZED keeps its serial in a register that can only be read
by *opening* the camera, and the visualizer is already streaming it: asked
instead of told, the library would open a camera that is busy, get nothing
back, and go quiet about the one camera it is certainly looking through.
Without a serial it falls back to probing the bus, which is right for a
visualizer that owns no camera and fails for one that does. `handeye=` on
`register_visualizer` hands a calibration over outright and skips the lookup
either way.

With `frame = "camera"` and no calibration filed on the bench, nothing is
drawn, and the reason is printed once. That is deliberate: camera coordinates
drawn as if they were base coordinates would be a confident picture of the
wrong place, about a metre out, and would cost somebody an afternoon. `poll()`
keeps running, so readings stay available.

### When something goes wrong

`register_visualizer` returns a handle. `stop()` takes the visualizer off the
scene; `Robot.close()` stops every one it handed out, before disconnecting the
arm, and the thread is always joined before `close()` is called — a camera
released while another thread is mid-read crashes inside the driver rather than
raising.

A visualizer that fails is disabled rather than left printing: `geometry()` and
`draw()` are pure, so one traceback and it stops; `poll()` touches hardware, so
it gets five tries first, because a dropped USB frame is not a reason to stay
dark for the rest of the session. Either way the arm keeps running, and so does
everything else registered. A browser tab that goes away takes the picture with
it and nothing else.

## Checking it without a robot

```bash
python -m xarm7_lib.visualize
```

The simulated arm with a stick tumbling around it — the whole path with no
hardware and no camera. `--mirror` drives a `RobotMirror` from a fake arm
instead, which is the only way to exercise the real-arm picture before there is
a real arm. `--rate`, `--churn`, `--bad-name` and `--raise-in {poll,geometry}`
provoke the failure paths on purpose; two of them are silent when they happen
by accident, which is why they have a switch.

```bash
pip install -e ".[dev]" && python -m pytest tests -q
```

The frame chain and the 120 mm are asserted there, against the real MJCF.
