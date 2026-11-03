"""The MuJoCo model of the arm, and the handful of things both pictures need.

Two classes draw the same arm: `SimulatedXArm7`, which is also running the
physics, and `visualize.RobotMirror`, which is drawing an arm that is somewhere
else entirely. Everything they share about the *model* — how to load it, where
the seven joints are in it, how to take the gripper out of the picture, and
where the base frame sits — lives here, as functions over a model rather than
methods on either of them.

The base frame is the reason this module exists rather than being copied. An
MJCF is free to put the frame joint configurations are written in anywhere in
its world, and Menagerie's xarm7 puts it 120 mm up; anything drawn in base
coordinates against the wrong frame lands 120 mm off the arm, which looks like
a calibration error and isn't one. One copy of `base_frame`, one place to get
it wrong.
"""

import collections

import mujoco
import numpy as np

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))

GRIPPER_BODIES = ("gripper", "finger", "knuckle")
HIDDEN_GROUP = 3  # outside the groups the visualizer draws

# How near a face of the safety box the arm has to be for that face to be
# drawn. Boxed in on all six sides at once the scene is mostly wall, and none
# of it is telling you anything; a face earns its place once it is close enough
# to be what stops you.
FACE_PROXIMITY = 0.15  # m

# ...except the box's z_min face, which the viewer draws as the surface the arm
# is standing on. Hiding the ground leaves the arm floating in an empty scene —
# the MJCF has no floor of its own — so it stays whatever the arm is doing.
FLOOR_FACE = "zmin"

ArmIndex = collections.namedtuple("ArmIndex", "jid qadr vadr act")


def load_arm_model(mjcf_path=None):
    """The MuJoCo model: Menagerie's xarm7 unless given another MJCF."""
    if mjcf_path is None:
        from robot_descriptions.loaders.mujoco import load_robot_description

        return load_robot_description("xarm7_mj_description")
    return mujoco.MjModel.from_xml_path(str(mjcf_path))


def index_arm(model):
    """Locate the arm's joints, DOFs and actuators (the MJCF also carries
    a gripper, which this API doesn't expose)."""
    jid, qadr, vadr, act = [], [], [], []
    for name in ARM_JOINTS:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise ValueError(f"joint {name!r} not found in the model")
        jid.append(j)
        qadr.append(model.jnt_qposadr[j])
        vadr.append(model.jnt_dofadr[j])

        driving = [
            a
            for a in range(model.nu)
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and model.actuator_trnid[a, 0] == j
        ]
        if len(driving) != 1:
            raise ValueError(f"expected exactly one actuator for {name!r}")
        act.append(driving[0])
    return ArmIndex(*(np.asarray(v) for v in (jid, qadr, vadr, act)))


def find_gripper_geoms(model):
    """Geoms belonging to the MJCF's gripper, by the bodies they hang off."""
    geoms = []
    for geom in range(model.ngeom):
        body = model.geom_bodyid[geom]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if any(part in name for part in GRIPPER_BODIES):
            geoms.append(geom)
    return geoms


def hide_gripper(model, geoms):
    """Take the gripper out of the contact set and out of the picture.

    The real arm this mirrors has a bare flange, and the guard's model has
    no gripper either. Leaving the MJCF's one collidable would have the
    simulator jam on a part the hardware doesn't carry — and jam silently,
    since the guard would have nothing to say about it. Only the geometry
    goes; the mass stays, standing in for whatever is actually bolted on.
    """
    for geom in geoms:
        model.geom_contype[geom] = 0
        model.geom_conaffinity[geom] = 0
        # MuJoCo's own way of saying "don't draw this", which the meshcat
        # visualizer follows too.
        model.geom_group[geom] = HIDDEN_GROUP


def base_frame(model, data, joint_id):
    """The arm's base frame in the world, as a 4x4.

    The frame every joint configuration is written in — and so the frame
    the safety box is written in — is the one joint1 turns in, which an
    MJCF is free to put anywhere in its world. Menagerie's xarm7 mounts
    `link_base` at z = 120 mm, so the two frames are not the same one and
    anything given in base coordinates has to be carried across.

    Read out of `data.xpos`, so at least one kinematics pass must have run.
    Both callers arrange that before asking — `SimulatedXArm7._reset_state`
    runs `mj_forward`, and `RobotMirror.__init__` calls `set_joints` — and a
    reordering that broke it would silently return the identity.
    """
    body = model.body_parentid[model.jnt_bodyid[joint_id]]
    frame = np.eye(4)
    frame[:3, :3] = data.xmat[body].reshape(3, 3)
    frame[:3, 3] = data.xpos[body]
    return frame


def draw_box(viz, guard, frame):
    """Show the guard's safety box in the viewer, if there is both."""
    if viz is None:
        return
    box = guard.box if guard is not None else None
    if box is None:
        viz.clear_safety_box()
    else:
        # In base coordinates, like the guard's own box — the picture would
        # otherwise sit 120 mm under the arm and libel safe poses.
        viz.draw_safety_box(box, frame=frame)


def update_faces(viz, guard, q):
    """Leave drawn only the floor and the faces the arm is getting close to.

    Measuring the six gaps costs around 150 us, a tenth of a physics tick,
    which is why this goes on the viewer's clock and not the simulation's.
    """
    if viz is None or guard is None:
        return
    clearances = guard.box_clearances(q)
    near = [face for face, gap in clearances.items() if gap <= FACE_PROXIMITY]
    viz.show_safety_faces(near + [FLOOR_FACE])
