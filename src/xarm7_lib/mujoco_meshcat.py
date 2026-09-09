"""Render a MuJoCo model in meshcat.

MuJoCo's own viewer wants the main thread (and `mjpython` on macOS), which is
awkward for a library that simulates on a background thread. This walks the
model's geoms once, publishes an equivalent meshcat scene, and then just pushes
`geom_xpos`/`geom_xmat` on every sync — so the browser is the display and the
simulation stays wherever it likes.

Only the geometry MuJoCo itself would draw is published: `geom_groups` defaults
to MuJoCo's own default-visible groups (0-2), which in the Menagerie models
means visual meshes without the collision primitives.

`draw_safety_box` adds the one thing the MuJoCo model has nothing to say about:
the volume the arm is allowed to occupy. It is drawn under its own path, so it
survives the model being re-published and can be moved or removed on its own,
and under that path's own transform, so it can be given in the robot's base
frame rather than the world's — see the `frame` argument.
"""

import threading

import meshcat
import meshcat.geometry as g
import mujoco
import numpy as np

DEFAULT_GEOM_GROUPS = (0, 1, 2)
_PLANE_EXTENT = 10.0  # metres drawn for an infinite plane
_PLANE_THICKNESS = 1e-3

# The safety box, drawn as six thin slabs. The floor reads as a surface the arm
# stands on, so it is solid enough to see; the other five are the boundary
# rather than an object, and stay faint enough to see the arm through.
_BOX_THICKNESS = 2e-3
_FLOOR_COLOR = 0x555F6B
_FLOOR_OPACITY = 0.55
_WALL_COLOR = 0xD08B2C
_WALL_OPACITY = 0.12

# three.js builds cylinders along +Y; MuJoCo's axis is +Z.
_Y_TO_Z = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _translation(xyz):
    T = np.eye(4)
    T[:3, 3] = xyz
    return T


class MeshcatVisualizer:
    """A meshcat scene mirroring a MuJoCo model.

    Parameters
    ----------
    model, data : the MuJoCo model and the state to render.
    zmq_url : connect to an existing meshcat server instead of starting one.
    open_browser : open the scene in a browser tab on construction.
    geom_groups : which MuJoCo geom groups to draw.
    prefix : path the scene is published under, so other objects can coexist.
    """

    def __init__(
        self,
        model,
        data,
        zmq_url=None,
        open_browser=True,
        geom_groups=DEFAULT_GEOM_GROUPS,
        prefix="mujoco",
    ):
        self.model = model
        self.data = data
        self.viewer = meshcat.Visualizer(zmq_url=zmq_url)
        # meshcat speaks over a REQ/REP ZMQ socket, which is not thread-safe.
        # sync() takes this lock; take it yourself before touching `viewer`.
        self.lock = threading.RLock()

        groups = set(int(x) for x in geom_groups)
        self._geoms = [
            gid for gid in range(model.ngeom) if int(model.geom_group[gid]) in groups
        ]
        self._safety_path = "safety"
        self._safety_faces = {}  # face name -> whether it is currently shown
        # World-attached geoms never move, so they only need placing once.
        self._moving = [gid for gid in self._geoms if model.geom_bodyid[gid] != 0]
        self._paths = {gid: f"{prefix}/geom_{gid}" for gid in self._geoms}

        self._load()
        self.sync(force=True)
        if open_browser:
            self.viewer.open()

    @property
    def url(self):
        return self.viewer.url()

    def _load(self):
        with self.lock:
            for gid in self._geoms:
                material = self._material(gid)
                for subpath, geometry, local in self._geometry(gid):
                    node = self.viewer[self._paths[gid]]
                    if subpath:
                        node = node[subpath]
                    node.set_object(geometry, material)
                    if local is not None:
                        node.set_transform(local)

    def _material(self, gid):
        matid = int(self.model.geom_matid[gid])
        rgba = (
            self.model.mat_rgba[matid] if matid >= 0 else self.model.geom_rgba[gid]
        )
        r, gr, b, a = (np.clip(rgba, 0.0, 1.0) * 255).astype(int)
        color = (int(r) << 16) + (int(gr) << 8) + int(b)
        opacity = float(np.clip(rgba[3], 0.0, 1.0))
        return g.MeshLambertMaterial(
            color=color, opacity=opacity, transparent=opacity < 1.0
        )

    def _geometry(self, gid):
        """(subpath, geometry, local transform) parts making up one geom."""
        gtype = self.model.geom_type[gid]
        size = self.model.geom_size[gid]
        T = mujoco.mjtGeom

        if gtype == T.mjGEOM_SPHERE:
            return [("", g.Sphere(float(size[0])), None)]

        if gtype == T.mjGEOM_BOX:
            return [("", g.Box((2.0 * size[:3]).tolist()), None)]

        if gtype == T.mjGEOM_ELLIPSOID:
            return [("", g.Ellipsoid(size[:3].tolist()), None)]

        if gtype == T.mjGEOM_CYLINDER:
            return [
                ("", g.Cylinder(2.0 * float(size[1]), float(size[0])), _Y_TO_Z),
            ]

        if gtype == T.mjGEOM_CAPSULE:
            radius, half = float(size[0]), float(size[1])
            return [
                ("cylinder", g.Cylinder(2.0 * half, radius), _Y_TO_Z),
                ("cap_pos", g.Sphere(radius), _translation([0.0, 0.0, half])),
                ("cap_neg", g.Sphere(radius), _translation([0.0, 0.0, -half])),
            ]

        if gtype == T.mjGEOM_PLANE:
            # MuJoCo planes are infinite when a half-extent is 0.
            sx = float(size[0]) if size[0] > 0 else _PLANE_EXTENT
            sy = float(size[1]) if size[1] > 0 else _PLANE_EXTENT
            box = g.Box([2.0 * sx, 2.0 * sy, _PLANE_THICKNESS])
            return [("", box, _translation([0.0, 0.0, -0.5 * _PLANE_THICKNESS]))]

        if gtype == T.mjGEOM_MESH:
            mesh = int(self.model.geom_dataid[gid])
            vadr = self.model.mesh_vertadr[mesh]
            fadr = self.model.mesh_faceadr[mesh]
            vertices = self.model.mesh_vert[vadr : vadr + self.model.mesh_vertnum[mesh]]
            faces = self.model.mesh_face[fadr : fadr + self.model.mesh_facenum[mesh]]
            return [("", g.TriangularMeshGeometry(vertices, faces), None)]

        return []  # heightfields and SDFs aren't rendered

    def draw_safety_box(self, box, floor=True, frame=None):
        """Draw the volume the arm must stay inside.

        `box` is ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in metres,
        read in `frame`: a 4x4 world transform, or None for the world frame the
        model is published in. A safety box is written in the robot's *base*
        frame, and an MJCF is free to put that frame anywhere — the Menagerie
        xarm7 stands its base 120 mm above the world origin — so pass the base
        frame rather than letting the two be confused. Drawn in the wrong one
        the faces land 120 mm off the arm, and configurations the guard is
        perfectly happy with appear to cross a wall.

        The floor is the z_min face, drawn solid enough to read as the surface
        the arm is standing on; pass `floor=False` to leave it out if the model
        already draws one.

        Replaces whatever was drawn before, so it can be called again whenever
        the box moves.
        """
        bounds = np.asarray(box, dtype=float).reshape(3, 2)
        centre = bounds.mean(axis=1)
        extent = bounds[:, 1] - bounds[:, 0]

        self.clear_safety_box()
        with self.lock:
            # The faces are placed in `frame`; the group carries it to the world.
            self.viewer[self._safety_path].set_transform(
                np.eye(4) if frame is None else np.asarray(frame, dtype=float)
            )
            for axis in range(3):
                for side, name in ((0, "min"), (1, "max")):
                    is_floor = axis == 2 and side == 0
                    if is_floor and not floor:
                        continue
                    # A slab spanning the other two axes, flattened onto this face.
                    size = extent.copy()
                    size[axis] = _BOX_THICKNESS
                    position = centre.copy()
                    position[axis] = bounds[axis, side]

                    colour = _FLOOR_COLOR if is_floor else _WALL_COLOR
                    opacity = _FLOOR_OPACITY if is_floor else _WALL_OPACITY
                    face = f"{'xyz'[axis]}{name}"
                    self._safety_faces[face] = True
                    node = self.viewer[f"{self._safety_path}/{face}"]
                    node.set_object(
                        g.Box(size.tolist()),
                        g.MeshLambertMaterial(
                            color=colour, opacity=opacity, transparent=True
                        ),
                    )
                    node.set_transform(_translation(position))

    def show_safety_faces(self, names):
        """Show these faces of the safety box and hide the rest.

        `names` are the ones `draw_safety_box` made, "xmin" through "zmax".
        This is cheap enough to call every frame: the faces stay in the scene
        and only their visibility is flipped, so nothing is re-published, and
        a face whose state hasn't changed isn't even sent.
        """
        wanted = set(names)
        with self.lock:
            for face, shown in self._safety_faces.items():
                if shown == (face in wanted):
                    continue
                self._safety_faces[face] = not shown
                self.viewer[f"{self._safety_path}/{face}"].set_property(
                    "visible", self._safety_faces[face]
                )

    def clear_safety_box(self):
        """Remove the safety box from the scene."""
        with self.lock:
            self._safety_faces.clear()
            self.viewer[self._safety_path].delete()

    def sync(self, force=False):
        """Push the current geom placements to the browser."""
        geoms = self._geoms if force else self._moving
        with self.lock:
            for gid in geoms:
                T = np.eye(4)
                T[:3, :3] = self.data.geom_xmat[gid].reshape(3, 3)
                T[:3, 3] = self.data.geom_xpos[gid]
                self.viewer[self._paths[gid]].set_transform(T)

    def close(self):
        """Drop the connection and stop the server we started.

        `meshcat.Visualizer.close()` is unusable here — it forwards to a
        `ViewerWindow.close()` that doesn't exist — so release the pieces
        directly, defensively enough to survive meshcat changing its internals.
        The browser tab goes dead with the server.
        """
        with self.lock:
            window = getattr(self.viewer, "window", None)
            if window is None:
                return
            socket = getattr(window, "zmq_socket", None)
            if socket is not None:
                socket.close(linger=0)
            server = getattr(window, "server_proc", None)
            if server is not None:
                server.terminate()
