"""Self-collision and workspace checking for the xArm7, shared by both backends.

Everything here is a question about a *configuration*, never about the robot:
`check(q)` says whether the arm may be at `q`, and knows nothing about how it
got there or which backend is asking. That is what lets the simulator and the
real arm enforce the same rule — they differ only in what they do with the
answer (the sim stops at the boundary and warns, the real arm refuses).

The geometry is UFACTORY's own, via Pinocchio and Coal:

    model       `xarm7_description`, 7 DoF and no gripper, so `q` is exactly the
                7 joint values the rest of this library passes around.
    meshes      the simplified `xarm7_1305/collision` meshes, reduced further to
                their convex hulls — as loaded they are triangle-soup BVHs and a
                single check costs 11 ms, which is 500x too slow to sit inside a
                servo loop. Convex hulls bring that to ~30 us.
    pairs       from the arm's own MoveIt SRDF. Of the 28 link pairs, 17 are
                disabled there as Adjacent/Default/Never, leaving 11 that can
                genuinely collide.

That SRDF is doing real work. Links which are merely *adjacent* rest against
each other permanently — `link1` and `link_base` sit fractions of a millimetre
apart — so any rule of the form "flag pairs closer than X" flags the home pose
and every other pose with it. But the pairs that need flagging are no further
away: `link2` and `link4` pass within 10 mm at the elbow and genuinely collide
elsewhere in the workspace. Distance alone cannot separate the two cases; only
knowing which pairs are *allowed* to touch can, and that list is what the SRDF
provides.

With the adjacent pairs disabled, a clearance margin becomes meaningful again,
and `margin` is applied as Coal's security margin — the check trips that far
before contact, rather than after.

The safety box is carried in the same collision problem as six half-spaces, one
per face, each oriented so that the solid half is the *outside*. Leaving the box
is then just another collision, found by the same query and reported the same
way, and the offending face names itself.
"""

import os
import re

import coal
import numpy as np
import pinocchio as pin

# Metres, in the robot's base frame, sized to the arm's actual envelope: the
# shoulder sweeps back to x = -204 mm in ordinary poses and the arm stands over
# a metre tall, so a box drawn only around the reachable *workspace* would
# exclude the arm holding still at home. z = 0 is the mounting surface.
DEFAULT_BOX = ((-0.10, 0.70), (-0.45, 0.45), (0.00, 0.80))

# 10 mm of clearance, with room to spare: the home pose, where some link pairs
# legitimately sit close together, only starts reading as a collision at 30 mm.
DEFAULT_MARGIN = 0.010

_DESCRIPTION = "xarm7_description"
_SRDF_RELPATH = "xarm_moveit_config/srdf/_xarm7_macro.srdf.xacro"

# The base is bolted down. Including it in the box would mean reporting a
# violation the caller has no way to move out of.
_ANCHORED = "link_base"

# Qhull, asked for a triangulated hull.
_QHULL_TRIANGULATE = "Qt"

_AXES = "xyz"
_DEFAULT_PATH_STEP = 0.02  # rad between samples when scanning a path


class SafetyError(ValueError):
    """A commanded configuration would self-collide or leave the safety box.

    A `ValueError`, like the joint-range check it sits next to: asking the arm
    to go somewhere it cannot legally be is a bug in the caller, not a fault in
    the robot.
    """

    def __init__(self, violation):
        super().__init__(str(violation))
        self.violation = violation


class Violation:
    """Why a configuration was rejected.

    `kind` is "self-collision" or "safety-box"; `detail` names the two links
    that met, or the face that was crossed.
    """

    def __init__(self, kind, detail, q):
        self.kind = kind
        self.detail = detail
        self.q = np.asarray(q, dtype=float).copy()

    def __str__(self):
        joints = ", ".join(f"{v:.3f}" for v in self.q)
        return f"{self.kind}: {self.detail} at joints [{joints}]"

    def __repr__(self):
        return f"<Violation {self}>"


class SafetyGuard:
    """Answers whether the arm may be at a given configuration.

    Parameters
    ----------
    box : ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in metres, in the
        robot's base frame; every part of the arm but the base must stay inside
        it. None turns the box off and leaves only self-collision checking.
    margin : clearance in metres. The check trips this far before contact.

    Building one loads and convexifies the meshes, which takes a moment; the
    checks themselves are tens of microseconds, so share a guard rather than
    building one per call.
    """

    def __init__(self, box=DEFAULT_BOX, margin=DEFAULT_MARGIN):
        from robot_descriptions.loaders.pinocchio import load_robot_description

        robot = load_robot_description(_DESCRIPTION)
        self.model = robot.model
        self.geometry = robot.collision_model
        self.nq = self.model.nq

        # Triangle meshes make this 500x slower for no benefit here: the links
        # are convex enough that their hulls are what the arm's own planner
        # checks anyway.
        #
        # It has to be `buildConvexHull`, not the `buildConvexRepresentation`
        # sitting next to it in Coal's API: that one relabels the mesh as a
        # Convex without hulling anything ("it does not check that the object is
        # convex, it does not compute a convex hull", says its own header). The
        # support function then hill-climbs the neighbour graph of a polytope
        # that isn't convex, settles on a local extreme, and places the arm
        # further inside the box than it is — which loses box violations
        # outright: 49 of 3000 random configurations here read as safe with the
        # arm outside the box, the worst of them by 47 mm.
        for obj in self.geometry.geometryObjects:
            # The return says whether the mesh was *already* convex, not whether
            # the hull was built; these aren't, which is the point of the call.
            obj.geometry.buildConvexHull(True, _QHULL_TRIANGULATE)
            if obj.geometry.convex is None:
                raise RuntimeError(f"no convex hull built for {obj.name}")
            obj.geometry = obj.geometry.convex

        self.geometry.addAllCollisionPairs()
        self._apply_srdf()
        self._n_self_pairs = len(self.geometry.collisionPairs)

        self._wall_names = self._add_walls()
        self._data = self.model.createData()
        self._geom_data = self.geometry.createData()

        self.margin = float(margin)
        self._set_box(box)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def margin(self):
        """Clearance in metres at which the check trips."""
        return self._margin

    @margin.setter
    def margin(self, value):
        self._margin = float(value)
        for request in self._geom_data.collisionRequests:
            request.security_margin = self._margin

    @property
    def box(self):
        """The safety box, or None if there isn't one."""
        return self._box

    def _set_box(self, box):
        """Place the safety box, or None to switch it off.

        Construction-time only, and deliberately so: a guard whose boundary can
        be moved by the code it is guarding is not a guard. The six faces are
        already in the collision problem, so this only slides them onto the
        bounds and arms their pairs.
        """
        if box is None:
            self._box = None
            for index in self._wall_pairs:
                self._geom_data.deactivateCollisionPair(index)
            return

        bounds = np.asarray(box, dtype=float).reshape(3, 2)
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError(f"safety box has a non-positive extent: {box}")
        self._box = tuple((float(lo), float(hi)) for lo, hi in bounds)

        for axis in range(3):
            for sign, bound in ((-1.0, bounds[axis, 0]), (1.0, bounds[axis, 1])):
                gid = self._wall_names[_wall_name(axis, sign)]
                # The half-space is a plane through the geometry's own origin,
                # so placing that origin on the face puts the plane there.
                placement = self.geometry.geometryObjects[gid].placement
                placement.translation = np.eye(3)[axis] * bound
        for index in self._wall_pairs:
            self._geom_data.activateCollisionPair(index)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check(self, q):
        """Return a `Violation` if the arm may not be at `q`, else None."""
        q = self._as_config(q)
        if not pin.computeCollisions(
            self.model, self._data, self.geometry, self._geom_data, q, True
        ):
            return None
        return self._first_violation(q)

    def check_path(self, start, goal, max_step=_DEFAULT_PATH_STEP):
        """Check the straight joint-space line from `start` to `goal`.

        Both backends move this way for a position command — the controller
        time-synchronizes the joints, and the simulator ramps its setpoint along
        the same line — so sampling the line is a faithful model of the path,
        not an approximation of one.

        Returns `(violation, reached)`: `violation` is None if the whole path is
        clear, and `reached` is the fraction of the way along it that is safe.
        """
        start = self._as_config(start)
        goal = self._as_config(goal)
        span = float(np.max(np.abs(goal - start)))
        steps = max(1, int(np.ceil(span / float(max_step))))

        safe = 0.0
        for step in range(steps + 1):
            fraction = step / steps
            violation = self.check(start + fraction * (goal - start))
            if violation is not None:
                return violation, safe
            safe = fraction
        return None, 1.0

    def box_clearances(self, q):
        """How close the arm is to each face of the box at `q`, in metres.

        Returns `{face_name: gap}` over the same six faces `check` reports —
        "xmin", "xmax", "ymin" and so on — where the gap is the room left
        before that face and goes negative once the arm is past it. Empty if
        there is no box.

        A face is an axis-aligned plane, so the room left before it is just the
        distance from it to the arm's furthest point along that axis: no
        distance solver, only the extreme vertex of each hull.

        This is a question for a viewer, not for a controller. `check` remains
        the thing that decides what the arm is allowed to do.
        """
        if self._box is None:
            return {}
        q = self._as_config(q)
        pin.updateGeometryPlacements(
            self.model, self._data, self.geometry, self._geom_data, q
        )

        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for gid, points in self._boxed_points.items():
            placement = self._geom_data.oMg[gid]
            world = points @ placement.rotation.T + placement.translation
            low = np.minimum(low, world.min(axis=0))
            high = np.maximum(high, world.max(axis=0))

        gaps = {}
        for axis in range(3):
            bottom, top = self._box[axis]
            gaps[_wall_name(axis, -1.0)] = float(low[axis] - bottom)
            gaps[_wall_name(axis, 1.0)] = float(top - high[axis])
        return gaps

    def check_ray(self, q, velocity, horizon, max_step=_DEFAULT_PATH_STEP):
        """Check where a constant joint velocity leads, out to `horizon` seconds.

        Returns `(violation, time)`: `violation` is None if nothing is hit
        within the horizon, and `time` is how long the arm can hold this
        velocity before it would be.
        """
        q = self._as_config(q)
        velocity = np.asarray(velocity, dtype=float).reshape(-1)
        if velocity.size != self.nq:
            raise ValueError(f"expected {self.nq} velocities, got {velocity.size}")

        horizon = float(horizon)
        fastest = float(np.max(np.abs(velocity)))
        if fastest <= 0.0 or horizon <= 0.0:
            return self.check(q), 0.0

        steps = max(1, int(np.ceil(fastest * horizon / float(max_step))))
        safe = 0.0
        for step in range(steps + 1):
            t = horizon * step / steps
            violation = self.check(q + t * velocity)
            if violation is not None:
                return violation, safe
            safe = t
        return None, horizon

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _as_config(self, values):
        q = np.asarray(values, dtype=float).reshape(-1)
        if q.size != self.nq:
            raise ValueError(f"expected {self.nq} joint values, got {q.size}")
        return q

    def _apply_srdf(self):
        """Disable the link pairs the arm's MoveIt config says can't collide.

        The SRDF ships as a xacro macro, which `removeCollisionPairsFromXML`
        won't parse — and it doesn't complain, it just removes nothing and
        leaves every adjacent pair armed. So lift the rows out and rewrap them,
        then check that pairs actually went away.
        """
        import robot_descriptions.xarm7_description as description

        path = os.path.join(description.REPOSITORY_PATH, _SRDF_RELPATH)
        with open(path) as handle:
            rows = re.findall(
                r"<disable_collisions[^>]*/>", handle.read().replace("${prefix}", "")
            )
        if not rows:
            raise RuntimeError(f"no disable_collisions rows found in {path}")

        before = len(self.geometry.collisionPairs)
        pin.removeCollisionPairsFromXML(
            self.model,
            self.geometry,
            f'<robot name="xarm7">{"".join(rows)}</robot>',
            verbose=False,
        )
        if len(self.geometry.collisionPairs) >= before:
            raise RuntimeError(
                f"the SRDF at {path} disabled none of the {before} collision "
                "pairs; without it every adjacent link reads as a collision"
            )

    def _add_walls(self):
        """Add the six box faces as half-spaces and pair every link with them."""
        names = {}
        for axis in range(3):
            for sign in (-1.0, 1.0):
                # Coal's half-space is solid where `normal . x <= offset`, and
                # what we want to detect is the arm reaching the *far* side of
                # the face, so the normal points outward from the box.
                normal = np.zeros(3)
                normal[axis] = -sign
                names[_wall_name(axis, sign)] = self.geometry.addGeometryObject(
                    pin.GeometryObject(
                        _wall_name(axis, sign),
                        0,  # anchored to the world, not to a joint
                        pin.SE3.Identity(),
                        coal.Halfspace(normal, 0.0),
                    )
                )

        self._wall_pairs = []
        # The points of everything the box applies to, in their own frames, for
        # `box_clearances` to measure against the faces.
        self._boxed_points = {}
        walls = set(names.values())
        for gid, obj in enumerate(self.geometry.geometryObjects):
            if gid in walls or obj.name.startswith(_ANCHORED):
                continue
            self._boxed_points[gid] = np.asarray(obj.geometry.points(), dtype=float)
            for wall in walls:
                self.geometry.addCollisionPair(pin.CollisionPair(gid, wall))
                self._wall_pairs.append(len(self.geometry.collisionPairs) - 1)
        return names

    def _first_violation(self, q):
        """Name whichever pair tripped. Caller must have just run the query."""
        for index, result in enumerate(self._geom_data.collisionResults):
            if not result.isCollision():
                continue
            pair = self.geometry.collisionPairs[index]
            first = _link_name(self.geometry.geometryObjects[pair.first].name)
            second = _link_name(self.geometry.geometryObjects[pair.second].name)
            if index >= self._n_self_pairs:
                return Violation("safety-box", f"{first} crossed {second}", q)
            return Violation("self-collision", f"{first} and {second} met", q)
        return None  # the query said yes but no pair owns it; treat as safe


def _wall_name(axis, sign):
    return f"{_AXES[axis]}{'max' if sign > 0 else 'min'}"


def _link_name(name):
    """`link4_0` is how the loader names link 4's first geometry."""
    return re.sub(r"_\d+$", "", name)
