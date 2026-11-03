"""The shortest thing that proves the install works.

    python testing.py

With `ROBOT_IP` set this connects to the arm on the bench; without it you get
the simulator instead, and everything below behaves the same way. Either way a
meshcat scene opens and follows the arm -- the URL is printed as it starts.
"""

import time

from xarm7_lib import HOME_POSE, DemoVisualizer, Robot

# `with` releases the backend on the way out -- the connection for the real
# arm, the simulation thread and the viewer for the sim -- including on a
# Ctrl-C. The real arm is left stopped with its motors still holding:
# switching them off would let it fall under its own weight.
with Robot() as robot:
    # A stick tumbling around the arm, standing in for a detector. Anything
    # with a `poll` and a `geometry` can go here; see "Drawing other things
    # beside the arm" in the README.
    robot.register_visualizer(DemoVisualizer())

    print("joints:", robot.joint_values)
    robot.set_joint_targets(HOME_POSE)
    print("at home:", robot.joint_values)

    time.sleep(5.0)  # long enough to look at it
