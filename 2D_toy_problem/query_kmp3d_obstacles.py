import pickle as pkl
import time

import numpy as np

from itertools import count
from kmp3d import KMP, GMM, ReferenceTrajectoryPoint

from pyrobolearn.tools.interfaces.controllers.xbox import XboxControllerInterface
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA

class Environment:
    def __init__(self, spheres = [[0.09, -0.44, 0.715, 1]], cuboids = None):
        # Define size of the map
        self.res = 1.
        # Initialize XBox controller
        self.controller = XboxControllerInterface(use_thread=True, verbose=False)
        # Create simulator
        self.sim = Bullet()
        # create world
        self.world = BasicWorld(self.sim)
        # create robot
        self.robot = KukaIIWA(self.sim)
        # define start/end pt
        self.start_pt = [-0.75, 0., 0.75]
        self.end_pt = [0.75, 0., 0.75]
        # Initialize robot location
        self.send_robot_to(np.array(self.start_pt), dt=2)

        # Get via points from user
        via_pts = self.get_via_pts()  # [[-0.19, -0.65, 0.77], [0.51, -0.44, 0.58]]
        self.motion_targets = [self.start_pt] + via_pts + [self.end_pt]

    def send_robot_to(self, location, dt):
        start = time.perf_counter()
        for _ in count():
            joint_ids = self.robot.joints
            link_id = self.robot.get_end_effector_ids(end_effector=0)
            qIdx = self.robot.get_q_indices(joint_ids)

            x = self.robot.sim.get_link_world_positions(body_id=self.robot.id, link_ids=link_id)
            q = self.robot.calculate_inverse_kinematics(link_id, position=location)
            self.robot.set_joint_positions(q[qIdx], joint_ids)
            self.world.step()

            if time.perf_counter() - start >= dt:
                break

    def get_via_pts(self):
        print("Move green dot to via point. Press 'A' to save. Press 'X' to finish.")

        X_desired = [0., 0., 1.3]
        speed_scale_factor = 0.01
        via_pts = []

        sphere = None
        for _ in count():

            last_trigger = self.controller.last_updated_button
            movement = self.controller[last_trigger]

            if last_trigger == 'LJ':
                X_desired[0] = X_desired[0] + round(movement[0], 1) * speed_scale_factor
                X_desired[1] = X_desired[1] + round(movement[1], 1) * speed_scale_factor
            elif last_trigger == 'RJ':
                X_desired[2] = X_desired[2] + round(movement[1], 1) * speed_scale_factor
            elif last_trigger == 'A' and movement == 1:
                print("Via point added")
                via_pts.append(X_desired)
                time.sleep(0.5)
            elif last_trigger == 'X' and movement == 1:
                self.world.sim.remove_body(sphere)
                print("Via point generation complete.")
                break

            if sphere:
                self.world.sim.remove_body(sphere)
            sphere = self.world.load_visual_sphere(X_desired, radius=0.01, color=(0, 1, 0, 1))

        return via_pts

if __name__ == "__main__":
    env = Environment()