import copy
import pickle as pkl
import time

import numpy as np

from itertools import count
from kmp3d import KMP, GMM, ReferenceTrajectoryPoint

from pyrobolearn.tools.interfaces.controllers.xbox import XboxControllerInterface
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA


def send_robot_to(location, dt):
    start = time.perf_counter()
    for _ in count():
        joint_ids = robot.joints
        link_id = robot.get_end_effector_ids(end_effector=0)
        qIdx = robot.get_q_indices(joint_ids)

        x = robot.sim.get_link_world_positions(body_id=robot.id, link_ids=link_id)
        q = robot.calculate_inverse_kinematics(link_id, position=location)
        robot.set_joint_positions(q[qIdx], joint_ids)
        world.step()

        if time.perf_counter()-start >= dt:
            break




def get_via_pts():
    print("Move green dot to via point. Press 'A' to save. Press 'X' to finish.")

    X_desired = [-0.75, 0., 0.75]
    speed_scale_factor = 0.01
    via_pts = []

    sphere = None
    for _ in count():
        # print("Via Points: ", via_pts)
        last_trigger = controller.last_updated_button
        movement = controller[last_trigger]

        if last_trigger == 'LJ':
            X_desired[0] = X_desired[0] + round(movement[0], 1) * speed_scale_factor
            X_desired[1] = X_desired[1] + round(movement[1], 1) * speed_scale_factor
        elif last_trigger == 'RJ':
            X_desired[2] = X_desired[2] + round(movement[1], 1) * speed_scale_factor
        elif last_trigger == 'A' and movement == 1:
            print("Via point added ", X_desired)
            via_pts.append(copy.deepcopy(X_desired))
            time.sleep(0.5)
        elif last_trigger == 'X' and movement == 1:
            print("Via point generation complete.")
            break

        if sphere:
            world.sim.remove_body(sphere)
        sphere = world.load_visual_sphere(X_desired, radius=0.01, color=(0, 1, 0, 1))

    return via_pts


def extractPath(traj):
    path = np.empty((len(traj), traj[0].mu.shape[0]))
    for i in range(len(traj)):
        path[i, :] = traj[i].mu.flatten()
    return path

def display_path(pred_path):
    for pt in pred_path:
        world.load_visual_sphere(pt[:3], radius=0.01, color=(1, 0, 0, 1))

def move_robot(pred_path):
    for pt in pred_path:
        dt = kmp.output_dt
        send_robot_to(np.array(pt[:3]), 0.05)


if __name__ == "__main__":
    # Initialize XBox controller
    controller = XboxControllerInterface(use_thread=True, verbose=False)
    # Create simulator
    sim = Bullet()
    # create world
    world = BasicWorld(sim)
    # create robot
    robot = KukaIIWA(sim)
    send_robot_to(np.array([-0.75, 0., 0.75]), dt=2)

    # Load trained kmp
    filehandler = open("./trained_kmps/kmp.obj", 'rb')
    kmp = pkl.load(filehandler)

    # Get via points from user
    start_pt = [[-0.75, 0., 0.75]]
    end_pt = [[0.75, 0., 0.75]]
    via_pts =  get_via_pts() # [[-0.19, -0.65, 0.77], [0.51, -0.44, 0.58]]
    via_pts = start_pt + via_pts + end_pt
    print("Via Points: ", via_pts)
    via_pt_var = 1e-6 * np.eye(kmp.ref_traj[0].sigma.shape[0])

    # Query KMP
    pred_traj = kmp.prediction(via_pts, via_pt_var)
    pred_path = extractPath(pred_traj)

    #Display Path in Pybullet
    print("Displaying Path")
    display_path(pred_path)

    # Move Robot
    print("Moving Robot")
    move_robot(pred_path)




