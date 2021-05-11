import numpy as np
import os, time, csv
from itertools import count

from pyrobolearn.tools.interfaces.controllers.xbox import XboxControllerInterface
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA


# #################### CLASS Robot ###################
class Robot:
    def __init__(self, sim, world):
        self.robot = KukaIIWA(sim)
        self.world = world

        # change the robot visual
        self.robot.change_transparency()
        self.robot.draw_link_frames(link_ids=[-1, 0])

        self.start_pos = np.array([-0.75, 0., 0.75])
        self.end_pos = np.array([0.75, 0., 0.75])
        self.pid = {'kp': 100, 'kd': 0}

        self.ee_id = self.robot.get_end_effector_ids(end_effector=0)

    def send_robot_to(self, location):
        # define useful variables for IK
        dt = 1. / 240
        link_id = self.robot.get_end_effector_ids(end_effector=0)
        joint_ids = self.robot.joints  # actuated joint
        # joint_ids = joint_ids[2:]
        damping = 0.01  # for damped-least-squares IK
        qIdx = self.robot.get_q_indices(joint_ids)

        # for t in count():
        # get current position in the task/operational space
        x = self.robot.sim.get_link_world_positions(body_id=self.robot.id, link_ids=link_id)
        dx = self.robot.get_link_world_linear_velocities(link_id)

        # Get joint configuration
        q = self.robot.sim.get_joint_positions(self.robot.id, self.robot.joints)

        # Get linear jacobian
        if self.robot.has_floating_base():
            J = self.robot.get_linear_jacobian(link_id, q=q)[:, qIdx + 6]
        else:
            J = self.robot.get_linear_jacobian(link_id, q=q)[:, qIdx]

        # Pseudo-inverse
        Jp = self.robot.get_damped_least_squares_inverse(J, damping)

        # evaluate damped-least-squares IK
        dq = Jp.dot(self.pid['kp'] * (location - x) - self.pid['kd'] * dx)

        # set joint velocities
        # robot.set_joint_velocities(dq)

        # set joint positions
        q = q[qIdx] + dq * dt
        self.robot.set_joint_positions(q, joint_ids=joint_ids)

        return x


    def go_home(self):
        for t in count():
            x = self.send_robot_to(self.start_pos)

            # step in simulation
            self.world.step()

            # Check if robot has reached the required position
            error = np.linalg.norm(self.start_pos - x)
            if error < 0.01 or t > 500:
                break

    def go_to_end(self):
        self.send_robot_to(self.end_pos)


# #################### /CLASS Robot ###################

# #################### CLASS ENV ###################

class DemoRecEnv:
    def __init__(self, demo_dt, demo_dir):

        # Initialize XBox controller
        self.controller = XboxControllerInterface(use_thread=True, verbose=False)

        # Create simulator
        self.sim = Bullet()

        # create world
        self.world = BasicWorld(self.sim)

        # Create Robot
        self.robot = Robot(self.sim, self.world)
        self.robot.go_home()

        self.workspace_outer_bound = {'rad': 1.0}
        self.workspace_inner_bound = {'rad': 0.25, 'height': 0.7}

        self.env_objects = []
        self.demo_dir = demo_dir
        self.demo_dt = demo_dt

    def init_task_recorder(self):
        num_viapts = 2  # int(input("How many via points?"))
        via_pts = []

        for i in range(num_viapts):
            pt = generate_pt(via_pts, self.workspace_outer_bound, self.workspace_inner_bound)
            via_pts.append(pt)
            self.env_objects.append(self.world.load_visual_sphere(pt, radius=0.05, color=(1, 0, 0, 1)))

        print("Via-pts generated.")
        print("Press 'start' to start/stop recording demonstrations,select to end training and Y to reset via pts")
        while True:
            last_trigger = self.controller.last_updated_button
            trigger_value = self.controller[last_trigger]

            if last_trigger == 'menu' and trigger_value == 1:
                self.start_recording()
            elif last_trigger == 'view' and trigger_value == 1:
                print("Ending Training")
                self.reset_env()
                break
            elif last_trigger == 'Y' and trigger_value == 1:
                for obj in self.env_objects:
                    self.world.sim.remove_body(obj)
                self.init_task_recorder()
        return

    def start_recording(self):
        time.sleep(1)
        rec_traj = []
        print("Started Recording. Press 'start' again to stop recording and 'A' to send robot to end")

        sphere = None
        X_desired = self.robot.robot.sim.get_link_world_positions(body_id=self.robot.robot.id,
                                                                  link_ids=self.robot.ee_id)
        speed_scale_factor = 0.01
        time_ind = 0
        save_every = 3

        for t in count():
            last_trigger = self.controller.last_updated_button
            trigger_value = self.controller[last_trigger]
            movement = self.controller[last_trigger]

            if last_trigger == 'menu' and trigger_value == 1:
                print("Stopping recording and resetting environment")
                self.save_traj(rec_traj)
                self.reset_env()
            elif last_trigger == 'A' and trigger_value == 1:
                X_desired = self.robot.end_pos
            # Update target robot location from controller input
            elif last_trigger == 'LJ':
                X_desired[0] = X_desired[0] + round(movement[0], 1) * speed_scale_factor
                X_desired[1] = X_desired[1] + round(movement[1], 1) * speed_scale_factor
            elif last_trigger == 'RJ':
                X_desired[2] = X_desired[2] + round(movement[1], 1) * speed_scale_factor

            # Record points every 10 iterations
            if not t % save_every:
                x = self.robot.robot.sim.get_link_world_positions(body_id=self.robot.robot.id,
                                                                  link_ids=self.robot.ee_id)
                dx = self.robot.robot.get_link_world_linear_velocities(self.robot.ee_id)
                rec_traj.append((time_ind * self.demo_dt,) + tuple(round(i, 3) for i in x) + tuple(round(i, 3) for i in dx))
                time_ind = time_ind+1

            # Visualize target EE location
            if sphere:
                self.world.sim.remove_body(sphere)
            sphere = self.world.load_visual_sphere(X_desired, radius=0.01, color=(0, 1, 0, 1))

            # Send robot to target location
            self.robot.send_robot_to(X_desired)

            self.world.step()

    def reset_env(self):
        self.robot.go_home()
        self.remove_objects()
        self.init_task_recorder()

    def remove_objects(self):
        for obj in self.env_objects:
            self.world.sim.remove_body(obj)

    def step_env(self):
        self.controller.step()
        self.world.step()

    def save_traj(self, rec_traj):
        # Finding the index of last file saved
        print(rec_traj[:10])
        listdir = os.listdir(self.demo_dir)
        listdir = list(map(int, listdir))
        listdir.sort()
        # print(listdir)
        ind_last_file = 0 if len(listdir) == 0 else int(listdir[-1])
        # print(ind_last_file)
        del (rec_traj[0])
        with open(self.demo_dir + str(ind_last_file + 1), 'w') as f:
            write = csv.writer(f)
            write.writerows(rec_traj)


# #################### /CLASS ENV ###################

def generate_pt(via_pts, outer_bounds, inner_bounds):
    pt = None
    if via_pts:
        while True:
            x = np.random.uniform(via_pts[-1][0], outer_bounds['rad'])
            y = np.random.uniform(-outer_bounds['rad'], 0)
            z = np.random.uniform(0, outer_bounds['rad'])
            if is_within_bounds(np.array([x, y, z]), outer_bounds, inner_bounds):
                pt = np.array([x, y, z])
                break
    else:
        while True:
            x = np.random.uniform(-outer_bounds['rad'], outer_bounds['rad'])
            y = np.random.uniform(-outer_bounds['rad'], 0)
            z = np.random.uniform(0, outer_bounds['rad'])
            if is_within_bounds(np.array([x, y, z]), outer_bounds, inner_bounds):
                pt = np.array([x, y, z])
                break
    return pt


def is_within_bounds(pt, outer_bounds, inner_bounds):
    if np.linalg.norm(pt) < outer_bounds['rad']:
        if pt[2] > inner_bounds['height'] or np.linalg.norm(pt[:2]) > inner_bounds['rad']:
            return True
        else:
            return False
    else:
        return False


if __name__ == "__main__":
    env = DemoRecEnv(demo_dt=0.01, demo_dir="./test_delete_later/")
    env.init_task_recorder()
    # train_kmp()
    # query_kmp()
