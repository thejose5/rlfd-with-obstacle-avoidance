import numpy as np
from itertools import count

from pyrobolearn.tools.interfaces.controllers.xbox import XboxControllerInterface
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA

X_desired = np.array([0.,0.,1.3])
speed_scale_factor = 0.05

if __name__ == "__main__":

    # Initialize XBox controller
    controller = XboxControllerInterface(use_thread=True, verbose=False)

    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = KukaIIWA(sim)
    robot.print_info()

    # define useful variables for IK
    dt = 1. / 240
    link_id = robot.get_end_effector_ids(end_effector=0)
    joint_ids = robot.joints  # actuated joint
    # joint_ids = joint_ids[2:]
    damping = 0.01  # for damped-least-squares IK
    wrt_link_id = -1  # robot.get_link_ids('iiwa_link_1')

    # change the robot visual
    # robot.change_transparency()
    robot.draw_link_frames(link_ids=[-1, 0])
    robot.draw_bounding_boxes(link_ids=joint_ids[0])
    # robot.draw_link_coms([-1,0])

    qIdx = robot.get_q_indices(joint_ids)
    sphere = None
    for t in count():
        controller.step()

        ## Update target EE location
        last_trigger = controller.last_updated_button
        movement = controller[last_trigger]
        # print("Last updated button: {} with value: {}".format(last_trigger, movement))
        if last_trigger == 'LJ':
            X_desired[0] = X_desired[0] + round(movement[0], 1) * speed_scale_factor
            X_desired[1] = X_desired[1] + round(movement[1], 1) * speed_scale_factor
        elif last_trigger == 'RJ':
            X_desired[2] = X_desired[2] + round(movement[1], 1) * speed_scale_factor

        if sphere:
            world.sim.remove_body(sphere)
        sphere = world.load_visual_sphere(X_desired, radius=0.005, color=(1, 0, 0, 0.5))


        ## Move EE
        # get current position in the task/operational space
        # x = robot.get_link_world_positions(link_id)
        # x = robot.sim.get_link_world_positions(body_id=robot.id, link_ids=link_id)
        # # print("(xd - x) = {}".format(xd - x))
        #
        # # perform full IK
        # q = robot.calculate_inverse_kinematics(link_id, position=X_desired)
        #
        # # set the joint positions
        # robot.set_joint_positions(q[qIdx], joint_ids)
        #
        # # step in simulation
        # world.step(sleep_dt=dt)
        #


        kp = 50  # 5 if velocity control, 50 if position control
        kd = 0  # 2*np.sqrt(kp)

        # get current position in the task/operational space
        # x = robot.get_link_world_positions(link_id)
        x = robot.sim.get_link_world_positions(body_id=robot.id, link_ids=link_id)
        dx = robot.get_link_world_linear_velocities(link_id)
        # print("(xd - x) = {}".format(xd - x))

        # Get joint configuration
        # q = robot.get_joint_positions()
        q = robot.sim.get_joint_positions(robot.id, robot.joints)

        # Get linear jacobian
        if robot.has_floating_base():
            J = robot.get_linear_jacobian(link_id, q=q)[:, qIdx + 6]
        else:
            J = robot.get_linear_jacobian(link_id, q=q)[:, qIdx]

        # Pseudo-inverse
        # Jp = robot.get_pinv_jacobian(J)
        # Jp = J.T.dot(np.linalg.inv(J.dot(J.T) + damping*np.identity(3)))   # this also works
        Jp = robot.get_damped_least_squares_inverse(J, damping)

        # evaluate damped-least-squares IK
        # print("Error: ", (X_desired - x))
        dq = Jp.dot(kp * (X_desired - x) - kd * dx)

        # set joint velocities
        # robot.set_joint_velocities(dq)

        # set joint positions
        q = q[qIdx] + dq * dt
        robot.set_joint_positions(q, joint_ids=joint_ids)

        # step in simulation
        world.step(sleep_dt=dt)

        if not t%10:
            print("Location of EE: ", x)

