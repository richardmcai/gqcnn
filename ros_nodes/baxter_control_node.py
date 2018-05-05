#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""

"""
Example node for planning grasps from point clouds using the gqcnn module and
executing the grasps with an ABB YuMi.
Additionally depends on the dex-net, meshpy, and yumipy modules.

This file is intended as an example, not as code that will run with standard installation.

Author: Vishal Satish
"""
import rospy
import logging
import numpy as np
import signal
import time
import sys

from autolab_core import RigidTransform
from autolab_core import YamlConfig
from baxter_interface import Limb
from baxter_interface import Gripper
import moveit_commander

# from visualization import Visualizer2D as vis
import perception as perception
from perception import RgbdDetectorFactory
# from gqcnn import Visualizer as vis

from gqcnn.msg import GQCNNGrasp, BoundingBox
from sensor_msgs.msg import Image, CameraInfo
from gqcnn.srv import GQCNNGraspPlanner

from cv_bridge import CvBridge, CvBridgeError

from recieve_images import input_reader

### CONFIG ###
CFG_PATH = '../cfg/ros_nodes/'

# Experiment Flow
ENABLE_ROBOT = True
SHAKE_TEST = False
TEST_COLLISION = False
VISUALIZE_DETECTOR_OUTPUT = False

# Poses; RigidTransform from tool to base
L_HOME_STATE = RigidTransform.load(CFG_PATH + 'L_HOME_STATE.tf')
R_HOME_STATE = RigidTransform.load(CFG_PATH + 'R_HOME_STATE.tf')
L_PREGRASP_POSE = RigidTransform.load(CFG_PATH + 'L_PREGRASP_POSE.tf')
L_KINEMATIC_AVOIDANCE_POSE = RigidTransform.load(CFG_PATH + 'L_KINEMATIC_AVOIDANCE_POSE.tf')

# Grasping params
MIN_GRIPPER_DEPTH = 0.0125
GRASP_APPROACH_DIST = 0.075
GRASP_LIFT_HEIGHT = 0.1
GRASP_PICKUP_MIN_WIDTH = 0.0001
GRIPPER_CLOSE_FORCE = 30.0 # percentage [0.0, 100.0]

# Velocity params; fractions [0.0,1.0]
APPROACH_VELOCITY = 0.2#0.5
STANDARD_VELOCITY = 0.2#1.0
SHAKE_VELOCITY = 0.2#1.0

# Shake config
SHAKE_RADIUS = 0.2
SHAKE_ANGLE = 0.03
NUM_SHAKES = 3

# Visualize configs
INPAINT_RESCALE_FACTOR = 1.0

########

def close_gripper(gripper, force=30.0):
    """closes the gripper; force is a percentage of max force"""
    gripper.set_holding_force(force)
    gripper.close(block=True)
    rospy.sleep(1.0)

def open_gripper(gripper):
    """opens the gripper"""
    gripper.open(block=True)
    rospy.sleep(1.0)

def go_to_pose(arm, pose, v_scale=1.0):
    """Uses Moveit to go to the pose specified
    Parameters
    ----------
    pose : :obj:`geometry_msgs.msg.Pose` or RigidTransform
        The pose to move to
    v_scale : fraction of max possible velocity
    """
    if isinstance(pose, RigidTransform):
        pose = pose.pose_msg
    arm.set_start_state_to_current_state()
    arm.set_pose_target(pose)
    arm.set_max_velocity_scaling_factor(v_scale)
    arm.plan()
    arm.go()

def get_pose(limb):
    """Returns a RigidTransform from limb toolframe to base frame"""
    cur_pose = limb.endpoint_pose()
    orientation, position = cur_pose['orientation'], cur_pose['position']
    rotation = np.asarray([orientation.w, orientation.x, orientation.y, orientation.z])
    translation = np.asarray([position.x, position.y, position.z])    
    T_gripper_world = RigidTransform(rotation, translation, 'gripper', 'world')
    return T_gripper_world

def process_GQCNNGrasp(grasp, robot, left_arm, right_arm, left_gripper, right_gripper, limb, home_pose, config):
    """ Processes a ROS GQCNNGrasp message and executes the resulting grasp on the ABB Yumi """
    grasp = grasp.grasp
    rospy.loginfo('Processing Grasp')

    rotation_quaternion = np.asarray([grasp.pose.orientation.w, grasp.pose.orientation.x, grasp.pose.orientation.y, grasp.pose.orientation.z]) 
    translation = np.asarray([grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z])
    T_grasp_camera = RigidTransform(rotation_quaternion, translation, 'grasp', T_camera_world.from_frame)
    T_gripper_world = T_camera_world * T_grasp_camera * T_gripper_grasp
    
    if ENABLE_ROBOT:
        rospy.loginfo('Executing Grasp!')
        lifted_object, lift_gripper_width, lift_torque = execute_grasp(T_gripper_world, robot, left_arm, right_arm, left_gripper, right_gripper, limb, config)
    
        # bring arm back to home pose 
        rospy.loginfo('Going Home')
        go_to_pose(left_arm, home_pose)
        open_gripper(left_gripper)

        return lift_gripper_width, T_gripper_world

def execute_grasp(T_gripper_world, robot, left_arm, right_arm, left_gripper, right_gripper, limb, config):
    """ Executes a single grasp for the hand pose T_gripper_world up to the point of lifting the object """
    # snap gripper to valid depth
    if T_gripper_world.translation[2] < MIN_GRIPPER_DEPTH:
        T_gripper_world.translation[2] = MIN_GRIPPER_DEPTH

    # get cur pose
    T_cur_world = get_pose(limb)

    # compute approach pose
    t_approach_target = np.array([0,0,GRASP_APPROACH_DIST])
    T_gripper_approach = RigidTransform(translation=t_approach_target,
                                        from_frame='gripper',
                                        to_frame='gripper')
    T_approach_world = T_gripper_world * T_gripper_approach.inverse()
    t_lift_target = np.array([0,0,GRASP_LIFT_HEIGHT])
    T_gripper_lift = RigidTransform(translation=t_lift_target,
                                    from_frame='gripper',
                                    to_frame='gripper')
    T_lift_world = T_gripper_world * T_gripper_lift.inverse()

    # compute lift pose
    t_delta_approach = T_approach_world.translation - T_cur_world.translation

    # perform grasp on the robot, up until the point of lifting
    open_gripper(left_gripper)
    rospy.loginfo('Going Kin avoid')
    go_to_pose(left_arm, L_KINEMATIC_AVOIDANCE_POSE)
    rospy.loginfo('going approach')
    go_to_pose(left_arm, T_approach_world)

    # grasp
    rospy.loginfo('going grasp')
    if TEST_COLLISION:
        T_gripper_world.translation[2] = 0.0
        go_to_pose(left_arm, T_gripper_world, v_scale=APPROACH_VELOCITY)
        T_cur_gripper_world = get_pose(limb)
        dist_from_goal = np.linalg.norm(T_cur_gripper_world.translation - T_gripper_world.translation)
        collision = False
        while dist_from_goal > 1e-3:
            T_cur_gripper_world = get_pose(limb)
            dist_from_goal = np.linalg.norm(T_cur_gripper_world.translation - T_gripper_world.translation)
            if limb.joint_effort('left_e0') > 0.001: # TODO: identify correct joint
                logging.info('Detected collision!!!!!!')
                go_to_pose(left_arm, T_approach_world, v_scale=APPROACH_VELOCITY)
                logging.info('Commanded!!!!!!')
                collision = True
                break
            go_to_pose(left_arm, T_gripper_world, v_scale=APPROACH_VELOCITY)
    else:
        go_to_pose(left_arm, T_gripper_world)
    
    # pick up object
    rospy.loginfo('grasping')
    close_gripper(left_gripper, force=GRIPPER_CLOSE_FORCE)
    pickup_gripper_width = left_gripper.position() # a percentage
    
    rospy.loginfo('lifting')
    go_to_pose(left_arm, T_lift_world, v_scale=STANDARD_VELOCITY)
    rospy.loginfo('return to kin avoid')
    go_to_pose(left_arm, L_KINEMATIC_AVOIDANCE_POSE, v_scale=STANDARD_VELOCITY)
    
    rospy.loginfo('going home')
    go_to_pose(left_arm, L_PREGRASP_POSE, v_scale=STANDARD_VELOCITY)

    # shake test
    if SHAKE_TEST:
        # compute shake poses
        radius = SHAKE_RADIUS
        angle = SHAKE_ANGLE * np.pi
        delta_T = RigidTransform(translation=[0,0,radius], from_frame='gripper', to_frame='gripper')
        R_shake = np.array([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]])
        delta_T_up = RigidTransform(rotation=R_shake, translation=[0,0,-radius], from_frame='gripper', to_frame='gripper')
        delta_T_down = RigidTransform(rotation=R_shake.T, translation=[0,0,-radius], from_frame='gripper', to_frame='gripper')
        T_shake_up = L_PREGRASP_POSE.as_frames('gripper', 'world') * delta_T_up * delta_T
        T_shake_down = L_PREGRASP_POSE.as_frames('gripper', 'world') * delta_T_down * delta_T

        for i in range(NUM_SHAKES):
            go_to_pose(left_arm, T_shake_up, v_scale=SHAKE_VELOCITY)
            go_to_pose(left_arm, L_PREGRASP_POSE, v_scale=SHAKE_VELOCITY)
            go_to_pose(left_arm, T_shake_down, v_scale=SHAKE_VELOCITY)
            go_to_pose(left_arm, L_PREGRASP_POSE, v_scale=SHAKE_VELOCITY)

    # check gripper width
    lift_torque = limb.joint_effort('left_w0') # TODO: identify correct joint
    lift_gripper_width = left_gripper.position() # a percentage

    # check drops
    lifted_object = False
    if np.abs(lift_gripper_width) > GRASP_PICKUP_MIN_WIDTH:
        lifted_object = True

    return lifted_object, lift_gripper_width, lift_torque

def init_robot(config):
    """ Initializes the robot """
    limb = None
    initialized = False
    while not initialized:
        try:
            moveit_commander.roscpp_initialize(sys.argv)
            robot = moveit_commander.RobotCommander()

            left_arm = moveit_commander.MoveGroupCommander('left_arm')
            go_to_pose(left_arm, L_HOME_STATE)
            home_pose = L_PREGRASP_POSE

            right_arm = moveit_commander.MoveGroupCommander('right_arm')
            go_to_pose(right_arm, R_HOME_STATE)

            left_gripper = Gripper('left')
            open_gripper(left_gripper)

            right_gripper = Gripper('right')
            open_gripper(right_gripper)

            limb = Limb('left')

            initialized = True
        except rospy.ServiceException as e:
            print e
    return robot, limb, left_arm, right_arm, left_gripper, right_gripper, home_pose

def run_experiment():
    """ Run the experiment """

    if ENABLE_ROBOT:
        rospy.loginfo('Initializing Baxter')
        robot, limb, left_arm, right_arm, left_gripper, right_gripper, home_pose = init_robot(config)
    
    # create ROS CVBridge
    cv_bridge = CvBridge()
    
    # wait for Grasp Planning Service and create Service Proxy
    rospy.wait_for_service('plan_gqcnn_grasp')
    plan_grasp = rospy.ServiceProxy('plan_gqcnn_grasp', GQCNNGraspPlanner)

    camera_intrinsics = perception.PrimesenseSensor().ir_intrinsics
    camera_intrinsics._frame = T_camera_world.from_frame

    # setup experiment logger

    object_keys = config['test_object_keys']

    while True:
        
        # rospy.loginfo('Please place object: ' + obj + ' on the workspace.')
        raw_input("Press ENTER when ready ...")
        # start the next trial

        # get the images from the sensor
        raw_color, raw_depth = None, None
        while raw_color == None or raw_depth == None:
            raw_color = sensor.rgb_image
            raw_depth = sensor.depth_image

        ### Create wrapped Perception RGB and Depth Images by unpacking the ROS Images using CVBridge ###
        try:
            color_image = perception.ColorImage(cv_bridge.imgmsg_to_cv2(raw_color, "rgb8"), frame=camera_intrinsics.frame)
            depth_image = perception.DepthImage(cv_bridge.imgmsg_to_cv2(raw_depth, desired_encoding = "passthrough"), frame=camera_intrinsics.frame)
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)
        
        # log some trial info        

        # inpaint to remove holes
        inpainted_color_image = color_image.inpaint(rescale_factor=INPAINT_RESCALE_FACTOR)
        inpainted_depth_image = depth_image.inpaint(rescale_factor=INPAINT_RESCALE_FACTOR)

        # detector = RgbdDetectorFactory.detector('point_cloud_box')
        # detection = detector.detect(inpainted_color_image, inpainted_depth_image, detector_cfg, camera_intrinsics, T_camera_world, vis_foreground=False, vis_segmentation=False
        #     )[0]

        if VISUALIZE_DETECTOR_OUTPUT:
            vis.figure()
            vis.subplot(1,2,1)
            vis.imshow(detection.color_thumbnail)
            vis.subplot(1,2,2)
            vis.imshow(detection.depth_thumbnail)
            vis.show()

        boundingBox = BoundingBox()
        # boundingBox.minY = detection.bounding_box.min_pt[0]
        # boundingBox.minX = detection.bounding_box.min_pt[1]
        # boundingBox.maxY = detection.bounding_box.max_pt[0]
        # boundingBox.maxX = detection.bounding_box.max_pt[1]

        boundingBox.minY = 30 * INPAINT_RESCALE_FACTOR
        boundingBox.minX = 50 * INPAINT_RESCALE_FACTOR
        boundingBox.maxY = 240 * INPAINT_RESCALE_FACTOR
        boundingBox.maxX = 590 * INPAINT_RESCALE_FACTOR

        try:
            start_time = time.time()
            planned_grasp_data = plan_grasp(inpainted_color_image.rosmsg, inpainted_depth_image.rosmsg, camera_intrinsics.rosmsg, boundingBox)
            grasp_plan_time = time.time() - start_time

            lift_gripper_width, T_gripper_world = process_GQCNNGrasp(planned_grasp_data, robot, left_arm, right_arm, left_gripper, right_gripper, limb, home_pose, config)

            # get human label
            
            # log result
        except rospy.ServiceException as e:
            print e

if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('Baxter_Control_Node')

    config = YamlConfig(CFG_PATH + 'baxter_control_node.yaml')

    # Tf gripper frame to grasp cannonical frame (y-axis = grasp axis, x-axis = palm axis)
    # On Baxter this is just the identity
    rospy.loginfo('Loading T_gripper_grasp')
    T_gripper_grasp = RigidTransform(from_frame='gripper', to_frame='grasp')

    # TODO: write a script to calibrate this automatically
    rospy.loginfo('Loading T_camera_world')
    T_camera_world = RigidTransform.load(CFG_PATH + 'kinect_to_world.tf')

    detector_cfg = config['detector']

    # create rgbd sensor
    rospy.loginfo('Creating RGBD Sensor')
    sensor = input_reader()
    rospy.loginfo('Sensor Running')

    robot = None
    # setup safe termination
    def handler(signum, frame):
        logging.info('caught CTRL+C, exiting...')        
        if robot is not None:
            robot.stop()
        exit(0)
    signal.signal(signal.SIGINT, handler)
    
    # run experiment
    run_experiment()

    rospy.spin()
