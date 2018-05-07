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
executing the grasps with a Rethink Baxter.
Additionally depends on the dex-net, meshpy, and moveit modules.

Author: Vishal Satish, Richard Cai
"""
import rospy
import tf
import logging
import numpy as np
import signal
import time
import sys

from baxter_interface import Gripper
import moveit_commander

from autolab_core import RigidTransform, YamlConfig
import perception as perception
from perception import RgbdDetectorFactory
# from gqcnn import Visualizer as vis

from gqcnn.msg import GQCNNGrasp, BoundingBox
from sensor_msgs.msg import Image, CameraInfo
from gqcnn.srv import GQCNNGraspPlanner

from cv_bridge import CvBridge, CvBridgeError

### CONFIG ###
CFG_PATH = '../cfg/ros_nodes/'

# Experiment Flow
ENABLE_ROBOT = True
DEBUG = False
VISUALIZE_DETECTOR_OUTPUT = False

# Poses
L_HOME_STATE = RigidTransform.load(CFG_PATH + 'L_HOME_STATE.tf')
R_HOME_STATE = RigidTransform.load(CFG_PATH + 'R_HOME_STATE.tf')
L_PREGRASP_POSE = RigidTransform.load(CFG_PATH + 'L_PREGRASP_POSE.tf')

# Planning Params
INPAINT_RESCALE_FACTOR = 1.0
BOUNDBOX_MIN_X = 0
BOUNDBOX_MIN_Y = 0
BOUNDBOX_MAX_X = 640
BOUNDBOX_MAX_Y = 280
CAMERA_MESH_DIM = (1,1,1)

# Grasping params
MIN_GRIPPER_DEPTH = -0.175
GRASP_APPROACH_DIST = 0.1
GRIPPER_CLOSE_FORCE = 30.0 # percentage [0.0, 100.0]

# Velocity params; fractions [0.0,1.0]
APPROACH_VELOCITY = 0.5
STANDARD_VELOCITY = 1.0

########

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

def init_robot():
    """ Initializes the robot """
    initialized = False
    while not initialized:
        try:
            moveit_commander.roscpp_initialize(sys.argv)
            robot = moveit_commander.RobotCommander()
            scene = moveit_commander.PlanningSceneInterface()

            left_arm = moveit_commander.MoveGroupCommander('left_arm')
            go_to_pose(left_arm, L_HOME_STATE)

            right_arm = moveit_commander.MoveGroupCommander('right_arm')
            go_to_pose(right_arm, R_HOME_STATE)

            left_gripper = Gripper('left')
            left_gripper.set_holding_force(GRIPPER_CLOSE_FORCE)
            left_gripper.open()

            right_gripper = Gripper('right')
            left_gripper.set_holding_force(GRIPPER_CLOSE_FORCE)
            right_gripper.open()

            initialized = True
        except rospy.ServiceException as e:
            print e
    return robot, scene, left_arm, right_arm, left_gripper, right_gripper

def process_GQCNNGrasp(grasp_msg):
    """ Processes a ROS GQCNNGrasp message and executes the resulting grasp on the ABB Yumi """
    rospy.loginfo('Processing Grasp')
    grasp = grasp_msg.grasp

    # compute grasp in world frame
    rotation_quaternion = np.asarray([grasp.pose.orientation.w, grasp.pose.orientation.x, grasp.pose.orientation.y, grasp.pose.orientation.z]) 
    translation = np.asarray([grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z])
    T_grasp_camera = RigidTransform(rotation_quaternion, translation, 'grasp', T_camera_world.from_frame)
    T_gripper_world = T_camera_world * T_grasp_camera * T_gripper_grasp

    # execute grasp
    lift_gripper_width = None
    if ENABLE_ROBOT:
        rospy.loginfo('Executing Grasp!')
        lifted_object, lift_gripper_width = execute_grasp(T_gripper_world)
    
    return lift_gripper_width, T_gripper_world

def execute_grasp(T_gripper_world):
    """ Executes a single grasp for the hand pose T_gripper_world up to the point of lifting the object """
    # snap gripper to valid depth
    if T_gripper_world.translation[2] < MIN_GRIPPER_DEPTH:
        T_gripper_world.translation[2] = MIN_GRIPPER_DEPTH

    # compute approach pose
    t_approach_target = np.array([0,0,GRASP_APPROACH_DIST])
    T_gripper_approach = RigidTransform(translation=t_approach_target, from_frame='gripper', to_frame='gripper')
    T_approach_world = T_gripper_world * T_gripper_approach.inverse()

    # perform grasp on the robot, up until the point of lifting
    rospy.loginfo('Approaching')
    go_to_pose(left_arm, T_approach_world, v_scale=STANDARD_VELOCITY)

    # grasp
    rospy.loginfo('Grasping')
    go_to_pose(left_arm, T_gripper_world, v_scale=APPROACH_VELOCITY)
    left_gripper.close()
    
    #lift object
    rospy.loginfo('Lifting')
    go_to_pose(left_arm, T_approach_world, v_scale=STANDARD_VELOCITY)
    
    # Drop in bin
    rospy.loginfo('Going Home')
    go_to_pose(left_arm, L_PREGRASP_POSE, v_scale=STANDARD_VELOCITY)
    left_gripper.open()

    # record gripper width
    lift_gripper_width = left_gripper.position() # a percentage

    # check drops
    lifted_object = lift_gripper_width > 0.0

    return lifted_object, lift_gripper_width

def run_experiment():
    """ Run the experiment """
    # get the images from the sensor
    rospy.loginfo("Waiting for images")
    raw_color = rospy.wait_for_message("/camera/rgb/image_color", Image)
    raw_depth = rospy.wait_for_message("/camera/depth_registered/image", Image)

    ### Create wrapped Perception RGB and Depth Images by unpacking the ROS Images using CVBridge ###
    try:
        color_image = perception.ColorImage(cv_bridge.imgmsg_to_cv2(raw_color, "rgb8"), frame=T_camera_world.from_frame)
        depth_image = perception.DepthImage(cv_bridge.imgmsg_to_cv2(raw_depth, desired_encoding = "passthrough"), frame=T_camera_world.from_frame)
    except CvBridgeError as cv_bridge_exception:
        rospy.logerr(cv_bridge_exception)
    
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

    try:
        rospy.loginfo('Planning Grasp')
        start_time = time.time()
        planned_grasp_data = plan_grasp(inpainted_color_image.rosmsg, inpainted_depth_image.rosmsg, camera_intrinsics, boundingBox)
        grasp_plan_time = time.time() - start_time

        lift_gripper_width, T_gripper_world = process_GQCNNGrasp(planned_grasp_data)

        if DEBUG:
            T_gripper_world.publish_to_ros()
    except rospy.ServiceException as e:
        print e

if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('Baxter_Control_Node')

    # load detector config
    detector_cfg = YamlConfig(CFG_PATH + 'baxter_control_node.yaml')['detector']

    # Tf gripper frame to grasp cannonical frame (y-axis = grasp axis, x-axis = palm axis)
    rospy.loginfo('Loading T_gripper_grasp')
    rotation = RigidTransform.y_axis_rotation(float(np.pi/2))
    T_gripper_grasp = RigidTransform(rotation, from_frame='gripper', to_frame='grasp')

    # load camera tf and intrinsics
    rospy.loginfo('Loading T_camera_world')
    T_camera_world = RigidTransform.load(CFG_PATH + 'kinect_to_world.tf')
    rospy.loginfo("Loading camera intrinsics")
    camera_intrinsics = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo)

    # initialize robot
    if ENABLE_ROBOT:
        rospy.loginfo('Initializing Baxter')
        robot, scene, left_arm, right_arm, left_gripper, right_gripper = init_robot()

        # Add collision meshes for moveit planner
        scene.add_box('camera', T_camera_world.pose_msg, CAMERA_MESH_DIM)
        t_world_table = np.array([0,0,MIN_GRIPPER_DEPTH])
        T_table_world = RigidTransform(translation=t_world_table)
        scene.add_plane('table', T_table_world.pose_msg)

    # initalize image processing objects
    cv_bridge = CvBridge()
    boundingBox = BoundingBox(BOUNDBOX_MIN_X, BOUNDBOX_MIN_Y, BOUNDBOX_MAX_X, BOUNDBOX_MAX_Y)
    
    # wait for Grasp Planning Service and create Service Proxy
    rospy.loginfo("Waiting for planner node")
    rospy.wait_for_service('plan_gqcnn_grasp')
    plan_grasp = rospy.ServiceProxy('plan_gqcnn_grasp', GQCNNGraspPlanner)

    # setup safe termination
    def handler(signum, frame):
        logging.info('caught CTRL+C, exiting...')        
        if robot is not None:
            robot.stop()
        exit(0)
    signal.signal(signal.SIGINT, handler)

    # run experiment
    raw_input("Press ENTER when ready ...")
    while True:
        run_experiment()
    
    rospy.spin()
