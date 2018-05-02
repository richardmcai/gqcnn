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
#!/usr/bin/env python
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

from autolab_core import RigidTransform
from autolab_core import YamlConfig
from dexnet.grasping import RobotGripper
from baxter_interface import Limb
from baxter_interface import Gripper

from visualization import Visualizer2D as vis
import perception as perception
from perception import RgbdDetectorFactory, RgbdSensorFactory
from gqcnn import Visualizer as vis

from gqcnn.msg import GQCNNGrasp, BoundingBox
from sensor_msgs.msg import Image, CameraInfo
from gqcnn.srv import GQCNNGraspPlanner

from cv_bridge import CvBridge, CvBridgeError

from gqcnn import GraspIsolatedObjectExperimentLogger

from recieve_images import input_reader


def close_gripper(gripper, force=30.0):
    """closes the gripper"""
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
    """
    if isinstance(pose, RigidTransform)
        pose = pose.pose_msg
    arm.set_start_state_to_current_state()
    arm.set_pose_target(pose)
    arm.set_max_velocity_scaling_factor(v_scale)
    arm.plan()
    arm.go()

def get_pose(limb)
    """Returns a RigidTransform from limb toolframe to base frame"""
    cur_pose = limb.endpoint_pose()
    rotation = np.asarray([cur_pose.orientation.w, cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z])
    translation = np.asarray([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z])    
    T_tool_base = RigidTransform(r_cur_world, t_cur_world, T_gripper_world.from_frame, T_gripper_world.to_frame)
    return T_tool_base

def process_GQCNNGrasp(grasp, robot, left_arm, right_arm, left_gripper, right_gripper, home_pose, config):
    """ Processes a ROS GQCNNGrasp message and executes the resulting grasp on the ABB Yumi """
    grasp = grasp.grasp
    rospy.loginfo('Processing Grasp')

    rotation_quaternion = np.asarray([grasp.pose.orientation.w, grasp.pose.orientation.x, grasp.pose.orientation.y, grasp.pose.orientation.z]) 
    translation = np.asarray([grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z])
    T_grasp_camera = RigidTransform(rotation_quaternion, translation, 'grasp', T_camera_world.from_frame)
    T_gripper_world = T_camera_world * T_grasp_camera * T_gripper_grasp
    
    if not config['robot_off']:
        rospy.loginfo('Executing Grasp!')
        lifted_object, lift_gripper_width, lift_torque = execute_grasp(T_gripper_world, robot, left_arm, right_arm, left_gripper, right_gripper, limb, config)
    
        # bring arm back to home pose 
        go_to_pose(left_arm, home_pose)
        open_gripper(left_gripper)

        return lift_gripper_width, T_gripper_world

def execute_grasp(T_gripper_world, robot, left_arm, right_arm, left_gripper, right_gripper, limb, config):
    """ Executes a single grasp for the hand pose T_gripper_world up to the point of lifting the object """
    # snap gripper to valid depth
    if T_gripper_world.translation[2] < config['grasping']['min_gripper_depth']:
        T_gripper_world.translation[2] = config['grasping']['min_gripper_depth']

    # get cur pose
    T_cur_world = get_pose(limb)

    # compute approach pose
    t_approach_target = np.array([0,0,config['grasping']['approach_dist']])
    T_gripper_approach = RigidTransform(translation=t_approach_target,
                                        from_frame='gripper',
                                        to_frame='gripper')
    T_approach_world = T_gripper_world * T_gripper_approach.inverse()
    t_lift_target = np.array([0,0,config['grasping']['lift_height']])
    T_gripper_lift = RigidTransform(translation=t_lift_target,
                                    from_frame='gripper',
                                    to_frame='gripper')
    T_lift_world = T_gripper_world * T_gripper_lift.inverse()

    # compute lift pose
    t_delta_approach = T_approach_world.translation - T_cur_world.translation

    # perform grasp on the robot, up until the point of lifting
    open_gripper(left_arm)
    go_to_pose(left_arm, YMC.L_KINEMATIC_AVOIDANCE_POSE)
    go_to_pose(left_arm, T_approach_world)

    # grasp
    v_scale = config['control']['approach_velocity']
    if config['control']['test_collision']:
        T_gripper_world.translation[2] = 0.0
        go_to_pose(left_arm, T_gripper_world, v_scale)
        T_cur_gripper_world = get_pose(limb)
        dist_from_goal = np.linalg.norm(T_cur_gripper_world.translation - T_gripper_world.translation)
        collision = False
        while dist_from_goal > 1e-3:
            T_cur_gripper_world = get_pose(limb)
            dist_from_goal = np.linalg.norm(T_cur_gripper_world.translation - T_gripper_world.translation)
            if limb.joint_effort('left_e0') > 0.001: # TODO: identify correct joint
                logging.info('Detected collision!!!!!!')
                go_to_pose(left_arm, T_approach_world, v_scale)
                logging.info('Commanded!!!!!!')
                collision = True
                break
            go_to_pose(left_arm, T_gripper_world, v_scale)
    else:
        go_to_pose(left_arm, T_gripper_world)
    
    # pick up object
    close_gripper(left_gripper, force=config['control']['gripper_close_force'])
    pickup_gripper_width = left_gripper.position() # a percentage
    
    go_to_pose(left_arm, T_lift_world, v_scale=config['control']['standard_velocity'])
    go_to_pose(left_arm, YMC.L_KINEMATIC_AVOIDANCE_POSE, v_scale=config['control']['standard_velocity'])
    go_to_pose(left_arm, YMC.L_PREGRASP_POSE, v_scale=config['control']['standard_velocity'])

    # shake test
    if config['control']['shake_test']:

        # compute shake poses
        radius = config['control']['shake_radius']
        angle = config['control']['shake_angle'] * np.pi
        delta_T = RigidTransform(translation=[0,0,radius], from_frame='gripper', to_frame='gripper')
        R_shake = np.array([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]])
        delta_T_up = RigidTransform(rotation=R_shake, translation=[0,0,-radius], from_frame='gripper', to_frame='gripper')
        delta_T_down = RigidTransform(rotation=R_shake.T, translation=[0,0,-radius], from_frame='gripper', to_frame='gripper')
        T_shake_up = YMC.L_PREGRASP_POSE.as_frames('gripper', 'world') * delta_T_up * delta_T
        T_shake_down = YMC.L_PREGRASP_POSE.as_frames('gripper', 'world') * delta_T_down * delta_T

        v_scale = config['control']['shake_velocity']
        for i in range(config['control']['num_shakes']):
            go_to_pose(left_arm, T_shake_up, v_scale)
            go_to_pose(left_arm, YMC.L_PREGRASP_POSE, v_scale)
            go_to_pose(left_arm, T_shake_down, v_scale)
            go_to_pose(left_arm, YMC.L_PREGRASP_POSE, v_scale)
        v_scale = config['control']['standard_velocity']

    # check gripper width
    lift_torque = limb.joint_effort('left_w0') # TODO: identify correct joint
    lift_gripper_width = left_gripper.position() # a percentage

    # check drops
    lifted_object = False
    if np.abs(lift_gripper_width) > config['grasping']['pickup_min_width']:
        lifted_object = True

    return lifted_object, lift_gripper_width, lift_torque

def init_robot(config):
    """ Initializes the robot """
    robot = None
    limb = None
    initialized = False
    while not initialized:
        try:
            moveit_commander.roscpp_initialize(sys.argv)
            rospy.init_node('moveit_node')
            robot = moveit_commander.RobotCommander()

            left_arm = moveit_commander.MoveGroupCommander('left_arm')
            go_to_pose(left_arm, YMC.L_HOME_STATE)
            home_pose = YMC.L_PREGRASP_POSE

            right_arm = moveit_commander.MoveGroupCommander('right_arm')
            go_to_pose(right_arm, YMC.R_HOME_STATE)

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

    if not config['robot_off']:
        rospy.loginfo('Initializing Baxter')
        robot, limb, left_arm, right_arm, left_gripper, right_gripper, home_pose = init_robot(config)
    
    # create ROS CVBridge
    cv_bridge = CvBridge()
    
    # wait for Grasp Planning Service and create Service Proxy
    rospy.wait_for_service('plan_gqcnn_grasp')
    plan_grasp = rospy.ServiceProxy('plan_gqcnn_grasp', GQCNNGraspPlanner)

    # TODO: get camera intrinsics
    camera_intrinsics = #sensor.ir_intrinsics

    # setup experiment logger

    object_keys = config['test_object_keys']

    while True:
        
        rospy.loginfo('Please place object: ' + obj + ' on the workspace.')
        raw_input("Press ENTER when ready ...")
        # start the next trial

        # get the images from the sensor
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
        inpainted_color_image = color_image.inpaint(rescale_factor=config['inpaint_rescale_factor'])
        inpainted_depth_image = depth_image.inpaint(rescale_factor=config['inpaint_rescale_factor'])

        detector = RgbdDetectorFactory.detector('point_cloud_box')
        detection = detector.detect(inpainted_color_image, inpainted_depth_image, detector_cfg, camera_intrinsics, T_camera_world, vis_foreground=False, vis_segmentation=False
            )[0]

        if config['vis']['vis_detector_output']:
            vis.figure()
            vis.subplot(1,2,1)
            vis.imshow(detection.color_thumbnail)
            vis.subplot(1,2,2)
            vis.imshow(detection.depth_thumbnail)
            vis.show()

        boundingBox = BoundingBox()
        boundingBox.minY = detection.bounding_box.min_pt[0]
        boundingBox.minX = detection.bounding_box.min_pt[1]
        boundingBox.maxY = detection.bounding_box.max_pt[0]
        boundingBox.maxX = detection.bounding_box.max_pt[1]

        try:
            start_time = time.time()
            planned_grasp_data = plan_grasp(inpainted_color_image.rosmsg, inpainted_depth_image.rosmsg, camera_intrinsics.rosmsg, boundingBox)
            grasp_plan_time = time.time() - start_time

            lift_gripper_width, T_gripper_world = process_GQCNNGrasp(planned_grasp_data, robot, left_arm, right_arm, limb, home_pose, config)

            # get human label
            
            # log result
        except rospy.ServiceException as e:
            print e

if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('Baxter_Control_Node')

    # TODO
    config = YamlConfig('./baxter_control_node.yaml')

    # TODO
    # Tf from tool frame to frame centered on point contact pair, i.e point between tip of parallel jaws
    rospy.loginfo('Loading T_gripper_grasp')
    T_gripper_grasp = RigidTransform.load('./T_gripper_grasp.tf')

    # TODO
    rospy.loginfo('Loading T_camera_world')
    T_camera_world = RigidTransform.load('./kinect_to_world.tf')

    detector_cfg = config['detector']

    # create rgbd sensor
    rospy.loginfo('Creating RGBD Sensor')
    sensor = input_reader()
    rospy.loginfo('Sensor Running')

    # setup safe termination
    def handler(signum, frame):
        logging.info('caught CTRL+C, exiting...')        
        if sensor is not None:
            sensor.stop()
        if robot is not None:
            robot.stop()
        exit(0)
    signal.signal(signal.SIGINT, handler)
    
    # run experiment
    run_experiment()

    rospy.spin()
