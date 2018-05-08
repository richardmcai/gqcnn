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

from autolab_core import RigidTransform, Point, YamlConfig
import perception as perception
from perception import RgbdDetectorFactory, BinaryImage
from gqcnn import Visualizer as vis

from gqcnn.msg import GQCNNGrasp, BoundingBox
from sensor_msgs.msg import Image, CameraInfo
from gqcnn.srv import GQCNNGraspPlanner, GQCNNGraspExecuter

from cv_bridge import CvBridge, CvBridgeError

### CONFIG ###
CFG_PATH = '../cfg/ros_nodes/'

# Experiment Flow
DEBUG = False
DETECT_OBJECTS = False
VISUALIZE_DETECTOR_OUTPUT = False

# Planning Params
INPAINT_RESCALE_FACTOR = 1.0
BOUNDBOX_MIN_X = 80
BOUNDBOX_MIN_Y = 0
BOUNDBOX_MAX_X = 525
BOUNDBOX_MAX_Y = 260

CIRCLE_RAD = 70
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

########

def run_experiment():
    """ Run the experiment """
    # get the images from the sensor
    previous_grasp = None
    while True:
        rospy.loginfo("Waiting for images")
        start_time = time.time()
        raw_color = rospy.wait_for_message("/camera/rgb/image_color", Image)
        raw_depth = rospy.wait_for_message("/camera/depth_registered/image", Image)
        image_load_time = time.time() - start_time
        rospy.loginfo('Images loaded in: ' + str(image_load_time) + ' secs.')

        ### Create wrapped Perception RGB and Depth Images by unpacking the ROS Images using CVBridge ###
        try:
            color_image = perception.ColorImage(cv_bridge.imgmsg_to_cv2(raw_color, "rgb8"), frame=T_camera_world.from_frame)
            depth_image = perception.DepthImage(cv_bridge.imgmsg_to_cv2(raw_depth, desired_encoding = "passthrough"), frame=T_camera_world.from_frame)
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)
        
        # inpaint to remove holes
        inpainted_color_image = color_image.inpaint(rescale_factor=INPAINT_RESCALE_FACTOR)
        inpainted_depth_image = depth_image.inpaint(rescale_factor=INPAINT_RESCALE_FACTOR)

        if DETECT_OBJECTS:
            detector = RgbdDetectorFactory.detector('point_cloud_box')
            detections = detector.detect(inpainted_color_image, inpainted_depth_image, detector_cfg, camera_intrinsics, T_camera_world, vis_foreground=False, vis_segmentation=False)

        detected_obj = None
        if previous_grasp is not None:
            position = previous_grasp.pose.position
            position = np.array([position.x, position.y, position.z])
            center_point = Point(position, camera_intrinsics.frame)
            center_pixel = camera_intrinsics.project(center_point, camera_intrinsics.frame)
            i, j = center_pixel.y, center_pixel.x
            if DETECT_OBJECTS:
                for detection in detections:
                    binaryIm = detection.binary_im
                    if binaryIm[i,j]: 
                        segmask = binaryIm
                        detected_obj = detection
                        break
            else:
                # Generate an ellipse inverse segmask centered on previous grasp
                y, x = np.ogrid[-i:IMAGE_HEIGHT-i, -j:IMAGE_WIDTH-j]
                circlemask = x*x + y*y <= CIRCLE_RAD*CIRCLE_RAD
                segmask_data = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)*255
                segmask_data[circlemask] = 0
                segmask = BinaryImage(segmask_data, camera_intrinsics.frame)
        else:
            segmask = BinaryImage(np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)*255, camera_intrinsics.frame)
        segmask._encoding = 'mono8'

        if VISUALIZE_DETECTOR_OUTPUT:
            vis.figure()
            vis.subplot(1,2,1)
            vis.imshow(detected_obj.color_thumbnail)
            vis.subplot(1,2,2)
            vis.imshow(detected_obj.depth_thumbnail)
            vis.show()

        try:
            rospy.loginfo('Planning Grasp')
            start_time = time.time()
            planned_grasp_data = plan_grasp(inpainted_color_image.rosmsg, inpainted_depth_image.rosmsg, segmask.rosmsg, raw_camera_info, boundingBox)
            grasp_plan_time = time.time() - start_time
            rospy.loginfo('Total grasp planning time: ' + str(grasp_plan_time) + ' secs.')

            rospy.loginfo('Queueing Grasp')
            previous_grasp = planned_grasp_data.grasp
            execute_grasp(previous_grasp)
            # raw_input("Press ENTER to resume")
        except rospy.ServiceException as e:
            rospy.logerr(e)
            previous_grasp = None
            raw_input("Press ENTER to resume")

if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('Baxter_Control_Node')

    # load detector config
    detector_cfg = YamlConfig(CFG_PATH + 'baxter_control_node.yaml')['detector']

    # load camera tf and intrinsics
    rospy.loginfo('Loading T_camera_world')
    T_camera_world = RigidTransform.load(CFG_PATH + 'kinect_to_world.tf')
    rospy.loginfo("Loading camera intrinsics")
    raw_camera_info = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo)
    camera_intrinsics = perception.CameraIntrinsics(raw_camera_info.header.frame_id, raw_camera_info.K[0], raw_camera_info.K[4], raw_camera_info.K[2], raw_camera_info.K[5], raw_camera_info.K[1], raw_camera_info.height, raw_camera_info.width)

    # initalize image processing objects
    cv_bridge = CvBridge()
    boundingBox = BoundingBox(BOUNDBOX_MIN_X, BOUNDBOX_MIN_Y, BOUNDBOX_MAX_X, BOUNDBOX_MAX_Y)
    
    # wait for Grasp Planning Service and create Service Proxy
    rospy.loginfo("Waiting for planner node")
    rospy.wait_for_service('plan_gqcnn_grasp')
    plan_grasp = rospy.ServiceProxy('plan_gqcnn_grasp', GQCNNGraspPlanner)

    # wait for Grasp Execution Service and create Service Proxy
    if not DEBUG:
        rospy.loginfo("Waiting for execution node")
        rospy.wait_for_service('execute_gqcnn_grasp')
        execute_grasp = rospy.ServiceProxy('execute_gqcnn_grasp', GQCNNGraspExecuter)

    # setup safe termination
    def handler(signum, frame):
        logging.info('caught CTRL+C, exiting...')        
        rospy.loginfo('caught CTRL+C, Aborted!')
        exit(0)
    signal.signal(signal.SIGINT, handler)

    # run experiment
    raw_input("Press ENTER when ready ...")
    run_experiment()
    
    rospy.spin()
