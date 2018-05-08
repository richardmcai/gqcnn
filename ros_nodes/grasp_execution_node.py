import sys
import rospy
import numpy as np
from copy import deepcopy
from Queue import Queue

from baxter_interface import Gripper
import moveit_commander

from autolab_core import RigidTransform

from geometry_msgs.msg import PoseStamped
from gqcnn.srv import GQCNNGraspExecuter

CFG_PATH = '../cfg/ros_nodes/'

# Poses
L_HOME_STATE = RigidTransform.load(CFG_PATH + 'L_HOME_STATE.tf').pose_msg
R_HOME_STATE = RigidTransform.load(CFG_PATH + 'R_HOME_STATE.tf').pose_msg
L_PREGRASP_POSE = RigidTransform.load(CFG_PATH + 'L_PREGRASP_POSE.tf').pose_msg

CAMERA_MESH_DIM = (0.3,0.05,0.05)

# Grasping params
TABLE_DEPTH = -0.190
GRASP_APPROACH_DIST = 0.1
GRIPPER_CLOSE_FORCE = 30.0 # percentage [0.0, 100.0]

# Velocity params; fractions [0.0,1.0]
MAX_VELOCITY = 1.0

class GraspExecuter(object):
    def __init__(self, max_vscale=1.0):
        self.grasp_queue = Queue()

        initialized = False
        while not initialized:
            try:
                moveit_commander.roscpp_initialize(sys.argv)
                self.robot = moveit_commander.RobotCommander()
                self.scene = moveit_commander.PlanningSceneInterface()

                self.left_arm = moveit_commander.MoveGroupCommander('left_arm')
                self.left_arm.set_max_velocity_scaling_factor(max_vscale)
                self.go_to_pose('left', L_HOME_STATE)

                self.right_arm = moveit_commander.MoveGroupCommander('right_arm')
                self.right_arm.set_max_velocity_scaling_factor(max_vscale)
                self.go_to_pose('right', R_HOME_STATE)

                self.left_gripper = Gripper('left')
                self.left_gripper.set_holding_force(GRIPPER_CLOSE_FORCE)
                self.left_gripper.open()

                self.right_gripper = Gripper('right')
                self.left_gripper.set_holding_force(GRIPPER_CLOSE_FORCE)
                self.right_gripper.open()

                initialized = True
            except rospy.ServiceException as e:
                rospy.logerr(e)

    def queue_grasp(self, req):
        grasp = req.grasp
        rotation_quaternion = np.asarray([grasp.pose.orientation.w, grasp.pose.orientation.x, grasp.pose.orientation.y, grasp.pose.orientation.z]) 
        translation = np.asarray([grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z])
        T_grasp_camera = RigidTransform(rotation_quaternion, translation, 'grasp', T_camera_world.from_frame)
        T_gripper_world = T_camera_world * T_grasp_camera * T_gripper_grasp
        self.grasp_queue.put(T_gripper_world.pose_msg)

        return True

    def execute_grasp(self, grasp):
        if grasp.position.z < TABLE_DEPTH + 0.02:
            grasp.position.z = TABLE_DEPTH + 0.02

        approach = deepcopy(grasp)
        approach.position.z = grasp.position.z + GRASP_APPROACH_DIST

         # perform grasp on the robot, up until the point of lifting
        rospy.loginfo('Approaching')
        self.go_to_pose('left', approach)

        # grasp
        rospy.loginfo('Grasping')
        self.go_to_pose('left', grasp)
        self.left_gripper.close(block=True)
        
        #lift object
        rospy.loginfo('Lifting')
        self.go_to_pose('left', approach)
        
        # Drop in bin
        rospy.loginfo('Going Home')
        self.go_to_pose('left', L_PREGRASP_POSE)
        self.left_gripper.open()

        lift_gripper_width = self.left_gripper.position() # a percentage

        # check drops
        lifted_object = lift_gripper_width > 0.0

        return lifted_object, lift_gripper_width

    def go_to_pose(self, arm, pose):
        """Uses Moveit to go to the pose specified
        Parameters
        ----------
        pose : :obj:`geometry_msgs.msg.Pose` or RigidTransform
            The pose to move to
        max_velocity : fraction of max possible velocity
        """
        if arm == 'left':
            arm = self.left_arm
        else:
            arm = self.right_arm
        arm.set_start_state_to_current_state()
        arm.set_pose_target(pose)
        arm.plan()
        arm.go()
        

if __name__ == '__main__':
        # initialize the ROS node
        rospy.init_node('Grasp_Execution_Server')

        # Tf gripper frame to grasp cannonical frame (y-axis = grasp axis, x-axis = palm axis)
        rospy.loginfo('Loading T_gripper_grasp')
        rotation = RigidTransform.y_axis_rotation(float(np.pi/2))
        T_gripper_grasp = RigidTransform(rotation, from_frame='gripper', to_frame='grasp')

        # load camera tf
        rospy.loginfo('Loading T_camera_world')
        T_camera_world = RigidTransform.load(CFG_PATH + 'kinect_to_world.tf')

        # Initialize Baxter
        rospy.loginfo('Initializing Baxter')
        grasp_executer = GraspExecuter(MAX_VELOCITY)

        # Add collision meshes for moveit planner
        rospy.loginfo('Constructing planning scene')
        camera_pose = PoseStamped()
        camera_pose.header.frame_id = 'world'
        camera_pose.header.stamp = rospy.Time(0)
        camera_pose.pose = T_camera_world.pose_msg
        grasp_executer.scene.add_box('camera', camera_pose, CAMERA_MESH_DIM)

        table_pose = PoseStamped()
        table_pose.header.frame_id = 'world'
        table_pose.header.stamp = rospy.Time(0)
        table_pose.pose.position.z = TABLE_DEPTH
        table_pose.pose.position.x = 0.0
        table_pose.pose.position.y = 0.0
        table_pose.pose.orientation.w = 1.0
        grasp_executer.scene.add_plane('table', table_pose)

        # initialize the service        
        service = rospy.Service('execute_gqcnn_grasp', GQCNNGraspExecuter, grasp_executer.queue_grasp)
        rospy.loginfo('Grasp Execution Server Initialized')

        while True:
            grasp = None
            try:
                grasp = grasp_executer.grasp_queue.get_nowait()
            except:
                continue
            if grasp:
                print 'executing'
                rospy.loginfo('Executing Grasp')
                grasp_executer.execute_grasp(grasp)
