#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image


class input_reader:

	def __init__(self):
		self.depth_image = None
		self.rgb_image = None
		self.depth_sub = rospy.Subscriber("/camera/depth/image", Image, self.readDepth)
		self.rgb_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.readRGB)

	def readDepth(self, data):
		if data is not None:
			self.depth_image = data
		else:
			print("No data recieved from depth image!")

	def readRGB(self, data):
		if data is not None:
			self.rgb_image = data
		else:
			print("No data recieved from RGB image!")
