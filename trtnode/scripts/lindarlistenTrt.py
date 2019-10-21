#!/usr/bin/env python
#-*-coding:UTF-8-*-
from __future__ import print_function
 
import sys
import rospy
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
import cv2
from geometry_msgs.msg import Pose2D
from lgsvl_msgs.msg import Detection2D
from lgsvl_msgs.msg import Detection2DArray
from lgsvl_msgs.msg import BoundingBox2D
from myColormap import Myclasses
global runStart
runStart = 0
global Detections
Detections = []
def callback(data):
    #global process_this_frame,boxes,scores,classes,num_detections
    global Detections
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    image = np.array(image)
    im_h = image.shape[0]
    im_w = image.shape[1]
    try:
        for i in Detections:
            if i.id==1:
                left = int(i.bbox.left)
                top = int(i.bbox.top)
                right = int(i.bbox.right)
                bottom = int(i.bbox.bottom)
                image = cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 3)
                # display class index and score
                cv2.rectangle(image, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                try:
                    cv2.putText(image, categories[int(classes[i] - 1)]['name'], (left, bottom), font, 1.0,
                                (255, 255, 255), 1)
                except:
                    print('Class i is ' + str(classes[i]))
    except:
        pass
    cv2.imshow('trt',image)
    cv2.waitKey(3)
def runDetect(dataNum):
    global runStart,Detections
    Detections = dataNum.detections

    Distance = []
    BoxList = []
    for i in Detections:
        if i.id == 1:
            left = int(i.bbox.left)
            top = int(i.bbox.top)
            right = int(i.bbox.right)
            bottom = int(i.bbox.bottom)
            centerIsX = abs(int(right - left) / 2)
            centerIsY = abs(int(bottom - top) / 2)
            distancePoint = abs(centerIsX - 320)
            Distance.append(distancePoint)
            BoxList.append(i)
    print(np.argmin(Distance))
    print(' is ')
    Detections = [BoxList[np.argmin(Distance)]]

def main():
    rospy.init_node('ssd_lis_node', anonymous=True)
    #Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型 第三个参数是回调函数的名称
    rospy.Subscriber('/usb_cam/image_raw', Image, callback)
    rospy.Subscriber('/trt_node/ssd_v2', Detection2DArray, runDetect)
    rospy.spin()
if __name__ == '__main__':
    main()
