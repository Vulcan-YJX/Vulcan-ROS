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
global left, top, right, bottom, Color, ID, voice_id
voice_id = 0
left= 0
top = 0
right = 0
bottom = 0
Color = []
ID = []
def get_id(data):
    global voice_id
    voice_id = data.data
    
def callback(data):
    caidan = cv2.imread('1.png')
    global left,top,right,bottom,Color,ID,  voice_id
    global Detections,runStart
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    image = np.array(image)
    if runStart:
        for i in Detections:
            if Myclasses.has_key(str(i.id)):
                source = int(i.score)
                left = int(i.bbox.left)
                top = int(i.bbox.top)
                right = int(i.bbox.right)
                bottom = int(i.bbox.bottom)
                ID = Myclasses[str(i.id)]['name']
                Color = Myclasses[str(i.id)]['color']
                image = cv2.rectangle(image,(left,top),(right,bottom),Color,3)
        # display class index and score
                cv2.rectangle(image, (left, bottom - 20), (right, bottom),Color,cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                try:
                    cv2.putText(image,ID+str(source), (left, bottom), font, 1.0,(255,255,255), 1)
                except:
                    print('Class i is '+str(classes[i]))
    else:
        try:
            image = cv2.rectangle(image,(left,top),(right,bottom),Color,3)
#    # display class index and score
            cv2.rectangle(image, (left, bottom - 20), (right, bottom), Color,cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image,ID, (left, bottom), font, 1.0,(255,255,255), 1)
        except:
            pass
    runStart = 0
    out_win = "trt"
    cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN)
     
    if(voice_id==1): 
	    cv2.imshow(out_win,caidan)
	    cv2.waitKey(3)
    else: 
	    cv2.imshow(out_win,image)
	    cv2.waitKey(3)
def runDetect(dataNum):
    global runStart,Detections
    runStart = 1
    Detections = dataNum.detections

def main():
    rospy.init_node('ssd_lis_node', anonymous=True)
    #Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型 第三个参数是回调函数的名称
#    rospy.Subscriber('/camera/color/image_raw', Image, callback)
    rospy.Subscriber('/hkCamera/image_raw', Image, callback)
    rospy.Subscriber('/voice', Int32, get_id)
    rospy.Subscriber('/trt_node/ssd_v2', Detection2DArray, runDetect)
    rospy.Subscriber('/voice', Int32, get_id)
    rospy.spin()
if __name__ == '__main__':
    main()
