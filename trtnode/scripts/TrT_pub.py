#!/usr/bin/env python
#-*-coding:UTF-8-*-
from __future__ import print_function
 
import sys
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import time
from tf_trt_models.detection import download_detection_model, build_detection_graph
import cv2
from utils import label_map_util
label_map = label_map_util.load_labelmap('/home/nvidia/catkin_ws/src/trtnode/scripts/mscoco_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=90, use_display_name=True)
from geometry_msgs.msg import Pose2D
from lgsvl_msgs.msg import Detection2D
from lgsvl_msgs.msg import Detection2DArray
from lgsvl_msgs.msg import BoundingBox2D

global process_this_frame,boxes,scores,classes,num_detections,runMark
runMark = 0
ssd_v2_pub = rospy.Publisher('/trt_node/ssd_v2',Detection2DArray,queue_size=10)
pernum_pub = rospy.Publisher('/voice/wx_talk',String,queue_size=1)
tts_pub = rospy.Publisher('/voice/xf_tts_topic', String , queue_size=1)
# Initialize some variables
process_this_frame = True
sendNuberMark = False
boxes=[]
scores=[]
classes=[]
num_detections =[]

frozen_g = tf.GraphDef()
with open('/home/nvidia/catkin_ws/src/trtnode/scripts/ssd_inception_v2_coco_trt.pb', 'rb') as f:
    frozen_g.ParseFromString(f.read())
print('config net now')
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(frozen_g, name='')
tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')
print("ok")
time.sleep(1)
def callback(data):
    global process_this_frame,boxes,scores,classes,num_detections,runMark
    global sendNuberMark
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = np.array(cv2.resize(image,(300, 300)))
    image = np.array(image)
    if process_this_frame:
        scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
            tf_input: image_resized[None, ...]
        })
        boxes = boxes[0] # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = num_detections[0]

    process_this_frame = not process_this_frame
    # plot boxes exceeding score threshold
    DeArray = []
    personNuber = 0
    for i in range(int(num_detections)):
        # scale box to image coordinates
        if classes[i] == 1:
            personNuber = personNuber + 1
        box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
        left,top,right,bottom = box[1],box[0],box[3],box[2]
        DeArray.append(Detection2D(classes[i],scores[i],BoundingBox2D(left,top,right,bottom)))
    ssd_v2_pub.publish(DeArray)
    if sendNuberMark:
        if personNuber == 0:
            sendPernum ='我没有看到任何人'
        else:
            sendPernum = '我看到了'+str(personNuber)+'个人'
        pernum_pub.publish(sendPernum)
        tts_pub.publish(sendPernum)
        sendNuberMark = False
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('trt',image)
    runMark = 0
    cv2.waitKey(3)
def pernum_cb(data):
    global sendNuberMark
    if data.data == 1:
        sendNuberMark = True
    else:
        sendNuberMark = False       
def main():
    rospy.init_node('ssd_pub_node', anonymous=True)
    rospy.Subscriber('/hkCamera/image_raw', Image, callback)
    rospy.Subscriber('/trt_node/pernum', Int32, pernum_cb)
    #rospy.Subscriber('/camera/color/image_raw', Image, callback)
    rospy.spin()
if __name__ == '__main__':
    main()
