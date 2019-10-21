#!/usr/bin/env python
#-*-coding:UTF-8-*-
from __future__ import print_function
 
import sys
import rospy
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import time
from tf_trt_models.detection import download_detection_model, build_detection_graph
import cv2
from utils import label_map_util
label_map = label_map_util.load_labelmap('mscoco_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=90, use_display_name=True)

global process_this_frame,boxes,scores,classes,num_detections
# Initialize some variables
process_this_frame = True
boxes=[]
scores=[]
classes=[]
num_detections =[]

frozen_g = tf.GraphDef()
with open('./ssd_inception_v2_coco_trt.pb', 'rb') as f:
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
    global process_this_frame,boxes,scores,classes,num_detections
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
    for i in range(int(num_detections)):
        # scale box to image coordinates
        box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
        left,top,right,bottom = int(box[1]),int(box[0]),int(box[3]),int(box[2])
        # display rectangle
        image = cv2.rectangle(image,(left,top),(right,bottom),(0,255,255),3)
        # display class index and score
        cv2.rectangle(image, (left, bottom - 25), (right, bottom), (0, 0, 255),cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('trt',image)
    cv2.waitKey(3)
 
def main():
    rospy.init_node('myv2_node', anonymous=True)
    #Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型 第三个参数是回调函数的名称
    rospy.Subscriber('/hkCamera/image_raw', Image, callback)
    rospy.spin()
if __name__ == '__main__':
    main()
