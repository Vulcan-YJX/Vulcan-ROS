#!/usr/bin/env python
#coding=utf-8

from __future__ import unicode_literals
from threading import Timer
from wxpy import *
import requests
from std_msgs.msg import String
import rospy
import time
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#倒入自定义的数据类型

robot_name =u'vulcanRobot'
save_image = '/home/nvidia/Pictures/watchDog.png'
talk_image = '/home/nvidia/Pictures/send_img.png'
bot = Bot(console_qr=2,cache_path="botoo.pkl")#linux环境上使用
rospy.init_node('wxtalker_node', anonymous=True)
wx_pub = rospy.Publisher('/voice/vcommand',String, queue_size=1)
my_friend = bot.friends().search(robot_name)[0]
bridge = CvBridge()
send_mark = False


def send_news(R_msg):
    try:
        my_friend.send(unicode(R_msg))
    except:
        print("Failure to send message")

def send_image(save_path):
    try:
        my_friend.send_image(save_path)
    except:
        print("Failure to send image")

def takeaPhoto():
    global send_mark
    send_mark = True


def sendimgCB(img_msg):
    global send_mark
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
#        cv_image = cv_image[:, :, ::-1]
    except CvBridgeError as e:
        print(e)
    if send_mark:
        print send_mark
        cv_image = cv2.resize(cv_image,(480,360))
        cv2.imwrite(talk_image,cv_image)
        send_mark = False
        time.sleep(1)
        send_image(talk_image)
    cv2.waitKey(3)

def talker():
    rospy.Subscriber('/voice/wx_talk', String, callback) 
    rospy.Subscriber('/hkCamera/image_raw', Image, sendimgCB)
    while not rospy.is_shutdown():
        rospy.spin()

@bot.register(my_friend)
def print_others(msg):
    global R_msg
    if str(msg.sender)[9:-1] == 'vulcanRobot':
        wx_pub.publish(msg.text)

def callback(data):
    send_wx = str(data.data).decode('utf8')
    if 'kmg_image' in send_wx:
        send_image(save_image)
    elif '照片' in send_wx:
        takeaPhoto()
    else:
        send_news(send_wx)
    send_wx = ''
    time.sleep(1)
if __name__ == '__main__':
    talker()
 
