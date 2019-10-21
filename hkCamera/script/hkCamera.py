#!/usr/bin/env python
#-*-coding:UTF-8-*-
import rospy
from camConfig import *
import os
import cv2
import gc
from multiprocessing import Process, Manager
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def ip2Str(usrName,passwd,ip):
    return "rtsp://"+str(usrName)+":"+str(passwd)+"@"+str(ip)+":557/h264/ch33/main/av_stream"

# 向共享缓冲栈中写入数据:
def write(stack, cam, top):
    """e4
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :param cam: 摄像头参数
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    while not rospy.is_shutdown():
        _, img = cap.read()
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[:]
                gc.collect()

# 在缓冲栈中读取数据:
def read(stack):
    print('Process to read: %s' % os.getpid())
    rospy.init_node('hkCamera_node', anonymous=True)
    hkPub = rospy.Publisher('/hkCamera/image_raw', Image, queue_size=100)
    bridge = CvBridge()
    count = 0
    rate = rospy.Rate(5)  # 5hz
    while not rospy.is_shutdown():
        if len(stack) != 0:
            try:
                value = stack.pop()
                value = cv2.resize(value,(hk_imgW,hk_imgH),interpolation=cv2.INTER_AREA)
                msg = bridge.cv2_to_imgmsg(value, encoding="bgr8")
                hkPub.publish(msg)
            except:
                pass
            rate.sleep()

if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    q = Manager().list()
    pw = Process(target=write, args=(q,ip2Str(camMessage['user'],camMessage['passwd'],camMessage['ip']), 100))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pr结束:
    pr.join()
    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw.terminate()
