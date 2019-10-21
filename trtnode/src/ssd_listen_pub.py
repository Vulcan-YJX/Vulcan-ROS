#!/usr/bin/env python
'''ssd_listen ROS Node'''
# license removed for brevity
import rospy
from std_msgs.msg import String

def talker():
    '''ssd_listen Publisher'''
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('ssd_listen', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
