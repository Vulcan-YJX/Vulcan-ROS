#!/usr/bin/env python3
#-*-coding:UTF-8-*-
from __future__ import print_function
 
import sys
import rospy
import cv2
import face_recognition
import numpy as np
from std_msgs.msg import Int32
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import glob
from std_msgs.msg import Int32MultiArray
from nameConfig import known_face_names
import time
global known_face_encodings,known_face_names,face_locations,face_encodings,face_names,process_this_frame,runStart
runStart = 0

#print(known_face_names)
face_pub = rospy.Publisher('/faceID/pub',Int32MultiArray,queue_size=1)
dog_pub = rospy.Publisher('/faceID/watchDog',Int32,queue_size=1)
tts_pub = rospy.Publisher('/voice/xf_tts_topic', String , queue_size=10)
# Load a sample picture and learn how to recognize it.
known_face_encodings = []
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
nameList = []
englishName = []

#loading and decoding facePic in kownFace Dir 
for name in glob.glob('kownFace/*'):
    frame = cv2.imread(name)
    image = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    nameList.append(known_face_names[name.split('/')[1][:-4]])
    englishName.append(name.split('/')[1][:-4])
    #image = face_recognition.load_image_file(name)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
print('load face ok!')

 
def callback(data):
    global known_face_encodings,known_face_names,face_locations,face_encodings,face_names,process_this_frame
    global runStart
    faceIDArray = []
    bridge = CvBridge()
    try:
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    # Resize frame of video to 1/4 size for faster face recognition processing
    #small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_frame=frame
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if runStart==1:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame,model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            print(best_match_index)
            if matches[best_match_index]:
                faceIDArray.append(best_match_index)
                #face_pub.publish(known_face_names[best_match_index])
            else:
                faceIDArray.append(-1)
        face_pub.publish(data=faceIDArray)
        runStart = 0
    if runStart==2:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame,model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                tts_pub.publish('身份确认,'+nameList[best_match_index])
        runStart = 0
    
    if runStart == 3:
        name = "未知身份"
        myBoss = ''
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if process_this_frame:
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame,model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = englishName[best_match_index]
                face_names.append(name)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
            # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

            # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom + 20), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                if name == 'vulcan':
                    myBoss = name
                cv2.putText(frame, name, (left + 6, bottom + 20), font, 1.0, (255, 255, 255), 1)
            if len(face_names) > 0:
                frame = cv2.resize(frame,(480,360))
                cv2.imwrite('/home/nvidia/Pictures/watchDog.png',frame)
                time.sleep(0.5)#给写图片的时间延迟
                if myBoss == 'vulcan':
                    myBoss = ''
                    dog_pub.publish(2)
                    runStart = 0
                else:
                    myBoss = ''    
                    dog_pub.publish(1)
                face_names = []
            else:
                dog_pub.publish(0)
                face_names = []
        process_this_frame = not process_this_frame
    cv2.waitKey(3)

def runDetect(dataNum):
    global runStart
    runStart = dataNum.data
 
def main():
    rospy.init_node('faceID_node', anonymous=True)
    #Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型 第三个参数是回调函数的名称
    rospy.Subscriber('/hkCamera/image_raw', Image, callback)
    rospy.Subscriber('/faceID/Srv', Int32, runDetect)
    rospy.spin()
if __name__ == '__main__':
    main()
