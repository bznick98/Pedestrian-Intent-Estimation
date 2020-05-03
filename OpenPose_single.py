'''
Created by Anthony Chemaly and Zongnan Bao for ECE498 SMA final prject
'''
##Imports##

import sys
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
#from pygame import mixer

##Setup##
path=os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
border_thiccness = 2
#mixer.init()
#warning_sound=mixer.Sound('XP_SS.wav')
protoFile = r"D:\Documents\Class\ECE498\Final_Project\pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = r"D:\Documents\Class\ECE498\Final_Project\pose/mpi/pose_iter_160000.caffemodel"
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
inWidth = 368
inHeight = 368
threshold = 0.1
frame_num=0
uncertainty_score=0
switch="Safe"
neck_chest = []
chest = []
neck=[]
#shoulders = []
#hips = []
#knees = []

input_source = r"D:\Documents\Class\ECE498\Final_Project/walk_Trim.mp4"
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

out = cv2.VideoWriter(r'D:\Documents\Class\ECE498\Final_Project\MPI_walk_nc.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
while (cap.isOpened()):
    try:
        t = time.time()
        hasFrame, frame = cap.read()
        w = frame.shape[1]
        h = frame.shape[0]
        cv2.rectangle(frame, (0,h-50), (225,h), (0,0,0), thickness=cv2.FILLED)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = (w * point[0]) / W
            y = (h * point[1]) / H

            if prob > threshold :
                if (i==1 or i==14):
                    cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else :
                points.append(None)
        try:
            neck_chest.append(abs(points[1][1]-points[14][1]))
            chest.append(points[14][0])
            neck.append(points[1][0])
            if (len(chest) > 1) and (abs(chest[-1]-chest[-2])>20):
                uncertainty_score+=1
                switch="Unsafe"
                score+=1
            if (len(neck)>1) and (abs(neck[-1]-chest[-1])>15):
                uncertainty_score+=1
                switch="Unsafe"
                score+=1
            if (len(neck_chest)>=1) and (neck_chest[-1]==0):
                uncertainty_score+=1
                switch="Unsafe"
                score+=1
        except:
            neck_chest.append(neck_chest[-1])
            uncertainty_score+=1
        """
        try:
            shoulders.append(abs(points[2][0]-points[5][0]))
        except:
            shoulders.append(shoulders[-1])
        try:
            hips.append(abs(points[8][0]-points[11][0]))
        except:
            hips.append(hips[-1])
        try:
            knees.append(abs(points[9][0]-points[12][0]))
        except:
            knees.append(knees[-1])
        """
        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
                    
            if points[partA] and points[partB] and pair==[1,14]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 2, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 2, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        ##If both eyes are closed##
        if(len(neck_chest)>1 and (neck_chest[-1]>neck_chest[-2])):
            score+=1
            switch = "Unsafe"
            cv2.putText(frame,switch,(10,h-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        elif(len(neck_chest)>1 and (neck_chest[-1]<neck_chest[-2])):
            score-=3
            switch = "Safe"
            cv2.putText(frame,switch,(10,h-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        else:
            cv2.putText(frame,switch,(10,h-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if score<0:
            score=0
        cv2.putText(frame,'Score:'+str(score),(100,h-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        ##If person appears distracted
        if(score>10):
            #cv2.imwrite(os.path.join(path,'asleep.jpg'),frame)
            try:
                warning_sound.play()    
            except:
                pass
            if(border_thiccness<16):
                border_thiccness+=8
            else:
                border_thiccness-=2
                if(border_thiccness<2):
                    border_thiccness=2
            cv2.rectangle(frame,(0,0),(w,h),(0,0,255),border_thiccness)
        #else:
            #cv2.imwrite(os.path.join(path,'awake.jpg'),frame)
        #cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        #cv2.imshow('Output-Keypoints', frameCopy)
        cv2.imshow('Output-Skeleton', frame)
        out.write(frame)
        print(frame_num)
        frame_num+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
print(uncertainty_score)
x = np.linspace(0,frame_num,frame_num)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,neck_chest,c='b',marker='.',label='neck_chest')
#ax.plot(x,shoulders,c='g',marker='.',label='shoulders')
#ax.plot(x,hips,c='r',marker='.',label='hips')
#ax.plot(x,knees,c='k',marker='.',label='knees')
plt.xlabel('Frame Number')
plt.ylabel('Distance [px]')
plt.title('Distance of Keypoints over time')
plt.legend(loc='upper left')
plt.show()