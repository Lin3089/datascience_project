from ctypes import *
import math
import random
import os
import time
from itertools import combinations
#計算歐式距離
def is_close(p1, p2):
    dst = math.sqrt(p1**2 + p2**2)
    return dst 

import cv2
import numpy as np
net = cv2.dnn.readNetFromDarknet("yolov4.cfg","yolov4.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = [line.strip() for line in open("coco.names")]
#colors = [(0,0,255),(255,0,0),(0,255,0)]

def yolo_detect(frame):
    # forward propogation
    img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape 
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # get detection boxes
    img = cv2.rectangle(img, (25, 140), (70,80), (0,0,255), 2)
    obj_midpoint = (47,110)
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            tx, ty, tw, th, confidence = detection[0:5]
            scores = detection[5:]
            class_id = np.argmax(scores)  
            if confidence > 0.3:   
                center_x = int(tx * width)
                center_y = int(ty * height)
                w = int(tw * width)
                h = int(th * height)

                # 取得箱子方框座標
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    #開始畫框框
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)#過濾多餘的框框
    font = cv2.FONT_HERSHEY_PLAIN
   
    #centroid_dict = dict() 
    #objectId = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            #color = colors[class_ids[i]]
            if label == 'person':
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
                #cv2.putText(img, label, (x, y -5), font, 3, (0,255,0), 3)
                #x_center = x + w/2
                y_center = y + h/2
                dx_min, dx_max, dy = x - obj_midpoint[0], x+w - obj_midpoint[0], y_center - obj_midpoint[1] # 將x邊界與y中心點與指定點距離相減
                distance_xmin = is_close(dx_min, dy)# 計算在右側的距離
                distance_xmax = is_close(dx_max, dy)#計算在左側的距離 
                #centroid_dict[objectId] = (int(x_center), int(y_center),int(x),int(y),int(x+w),int(y+h))#記錄框框的邊界點
                #objectId += 1
                if distance_xmin < 100 or distance_xmax < 100:#左側或右側過近
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) 
                    cv2.putText(img, "too close", (70,35), font, 3, (0, 0, 255), 2)
          
    #red_zone_list = []#紀錄過近的點  
    #red_line_list = []
    #for (id1, p1) in centroid_dict.items():#取得每個框框的邊界點
    #    dx_min, dx_max, dy = p1[2] - obj_midpoint[0], p1[4] - obj_midpoint[0], p1[1] - obj_midpoint[1] # 將x邊界與y中心點與指定點距離相減
    #    distance_xmin = is_close(dx_min, dy)# 計算在右側的距離
    #    distance_xmax = is_close(dx_max, dy)#計算在左側的距離 
    #    if distance_xmin < 100 or distance_xmax < 100:#左右側過近
    #        cv2.rectangle(img, (p1[2], p1[3]), (p1[4], p1[5]), (0, 0, 255), 2) 
    #        cv2.putText(img, "too close", (70,35), font, 3, (0, 0, 255), 2)
            
            
                
                
      
    
    #for idx, box in centroid_dict.items():  # 若是距離過近，則顯示too close於畫面
    #   if idx in red_zone_list:
    #       cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2) 
    #        cv2.putText(img, "too close", (70,35), font, 3, (0, 0, 255), 2)
            
    
    #end_point = (47,110)#設定指定點為連線端點
    #for check in range(0, len(red_line_list)):# 畫出兩點點的連線
    #   start_point = red_line_list[check] 
    #    check_line_x = abs(end_point_point[0] - start_point[0])# x座標 
    #    check_line_y = abs(end_point[1] - start_point[1])# y座標
    #   if (check_line_x < 125) and (check_line_y < 50):  
    #        cv2.line(img, start_point, end_point, (0, 0, 255), 2)
    #       cv2.putText(img,str(int(is_close(check_line_x,check_line_y))),(start_point[0]+10,start_point[1]), font, 2, (0, 0, 255), 2)
    
    return img

import imutils
import time

VIDEO_IN = cv2.VideoCapture(0)

while True:
    hasFrame, frame = VIDEO_IN.read()
    
    img = yolo_detect(frame)
    cv2.imshow("Frame", imutils.resize(img, width=800))#調整視窗大小

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
VIDEO_IN.release()
cv2.destroyAllWindows()