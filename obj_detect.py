#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:02:54 2019

@author: anik
"""
# Importing Libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio


def detect(frame, net, transform): #We define a detect function that will take as inputs, a frame, a ssd neural network, and a transformation to be applied on the images, and that will return the frame with the detector rectangle.
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]# Apply transformation to the frame
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # Frame to torch tensor #GBR color
    x = Variable(x.unsqueeze(0)) 
    y = net(x) #We feed the neural network ssd with the image and we get the output y.
    detections = y.data #Create the detections tensor
    scale = torch.Tensor([width, height, width, height])
    
    for i in range(detections.size(1)): #detecting objects in every class
            j = 0
            while detections(0, i, j, 0) >=0.6:
                pt = (detections[0, i, j, 1:] * scale).numpy()
                cv2.rectangle(frame, int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]), (255, 0, 0))
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
    return frame


net = build_ssd('test') #Create object that becomes our our neural network ssd.
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) #Create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.


video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read() # detect(frame, net.eval(), transform)
    cv2.imshow('Video', detect(frame, net.eval(), transform))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
    


    