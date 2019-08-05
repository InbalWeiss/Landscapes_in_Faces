# *******************************************************************
# EDITED VERSION 
# Authors: Roz Uslan & Weiss Inbal, 2019
# Email: yussiroz@gmail.com, inbalweissnew@gmail.com
#
#Facial key points detection using Convo (CNN) & YOLOv3
#********************************************************************
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : utils.py
# This file contains the code of the parameters and help functions
#
# *******************************************************************


import datetime
import numpy as np
import cv2
from keras.layers import Input, Dense, Dropout, LeakyReLU, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from PIL import Image

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLACK = (0, 0, 0)


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------

# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box & facial key points
def draw_predict(frame, conf, left, top, right, bottom,x,y):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)
    # Draw facial key points 
    for i,j in zip(x,y):
        cv2.circle(frame,(int(i),int(j)),3,COLOR_BLACK,-1)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)


def post_process(frame, outs, conf_threshold, nms_threshold,convo_weights):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    coor=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        
        # x,y coordinates of facial ROI
        x_f,y_f=face_detection_roi(frame,top,left,width,height,convo_weights)
        # list of x,y coordinates of ROI
        coor.append([x_f,y_f])
        left, top, right, bottom = refined_box(left, top, width, height)
        # draw_predict(frame, confidences[i], left, top, left + width,
        #              top + height)
        draw_predict(frame, confidences[i], left, top, right, bottom,x_f,y_f)
    return final_boxes,coor


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._num_frames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._num_frames / self.elapsed()

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom

# -------------------------------------------------------------------
# ROI Face Detection (facial key points detection)
# -------------------------------------------------------------------

# excute Convo model with pretrained weights
def face_detection_roi(frame,top,left,width,height,convo_weights):
        # crop face from image
        face_im=frame[top:top+height,left:left+width]
        #resize to 96x96 for CNN with antialias
        face_resize=Image.fromarray(face_im).convert('L').resize((96,96),Image.ANTIALIAS)
        # convert to np array again and normalize from 0-1
        face_norm=np.array(face_resize)/255.0
        face_norm=face_norm.reshape((1,96,96,1))
        
        #load model & pretrained weights
        my_model=convo(pretrained_weights=convo_weights)
        #predict coordinates (roi)
        pred=my_model.predict(face_norm)
        
        #x,y according to crop image (15 key points)
        x_s = [(x/96)*face_im.shape[1] for x in pred[:,:30:2]]
        y_s = [(x/96)*face_im.shape[0] for x in pred[:,1:30:2]]
        
        #x,y according to frame (convert to list inside a list)
        x_f=np.array(x_s)+left
        x_f=x_f.tolist()
        y_f=np.array(y_s)+top
        y_f=y_f.tolist()
        
        return x_f[0],y_f[0]
        
# Convo = CNN model for facial key points detection
def convo(pretrained_weights = None,input_size = (96,96,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = LeakyReLU(alpha=0.3), padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv2 = Conv2D(16, 3, activation = LeakyReLU(alpha=0.3), padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, 3, activation = LeakyReLU(alpha=0.3), padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv4 = Conv2D(32, 3, activation = LeakyReLU(alpha=0.3), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    batch_1 = BatchNormalization()(conv4)
    conv5 = Conv2D(64, 3, activation = LeakyReLU(alpha=0.3), padding = 'same', kernel_initializer = 'he_normal')(batch_1)
    conv6 = Conv2D(64, 3, activation = LeakyReLU(alpha=0.3), padding = 'same', kernel_initializer = 'he_normal')(conv5)       
    flatten_1 = Flatten()(conv6)
    drop1 = Dropout(0.2)(flatten_1)   
    output = Dense(30, kernel_initializer = 'he_normal')(drop1)
    
    model = Model(input = inputs, output = output)
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['accuracy'])
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    
    return model  
# -------------------------------------------------------------------------------- #