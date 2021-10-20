import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import numpy as np
import os
import os.path

import gdown

from os import listdir
from os.path import isfile, join

path_gun='https://drive.google.com/uc?id=1bPPR0LVD_nn9oNAM1mTAdZxclcKxX0YJ'
path_coco='https://drive.google.com/uc?id=16Avmk9pIZRfrqr2lFiDEH0vzGitO9QAh'

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = 'diversos'
        self.threshold1 = 0.5
        self.threshold2 = 0.5
        weightsPath = os.path.sep.join([r"coco", "yolov4.weights"])
        
        if not(os.path.isfile(weightsPath)): # baixar         
            gdown.download(path_coco,weightsPath)
        configPath = os.path.sep.join([r"coco", "yolov4.cfg"])

        labelsPath = os.path.sep.join([r"coco", "obj.names"])
        #LABELS = open(labelsPath).read().strip().split("\n")

        self.net = cv2.dnn.readNet(configPath, weightsPath)
        self.classes = []
        with open(labelsPath, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.font = cv2.FONT_HERSHEY_PLAIN

    def transform(self, frame):
        #print('Modelo:',self.model)
        img = frame.to_ndarray(format="bgr24")
        height, width, channels = img.shape

        '''
        img = cv2.cvtColor(
            cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR
        )
        '''
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                #if confidence > 0.5:
                if confidence > 0.1:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.threshold1, self.threshold2)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                #cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y + 30), self.font, 3, color, 3)



        #elapsed_time = time.time() - starting_time
        #fps = frame_id / elapsed_time
        #cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), self.font, 4, (0, 0, 0), 3)
        #cv2.imshow("Image", frame)

        return img
        #return frame


def setModel(self, model):
    """ Muda modelo """
    if (self.model != model):
        self.model = model
        if (model == 'diversos'):
            weightsPath = os.path.sep.join([r"coco", "yolov4.weights"])
            if not(os.path.isfile(weightsPath)): # baixar         
                gdown.download(path_coco,weightsPath)
            configPath = os.path.sep.join([r"/coco", "yolov4.cfg"])
            labelsPath = os.path.sep.join([r"coco", "obj.names"])
            #LABELS = open(labelsPath).read().strip().split("\n")
        else:
            weightsPath = os.path.sep.join([r"gun", "yolov4.weights"])
            if not(os.path.isfile(weightsPath)): # baixar         
                gdown.download(path_gun,weightsPath)
            configPath = os.path.sep.join([r"/gun", "yolov4.cfg"])
            labelsPath = os.path.sep.join([r"gun", "obj.names"])
        
        self.net = cv2.dnn.readNet(configPath, weightsPath)
        self.classes = []
        with open(labelsPath, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.font = cv2.FONT_HERSHEY_PLAIN

#ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
ctx = webrtc_streamer(key="diversos", video_transformer_factory=VideoTransformer)


if ctx.video_transformer:
    ctx.video_transformer.threshold1 = st.slider("Threshold1", 0.0, 1.0, 0.5)
    ctx.video_transformer.threshold2 = st.slider("Threshold2", 0.0, 1.0, 0.5)
    st.header("Paraná Visão Computacional - Analise de video em tempo real")
    armas = "Armas de fogo"
    diversos = "Diversos objetos"

    app_mode = st.sidebar.selectbox(
        "O que deseja identificar?",
        [
            diversos,
            armas
        ],
    )
    st.subheader(app_mode)
    if app_mode == armas:
        #ctx.video_transformer.model='armas'
        setModel(ctx.video_transformer, 'armas')
    else:
        #ctx.video_transformer.model='diversos'
        setModel(ctx.video_transformer, 'diversos')

