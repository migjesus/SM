import cv2
import numpy as np

cap = cv2.VideoCapture('test.mp4')
width = 320

classesFile = 'coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfig = 'tiny.cfg'
modelWeights = 'tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    height, width, channels = img.shape
    bbox = []
    classIds = []
    confidenceVal = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax((scores))
            confidence = scores[classId]
            if confidence > 0.5:
                w, h = int(
                    detection[2]*width), int(detection[3]*height)
                x, y = int(detection[0]*width-w /
                           2), int(detection[1]*height-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confidenceVal.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(bbox, confidenceVal, 0.5, 0.3)

    for i in indexes:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(
            img, f'{classNames[classIds[i]].upper()} {int(confidenceVal[i]*100)}%',
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(
        img, 1/255, (width, width), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    findObjects(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
