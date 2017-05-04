#!/usr/bin/python

import cv2 as opencv, os
import numpy
from PIL import Image

# Simple Program that implements face recognition using OpenCV
# Requires Python 3 and OpenCV 3

#---------------------------------------------------------------------------

def getImages(localPath):

    facedb = [os.path.join(localPath, f) for f in os.listdir(localPath) if not f.endswith('.wink')]

    pictures = []
    labels = []

    print("Loading Faces for Training")

    for faceloc in facedb:
        trainingSet = Image.open(faceloc).convert('L')
        trainingImage = numpy.array(trainingSet, 'uint8')

        nbr,facesToTrain = __prepare_image(faceloc,trainingImage)
        
        for (left, right, width, height) in facesToTrain:
            pictures.append(trainingImage[right: right + height, left: left + width])
            labels.append(nbr)
            opencv.imshow("Currently Populating Training Set", trainingImage[right: right + height, left: left + width])
            opencv.waitKey(50)

    return pictures, labels


#---------------------------------------------------------------------------

def __prepare_image(faceloc,trainingImage):

        
    labels = int(os.path.split(faceloc)[1].split(".")[0].replace("subject", ""))
    
    facesToTrain = opencv.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(trainingImage)

    return labels,facesToTrain


recognizer = opencv.face.createLBPHFaceRecognizer()
#recognizer = opencv.face.createEigenFaceRecognizer()
#recognizer = opencv.face.createFisherFaceRecognizer()

localPath = './yalefaces'

pictures, labels = getImages(localPath)
opencv.destroyAllWindows()
recognizer.train(pictures, numpy.array(labels))

cap = opencv.VideoCapture(0)
while 1: 
 

    ret, img = cap.read() 
 
    gray = opencv.cvtColor(img, opencv.COLOR_BGR2GRAY)
    faces = opencv.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
 
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        opencv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        nbr_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])
        if nbr_predicted == 16:
            nbr_predicted = "Shikha"
        if nbr_predicted == 17:
            nbr_predicted = "Aniketh"
        if nbr_predicted == 45:
            nbr_predicted = "Devyash"
        if conf<=90:
            print ("{} is Correctly Recognized with confidence {}".format(nbr_predicted, conf))
        else:
            nbr_predicted = "Unknown";
            print ("Subject is categorized as {} with confidence {}".format(nbr_predicted, conf))

        font = opencv.FONT_HERSHEY_DUPLEX
        opencv.putText(img, nbr_predicted, (x + 6, y+h - 6), font, 1.0, (255, 255, 255), 1)
      
        opencv.waitKey(100)
 
    opencv.imshow('img',img)
 
    k = opencv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()