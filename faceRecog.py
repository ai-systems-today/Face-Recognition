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

#---------------------------------------------------------------------------
meanConf=0
minConf=3000
maxConf=0
facedb = [os.path.join(localPath, f) for f in os.listdir(localPath) if f.endswith('.wink')]
for faceloc in facedb:
    imageToPredict = Image.open(faceloc).convert('L')
    inputToPredict = numpy.array(imageToPredict, 'uint8')
    faces = opencv.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(inputToPredict)
    for (left, right, width, height) in faces:
        predictedSubj, conf = recognizer.predict(inputToPredict[right: right + height, left: left + width])
        meanConf=meanConf+conf
        if conf<minConf:
            minConf = conf
        if conf>maxConf:
            maxConf = conf
        actualSubj = int(os.path.split(faceloc)[1].split(".")[0].replace("subject", ""))
        if actualSubj == predictedSubj:
            print ("An Image of Subject {} has been correctly Recognized as Subject {} Confidence : {}".format(actualSubj,predictedSubj, conf))
        else:
            print ("An Image of Subject {} has been Incorrectly Recognized as Subject {}".format(actualSubj, predictedSubj))
        
        opencv.imshow("Face Recognition Running...", inputToPredict[right: right + height, left: left + width])
        opencv.waitKey(1000)

#---------------------------------------------------------------------------

print ("Mean Confidence is {}".format(meanConf/15))
print ("Min Confidence is {}".format(minConf))
print ("Max Confidence is {}".format(maxConf))