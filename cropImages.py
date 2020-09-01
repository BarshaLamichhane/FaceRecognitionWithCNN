
import cv2
import tensorflow as tf
from keras_vggface.vggface import VGGFace

import mtcnn.mtcnn as MTCNN
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import sys
import glob
import keras_vggface.vggface ## you need to install keras-applications also
###https://www.digitalocean.com/community/tutorials/how-to-detect-and-extract-faces-from-an-image-with-opencv-and-python

# print(keras_vggface.__version__)
main_dir = "yalefaces"
original_faces_directory = "yalefaces"
path = os.path.join(main_dir,original_faces_directory )
filenames = os.listdir(path) ##original filename
jpgImageDirectory = 'yalefaces/jpgImages'
croppedImageDirectory = 'yalefaces/croppedImages'
boxPlotImageDirectory = 'yalefaces/boxPlotOfTheDetectedFace'
directories=['croppedImages','boxPlotOfTheDetectedFace','jpgImages']

##creating necessary directories
def createDirectories(directories):
    for dir in directories:
        if(os.path.exists(os.path.join(main_dir,dir))):
            shutil.rmtree(os.path.join(main_dir,dir))
            print("[INFO] removed directory %s successfully!"%dir)
            os.mkdir(os.path.join(main_dir,dir))
            print("[INFO] created directory %s successfully!"%dir)
        else:
            os.mkdir(os.path.join(main_dir,dir))
            print("[INFO] created directory %s successfully!"%dir)

##calling the createDirectories function
createDirectories(directories)

##this is for test
def convertImage(filename):
    check = filename.split('.')
    if check[len(check)-1]=='gif':
        return
    else:
        os.rename(os.path.join(path, filename), os.path.join(path, filename + '.gif'))

## the original images do not have gif extension so it is not openend hence provided gif extension
def imageToGIF(filename):
    os.rename(os.path.join(path, filename), os.path.join(path, filename + '.gif'))


###replace gif with jpg name ## this will only change the name but not format ## for creating name it is created
def saveToJPGFormat(filename):
    last_char_index = filename.rfind("gif")
    new_string = filename[:last_char_index]+"jpg"
    return new_string

## convert iamges to jpg format and save in the folder
def convertImageToJPG(filename):
    im = Image.open(os.path.join(path,filename))
    # im.save(saveToJPGFormat(filename))
    im.convert('RGB').save(os.path.join('yalefaces/jpgImages/',saveToJPGFormat(filename)))
##read the images with cv2
def readImageObjectWithCv2(directory,filename):
    # imgg = Image.open(os.path.join('yalefaces/jpgImages/',filename)) ##using PIL
    image = cv2.imread(os.path.join(directory, filename))
    # print("Input Images are:", image)
    return image

##read the images with PIL
def readImageObjectWithPIL(directory,filename):
    image = Image.open(os.path.join(directory,filename)) ##using PIL
    return image

##convert image to grayscale object
def convertToGrayScale (filename):
    image = cv2.imread(os.path.join(jpgImageDirectory, filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("grayscale inputs are:", gray)
    return gray

def boxplotOfTheDetectedFace(filename):
    image = cv2.imread(os.path.join(jpgImageDirectory, filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    if len(faces) == 0:
        print("No Faces Found")
        return
    else:
        print("[INFO] Found {0} Faces!".format(len(faces)))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            status = cv2.imwrite(os.path.join(boxPlotImageDirectory, filename), image) ## it will simply draw rectangle on detected
            # without cropping
            print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

def cropFace(filename):
    image = cv2.imread(os.path.join(jpgImageDirectory, filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    if len(faces) == 0:
       print("No Faces Found")
       return
    else:
        print("[INFO] Found {0} Faces!".format(len(faces)))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            status = cv2.imwrite(os.path.join(croppedImageDirectory, filename), roi_color)
            print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

for filename in filenames:
    convertImageToJPG(filename)

jpgfilenames = os.listdir(jpgImageDirectory)

for filename in jpgfilenames:
    cropFace(filename)
    boxplotOfTheDetectedFace(filename)















