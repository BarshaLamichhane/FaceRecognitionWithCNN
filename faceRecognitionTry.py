###https://towardsdatascience.com/image-pre-processing-c1aec0be3edf
###https://keras.io/api/preprocessing/image/
####https://stackoverflow.com/questions/38025838/normalizing-images-in-opencv/39037135#39037135
###https://blog.cambridgespark.com/robust-one-hot-encoding-in-python-3e29bfcec77e
###https://www.digitalocean.com/community/tutorials/how-to-detect-and-extract-faces-from-an-image-with-opencv-and-python
###https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
######https://medium.com/@Intellica.AI/a-guide-for-building-your-own-face-detection-recognition-system-910560fe3eb7


##*********************************Note please run dataPreparationYalefaces.py before running this*****************************************#

###Xilinix______________________________________________________________________________________________________________
##https://www.digikey.kr/en/articles/build-and-program-fpga-based-designs-quickly-python-jupyter-notebooks

##______________________________________________________________________________________________________________________
import cv2
import tensorflow as tf
from keras_vggface.vggface import VGGFace
import mtcnn.mtcnn as MTCNN
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil
import sys
import glob
import keras_vggface.vggface  ## you need to install keras-applications also
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MultiLabelBinarizer

main_dir = "yalefaces"
original_faces_directory = "yalefaces"
path = os.path.join(main_dir, original_faces_directory)
filenames = os.listdir(path)  ##original filename
jpgImageDirectory = 'yalefaces/jpgImages'
croppedImageDirectory = 'yalefaces/croppedImages'
boxPlotImageDirectory = 'yalefaces/boxPlotOfTheDetectedFace'
trainImageDirectory = 'yalefaces/train'
testImageDirectory = 'yalefaces/test'
directories = ['croppedImages', 'boxPlotOfTheDetectedFace', 'jpgImages', 'train', 'test']
trainFilenames = os.listdir(trainImageDirectory)
testFilenames = os.listdir(testImageDirectory)

categories=[]
X_train=[]
X_test=[]
Y_train=[]
normalized_X_Train=[]
normalized_X_Test=[]
one_hot_Y=[]

def create_train_data(images): ## images = trainFileNames
    for filename in images:
        category = filename.split('.')[0]
        # categories.append(category)
        img_array_train = cv2.imread(os.path.join(trainImageDirectory, filename), cv2.IMREAD_UNCHANGED)


        new_img_array_train = cv2.resize(img_array_train, dsize=(100, 100))
        norm_image_train = cv2.normalize(new_img_array_train , None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        normalized_X_Train.append(norm_image_train)
        X_train.append((new_img_array_train))
        Y_train.append(category)


    # Y_train.append(category)
def create_test_data(images): ## images = testFileNames
    for filename in images :
        img_array_test = cv2.imread(os.path.join(testImageDirectory, filename), cv2.IMREAD_UNCHANGED)
        new_img_array_test = cv2.resize(img_array_test, dsize=(100, 100))
        norm_image_test = cv2.normalize(new_img_array_test, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F)
        normalized_X_Test.append(norm_image_test)
        X_test.append((new_img_array_test))

 ###____________________________________________________Do not delete these portions
    # plt.imshow(X_test[0], cmap="gray")
    # plt.show()
###_______________________________________________
def convert_list_to_string(org_list, seperator=''):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    return seperator.join(org_list)

def create_model():

    model = Sequential()

    model.add(Conv2D(64,(3,3), activation = 'relu', input_shape =normalized_X_Train.shape[1:]))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(15, activation='softmax')) ##since there are 15 categoies we need to write 15 categories here

    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def main():
    models = create_model()
    models.fit(normalized_X_Train, multilabely_Y, epochs=30, batch_size=1, shuffle=True)
    models.save("FaceRecognition.h5")
    # reconstructed_model = tf.keras.models.load_model("FaceRecognition.h5")
    # print("reconstructed model is:", reconstructed_model)
    # predict = reconstructed_model.predict(normalized_X_Test)
    # print(predict)

if __name__ == '__main__':
    create_train_data(trainFilenames)
    # create_test_data(testFilenames)
    create_test_data(np.asarray(['subject02.happy.jpg']))
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    normalized_X_Train = np.asarray(normalized_X_Train)
    normalized_X_Test = np.asarray(normalized_X_Test)
    label_encoder = LabelEncoder()  ## at first string must be labelled to integer value then only other
    nonReshapedEncoder = label_encoder.fit_transform(Y_train)
    originalLevel = label_encoder.inverse_transform(nonReshapedEncoder)
    encoder_Y = label_encoder.fit_transform(Y_train)
    encoder_Y = encoder_Y.reshape(-1, 1)
    # ____________one hot encoding process 1____________
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')  ## creating instance of one hot encoder
    enc_df = enc.fit_transform(
        encoder_Y)  ## this will be fed to training data if needed but in our case we have used another one
    ## while feeeding this accuracy was very poor
    print("enc_f shape", enc_df.shape)
    ###____________________________________________________Do not delete these portions
    # plt.imshow(X_train[0], cmap="gray")
    # plt.show()
    ###_______________________________________________
    ##______________another process of labelling but this is binary type and as excellent accuracy;this is used here__________
    one_hot = MultiLabelBinarizer()
    multilabely_Y = one_hot.fit_transform(encoder_Y)
    # print("multilabel inverse transform is:",one_hot.inverse_transform(multilabely_Y))
    ####______________ another method of one hot encoding i.e. process 3 not used in our case
    encoded = to_categorical(encoder_Y)
    #main()
    test_df = pd.DataFrame({
        'filename':testFilenames
    })
    reconstructed_model = tf.keras.models.load_model("FaceRecognition.h5")
    predict = reconstructed_model.predict(normalized_X_Test)
    labelOfPredict = np.argmax(predict, axis=-1)
    predicted = label_encoder.inverse_transform(labelOfPredict)
    # Convert list of strings to string
    full_str = convert_list_to_string( predicted)
    print("This is",full_str+"'s face")





####_________________ please don't delete this, this will be used while using training model directly without saving and predicting#_____
# predictions = model.predict(normalized_X_Test)
# predicted_val = [int(round(p[0])) for p in predictions]
# print(predicted_val)

