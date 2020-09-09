##https://forums.fast.ai/t/accessing-filenames-while-batch-processing-images/1981/8
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

jpgImageDirectory = 'yalefaces/jpgImages'
trainImageDirectory = 'yalefaces/train_test'
testImageDirectory = 'yalefaces/test_test'
trainFilenames = os.listdir(trainImageDirectory)
testFilenames = os.listdir(testImageDirectory)
X_train=[]
Y_train=[]
normalized_X_Train=[]
height=100
width=100
batchSize=38
train_epochs=30
no_of_batch = len(trainFilenames)//batchSize

def imageLoader(files, batch_size):

    L = len(files)
    print("shape of L is:",L)
    # this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            print("limit is:",limit)
            # for f in range(batch_start, batch_end):
                # print(os.path.join(trainImageDirectory, files[batch_start:limit]))

                # X = cv2.imread(os.path.join(trainImageDirectory, files[batch_start:limit]), cv2.IMREAD_UNCHANGED)
                # X = cv2.imread(os.path.join(trainImageDirectory, files[f]), cv2.IMREAD_UNCHANGED)
                # Y = files[batch_start:limit]
                # Y = [j.split('.')[0] for j in Y]
            # X = someMethodToLoadImages(files[batch_start:limit])
            # Y = someMethodToLoadTargets(files[batch_start:limit])

            # yield (X, Y)  # a tuple with two numpy arrays with batch_size samples

            ##__________________________________________________________________

            X = files[batch_start:limit]
            category = [j.split('.')[0] for j in X]
            # for f in X:
            #     category = f.split('.')[0]
            #     Y_train = category
            #
            #     img_array_train = cv2.imread(os.path.join(trainImageDirectory, f), cv2.IMREAD_UNCHANGED)
            #     new_img_array_train = cv2.resize(img_array_train, dsize=(100, 100))
            #     norm_image_train = cv2.normalize(new_img_array_train, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
            #                                      dtype=cv2.CV_32F)
            yield (X, category)
            ##_____________________________________________________________________




            batch_start += batch_size
            batch_end += batch_size
        break
        # print("images are:",X)
        # print("labels are:",Y)



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

    model.add(Dense(multilabel_Y.shape[1], activation='softmax')) ##since there are 15 categoies we need to write 15 categories here

    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def main():
    models = create_model()
    # models.fit(normalized_X_Train, multilabel_Y, epochs=8, steps_per_epoch=19, batch_size=batchSize)
    models.fit(normalized_X_Train,multilabel_Y, epochs=1, batch_size=batchSize)
    models.save("FaceRecognitionBarsha2.h5")
    # reconstructed_model = tf.keras.models.load_model("FaceRecognition.h5")
    # print("reconstructed model is:", reconstructed_model)
    # predict = reconstructed_model.predict(normalized_X_Test)
    # print(predict)


if __name__ == '__main__':


    segmentedDatasets = imageLoader(trainFilenames, batchSize)
    label_encoder = LabelEncoder()
    for value in segmentedDatasets:
        X = np.asarray(value[0])
        X_train = [cv2.imread(os.path.join(trainImageDirectory, j), cv2.IMREAD_UNCHANGED) for j in X]
        resize_X_train = [cv2.resize(j, dsize=(width, height)) for j in X_train]
        normalized_X_train = [cv2.normalize(j , None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for j in resize_X_train]
        normalized_X_Train = np.asarray(normalized_X_train )
        Y_train = np.asarray(value[1])
        encoder_Y = label_encoder.fit_transform(Y_train)
        encoder_Y = encoder_Y.reshape(-1, 1)
        one_hot = MultiLabelBinarizer()
        multilabel_Y = one_hot.fit_transform(encoder_Y)
        multilabel_Y = np.asarray(multilabel_Y)
        print(Y_train)
        print(X)
        print(multilabel_Y.shape[1])
        main()
        ##___________________________________________
            # plt.imshow(norm_X_train[37], cmap="gray")
            # plt.show()
            ###______________________________________________




###____________________________________________________Do not delete these portions
# plt.imshow(X_train[37], cmap="gray")
# plt.show()
###_______________________________________________




