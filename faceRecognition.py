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
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.utils import np_utils




###https://www.digitalocean.com/community/tutorials/how-to-detect-and-extract-faces-from-an-image-with-opencv-and-python

##*********************************Note please run dataPreparationYalefaces.py before running this*****************************************#

# print(keras_vggface.__version__)
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
FAST_RUN = True
batch_size = 32
img_height = 180
img_width = 180
IMAGE_CHANNELS = 3

categories=[]
X_train=[]
Y_train=[]
X_test=[]
Y_test = []
normalized_X_Train =[]
normalizes_X_Train=[]
##creating necessary directories

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

    model.add(Dense(15, activation='softmax'))

    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(normalized_X_Train, Y_train, epochs=10, batch_size=32, shuffle=True)
    # return model

##generate data
def create_train_data(path):

    for filename in path:
        category = filename.split('.')[0]
        categories.append(category)
        img_array = cv2.imread(os.path.join(trainImageDirectory, filename), cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(100, 100))
        X_train.append(np.asarray(new_img_array))
    files_df = pd.DataFrame({
        'filename': path,
        'image_array': X_train,
        'category': categories
    })
    print("categories are:",categories)
    print(files_df.category)
    ##use y_train[0]
    # Y_train.append(np.asarray(pd.get_dummies(files_df.category)))
    # for i in (categories):
    #     print("hhhhhhhhhhhhhhhh",i)
    #     encs = OneHotEncoder(handle_unknown='ignore')
    #     # enc_df = enc.fit_transform(files_df[['category']]).toarray().reshape(132,-1)
    #     encs_df = encs.fit_transform(files_df[['category']]).toarray().reshape(132, -1)
    #     print("iiiiiiiiiiiiiiii",encs_df)
    #     # Y_train.append(encs_df)
    abc =  LabelEncoder()
    abc.fit(categories)
    encoder_Y = abc.transform(categories)
    print("to categorical values are:",encoder_Y)
    dummy_y = np_utils.to_categorical(encoder_Y)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # # enc_df = enc.fit_transform(files_df[['category']]).toarray().reshape(132,-1)
    enc_df = enc.fit_transform(files_df[['category']])
    print("one hot encoder values are:",enc_df)
    # enc =  LabelEncoder()
    # enc_df = LabelEncoder.fit(categories)
    #
    Y_train.append(enc_df)


    ### for testing###____________________________ Do not delete these portions
    # print(X_train[0].shape)
    # plt.imshow(X_train[0], cmap="gray")
    # plt.show()
    # print(categories[0])
    # print(trainFilenames[0])
    ##______________________________


def main():
    print("execution starts here---")
    for i in range(132): ## 132 because length of X_train is 132
        norm = MinMaxScaler().fit(X_train[i])
        # transform training data
        X_train_norm = norm.transform(X_train[i])
        # print(X_train_norm)
        normalized_X_Train.append(X_train_norm)

if __name__ == '__main__':
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    create_train_data(trainFilenames)
    # print(X_train[0])
    # print(MinMaxScaler().fit(X_train[0]).transform(X_train[0]))
    # print(len(X_train[0]))
    # print(MinMaxScaler().fit(X_train[0][0]).transform(X_train[0][0]))
    # print(len(X_train[0][1]))
    # print(len(X_train[1]))
    # print(len(X_train))
    main()
    normalized_X_Train = np.asarray(normalized_X_Train)
    normalized_X_Train = np.asarray(normalized_X_Train).reshape(-1,100,100,1)
    print("vcxvcxb",normalized_X_Train)
    # Y_train = np.asarray(Y_train[0])
    # print(np.asarray(X_train.shape))
    # print(Y_train.shape)
    X_train=np.array(X_train)
    print(X_train.shape)
    print(np.array(Y_train[0]).shape)
    print(Y_train)
    Y_train = np.array(Y_train)
    print("jdhfkjhdjkf",Y_train)
    # print("normalized 4D output is:",normalized_X_Train.shape[1:])

    # print(normalized_X_Train.shape)
    # print(np.array(X_train).shape)
    # print("normalized",normalized_X_Train)
    models = create_model()
    fit = models.fit(normalized_X_Train, Y_train, epochs=10, batch_size=32,shuffle=True)
    # create_model()

    # print(np.asarray(normalized_X_Train).shape[1:])




