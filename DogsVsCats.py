import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

### https://www.tensorflow.org/guide/keras/save_and_serialize
## https://www.kaggle.com/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial/data
## https://www.kaggle.com/ishaanaditya/dogs-vs-cats?select=sampleSubmission.csv
###https://www.kaggle.com/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial
##https://www.kaggle.com/bulentsiyah/dogs-vs-cats-classification-vgg16-fine-tuning
##https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8
##https://www.pyimagesearch.com/2019/06/24/change-input-shape-dimensions-for-fine-tuning-with-keras/
## https://medium.com/@ODSC/image-augmentation-for-convolutional-neural-networks-18319e1291c
##https://www.kaggle.com/learn/deep-learning
## http://www.robots.ox.ac.uk/~vgg/data/
##https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
##https://medium.com/analytics-vidhya/how-to-implement-face-recognition-using-vgg-face-in-python-3-8-and-tensorflow-2-0-a0593f8c95c3
##https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5
##https://analyticsindiamag.com/10-face-datasets-to-start-facial-recognition-projects/
##https://note.nkmk.me/en/python-pillow-basic/


import matplotlib.pyplot as plt
import random
import os

FAST_RUN =True
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

main_dir = "DogsVsCats/dogs-vs-cats/"
train_dir = "train"
path = os.path.join(main_dir,train_dir)

filenames = os.listdir(path)
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    categories.append(category)
    # image = cv2.imread(os.path.join(path, filename))

# print(categories)
files_df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(files_df,
                                         test_size=0.20,
                                         random_state=0)
train_df = train_df.reset_index(drop=True)
print("train df are:",train_df)
validate_df = validate_df.reset_index(drop=True)
print(validate_df)
total_train = train_df.shape[0]
print("shape of total_train is:",total_train)
total_validate = validate_df.shape[0]
strategy = tf.distribute.MirroredStrategy()
batch_size=15*strategy.num_replicas_in_sync
print(batch_size)

######Preparing Inage Data
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
print("train_generatot:",train_generator)

validation_datagen = ImageDataGenerator(
    # rotation_range=15,
    rescale=1. / 255
    # shear_range=0.1,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1
)

validation_generator = validation_datagen.flow_from_dataframe(validate_df,
    path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3),
                     activation='relu',
                     input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(60, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    # model.summary()
    return model

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

import time

epochs = 3 if FAST_RUN else 30

epochs = 10
with strategy.scope():
    model = create_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# fit_data = model.fit(train_generator,
#                                epochs=epochs,
#                                validation_data=validation_generator,
#                                validation_steps=1,
#                                steps_per_epoch=total_train // batch_size,
#                                callbacks=callbacks)
# print("fit datas are:",fit_data)
# model.save("DogsVsCatClassification.h5")

reconstructed_model = tf.keras.models.load_model("DogsVsCatClassification.h5")
print("reconstructed model is:",reconstructed_model)
#
test_filenames = os.listdir("DogsVsCats/dogs-vs-cats/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
print("test_df are:",test_df)
idlist=[]
for file in test_df["filename"]:
    idlist.append(file.split(".")[0])

test_df['ids'] = list
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "DogsVsCats/dogs-vs-cats/test1",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)
print(test_generator)
predict = reconstructed_model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))
print("predict are:",predict)
test_df['category'] = np.argmax(predict, axis=-1)
print("arg max is:",np.argmax(predict, axis=-1).shape)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
print("label map are",label_map)
print("train generator class are:",train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })
print("test_df category",test_df['category'])
