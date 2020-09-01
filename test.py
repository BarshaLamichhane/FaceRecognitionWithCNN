import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import OrderedDict
# visualization
from PIL import Image
from mtcnn.mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

allFilePath = []
allFilePath = os.listdir('KaggFaceRecognition/lfw-deepfunneled/lfw-deepfunneled/')
# for root, dirs, files in os.walk('KaggFaceRecognition/'):
#     print(files)
#
# i = 0
# for batch in range(20):
#     i += 1
#     print('000'+ str(i))

# print(allFilePath)
# with os.scandir('KaggFaceRecognition/lfw-deepfunneled/lfw-deepfunneled/') as entries:
#
#     for idx, entry in enumerate(entries):
#         # apple=[]
#         # apple=entry.name
#         print(entry.name)
#
#         with os.scandir('KaggFaceRecognition/lfw-deepfunneled/lfw-deepfunneled/'+entry.name) as ent:
#             # while idx < 2:
#                 for en in ent:
#                         print(en.name)





img = load_img('KaggFaceRecognition/lfw-deepfunneled/lfw-deepfunneled/Aaron_Patterson/Aaron_Patterson_0001.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (300, 300, 3)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 300, 300, 3)
print(x)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=10, save_to_dir='KaggFaceRecognition/preview/', save_prefix='Aaron_Patterson', seed=0, save_format='jpg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely

Labels_fileNames=[]

def remove_last_underscore(iterable):

    # print(iterable.replace('__',''))
    return iterable.replace('__','')
     # if iterable[-5] == '_':
     #    # return iterable[:len(iterable)-1]
     #    return iterable.replace('--','')
     # else:
     #   return iterable
def remove_number_from_string(fileName):
    remove_number = ''.join([i for i in fileName if not i.isdigit()])
    return remove_number

with os.scandir('KaggFaceRecognition/preview') as entries:
    for idx, entry in enumerate(entries):
        fileName=entry.name
        p="KaggFaceRecognition/preview/"
        #remove_number = ''.join([i for i in fileName if not i.isdigit()])
        remove_number = remove_number_from_string(fileName)
        removeLastUnderscore = remove_last_underscore(remove_number)
        Labels_fileNames.append(removeLastUnderscore)
        # os.rename(os.path.join(p, fileName), os.path.join(p, removeLastUnderscore))
        # print(os.path.join(p, fileName))
        # print(os.path.join(p, removeLastUnderscore))

        # if (os.path.exists("kaggFaceRecognition/preview/abcd.txt")):
        #     print("hello")
        # else:
        #     print(os.path.join(p, fileName))
        #     print(os.path.join(p, removeLastUnderscore))
        #     print("hi")
        # if(os.path.isfile(os.path.join(path,fileName))):
        #     print("hello")
        # else:
        #     print(os.path.join(path,fileName))
        #     print(os.path.join(path,removeLastUnderscore))
        #     print("hi")
        # print(fileName)



        # os.rename(fileName,removeLastUnderscore)
        # cde.append(fileName)

print(len(Labels_fileNames))
# if(os.path.isfile('KaggFaceRecognition/preview/abcd.txt')):
#     # os.remove('KaggFaceRecognition/preview/abcd.txt')
#     # print("file deleted")
#     # with open("KaggFaceRecognition/preview/abcd.txt", "w") as file:
#     #     file.write("Your text goessss here")
#     # print("file deleted and created")
#     path="KaggFaceRecognition/preview/"
#     fileName="abcd.txt"
#     removeLastUnderscore="abc.txt"
#     os.rename(os.path.join(path, fileName), os.path.join(path, removeLastUnderscore))
#     # os.rename("KaggFaceRecognition/preview/abcd.txt","KaggFaceRecognition/preview/abc.txt")
#
#
# else:
#     if(os.path.isfile('KaggFaceRecognition/preview/abc.txt')):
#         os.remove('KaggFaceRecognition/preview/abc.txt')
#         print("file abc deleted successfully")
#     with open("KaggFaceRecognition/preview/abcd.txt", "w") as file:
#         file.write("Your text goes here")
#
#     print("file abcd.txt created")

# print(os.rename('KaggFaceRecognition/preview/abcd.txt','abc.txt'))
# print(cde)



# lfw_allnames = pd.read_csv("KaggFaceRecognition/lfw_allnames.csv")
# matchpairsDevTest = pd.read_csv("KaggFaceRecognition/matchpairsDevTest.csv")
# matchpairsDevTrain = pd.read_csv("KaggFaceRecognition/matchpairsDevTrain.csv")
# mismatchpairsDevTest = pd.read_csv("KaggFaceRecognition/mismatchpairsDevTest.csv")
# mismatchpairsDevTrain = pd.read_csv("KaggFaceRecognition/mismatchpairsDevTrain.csv")
# pairs = pd.read_csv("KaggFaceRecognition/pairs.csv")
# # tidy pairs data:
# pairs = pairs.rename(columns ={'name': 'name1', 'Unnamed: 3': 'name2'})
# matched_pairs = pairs[pairs["name2"].isnull()].drop("name2",axis=1)
# mismatched_pairs = pairs[pairs["name2"].notnull()]
# people = pd.read_csv("KaggFaceRecognition/people.csv")
# # remove null values
# people = people[people.name.notnull()]
# peopleDevTest = pd.read_csv("KaggFaceRecognition/peopleDevTest.csv")
# peopleDevTrain = pd.read_csv("KaggFaceRecognition/peopleDevTrain.csv")
#
# # shape data frame so there is a row per image, matched to relevant jpg file
#
# image_paths = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]
# print("image paths are:",image_paths)
# # print(lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])])
# # allIndex = lfw_allnames.index.repeat(lfw_allnames['images'])
# # print(lfw_allnames.index)
#
# image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()
# print("sdfdsfgdg", image_paths)
# image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))
# image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"
# image_paths = image_paths.drop("images",1)


# take a random sample: 80% of the data for the test set
#lfw_train, lfw_test = train_test_split(image_paths, test_size=0.2)
# lfw_train = lfw_train.reset_index().drop("index",1)
# lfw_test = lfw_test.reset_index().drop("index",1)

# verify that there is a mix of seen and unseen individuals in the test set
# print(len(set(lfw_train.name).intersection(set(lfw_test.name))))
# print(len(set(lfw_test.name) - set(lfw_train.name)))
#
# im = Image.open("KaggFaceRecognition/lfw-deepfunneled/lfw-deepfunneled/" + str(lfw_train.image_path[0]))
# plt.imshow(im)
# plt.show()

# print(lfw_allnames.index.repeat(lfw_allnames['images']))
# print(lfw_allnames.index)
# print(lfw_allnames['images'])
# print(lfw_allnames.index.repeat(1))
# print(lfw_allnames.index.repeat(lfw_allnames[0]))
