# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:30:42 2019

@author: LENOVO
"""

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:
        for each in range(10):
            image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg')
            # read each image of each character
            img_details = imread(image_path, as_gray=True)
            # converts each character image to binary image
            binary_image = img_details < threshold_otsu(img_details)
            # the 2D array of each image is flattened because the machine learning
            # classifier requires that each sample is a 1D array
            # therefore the 20*20 image becomes 1*400
            # in machine learning terms that's 400 features with each pixel
            # representing a feature
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)


# current_dir = os.path.dirname(os.path.realpath(__file__))
#
# training_dataset_dir = os.path.join(current_dir, 'train')
print('reading data')
training_dataset_dir = './train20X20'
image_data, target_data = read_training_data(training_dataset_dir)
print('reading data completed')

# the kernel can be 'linear', 'poly' or 'rbf'
# the probability was set to True so as to show
# how sure the model is of it's prediction
svc_model = SVC(kernel='linear', probability=True)

cross_validation(svc_model, 4, image_data, target_data)

print('training model')

# let's train the model with all the input data
svc_model.fit(image_data, target_data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(image_data)
pca_2d = pca.transform(image_data)

from sklearn import svm
import pylab as pl
svmClassifier_2d =   svm.LinearSVC(random_state=111).fit(   pca_2d, target_data)
for i in range(0, pca_2d.shape[0]):
    if(target_data[i] == '0'):
            c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    marker='+')
    elif(target_data[i] == '1'):
            c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    marker='o')
    elif(target_data[i] == '2'):
            c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',    marker='*')
    elif(target_data[i] == '3'):
           c4 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='y',    marker='+')  
    elif(target_data[i] == '4'):
           c5 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='c',    marker='+')
    elif(target_data[i] == '5'):
           c6 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='c',    marker='*')       
pl.legend([c1, c2, c3,c4,c5,c6], ['0', '1',    '2','3','4','5'])
pl.title('dataset with 3 classes and    known outcomes')
pl.show()


# save_directory = os.path.join(current_dir, 'models/svc/')
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)
# joblib.dump(svc_model, save_directory+'/svc.pkl')

import pickle
print("model trained.saving model..")
filename = './finalized_model.sav'
pickle.dump(svc_model, open(filename, 'wb'))
print("model saved")