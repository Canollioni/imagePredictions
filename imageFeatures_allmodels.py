# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:00:19 2019

Extracts and saves features from images using four different models (VGG16, VGG19, ResNet50, InceptionV3)
Purpose is to imput these into a K means clustering algorithm to group like-photos.

Only need to specify folder for images (imgFolder).

@author: night
"""
from keras.preprocessing import image #Not working
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_prep_inp
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_prep_inp
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_prep_inp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as incv3_prep_inp
import numpy as np
import os
import pandas as pd

imgFolder = 'ImagesCompile'

def getList(dir):
    idList = []
    for filename in os.listdir(dir):
        basename, file_extension = os.path.splitext(filename)
        idList.append([basename,file_extension])
    return idList

def imResize(img, new_dim):
    w, h = img.size
    min_dim = min(w,h)
    left = np.floor(w/2) - np.floor(min_dim/2)
    right = np.floor(w/2) + np.floor(min_dim/2)
    top = np.floor(h/2) - np.floor(min_dim/2)
    bottom = np.floor(h/2) + np.floor(min_dim/2)
    new_im = img.crop((left, top, right, bottom))
    return new_im.resize((new_dim, new_dim))

#Get image

idList = getList(imgFolder)
topics_dict = { "id":[]}

#Create models
model_vgg16 = VGG16(weights='imagenet', include_top=False)
#model_vgg16.summary()
model_vgg19 = VGG19(weights='imagenet', include_top=False)
#model_vgg19.summary()
model_resnet50 = ResNet50(weights='imagenet', include_top=False)
#model_resnet50.summary()
model_inceptv3 = InceptionV3(weights='imagenet', include_top=False)
#model_inceptv3.summary()



vgg16_feature_list = []
vgg19_feature_list = []
resnet50_feature_list = []
inceptv3_feature_list = []
  
for img in idList:
    
    imgID = img[0]
    imgExt = img[1]
#    print(imgID )
    imgDir = imgFolder + '/' + imgID + imgExt
    img_o = Image.open(imgDir)
    
    try:
        img = imResize(img_o,224)
        img_data = image.img_to_array(img)
        img_data_orig = np.expand_dims(img_data, axis=0)
        
        img_data = vgg16_prep_inp(img_data_orig)
        #VGG16 features
        vgg16_feature = model_vgg16.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())
        
        img_data = vgg19_prep_inp(img_data_orig)
        #VGG19 features
        vgg19_feature = model_vgg19.predict(img_data)
        vgg19_feature_np = np.array(vgg19_feature)
        vgg19_feature_list.append(vgg19_feature_np.flatten())
        
        img_data = resnet50_prep_inp(img_data_orig)
        #ResNet50 features
        resnet50_feature = model_resnet50.predict(img_data)
        resnet50_feature_np = np.array(resnet50_feature)
        resnet50_feature_list.append(resnet50_feature_np.flatten())
        topics_dict["id"].append(imgID)
        
        
        
#        #Resizing image to 299 for InceptionV3
        img = imResize(img_o,299)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = incv3_prep_inp(img_data)
        #InceptionV3 features
        inceptv3_feature = model_inceptv3.predict(img_data)
        inceptv3_feature_np = np.array(inceptv3_feature)
        inceptv3_feature_list.append(inceptv3_feature_np.flatten())
    except:
        print(imgID)   


path = './model_params/vgg16'
if not(os.path.isdir(path)):
    os.mkdir(path)
vgg16_feature_list_np = np.array(vgg16_feature_list)
path = './model_params/vgg16/vgg16_features_final'
np.save(path,vgg16_feature_list_np)

path = './model_params/vgg19'
if not(os.path.isdir(path)):
    os.mkdir(path)
vgg19_feature_list_np = np.array(vgg19_feature_list)
path = './model_params/vgg19/vgg19_features_final'
np.save(path,vgg19_feature_list_np)

path = './model_params/resnet50'
if not(os.path.isdir(path)):
    os.mkdir(path)
resnet50_feature_list_np = np.array(resnet50_feature_list)
np.save((path+'/resnet50_features_final'),resnet50_feature_list_np)

topics_data = pd.DataFrame(topics_dict)    
topics_data.to_csv((path+'/id_list_success_final.csv'), index=False)

path = './model_params/inceptionv3'
if not(os.path.isdir(path)):
    os.mkdir(path)
inceptv3_feature_list_np = np.array(inceptv3_feature_list)
path = path+'/inceptionv3_features_final'
np.save(path,inceptv3_feature_list_np)


