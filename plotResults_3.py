# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 09:22:11 2019
Visualize the clusters that models obtain.
@author: night
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
 
#Set folder from which to draw images
imgFolder = 'ImagesCompile'

def getList(dir):
    idList = []
    for filename in os.listdir(dir):
        basename, file_extension = os.path.splitext(filename)
        idList.append([basename,file_extension])
    return idList


# Load and plot model silhouette scores
path = './model_params/vgg16/kmeans_scores_final.npy'
vgg16score = np.load(path)
path = './model_params/vgg19/kmeans_scores_final.npy'
vgg19score = np.load(path)
path = './model_params/resnet50/kmeans_scores_final.npy'
rn50score = np.load(path)
path = './model_params/inceptionv3/kmeans_scores_final.npy'
inv3score = np.load(path)

fig,ax = plt.subplots()
x=[2,3,4,5]
ax.plot(x,vgg16score,label='VGG16')
ax.plot(x,vgg19score,label='VGG19')
ax.plot(x,rn50score,label='Resnet50')
ax.plot(x,inv3score,label='InceptionV3')
ax.legend(loc='upper right')

plt.show()

input("Press enter to continue")

# Show examples of images in each label

#Load labels
idList = getList(imgFolder)
path = './model_params/resnet50/labels_3cluster_final.npy'
rn50_cluster_label = np.load(path)

i_0clust = np.where(rn50_cluster_label==0)
i_1clust = np.where(rn50_cluster_label==1)
i_2clust = np.where(rn50_cluster_label==2)

#Load and display some images for each cluster
num_img = 5
print("Showing 1st cluster images")
for i in range(70,70+num_img):
    i_file = i_0clust[0][i]
    im_file = ('./'+imgFolder+'/'+idList[i_file][0]+idList[i_file][1])
    im = Image.open(im_file)
    im.show()
    
input("Press enter to continue")
print("Showing 2nd cluster images")    
for i in range(70,70+num_img):
    i_file = i_1clust[0][i]
    im_file = ('./'+imgFolder+'/'+idList[i_file][0]+idList[i_file][1])
    im = Image.open(im_file)
    im.show()
    
input("Press enter to continue")
print("Showing 3rd cluster images")    
for i in range(70,70+num_img):
    i_file = i_2clust[0][i]
    im_file = ('./'+imgFolder+'/'+idList[i_file][0]+idList[i_file][1])
    im = Image.open(im_file)
    im.show()
        

#Get most used words for these images (bag of words from feature list)

#Import data
dataFilename = './model_params/resnet50/id_list_success_final.csv'
image_ids = pd.read_csv(dataFilename,encoding = "ISO-8859-1")

dataFilename = 'imageFeatures_190412.csv'
image_simplefeatures = pd.read_csv(dataFilename,encoding = "ISO-8859-1")


#Get titles for each cluster 
titles_0=[]
titles_1=[]
titles_2=[]

for i in i_0clust[0]:
    id_now = image_ids.id[i]
    title_i = image_simplefeatures.title[image_simplefeatures.id==id_now]
    titles_0.append(title_i.tolist()[0])
    
for i in i_1clust[0]:
    id_now = image_ids.id[i]
    title_i = image_simplefeatures.title[image_simplefeatures.id==id_now]
    titles_1.append(title_i.tolist()[0])
    
for i in i_2clust[0]:
    id_now = image_ids.id[i]
    title_i = image_simplefeatures.title[image_simplefeatures.id==id_now]
    titles_2.append(title_i.tolist()[0])

#Get complete set of titles    
titles_complete = titles_0+titles_1+titles_2

#Create vocabulary from complete set of titles
cv = CountVectorizer()
data = cv.fit_transform(titles_complete)
vocab = cv.get_feature_names()
dist = np.sum(data,axis=0)
freq = dist / (np.sum(dist))

#Get most disproportionate titles for each cluster
cv0 = CountVectorizer(vocabulary=cv.vocabulary_)
data0 = cv0.fit_transform(titles_0)
dist_0 = np.sum(data0,axis=0)
freq_0 = dist_0 / (np.sum(dist_0))
rel_freq_0 = np.array(freq_0 - freq)
top10_0 = rel_freq_0.argsort()[0][-10:]

cv1 = CountVectorizer(vocabulary=cv.vocabulary_)
data1 = cv1.fit_transform(titles_1)
dist_1 = np.sum(data1,axis=0)
freq_1 = dist_1 / (np.sum(dist_1))
rel_freq_1 = np.array(freq_1 - freq)
top10_1 = rel_freq_1.argsort()[0][-10:]

cv2 = CountVectorizer(vocabulary=cv.vocabulary_)
data2 = cv2.fit_transform(titles_2)
dist_2 = np.sum(data2,axis=0)
freq_2 = dist_2 / (np.sum(dist_2))
rel_freq_2 = np.array(freq_2 - freq)
top10_2 = rel_freq_2.argsort()[0][-10:]

print('Most overrepresented words in cluster 0:')
for ind in top10_0:
    print(vocab[ind])
    
print('')    
print('Most overrepresented words in cluster 1:')
for ind in top10_1:
    print(vocab[ind])

print('')    
print('Most overrepresented words in cluster 2:')
for ind in top10_2:
    print(vocab[ind])


#Take grouped images
#Reapply to prediction models.



