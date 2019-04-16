# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:59:03 2019

Imports feature lists for different models and uses k means clustering. 


@author: night
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
min_clust = 2
max_clust = 5

#Load model features
path = './model_params/vgg16/vgg16_features_final.npy'
vgg16_feature_list_np = np.load(path)

path = './model_params/vgg19/vgg19_features_final.npy'
vgg19_feature_list_np = np.load(path)

path = './model_params/resnet50/resnet50_features_final.npy'
resnet50_feature_list_np = np.load(path)

path = './model_params/inceptionv3/inceptionv3_features_final.npy'
inceptv3_feature_list_np = np.load(path)

score_vgg16 = np.zeros(max_clust-min_clust+1)
score_vgg19 = np.zeros(max_clust-min_clust+1)
score_rn50 = np.zeros(max_clust-min_clust+1)
score_icv3 = np.zeros(max_clust-min_clust+1)

for i in range(min_clust,max_clust+1):
    
    kmeans_model = KMeans(n_clusters=i, random_state=1).fit(vgg16_feature_list_np)
    labels = kmeans_model.labels_
    path = './model_params/vgg16/labels_' + str(i) + 'cluster_final'
    np.save(path,labels)
    score_vgg16[i-min_clust] = metrics.silhouette_score(vgg16_feature_list_np, labels, metric='euclidean')

    kmeans_model = KMeans(n_clusters=i, random_state=1).fit(vgg19_feature_list_np)
    labels = kmeans_model.labels_
    path = './model_params/vgg19/labels_' + str(i) + 'cluster_final'
    np.save(path,labels)
    score_vgg19[i-min_clust] = metrics.silhouette_score(vgg19_feature_list_np, labels, metric='euclidean')
    
    kmeans_model = KMeans(n_clusters=i, random_state=1).fit(resnet50_feature_list_np)
    labels = kmeans_model.labels_
    path = './model_params/resnet50/labels_' + str(i) + 'cluster_final'
    np.save(path,labels)
    score_rn50[i-min_clust] = metrics.silhouette_score(resnet50_feature_list_np, labels, metric='euclidean')
    
    kmeans_model = KMeans(n_clusters=i, random_state=1).fit(inceptv3_feature_list_np)
    labels = kmeans_model.labels_
    path = './model_params/inceptionv3/labels_' + str(i) + 'cluster_final'
    np.save(path,labels)
    score_icv3[i-min_clust] = metrics.silhouette_score(inceptv3_feature_list_np, labels, metric='euclidean')
    print(i)
print("VGG16 Scores")
print(score_vgg16)
path = './model_params/vgg16/kmeans_scores_final'
np.save(path,score_vgg16)

print("VGG19 Scores")
print(score_vgg19)
path = './model_params/vgg19/kmeans_scores_final'
np.save(path,score_vgg19)

print("ResNet50 Scores")
print(score_rn50)
path = './model_params/resnet50/kmeans_scores_final'
np.save(path,score_rn50)

print("InceptionV3 Scores")
print(score_icv3)
path = './model_params/inceptionv3/kmeans_scores_final'
np.save(path,score_icv3)
