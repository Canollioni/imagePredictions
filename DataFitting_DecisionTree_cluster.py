# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:57:44 2019
Uses previously obtained clusters and makes decision trees and random forests for those labels.
@author: night
"""
# Import function to create training and test set splits
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import time

def getTimestamp(date):
    return time.mktime(dt.datetime.strptime(date,"%d/%m/%Y").timetuple())

def get_mae_RF(max_LN, x, y):
    pipeline_RF = make_pipeline(RandomForestRegressor(max_leaf_nodes = max_LN,n_estimators=10))
    scores = cross_val_score(pipeline_RF, x, y, scoring="neg_mean_absolute_error", cv=5)
    mae = (-1 * scores.mean())
    return(mae)

def get_mae_DT(max_leaf_nodes, x, y):
    pipeline_DT = make_pipeline(DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes))
    scores = cross_val_score(pipeline_DT, x, y, scoring="neg_mean_absolute_error", cv=5)
    mae = (-1 * scores.mean())
    return(mae)

#Import data
dataFilename = 'imageFeatures_190412.csv'
imageDF = pd.read_csv(dataFilename,encoding = "ISO-8859-1")
imageDF = imageDF.set_index('id')

dataFilename = './model_params/resnet50/id_list_success_final.csv'
image_ids = pd.read_csv(dataFilename,encoding = "ISO-8859-1")


#Load in ResNet50 clustered labels (3)
path = './model_params/resnet50/labels_3cluster_final.npy'
rn50_cluster_label = np.load(path)

i_0clust = np.where(rn50_cluster_label==0)
i_1clust = np.where(rn50_cluster_label==1)
i_2clust = np.where(rn50_cluster_label==2)

#Get title list for each cluster
titles_0=[]
titles_1=[]
titles_2=[]
for i in i_0clust[0]:
    id_now = image_ids.id[i]
    titles_0.append(id_now)
    
for i in i_1clust[0]:
    id_now = image_ids.id[i]
    titles_1.append(id_now)
    
for i in i_2clust[0]:
    id_now = image_ids.id[i]
    titles_2.append(id_now)
    
    
clust_0_df = imageDF.loc[titles_0]
clust_1_df = imageDF.loc[titles_1]
clust_2_df = imageDF.loc[titles_2]


#Set filtering parameters
beginTS = getTimestamp("01/02/2019")

c0_logic = clust_0_df.created>beginTS
c1_logic = clust_1_df.created>beginTS
c2_logic = clust_2_df.created>beginTS


c0_df_filt = clust_0_df[c0_logic]
c1_df_filt = clust_1_df[c1_logic]
c2_df_filt = clust_2_df[c2_logic]


imageModel_DT = DecisionTreeRegressor(random_state=1)
imageModel_RF = RandomForestRegressor(random_state=1)


#Select features to use for fit
#feature_list = ['aspRatio', 'bluLRBal', 'bluTBBal', 'globalContrast', 'grnLRBal', 'grnTBBal', 'imgSz', 'lumLRBal', 'lumTBBal', 'luminance', 'redLRBal', 'redTBBal', 'saturation']
feature_list = ['coarseContrast', 'globalContrast', 'imgSz', 'lumTBBal', 'luminance', 'saturation']


#Select features and scores
X0 = c0_df_filt[feature_list]
X1 = c1_df_filt[feature_list]
X2 = c2_df_filt[feature_list]

#Select features and scores
y0 = c0_df_filt.score
y1 = c1_df_filt.score
y2 = c2_df_filt.score

#Normalize the data. Logically, higher scores are arbitrarily inflated due to higher visibility.
#This is an attempted correction
y0 = np.log(y0)
y1 = np.log(y1)
y2 = np.log(y2)
plt.hist(y0)
plt.title('Cluster 0 distribution')
plt.show()
plt.hist(y1)
plt.title('Cluster 1 distribution')
plt.show()
plt.hist(y2)
plt.title('Cluster 2 distribution')
plt.show()

maxNode = 100

MAE_RF = np.zeros(maxNode)
MAE_DT = np.zeros(maxNode)

## compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in range(2,maxNode):
    MAE_DT[max_leaf_nodes] = np.exp(get_mae_DT(max_leaf_nodes, X0, y0))
print('Minimum MAE, Cluster 0, Decision tree')
print(np.min(MAE_DT[2:]))
plt.plot(MAE_DT[2:])
plt.title('Cluster 0, Decision tree')
plt.show() 

    
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in range(2,maxNode):
    MAE_RF[max_leaf_nodes] = np.exp(get_mae_RF(max_leaf_nodes, X0, y0))
print('Minimum MAE, Cluster 0, Random forest')
print(np.min(MAE_RF[2:]))
plt.plot(MAE_RF[2:])
plt.title('Cluster 0, Random forest')
plt.show() 

## compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in range(2,maxNode):
    MAE_DT[max_leaf_nodes] = np.exp(get_mae_DT(max_leaf_nodes, X1, y1))
print('Minimum MAE, Cluster 0, Decision tree')
print(np.min(MAE_DT[2:]))
plt.plot(MAE_DT[2:])
plt.title('Cluster 1, Decision tree')
plt.show() 

    
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in range(2,maxNode):
    MAE_RF[max_leaf_nodes] = np.exp(get_mae_RF(max_leaf_nodes, X1, y1))
print('Minimum MAE, Cluster 0, Random forest')
print(np.min(MAE_RF[2:]))
plt.plot(MAE_RF[2:])
plt.title('Cluster 1, Random forest')
plt.show() 


## compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in range(2,maxNode):
    MAE_DT[max_leaf_nodes] = np.exp(get_mae_DT(max_leaf_nodes, X2, y2))
print('Minimum MAE, Cluster 0, Decision tree')
print(np.min(MAE_DT[2:]))
plt.plot(MAE_DT[2:])
plt.title('Cluster 0, Decision tree')
plt.show() 

    
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in range(2,maxNode):
    MAE_RF[max_leaf_nodes] = np.exp(get_mae_RF(max_leaf_nodes, X2, y2))
print('Minimum MAE, Cluster 0, Random forest')
print(np.min(MAE_RF[2:]))
plt.plot(MAE_RF[2:])
plt.title('Cluster 2, Random forest')
plt.show() 
