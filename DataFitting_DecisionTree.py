"""
Created on Thu Feb 21 17:57:44 2019

@author: Canollioni
"""
# Import function to create training and test set splits
from sklearn.cross_validation import train_test_split

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
    pipeline_RF = make_pipeline(RandomForestRegressor(max_leaf_nodes = max_LN))
    scores = cross_val_score(pipeline_RF, x, y, scoring="neg_mean_absolute_error")
    mae = (-1 * scores.mean())
    return(mae)

def get_mae_DT(max_leaf_nodes, x, y):
    pipeline_DT = make_pipeline(DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes))
    scores = cross_val_score(pipeline_DT, x, y, scoring="neg_mean_absolute_error")
    mae = (-1 * scores.mean())
    return(mae)

#Import data
dataFilename = 'imageFeatures_190321.csv'
imageDF = pd.read_csv(dataFilename,encoding = "ISO-8859-1")

#Set filtering parameters
beginTS = getTimestamp("01/02/2019")
logic_filt = imageDF.created>beginTS #(old_data.score>1000)# & (old_data.score<30000)
filtered_data = imageDF[logic_filt]

imageModel_DT = DecisionTreeRegressor(random_state=1)
imageModel_RF = RandomForestRegressor(random_state=1)


#Select features to use for fit
#feature_list = ['aspRatio', 'bluLRBal', 'bluTBBal', 'globalContrast', 'grnLRBal', 'grnTBBal', 'imgSz', 'lumLRBal', 'lumTBBal', 'luminance', 'redLRBal', 'redTBBal', 'saturation']
feature_list = ['aspRatio', 'coarseContrast', 'globalContrast', 'imgSz', 'lumTBBal', 'luminance', 'saturation']


#Select features and scores
x = filtered_data[feature_list]
y = filtered_data.score
plt.hist(y)
plt.show()
#Normalize the data. Logically, higher scores are arbitrarily inflated due to higher visibility.
#This is an attempted correction
y = np.log(y)
plt.hist(y)
plt.show()

maxNode = 100

MAE_RF = np.zeros(maxNode)
MAE_DT = np.zeros(maxNode)

## compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in range(2,maxNode):
    MAE_DT[max_leaf_nodes] = np.exp(get_mae_DT(max_leaf_nodes, x, y))
print(np.min(MAE_DT[2:]))
plt.plot(MAE_DT[2:])
plt.show() 

    
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in range(2,maxNode):
    MAE_RF[max_leaf_nodes] = np.exp(get_mae_RF(max_leaf_nodes, x, y))
print(np.min(MAE_RF[2:]))
plt.plot(MAE_RF[2:])
plt.show() 
