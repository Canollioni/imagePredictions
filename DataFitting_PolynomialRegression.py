# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:57:44 2019

@author: night
"""
# Import function to create training and test set splits
from sklearn.cross_validation import train_test_split
# Import function to automatically create polynomial features! 
from sklearn.preprocessing import PolynomialFeatures
# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import time

def getTimestamp(date):
    return time.mktime(dt.datetime.strptime(date,"%d/%m/%Y").timetuple())

#Import data
dataFilename = 'imageFeatures_190321.csv'
imageDF = pd.read_csv(dataFilename,encoding = "ISO-8859-1")

#Set filtering parameters
beginTS = getTimestamp("01/01/2019")
logic_filt = imageDF.created>beginTS #(old_data.score>1000)# & (old_data.score<30000)
filtered_data = imageDF[logic_filt]

#Set variables
x = filtered_data[feature].values


#Select features to use for fit
#feature_list = ['aspRatio', 'bluLRBal', 'bluTBBal', 'globalContrast', 'grnLRBal', 'grnTBBal', 'imgSz', 'lumLRBal', 'lumTBBal', 'luminance', 'redLRBal', 'redTBBal', 'saturation']
feature_list = ['aspRatio', 'coarseContrast', 'globalContrast', 'imgSz', 'lumTBBal', 'luminance', 'saturation']

#Select features and scores
x = filtered_data[feature_list].values
y = filtered_data.score


# Alpha (regularization strength) of LASSO regression
lasso_eps = 0.001
lasso_nalpha=40
lasso_iter=500000
# Min and max degree of polynomials features to consider
degree_min = 7
degree_max = 9
# Test/train split
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.1)
# Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)

RMSE = np.zeros(degree_max-degree_min+1)
test_score = np.zeros(degree_max-degree_min+1)

for degree in range(degree_min,degree_max+1):
    model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,
normalize=True,cv=5))
    model.fit(X_train,y_train)
    test_pred = np.array(model.predict(X_test))
    RMSE[degree-degree_min]=np.sqrt(np.sum(np.square(test_pred-y_test)))
    test_score[degree-degree_min] = model.score(X_test,y_test)
    print(RMSE)

plt.plot(RMSE)
plt.show()
plt.plot(test_score)
plt.show()
    
