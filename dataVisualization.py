# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 17:04:18 2019

@author: night
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import datetime as dt
import time

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def getTimestamp(date):
    return time.mktime(dt.datetime.strptime(date,"%d/%m/%Y").timetuple())


#import numpy.polynomial.polynomial.polyfit
dataFilename = 'imageFeatures_190321.csv'
old_data = pd.read_csv(dataFilename,encoding = "ISO-8859-1")

#Set features to visualize
feature_list = ['aspRatio', 'bluLRBal', 'bluTBBal', 'comms_num', 'created', 'globalContrast', 'grnLRBal', 'grnTBBal', 'imgSz', 'lumLRBal', 'lumTBBal', 'luminance', 'redLRBal', 'redTBBal', 'saturation', "redHue", "grnHue", "bluHue", "coarseContrast"]

#Set filtering parameters
beginTS = getTimestamp("01/01/2019")
logic_filt = old_data.created>beginTS #(old_data.score>1000)# & (old_data.score<30000)
filtered_data = old_data[logic_filt]
15
#fig, axs = plt.subplots(3, 5, sharey=True)
i=0
y = filtered_data.score

for feature in feature_list:
    
    x = filtered_data[feature].values
#    n = len(x)                          #the number of data
#    mean = sum(x*y)/n                   #note this correction
#    sigma = sum(y*(x-mean)**2)/n        #note this correction
#    
#    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])
#    plt.plot(x,gaus(x,*popt),'ro:',label='fit')
    
    x_new = np.linspace(min(x), max(x), num=len(x))
    coefs = poly.polyfit(x, y, 1)
    fitline = poly.polyval(x_new,coefs)
    
    
    
    plt.scatter(x, y)
    plt.plot(x_new, fitline, 'r--')
    plt.title(feature)
    plt.show()
