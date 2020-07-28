# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 23:12:01 2020

@author: Ritik Jain
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Foodtruck.csv')
features= df.iloc[:,0].values
labels= df.iloc[:,1].values

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train.reshape(-1,1),labels_train)

regressor.predict(np.array(3.073).reshape(-1,1))

plt.scatter(features_train,labels_train, color="red")
plt.plot(features_train,regressor.predict(features_train), color="blue")
plt.title("predicted graph")
plt.xlabel('y of exp')
plt.ylabel("salary")
plt.show()

plt.scatter(features_test,labels_test, color="red")
plt.plot(features_test,regressor.predict(features_test), color="blue")
plt.title("predicted graph")
plt.xlabel('y of exp')
plt.ylabel("salary")
plt.show()