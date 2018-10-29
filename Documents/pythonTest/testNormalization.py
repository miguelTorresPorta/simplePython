#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:20:27 2018

@author: migueltorresporta
"""

from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get dataset
df = pd.read_csv("california_housing_train.csv", sep=",")

df_descriptive = df.describe().transpose()


plt.hist(df.total_bedrooms,bins='auto')
plt.hist(df.median_income,bins='auto')
plt.hist(df.median_house_value,bins=1000)




#################### STANDARIZATION ####################


# Get column names first
names = df.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)

df_descriptive_standarized = scaled_df.describe().transpose()

plt.hist(scaled_df.median_income,bins=1000)




#################### NORMALIZATION ####################
# Normalize total_bedrooms column
total_beds_array = np.array(df['total_bedrooms'])
normalized_total_beds = preprocessing.normalize([total_beds_array])
df.total_bedrooms = pd.DataFrame(normalized_total_beds.T)[0]

# Normalize median house value column
total_median_house = np.array(df['median_house_value'])
normalized_median_house = preprocessing.normalize([total_median_house])
df.median_house_value = pd.DataFrame(normalized_median_house.T)[0]




plt.hist(normalized_X)


plt.hist(df.median_income,bins='auto')
