#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC  

# Read the data
df = pd.read_csv('train_data.csv', header=None)

# Read labels
labels_df = pd.read_csv('train_labels.csv', header=None)

# Read training set
test_df = pd.read_csv('test_data.csv', header=None)

# Read dummy_test
dummy_df = pd.read_csv('dummy_solution_accuracy.csv', index_col = 'Sample_id')
dummy_df_logloss = pd.read_csv('dummy_solution_logloss.csv', index_col = 'Sample_id')

# First 10 lines
df.head(10)

# Descriptive analysis of the data
descDf = df.describe().transpose()

# Observer min and max values 
equalValues = np.where(descDf['min'] == descDf['max'])[0]

# These columns does not give any useful information
# Dropping...
#df = df.drop(columns = equalValues)

# Delete same colums to the test_df
#test_df = test_df.drop(columns = equalValues)

# Get de values
listValues = df.values

#Standarizing the data 
# Standard deviation and vanrian equal to 1
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled)

test_scaled = scaler.fit_transform(test_df)
test_scaled_df = pd.DataFrame(test_scaled)

#################3 fit ########################
pca = PCA(.90)
pca.fit(scaled_df)
pca.n_components_


model_scaled_df = pca.transform(scaled_df)
model_test_scaled_df = pca.transform(test_scaled_df)

##############################################3


################### MODEL SVM ###################
svclassifier = SVC(kernel='poly', degree=8)  
svclassifier.fit(model_scaled_df, labels_df)  


lastPredictions = svclassifier.predict(model_test_scaled_df)  


##############################################




################### MODEL LOGISTIC REGRESSION ###################
logisticRegr = LogisticRegression(solver = 'lbfgs')

#With PCA
logisticRegr.fit(model_scaled_df, labels_df)
#Without PCA
logisticRegr.fit(scaled_df, labels_df)

# Returns a NumPy Array
# Predict for One Observation (image) PCA
logisticRegr.predict(model_test_scaled_df[0].reshape(1,-1))

logisticRegr.predict(model_test_scaled_df[0:10])
lastPredictions = logisticRegr.predict(model_test_scaled_df)

###### Log loss ######

lastPredictionLogLoss = logisticRegr.predict_proba(model_test_scaled_df)

#WithoutPCA
logisticRegr.predict(test_scaled[0].reshape(1,-1))

logisticRegr.predict(test_scaled[0:10])
lastPredictions = logisticRegr.predict(test_scaled)

###### Write output labels
dummy_df['Sample_label'] = lastPredictions
dummy_df.to_csv('../dummy_example_solution4.csv')
###### Write output logloss
dummy_df_logloss['Class_1'] = lastPredictionLogLoss[:,0]
dummy_df_logloss['Class_2'] = lastPredictionLogLoss[:,1]
dummy_df_logloss['Class_3'] = lastPredictionLogLoss[:,2]
dummy_df_logloss['Class_4'] = lastPredictionLogLoss[:,3]
dummy_df_logloss['Class_5'] = lastPredictionLogLoss[:,4]
dummy_df_logloss['Class_6'] = lastPredictionLogLoss[:,5]
dummy_df_logloss['Class_7'] = lastPredictionLogLoss[:,6]
dummy_df_logloss['Class_8'] = lastPredictionLogLoss[:,7]
dummy_df_logloss['Class_9'] = lastPredictionLogLoss[:,8]
dummy_df_logloss['Class_10'] = lastPredictionLogLoss[:,9]
dummy_df_logloss.to_csv('../dummy_logloss_solution4.csv')

################### fit transform ##############3
pca = PCA(n_components=0.90, whiten=True)
scaled_pca = pca.fit_transform(scaled_df)

print('Original number of features:', scaled_df.shape[1])
print('Reduced number of features:', scaled_pca.shape[1])

##############################################


#############################3
# Scaling the data 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(listValues)
df_scaled = pd.DataFrame(x_scaled)

df_descriptive = df_scaled.describe().transpose()

