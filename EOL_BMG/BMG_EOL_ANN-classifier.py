# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:19:33 2020

@author: jschoeck

POSSIBLE IMPROVEMENTS:
    - Use Variance Analysis to perform feature selection (e.g. via PCA)
    
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing._encoders import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# dataset = pd.read_csv('example.csv', sep=';', decimal=',')
dataset = pd.read_csv('EOL_all_2016_combined.csv', sep=';', decimal=',')

'''
Dependant Variable:
    'error_code'
Independant Variables:
    'I0','N0','I_stall0','Te0','I1','I1_1','N1','I_stall1','M_stall1','Te1','Kt',
    'U2','N2','M2','CR','balance','Tr_2_H','isolation_resistance','I0_2','N0_2',
    'I_stall0_2','Te0_2','U1_2','I1_2','I1_1_2','N1_2','I_stall1_2','M_stall1_2',
    'Te1_2','Rm_2'
'''
df = dataset[['I0','N0','I_stall0','Te0','I1','I1_1','N1','I_stall1','M_stall1','Te1','Kt','U2',
              'N2','M2','CR','balance','Tr_2_H','isolation_resistance','I0_2','N0_2','I_stall0_2',
              'Te0_2','U1_2','I1_2','I1_1_2','N1_2','I_stall1_2','M_stall1_2','Te1_2','Rm_2',
              'error_code']]

# Data Analysis
# dataset.head()
# dataset.describe()
# dataset.columns.values
# dataset.error_code.unique()
# dataset.error_code.isna().sum() # Rows with NaN = 1
''' dataset.error_code.unique() --> 
-1., 38., 39., 30.,  3., 52., 55., 51., 33., 34., 37., 31., 53.,
1., 44., 54., 32., nan,  4., 21., 16., 35., 15., 56., 36.
'''
# df_y.isna().sum() # Rows with NaN = 0

df = df.dropna(axis=0)
df_y = df.pop('error_code')

for _ in range(0, len(df.columns.values)):
    # print('{}: INT? {}; {} unique; mean = {}'.format(
    print('{}: {} unique; mean = {}'.format(
        df.columns.values[_],
        # (df[df.columns.values[_]] % 1 == 0).all(), # check if MOD 1 can be calculated --> if column is INT or not
        # df[df.columns.values[_]].apply(float.is_integer).all(), # slower check if all values are INT or not
        len(df[df.columns.values[_]].unique()),
        round(df[df.columns.values[_]].mean(), ndigits=2)))
        # df[df.columns.values[_]].sample(5).values))
'''
All columns (up to Rm_2) contain FLOAT values.
Number of rows: 1037648    (-1 due to NaN in error_code)
'''

# Define dtypes
for _ in range(0, len(df.columns.values)):
    df[df.columns.values[_]].astype(float)
df_y = df_y.astype('int16')
# df.dtypes # All float64
# df_y.dtypes # Int16

# Define data for training
X = df.values
y = df_y.values

# np.unique(y, axis=0)
''' Unique error codes in y_train:
array([-1,  1,  3,  4, 15, 16, 21, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
       44, 51, 52, 53, 54, 55, 56], dtype=int16)
'''

'''
Model:
ANN classifier

https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
'''

# OneHotEncode y
# Creates sparse matrix with 0 or 1 / hot for each error_code category
encoder_y = OneHotEncoder()
y = encoder_y.fit_transform(y.reshape(-1,1))
# Translation of OneHotEncoded error_codes
y_features = pd.DataFrame(columns={'encoded', 'error_code'})
y_features.encoded = range(0,24)
y_features.error_code = encoder_y.get_feature_names()
y_features.error_code = y_features.error_code.apply(lambda x: x[3:])
y = y.toarray()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# 70/30 split to reduce risk of overfitting

# Normalize data / Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

input_dim = X.shape[1] # --> 30 features
output_dim =  y.shape[1] # --> 24 known unique values for error code
# Fitting classifier to the Training set
classifier = Sequential()
classifier.add(Dense(units=27, kernel_initializer='uniform', activation='relu', input_dim=input_dim))
classifier.add(Dense(units=27, kernel_initializer='uniform', activation='relu')) # Nodes in hidden layer = 1/2 sum(input+output)
classifier.add(Dense(units=output_dim, kernel_initializer='uniform', activation='softmax'))
# Compiling the ANN
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # try 'sgd' optimizer
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, epochs = 10)
# --> 98.41 % accuracy (optimizer adam)
# --> 98.22 % accuracy (optimizer sdg; faster than adam by ~10s/epoch)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)*1

# Transform back from OneHotEncoding to actual error_code

columns_X = {'I0','N0','I_stall0','Te0','I1','I1_1','N1','I_stall1','M_stall1','Te1','Kt','U2',
              'N2','M2','CR','balance','Tr_2_H','isolation_resistance','I0_2','N0_2','I_stall0_2',
              'Te0_2','U1_2','I1_2','I1_1_2','N1_2','I_stall1_2','M_stall1_2','Te1_2','Rm_2'}

df_results = pd.DataFrame(data=sc.inverse_transform(X_test), columns=columns_X)
# df_results = pd.DataFrame(data=encoder_y.inverse_transform(y_test), columns={'error_code'})
df_results['error_code'] = encoder_y.inverse_transform(y_test)
df_results['prediction'] = encoder_y.inverse_transform(y_pred)

wrong_predictions = df_results[df_results.error_code != df_results.prediction]
