# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:19:33 2020

@author: jschoeck
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# dataset = pd.read_csv('example.csv', sep=';', decimal=',')

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

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalize data / Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
Model:
KNN classifier
'''
# Fitting classifier to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
df_result = pd.DataFrame(data=(y_test, y_pred))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_norm = confusion_matrix(y_test, y_pred, normalize='pred')
true_positives = np.sum(np.diag(cm))
prediction_ratio = round(true_positives/np.sum(cm), ndigits=3)
print('\nModel: kNN classifier\nCorrect predictions: {}\nTotal test points:\n{}\nPercentage: {} %'.format(true_positives, np.sum(cm), prediction_ratio*100))
# --> 98.2 %

'''
Possible future improvements:
- Cross-validation
- Grid_search
'''