# -*- coding: utf-8 -*-
"""
Created on Sun May 17 02:14:00 2020

@author: Johannes Schöck
For Kaggle competition Tweet Sentiment Extraction ( https://www.kaggle.com/c/tweet-sentiment-extraction )

STRATEGY:
    - 
    - X: 
    - y: 

DOCS:
    - https://www.nltk.org/api/nltk.sentiment.html
    - https://www.nltk.org/howto/sentiment.html
    - http://www.nltk.org/howto/

RESULTS:
    - 

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score # doesn't work on multi-feature data
# from sklearn.model_selection import cross_val_score # will be very slow, maybe also not functional for multi-feature data
from nltk.sentiment import SentimentAnalyzer
from tqdm import tqdm # ( Progress bar in iterable code with: tqdm(iterable); https://github.com/tqdm/tqdm )

def import_data():
    '''
    Load train.csv
    In future test.csv -> return both then and call with:
    df_train, df_test = import_data()
    '''
    dataset = pd.read_csv('train.csv', index_col=False) # take care for quotes etc.
    # dataset_test = pd.read_csv('test.csv')
    dataset.replace('nan', np.NaN, inplace=True)
    dataset.dropna(axis=0, inplace=True) # drop rows with NaN data
    dataset.reset_index(drop=True, inplace=True) # adds Index column :(
    # dataset.fillna('nan')
    return dataset.iloc[:, 1:] #dataset_test.iloc[:, 1:] ?
### DO NOT CLEAN tweets! Part of model input and submission content!

def feature_engineering(df):
    '''
    Create bag of words for text columns and return them as arrays.
    Transform based on fit for 'text' column, so features reference same words.
    '''    

    return df.iloc[:, -1]


def encode_df(df):
    '''
    Define X and y and apply LabelEncode on sentiment columns.
    Encode sentiment: 0 = negative, 1 = neutral, 2 = positive
    '''
    X_sentiments = df.values
    labelencoder_sentiment = LabelEncoder()
    X_sentiments = labelencoder_sentiment.fit_transform(X_sentiments)
    return (X_sentiments)

def split(X, y):
    '''
    Perform split of input data in test and train sets for X and y with sklearn.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    '''
    Train classifier with X_train and y_train using kNN classifier.
    '''
    classifier = 
    classifier.fit(X_train, y_train)
    return classifier

def predict(classifier, X_test):
    '''
    Predict classifier for X_test.
    '''
    return classifier.predict(X_test) # --> y_pred

def metrics(y_test, y_pred):
    '''
    Calculate Confusion Matrix for y_pred on y_test.
    '''
    cm = confusion_matrix(y_test, y_pred)
    print('Mean: {} %'.format(round(accuracy_score(y_test, y_pred)*100, 2)))
    return cm

def cross_validation(classifier, X_train, y_train):
    '''
    Do Cross-validation for y_pred on y_test, returning mean and standard deviation.
    '''
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)
    print('Mean: {} %\nStd: {} %'.format(round(accuracies.mean()*100, 2),
                                         round(accuracies.std()*100, 2)))
    return accuracies

def main():
    '''
    Run program.
    '''
    df = import_data()
    # df_train, df_test = import_data()
    df = feature_engineering(df)
    X_sentiments = encode_df(df)
    X_train, X_test, y_train, y_test = split(X, y)
    classifier = train(X_train, y_train)
    y_pred = predict(classifier, X_test)
    metrics(y_test, y_pred)
    cross_validation(classifier, X_train, y_train)
    return True

main()

'''

'''