# -*- coding: utf-8 -*-
"""
Created on Sun May 10 00:16:00 2020

@author: Johannes Schöck
For Kaggle competition Tweet Sentiment Extraction

ATTENTION - model analyzes sentiment as dependant variable, NOT selected_text!

STRATEGY:
    - Use text word count to predict sentiment
    - Use selected_text word count to predict sentiment
    - Compare with random selection
    - Analyze effect of cleaning --> not performed

RESULTS:
    - Randomized                                 --> 34 %
    - Word count "total" before cleaning           --> 36 +- 2 %
    - Word count "selected" before cleaning        --> 53 +- 2 %
    
"""

# Tweet_Sentiment_Extraction_EAIML#1
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import nltk
# nltk.download('stopwords')
# import re
from sklearn.preprocessing import LabelEncoder

def import_data():
    dataset = pd.read_csv('train.csv') # take care for quotes etc. for real import
# print(dataset.head(50))
# print(dataset.sentiment.describe()) # negative, neutral, positive
    return dataset.iloc[:,1:]

def word_count(words):
    return len(str(words).split())

def feature_engineering(df):
    # Add word count as feature
    df.insert(loc = len(df.columns), column='WordCount', value=None)
    return df

def count_words(df):
    # 1. Word count total before cleaning
    # df['WordCount'] = df.apply(lambda x: word_count(x['text']),axis=1)
    # 2. Word count selected before cleaning
    df['WordCount'] = df.apply(lambda x: word_count(x['selected_text']),axis=1)
    # 3. Word count total after cleaning
    # df['WordCount'] = df.apply(lambda x: word_count(x['text']),axis=1)
    # 4. Word count selected after cleaning
    # df['WordCount'] = df.apply(lambda x: word_count(x['selected_text']),axis=1)
    return df

### CLEANING, stopwords...
def clean_tweets(df):
    for i in range(0, df.text.size):
        df['text'][i] = str(df.text[i]).lower()
    for i in range(0, df.selected_text.size):
        df['selected_text'][i] = str(df.selected_text[i]).lower()        
    return df


def encode_df(df):
    X = df.iloc[:, 3].values.reshape(-1, 1) # pick wordcount as X
    y = df.iloc[:, 2].values

    # Encode sentiment: 0 = negative, 1 = neutral, 2 = positive
    labelencoder_sentiment = LabelEncoder()
    y = labelencoder_sentiment.fit_transform(y)
    return (X, y)

# CHECK (linearity, variance, codependence)
# plt.hist(X[:,0])
# plt.hist(X[:,1])
# plt.xlabel('0/1/2 & Word Count')
# plt.ylabel('Occurences')
# plt.show()
# sns.relplot(data=pd.DataFrame(X))

def split(X, y):
    # SPLIT DATA
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

# TRAIN MODEL
def train(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    return classifier

def predict(classifier, X_test):
    return classifier.predict(X_test) # --> y_pred

def predict_random(X_test):
    y_pred = [0] * X_test.size
    # 0. Randomized y
    from random import seed, randint
    seed(0)
    for i in range(X_test.size):
        y_pred[i] = randint(0,2)
    return y_pred

def metrics(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    print('Mean: {} %'.format(round(accuracy_score(y_test, y_pred)*100, 2)))
    return cm    

def cross_validation(classifier, X_train, y_train):
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print('Mean: {} %\nStd: {} %'.format(round(accuracies.mean()*100, 2), round(accuracies.std()*100, 2)))
    return

# VISUALIZE
# plt.scatter(X[:,1], X[:,0])
# plt.show()

def main():
    df = import_data()
    feature_engineering(df)
    count_words(df)
    clean_tweets(df)
    X, y = encode_df(df)
    X_train, X_test, y_train, y_test = split(X, y)
    classifier = train(X_train, y_train)
    
    # Select one model:
    # y_pred = predict_random(X_test)
    y_pred = predict(classifier, X_test)
    
    cm = metrics(y_test, y_pred)
    cross_validation(classifier, X_train, y_train)
    return

main()