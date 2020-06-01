# -*- coding: utf-8 -*-
"""
Created on Sun May 10 00:16:00 2020

@author: Johannes Schöck
For Kaggle competition Tweet Sentiment Extraction

TO DO:
    --> Submit as notebook online

STRATEGY:
    - Make bag of words for both text columns
    - X: total_text bag of words and sentiment
    - y: selected_text bag of words

RESULTS:
    - Jaccard score of kNN model with full training set = 0.235
    - kNN classifier on bag of words yields bad result
    - sentiment feature irrelevant with the high number of features from bag of words
    - almost all words get selected
    - important words are often not selected for feature matrix with CountVectorizer.fit

    --> try NLTK sentiment analyzer
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

def import_data():
    '''
    Load train.csv
    In future test.csv -> return both then and call with:
    df_train, df_test = import_data()
    '''
    if spyder():
        path = ''
    else:
        path = '../input/tweet-sentiment-extraction/'
    # dataset = pd.read_csv(path+'train.csv', index_col=False, nrows=1000)
    dataset_train = pd.read_csv(path+'train.csv', index_col=False)
    dataset_test = pd.read_csv(path+'test.csv')
    # DO NOT CLEAN tweets! Part of model input and submission content!
    dataset.replace('nan', np.NaN, inplace=True)
    dataset.dropna(axis=0, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    dataset_test.replace('nan', np.NaN, inplace=True)
    dataset_test.dropna(axis=0, inplace=True)
    dataset_test.reset_index(drop=True, inplace=True)
    return dataset, dataset_test

def feature_engineering(df, max_words):
    '''
    Create bag of words for text columns and return them as arrays.
    Transform based on fit for 'text' column, so features reference same words.
    '''
    corpus = [] # Create bag of words from 'text' column
    for i in tqdm(range(0, df.text.size)):
        text = df['text'][i]
        corpus.append(text)
    cv = CountVectorizer(max_features=max_words) # Play around, maybe grid_search
    X_text = cv.fit_transform(corpus).toarray()
    
    corpus = [] # Create bag of words from 'selected_text' column
    for i in range(0, df.selected_text.size):
        text = df['selected_text'][i]
        corpus.append(text)
    '''
    Must use same fit as for total_words to result in same words as features!
    Proven by difference in X_selected_words.sum() and comparison of dictionaries
    that are created by cv
    '''
    y_selected_words = cv.transform(corpus).toarray()
    return cv, df, X_text, y_selected_words

def build_X(X_text, sentiments):
    '''
    Combine encoded sentiment with bag of words for text via a df to build X.
    '''
    df = pd.DataFrame(X_text)
    df['sentiment'] = sentiments
    return df.values

def encode_df(df):
    '''
    Define X and y and apply LabelEncode on sentiment columns.
    Encode sentiment: 0 = negative, 1 = neutral, 2 = positive
    '''
    X_sentiments = df.values
    labelencoder_sentiment = LabelEncoder()
    X_sentiments = labelencoder_sentiment.fit_transform(X_sentiments)
    return (X_sentiments)

def train(X_train, y_train):
    '''
    Train classifier with X_train and y_train using kNN classifier.
    '''
    classifier = KNeighborsClassifier(n_jobs=-1)
    classifier.fit(X_train, y_train)
    return classifier

def predict(classifier, X_test):
    '''
    Predict classifier for X_test.
    '''
    return classifier.predict(X_test) # --> y_pred

def jaccard(df):
    '''
    gt = ground truth, pt = prediction
    Jaccard score between submission string and correct answer string.
    '''
    gt = set(str(df['selected_text']).lower().split()) 
    pt = set(str(df['prediction']).lower().split())
    c = gt.intersection(pt)
    return float(len(c)) / (len(gt) + len(pt) - len(c))

def jacc_score(df):
    '''
    Official grading mechanism for this competition.
    score=1/n ∑{n;i=1}(jaccard(gt_i, dt_i)) # gt=ground_truth, dt=prediction
    
    Input dataframe with column for selected_text and result.
    Apply Jaccard scoring on all of them according to algorithm.
    '''
    '''
    EXAMPLE df
    strs1 = ['in miss sad san sooo will you', 'is me my', 'interview leave me what']
    strs2 = ['Sooo SAD', 'bullying me', 'leave me alone']
    data = {'prediction': strs1, 'selected_text': strs2}
    df = pd.DataFrame(data)
    df['jaccard'] = df.apply(jaccard, axis=1)
    '''
    summed_score = 0
    for i in range(0, df.size):
        summed_score += jaccard(df)
    return summed_score/df.size

def spyder():
    from os import environ
    if any('SPYDER' in name for name in environ):
        return True
    else:        
        return False

if __name__ == "__main__" :
    # maximum number of features for bag of words model
    max_words = 1500
    # Import training and test data sets separately
    df_train, df_test = import_data()
    # Create bag of words for training data
    cv, df_train, X_train, y_train = feature_engineering(df_train, max_words)
    # Encode sentiment feature
    sentiments_train = encode_df(df_train['sentiment'])
    sentiments_test = encode_df(df_test['sentiment'])
    # Add encoded sentiment to bag of words for training data
    X_train = build_X(X_train, (sentiments_train+1)*max_words) # Increase weight of sentiment by size of bag of words
    # Train on training set X and y
    classifier = train(X_train, y_train)
    
    # Create bag of words for test data, based on trained model
    X_test = cv.transform(df_test['text']).toarray()
    # Add encoded sentiment to bag of words for test data
    X_test = np.insert(X_test, max_words, sentiments_test, axis=1)
    
    # Predict on training data itself
    y_pred_train = predict(classifier, X_train) 
    # Predicton test data
    y_pred_test = predict(classifier, X_test)
    
    # Reverse bag of words to actual words, leaving out sentiment
    df_result_train = pd.DataFrame(cv.inverse_transform(X_train[:,:-1]))
    # Combine selected words from bag of words model into one column
    df_train['prediction'] = df_result_train[df_result_train.columns[1:]].apply(
        lambda x: ','.join(x.dropna().astype(str)), axis=1)
    
    # Reverse bag of words to actual words, leaving out sentiment
    df_result_test = pd.DataFrame(cv.inverse_transform(X_test[:,:-1]))
    # Combine selected words from bag of words model into one column
    df_test['selected_text'] = df_result_test[df_result_test.columns[1:]].apply(
        lambda x: ','.join(x.dropna().astype(str)), axis=1)
    
    df_sub = df_test.drop(labels={'text', 'sentiment'}, axis=1)
    
    if spyder():
        print('\nModel Jaccard score: {}'.format(round(jacc_score(df_train), ndigits=3))) # pass over prediction and selected_text
        df_sub.to_csv('submit_CV_kNC_20200601B.csv', index=False) # Offline simulation
    else:
        df_sub.to_csv('../working/submission.csv', index=False) # Online submission
