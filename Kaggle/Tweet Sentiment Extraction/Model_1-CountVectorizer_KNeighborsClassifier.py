# -*- coding: utf-8 -*-
"""
Created on Sun May 10 00:16:00 2020

@author: Johannes Schöck
For Kaggle competition Tweet Sentiment Extraction

TO DO:
    - Combine predicted words with original textID
    - Output submission csv file
    - Perfom training and prediction on shortened training data set
    - DONE  Calculate Jaccard score
    - Remove nrows=1000 in import_data for final training
    - Perform final training on 100% of data set
    - Predict with trained model on test data set


STRATEGY:
    - Make bag of words for both text columns
    - X: total_text bag of words and sentiment
    - y: selected_text bag of words

RESULTS:
    - kNN classifier on bag of words yields bad result
    - sentiment feature irrelevant with the high number of features from bag of words
    - almost all words get selected
    - important words are often not selected for feature matrix with CountVectorizer.fit

    --> try NLTK sentiment analyzer
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score # doesn't work on multi-feature data
# from sklearn.model_selection import cross_val_score # will be very slow, maybe also not functional for multi-feature data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

def import_data():
    '''
    Load train.csv
    In future test.csv -> return both then and call with:
    df_train, df_test = import_data()
    '''
    dataset = pd.read_csv('train.csv', index_col=False, nrows=1000) # take care for quotes etc.
    dataset_test = pd.read_csv('test.csv')
    dataset.replace('nan', np.NaN, inplace=True)
    dataset.dropna(axis=0, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    dataset_test.replace('nan', np.NaN, inplace=True)
    dataset_test.dropna(axis=0, inplace=True)
    dataset_test.reset_index(drop=True, inplace=True)
    return dataset, dataset_test
### DO NOT CLEAN tweets! Part of model input and submission content!

def feature_engineering(df):
    '''
    Create bag of words for text columns and return them as arrays.
    Transform based on fit for 'text' column, so features reference same words.
    '''
    corpus = [] # Create bag of words from 'text' column
    for i in range(0, df.text.size):
        text = df['text'][i]
        # No split needed - CountVectorizer needs single string input, not list
        # try:
        #     text = text.split()
        # except:
        #     pass
        corpus.append(text)
    cv = CountVectorizer(max_features=1500) # Play around, maybe grid_search
    X_text = cv.fit_transform(corpus).toarray()
    
    corpus = [] # Create bag of words from 'selected_text' column
    for i in range(0, df.selected_text.size):
        text = df['selected_text'][i]
        corpus.append(text)
    '''
    Must use same fit as for total_words to result in same words as features!
    Proven by difference in X_selected_words.sum() and comparison of dictionaries that are created by cv
    '''
    y_selected_words = cv.transform(corpus).toarray()
    df = df.drop(labels={'text', 'selected_text'}, axis=1)

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

# def split(X, y):
#     '''
#     Perform split of input data in test and train sets for X and y with sklearn.
#     '''
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#     return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    '''
    Train classifier with X_train and y_train using kNN classifier.
    '''
    classifier = KNeighborsClassifier()
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

def output(df_textID, df_prediction):
    '''    
    For each ID in the test set, you must predict the string that best supports
    the sentiment for the tweet in question. Note that the selected text needs
    to be quoted and complete (include punctuation, etc. - the above code splits
    ONLY on whitespace) to work correctly. The file should contain a header and
    have the following format:
        textID,selected_text
        2,"very good"
    '''
    sub_df = pd.DataFrame()
    sub_df['textID'] = df_textID['textID']
    sub_df['prediction'] = df_prediction['prediction']
    # sub_df = pd.DataFrame(data, columns=['textID', 'selected_text'])
    sub_df.to_csv('submit_CV_kNC_20200517A.csv', index=False)
    return True

if __name__ == "__main__" :
    df_train, df_test = import_data()
    cv, df_train, X_text, y = feature_engineering(df_train)
    sentiments = encode_df(df_train['sentiment'])
    X = build_X(X_text, sentiments)
    # X_train, X_test, y_train, y_test = split(X, y) # Don't use split, instead use test data file
    classifier = train(X, y)
    X_test = cv.transform(df_test['text']).toarray()
    # use model fitted bag of words for test data and make prediction based on that?
    # y_pred = predict(classifier, X) # test prediction on training data itself
    y_pred = predict(classifier, X_test)
    # y_pred = predict(classifier, df_test.drop(labels='textID', axis=1))
    
    # df bauen für Bewertung und submission
    # y_pred -> 
    # bag of words mit tatsächlichen Worten ersetzen
    df_result = pd.DataFrame(cv.inverse_transform(X_text))
    
    
    jacc_score(df)

'''
# CHECK RESULTS:
#  Compare df_result with df.selected_text
df_result = pd.DataFrame(cv.inverse_transform(X_text))
df['prediction'] = df_result[df_result.columns[1:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1)
'''