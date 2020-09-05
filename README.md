# Johannes Schöck
# Data Science and Machine Learning Portfolio

## Motivation
After spending a few years working in different fields, I found the type of work where my motivation and passion always was at its max: *working with data*.

My goal is to make the career switch some time between spring 2020 and spring 2021.

Since spring 2019 I am going after this goal by enhancing and completing my skillset as a data scientist, both in theory and practical work. This repository showcases most of my work as far as possible due to trade secrets. It is a growing, living collection with content from different periods of my self-learning journey, so the quality and aptness of how I use tools can change between projects.

If you like my work or have any questions, please contact me at johannes(at)schoeck(dot)org or find me on [LinkedIn](https://www.linkedin.com/in/johannes-sch%C3%B6ck-a87547195/).

## Content Overview
[1. Skills, Tools and Domain Knowledge](#1-skills)
[2. Work Data Projects](#2-work-data-projects)
[3. Case Studies](#3-case-studies)
[4. Private Data Projects and Demos](#4-private-data-projects-and-demos)
[5. Kaggle Competitions](#5-competitions)

### 1. Skills
- Problem analysis
- Communication
- International experience
- Presentation and sharing of insights
- Interdisciplinary mindset
- Data analysis
- Data cleaning and wrangling
- Regression models for classification and quantitative prediction
- Geodemographic segmentation
- Machine learning models for regression and classification
- Neural network techniques for estimation and binary- and multi-classification of structured and unstructured data
- Simple generative adversial networks (GAN)

#### Data Science Tools
- Python
- Pandas
- Numpy
- Matplotlib / Seaborne
- SciKit-Learn
- NLTK
- Tensorflow / Keras
- BeautifulSoup
- Google Colab / AutoML
- Jupyter notebook
- Spyder IDE

#### Domain Knowledge
- Semiconductor design, production and testing
- Automotive product development and quality assurance
- Electronics production and testing
- Data analysis of serial production testers in automotive and electronics production lines
- Research data aggregation, analysis and publishing
- NGO structures and membership analysis

### 2. Work Data Projects
#### [CVD_prediction](https://github.com/JSchoeck/portfolio/tree/master/CVD_prediction)
Developed a model to predict oxide layer thickness in a TEOS CVD process. Training data based on a design of experiment (DOE) concept was used and led to a prediction quality of 98%. Incorporating a physics-based model to include temperature variation, without T being available from the training data, made the project especially powerfull. The ANN put out 49 measurement points on the wafer, allowing to precisely simulate real CVD tools.
Creating a simple GUI with Tkinter demonstrated the ability to roll out the prediction tool to end-users.
- Tools: Python, Keras/Tensorflow, Pandas, numpy, matplotlib, Tkinter

#### [EOL_BMG](https://github.com/JSchoeck/portfolio/tree/master/EOL_BMG)
Analysis and classification of serial production end-of-line tester data using a kNN-classifier and an ANN on the error code as a OneHotEncoded multi-class dependant variable. The dataset consists of over 1 M observations with 30 selected features. Model quality was quantified using appropriate accuracy measurements for each model.
The results yield insight into reasons of failure and help increase the yield of the production line.
- Tools: Python, Pandas, numpy, matplotlib, scikit-learn, Keras/Tensorflow

#### [SPEA_csv_import](https://github.com/JSchoeck/portfolio/tree/master/SPEA_csv_import)
A tool to deal with data from an end of line tester in a production environment for electronics (PCBA production). Imports large amount of original csv data files, stiches them together, selects relevant features, handles missing and incomplete data and tests, adds features and creates different plots with insights into tests results. Compresses data to store whole dataset in memory.
- Tools: Python, Pandas, numpy, matplotlib

### 3. Case Studies
#### [Human Resources - Employee Attrition](https://github.com/JSchoeck/portfolio/blob/master/Demos/Case%20Study%20Human%20Resources/Case%20Study%20Human%20Resources.ipynb)
Analyzed a realistic HR data set regarding employee attrition. Asked and tried to answer questions like who leaves, why do they leave and what prevents employees from leaving? Created different models to predict employee attrition: Logistic Regression, Random Forrest, dense ANNs. I added F1 score as a custom metric to the ANN model and introduced early stopping using the metrics Accuracy and F1 score.
- Tools: Python, Pandas, Numpy, matplotlib, Seaborn, Scikit-learn, Keras

### 4. Private Data Projects and Demos
#### [GAN_1D_Keras](https://github.com/JSchoeck/portfolio/blob/master/Demos/GAN_1D/GAN_1D_Keras.ipynb)
Created a generative adversial network that learns to output increasingly indistinguishable data points from an original mathematical function. Training of the generator model happens via classification by a discriminator model, which adjusts the generator model's weights in a combined logical GAN model. Progress of the training can be watched by plots of both real and generated data points, as well as the classification accuracy of the discriminator.
- Tools: Python, Pandas, Numpy, Keras, matplotlib

#### [Disc Sports Twitter Sentiment Tracker](https://github.com/JSchoeck/portfolio/tree/master/DiscSports/Disc_Sports_Twitter_Sentiment_Tracker)
Tracking of all tweets containing the word 'frisbee' on Twitter and running a sentiment analysis on it. The results are visualized and saved in a csv file. Actually running 'in production' and updating the dataset and plots daily. The [latest plots](https://www.dropbox.com/sh/dmhv503ni3q0sb0/AABsV2t47-KIwS74RsZ3HRLOa?dl=0) are available online, including an [interactive plot](https://www.dropbox.com/s/m0scddrx0aaxk41/Daily_number_of_%27frisbee%27_tweets_per_sentiment_line_latest.html?dl=0) using Bokeh (not viewable in Dropbox).
- Tools: Python, Pandas, Twython, NLTK, VADER, matplotlib

#### [German Ultimate Frisbee Clubs Count per State](https://github.com/JSchoeck/portfolio/blob/master/DiscSports/German%20Ultimate%20Frisbee%20Clubs%20Count%20per%20State.ipynb)
A small web-scraping project using BeautifulSoup to get the number of Ultimate Frisbee clubs in each state federation. The result is displayed on a map in Tableau.
[Tableau map](https://public.tableau.com/profile/johannes.sch.ck#!/vizhome/DFV_Vereine_nach_Bundesland/Dashboard1)
- Tools: Python, Pandas, BeautifulSoup, Tableau

### 5. Competitions
#### [Tweet Sentiment Extraction](https://github.com/JSchoeck/portfolio/tree/master/Kaggle/Tweet%20Sentiment%20Extraction)
Text analysis challenge to find the parts of a tweet that have been associated with positive, neutral or negative sentiments. Used bag of word method to feed into kNN classification algorithm. Also played with the data to predict which sentiment would be assigned based on word count of the tweet or of the selected part of it and compared it to a random selection model. As this was my first serious attempt at NLP, my goal was to build a functional ML framework, not to find an optimized model.
Kaggle submission page: https://www.kaggle.com/jschoeck/competitions
- Tools: Python, Pandas, NLTK, scikit-learn

#### [Titanic](https://github.com/JSchoeck/portfolio/tree/master/Kaggle/Titanic)
Classic data analysis / machine learning entry classification data set to predict survivors of the Titanic accident. Analyzed the data, performed feature selection and engineering. Applied different classification algorithms and compared their performance with hyperparameter optimization.
Kaggle submission page: https://www.kaggle.com/jschoeck/competitions
- Tools: Python, Pandas, numpy, matplotlib, scikit-learn
