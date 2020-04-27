# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:35:44 2020

@author: Prince
"""

#natural language processing
#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv("C:/Users/Prince/Downloads/P14-Natural-Language-Processing/Natural_Language_Processing/Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

#cleaning the dataset


#removing non significant words
#like the,this,that,all etc

#stemming 
#its all about taking the roots of the word
#like (loved,loving,lovable)-love 
#just a simplfication to reduce the sparse matrix
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#creating the  bag of model
#by using process of tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()


#training machine learning models
y=dataset.iloc[:,1].values

#now classification
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



