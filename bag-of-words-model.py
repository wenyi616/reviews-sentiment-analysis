#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:51:22 2019

@author: wenyi
"""

import pandas as pd
import numpy as np
import sys, re, string, unidecode

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

# np.set_printoptions(threshold=np.inf)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# helper function for lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# data (np.array) -> new_data: np.array
def pre_process_data(data):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    new_data = []
    for i in range(len(data)):
        review = data[i]
        # convert to lowercase
        review = review.lower()

        # add whiltespace after punctuation
        review = re.sub( r'([a-zA-Z])([,.!])', r'\1\2 ', review)

        # remove punctuation
        review = review.translate(string.maketrans("",""), string.punctuation)

        filtered_review = "" 
        for w in word_tokenize(review): 
            # Lemmatize with POS Tag
            try:
                w = lemmatizer.lemmatize(w, get_wordnet_pos(w))
            except:
                # fiancé, café, crêpe, puréed
                # w = unidecode.unidecode(unicode(w, "utf-8"))
                continue

            # remove stop word
            if w not in stop_words:
                filtered_review =filtered_review + w + " "

        review = filtered_review
        new_data.append(review)  

    return np.array(new_data)


# load data
amazon_df = pd.read_csv('./data/amazon_cells_labelled.txt', sep="\t", header=None)
amazon_df.columns = ['Reviews', 'Labels']
yelp_df = pd.read_csv('./data/yelp_labelled.txt', sep="\t", header=None)
yelp_df.columns = ['Reviews', 'Labels']
imdb_df = pd.read_csv('./data/imdb_labelled.txt', sep="\t+", header=None, engine='python')
imdb_df.columns = ['Reviews', 'Labels']

# check if labels are balanced
# print(amazon_df.query('Labels == "0"').Reviews.count())
# print(amazon_df.query('Labels == "1"').Reviews.count())

# seperating training and testing set
amazon_training_df = pd.concat([amazon_df[amazon_df['Labels']==0].iloc[:400], amazon_df[amazon_df['Labels']==1].iloc[:400]])
amazon_testing_df = pd.concat([amazon_df[amazon_df['Labels']==0].iloc[-100:], amazon_df[amazon_df['Labels']==1].iloc[-100:]])

yelp_training_df = pd.concat([yelp_df[yelp_df['Labels']==0].iloc[:400], yelp_df[yelp_df['Labels']==1].iloc[:400]])
yelp_testing_df = pd.concat([yelp_df[yelp_df['Labels']==0].iloc[-100:], yelp_df[yelp_df['Labels']==1].iloc[-100:]])

imdb_training_df = pd.concat([imdb_df[imdb_df['Labels']==0].iloc[:400], imdb_df[imdb_df['Labels']==1].iloc[:400]])
imdb_testing_df = pd.concat([imdb_df[imdb_df['Labels']==0].iloc[-100:], imdb_df[imdb_df['Labels']==1].iloc[-100:]])

# training
training_data = np.concatenate((np.array(amazon_training_df["Reviews"]),
    np.array(yelp_training_df["Reviews"]), np.array(imdb_training_df["Reviews"])),axis=0)
training_label = np.concatenate((np.array(amazon_training_df["Labels"]),
    np.array(yelp_training_df["Labels"]), np.array(imdb_training_df["Labels"])),axis=0)

# testing
testing_data = np.concatenate((np.array(amazon_testing_df["Reviews"]),
    np.array(yelp_testing_df["Reviews"]), np.array(imdb_testing_df["Reviews"])),axis=0)
testing_label = np.concatenate((np.array(amazon_testing_df["Labels"]),
    np.array(yelp_testing_df["Labels"]), np.array(imdb_testing_df["Labels"])),axis=0)

print("****** Done loading data ******")

training_data = pre_process_data(training_data)
testing_data = pre_process_data(testing_data)
print(training_data.shape)
print(testing_data.shape)
print("****** Done pre-prossing data ******")

# Bag of Words Model 
word_dict = {}

# iterate thru all reviews in the training set
for i in range(len(training_data)):
    review = training_data[i]
    
    for w in word_tokenize(review):
        if w not in word_dict:
            word_dict[w] = 0

print(len(word_dict)) #3846
print("****** Done building word dictionary ******")

# iterate all reviews in both sets to create review feature vectors
# training vector of size (2400*3905) 
training_vectors = []

for i in range(len(training_data)):
    # initialize a vector of size (1*3905) 
    review_vector = np.zeros(len(word_dict))
    review = training_data[i]

    for w in word_tokenize(review):
        if w in word_dict:
            index = word_dict.keys().index(w)
            review_vector[index] += 1   
    
    training_vectors.append(review_vector)


# # testing vector of size (600*3905) 
testing_vectors = []
for i in range(len(testing_data)): 
    review_vector = np.zeros(len(word_dict))
    
    review = testing_data[i]
    for w in word_tokenize(review):
        if w in word_dict:
            index = word_dict.keys().index(w)
            review_vector[index] += 1   
    
    testing_vectors.append(review_vector)


training_vectors = np.array(training_vectors)
print(training_vectors.shape)
print(training_vectors)

testing_vectors = np.array(testing_vectors)
print(testing_vectors.shape)
# print(testing_vectors)
print("****** Done building training & testing feature vectors ******")


# L2 normalization
training_vectors_norm = preprocessing.normalize(training_vectors, norm='l2')
testing_vectors_norm = preprocessing.normalize(testing_vectors, norm='l2')
print("****** Done normalizations ******")


# Sentiment prediction using Logistic Regression
logreg = LogisticRegression()
logreg.fit(training_vectors_norm, training_label)
pred_logreg = logreg.predict(testing_vectors_norm)
acc_logreg = accuracy_score(pred_logreg, testing_label)  
cm_logreg = confusion_matrix(testing_label, pred_logreg)

# print Logistic Regression results
print("\n**** Logistic Regression ****")   
print("Accuracy = %.2f" % acc_logreg)
print(cm_logreg)
print("****** Done reporting performance ******")

# returns a matrix of weights (coefficients)
# AKA, returns the index of most significant words (take negation to argsort in descending order) 
co = np.argsort(np.negative(np.absolute(logreg.coef_)))[0] 


# Sentiment prediction using Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(training_vectors_norm, training_label)
pred_gnb = gnb.predict(testing_vectors_norm)
acc_gnb = accuracy_score(pred_gnb, testing_label)  
cm_gnb = confusion_matrix(testing_label, pred_gnb)

# print Naive Bayes results
print("\n**** Naive Bayes  ****")   
print("Accuracy = %.2f" % acc_gnb)
print(cm_gnb)
print("****** Done reporting performance ******")

# print 10 most significant words
important_words = []
for i in range(10):
    important_words.append(word_dict.keys()[co[i]])
print(important_words)
print("****** Well Done! ******")



# implement PCA to reduce the dimension of features (use SVD to perform PCA)
from numpy import linalg

# X - feature vector, p - number of features in the reduced matrix 
def my_pca_implementation(X, p):
    X -= np.mean(X, axis=0)
        
    # p x p covariance matrix
    C = np.cov(X, rowvar=False)

    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]

    evecs, evals = evecs[:, idx], evals[idx]
    evecs, evals = evecs[:, :p], evals[:p]
    principal_components = evecs.T.dot(X.T).T
    
    return principal_components

def run_PCA_then_LR(p):
    training_vectors_principle = my_pca_implementation(training_vectors_norm, p)
    testing_vectors_principle = my_pca_implementation(testing_vectors_norm, p)

    # print(training_vectors_principle.shape)
    # print(testing_vectors_principle.shape)

    # run Logistic Regression (w/ PCA applied)
    logreg.fit(training_vectors_principle, training_label)
    pred = logreg.predict(testing_vectors_principle)
    acc_2 = accuracy_score(pred,testing_label)  
    cm_2 = confusion_matrix(testing_label,pred)

    print("\n**** Logistic Regression (PCA p=%d)****" % p)   
    print("Accuracy = %.2f" % acc_2)
    print(cm_2)

    # returns a matrix of weights (coefficients)
    # AKA, returns the index of most significant words (take negation to argsort in descending order) 
    co_2 = np.argsort(np.negative(np.absolute(logreg.coef_)))[0] 

    # print 10 most significant words
    important_words_2 = []
    for i in range(10):
        important_words_2.append(word_dict.keys()[co_2[i]])
    print(important_words_2)

# # Double check using existing implementation for PCA
# from sklearn.decomposition import PCA

# pca = PCA(n_components=100)
# training_vectors_principle = pca.fit_transform(training_vectors_norm)
# testing_vectors_principle = pca.fit_transform(testing_vectors_norm)

# run_PCA_then_LR(10)
# run_PCA_then_LR(50)
# run_PCA_then_LR(100)