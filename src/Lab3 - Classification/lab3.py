import sys
import numpy as np
import csv
import time
import gc
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split


def split_data(filename,split_ratio):
	
	dataMatrix = []
	train_data = []
	test_data = []
	train_labels = []
	test_labels = []

	with open(filename, 'rb') as csv_file:
	    rowHeaders = next(csv_file)
	    for row in csv_file:
	        dataMatrix.append([x for x in row.split(",")])

	train, test = train_test_split(dataMatrix, test_size=split_ratio, random_state=42)
	
	for each in train:
		train_labels.append(each[0])
		train_data.append(each[2:])

	for each in test:
		test_labels.append(each[0])
		test_data.append(each[2:])

	# print len(train_labels), len(X), len(test_labels), len(Y)

	return np.array(train_data).astype(np.float),np.array(test_data).astype(np.float),np.array(train_labels),np.array(test_labels)

def KNNClassifier(filename, split_ratio):
	print "-"*15,"K Nearest Neighbors classfier","-"*15
	
	X, Y, X_labels, Y_labels = split_data(filename,split_ratio)

	# print X.shape, Y.shape, X_labels.shape, Y_labels.shape

	for i in range(1,15):
		knn_model = KNeighborsClassifier(n_neighbors=i)
		knn_model.fit(X, X_labels) 
		print "\n for k = ", i, ", accuracy = ", knn_model.score(Y,Y_labels,sample_weight=None)

	print "-"*50

def NBClassifier(filename, split_ratio):
	print "-"*15,"Naive Bayes Classfier","-"*15

	X, Y, X_labels, Y_labels = split_data(filename,split_ratio)

	# print X.shape, Y.shape, X_labels.shape, Y_labels.shape

	nb_model = GaussianNB()
	nb_model.fit(X, X_labels)

	print "\n accuracy =", nb_model.score(Y,Y_labels,sample_weight=None)

	print "-"*50


NBClassifier("doc_tfidf.csv",0.20)
NBClassifier("doc_tfidf.csv",0.30)
NBClassifier("doc_tfidf.csv",0.40)
KNNClassifier("doc_tfidf.csv",0.20)
KNNClassifier("doc_tfidf.csv",0.30)
KNNClassifier("doc_tfidf.csv",0.40)
