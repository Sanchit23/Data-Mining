import sys
import csv
import time
import gc
import math
import numpy as np
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
from collections import Counter
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering

def getdata(filename):
	dataMatrix = []
	labels = []
	with open(filename, 'rb') as csv_file:
	    rowHeaders = next(csv_file)
	    for row in csv_file:
	        dataMatrix.append([float(x) for x in row.split(",")[2:]])
	        labels.append(row.split(",")[0])
	return dataMatrix, labels

def variance(clusters):
    return np.var(Counter(clusters).values())

def entropy(class_labels, clusters, n_clusters, total_records):
	count = []
	labels = []
	entropies = []

	for c in range(0,n_clusters):
	    labels =[]
	    for (cluster_label,label) in zip(clusters,class_labels):
	        if cluster_label == c:
	            labels.append(label[0])
	    entropy_cluster = get_entropy(labels)
	    entropies.append(entropy_cluster)
	    count.append(len(labels))
	    # print "Records in cluster:", len(labels)
	    # print "Entropy of cluster:", entropy_cluster
	total_entropy = 0.0
	for (h,c) in zip(entropies,count):
	    total_entropy += h * c
	total_entropy = total_entropy/total_records

	return total_entropy

def get_entropy(labels):
	if len(labels) > 1:
		probs = [c/float(len(labels)) for c in Counter(labels).values()]
		classes = np.count_nonzero(probs)
		if classes > 1:
		    h = 0
		    for i in probs:
		        h = h - (i * math.log(i, 2))
		    return h
		else:
			return 0
	return 0


while True:
	option = input("Press 1 to run on small data set, Press 2 to run on the full data set \n")

	# get data points and class labels
	if option == 1:
		X, class_labels = getdata("doc_tfidf_small.csv")
		break
	elif option == 2:
		X, class_labels = getdata("doc_tfidf.csv")
		break
	else:
		print "\nWrong input. Please try again."


start = time.time()

# transform to numpy array
X = np.array(X)

k = [50,100,150] # number of clusters
metrics = ["cityblock","euclidean","cosine"] # distance functions
linkages = ["average","complete"] # linkage types

for n_clusters in k:
	for metric in metrics:
		for linkage in linkages:
			loop_time = time.time()

			print "\n","-"*60
			print "Clusters = ", n_clusters, ", Metric = ", metric, ", Linkage = ", linkage
			print "-"*60
			# create a model with specified metric, linkage and number of clusters
			model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, affinity=metric)
			
			# cluster
			model.fit(X)
			
			# get cluster labels
			clusters = model.labels_

			print "Time taken to cluster: ", time.time() - loop_time			
			print "Entropy  : ", entropy(class_labels, clusters, n_clusters, int(X.shape[0])) # calculate entropy
			print "Variance : ", variance(clusters) # calculate variance
			# print "The average silhouette_score is :", silhouette_score(X, clusters) # calculate silhouette_score (optional)
			print "-"*60

print "\n total time taken : ", time.time() - start
			
