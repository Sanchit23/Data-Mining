from __future__ import division
import csv
import time
import random
import numpy as np
from math import sqrt, log


def getdata(filename):
    temp=[]
    labels=[]
    data=[]
    topics=[]
    doc_id = 1
    
    with open(filename,'rb') as csvfile:
        csvfile.next()
        reader = csv.reader(csvfile)
        for row in reader:
            labels = row[0:1]
            temp = row[2:]
            temp = map(float, temp)
            for i in range(0,temp.__len__()):
                if temp[i]>0:
                    temp[i]=1
                else:
                    temp[i]=0
            
            temp.insert(0,doc_id)
            labels.insert(0,doc_id)
            data.append(temp)
            topics.append(labels)
            doc_id=doc_id+1
            temp=[]
            labels=[]
            
    return data, topics

def calc_variance(clusters,count):
    variance=[]
    for i in range(0,count):
        variance.append(clusters[i].__len__())
    print "\n variance ", np.var(variance)

def get_distance(u,v,metric):
    u=u[1:]
    v=v[1:]
    u=np.array(map(float, u))
    v=np.array(map(float, v))
    
    if(metric=="euclidean"):    #euclidean
        # print "\n using euclidean" 
        diff = u - v
        return sqrt(np.dot(diff, diff))
    else:    #cosine
        # print "\n using cosine" 
        return 1 - (np.dot(u, v) / (sqrt(np.dot(u, u)) * sqrt(np.dot(v, v))))


def get_entropy(cluster_dict,size):
    ans=0
    for values in cluster_dict.itervalues():
        P=values/size;
        ans= ans - (P*log(P if P>0 else 1)/log(2))
    return ans

def center(list_of_vectors):
    list_of_vectors=np.array(list_of_vectors)
    return np.mean(list_of_vectors,axis=0)


def kmeans(dist_type, k):
    print "\n","-"*50
    print "Clusters = ", k, ", Metric = ", dist_type

    cluster_track={}

    for i in range(1,dataMatrix.__len__()+1):
        cluster_track[i] = -1

    clusters = random.sample(dataMatrix,k)
    
    count = len(clusters)
    iteration = 0
    total_updates = 0
    max_iterations = int(10)

    #k-means algorithm starts here
    start_time = time.clock()

    while True:
        iteration = iteration+1
        cluster_lists = [[] for c in clusters]
        total_updates = 0
        
        for row in dataMatrix:
            if len(clusters[0])!=0:
                smallest_distance = get_distance(row, clusters[0],dist_type)
            tempIndex = 0
            for i in range(count-1):
                if len(clusters[i+1]) != 0:
                    distance = get_distance(row,clusters[i+1],dist_type)

                if distance < smallest_distance:
                    smallest_distance = distance
                    tempIndex = i+1
            
            cluster_lists[tempIndex].append(row)

            if cluster_track[row[0]] != tempIndex:
                total_updates = total_updates + 1
                cluster_track[row[0]] = tempIndex

        if total_updates <= count or iteration > max_iterations:
            break

        else:
            for k in range(0,cluster_lists.__len__()):
                if cluster_lists[k].__len__() > 1:
                    clusters[k] = center(cluster_lists[k])
                else:
                    max = 0
                    for l in range(0,count):
                        if cluster_lists[l].__len__() > max:
                            max = cluster_lists[l].__len__()
                            test = l
                    clu = random.sample(cluster_lists[test],1)
                    clusters[k] = clu[0]
                    clu = []
            count=len(clusters)
        # print iteration
    end_time = time.clock()
    print "\n time taken to cluster", end_time-start_time,"s"
    calc_variance(cluster_lists,count)
    entropy(cluster_lists)

def entropy(cluster_lists):
    majority_count={}
    entropies=[]
    for i in range (0,cluster_lists.__len__()):
        for j in range (0,cluster_lists[i].__len__()):
            if isinstance(cluster_lists[i][j][0], int):
                key=(topics[int(cluster_lists[i][j][0])-1][1])
                if majority_count.has_key(key):
                  majority_count[key]=majority_count.get(key)+1
                else:
                    majority_count[key]=1
        entropy=get_entropy(majority_count,cluster_lists[i].__len__())
        entropies.append(entropy)
        majority_count.clear()

    # final weighted entropy
    final_entropy=0
    for i in range(0,cluster_lists.__len__()):
        final_entropy=final_entropy+(entropies[i]*cluster_lists[i].__len__()/dataMatrix.__len__())

    #print entropies
    print "\n Entropy " , final_entropy


while True:
    option = input("Press 1 to run on small data set, otherwise press 2 \n")

    if option == 1:
        dataMatrix, topics = getdata("doc_tfidf_small.csv")
        break
    elif option == 2:
        dataMatrix, topics = getdata("doc_tfidf.csv")
        break
    else:
        print "\n wrong input. Please try again."

kmeans("euclidean",25)
kmeans("euclidean",50)
kmeans("euclidean",75)
kmeans("euclidean",100)
kmeans("Cosine",25)
kmeans("Cosine",50)
kmeans("Cosine",75)
kmeans("Cosine",100)

