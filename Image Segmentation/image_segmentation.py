# -*- coding: utf-8 -*-
"""image_segmentation_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14_2mKVyS9VXeYgsbtRCxqAaaIWnSBmUD
"""

from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import os
import math
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import rbf_kernel
import scipy.misc

warnings.filterwarnings('ignore') # remove any warning output

from zipfile import ZipFile
file_name = "BSR.zip"
with ZipFile(file_name,'r') as zip:
    zip.extractall()
    print("Done")

def displayImages_GroundTruth(groundTruth):
    imgs = []
    ground_truth_images = []
    
    path_groundtruth = 'BSR/BSDS500/data/groundTruth/'+groundTruth
    path_images = 'BSR/BSDS500/data/images/'+groundTruth
    groundTruth_files = [f for f in os.listdir(path_groundtruth) if f.endswith('.mat')]
    for ground_Truth in groundTruth_files:  
        #print(ground_Truth)
        test = sio.loadmat(path_groundtruth + '/' + ground_Truth)
        my_images = Image.open(path_images + '/' + ground_Truth[:-3] + "jpg")
        my_images = np.array(my_images)
       # my_images = my_images.reshape((154401,3))
        imgs.append(my_images)   
        img = Image.fromarray(my_images)
        img = img.convert("RGB")
       # display(img)
        array = np.array(test['groundTruth'])
        image_of_groundtruth = []
        for ar1 in array[0]:
               for ar2 in ar1:
                    for ar3 in ar2:
                        for ar4 in ar3:
                            ar4 = 255 - ar4
                            img = Image.fromarray(ar4)
                            img = img.convert("RGB")
                            #display(img)
                            #plt.figure()
                            #plt.imshow(ar4.astype(np.uint8))
                            image_of_groundtruth.append(ar4) 
        ground_truth_images.append(image_of_groundtruth)                    
    return imgs,ground_truth_images

images,ground_truth_images = displayImages_GroundTruth("train")

print(images[0].shape)
print(len(ground_truth_images[0]))

plt.imshow(images[0])

num_of_clusters = 3

def kmeans_image(index,image,num_of_clusters):
    rows,cols,_  =image.shape
    kmeans = KMeans(n_clusters= num_of_clusters)
    kmeans_cluster = kmeans.fit(image.reshape((rows*cols,3)))
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    plt.imshow((cluster_centers[cluster_labels].reshape(rows,cols, 3)).astype(np.uint8)) 
    scipy.misc.imsave('outfile'+str(index)+'.jpg', image)
    return cluster_centers ,cluster_labels

#for segments in range(len(ground_truth_images[0]))

def kmeans_segment(image,num_of_clusters):
    rows,cols  =image.shape
    kmeans = KMeans(n_clusters=num_of_clusters)
    kmeans_cluster_gt = kmeans.fit(image.reshape(rows*cols,1))
    cluster_centers_gt = kmeans_cluster_gt.cluster_centers_
    cluster_labels_gt = kmeans_cluster_gt.labels_
    plt.figure()
    plt.imshow((cluster_centers_gt[cluster_labels_gt].reshape(rows,cols)).astype(np.uint8))
    truth_points = []
    for i in range(kmeans.n_clusters):
        i= np.where(kmeans_cluster_gt.labels_ == i)[0]
        truth_points.append(i)
    return truth_points

def get_clusters(truth_points):
    colors = np.empty((num_of_clusters,num_of_clusters))
    row = 0
    for i in truth_points:
        point = i
        truth = cluster_centers[cluster_labels][point]
        for x in range(num_of_clusters):
            testing = cluster_centers[x]
            count = 0
            for z in truth:
                if((z == testing).all()):
                    count += 1
            colors[row,x] = count
        row += 1
    return colors

compatible_images = 0

def f_measure(num_of_clusters,colors):
    #calculate F-Measure
    precision = np.zeros((num_of_clusters,1))
    recall = np.zeros((num_of_clusters,1))
    f = np.zeros((num_of_clusters,1))
    x = 0
    for i in colors:
        precision[x] = i[x]/np.sum(i)
        recall[x] = i[x]/np.sum(colors[:,x])
        f[x] = (2*precision[x]*recall[x])/(precision[x] + recall[x])
        #print("prec[",x,'] =' ,precision[x])
        #print("reca[",x,'] =' ,recall[x])
        #print("f[",x,'] =' ,f[x])
        x += 1

    F = (np.sum(f))/num_of_clusters
    if(math.isnan(F)):
        return 0
    else:
        #print('F-measure = {:.2f}'.format(F))
        return F

#calculate Entropy
def entropy(num_of_clusters,colors):
    H = 0
    tot_points = 154401
    iteration = 0
    h = np.zeros((num_of_clusters,1))
    entroy_of_cluster = 0
    for i in colors:
        entroy_of_cluster = 0
        for x in range(len(i)):
            coef = i[x]/np.sum(i)
            if(coef != 0):
                entroy_of_cluster -= coef*math.log(coef,10)
        h[iteration] = entroy_of_cluster
        #print(h[iteration])  
        H += (np.sum(i)/tot_points)*h[iteration]
    float_formatter = lambda x: "%.2f" % x
    if(math.isnan(H)):
        #print("Image not compatible")
        return 0
    else:
        #print('Entropy = ' , float_formatter(H))
        return H[0]

for image in range(5):
    f_measure_tot = 0
    entropy_tot = 0
    count = 0
    f_measure_ = 0
    entropy_ = 0
    cluster_centers , cluster_labels = kmeans_image(image,images[image],num_of_clusters)
    plt.figure()
    plt.imshow(images[image]) 
    for i in range(len(ground_truth_images[image])):
        truth_points = kmeans_segment(ground_truth_images[image][i],num_of_clusters)
        colors = get_clusters(truth_points)
        f_measure_ = f_measure(num_of_clusters,colors)
        entropy_ = entropy(num_of_clusters,colors)
        if((f_measure_ != 0) and (entropy_ != 0)):         
            f_measure_tot += f_measure_
            entropy_tot += entropy_
            count += 1
    print("total f-measure = " , f_measure_tot / count)
    print("total entropy = " , entropy_tot / count)
    print('--')

# def KNN_clustering(X,k):
#     iterator = 0
#     for row in X:
#         sorted_row = -np.sort(-row)
#         min_value = sorted_row[k]
#         for i in range(len(row)):
#             if row[i] < min_value:
#                 row[i] = 0
#         X[iterator] = row
#         iterator += 1
#     return X

def sim(cl,k,length):
    for rows in range(len(cl)):
        x = np.empty((length,1))
        lis = []
        for i in range(len(cl[0])):
            lis.append(cl[rows,i] - cl[rows,rows])
        x = np.array(lis)
        x = np.absolute(x)
        y = []
        for n in range(0,k+1):
            z = np.where(x == np.min(x))[0]
            x[np.where(x == np.min(x))[0][0]] = 1000
            y.append((z[0]))
        y = np.array(y)
        counter = 0
        y = np.sort(y)
        for i in range(len(cl[0])):
            if(y[counter] != i):
                cl[rows,i] =  0
            elif(y[counter] == i):
                cl[rows,i] = 1
                counter += 1
                if(counter == len(y)):
                    counter = 0
    return cl

from sklearn.preprocessing import normalize
def spectral_clustering(A,k):
    D = np.sum(A,axis = 1)
    D = np.diag(D)
    L = np.empty((A.shape[0],A.shape[1]))
    L = D - A 
    eigVal , eigVec = np.linalg.eigh(L)
    X_normalized = normalize(eigVec.transpose(), norm='l2')
    eigVec = X_normalized.transpose()
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(eigVec[:,1:k])
    centroids = kmeans.cluster_centers_
    return kmeans,eigVec[:,1:k]

from skimage.transform import rescale
for image in range(5):
    for i in range(len(ground_truth_images[image])):
      plt.figure()
      plt.imshow(ground_truth_images[image][i].astype(np.uint8))
      image_rescaled = rescale(ground_truth_images[image][i],1/6)
      plt.figure()
      plt.imshow(image_rescaled)
      i,j = image_rescaled.shape
      X = image_rescaled.reshape((i*j),1).flatten()
      X = np.array([X])
      X = np.repeat(X,i*j,axis=0)

      k_ways_nc = 3
      A = sim(X,k_ways_nc,i*j)
      normalized_kmeans , eigVec = spectral_clustering(A,k_ways_nc)
      
      centroids_nc = normalized_kmeans.cluster_centers_
      labels_nc = normalized_kmeans.labels_
      rows , cols = image_rescaled.shape
      predicted_image = np.empty((rows,cols)) 
      count = 0
      for i in range(rows):
        for j in range(cols):
          if(labels_nc[count] == 0):
              predicted_image[i,j] = 255
          elif(labels_nc[count] == 1):
              predicted_image[i,j] = 100
          elif(labels_nc[count] == 2):
              predicted_image[i,j] = 2
          count += 1