import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def round(x) :
    xx = math.floor(x)
    if(x - xx >= 0.5): return math.ceil(x)
    return xx
def CalDistance(point, center) :
    return np.sqrt(sum((point - center)**2))
def CalCost(clustering_method, data, centers, label):
    cost = 0.0
    print(clustering_method)
    for i in range(len(data)):
        if clustering_method == 'kmedoids' :
            cost += sum((data[i] - centers[label[i]])**2)
        if clustering_method == 'kmedian' :
            cost += np.sqrt(sum((data[i] - centers[label[i]])**2))
        if clustering_method == 'kcenter' :
            cost = max(np.sqrt(sum((data[i] - centers[label[i]]) ** 2)), cost)
        #cost += CalDistance(data[i], centers[label[i]])
    return cost
def my_metric(point1, point2) :
    return sum((point1 - point2)**2)
def CalCenter(data, label, num) :
    centers = np.zeros((num, len(data[0])), dtype=np.double)
    for i in range(num) :
        cnt = 0
        for j in range(len(label)) :
            if(label[j] == i) :
                cnt += 1
                centers[i] += data[j]
        if cnt :
            centers[i] /= cnt
    return centers

def CalGFViolation(data, center, label, color, kl, kr) :
    cluster_sz = []
    for i in range(len(center)) :
        cluster_sz.append(0)
    for i in range(len(label)) :
        cluster_sz[label[i]] += 1
    GFViolation = 0
    for i in range(len(center)):
        if(cluster_sz[i] == 0) : continue
        color_sz = []
        for j in range(len(kl)):
            color_sz.append(0)
        for j in range(len(label)):
            if(i == label[j]) :
                color_sz[color[j]] += 1
       # print("color_sz")
        #print(color_sz)
        for j in range(len(kl)) :
            #if(color_sz[j] == 0) : continue
            if(color_sz[j] < round(kl[j] * cluster_sz[i])):
                GFViolation = max(GFViolation, round(kl[j] * cluster_sz[i]) - color_sz[j])
            if(color_sz[j] > round(kr[j] * cluster_sz[i])):
                GFViolation = max(GFViolation, color_sz[j] - round(kr[j] * cluster_sz[i]))
    return GFViolation

def CalDSViolation(data, center, label, color, kl, kr):
    cluster_sz = []
    for i in range(len(center)):
        cluster_sz.append(0)
    for i in range(len(label)):
        cluster_sz[label[i]] += 1
    _k = 0
    for i in range(len(center)):
        if(cluster_sz[i] == 0) : continue
        _k += 1
    color_sz = []
    for i in range(len(kl)):
        color_sz.append(0)
    for i in range(len(data)):
        for j in range(len(center)):
            if(cluster_sz[j] == 0) : continue
            if(np.array_equal(np.array(data[i]), np.array(center[j]))):
                color_sz[color[i]] += 1
    vio = 0
    for i in range(len(kl)):
        if(color_sz[i] < kl[i]): vio = max(vio, kl[i] - color_sz[i])
        if(color_sz[i] > kr[i]): vio = max(vio, color_sz[i] - kr[i])
    return vio

