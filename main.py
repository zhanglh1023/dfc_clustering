import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#df = pd.read_csv('adult/adult.data', sep=',', names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
#df = df.sample(n = 20000)
#df.head()
#df.info()
#print(df.index)
#print(df.apply(lambda x : np.sum(x == " ?")))
#df.replace(" ?", pd.NaT, inplace = True)
#df.replace(" >50K", 1, inplace=True)
#df.replace(" <=50K", 0, inplace=True)
#trans = {'workclass' : df['workclass'].mode()[0], 'occupation' : df['occupation'].mode()[0], 'native-country' : df['native-country'].mode()[0]}
#df.fillna(trans, inplace = True)
#print(df.describe())
#print(df.apply(lambda x : np.sum(x == " ?")))

#labelEncoder = LabelEncoder()
#labelEncoder.fit(df['sex'])
#df['sex'] = labelEncoder.transform(df['sex'])
#labelEncoder.fit(df['occupation'])
#df['occupation'] = labelEncoder.transform(df['occupation'])
#labelEncoder.fit(df['workclass'])
#df['workclass'] = labelEncoder.transform(df['workclass'])
#labelEncoder.fit(df['education'])
#df['education'] = labelEncoder.transform(df['education'])
#labelEncoder.fit(df['marital-status'])
#df['marital-status'] = labelEncoder.transform(df['marital-status'])
#labelEncoder.fit(df['relationship'])
#df['relationship'] = labelEncoder.transform(df['relationship'])
#labelEncoder.fit(df['race'])
#df['race'] = labelEncoder.transform(df['race'])
#labelEncoder.fit(df['native-country'])
#df['native-country'] = labelEncoder.transform(df['native-country'])

#(df)
#kmeans = KMeans(n_clusters=4)
#kmeans.fit(df)

#label = kmeans.labels_
#centers = kmeans.cluster_centers_
#print(centers[0])
#data = df.values
#print(data[0])
#print(np.size(data))
#print(centers[0])
#cost = 0.0
#print(df.size)
#print(df.iloc[0])
#for i in range(0, df.size) :
   #print(df[i])
#print(np.array(object= 0, dtype=float, ndmin=4))
def round(x) :
    xx = math.floor(x)
    if(x - xx >= 0.5): return math.ceil(x)
    return xx
def CalDistance(point, center) :
    return np.sqrt(sum((point - center)**2))
def CalCost(data, centers, label):
    cost = 0.0
    for i in range(len(data)):
        cost += sum((data[i] - centers[label[i]])**2)
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
#print(sum)
#print(CalCost(data, centers, label))
