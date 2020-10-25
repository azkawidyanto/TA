# AGGLOMERATIVE CLUSTERING
import math
import sys

import pandas as pd
from statistics import mean
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

# data iris
# iris = load_iris()
# data_input = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#                      columns= iris['feature_names'] + ['target'])
# targets = data_input.iloc[:,-1]

# #data is training data, udh dipisah sama target
# data = data_input.iloc[:, :-1].reset_index(drop=True)

# print(data)


def findAverage(points):
    return mean(points)

# input P: [x, y, z, ..]


def findDistance(p1, p2):
    total = 0.0
    for i in range(len(p1)):
        total += pow((p1[i] - p2[i]), 2)
    return round(math.sqrt(total), 2)


def minDistance(d1, d2):
    if (d1 < d2):
        return d1
    else:
        return d2


def maxDistance(d1, d2):
    if (d1 > d2):
        return d1
    else:
        return d2


def avgDistance(data, cluster, x, y):
    if (x == y):
        return 0
    else:
        c1 = data[cluster[x]]
        c2 = data[cluster[y]]
        distances = []
        for i in range(len(c1)):
            for j in range(len(c2)):
                distances.append(findDistance(c1[i], c2[j]))
        return findAverage(distances)


def avgGroupDistance(data, cluster, x, y):
    if (x == y):
        return 0
    else:
        c1 = data[cluster[x]]
        c2 = data[cluster[y]]
        mean1 = np.mean(c1, axis=0)
        mean2 = np.mean(c2, axis=0)
        return findDistance(mean1, mean2)


def createDissimilarityMtx(data):
    dissmilarity = np.zeros((len(data), len(data)))
    i = 0
    while (i < len(data)):
        j = i
        while (j < len(data)):
            dissmilarity[i][j] = findDistance(data[i], data[j])
            j += 1
        i += 1
    return dissmilarity


def getMinDissimilarity(mtx):
    mindis = sys.float_info.max
    row, col = -1, -1
    i = 0
    while (i < len(mtx)):
        j = i
        while (j < len(mtx)):
            if (mtx[i][j] < mindis and i != j):
                mindis = mtx[i][j]
                row = i
                col = j
            j += 1
        i += 1
    return row, col


def updateDissimilatrity(data, mtx, cluster, row, col, linkage):
    newmtx = np.delete(mtx, col, 0)
    newmtx = np.delete(newmtx, col, 1)
    for i in range(len(cluster)):
        if (i < col):
            if (i < row):
                dist1, dist2 = mtx[i][row], mtx[i][col]
                newrow, newcol = i, row
            elif (i == row):
                dist1, dist2 = 0, 0
                newrow, newcol = i, row
            else:
                dist1, dist2 = mtx[row][i], mtx[i][col]
                newrow, newcol = row, i
        elif (i >= col):
            if (i < row):
                dist1, dist2 = mtx[i+1][row], mtx[col][i+1]
                newrow, newcol = i, row
            elif (i == row):
                dist1, dist2 = 0, 0
                newrow, newcol = i, row
            else:
                dist1, dist2 = mtx[row][i+1], mtx[col][i+1]
                newrow, newcol = row, i
        if (linkage == 'single'):
            distance = minDistance(dist1, dist2)
        elif (linkage == 'complete'):
            distance = maxDistance(dist1, dist2)
        elif (linkage == 'average'):
            distance = avgDistance(data, cluster, row, i)
        elif (linkage == 'average group'):
            distance = avgGroupDistance(data, cluster, row, i)
        newmtx[newrow][newcol] = distance
    return newmtx


def createClusterList(data):
    cluster = [[]] * len(data)
    for i in range(len(data)):
        cluster[i] = [i]
    return cluster


def updateCluster(cluster, row, col):
    cluster[row].extend(cluster[col])
    del cluster[col]
    return cluster


def agglomerative(data, n, linkage):
    # linkage = input("single/complete/average/average group?\n")
    cluster = createClusterList(data)
    dissmilarity = createDissimilarityMtx(data)
    while (len(cluster) > n):
        i, j = getMinDissimilarity(dissmilarity)
        updateCluster(cluster, i, j)
        dissmilarity = updateDissimilatrity(
            data, dissmilarity, cluster, i, j, linkage)
    return(cluster)


def getResult(cluster, n):
    result = [0] * n
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            result[cluster[i][j]] = i
    return result

# data = [[0.4, 0.53],
# 		[0.22, 0.38],
# 		[0.35, 0.32],
# 		[0.26, 0.19],
# 		[0.08, 0.41],
# 		[0.45, 0.30]]

# # data = data.iloc[:-140, :].reset_index(drop=True).values

# print(agglomerative(data, 1, 'complete'))
