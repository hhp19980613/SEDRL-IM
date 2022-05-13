import time
import numpy as np
import networkx as nx



networkName = "EGFR"
import scipy.io as scio
path = '../data/' + networkName + '.mat'
data = scio.loadmat(path)
data1 = data['adj_mat'].A
print(data1.shape)
l = data1.shape[0]
print(l)
newedges = np.array([])
for i in range(l):
    for j in range(l):
            if data1[i][j] != 0:
                newedges = np.append(newedges, (i+1, j+1, data1[i][j]))



newedges = newedges.reshape((len(newedges)//3, 3))
print(newedges.shape)
np.savetxt('../data/' + networkName + '.txt', newedges, fmt='%d', delimiter=' ')