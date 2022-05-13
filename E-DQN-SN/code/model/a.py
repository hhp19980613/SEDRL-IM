import matlab.engine
import scipy.io as scio
import numpy as np
import os
import networkx as nx
import torch
import random

networkName = "LFR-0.10"

# graphfile = '../data/' + networkName + '.txt'
graphfile = '../data/LFR/500/0.10.txt'
newedges = np.array([])
edges = np.loadtxt(graphfile)
graph = nx.DiGraph()
# fh = open('../data/Finder/' + networkName + '.txt', 'w', encoding='utf-8')

for u, v in edges:
    if not graph.has_edge(u, v) and not graph.has_edge(v, u):
        graph.add_edge(u ,v)
        s = str(int(u-1)) + " " + str(int(v-1)) + " {}\n"
        newedges = np.append(newedges, s)
        # fh.write(s)


print(len(graph.nodes()))
newedges = newedges.reshape((len(newedges)//2, 2))
np.savetxt('../data/' + networkName + '1.txt', newedges, fmt='%d', delimiter=' ')

# i = 10
# for file in os.listdir(nameList[i]):  # calculate layer num of multilex
#     print(file)
#     path = nameList[i] + "/" + file
#     edges = np.loadtxt(path)
#     matrix = np.zeros((500,500))
#     for edge in edges:
#         u, v = edge
#         u = int(u) - 1
#         v = int(v) - 1
#         matrix[u][v] = 1
#     np.savetxt("../data/matrix/" + networkName + str(i) + file, matrix)
