import numpy as np
import networkx as nx
import os
nodeNum = 118
maxSeedsNum = nodeNum * 1 // 10
networkName = 'sfi'
# nameList = ['../data/' + networkName + '/' + str(nodeNum) + '/0.00.txt',
#             '../data/' + networkName + '/' + str(nodeNum) + '/0.05.txt',
#             '../data/' + networkName + '/' + str(nodeNum) + '/0.10.txt',
#             '../data/' + networkName + '/' + str(nodeNum) + '/0.15.txt',
#             '../data/' + networkName + '/' + str(nodeNum) + '/0.20.txt',
#             '../data/' + networkName + '/' + str(nodeNum) + '/0.25.txt',
#             '../data/' + networkName + '/' + str(nodeNum) + '/0.30.txt',
#             '../data/' + networkName + '/' + str(nodeNum) + '/0.35.txt',
#             '../data/' + networkName + '/' + str(nodeNum) + '/0.40.txt',
#             '../data/' + networkName + '/' + str(nodeNum) + '/0.45.txt',
#             '../data/' + networkName + '/' + str(nodeNum) + '/0.50.txt']
# nameList = ['../data/'+networkName+'/network1.txt', '../data/'+networkName+'/network2.txt',
#             '../data/'+networkName+'/network3.txt', '../data/'+networkName+'/network4.txt',
#             '../data/'+networkName+'/network5.txt', '../data/'+networkName+'/network6.txt',
#             '../data/'+networkName+'/network7.txt', '../data/'+networkName+'/network8.txt',
#             '../data/'+networkName+'/network9.txt', '../data/'+networkName+'/network10.txt']
nameList = ['../data/' + networkName + '.txt']
# graphIndex = 0

for graphIndex in range(len(nameList)):
    graph_file = nameList[graphIndex]
    # graph_file = '../data/test.txt'
    edges = np.loadtxt(graph_file)
    graph = nx.DiGraph()
    for u, v in edges:
        u = int(u) - 1
        v = int(v) - 1
        # u = int(u)
        # v = int(v)
        graph.add_edge(u, v)


    A=np.array(nx.adjacency_matrix(graph).todense())
    if not os.path.exists(networkName):
        os.makedirs(networkName)
    resultPath = networkName+'/adj' + str(graphIndex) + '.txt'
    file = open(resultPath, 'w')

    for i in range(len(A)):
        line = ""
        for j in range(len(A)):
            if graph.has_edge(i,j):
                line = line + "1"
            else:
                line = line + "0"
        line = line + "\n"
        file.write(line)