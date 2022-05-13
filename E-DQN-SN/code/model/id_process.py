import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import networkx as nx


class Env:

    def __init__(self):

        self.nodeNum = 0
        self.edgeList = []
        self.seeds = set()
        self.influence = 0
        self.graphNum = 0
        self.graphIndex = -1
        self.degreeScore = []
        self.randomScore = []
        self.CIScore = []
        self.nodeNum = 1000
        # self.maxSeedsNum = self.nodeNum * 1 // 10
        self.maxSeedsNum = 10
        self.networkName = 'wiki1000'
        # self.nameList = ['../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.00.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.05.txt',
        #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.10.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.15.txt',
        #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.20.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.25.txt',
        #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.30.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.35.txt',
        #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.40.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.45.txt',
        #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.50.txt']
        # self.nameList = ['../data/'+self.networkName+'/network1.txt', '../data/'+self.networkName+'/network2.txt',
        #                  '../data/'+self.networkName+'/network3.txt', '../data/'+self.networkName+'/network4.txt',
        #                  '../data/'+self.networkName+'/network5.txt', '../data/'+self.networkName+'/network6.txt',
        #                  '../data/'+self.networkName+'/network7.txt', '../data/'+self.networkName+'/network8.txt',
        #                  '../data/'+self.networkName+'/network9.txt', '../data/'+self.networkName+'/network10.txt']
        self.nameList = ['../data/'+self.networkName+'.txt']

        self.graphIndex = -1
        self.nextGraph()


    def constrctGraph(self, edges):
        graph = nx.DiGraph()
        graphP = nx.DiGraph()

        for u, v, p in edges:
            u = int(u)
            v = int(v)
            graph.add_edge(u,v)

        nodesList = list(graph.nodes())     # id: 0~nodeNum-1
        nodeMap = dict()
        index = 0
        for node in nodesList:
            nodeMap[node] = index
            index += 1

        edges1 = np.array([])
        indegree = graph.in_degree()
        outdegree = graph.out_degree()
        for edge in graph.edges():
            u, v = edge


            u = nodeMap[u]
            v = nodeMap[v]
            graphP.add_edge(u, v)
            edges1 = np.append(edges1, (u, v))

        edges1 = edges1.reshape((len(edges), 2))
        return graphP, edges1


    def nextGraph(self):
        self.graphIndex += 1
        graph_file = self.nameList[self.graphIndex]
        self.graph, self.edges = self.constrctGraph(np.loadtxt(graph_file))
        print(self.edges.shape)
        print("nodeNum:", len(self.graph.nodes()))
        print("edgeNum:", len(self.graph.edges()))
        np.savetxt('../data/'+self.networkName+'p.txt', self.edges, fmt='%d', delimiter=' ')




if __name__ == "__main__":
    env = Env()

