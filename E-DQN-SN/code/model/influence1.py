import numpy as np
import networkx as nx

import time
import os
# import matlab.engine
import scipy.io as scio

seed = 123
np.random.seed(seed)
class Env:

    def __init__(self, mainPath, maxSeedNum):
        # self.eng = matlab.engine.start_matlab()  # 可以为所欲为的调用matlab内置函数
        self.eng = []  # 可以为所欲为的调用matlab内置函数
        self.mainPath = mainPath
        self.dim = 64 + 3
        self.graph_dim = 64
        self.nodeNum = 166
        self.maxSeedsNum = maxSeedNum
        self.networkName = 'War'
        self.nameList = ['../data/'+self.networkName+".txt"]
        # self.nameList = ['../data/' + self.networkName + '/' + str(self.nodeNum) + '/0.10.txt',
        #                  '../data/' + self.networkName + '/' + str(self.nodeNum) + '/0.30.txt']

        self.localInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        self.oneHopInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        self.graphIndex = -1
        self.nextGraph()

    def constrctGraph(self,edges):
        graph = nx.DiGraph()
        posi_graph = nx.DiGraph()
        neg_graph = nx.DiGraph()
        for u, v, sign in edges:
            #u = int(u) - 1
            #v = int(v) - 1
            u = int(u)
            v = int(v)
            p = 0.1

            if sign == 1:
                posi_graph.add_edge(u, v, weight=p)
                graph.add_edge(u, v, weight=p)
            elif sign == -1:
                graph.add_edge(u, v, weight=-p)
                neg_graph.add_edge(u, v, weight=p)

        return graph, posi_graph, neg_graph, np.array(graph.edges())

    def nextGraph(self):
        self.graphIndex += 1
        graph_file = self.nameList[self.graphIndex]
        self.graph, self.posi_graph, self.neg_graph, self.edges = self.constrctGraph(np.loadtxt(graph_file))

        self.embedInfo = self.getembedInfo()
        self.nodeNum = len(self.graph.nodes())
        self.seeds = set()
        self.influence = 0
        self.localInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        self.oneHopInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded

    def reset(self):
        self.seeds = set([])
        self.influence = 0
        return self.seeds2input(self.seeds)

    def step(self, node):
        state = None
        if node in self.seeds:
            print("choose repeated node!!!!!!!!!!!!!")
            state = self.seeds2input(self.seeds)
            return state, 0, False

        self.seeds.add(node)
        reward = self.getInfluence(self.seeds) - self.influence

        self.influence += reward

        isDone = False
        if len(self.seeds) == self.maxSeedsNum:
            isDone = True

        state = self.seeds2input(self.seeds)
        return state, reward, isDone


    def seeds2input(self,seeds):
        input = np.array(self.embedInfo)
        flagList = np.array([])
        degreeList = np.array([])
        posi_degreeList = np.array([])
        # print(input.shape)
        # print(self.posi_graph.number_of_nodes())
        # print(self.neg_graph.number_of_nodes())
        # print(self.graph.number_of_nodes())
        # print(self.posi_graph.out_degree[1])
        # print(self.posi_graph.out_degree)
        for i in range(self.nodeNum):
            degreeList = np.append(degreeList, self.graph.out_degree[i])
            #posi_degreeList = np.append(posi_degreeList, self.posi_graph.out_degree[i])
            try:
                posi_degreeList = np.append(posi_degreeList, self.posi_graph.out_degree[i])
            except:
                posi_degreeList = np.append(posi_degreeList, 0)
            if i in seeds:
                flagList = np.append(flagList, 0)
            else:
                flagList = np.append(flagList, 1)

        flagList = flagList.reshape((self.nodeNum,1))
        degreeList = degreeList.reshape((self.nodeNum, 1))
        posi_degreeList = posi_degreeList.reshape((self.nodeNum, 1))
        
        self.max_out_degree = np.max(degreeList)
        for i in range(self.nodeNum):
            degreeList[i] = degreeList[i] / self.max_out_degree * 1.0
            
        self.max_out_posi_degree = np.max(posi_degreeList)
        for i in range(self.nodeNum):
            posi_degreeList[i] = posi_degreeList[i] / self.max_out_posi_degree * 1.0

        input = np.hstack((degreeList, input))
        input = np.hstack((posi_degreeList, input))
        input = np.hstack((flagList,input))
        return input

    def getembedInfo(self):
        try:
            print("graph name == ", self.networkName)
            print("seed num == ", self.maxSeedsNum)
            embedInfo = np.loadtxt("../data/embedding/" + self.networkName + ".txt")
            #np.savetxt(self.mainPath + "/embedding/" + self.networkName + str(self.graphIndex), embedInfo)
            # embedInfo = np.loadtxt("../data/embedding/" + self.networkName + "matrix" + str(self.graphIndex))
        except:
            self.eng.init_embed(self.networkName, self.nodeNum, self.dim - 3)

            embedInfo = np.loadtxt("../data/embedding/" + self.networkName + str(self.graphIndex))
            np.savetxt(self.mainPath + "/embedding/" + self.networkName + str(self.graphIndex), embedInfo)

        return embedInfo

    def getInfluence(self, S):
        influence = 0
        for s in S:
            influence += self.getLocalInfluence(s)

        influence -= self.getEpsilon(S)

        for s in S:
            Cs = set(self.graph.successors(s))
            S1 = S & Cs
            for s1 in S1:
                influence -= self.graph[s][s1]['weight'] * self.getOneHopInfluence(s1)
        return influence

    # one node local influence
    def getLocalInfluence(self, node):
        if self.localInfluenceList[node] >= 0:
            return self.localInfluenceList[node]

        result = 1
        Cu = set(self.graph.successors(node))
        for c in Cu:
            temp = self.getOneHopInfluence(c)
            Cc = set(self.graph.successors(c))
            if node in Cc:      # if egde (c,node) exits
                 temp = temp - self.graph[c][node]['weight']
            temp = temp * self.graph[node][c]['weight']
            result += temp
        self.localInfluenceList[node] = result
        return result

    # input a node
    def getOneHopInfluence(self, node):
        if self.oneHopInfluenceList[node] >= 0:
            return self.oneHopInfluenceList[node]

        result = 1
        for c in self.graph.successors(node):
            result += self.graph[node][c]['weight']

        self.oneHopInfluenceList[node] = result
        return result

    # input a set of nodes
    def getEpsilon(self, S):
        result = 0

        for s in S:
            Cs = set(self.graph.successors(s))  # neighbors of node s
            S1 = Cs - S
            for c in S1:
                Cc = set(self.graph.successors(c))  # neighbors of node c
                S2 = Cc & S
                # S2 = S2 - {s}
                result += (0.01 * len(S2))
                # for d in S2:
                #     result += self.graph[s][c]['weight'] * self.graph[c][d]['weight']
        return result
