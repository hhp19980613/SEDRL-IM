import numpy as np
import networkx as nx
import time
import os
import scipy.io as scio

seed = 123
np.random.seed(seed)
class Env:

    def __init__(self, mainPath, maxSeedNum):
        self.mainPath = mainPath # 设置主路径，用于存取数据
        self.dim = 64 + 3 # 设置DQN输出维数
        self.nodeNum = 166 # 节点数
        self.maxSeedsNum = maxSeedNum # 设置种子规模
        self.networkName = 'War' # 网络数据集名称
        self.nameList = ['../data/'+self.networkName+".txt"] # 网络数据集路径
        self.localInfluenceList = np.zeros(self.nodeNum)-1  # 记录每个种子节点的影响分数
        self.oneHopInfluenceList = np.zeros(self.nodeNum)-1  # 记录每个种子节点的影响分数
        self.graphIndex = -1 # 数据集标记
        self.nextGraph() # 切换网络数据集

    # 基于邻接表构造邻接矩阵 #
    def constrctGraph(self,edges):
        graph = nx.DiGraph()
        posi_graph = nx.DiGraph()
        neg_graph = nx.DiGraph()
        for u, v, sign in edges:
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

    # 切换网络数据集，并读取相应的邻接表和降维向量 #
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

    # 清空种子节点信息，重置降维向量 #
    def reset(self):
        self.seeds = set([])
        self.influence = 0
        return self.seeds2input(self.seeds)

    # 基于DQN选出的种子节点，模拟影响传播过程，最终将两跳以内的影响分数作为reward #
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

    # 将当前种子节点的信息作为新向量，拼接成网络数据集的新降维向量 #
    def seeds2input(self,seeds):
        input = np.array(self.embedInfo)
        flagList = np.array([])
        degreeList = np.array([])
        posi_degreeList = np.array([])
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
        
        

        input = np.hstack((degreeList, input))
        input = np.hstack((posi_degreeList, input))
        input = np.hstack((flagList,input))
        return input

    # 获取网络数据集的降维向量 #
    def getembedInfo(self):

        print("graph name == ", self.networkName)
        print("seed num == ", self.maxSeedsNum)
        embedInfo = np.loadtxt("../data/embedding/" + self.networkName + ".txt")

        return embedInfo

    # 计算种子节点两跳以内的有效影响分数 #
    def getInfluence(self, S):
        # 计算种子节点两跳以内激活其他节点的影响分数
        influence = 0
        for s in S:
            influence += self.getLocalInfluence(s)

        # 剔除种子节点一两跳以内激活其他种子节点的影响分数
        influence -= self.getEpsilon(S)
        # 剔除种子节点一两跳以内激活其他种子节点的影响分数
        for s in S:
            Cs = set(self.graph.successors(s))
            S1 = S & Cs
            for s1 in S1:
                influence -= self.graph[s][s1]['weight'] * self.getOneHopInfluence(s1)
        return influence

    # 计算种子节点两跳以内激活其他节点的影响分数 #
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

    # 计算种子节点两跳以内激活其他种子节点影响分数 #
    def getOneHopInfluence(self, node):
        if self.oneHopInfluenceList[node] >= 0:
            return self.oneHopInfluenceList[node]

        result = 1
        for c in self.graph.successors(node):
            result += self.graph[node][c]['weight']

        self.oneHopInfluenceList[node] = result
        return result

    # 计算种子节点一跳以内激活其他种子节点影响分数 #
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
