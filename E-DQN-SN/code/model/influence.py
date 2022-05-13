import numpy as np
import networkx as nx
from line import embedding_main
import os

seed = 123
np.random.seed(seed)
class Env:

    def __init__(self):
        self.dim = 32
        graph_file = '../data/wiki.txt'
        self.graph, self.edges = self.constrctGraph(np.loadtxt(graph_file))
        print(self.edges.shape)
        self.nodeNum = len(self.graph.nodes())
        self.VNindex = self.nodeNum + 1
        for i in range(self.nodeNum):
            self.graph.add_edge(self.VNindex, i, weight=1)

        print("VN finished")
        self.seeds = set()
        self.influence = 0

        self.state = self.seeds2embedInfo(self.seeds)   # (n+1)*d

    # 由有向无权图构造有向有权图
    def constrctGraph(self,edges):
        graph = nx.DiGraph()
        graphP = nx.DiGraph()

        for u,v in edges:
            u = int(u)
            v = int(v)
            graph.add_edge(u,v)

        nodesList = list(graph.nodes())     # 把点的id映射到0~n-1
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

            # p = outdegree[u] / (outdegree[u] + indegree[v])
            # p = (outdegree[u] + indegree[u]) / (outdegree[u] + indegree[u] + outdegree[v] + indegree[v])
            p = 1 / (outdegree[v] + indegree[v])
            # p = 1 / (indegree[v])
            # p = 0.01

            u = nodeMap[u]
            v = nodeMap[v]
            graphP.add_edge(u, v, weight=p)
            edges1 = np.append(edges1,(u,v,p))

        edges1 = edges1.reshape((len(edges),3))
        return graphP, edges1

    def reset(self):
        self.seeds = set([])
        self.influence = 0

    def step(self,node):
        if node in self.seeds:
            return self.state.copy(), 0, False
        self.seeds.add(node)
        reward = self.getInfluence(self.seeds) - self.influence
        self.influence += reward

        isDone = False
        if len(self.seeds) == len(self.graph.nodes()):
            isDone = True

        self.state = self.seeds2embedInfo(self.seeds)
        if reward<0:
            print("error!!!!!!")
            print(node)
            print(self.seeds)
        return self.state.copy(), reward, isDone

    # seeds -> (n,d*d)
    def seeds2input(self,seeds):
        embedInfo = self.seeds2embedInfo(seeds)    # (n+1, d)
        input = []
        VN = embedInfo[-1]

        VN = VN.reshape((self.dim,1))
        for i in range(self.nodeNum):
            embed = embedInfo[i].reshape((1,self.dim))
            temp = np.dot(VN,embed)
            temp = temp.reshape((self.dim*self.dim,))
            input.append(temp)

        input = np.array(input)

        return input

    # (n+1)*d -> (n,d*d)
    def embed2input(self,embedInfo):
        input = []
        VN = embedInfo[-1]

        VN = VN.reshape((self.dim,1))
        for i in range(self.nodeNum):
            embed = embedInfo[i].reshape((1,self.dim))
            temp = np.dot(VN,embed)
            temp = temp.reshape((self.dim*self.dim,))
            input.append(temp)

        input = np.array(input)

        return input

    # seeds -> (n+1)*d
    def seeds2embedInfo(self,seeds):
        edges = self.edges.copy()

        for i in range(self.nodeNum):
            if i not in seeds:
                edges = np.vstack((edges, (self.VNindex, i, 1)))

        path = '../data/graphwithVN.txt'
        np.savetxt(path, edges)
        os.system('python line.py --embedding_dim ' + str(self.dim))
        embedInfo = np.loadtxt("../data/_embedding.txt")
        return embedInfo


    def render(self):
        ...

    def getInfluence(self,S):
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
    def getLocalInfluence(self,node):
        result = 1
        Cu = set(self.graph.successors(node))
        for c in Cu:
            temp = self.getOneHopInfluence(c)
            Cc = set(self.graph.successors(c))
            if node in Cc:      # if egde (c,node) exits
                 temp = temp - self.graph[c][node]['weight']
            temp = temp * self.graph[node][c]['weight']
            result += temp

        return result

    # input a node
    def getOneHopInfluence(self, node):
        result = 1
        for c in self.graph.successors(node):
            result += self.graph[node][c]['weight']
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
                for d in S2:
                    result += self.graph[s][c]['weight'] * self.graph[c][d]['weight']
        return result

