
import numpy as np
import random
import torch
import networkx as nx
# import matlab.engine
import os
import scipy.io as scio


class Env:

    def __init__(self):
        self.eng = matlab.engine.start_matlab()  # 可以为所欲为的调用matlab内置函数
        self.dim = 64 + 3
        self.edgeList = []
        self.seeds = set()
        self.influence = 0
        self.graphNum = 0
        self.graphIndex = -1
        self.degreeScore = []
        self.posi_degreeScore = []
        self.randomScore = []

        self.nodeNum = 329
        self.maxSeedsNum = 10
        self.networkName = 'EGFR'
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
        # self.nameList = ['../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.10.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.30.txt']

        self.localInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        self.oneHopInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded


        self.graphIndex = -1
        self.nextGraph()

    def constrctGraph(self,edges):
        graph = nx.DiGraph()
        posi_graph = nx.DiGraph()
        neg_graph = nx.DiGraph()
        for u, v, sign in edges:
            u = int(u) - 1
            v = int(v) - 1
            p = 0.1
            graph.add_edge(u, v, weight=p)
            if sign == 1:
                posi_graph.add_edge(u, v, weight=p)

            elif sign == -1:

                neg_graph.add_edge(u, v, weight=p)



        return graph, posi_graph, neg_graph, np.array(graph.edges())

    def nextGraph(self):
        self.graphIndex += 1
        graph_file = self.nameList[self.graphIndex]
        self.graph, self.posi_graph, self.neg_graph, self.edges = self.constrctGraph(np.loadtxt(graph_file))
        self.eng.init_embed(self.networkName, self.nodeNum, self.dim - 3)
        self.embedInfo = self.getembedInfo()
        self.nodeNum = len(self.graph.nodes())
        self.seeds = set()
        self.influence = 0
        self.localInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        self.oneHopInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded

    def reset(self):
        self.seeds = set([])
        self.influence = 0

    def degree(self):
        outdegree = self.graph.out_degree()
        a = sorted(outdegree.items(), key=lambda i: i[1], reverse=True)
        self.seeds = set()
        for i in range(self.maxSeedsNum):
            self.seeds.add(a[i][0])
        self.degreeScore.append(self.getInfluence(self.seeds))
        print("degree seeds:", self.seeds)


    def degreePosi(self):
        outdegree = self.posi_graph.out_degree()
        a = sorted(outdegree.items(), key=lambda i: i[1], reverse=True)
        self.seeds = set()
        for i in range(self.maxSeedsNum):
            self.seeds.add(a[i][0])

        self.posi_degreeScore.append(self.getInfluence(self.seeds))
        print("positive degree seeds:", self.seeds)

    def randomChoose(self):
        seeds = set([])
        sum = 0
        random_times = 20
        for _ in range(random_times):
            seeds = set([])
            a = random.sample(range(self.nodeNum), self.maxSeedsNum)
            for node in a:
                seeds.add(node)
            # sum += self.getSingleNetworkInfluence(seeds,self.sumGraph)
            i = self.getInfluence(seeds)
            # print(seeds)
            # print(i)
            sum += i
        self.randomScore.append(sum / random_times)

    def seeds2input(self,seeds):
        input = np.array(self.embedInfo)
        flagList = np.array([])
        degreeList = np.array([])
        posi_degreeList = np.array([])
        for i in range(self.nodeNum):
            degreeList = np.append(degreeList, self.graph.out_degree[i])
            posi_degreeList = np.append(degreeList, self.posi_graph.out_degree[i])
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


    def getembedInfo(self):
        try:
            embedInfo = np.loadtxt("../data/e/" + self.networkName + str(self.graphIndex))
            # embedInfo = np.loadtxt("../data/embedding/" + self.networkName + str(self.graphIndex))
            np.savetxt("result/embedding/" + self.networkName + str(self.graphIndex), embedInfo)
            # embedInfo = np.loadtxt("../data/embedding/" + self.networkName + "matrix" + str(self.graphIndex))
        except:
            path = '..//data//embedding//' + self.networkName + '.mat'
            data = scio.loadmat(path)
            embedInfo = data['embedInfo']

        return embedInfo


    def getInfluence(self, S):
        influence = 0
        for s in S:
            temp = self.getLocalInfluence(s)
            # print(s,":",temp)
            influence += temp

        # print("1111111111111111111:",influence)
        temp = self.getEpsilon(S)
        influence -= temp
        # print("222222222222222222:",influence)
        for s in S:
            Cs = set(self.graph.successors(s))
            S1 = S & Cs
            for s1 in S1:
                influence -= self.graph[s][s1]['weight'] * self.getOneHopInfluence(s1)
        return influence

    # one node local influence
    def getLocalInfluence(self, node):
        if self.localInfluenceList[node] >= 0:
            ...
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
            ...
            return self.oneHopInfluenceList[node]

        result = 1
        Cc = self.graph.successors(node)
        for c in Cc:
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

    def showScore(self):
        out = ""
        print("degree:")
        for i in range(len(self.degreeScore)):
            out = out + str(self.degreeScore[i])
            out = out + "\t"
        print(out)
        out = ""
        print("positive degree:")
        for i in range(len(self.posi_degreeScore)):
            out = out + str(self.posi_degreeScore[i])
            out = out + "\t"
        print(out)
        out = ""
        print("random:")
        for i in range(len(self.randomScore)):
            out = out + str(self.randomScore[i])
            out = out + "\t"
        print(out)


if __name__ == "__main__":
    env = Env()
    # l = [6098,6960,6655,5351,7297,3784,4662,1227,6932,1143,1689,5848,5159,7640,369,7129,4516,5529,4466,5121,5231,4951,5345,7286,521,7102,1495,
    #      435,6859,6867,6765,6555,6608,970,2348,7338,7144,897,7061,3870,3346,7062,7585,2271,3876,7173,604,6924,6318,3582]
    # l = l[0:10]
    # for i in range(len(l)):
    #     l[i] = l[i]
    # seeds = set(l)
    # print(len(seeds))
    # print(env.getInfluence(seeds))

    for i in range(len(env.nameList)):
        seeds = []
        print("env.nodeNum", env.nodeNum)
        env.degree()
        env.degreePosi()
        env.randomChoose()

        # print("nodeNum:", len(env.graph.nodes()))
        # print(env.getInfluence(seeds))


        if i < len(env.nameList) - 1:
            env.nextGraph()

    env.showScore()
