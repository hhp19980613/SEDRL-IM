# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000

class DQNTestAgent:
    def __init__(self, output_size, node_num):

        self.output_size = output_size
        self.node_num = node_num
        self.gamma = 0.95  # discount rate
        self.epsilon = 0  # exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.env = gym.make('test_SNetwork-v0')
        self.freeze = deque(maxlen=int(node_num / 50))  # 最多冻结50分之一的点
        self.freeze1 = deque(maxlen=int(node_num))  # 最多冻结50分之一的点
        # self.standNum = 0  # times of standing in the interval
        # self.intervalTop = 0
        # self.intervalBottom = 0
        # self.kmeansFlag = False  # denotes whether have reached kmeans baseline
        self.kmeansNum = 0  # kmeans score

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.output_size, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        input_state = self.env.state2input(state)
        act_values = self.model.predict(input_state)
        act_values = np.array(act_values).T
        # action = np.argmax(act_values[0])
        #print("act_values = ", act_values)
        # print("is all elements equal to each other =======================",self.judgeRepeatedThird(act_values[0]))
        action = self.getAction(act_values[0].tolist())
        # self.env.appendData("q_value", act_values[0][action])
        print("greedy select",action)
        return action  # returns action

    def reset_parameter(self, node_num, kmeansNum):
        print("reset_parameter")
        self.node_num = node_num
        self.kmeansNum = kmeansNum
        self.freeze = deque(maxlen=int(node_num / 50))  # 最多冻结50分之一的点
        self.freeze1 = deque(maxlen=int(node_num))      # 永久冻结

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def getAction(self, act_values):
        act_values = np.argsort(np.array(act_values))
        act_values = act_values[::-1]
        for i in range(0, self.node_num):
            # reward = self.env.getReward(act_values[i])
            # print("rrrrrrrrrrrrrrrrrreward:",reward)
            # if reward<=0:
            #     continue
            if act_values[i] not in self.freeze and act_values[i] not in self.freeze1:
                # self.freeze.append(act_values[i])
                return act_values[i]
        return -1

    def judgeRepeatedThird(self, array):
        if len(set(array)) == 1:
            return True
        else:
            return False


    def test(self):
        done = False
        Flag = 0
        lastNode = 0
        action = 0

        self.env.reset()
        for i in range(self.env.graphNum):  # traversal all graph
            name = "model//model_parameter" + str(i) + "_8"
            self.model.load_weights(name)

            print("env.graphNum:", self.env.graphNum)
            totalNodeNum = self.env.nextGraph()
            subGraphNum = 10
            print("subGraphNum:",subGraphNum)
            for j in range(subGraphNum):  # traversal all subgraph in i'th graph

                state, output_size, node_num, kmeansNum, min = self.env.nextSubGraph()
                self.reset_parameter(node_num, kmeansNum)
                self.env.initData("reward")
                self.env.initData("error_rate")
                self.env.initData("q_value")
                self.env.initData("action")
                self.env.initData("result")
                numList = np.zeros(node_num)
                error_rate = 1

                minIdx = state.copy()
                for time in range(1000):  # repeat train one subgraph
                    print("graph:", self.env.graphSet[self.env.graphIndex], " subGraph:", self.env.subGraphIndex, " DQN:", time," kmeans:",self.kmeansNum)
                    lastNode = action
                    action = self.act(state)
                    if action==-1:
                        break
                    self.env.appendData("action", action)

                    if lastNode == action:
                        Flag += 1
                    else:
                        Flag = 0
                    numList[action] += 1
                    print("Flag = ", Flag,"   /   numList:",numList[action])
                    if numList[action] >= 7:
                        self.freeze1.append(action)
                    if Flag >= 3:
                        Flag = 0
                        self.freeze.append(action)
                        print("freeze:", self.freeze)

                    next_state, reward, done, _, error_rate = self.env.step(action)
                    if min > error_rate:
                        min = error_rate
                        minIdx = next_state.copy()
                    state = next_state
                # end of "time" for

                if min>0.3:
                    state, output_size, node_num, kmeansNum = self.env.repeatSubGraph()
                    self.reset_parameter(node_num, kmeansNum)
                    # self.env.initData("reward")
                    # self.env.initData("error_rate")
                    # self.env.initData("q_value")
                    # self.env.initData("action")
                    # self.env.initData("result")
                    numList = np.zeros(node_num)
                    error_rate = 1

                    minIdx = state.copy()
                    for time in range(1000):  # repeat train one subgraph
                        print("graph:", self.env.graphSet[self.env.graphIndex], " subGraph:", self.env.subGraphIndex,
                              " DQN:", time, " kmeans:", self.kmeansNum)
                        lastNode = action
                        action = self.act(state)
                        if action == -1:
                            break
                        self.env.appendData("action", action)

                        if lastNode == action:
                            Flag += 1
                        else:
                            Flag = 0
                        numList[action] += 1
                        print("Flag = ", Flag, "   /   numList:", numList[action])
                        if numList[action] >= 7:
                            self.freeze1.append(action)
                        if Flag >= 3:
                            Flag = 0
                            self.freeze.append(action)
                            print("freeze:", self.freeze)

                        next_state, reward, done, _, error_rate = self.env.step(action)
                        if min > error_rate:
                            min = error_rate
                            minIdx = next_state.copy()
                        state = next_state
                # end of "time" for
                print("min1:", min)
                self.env.appendData("result", min)
                idx, min = self.env.greedySearch(minIdx)
                print("min2:", min)
                self.env.appendData("result", min)
            # end of "j" for
        # end of "i" for










def main():
    agent = DQNTestAgent(64 * 64, 700)
    agent.test()


if __name__ == '__main__':
    main()
    # print(__name__)
