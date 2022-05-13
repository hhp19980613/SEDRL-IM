import random

import numpy as np
from collections import deque

from torch.optim import Adam
import torch.nn as nn
import torch
import os
import sys

o_path = os.getcwd()  # 返回当前工作目录
sys.path.append(o_path)

from influence1 import Env
from DQN_Model import DQN

class DQNAgent:
    def __init__(self, dim, node_num, nstep):
        self.dqn = DQN(dim)
        self.output_size = dim
        self.optim = Adam(self.dqn.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

        self.node_num = node_num
        self.memory = deque(maxlen=3000)
        self.memory1 = deque(maxlen=1000)
        self.memoryFlag = 0

        self.gamma = 0.8  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.00001
        self.epsilon_decay = 0.6
        self.learning_rate = 0.001
        self.nstep = nstep

        self.stateList = deque(maxlen=self.nstep)  # store last nstep states
        self.rewardList = deque(maxlen=self.nstep)  # store last nstep rewards
        self.actionList = deque(maxlen=self.nstep)  # store last nstep actions

    def memorize(self, state, action, reward, next_state, done):
        sumReward = sum(self.rewardList)
        self.memory.append((self.stateList[0], self.actionList[0], sumReward, self.stateList[-1], done))
        self.memoryFlag = self.memoryFlag + 1
        if self.memoryFlag == 25 and reward > 2:
            self.memoryFlag = 0
            self.memory1.append((self.stateList[0], self.actionList[0], sumReward, self.stateList[-1], done))
            sample = random.sample(self.memory1, 1)
            for state, action, reward, next_state, done in sample:
                self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.node_num)
            print("random select:", action, "    epsilon = ", self.epsilon)
            return action

        input_state = env.seeds2input(state)
        act_values = self.dqn.forward(torch.from_numpy(input_state).float())
        act_values = act_values.detach().numpy()

        act_values = act_values.reshape((self.node_num,))
        act_values = np.argsort(np.array(act_values))
        act_values = act_values[::-1]
        action = act_values[0]

        print("greedy select", action, "    epsilon = ", self.epsilon)
        return action  # returns action

    def replay(self, batch_size):
        print("replay")
        minibatch = random.sample(self.memory, batch_size)
        currentList = []
        targetList = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                input_next_state = env.seeds2input(next_state)
                next_act_values = self.dqn.forward(torch.from_numpy(input_next_state).float())
                next_act_values = next_act_values.detach().numpy()
                next_act_values = np.array(next_act_values).T
                target = (reward + self.gamma * np.amax(next_act_values[0]))

            input_state = env.seeds2input(state)
            target_f = self.dqn.forward(torch.from_numpy(input_state).float())
            target_f = target_f.detach().numpy()
            target_f = np.array(target_f).T

            currentList.append(target_f[0][action])
            target_f[0][action] = target
            targetList.append(target_f[0][action])
        # end for

        self.optim.zero_grad()
        currentList = torch.Tensor(currentList)
        currentList = currentList.requires_grad_()
        targetList = torch.Tensor(targetList)
        dt = self.loss(currentList, targetList)
        dt.backward()
        nn.utils.clip_grad_norm(self.dqn.parameters(), 10000)
        self.optim.step()

        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset_parameter(self, node_num, kmeansNum):
        print("reset_parameter")
        self.node_num = node_num

        self.epsilon = 1.0  # exploration rate
        self.stateList.clear()
        self.rewardList.clear()
        self.actionList.clear()

        self.kmeansNum = kmeansNum

    def load(self, name):
        torch.load(self.dqn, name)

    def save(self, name):
        torch.save(self.dqn, name)


# 被选中的点的embedding置0
if __name__ == "__main__":

    env = Env()

    EPISODES = 100
    # EPISODES = 0
    nstep = 1
    batch_size = 4
    targetUpdateStep = 4
    targetUpdateFlag = 0
    state = env.seeds
    influenceList = np.array([])
    agent = DQNAgent(32, env.nodeNum, nstep)
    for e in range(EPISODES):
        print("A new episode")
        env.reset()
        agent.epsilon = 1
        influence = 0
        agent.learning_rate *= 0.9

        for times in range(500):

            print("====================================")
            print("EPISODES:", e, "    time:", times)
            targetUpdateFlag += 1
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            print("reward:", reward, "      influence:", env.influence)
            agent.stateList.append(state)  # record for n-step
            agent.rewardList.append(reward)
            agent.actionList.append(action)

            influence += reward
            influenceList = np.append(influenceList,influence)

            if times >= nstep:
                agent.memorize(state, action, reward, next_state, done)

            state = next_state

            if times == 499:
                print(env.seeds)

            if times < nstep + batch_size:
                continue

            agent.replay(batch_size)
            if targetUpdateStep <= targetUpdateFlag:
                agent.save("model//model_parameter")
                # agent.TargetModel.load_weights("model//model_parameter")
                targetUpdateFlag = 0
                np.savetxt("influence.txt",influenceList)

    agent.save("model//model_parameter")