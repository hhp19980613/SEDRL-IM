import random

import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import sys


o_path = os.getcwd()  # 返回当前工作目录
sys.path.append(o_path)

from influence import Env

class DQNAgent:
    def __init__(self, output_size, node_num, nstep):

        self.output_size = output_size
        self.node_num = node_num
        self.memory = deque(maxlen=3000)
        self.memory1 = deque(maxlen=1000)
        self.memoryFlag = 0

        self.gamma = 0.8  # discount rate
        self.epsilon = 0  # exploration rate
        self.epsilon_min = 0.00001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.nstep = nstep

        self.stateList = deque(maxlen=self.nstep)  # store last nstep states
        self.rewardList = deque(maxlen=self.nstep)  # store last nstep rewards
        self.actionList = deque(maxlen=self.nstep)  # store last nstep actions
        self.model = self._build_model()
        self.TargetModel = self._build_model()

        self.kmeansNum = 0  # kmeans score


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.output_size, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.memoryFlag = self.memoryFlag + 1
        if self.memoryFlag == 25 and reward > 0:
            self.memoryFlag = 0
            self.memory1.append((state, action, reward, next_state, done))
            sample = random.sample(self.memory1, 1)
            for state, action, reward, next_state, done in sample:
                self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        print("epsilon = ", self.epsilon)
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.node_num)
            print("random select:", action)
            return action

        input_state = env.embed2input(state)
        act_values = self.model.predict(input_state)

        act_values = act_values.reshape((self.node_num,))
        act_values = np.argsort(np.array(act_values))
        act_values = act_values[::-1]
        action = act_values[0]

        print("greedy select", action)
        return action  # returns action

    def replay(self, batch_size):
        print("replay")
        minibatch = random.sample(self.memory, batch_size)
        inputList = []
        targetList = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                input_next_state = env.embed2input(next_state)
                next_act_values = self.model.predict(input_next_state)
                next_act_values = np.array(next_act_values).T
                target = (reward + self.gamma * np.amax(next_act_values[0]))

            input_state = env.embed2input(state)
            target_f = self.model.predict(input_state)
            target_f = np.array(target_f).T
            target_f[0][action] = target

            inputList.append(input_state[action])
            targetList.append(target_f[0][action])


        inputList = np.array(inputList).reshape((batch_size, self.output_size))
        targetList = np.array(targetList).reshape((batch_size,))
        history = self.model.fit(np.array(inputList), np.array(targetList), epochs=1, verbose=0, batch_size=1)
        loss = history.history['loss'][0]
        # env.appendData("loss", loss)

        # Keeping track of loss
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def reset_parameter(self, node_num, kmeansNum):
        print("reset_parameter")
        self.node_num = node_num

        self.epsilon = 1.0  # exploration rate
        self.stateList.clear()
        self.rewardList.clear()
        self.actionList.clear()



        self.kmeansNum = kmeansNum

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



if __name__ == "__main__":
    edges = [(1, 2, 0.5), (3, 1, 0.75), (4, 1, 0.3), (2, 4, 0.2), (4, 3, 0.3), (3, 5, 0.3), (5, 4, 0.4), (5, 6, 0.2), (6, 4, 0.3), (6, 2, 0.4)]
    env = Env()

    EPISODES = 1
    nstep = 4
    batch_size = 4
    targetUpdateStep = 4
    targetUpdateFlag = 0
    state = env.state

    agent = DQNAgent(32*32, env.nodeNum, nstep)
    for e in range(EPISODES):
        print("A new episode")
        env.reset()
        influence = 0
        influenceList = np.array([])
        for time in range(2000):
            targetUpdateFlag += 1
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            influence += reward
            influenceList = np.append(influenceList,influence)

            agent.memorize(state, action, reward, next_state, done)

            state = next_state

            if time < nstep + batch_size:
                continue

            loss = agent.replay(batch_size)
            if targetUpdateStep >= targetUpdateFlag:
                agent.save("model//model_parameter")
                agent.TargetModel.load_weights("model//model_parameter")
                targetUpdateFlag = 0
                np.savetxt("influence.txt",influenceList)


