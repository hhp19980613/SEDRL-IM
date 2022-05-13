import numpy as np, os, time, sys, random

import torch
from torch.optim import Adam
import torch.nn as nn
import replay_memory

import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from influence1 import Env
from SSNE import SSNE
from DQN_Model import DQN
import mod_utils as utils


class Agent:
    def __init__(self, env):
        self.is_cuda = True
        self.is_memory_cuda = True
        self.batch_size = 512
        self.use_done_mask = True
        self.pop_size = 100
        self.buffer_size = 10000
        self.randomWalkTimes = 20
        self.learningTimes = 3
        self.action_dim = None  # Simply instantiate them here, will be initialized later
        self.dim = env.dim

        self.env = env
        self.evolver = SSNE(self.pop_size)
        self.evalStep = env.maxSeedsNum  # step num during evaluation

        # Init population
        self.pop = []
        for _ in range(self.pop_size):
            self.pop.append(DQN(self.dim).cuda())
        self.all_fitness = []

        # Turn off gradients and put in eval mode
        for dqn in self.pop:
            dqn.eval()

        # Init RL Agent
        self.rl_agent = DQN(self.dim)
        self.gamma = 0.8  # discount rate
        self.optim = Adam(self.rl_agent.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=0.9995, last_epoch=-1)
        self.loss = nn.MSELoss()
        self.replay_buffer = replay_memory.ReplayMemory(self.buffer_size)

        # Trackers
        self.num_games = 0;
        self.num_frames = 0;
        self.gen_frames = 0

    def add_experience(self, state, action, next_state, reward, done):
        reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.is_cuda: reward = reward.cuda()
        if self.use_done_mask:
            done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.is_cuda: done = done.cuda()

        self.replay_buffer.push(state, action, next_state, reward, done)

    def evaluate(self, net, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        state = utils.to_tensor(state).unsqueeze(0)
        # print("======running======")
        if self.is_cuda:
            
            state = state.cuda()
        done = False
        seeds = []
        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            Qvalues = net.forward(state)
            Qvalues = Qvalues.reshape((Qvalues.numel(),))
            sorted, indices = torch.sort(Qvalues, descending=True)
            
            actionNum = 0

            for i in range(state.shape[1]):
                if state[0][indices[i]][0].item() == 1:  # choose node that is not seed
                    actionNum += 1
                    actionInt = indices[i].item()
                    seeds.append(actionInt)
                    action = torch.tensor([actionInt])

                    next_state, reward, done = self.env.step(actionInt)  # Simulate one step in environment

                    next_state = utils.to_tensor(next_state).unsqueeze(0)

                    if self.is_cuda:
                        next_state = next_state.cuda()
                    total_reward += reward
                    if store_transition: self.add_experience(state.cpu(), action, next_state.cpu(), reward, done)
                    state = next_state

                    if actionNum == self.evalStep or done:  # finish after self.evalStep steps
                        break
            # end of for
        # end of while
        if store_transition: self.num_games += 1

        return total_reward, seeds

    def randomWalk(self):
        total_reward = 0.0
        state = self.env.reset()
        state = utils.to_tensor(state).unsqueeze(0)
        if self.is_cuda:
            state = state.cuda()
        done = False
        actionList = [i for i in range(self.env.nodeNum)]
        actionIndex = 0
        random.shuffle(actionList)
        while not done:
            self.num_frames += 1
            self.gen_frames += 1
            actionInt = actionList[actionIndex]
            action = torch.tensor([actionInt])
            next_state, reward, done = self.env.step(actionInt)  # Simulate one step in environment
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            if self.is_cuda:
                next_state = next_state.cuda()
            total_reward += reward
            self.add_experience(state.cpu(), action, next_state.cpu(), reward, done)
            state = next_state
            actionIndex += 1
        self.num_games += 1
        return total_reward

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    def evaluate_all(self):
        self.all_fitness = []
        t1 = time.time()
        for net in self.pop:
            fitness, _ = self.evaluate(net)
            self.all_fitness.append(fitness)
        best_train_fitness = max(self.all_fitness)
        print("fitness_init:", best_train_fitness)
        t2 = time.time()
        print("evaluate finished.    cost time:", t2 - t1)

    def train(self):
        self.gen_frames = 0
        ####################### EVOLUTION #####################
        t1 = time.time()
        #for _ in range(self.randomWalkTimes):
        #    self.randomWalk()
        best_train_fitness = max(self.all_fitness)
        new_pop = self.evolver.epoch(self.pop, self.all_fitness)
        new_pop_fitness = []
        for net in new_pop:
            fitness, _ = self.evaluate(net)
            new_pop_fitness.append(fitness)
        self.pop, self.all_fitness = self.get_offspring(self.pop, self.all_fitness, new_pop, new_pop_fitness)
        t2 = time.time()
        print("epoch finished.    cost time:", t2 - t1)
        fitness_best, _ = self.evaluate(self.pop[0], True)
        ####################### RL Learning #####################
        t1 = time.time()
        for _ in range(self.learningTimes):
            # worst_index = self.all_fitness.index(min(self.all_fitness))
            index = random.randint(len(self.pop) // 2, len(self.pop) - 1)
            self.rl_to_evo(self.pop[0], self.rl_agent)
            if len(self.replay_buffer) > self.batch_size * 2:
                transitions = self.replay_buffer.sample(self.batch_size)
                batch = replay_memory.Transition(*zip(*transitions))
                self.update_parameters(batch) 
                fitness, _ = self.evaluate(self.rl_agent, True) 
                if fitness_best < fitness:
                  self.rl_to_evo(self.rl_agent, self.pop[index])
                  self.all_fitness[index] = fitness         
        t2 = time.time()
        print("learning finished.    cost time:", t2 - t1)
        return best_train_fitness, sum(self.all_fitness) / len(self.all_fitness), self.rl_agent, self.pop[
                                                                                                 0:len(self.pop) // 10]

    def update_parameters(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = None
        if self.use_done_mask: done_batch = torch.cat(batch.done)
        state_batch.volatile = False;
        next_state_batch.volatile = True;
        action_batch.volatile = False

        # Load everything to GPU if not already
        if self.is_cuda:
            self.rl_agent.cuda()
            state_batch = state_batch.cuda();
            next_state_batch = next_state_batch.cuda();
            action_batch = action_batch.cuda();
            reward_batch = reward_batch.cuda()
            if self.use_done_mask: done_batch = done_batch.cuda()

        currentList = torch.Tensor([])
        currentList = torch.unsqueeze(currentList, 1).cuda()
        targetList = torch.Tensor([])
        targetList = torch.unsqueeze(targetList, 1).cuda()
        # DQN Update
        for state, action, reward, next_state, done in zip(state_batch, action_batch, reward_batch, next_state_batch,
                                                           done_batch):
            target = torch.Tensor([reward])
            if not done:
                next_q_values = self.rl_agent.forward(next_state)
                pred, idx = next_q_values.max(0)
                target = reward + self.gamma * pred

            target_f = self.rl_agent.forward(state)

            current = target_f[action]
            current = torch.unsqueeze(current, 1)
            target = torch.unsqueeze(target, 1).cuda()
            currentList = torch.cat((currentList, current), 0)
            targetList = torch.cat((targetList, target), 0)

        self.optim.zero_grad()
        dt = self.loss(currentList, targetList)
        dt.backward()
        nn.utils.clip_grad_norm(self.rl_agent.parameters(), 10000)
        self.optim.step()

        # Nets back to CPU if using memory_cuda
        if self.is_memory_cuda and not self.is_cuda:
            self.rl_agent.cpu()

    def get_offspring(self, pop, fitness_evals, new_pop, new_fitness_evals):
        all_pop = []
        fitness = []
        offspring = []
        offspring_fitness = []
        for i in range(len(pop)):
            all_pop.append(pop[i])
            fitness.append(fitness_evals[i])
        for i in range(len(new_pop)):
            all_pop.append(new_pop[i])
            fitness.append(new_fitness_evals[i])

        index_rank = sorted(range(len(fitness)), key=fitness.__getitem__)
        index_rank.reverse()
        for i in range(len(pop) // 2):
            offspring.append(all_pop[index_rank[i]])
            offspring_fitness.append(fitness[index_rank[i]])

        randomNum = len(all_pop) - len(pop) // 2
        randomList = list(range(randomNum))
        random.shuffle(randomList)
        for i in range(len(pop) // 2, len(pop)):
            index = randomList[i - len(pop) // 2]
            offspring.append(all_pop[index])
            offspring_fitness.append(fitness[index])
            ...

        return offspring, offspring_fitness

    def showScore(self, score):
        out = ""
        for i in range(len(score)):
            out = out + str(score[i])
            out = out + "\t"
        print(out)


def getResultPath():
    for i in range(1, 1000):
        path = "result/" + str(i) + "/"
        if os.path.exists(path):
            # print(i, "repeat===========================")
            continue
        os.makedirs(path)
        os.makedirs(path + "/embedding/")
        os.makedirs(path + "/model/")
        print("create directory:", i)
        return path


def run(maxSeedsNum):
    # Create Env
    t1 = time.time()
    print("start===========================")
    mainPath = getResultPath()

    env = Env(mainPath, maxSeedsNum)
    # Random seed setting
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create Agent
    agent = Agent(env)

    next_save = 100;
    time_start = time.time()
    print("Start training...")
    maxList = np.array([])
    resultList = np.array([])
    timeList = np.array([])
    for graphIndex in range(len(agent.env.nameList)):  # 最后一张图不训练
        for i in range(100):  # Generation
            if i == 0:
                agent.evaluate_all()
            print("=================================================================================")
            print(
            graphIndex, "th graph      Generation:", i, "    ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            best_train_fitness, average, rl_agent, elitePop = agent.train()
            print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f' % best_train_fitness,
                  ' Avg:', average)
            maxList = np.append(maxList, best_train_fitness)
            # Save Policy

            torch.save(rl_agent, mainPath + "//model//rlmodel" + str(graphIndex))
            for eliteIndex in range(len(elitePop)):
                torch.save(elitePop[eliteIndex],
                           mainPath + "//model//elite_model" + str(graphIndex) + "_" + str(eliteIndex))
            np.savetxt(mainPath + "//maxList.txt", maxList)
            # print("Progress Saved")

        fitness, seeds = agent.evaluate(agent.pop[0])
        print("best fitness:", fitness)
        print("seeds:", seeds)
        resultList = np.append(resultList, fitness)
        t2 = time.time()
        timeList = np.append(timeList, t2 - t1)
        t1 = t2
        if graphIndex < len(agent.env.nameList) - 1:
            agent.env.nextGraph()
            agent.replay_buffer = replay_memory.ReplayMemory(agent.buffer_size)
        else:
            break

    print("time cost:")
    agent.showScore(timeList)

    print("influence:")
    agent.showScore(resultList)
    np.savetxt(mainPath + "//timeList.txt", timeList)
    np.savetxt(mainPath + "//resultList.txt", resultList)
    np.savetxt(mainPath + "//seeds.txt", seeds)
    file = 'result//timeList.txt'
    with open(file, 'a') as f:
        for i in range(len(timeList)):
            f.write(str(timeList[i]) + "\t")
        f.write("\n")
    file = 'result//resultList.txt'

    with open(file, 'a') as f:
        for i in range(len(resultList)):
            f.write(str(resultList[i]) + "\t")
        f.write("\n")


if __name__ == "__main__":
    #for i in range(3, 11):

    run(1)


