import numpy as np, os, time, sys, random
import torch
from torch.optim import Adam
import torch.nn as nn
import Replay_Memory
import argparse
from Influence_Propagation import Env
from Evolutionary_Algorithm import EA
from DQN_Model import DQN
import Mod_Utils as Utils

# 本程序基于CUDA技术利用GPU进行加速，此处设置采用1号显卡 #
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Agent:
    def __init__(self, env):
        self.is_cuda = True  # 设置为使用CUDA技术
        self.is_memory_cuda = True # 设置为使用GPU显存
        self.batch_size = 512 # 设置批次大小
        self.use_done_mask = True
        self.pop_size = 100 # 设置种群大小
        self.buffer_size = 10000 # 设置缓存池大小
        self.randomWalkTimes = 20 # 设置基于DQN随机选点次数
        self.learningTimes = 3 # 设置基于DRL技术加速DQN训练的次数
        self.dim = env.dim # 设置DQN输入层维数
        self.env = env # 初始化影响传播环境
        self.evolver = EA(self.pop_size) #初始化
        self.evalStep = env.maxSeedsNum  # 基于种子节点数设置DQN选点次数
        # 初始化DQN种群
        self.pop = []
        for _ in range(self.pop_size):
            self.pop.append(DQN(self.dim).cuda())
        # 初始化DQN种群对应的适应值数组
        self.all_fitness = []
        # Turn off gradients and put in eval mode
        for dqn in self.pop:
            dqn.eval()
        # 初始化最优DQN
        self.rl_agent = DQN(self.dim)
        # 初始DQN各项参数
        self.gamma = 0.8  # 设置更新比例
        self.optim = Adam(self.rl_agent.parameters(), lr=0.001) # 设置学习器
        self.loss = nn.MSELoss() # 设置使用均方误差作为损失函数
        self.replay_buffer = Replay_Memory.ReplayMemory(self.buffer_size) # 初始化缓冲池
        # 初始化性追踪器参数
        self.num_games = 0;
        self.num_frames = 0;
        self.gen_frames = 0

    # 基于CUDA技术将训练四元组数据存进缓存池 #
    def add_experience(self, state, action, next_state, reward, done):
        reward = Utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.is_cuda: reward = reward.cuda()
        if self.use_done_mask:
            done = Utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.is_cuda: done = done.cuda()

        self.replay_buffer.push(state, action, next_state, reward, done)

    # 基于DQN输出的节点分数选择种子节点并计算出适应值，同时将选点过程中的四元组数据缓存 #
    def evaluate(self, net, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        state = Utils.to_tensor(state).unsqueeze(0)
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
                if state[0][indices[i]][0].item() == 1:  # 选择非种子节点
                    actionNum += 1
                    actionInt = indices[i].item()
                    seeds.append(actionInt)
                    action = torch.tensor([actionInt])

                    next_state, reward, done = self.env.step(actionInt)  # 基于每个action的节点分数选择种子节点

                    next_state = Utils.to_tensor(next_state).unsqueeze(0)

                    if self.is_cuda:
                        next_state = next_state.cuda()
                    total_reward += reward
                    if store_transition: self.add_experience(state.cpu(), action, next_state.cpu(), reward, done)
                    state = next_state

                    if actionNum == self.evalStep or done:
                        break
            # end of for
        # end of while
        if store_transition: self.num_games += 1

        return total_reward, seeds

    def randomWalk(self):
        total_reward = 0.0
        state = self.env.reset()
        state = Utils.to_tensor(state).unsqueeze(0)
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

    # 将源DQN的网络权重复制给目标DQN的网络权重 #
    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    # 评估演化后的DQN种群的适应值 #
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

    # 基于演化算法和DRL技术训练演化DQN种群 #
    def train(self):
        self.gen_frames = 0
        ####################### EVOLUTION #####################

        t1 = time.time()
        #for _ in range(self.randomWalkTimes):
        #    self.randomWalk()

        # 获得最优适应值
        best_train_fitness = max(self.all_fitness)
        # 基于演化算法对DQN种群的网络权重进行演化，并更新新种群的适应值
        new_pop = self.evolver.epoch(self.pop, self.all_fitness)
        new_pop_fitness = []
        for net in new_pop:
            fitness, _ = self.evaluate(net)
            new_pop_fitness.append(fitness)
        self.pop, self.all_fitness = self.get_offspring(self.pop, self.all_fitness, new_pop, new_pop_fitness)
        t2 = time.time()
        print("epoch finished.    cost time:", t2 - t1)
        # 获得当前DQN种群的最优适应值
        fitness_best, _ = self.evaluate(self.pop[0], True)

        ####################### DRL Learning #####################
        # rl learning step
        t1 = time.time()
        # 基于DRL思想中n-step Q-learning技术利用缓存池中的经验数据反向更新最优DQN，并将其网络权重复制给适应值较差的DQN
        for _ in range(self.learningTimes):
            # 筛选出需要复制最优网络权重的目标DQN
            index = random.randint(len(self.pop) // 2, len(self.pop) - 1)
            # 获得最优DQN
            self.rl_to_evo(self.pop[0], self.rl_agent)
            # 基于训练数据更新最优DQN，并在更新后适应值提高的情况下将权重更新给目标DQN
            if len(self.replay_buffer) > self.batch_size * 2:
                transitions = self.replay_buffer.sample(self.batch_size)
                batch = Replay_Memory.Transition(*zip(*transitions))
                self.update_parameters(batch)
                fitness, _ = self.evaluate(self.rl_agent, True)
                if fitness_best < fitness:
                  self.rl_to_evo(self.rl_agent, self.pop[index])
                  self.all_fitness[index] = fitness

        t2 = time.time()
        print("learning finished.    cost time:", t2 - t1)
        return best_train_fitness, sum(self.all_fitness) / len(self.all_fitness), self.rl_agent, self.pop[
                                                                                                 0:len(self.pop) // 10]

    # 基于DRL思想中n-step Q-learning技术利用缓存池中的特定批次大小的经验数据计算误差值，并基于随机梯度下降技术采用误差值梯度反向更新DQN的网络权重 #
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

    # 在演化得到的超过种群规模上限的DQN种群中，先对所有DQN种群进行排序，然后选出其中前50个DQN种群保留，并在剩下的种群中随机选择50个种群进行保留 #
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

    # 格式化输出适应值分数 #
    def showScore(self, score):
        out = ""
        for i in range(len(score)):
            out = out + str(score[i])
            out = out + "\t"
        print(out)

# 设置程序结果和模型的存储路径 #
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

# 基于演化算法和强化学习思想求解特定种子规模下使得影响最大化的种子节点集 #
def run(maxSeedsNum):
    # Create Env
    t1 = time.time()
    print("start===========================")
    mainPath = getResultPath()

    env = Env(mainPath, maxSeedsNum)
    # 设置随机数种子
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
        # 运行100次迭代
        for i in range(100):  # Generation
            if i == 0:
                agent.evaluate_all()
            print("=================================================================================")
            print(
            graphIndex, "th graph      Generation:", i, "    ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # 进行训练
            best_train_fitness, average, rl_agent, elitePop = agent.train()
            print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f' % best_train_fitness,
                  ' Avg:', average)
            maxList = np.append(maxList, best_train_fitness)
            # 保存模型
            torch.save(rl_agent, mainPath + "//model//rlmodel" + str(graphIndex))
            for eliteIndex in range(len(elitePop)):
                torch.save(elitePop[eliteIndex],
                           mainPath + "//model//elite_model" + str(graphIndex) + "_" + str(eliteIndex))
            np.savetxt(mainPath + "//maxList.txt", maxList)


        fitness, seeds = agent.evaluate(agent.pop[0])
        print("best fitness:", fitness)
        print("seeds:", seeds)
        resultList = np.append(resultList, fitness)
        t2 = time.time()
        timeList = np.append(timeList, t2 - t1)
        t1 = t2
        # 更换网络并清空缓存池
        if graphIndex < len(agent.env.nameList) - 1:
            agent.env.nextGraph()
            agent.replay_buffer = Replay_Memory.ReplayMemory(agent.buffer_size)
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
    # 设置种子规模为1
    seedNum = 1
    run(seedNum)


