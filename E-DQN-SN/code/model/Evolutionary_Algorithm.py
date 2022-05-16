import random
import numpy as np
from scipy.special import expit
import math
import copy


# Neuroevolution SSNE
class EA:
    def __init__(self, pop_size):
        self.current_gen = 0 # 记录当前演化次数
        self.pop_size = pop_size # 设置演化种群规模
        self.elite_fraction = 0.2 # 设置竞标赛的保留比例
        self.num_elitists = int(self.elite_fraction * self.pop_size) # 设置竞标赛的保留数
        self.crossover_prob = 0.8 # 设置交叉概率
        self.mutation_prob = 0.2 # 设置变异概率
        if self.num_elitists < 1:
            self.num_elitists = 1

    # 基于三个为一组的竞标赛规则选出适应值较大的DQN种群作为优异种群 #
    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[random.randint(0, len(offsprings)-1)])
        return offsprings

    # 对种群适应值进行排序 #
    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    # 使得weight的绝对值不超过mag #
    def regularize_weight(self, weight, mag):
        if weight > mag:
            weight = mag
        if weight < -mag:
            weight = -mag
        return weight

    # 对两个DQN种群的权重采取交叉操作 #
    def crossover_inplace(self, gene1, gene2):
        for param1, param2 in zip(gene1.parameters(), gene2.parameters()):
            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2: #Weights no bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = random.randint(0, num_variables * 2 - 1)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = random.randint(0, W1.shape[0] - 1)  #
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        ind_cr = random.randint(0, W1.shape[0] - 1)
                        W2[ind_cr, :] = W1[ind_cr, :]

            elif len(W1.shape) == 1:    # Bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = random.randint(0, num_variables)      # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = random.randint(0, W1.shape[0] - 1)
                        W1[ind_cr] = W2[ind_cr]
                    else:
                        ind_cr = random.randint(0, W1.shape[0] - 1)
                        W2[ind_cr] = W1[ind_cr]

    # 对两个DQN种群的权重采取变异操作 #
    def mutate_inplace(self, gene):
        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        num_params = len(list(gene.parameters()))
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2
        model_params = gene.state_dict()

        for i, key in enumerate(model_params): # Mutate each param

            if key == 'lnorm1.gamma' or key == 'lnorm1.beta' or  key == 'lnorm2.gamma' or key == 'lnorm2.beta' or key == 'lnorm3.gamma' or key == 'lnorm3.beta': continue

            # References to the variable keys
            W = model_params[key]
            if len(W.shape) == 2:   # Weights, no bias

                num_weights= W.shape[0]*W.shape[1]
                ssne_prob = ssne_probabilities[i]

                if random.random() < ssne_prob:
                    num_mutations = random.randint(0, int(math.ceil(num_mutation_frac * num_weights)) - 1)  # Number of mutation instances
                    for _ in range(num_mutations):
                        ind_dim1 = random.randint(0, W.shape[0] - 1)
                        ind_dim2 = random.randint(0, W.shape[-1] - 1)
                        random_num = random.random()

                        if random_num < super_mut_prob:  # Super Mutation probability
                            W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
                        elif random_num < reset_prob:  # Reset probability
                            W[ind_dim1, ind_dim2] = random.gauss(0, 1)
                        else:  # mutauion even normal
                            W[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[ind_dim1, ind_dim2])

                        # Regularization hard limit
                        W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2], 1000000)

    # 将源DQN种群的网络权重复制给目标DQN种群的网络权重 #
    def clone(self, master, replacee):
        for target_param, source_param in zip(replacee.parameters(), master.parameters()):
            target_param.data.copy_(source_param.data)

    # 基于适应值和竞标赛规则划分优异种群与非优异种群，并对优异种群进行交叉操作，对非优异种群进行变异操作 #
    def epoch(self, pop, fitness_evals):
        pop_copy = []
        cross_pop = []
        mutate_pop = []
        for i in range(self.pop_size):
            pop_copy.append(copy.deepcopy(pop[i]))


        # 对DQN种群进行排序
        index_rank = self.list_argsort(fitness_evals)
        index_rank.reverse()

        # 优先选出特定比例的适应值较高的种群加入第一组优异种群
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # 基于竞标赛规则在剩余种群中选择适应值较高的种群加入第二组优异种群
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # 得到非优异种群
        unselects = []; new_elitists = []
        for i in range(self.pop_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # 随机淘汰一些优异种群
        for i in elitist_index:
            try: replacee = unselects.pop(0)
            except: replacee = offsprings.pop(0)
            new_elitists.append(replacee)
            self.clone(master=pop[i], replacee=pop[replacee])


        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            index = random.randint(0, len(unselects)-1)
            unselects.append(unselects[index])

        # 对第一组和第二组优异种群采取交叉操作
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists)
            off_j = random.choice(offsprings)
            self.clone(master=pop[off_i], replacee=pop[i])
            self.clone(master=pop[off_j], replacee=pop[j])
            self.crossover_inplace(pop[i], pop[j])
            cross_pop.append(copy.deepcopy(pop[i]))
            cross_pop.append(copy.deepcopy(pop[j]))

        # 对第一组优异种群的任意两个种群采取交叉操作
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.crossover_prob:
                self.crossover_inplace(pop[i], pop[j])
                cross_pop.append(copy.deepcopy(pop[i]))
                cross_pop.append(copy.deepcopy(pop[j]))

        # 对非优异种群采取交叉操作
        for i in range(self.pop_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.mutation_prob:
                    self.mutate_inplace(pop[i])
                    mutate_pop.append(copy.deepcopy(pop[i]))

        for i in range(self.pop_size):
            pop[i] = pop_copy[i]

        all_pop = []
        all_pop.extend(cross_pop)
        all_pop.extend(mutate_pop)
        return all_pop

# 重构数组维数，工具函数 #
def unsqueeze(array, axis=1):
    if axis == 0: return np.reshape(array, (1, len(array)))
    elif axis == 1: return np.reshape(array, (len(array), 1))


