# SEDRL-IM
## 一 代码所属论文说明
Code of Influence Maximization Research in Signed Networks by Using Evolutionary Deep Reinforcement Learning
## 二 python 环境说明
1）环境配置文件：environment.yml <br>
2）重要环境 <br>
  python                    3.7.0 <br>
  keras-applications        1.0.8 <br>
  keras-preprocessing       1.1.2 <br>
  tensorflow                1.14.0 <br>
  torch                     1.8.1+cu111 <br>
  networkx                  2.5.1 <br>
 ## 三 代码说明
 ### 3.1 代码文件总览
 1）Mod_Utils.py 工具函数<br> 
 2）DQN_Model.py DQN 网络模型<br>
 3）Replay_Memory.py 缓存池及存取函数<br>
 4）Evolutionary_Algorithm.py 演化算法相关函数<br>
 5）Influence_Propagation.py 影响传播函数<br>
 6）Run_SEDRL_IM.py 训练及求解函数<br>
 ### 3.2 运行代码指令
 python Run_SEDRL_IM.py
 ### 3.3 函数功能
 #### 3.3.1 Mod_Utils.py
 矩阵处理函数
 #### 3.3.2 DQN_Model.py
 DQN网络的数据结构，采用两层神经网络作为DQN网络
 #### 3.3.3 Replay_Memory.py
 缓存池的数据结构
 ##### 1) def push
 在容量未满的情况下存入四元组数据，如果容量超过上限，则覆盖之前的数据
 ##### 2) def sample
 以特定批次大小采用经验数据
 ##### 3) def __len__
 返回当前容量
 #### 3.3.4 Evolutionary_Algorithm.py
 ##### 1) def selection_tournament
 基于三个为一组的竞标赛规则选出适应值较大的DQN种群作为优异种群
 ##### 2) def list_argsort
 对种群适应值进行排序
 ##### 3) def regularize_weight
 使得weight的绝对值不超过mag
 ##### 4) def crossover_inplace
 对两个DQN种群的权重采取交叉操作对两个DQN种群的权重采取变异操作
 ##### 5) def mutate_inplace
 对两个DQN种群的权重采取变异操作
 ##### 6) def clone
 将源DQN种群的网络权重复制给目标DQN种群的网络权重
 ##### 7) def epoch
 基于适应值和竞标赛规则划分优异种群与非优异种群，并对优异种群进行交叉操作，对非优异种群进行变异操作
 ##### 8) def unsqueeze
 重构数组维数，工具函数
 #### 3.3.5 Influence_Propagation.py
 ##### 1) def constrctGraph
 基于邻接表构造邻接矩阵
 ##### 2) def nextGraph
 切换网络数据集，并读取相应的邻接表和降维向量
 ##### 3) def reset
 清空种子节点信息，重置降维向量
 ##### 4) def step
 基于DQN选出的种子节点，模拟影响传播过程，最终将两跳以内的影响分数作为reward
 ##### 5) def seeds2input
 将当前种子节点的信息作为新向量，拼接成网络数据集的新降维向量
 ##### 6) def getembedInfo 
 获取网络数据集的降维向量
 ##### 7) def getInfluence
 计算种子节点两跳以内的有效影响分数
 ##### 8) def getLocalInfluence
 计算种子节点两跳以内激活其他节点的影响分数
 ##### 9) def getOneHopInfluence
 计算种子节点两跳以内激活其他种子节点影响分数
 ##### 10) def getEpsilon
 计算种子节点一跳以内激活其他种子节点影响分数
 #### 3.3.6 Run_SEDRL_IM.py
 ##### 1) def add_experience
 基于CUDA技术将训练四元组数据存进缓存池
 ##### 2) def evaluate
 基于DQN输出的节点分数选择种子节点并计算出适应值，同时将选点过程中的四元组数据缓存
 ##### 3) def rl_to_evo
 将源DQN的网络权重复制给目标DQN的网络权重
 ##### 4) def evaluate_all
 评估演化后的DQN种群的适应值
 ##### 5) def train
 基于演化算法和DRL技术训练演化DQN种群
 ##### 6) def update_parameters
 基于DRL思想中n-step Q-learning技术利用缓存池中的特定批次大小的经验数据计算误差值，并基于随机梯度下降技术采用误差值梯度反向更新DQN的网络权重
 ##### 7) def get_offspring
 在演化得到的超过种群规模上限的DQN种群中，先对所有DQN种群进行排序，然后选出其中前50个DQN种群保留，并在剩下的种群中随机选择50个种群进行保留
 ##### 8) def showScore
 格式化输出适应值分数
 ##### 9) def getResultPath()
 设置程序结果和模型的存储路径
 ##### 10) def run
基于演化算法和强化学习思想求解特定种子规模下使得影响最大化的种子节点集
 
