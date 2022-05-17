# SEDRL-IM
## 代码说明
Code of Influence Maximization Research in Signed Networks by Using Evolutionary Deep Reinforcement Learning
## python 环境
1）环境配置文件：environment.yml <br>
2）重要环境 <br>
  python                    3.7.0 <br>
  keras-applications        1.0.8 <br>
  keras-preprocessing       1.1.2 <br>
  tensorflow                1.14.0 <br>
  torch                     1.8.1+cu111 <br>
  networkx                  2.5.1 <br>
 ## 代码说明
 ### 代码文件总览
 1）Mod_Utils.py <br>
 2）DQN_Model.py <br>
 3）Replay_Memory.py <br>
 4）Evolutionary_Algorithm.py <br>
 5）Influence_Propagation.py <br>
 6）Run_SEDRL_IM.py <br>
 ### 函数功能
 #### Mod_Utils.py
 矩阵处理函数
 #### DQN_Model.py
 DQN网络的数据结构，采用两层神经网络作为DQN网络
 #### Replay_Memory.py
 缓存池的数据结构
 ##### def push
 在容量未满的情况下存入四元组数据，如果容量超过上限，则覆盖之前的数据
 ##### def sample
 以特定批次大小采用经验数据
 ##### def __len__
 返回当前容量
 
 
  
