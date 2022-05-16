import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# ����ص����ݽṹ #
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity # ��������
        self.memory = [] # ����List��Ϊ�����
        self.position = 0 # ���ڼ�¼������ݴ����λ��

    # ������δ��������´�����Ԫ�����ݣ���������������ޣ��򸲸�֮ǰ������ #
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # ���ض����δ�С���þ������� #
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # ���ص�ǰ���� #
    def __len__(self):
        return len(self.memory)
