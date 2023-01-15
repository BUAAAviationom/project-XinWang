# 这个是我写的
import numpy as np
from UAV_env import Env
import torch
import torch.nn as nn
import torch.nn.functional
import matplotlib.pyplot as plt
import copy
import random

# hyper-parameters超参数
BATCH_SIZE = 128            # 样本数
LR = 0.0002                 # 学习率
GAMMA = 0.99                # γ
MEMORY_CAPACITY = 20_000    # 记忆库容量
Q_NETWORK_ITERATION = 10000  # 目标网络更新频率, ori:1000

params_env = {
    'seed': 21,
    'time_slot': 0.5,
    'area_l': 200,
    'area_w': 200,
    'H': 100,
    'uav_speed': 20,

    'num_node': 20,
    'packet_size': 1000,

    'bandwidth_node': 150_000,                  # 150kHz
    'bandwidth_uav': 150_000,                   # 100MHz
    'power_node': 10 ** (10 / 10) / 1000,       # node功率，20dBm
    'power_uav': 10 ** (10 / 10) / 1000,        # uav功率，20dBm
    'power_noise': 10 ** (-164 / 10) / 1000,    # 噪声功率170dBm，即~W/Hz

    'beta': 10 ** (-60 / 10),
    'alpha1': 3.5,
    'alpha2': 2.2,
    'antenna_gain': 20,
}


class Net(nn.Module):
    """定义1个神经网络"""
    def __init__(self, NUM_ACTIONS, NUM_STATES):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 256)   # 设置第一个全连接层，状态数个神经元64个神经元
        self.fc1.weight.data.normal_(0, 0.1)    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 = nn.Linear(256, 64)           # 第二个全连接层，64到31个神经元
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(64, NUM_ACTIONS)   # 隐藏层到输出层，16个到动作个神经元
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):                       # 定义前向传递函数，x为状态
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)         # 与上一句一起连接fc1层
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)         # 与上一句一起连接fc2层
        action_value = self.out(x)              # 连接out层，获得输出值（action）
        return action_value                     # 返回动作值


class DQN():
    """定义DQN类"""
    def __init__(self, seed, NUM_ACTIONS, NUM_STATES):
        super(DQN, self).__init__()
        torch.manual_seed(seed)                                               # 为CPU设置种子生成随机数
        np.random.seed(seed)
        self.eval_net = Net(NUM_ACTIONS, NUM_STATES)                          # 利用Net创建评估网络和目标网络
        self.target_net = Net(NUM_ACTIONS, NUM_STATES)
        self.num_action = NUM_ACTIONS
        self.num_state = NUM_STATES

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))         # 记忆空间维度（记忆库容量，状态数*2+2）
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # 使用Adam优化器，输入为评估网络的参数和学习率，后面反向传播用到
        self.loss_func = nn.MSELoss()                                         # 使用均方损失函数
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2300], gamma=0.1)

    def choose_action(self, state, epsilon):                      # 对神经网络输入state，输出action（用ε-greedy）
        state = torch.unsqueeze(torch.FloatTensor(state), 0)      # get a 1D array
        if random.random() >= epsilon:                            # greedy policy
            action_value = self.eval_net.forward(state)           # 对评估神经网络，输入state，得到action（多个值）
            action = torch.max(action_value, 1)[1].data.numpy()   # 输出每一行最大值的索引(序号)（即对应的action），并转化为多维数组形式
            action = int(action[0])                               # 输出action的第一个数
        else:                                                     # random policy
            action = random.randint(0, self.num_action-1)
        return action

    def store_transition(self, state, action, reward, next_state):     # 记忆存储函数
        transition = np.hstack((state, [action, reward], next_state))  # 在水平方向上拼接数组
        index = self.memory_counter % MEMORY_CAPACITY                  # 获取transition要置入的行数，这里如果超过记忆库容量则从1开始重新覆盖之前数据
        self.memory[index, :] = transition                             # 置入transition
        self.memory_counter += 1                                       # 计数器+1，让transition下一次放到下一个位置

    def learn(self):
        # 记忆库满了就开始学习
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:           # 每Q_NETWORK_ITERATION步
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络参数赋给目标网络
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)    # 从记忆库中随机抽128个数，可能重复
        batch_memory = self.memory[sample_index, :]                     # 把抽到的数对应的transition存入batch记忆库里
        batch_state = torch.FloatTensor(batch_memory[:, :self.num_state])
        # 取出其中的state，转换成张量tensor形式后存入batch_state，PyTorch的tensor可以在GPU上运行，而numpy的ndarray只能在cpu上运行
        batch_action = torch.LongTensor(batch_memory[:, self.num_state:self.num_state+1].astype(int))
        # 取出其中的action，转换形式后存入batch_action
        batch_reward = torch.FloatTensor(batch_memory[:, self.num_state+1:self.num_state+2])  # 取出其中的reward
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.num_state:])           # 取出其中的next_state

        q_eval = self.eval_net(batch_state).gather(1, batch_action)             # 评估神经网络通过评估state估计出（最优）action的q值
        q_next = self.target_net(batch_next_state).detach()
        # 目标神经网络输入batch中下一状态估计出下一状态各action的q值.detach防止反向传播，
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # 更新目标网络Q值
        q_target = torch.where(batch_reward > 0, batch_reward, q_target)

        loss = self.loss_func(q_eval, q_target)                                 # 输入128个评估值和128个目标值，使用均方损失函数

        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()             # 反向传播, 计算参数更新值
        self.optimizer.step()       # 更新评估网络的所有参数
        return loss


def test(test_env, train_steps, dqn, seed):
    test_state = test_env.reset()  # 重置环境
    while True:  # 每个step
        test_action = dqn.choose_action(test_state, -1)
        test_reward, test_next_state, test_done = test_env.step(test_action)
        test_state = test_next_state
        # 跑完step开始learn
        if test_done:
            'file write'
            fileobject = open('DQN_' + str(train_steps) + 'steps_seed' + str(seed) + '_evatime.txt', 'a')
            fileobject.write(str(test_env.delay))
            fileobject.write('\n')
            fileobject.close()
            break


def main(seed, num_packet, train_steps):
    env = Env(params_env, seed, num_packet)
    test_env = Env(params_env, seed, num_packet)
    NUM_ACTIONS = env.space_action  # action个数
    NUM_STATES = env.space_state  # state个数
    dqn = DQN(seed, NUM_ACTIONS, NUM_STATES)
    # episodes = int(7.5 * num_packet)
    epsilon = 0.9
    step = 0
    print("Collecting Experience....")
    while step <= train_steps:           # 共运行固定数量个step
        "normal training"
        state = env.reset()  # 重置环境
        ep_reward = 0
        while True:  # 每个step
            step += 1
            epsilon -= 2 / (train_steps - 10)
            action = dqn.choose_action(state, epsilon)
            reward, next_state, done = env.step(action)
            ep_reward += reward  # episode reward 相加
            dqn.store_transition(state, action, reward, next_state)  # 存储样本
            state = next_state
            # 跑完step开始learn
            if dqn.memory_counter >= MEMORY_CAPACITY:  # 超过记忆库容量则开始学习
                if step % 1000 == 0:
                    test(test_env, train_steps, dqn, seed)        # every 30 episodes evaluate once
                loss = dqn.learn()
                if done:
                    # dqn.scheduler.step()
                    # test(test_env, train_steps, dqn, seed)  # every 1 episodes evaluate once
                    print("step: {:0>3d}, reward: {:.5f}, ε: {:.3f}, delay: {:.3f}， loss: {: .12f}"
                          .format(step, ep_reward, epsilon, env.delay, loss))
            if done:
                break


if __name__ == '__main__':
    main(0, 800, 40000)
    main(1, 800, 40000)
    main(2, 800, 40000)
    main(3, 800, 40000)
    main(4, 800, 40000)

    main(0, 800, 30000)
    main(1, 800, 30000)
    main(2, 800, 30000)
    main(3, 800, 30000)
    main(4, 800, 30000)

    main(0, 800, 35000)
    main(1, 800, 35000)
    main(2, 800, 35000)
    main(3, 800, 35000)
    main(4, 800, 35000)

    main(0, 800, 45000)
    main(1, 800, 45000)
    main(2, 800, 45000)
    main(3, 800, 45000)
    main(4, 800, 45000)



