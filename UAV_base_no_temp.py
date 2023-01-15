import os
import numpy as np
from torch.optim import Adam
from memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from UAV_env import Env


params_env = {
    'seed': 21,
    'time_slot': 0.5,
    'area_l': 200,
    'area_w': 200,
    'H': 100,
    'uav_speed': 20,

    'num_node': 20,
    'packet_size': 1000,

    'bandwidth_node': 150_000,  # 150kHz
    'bandwidth_uav': 150_000,  # 100MHz
    'power_node': 10 ** (10 / 10) / 1000,  # node功率，20dBm
    'power_uav': 10 ** (10 / 10) / 1000,  # uav功率，20dBm
    'power_noise': 10 ** (-164 / 10) / 1000,  # 噪声功率170dBm，即~W/Hz

    'beta': 10 ** (-60 / 10),
    'alpha1': 3.5,
    'alpha2': 2.2,
    'antenna_gain': 20,
}


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class QNetwork(BaseNetwork):

    def __init__(self, num_states, num_actions):
        super().__init__()
        self.a_head = nn.Sequential(nn.Linear(num_states, 256), nn.ReLU(inplace=True),      # x-128
                                    nn.Linear(256, 64), nn.ReLU(inplace=True),            # 128-512
                                    nn.Linear(64, num_actions))                            # 512-y

    def forward(self, states):
        a = self.a_head(states)
        return a


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_states, num_actions):
        super(). __init__()
        self.Q1 = QNetwork(num_states, num_actions)
        self.Q2 = QNetwork(num_states, num_actions)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2

    def my_load_state_dict(self, path_model):
        self.load_state_dict(torch.load(path_model))
        self.eval()


class CateoricalPolicy(BaseNetwork):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.num_states = num_states
        self.head = nn.Sequential(nn.Linear(num_states, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, 64), nn.ReLU(inplace=True),
                                  nn.Linear(64, num_actions))

    def act(self, states):                                  # 最终策略选择确定性策略（选择最优action）
        states = states.view(1, -1)
        action_logits = self.head(states)
        greedy_actions = torch.argmax(action_logits, dim=1)
        return greedy_actions

    """对policy网络输入state，输出按softmax归一化后抽样的action，各action的softmax归一化值和归一化log值"""
    def sample(self, states):
        states = states.view(-1, self.num_states)           # 对输入的state转成num_states列的形式
        action_probs = F.softmax(self.head(states), dim=1)  # 给policy网络输入state，输出各action 的softmax值,行和为一
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)          # 按action归一化概率抽样，转为1列

        # 避免数值不稳定？
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return actions, action_probs, log_action_probs


class BaseAgent:
    def __init__(self, log_dir, num_steps=100000, batch_size=128, lr=0.00001, memory_size=20000,
                 gamma=0.99, multi_step=1, target_entropy_ratio=0.98, start_steps=20000, alpha=0.005, update_interval=4,
                 target_update_interval=8000, use_per=False, num_eval_episodes=1, max_episode_steps=27000,
                 eval_interval=1000, cuda=True, seed=0, mode='sac', num_packet=800):
        torch.manual_seed(seed)   # 为CPU设置种子生成随机数
        np.random.seed(seed)
        self.env = Env(params_env, seed, num_packet)
        self.test_env = Env(params_env, seed, num_packet)
        self.device = torch.device("cuda:2" if cuda and torch.cuda.is_available() else "cpu")

        # LazyMemory efficiently stores FrameStacked states.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(capacity=memory_size, state_shape=(self.env.space_state,),
                                                         device=self.device, gamma=gamma, multi_step=multi_step,
                                                         beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(capacity=memory_size, state_shape=(self.env.space_state,),
                                              device=self.device, gamma=gamma, multi_step=multi_step)

        # save data in profile
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_episodes = num_eval_episodes
        self.max_episode_steps = max_episode_steps
        self.eval_interval = eval_interval
        self.memory_counter = 0

        self.mode = mode
        self.seed = str(seed)
        self.num_packet = str(num_packet)
        self.train_steps = str(num_steps)
        """from sacd"""
        # 设置网络
        self.policy = CateoricalPolicy(self.env.space_state, self.env.space_action).to(self.device)
        self.online_critic = TwinnedQNetwork(self.env.space_state, self.env.space_action).to(device=self.device)
        self.target_critic = TwinnedQNetwork(self.env.space_state, self.env.space_action).to(device=self.device).eval()

        self.target_critic.load_state_dict(self.online_critic.state_dict())  # 参数硬复制
        for param in self.target_critic.parameters():                        # 阻止目标神经网络梯度下降
            param.requires_grad = False
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)
        'Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio)'
        self.target_entropy = -np.log(1.0 / self.env.space_action) * target_entropy_ratio
        self.alpha = alpha

    def run(self):
        while self.steps <= self.num_steps:
            self.episodes += 1
            episode_return = 0
            done = False
            self.env.reset()
            while not done:     # in an episode
                state = self.env.get_state()
                if self.start_steps > self.steps:
                    action = np.random.randint(0, self.env.space_action-1)    # 前几个action随机取
                else:
                    action, _, _ = self.explore(state)                             # 后面的按概率采样

                reward, next_state, done = self.env.step(action)
                self.memory.append(state, action, reward, next_state, done)  # 存数据
                self.steps += 1
                episode_return += reward

                # 每4步learn 一次,更新神经网络参数
                if self.steps % self.update_interval == 0 and self.steps > self.start_steps:
                    self.learn()
                # 每8000步，更新target网络
                if self.steps % self.target_update_interval == 0 and self.steps >= self.start_steps:
                    self.target_critic.load_state_dict(self.online_critic.state_dict())
                # # 每1000步，测试一下模型
                if self.steps % self.eval_interval == 0 and self.steps >= self.start_steps:
                    self.evaluate()
                    self.save_models(os.path.join(self.model_dir, 'final'))
            'end an episode'
            """
            if self.steps > self.start_steps:
                self.file_write('SAC_noTemp_' + str(self.alpha) + 'alpha_' + self.train_steps + 'steps_seed' + self.seed + '_entropy.txt', ave_entropy)
                self.file_write('SAC_noTemp_'+str(self.alpha)+'alpha_seed'+self.seed+'_return.txt', episode_return)
            """
    def explore(self, state):
        state = torch.Tensor(state[None, ...]).to(self.device).float()
        with torch.no_grad():
            action, action_probs, log_action_probs = self.policy.sample(state)
        return action.item(), action_probs, log_action_probs

    def exploit(self, state):
        state = torch.Tensor(state[None, ...]).to(self.device).float()
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()

    """类似于DQN，给online_critic网络输入state，action，输出对应的Q_eval值"""
    def calc_current_q(self, states, actions, rewards, next_states, dones, states_cache):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.view(self.batch_size, -1)                 # 输入state，输出128*action个Q值
        curr_q2 = curr_q2.view(self.batch_size, -1)                 # 同上
        curr_q1 = curr_q1.gather(1, actions.long())     # 索引相应action的Q值
        curr_q2 = curr_q2.gather(1, actions.long())     # 同上
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones, states_cache):
        with torch.no_grad():
            """对policy网络输入next_state，输出按softmax归一化后抽样的action，各action的softmax归一化值和归一化log值"""
            actions, action_probs_, log_action_probs_ = self.policy.sample(next_states)
            action_probs_ = action_probs_.view(self.batch_size, -1)
            log_action_probs_ = log_action_probs_.view(self.batch_size, -1)
            """给target_critic网络输入next_state，获得输出的各action的Q值"""
            next_q1, next_q2 = self.target_critic(next_states)
            next_q1 = next_q1.view(self.batch_size, -1)
            next_q2 = next_q2.view(self.batch_size, -1)
            min_q_next = torch.min(next_q1, next_q2)
            if self.mode == 'drsac':
                C = 1
                eta = 0.5
                _, action_probs, log_action_probs = self.policy.sample(states)
                curr_q1, curr_q2 = self.online_critic(states)
                curr_q1 = curr_q1.view(self.batch_size, -1)
                curr_q2 = curr_q2.view(self.batch_size, -1)
                states_array = states_cache  # 什么意思
                if self.use_per:
                    N_s = self.memory._get_state_count(states_array)  # 状态访问计数
                else:
                    N_s = 10000
                adv_eps = C / N_s ** eta  # 引入误差
                min_q_current = torch.min(curr_q1, curr_q2)
                # q的均值
                q_mean_state = torch.sum(min_q_current * action_probs, dim=1)  # online_critic网络的Q_eval*policy网络概率
                q_mean_state_ = torch.sum(min_q_next * action_probs_, dim=1)  # target_critic网络的Q_next*policy网络概率
                q_mean_state = q_mean_state.unsqueeze(1)  # 增加一维
                q_mean_state_ = q_mean_state_.unsqueeze(1)
                q_std = torch.sqrt(torch.sum(action_probs * ((min_q_current - q_mean_state) ** 2), dim=1)
                                   ).view(self.batch_size, 1)  # q的方差
                q_std_targ = torch.sqrt(torch.sum(action_probs_*((min_q_next - q_mean_state_) ** 2), dim=1)
                                        ).view(self.batch_size, 1)  # q_next的方差
                adv_reward_cor = (torch.tensor(np.sqrt(adv_eps / 2.0)).reshape((self.batch_size, 1))
                                  ).to(q_std.device) * (self.gamma_n * q_std_targ - q_std)
            elif self.mode == 'sac':
                adv_reward_cor = 0
            else:
                raise SystemExit('mode error')
            next_q = (action_probs_ * (min_q_next - self.alpha * log_action_probs_)).sum(dim=1, keepdim=True)
            a = rewards + (1.0 - dones) * self.gamma_n * next_q + adv_reward_cor
        return a

    def calc_critic_loss(self, batch, weights):  # weights的作用
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)
        errors = torch.abs(curr_q1.detach() - target_q) # TD errors for updating priority weights
        # We log means of Q to monitor training
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()                        # ？？？？
        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones, states_array = batch
        _, actions_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            q1, q2 = self.online_critic(states)
            q1 = q1.view(self.batch_size, -1)
            q2 = q2.view(self.batch_size, -1)
        # 计算熵
        entropies = -torch.sum(actions_probs * log_action_probs, dim=1, keepdim=True)
        q = torch.sum(torch.min(q1, q2) * actions_probs, dim=1, keepdim=True)
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()
        return policy_loss, entropies.detach()

    def learn(self):
        self.learning_steps += 1
        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            weights = 1

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)

        self.update_params(self.q1_optim, q1_loss)
        self.update_params(self.q2_optim, q2_loss)
        self.update_params(self.policy_optim, policy_loss)

        if self.use_per:
            self.memory.update_priority(errors)

        if self.learning_steps % 1000 == 0:
            print('------------------step:', self.steps, 'q1_loss:', q1_loss.item(), 'policy_loss:', policy_loss.item())

    def file_write(self, filename, txt):
        fileobject = open(filename, 'a')
        fileobject.write(str(txt))
        fileobject.write('\n')
        fileobject.close()

    def evaluate(self):
        num_episodes = 0                                    # evaluate x episodes in this function
        total_return = 0.0                                  # sum return of x episodes
        total_delay = 0.0
        entropy_list = []
        while num_episodes < self.num_eval_episodes:
            # self.env.reset()
            # state = self.env.get_state()
            self.test_env.reset()
            state = self.test_env.get_state()
            episode_return = 0.0                            # return in 1 of x episodes
            action_list = []                                # just print
            done = False
            while not done:
                # action = self.exploit(state)   # action选最优
                action, action_probs, log_action_probs = self.explore(state)     # sui ji
                entropy = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True).item()
                # reward, next_state, done = self.env.step(action)
                reward, next_state, done = self.test_env.step(action)
                entropy_list.append(entropy)
                action_list.append(action)
                episode_return += reward
                state = next_state
            'end an episode'
            # print(action_list)  # print(uav_trajectory)
            num_episodes += 1
            total_return += episode_return
            total_delay += self.test_env.delay
            # print('return:', episode_return)
        ave_episode_return = total_return / self.num_eval_episodes
        ave_delay = total_delay / self.num_eval_episodes
        ave_entropy = np.mean(entropy_list)
        print('return:', ave_episode_return, 'time:', ave_delay, 'entropy:', ave_entropy)

        'file write'
        self.file_write('SAC_noTemp_' + str(self.alpha) + 'alpha_' + self.train_steps + 'steps_seed' + self.seed + '_evatime.txt', ave_delay)
        self.file_write('SAC_noTemp_' + str(self.alpha) + 'alpha_' + self.train_steps + 'steps_seed' + self.seed + '_entropy.txt', ave_entropy)
        if total_return > self.best_eval_score:
            self.best_eval_score = total_return
            self.save_models(os.path.join(self.model_dir, 'best'))

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))

    def update_params(self, optim, loss, retain_graph=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optim.step()


def main(alpha, seed, num_packet, num_steps):
    # num_steps = 10 * 20000
    batch_size = 128
    lr = 0.0002
    gamma = 0.99
    multi_step = 1
    start_steps = 20000
    update_interval = 4
    target_update_interval = 10 * 200
    args = {'num_steps': num_steps, 'batch_size': batch_size, 'lr': lr, 'memory_size': start_steps, 'gamma': gamma,
            'multi_step': multi_step, 'target_entropy_ratio': 0.1, 'start_steps': start_steps, 'alpha': alpha,
            'update_interval': update_interval, 'target_update_interval':  target_update_interval, 'use_per': True,
            'num_eval_episodes': 100, 'max_episode_steps': 10, 'eval_interval': 1000, 'cuda': True, 'seed': seed,
            'num_packet': num_packet}
    log_dir = os.path.join('logs', 'GammaRay', f'h')  # 进入文件夹：logs\GammaRay\h，加f原本用于识别打括号内的{name=sacd}
    agent = BaseAgent(log_dir=log_dir, **args)  # 调用base
    agent.run()


"""seed放回参数堆里了"""
if __name__ == '__main__':
    main(0.0025, 0, 800, 200000)
    main(0.0025, 1, 800, 200000)
    main(0.0025, 2, 800, 200000)
    main(0.0025, 3, 800, 200000)
    main(0.0025, 4, 800, 200000)
    """
    main(0.002, 0, 800, 200000)
    main(0.002, 1, 800, 200000)
    main(0.002, 2, 800, 200000)
    main(0.002, 3, 800, 200000)
    main(0.002, 4, 800, 200000)

    main(0.003, 0, 800, 200000)
    main(0.003, 1, 800, 200000)
    main(0.003, 2, 800, 200000)
    main(0.003, 3, 800, 200000)
    main(0.003, 4, 800, 200000)

    main(0.01, 0, 800, 200000)
    main(0.01, 1, 800, 200000)
    main(0.01, 2, 800, 200000)
    main(0.01, 3, 800, 200000)
    main(0.01, 4, 800, 200000)

    main(0.001, 0, 800, 200000)
    main(0.001, 1, 800, 200000)
    main(0.001, 2, 800, 200000)
    main(0.001, 3, 800, 200000)
    main(0.001, 4, 800, 200000)

    main(0.0022, 0, 800, 200000)
    main(0.0022, 1, 800, 200000)
    main(0.0022, 2, 800, 200000)
    main(0.0022, 3, 800, 200000)
    main(0.0022, 4, 800, 200000)
    """
