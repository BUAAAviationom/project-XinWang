import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance  # 距离计算包


class Env:
    def __init__(self, params_env, rate_seed, num_packet):
        self.params_env = params_env                     # 所有参数丢这里面
        self.seed = self.params_env['seed']
        self.t_slot = self.params_env['time_slot']
        self.num_node = self.params_env['num_node']
        self.uav_speed = self.params_env['uav_speed'] * self.params_env['time_slot']  # 无人机单位时间速度
        self.num_packet = num_packet
        '''初始化,注意这里packet队列里面有M个节点以及无人机的队列，没有AP的'''
        self.delay = 0.
        self.queue_len = [num_packet] * self.num_node  # 生成节点的初始packet队列
        self.queue_len.append(0)                    # 加上无人机的队列
        self.action_tuple = ()                      # 获得空的action元组
        '''node，AP位置和uav轨迹均用ndarry类型'''
        self._get_loc()                             # 得到uav和ap初始位置，得到node分布
        self.rate_seed = rate_seed
        '''求出routing策略'''
        self.graph_list = distance.cdist(self.node_loc, self.node_loc)
        self.graph = copy.deepcopy(self.graph_list)
        self.routing = self._get_routing()          # routing元组中每一位对应该node传的下一个节点，指向AP则=-1
        self.update_order = self._get_order()       # 获得更新节点的顺序,数组中为节点编号，从左往右更新
        self.node_rate = [0] * self.num_node        # 声明节点routing速率(单位时间内)列表
        self._get_routing_rate()                    # 获得地面按照路由方案时的传输速率,第x个元素表示node_x向下一个node的传输速率
        self.tra = np.array([[self.init_loc_x, self.init_loc_y]])  # 以无人机初始位置作为飞行轨迹起始点
        '''动作和状态空间维数'''
        self.space_action = 4 * self.num_node                      # 动作空间                                             # 3,已修改
        space_fly = 2                                              # x坐标和y坐标各占一个维度
        space_queue = self.num_node + 1                            # 各node和uav queue各占一个维度
        self.space_state = space_fly + space_queue                 # 状态空间数
        # 状态空间维数

    def reset(self):
        self.delay = 0.
        self.queue_len = [self.num_packet] * self.num_node  # 生成节点的初始packet队列
        self.queue_len.append(0)  # 加上无人机的队列
        # self.volume = [0.] * (self.num_node + 1)  # 初始化所有node当前传输量                                              # 2
        self.tra = np.array([[self.init_loc_x, self.init_loc_y]])  # 以无人机初始位置作为飞行轨迹起始点
        self.action_tuple = ()  # 获得空的action元组
        state = np.array([self.init_loc_x / self.params_env['area_l'],
                          self.init_loc_y / self.params_env['area_w']] +
                         np.divide(self.queue_len, 3 * self.num_packet).tolist())
        return state

    def get_state(self):
        state = np.array([self.tra[-1, 0] / self.params_env['area_l'],
                          self.tra[-1, 1] / self.params_env['area_w']]
                         + np.divide(self.queue_len, 3 * self.num_packet).tolist())
        return state

    def _get_loc(self):  # 得到节点和ap及无人机初始位置坐标张量，并把AP坐标加在最后
        np.random.seed(self.seed)
        self.init_loc_x = np.random.randint(0, self.params_env['area_l'])
        self.init_loc_y = np.random.randint(0, self.params_env['area_w'])
        self.AP_loc_x = np.random.randint(0, self.params_env['area_l'])
        self.AP_loc_y = np.random.randint(0, self.params_env['area_w'])
        self.AP_loc = np.array([[self.AP_loc_x, self.AP_loc_y]])  # 获得AP位置,由于无人机位置保存在self.tra里面,需要归零,这里不写
        # 0到l范围内生成num_relay个x坐标，0到w范围内生成num_relay个y坐标
        loc_x = np.array([np.random.randint(0, self.params_env['area_l'], self.num_node)])
        loc_y = np.array([np.random.randint(0, self.params_env['area_w'], self.num_node)])
        self.node_loc = np.concatenate((loc_x, loc_y), axis=0).T  # node坐标整合
        self.node_loc = np.concatenate((self.node_loc, self.AP_loc), axis=0)  # 加上AP坐标

    def min_edge(self, select, candidate):  # 求已经确定的顶点集合与未选顶点集合中的最小边，用在prim算法中
        min_weight = np.inf  # 记录最小边权重
        v, u = 0, 0  # 记录最小边
        for i in select:  # 循环扫描已选顶点与未选顶点，寻找最小边
            for j in candidate:  # 如果存在比当前的最小边权重还小的边，则记录
                if min_weight > self.graph[i][j]:
                    min_weight = self.graph[i][j]
                    v, u = i, j
        return v, u  # 返回记录的最小边的两个顶点

    def prim(self):  # prim算法求MST得到routing
        vertex_num = len(self.graph)  # 顶点个数
        select = [self.num_node]  # 存储已选顶点，初始化时可随机选择一个起点:AP
        candidate = list(range(vertex_num - 1))  # 存储未选顶点
        edge = []  # 存储每次搜索到的最小生成树的边
        for i in range(vertex_num - 1):  # 由于连接n个顶点需要n-1条边，故进行n-1次循环，以找到足够的边
            v, u = self.min_edge(select, candidate)  # 调用函数寻找当前最小边
            edge.append((v, u))  # 添加到最小生成树边的集合中
            select.append(u)  # v是select中的顶点，u为candidate中的顶点，故将u加入candidate，以代表已经选择该顶点
            candidate.remove(u)  # 同时将u从candidate中删除
        return edge

    def _get_routing(self):
        """由node位置给出routing方法"""
        edge = self.prim()  # 求出prim算法得到的MST
        routing = [None] * self.num_node  # 初始化路由策略
        for x in range(self.num_node):
            routing[edge[x][1]] = edge[x][0]  # 将MST转换为routing方法
        routing = [-1 if x == self.num_node else x for x in routing]   # 传给AP的点标记为-1
        routing = tuple(routing)

        print("routing = ", routing)
        """
        for x in range(self.num_node):  # 画出routing路线
            plt.plot([self.node_loc[x][0], self.node_loc[routing[x]][0]],
                     [self.node_loc[x][1], self.node_loc[routing[x]][1]], c='c')
        plt.scatter(self.node_loc.T[0], self.node_loc.T[1], c='k')  # 画node位置
        plt.scatter(self.node_loc[-1][0], self.node_loc[-1][1], c='r')  # 画AP位置，红点标记
        plt.scatter(self.init_loc_x, self.init_loc_y, s=100, c='r', marker='p')  # 画UAV初始位置，红点标记
        plt.show()
        """
        return routing

    def _get_order(self):
        """该函数给所有节点排序，高优先级的先计算，低优先级的后计算"""
        level = 1
        node_level = [None] * self.num_node             # 初始化节点等级
        begin_node = [-1]                               # 从AP作为起点，AP的等级视为0，最后计算
        while None in node_level:                       # 只要列表里面还有None就继续循环
            node_group = []
            for node in begin_node:                     # 该循环将所有同优先级计算的节点放入node_group内
                node_group += [i for i, x in enumerate(self.routing) if x == node]  # 用于得到计算某节点的上一个或多个节点
            for j in node_group:                        # 给node_group内节点排计算顺序
                node_level[j] = level
                level += 1
            begin_node = node_group.copy()              # 更新上一优先级的节点集合
        # 这里获得的node_level表示各节点的优先级，下面将其转换为node更新顺序数组“order”
        order = [None] * self.num_node                  # 初始化顺序,更新节点时按order列表从左往右更新即可，order中为节点编号
        for i in range(self.num_node):
            order[i] = node_level.index(self.num_node - i)
        return order

    def _get_routing_rate(self):  # 计算节点routing速率(用数据包数表示)
        np.random.seed(self.rate_seed)
        """这里假设节点发射功率恒定"""
        bandwidth = self.params_env['bandwidth_node']
        for x in range(self.num_node):
            dis = np.linalg.norm(self.node_loc[x] - self.node_loc[self.routing[x]])
            k = np.random.normal(scale=1) + np.random.normal(scale=1) * 1j
            h = np.sqrt(self.params_env['beta'] * pow(dis, -self.params_env['alpha1'])) * k
            sinr = self.params_env['power_node'] * pow(abs(h), 2) / (bandwidth * self.params_env['power_noise'])
            self.node_rate[x] = bandwidth * np.log2(1 + sinr) * self.t_slot
            self.node_rate[x] = math.floor(self.node_rate[x] / self.params_env['packet_size'])  # 向下取整

    def update_queue(self):
        for i in self.update_order:
            if self.queue_len[i] > 0:               # 队列中有数据包才进行处理，没有直接跳过该循环
                rate = self.node_rate[i]
                next_node = self.routing[i]         # i的下一个节点（注意区分向AP传）
                "接下来判断是否能传输完所有数据包，并更新节点i和下一个节点的数据包数"
                if self.queue_len[i] > rate:        # if数据包没传输完，还剩点
                    next_node_get = rate
                    self.queue_len[i] -= rate
                else:                               # if数据包传完了
                    next_node_get = self.queue_len[i]
                    self.queue_len[i] = 0
                "然后判断是否向AP传，向AP传就不用管了"
                if self.routing[i] != -1:           # 不向AP传才会让后一个节点数据包增加对应数量，避免了-1同时表示AP和无人机
                    self.queue_len[next_node] += next_node_get
        """这里更新完节点数据包了"""
        done = bool(np.sum(self.queue_len) == 0)
        self.delay += self.params_env['time_slot']
        return done

    def step(self, action):
        done = self.update_queue()
        '''State包括已归一化的无人机xy坐标和各node及无人机的数据包队列长度'''
        next_state = np.array([self.tra[-1, 0]/self.params_env['area_l'],
                               self.tra[-1, 1]/self.params_env['area_w']] +
                              np.divide(self.queue_len, 3*self.num_packet).tolist())
        return 0, next_state, done
