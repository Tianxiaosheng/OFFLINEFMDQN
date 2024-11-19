import numpy as np
np.random.seed(0)
import math
import pandas as pd
import torch  
import torch.nn as nn  
import torch.optim as optim
import random
from collections import deque, namedtuple       # 队列类型
import matplotlib.pyplot as plt

from smarts.core.agent import Agent
from planner_base import PlanerBase
import time
from scipy.signal import savgol_filter
import ogm

EVA_PATH = "./examples/lon_plan_agent/eva_model_net.pth"
TARGET_PATH = "./examples/lon_plan_agent/target_model_net.pth"

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    def sample(self, batch_size):
        batch_data = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch_data)
        return state, action, reward, next_state, done
 
    def push(self, *args):
        # *args: 把传进来的所有参数都打包起来生成元组形式
        # self.push(1, 2, 3, 4, 5)
        # args = (1, 2, 3, 4, 5)
        self.memory.append(self.Transition(*args))
 
    def __len__(self):
        return len(self.memory)

# 定义DQN网络结构
class FMDQNNet(nn.Module):
    def __init__(self, input_dim=(4, 67, 133), conv_param={'filter1_size': 6, 'filter2_size': 16,
                                                          'filter_width': 3, 'pad': 0, 'stride': 1},
                                hidden1_size=128, hidden2_size=32, output_size=3):
        super(FMDQNNet, self).__init__()
        self.input_dim = input_dim
        filter1_size = conv_param['filter1_size']
        filter2_size = conv_param['filter2_size']
        filter_width = conv_param['filter_width']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        intput_channel_size = input_dim[0]
        input_height = input_dim[1]
        input_width = input_dim[2]
        conv1_height = 1 + (input_height + 2*filter_pad - filter_width) / \
                filter_stride
        conv1_width = 1 + (input_width + 2*filter_pad - filter_width) / \
                filter_stride

        conv2_height = 1 + (int(conv1_height / 2) + 2*filter_pad - filter_width) / \
                filter_stride
        conv2_width = 1 + (int(conv1_width / 2) + 2*filter_pad - filter_width) / \
                filter_stride

        pool2_output_size = int(filter2_size * int(conv2_height / 2) *
                               int(conv2_width / 2))
        print("conv2_height{}, conv2_width{}, pool2_output_size{}, hidden1_size{}, hidden2_size{}, output_size{}".format(
                conv2_height, conv2_width, pool2_output_size, hidden1_size, hidden2_size, output_size))

        self.conv_block = nn.Sequential(
            nn.Conv2d(intput_channel_size, filter1_size, filter_width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=filter_pad),
            nn.Conv2d(filter1_size, filter2_size, filter_width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=filter_pad)
        )

        self.fc = nn.Sequential(
            nn.Linear(pool2_output_size, out_features=hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size)
        )

    def forward(self, x):
        x = self.conv_block(x)
        if x.dim() == len(self.input_dim):
            x = x.view(-1)
        else:
            x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# 定义DQNAgent类
class DQNAgent(Agent):
    def __init__(self, net_kwargs={}, gamma=0.9, epsilon=0.05,\
                 batch_size=64, observation_dim=(4, 67, 133), action_size=3):
        # action_size == 3,  0: -1m/s, 1: 0, 2: +1m/s
        super().__init__()
        self.planerbase = PlanerBase(100.0, 100.0)
        self.observation_dim = observation_dim
        self.action_n = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.count = 0
        delta_x = 1.0
        delta_y = 1.0
        self.render = False
        self.ogm = ogm.OccupancyGrid(self.observation_dim, delta_x=delta_x, delta_y=delta_y, render=self.render)

        # 初始化评估网络和目标网络  
        self.evaluate_net = FMDQNNet(input_dim=self.observation_dim,
                                     conv_param={'filter1_size': 6, 'filter2_size': 16,
                                                          'filter_width': 5, 'pad': 0, 'stride': 1},
                                     hidden1_size= 120, hidden2_size=84, output_size=self.action_n).to(self.device)
        self.target_net = FMDQNNet(input_dim=self.observation_dim,
                                     conv_param={'filter1_size': 6, 'filter2_size': 16,
                                                          'filter_width': 5, 'pad': 0, 'stride': 1},
                                     hidden1_size= 120, hidden2_size=84, output_size=self.action_n).to(self.device)

        # 初始化目标网络的权重与评估网络相同  
        self.update_target_net()

        # 定义优化器
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=net_kwargs.get('learning_rate', 0.001))

        # 定义损失函数  
        self.criterion = nn.MSELoss()  

    def update_target_net(self):  
        self.target_net.load_state_dict(self.evaluate_net.state_dict())

    def learn(self,transition_dict):
        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1) # 扩充维度
        rewards = np.expand_dims(transition_dict.reward, axis=-1) # 扩充维度
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1) # 扩充维度

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        predict_q_values = self.evaluate_net(states).gather(1, actions)

        # 计算目标Q值
        with torch.no_grad():
            # max(1) 即 max(dim=1)在行向找最大值，这样的话shape(64, ), 所以再加一个view(-1, 1)扩增至(64, 1)
            max_next_q_values = self.target_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        l = self.criterion(predict_q_values, q_targets)

        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
 
        if transition_dict.done:
            # copy model parameters
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
 
        self.count += 1

    def decide(self, observation):
        # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)

        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        action = torch.argmax(self.evaluate_net(state)).item()

        return action

    def convert_observation(self, observation):
        ego_pos = observation["ego_vehicle_state"]["position"]
        ego_heading = observation["ego_vehicle_state"]["heading"] + math.pi / 2.0
        if ego_heading < 0.0:
            ego_heading += 2.0 * math.pi
        ego_bounding_box = observation["ego_vehicle_state"]["box"]
        ego_dist_to_cli = self.planerbase.calc_dist_to_cli_pt(ego_pos,\
                ego_heading)
        ego_vel = observation["ego_vehicle_state"]["speed"]

        self.ogm.preprocess_occupancy_grid(ego_pos[:2], ego_heading)
        self.ogm.update_occupancy_grid(ego_pos[:2], ego_heading, ego_vel, \
                                       ego_bounding_box[:2], 0.0)
        bird_eye_valid_dist = self.ogm.width * self.ogm.delta_x / 2.0

        objs =observation.get("neighborhood_vehicle_states", [])
        if objs and (ego_dist_to_cli < bird_eye_valid_dist and ego_dist_to_cli > 0.0):
            position = observation["neighborhood_vehicle_states"]["position"]
            heading = observation["neighborhood_vehicle_states"]["heading"]
            speed = observation["neighborhood_vehicle_states"]["speed"]
            id = observation["neighborhood_vehicle_states"]["id"]
            bounding_box = observation["neighborhood_vehicle_states"]["box"]

            for i in range(len(id)):
                if id[i]:
                    dist = self.planerbase.calc_dist_to_cli_pt(position[i], heading[i])
                    if (dist >= 0) and (dist < bird_eye_valid_dist):
                        obj_pos = position[i]
                        obj_vel = speed[i]
                        obj_bounding_box = bounding_box[i]
                        obj_heading = heading[i] + math.pi / 2.0
                        if obj_heading < 0.0:
                            obj_heading+=math.pi * 2.0

                        self.ogm.update_occupancy_grid(obj_pos[:2], obj_heading, obj_vel, \
                                                    obj_bounding_box[:2], dist)

                        if dist < (bird_eye_valid_dist) and self.render:
                            print('++++++++++++++++++obj_id{}+++++++++++++++++'.format(i))
                            self.ogm.dump_ogm_graph(0)
                            self.ogm.dump_ogm_graph(1)
                            self.ogm.dump_ogm_graph(2)
                            # self.ogm.dump_ogm_graph(3)

        return self.ogm.grid

    def normalize_state(self, state):
        ego_dist_to_cli = state[0]
        ego_vel = state[1]
        ego_acc = state[2]

        obj_dist_to_cli = state[3]
        obj_vel = state[4]
        obj_acc = state[5]

        ego_dist_to_cli_norm = ego_dist_to_cli / 200.0
        ego_vel_norm = max(ego_vel, 30) / 30.0
        ego_acc_norm = max(ego_acc, 10) / 10.0

        obj_dist_to_cli_norm = obj_dist_to_cli / 200.0
        obj_vel_norm = max(obj_vel, 30) / 30.0
        obj_acc_norm = max(obj_acc, 10) / 10.0

        return [ego_dist_to_cli_norm, ego_vel_norm, ego_acc_norm, obj_dist_to_cli_norm, obj_vel_norm, obj_acc_norm]

    def get_steps_from_observation(self, observation):
        return observation["steps_completed"]

    def get_ego_vel_from_observation(self, observation):
        return observation["ego_vehicle_state"]["speed"]

    def get_ego_pose_from_observation(self, observation):
        return observation["ego_vehicle_state"]["position"]

    def get_events_from_observation(self, observation):
        return observation["events"]

    def get_neighborhood_vehicle_states(self, observation):
        return observation["neighborhood_vehicle_states"]

    def calc_reward(self, observation):
        K_e = 10
        K_c = 2000

        truncated = observation["events"]['collisions']

        ego_vel= observation["ego_vehicle_state"]["speed"]
        ego_target_vel = 12.0

        R_e = -K_e * abs(ego_vel - ego_target_vel) / ego_target_vel

        delta_s = observation["ego_vehicle_state"]["speed"]

        if truncated:
            R_c = -K_c
        else:
            R_c = 0.0

        return R_e + R_c

    def act(self, state, ego_init_vel, **kwargs):
        action = self.decide(state)
        if action == 0:
            expt_vel = max(ego_init_vel - 1.0, 0.0)
        elif action == 1:
            expt_vel = ego_init_vel
        else:
            expt_vel = ego_init_vel + 1.0

        if self.render:
            print('init_vel:{}, action:{}, expt_vel:{}'.format(ego_init_vel, action-1, expt_vel))
        return [expt_vel, 0], action

    def plot_reward(self, episode_rewards, ego_vels, gamma, epsilon, batch_size, net_kwargs={}):
        fig = plt.figure(figsize=(10, 6))  # 设置图形大小

        # 添加第一个子图
        episode_rewards_filtered = savgol_filter(episode_rewards, 19, 1, mode='nearest')
        ax1 = fig.add_subplot(111)  # 使用 add_subplot 更简洁
        ax1.plot(episode_rewards, label='Episode_Reward', color='r', linestyle='--', linewidth=1)  # 设置线条宽度和标签
        ax1.plot(episode_rewards_filtered, label='Episode_Reward-filtered', color='r', linewidth=3)
        ax1.set_ylabel('Reward', color='r')  # 设置y轴标签和颜色
        ax1.tick_params(axis='y', labelcolor='r')  # 设置y轴刻度标签颜色

        # 添加第二个子图，共享x轴
        ax2 = ax1.twinx()
        ax2.plot(ego_vels, label='Ego Velocity', color='b', linewidth=1)  # 设置线条样式和宽度
        ax2.set_ylabel('Ego Velocity(m/s)', color='b')  # 设置y轴标签和颜色
        ax2.tick_params(axis='y', labelcolor='b')  # 设置y轴刻度标签颜色

        # 设置x轴标签
        ax1.set_xlabel('Episode')

        # 设置标题
        fig.suptitle('Params->hidden_sizes:{}, learning_rate:{}, gamma:{}, epsilon:{}, batch_size:{}'.
                format(net_kwargs.get("hidden_sizes"), net_kwargs.get("learning_rate"), gamma, epsilon, batch_size), fontsize=16)

        # 获取两个轴的 handles 和 labels
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        # 合并 handles 和 labels
        handles = handles1 + handles2
        labels = labels1 + labels2

        # 调整子图之间的间距
        fig.tight_layout()

        # 添加图例，并设置其位置
        fig.legend(handles, labels, loc='upper right')

        plt.show()

    def save_model_params(self):
        torch.save(self.evaluate_net.state_dict(), EVA_PATH)
        torch.save(self.target_net.state_dict(), TARGET_PATH)

    def load_model_params(self):
        self.evaluate_net.load_state_dict(torch.load(EVA_PATH))
        self.evaluate_net.eval()
        self.target_net.load_state_dict(torch.load(TARGET_PATH))
        self.target_net.eval()

def play_qlearning(env, agent, repalymemory, episode, train=False, render=False):
    episode_reward = 0
    ego_vel_sum = 0.0
    ego_vel_mean = 0.0
    observation, _ = env.reset()
    obj_info_prev = [200, 0.0, 0.0, ""]
    obj_info_prev = agent.planerbase.get_nearest_obj_info(observation, obj_info_prev)
    ego_vel_prev = -1.0
    state = agent.convert_observation(observation)
    ego_init_vel = agent.get_ego_vel_from_observation(observation)

    episode.record_scenario(env.unwrapped.scenario_log)
    if render:
        start_time = time.time()
    while True:
        action_converted, action = agent.act(state, ego_init_vel)
        next_observation, reward, terminated, truncated, info = env.step(action_converted)
        next_state = agent.convert_observation(next_observation)
        reward = agent.calc_reward(next_observation)

        # print("state:%{}, reward:{}".format(next_state, reward))

        repalymemory.push(state, action, reward, next_state, terminated)
        ego_init_vel = agent.get_ego_vel_from_observation(next_observation)
        episode_reward += reward
        ego_vel_sum += ego_init_vel

        # episode.record_step(observation, reward, terminated, truncated, info)

        if train:
            if len(repalymemory) > agent.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = repalymemory.sample(agent.batch_size)
                T_data = repalymemory.Transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                agent.learn(T_data)
        if terminated:
            break
        obj_info_prev = agent.planerbase.get_nearest_obj_info(next_observation, obj_info_prev)
        state = next_state

    ego_vel_mean = ego_vel_sum / agent.get_steps_from_observation(next_observation)
    if render:
        end_time = time.time()
        print('episode_reward:{}, ego_vel_mean:{}, count:{}, time:{}'
              .format(episode_reward, ego_vel_mean, agent.count, end_time-start_time))
    return episode_reward, ego_vel_mean
