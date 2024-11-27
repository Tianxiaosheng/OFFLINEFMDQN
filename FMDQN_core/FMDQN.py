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
from scipy.signal import savgol_filter
import FMDQN_core.ogm as ogm
import deserialization_core.deserialization
import time


EVA_PATH = "data/eva_model_net.pth"
TARGET_PATH = "data/target_model_net.pth"

# 访问scene枚举值
SCENE_UNKNOWN = 0
SCENE_FOLLOW = 1
SCENE_MERGE = 2
SCENE_MERGE_LANE_CHANGE = 3
SCENE_CROSS = 4
SCENE_OTHER = 5

LON_DECISION_MAX_SAFE_T = 10000

def normalize_neg(x):
    return math.exp(-x) / (1 + math.exp(-x)) * 2

def normalize_pos(x):
    return (1 - math.exp(-math.pow(x / 2, 3))) /\
            (1 + math.exp(-math.pow(x / 2, 3)))

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

    def print_frame(self, frame_index):
        if frame_index < 0 or frame_index >= len(self.memory):
            print("帧索引超出范围。")
        else:
            transition = list(self.memory)[frame_index]
            print(f"帧 {frame_index}:=========================")
            print(f"动作: {transition.action}")
            print(f"奖励: {transition.reward}")
            print(f"完成: {transition.done}")

    def get_frame(self, frame_index):
        if frame_index < 0 or frame_index >= len(self.memory):
            print("帧索引超出范围。")
        else:
            return list(self.memory)[frame_index]

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
        # print("conv2_height{}, conv2_width{}, pool2_output_size{}, hidden1_size{}, hidden2_size{}, output_size{}".format(
        #         conv2_height, conv2_width, pool2_output_size, hidden1_size, hidden2_size, output_size))

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
class DQNAgent:
    def __init__(self, net_kwargs={}, gamma=0.9, epsilon=0.05,\
                 batch_size=64, observation_dim=(4, 67, 133), action_size=3,
                 offline_RL_data_path=None):
        # action_size == 3,  0: -1m/s, 1: 0, 2: +1m/s
        super().__init__()
        self.offline_RL_data_path = offline_RL_data_path
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
        self.deserialization = deserialization_core.deserialization.Deserialization(offline_RL_data_path)

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

        self.replay_memory = DQNReplayer(capacity=40000)

    def load_replay_memory(self):
        self.deserialization.get_lon_decision_inputs_by_deserialization()
        self.fill_replay_memory(self.deserialization.get_lon_decision_inputs())

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
 
        if self.count % 100 == 0:
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

    def get_observation_from_lon_decision_input(self, lon_decision_input):
        ego_pos = self.deserialization.get_ego_position_from_lon_decision_input(lon_decision_input)
        ego_pos = [ego_pos.x, ego_pos.y]
        ego_heading = self.deserialization.get_ego_heading_from_lon_decision_input(lon_decision_input)

        width = self.deserialization.get_ego_width_from_lon_decision_input(lon_decision_input)
        length = self.deserialization.get_ego_length_from_lon_decision_input(lon_decision_input)
        ego_vel = self.deserialization.get_ego_vel_from_lon_decision_input(lon_decision_input)
        ego_bounding_box = [length, width]

        self.ogm.preprocess_occupancy_grid(ego_pos, ego_heading)
        self.ogm.update_occupancy_grid(ego_pos, ego_heading, ego_vel, \
                                       ego_bounding_box, 0.0)
        bird_eye_valid_dist = self.ogm.width * self.ogm.delta_x / 2.0

        for obj_idx, obj_info in enumerate(lon_decision_input.obj_set.obj_info):
            dist = self.deserialization.get_obj_dtc_from_obj_info(obj_info)
            if (dist >= 0) and (dist < bird_eye_valid_dist):
                obj_pos = self.deserialization.get_obj_position_from_obj_info(obj_info)
                obj_pos = [obj_pos.x, obj_pos.y]
                obj_vel = self.deserialization.get_obj_vel_from_obj_info(obj_info)
                obj_bounding_box = [self.deserialization.get_obj_length_from_obj_info(obj_info), \
                                    self.deserialization.get_obj_width_from_obj_info(obj_info)]
                obj_heading = self.deserialization.get_obj_heading_from_obj_info(obj_info)
                if obj_heading < 0.0:
                    obj_heading+=math.pi * 2.0

                self.ogm.update_occupancy_grid(obj_pos, obj_heading, obj_vel, \
                                                obj_bounding_box, dist)

        return self.ogm.grid

    def get_action_from_lon_decision_inputs(self, lon_decision_inputs, frame):
        if (frame < len(lon_decision_inputs)-1):
            ego_info = self.deserialization.\
                get_ego_info_from_lon_decision_input(lon_decision_inputs[frame+1])
            if (ego_info.prev_cmd_acc >0.3):
                return 2
            elif (ego_info.prev_cmd_acc < -0.8):
                return 0
            else:
                return 1
        else:
            return 1

    def calc_safe_time_of_cross_obj(self, obj_info, ego_info):
        obj_time_to_cli =\
            self.deserialization.get_obj_time_to_cli_from_obj_info(obj_info)
        obj_time_to_leave_cli = \
            self.deserialization.get_obj_time_to_leave_cli_from_obj_info(obj_info)
        ego_time_to_cli = \
            self.deserialization.get_ego_time_to_cli(obj_info, ego_info)
        ego_time_to_leave_cli = \
            self.deserialization.get_ego_time_to_leave_cli(obj_info, ego_info)

        if (ego_time_to_cli == LON_DECISION_MAX_SAFE_T or\
                obj_time_to_cli == LON_DECISION_MAX_SAFE_T):
            safe_time = LON_DECISION_MAX_SAFE_T
        elif (obj_time_to_cli > ego_time_to_leave_cli or\
                ego_time_to_cli > obj_time_to_leave_cli):
            delta_time1 = abs(ego_time_to_cli - obj_time_to_leave_cli)
            delta_time2 = abs(ego_time_to_leave_cli - obj_time_to_cli)
            safe_time = min(delta_time1, delta_time2)
        else:
            safe_time = 0.0

        return safe_time

    def calc_safe_time_of_merge_obj(self, obj_info, ego_info):
        obj_time_to_cli =\
            self.deserialization.get_obj_time_to_cli_from_obj_info(obj_info)
        obj_time_to_leave_cli = \
            self.deserialization.get_obj_time_to_leave_cli_from_obj_info(obj_info)
        ego_time_to_cli = \
            self.deserialization.get_ego_time_to_cli(obj_info, ego_info)
        ego_time_to_leave_cli = \
            self.deserialization.get_ego_time_to_leave_cli(obj_info, ego_info)

        if (ego_time_to_cli == LON_DECISION_MAX_SAFE_T or\
                obj_time_to_cli == LON_DECISION_MAX_SAFE_T):
            safe_time = LON_DECISION_MAX_SAFE_T
        elif (obj_time_to_cli > ego_time_to_leave_cli or\
                ego_time_to_cli > obj_time_to_leave_cli):
            delta_time1 = abs(ego_time_to_cli - obj_time_to_leave_cli)
            delta_time2 = abs(ego_time_to_leave_cli - obj_time_to_cli)
            safe_time = min(delta_time1, delta_time2)
        else:
            safe_time = 0.0

        return safe_time

    def update_payoff_of_safety_bak(self, lon_decision_input):
        k_c = 2000
        min_safe_time = LON_DECISION_MAX_SAFE_T

        ego_info = self.deserialization.get_ego_info_from_lon_decision_input(lon_decision_input)
        for obj_info in lon_decision_input.obj_set.obj_info:
            if (obj_info.scene == SCENE_CROSS):
                safe_time = self.calc_safe_time_of_cross_obj(obj_info, ego_info)
            elif (obj_info.scene == SCENE_MERGE or\
                    obj_info.scene == SCENE_MERGE_LANE_CHANGE):
                safe_time = self.calc_safe_time_of_merge_obj(obj_info, ego_info)
            else:
                safe_time = self.deserialization.get_ego_time_to_cli(obj_info, ego_info)

            min_safe_time = min(min_safe_time, safe_time)

        normalized_safe_time = -k_c * normalize_neg(min_safe_time)

        return normalized_safe_time

    def update_payoff_of_safety(self, lon_decision_input):
        k_c = 2000
        min_safe_time = LON_DECISION_MAX_SAFE_T
        mass = 1.0
        max_risk_value = 0.0
        risk_value = 0.0

        ego_info = self.deserialization.get_ego_info_from_lon_decision_input(lon_decision_input)
        for obj_info in lon_decision_input.obj_set.obj_info:
            if (self.deserialization.get_obj_dtc_from_obj_info(obj_info) < 0.5):
                safe_time = self.deserialization.get_ego_time_to_cli(obj_info, ego_info)
                obj_vel = self.deserialization.get_obj_vel_from_obj_info(obj_info)
                ego_vel = self.deserialization.get_ego_vel_from_ego_info(ego_info)
                relative_vel = ego_vel - obj_vel
                dist_to_obj = self.deserialization.get_obj_intersection_start_s_from_obj_info(obj_info)
                if (relative_vel <= 0.0):
                    risk_value = 0.0
                elif (dist_to_obj <= 0.0):
                    risk_value = 5.0
                else:
                    risk_value = mass * pow(relative_vel, 3) / (2 * dist_to_obj)
                    risk_value = min(risk_value, 5.0)
                max_risk_value = max(max_risk_value, risk_value)

        normalized_risk_value = -k_c * normalize_pos(max_risk_value)
        return normalized_risk_value

    def update_payoff_of_comfort(self, lon_decision_input):
        k_a = 100
        ego_acc = self.deserialization.get_ego_acc_from_lon_decision_input\
                (lon_decision_input)
        ego_vel = self.deserialization.get_ego_vel_from_lon_decision_input\
                (lon_decision_input)
        if (ego_acc < 0.0 and ego_vel > 0.0):
            normalized_ego_acc = normalize_pos(-ego_acc)
            payoff_of_comfort = -k_a * normalized_ego_acc
        else:
            payoff_of_comfort = 0.0

        return payoff_of_comfort

    def update_payoff_of_efficiency(self, lon_decision_input):
        k_e = 10

        ego_vel = self.deserialization.get_ego_vel_from_lon_decision_input\
                (lon_decision_input)
        ego_target_vel = self.deserialization.get_ego_max_vel_from_lon_decision_input\
                (lon_decision_input)
        if (ego_target_vel > 0.0):
            payoff_of_efficiency = -k_e * abs(ego_vel - ego_target_vel) / ego_target_vel
        else:
            payoff_of_efficiency = 0.0

        return payoff_of_efficiency

    def update_reward(self, lon_decision_input):
        payoff_of_safety = self.update_payoff_of_safety(lon_decision_input)
        payoff_of_comfort = self.update_payoff_of_comfort(lon_decision_input)
        payoff_of_efficiency = self.update_payoff_of_efficiency(lon_decision_input)
        reward = payoff_of_safety + payoff_of_comfort + payoff_of_efficiency

        return reward

    def fill_replay_memory(self, lon_decision_inputs):
        state = self.get_observation_from_lon_decision_input(lon_decision_inputs[0])
        for frame in range(1, len(lon_decision_inputs), 1):
            next_state = self.get_observation_from_lon_decision_input(lon_decision_inputs[frame])
            action = self.get_action_from_lon_decision_inputs(lon_decision_inputs, frame)
            reward = self.update_reward(lon_decision_inputs[frame-1])
            done = False
            self.replay_memory.push(state, action, reward, next_state, done)
            state = next_state

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

def play_qlearning(agent, train=False, render=False):
    if len(agent.replay_memory) < agent.batch_size:
        print("repalymemory is not enough, please collect more data")
        return

    # 计算训练的轮次
    print("replay_memory size: {}, batch_size: {}".format(len(agent.replay_memory), agent.batch_size))
    period = len(agent.replay_memory) // agent.batch_size

    for _ in range(period):
        if train:
            start_time = time.time()
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
                agent.replay_memory.sample(agent.batch_size)
            T_data = agent.replay_memory.Transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            agent.learn(T_data)
            end_time = time.time()
            print("period {} train time: {}".format(_, end_time - start_time))

def decide_action_of_frame(agent, frame):
    state = agent.replay_memory.get_frame(frame).state
    action = agent.decide(state)
    return action

def replay_from_memory(agent):
    totol_action = len(agent.replay_memory)
    accuracy_action = 0
    for frame in range(len(agent.replay_memory)):
        action = decide_action_of_frame(agent, frame)
        print("frame: {}, action: {}, agent->action: {}".format(frame, agent.replay_memory.get_frame(frame).action, action))
        if action == agent.replay_memory.get_frame(frame).action:
            accuracy_action += 1
    print("accuracy_action: {} / {} = {}".format(accuracy_action, totol_action, accuracy_action / totol_action))
