#! /usr/bin/env python3

import sys
import numpy as np
from FMDQN_core.FMCQL import CQLAgent
from deserialization_core.deserialization import *
import torch

class TestCase:
    def __init__(self, file_path):
        self.file_path = file_path
        # 初始化DQN agent
        net_kwargs = {'hidden_sizes': [64, 64], 'learning_rate': 0.0005}
        self.agent = CQLAgent(
            net_kwargs=net_kwargs,
            gamma=0.99,
            epsilon=0.0,
            batch_size=400,
            observation_dim=(3, 51, 101),
            action_size=6,
            offline_RL_data_path=file_path,
            cql_alpha=1.0
        )
        # 加载训练好的模型参数
        self.agent.load_model_params()
        self.agent.load_replay_memory()

    def create_basic_lon_decision_input(self):
        """创建基础的LonDecisionInputType对象"""
        lon_input = LonDecisionInputType()

        # 设置ego基础信息
        lon_input.ego_info = EgoInfoType()
        lon_input.ego_info.pose = PoseType()
        lon_input.ego_info.pose.pos = PosType()
        lon_input.ego_info.pose.pos.x = 0.0  # 自车初始位置设为原点
        lon_input.ego_info.pose.pos.y = 0.0
        lon_input.ego_info.pose.theta = 0.0  # 自车朝向默认为x轴正方向
        lon_input.ego_info.width = 2.0
        lon_input.ego_info.length = 3.6

        # 设置障碍物集合
        lon_input.obj_set = ObjSetType()
        lon_input.obj_set.obj_info = []
        
        # 设置额外信息 - 需要与ego_info保持一致
        lon_input.extra_info = ExtraInfoType()
        lon_input.extra_info.ego_extra_info = EgoExtraInfoType()
        lon_input.extra_info.ego_extra_info.pose = PoseType()
        lon_input.extra_info.ego_extra_info.pose.pos = PosType()
        lon_input.extra_info.ego_extra_info.pose.pos.x = 0.0
        lon_input.extra_info.ego_extra_info.pose.pos.y = 0.0
        lon_input.extra_info.ego_extra_info.pose.theta = 0.0
        
        return lon_input
        
    def create_single_obj_case(self, ego_vel, obj_vel, obj_dtc, obj_intersection_dist, 
                              obj_scene=SCENE_FOLLOW, 
                              ego_x=0.0, ego_y=0.0, ego_theta=0.0,
                              obj_y_offset=0.0, obj_theta=0.0):
        """创建单个障碍物的测试用例
        Args:
            ego_vel: 自车速度
            obj_vel: 障碍物速度
            obj_dtc: 障碍物到交叉口的距离(用于特征图dtc图层)
            obj_intersection_dist: 障碍物相对自车的实际距离(用于确定障碍物位置)
            obj_scene: 场景类型
            ego_x: 自车x坐标
            ego_y: 自车y坐标
            ego_theta: 自车航向角(弧度)
            obj_y_offset: 障碍物相对自车的横向偏移
            obj_theta: 障碍物航向角(弧度)
        """
        lon_input = self.create_basic_lon_decision_input()

        # 更新自车信息
        lon_input.ego_info.vel = ego_vel
        lon_input.ego_info.pose.pos.x = ego_x
        lon_input.ego_info.pose.pos.y = ego_y
        lon_input.ego_info.pose.theta = ego_theta
        
        # 同步更新extra_info中的自车信息
        lon_input.extra_info.ego_extra_info.pose.pos.x = ego_x
        lon_input.extra_info.ego_extra_info.pose.pos.y = ego_y
        lon_input.extra_info.ego_extra_info.pose.theta = ego_theta

        # 创建障碍物信息
        obj = ObjInfoType()
        obj.id = 1
        obj.vel = obj_vel
        obj.dist_to_intersection = obj_dtc  # 用于特征图dtc图层
        obj.scene = obj_scene
        obj.width = 2.0
        obj.length = 4.0
        obj.pose = PoseType()
        obj.pose.pos = PosType()

        # 根据自车位置和实际距离设置障碍物位置
        obj.pose.pos.x = ego_x + obj_intersection_dist * np.cos(ego_theta)
        obj.pose.pos.y = ego_y + obj_intersection_dist * np.sin(ego_theta) + obj_y_offset
        obj.pose.theta = obj_theta

        lon_input.obj_set.obj_info.append(obj)

        return lon_input

    def test_velocity_cases(self):
        """测试不同车速场景"""
        print("\n=== Testing different velocity scenarios ===")

        ego_velocities = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        ego_velocities = [vel / 3.6 for vel in ego_velocities]
        obj_dtc = 0.0  # 障碍物到交叉口的距离
        obj_intersection_dist = 40.0  # 障碍物相对自车的实际距离
        obj_vel = 18.0  # 固定障碍物速度
        obj_vel = obj_vel / 3.6
        obj_y_offset = 0.0
        obj_theta = 0.0

        # 收集所有结果
        results = []
        for ego_vel in ego_velocities:
            lon_input = self.create_single_obj_case(
                ego_vel=ego_vel,
                obj_vel=obj_vel,
                obj_dtc=obj_dtc,  # 用于特征图
                obj_intersection_dist=obj_intersection_dist,  # 用于位置计算
                ego_theta=0.0,
                obj_y_offset=obj_y_offset,
                obj_theta=obj_theta
            )
            observation = self.agent.get_observation_from_lon_decision_input(lon_input)
            # self.agent.ogm.dump_ogm_graphs(observation)
            action = self.agent.decide(observation)
            results.append((ego_vel, action))

            print(f"\nEgo velocity: {ego_vel:.1f} m/s")
            print(f"Object velocity: {obj_vel:.1f} m/s")
            print(f"Obj Dist to intersection: {obj_dtc:.1f} m")
            print(f"EGO Dist to intersection: {obj_intersection_dist:.1f} m")
            print(f"Object y-offset: {obj_y_offset:.1f} m")
            print(f"Object heading: {np.rad2deg(obj_theta):.1f} deg")
            print(f"Predicted action: {self._action_to_string(action)}")

        # 打印结果表格
        for ego_vel, action in results:
            print(f"{ego_vel:^15.1f} | {self._action_to_string(action):^20}")


    def test_distance_cases(self):
        """测试不同距离场景"""
        print("\n=== Testing different distance scenarios ===")
        distances = [5.0, 10.0, 20.0, 30.0, 50.0]
        ego_vel = 10.0  # 固定自车速度
        obj_vel = 5.4   # 固定障碍物速度
        
        for dtc in distances:
            lon_input = self.create_single_obj_case(ego_vel, obj_vel, dtc)
            observation = self.agent.get_observation_from_lon_decision_input(lon_input)
            action = self.agent.decide(observation)
            
            print(f"\nEgo velocity: {ego_vel:.1f} m/s")
            print(f"Object velocity: {obj_vel:.1f} m/s")
            print(f"Distance to object: {dtc:.1f} m")
            print(f"Predicted action: {self._action_to_string(action)}")
    
    def test_scene_cases(self):
        """测试不同场景类型"""
        print("\n=== Testing different scene scenarios ===")
        scenes = [SCENE_FOLLOW, SCENE_MERGE, SCENE_CROSS]
        ego_vel = 10.0  # 固定自车速度
        obj_vel = 8.0   # 固定障碍物速度
        obj_dtc = 20.0  # 固定距离

        for scene in scenes:
            lon_input = self.create_single_obj_case(ego_vel, obj_vel, obj_dtc, scene)
            observation = self.agent.get_observation_from_lon_decision_input(lon_input)
            action = self.agent.decide(observation)

            print(f"\nScene type: {self._scene_to_string(scene)}")
            print(f"Ego velocity: {ego_vel:.1f} m/s")
            print(f"Object velocity: {obj_vel:.1f} m/s")
            print(f"Distance to object: {obj_dtc:.1f} m")
            print(f"Predicted action: {self._action_to_string(action)}")

    def _action_to_string(self, action):
        """将action数值转换为可读字符串"""
        action_map = {
            0: "EMERGENCY DECELERATE",
            1: "HARD DECELERATE",
            2: "SOFT DECELERATE",
            3: "MAINTAIN",
            4: "SOFT ACCELERATE",
            5: "HARD ACCELERATE"
        }
        return action_map.get(action, "UNKNOWN")
    
    def _scene_to_string(self, scene):
        """将场景类型转换为可读字符串"""
        scene_map = {
            SCENE_UNKNOWN: "UNKNOWN",
            SCENE_FOLLOW: "FOLLOW",
            SCENE_MERGE: "MERGE",
            SCENE_MERGE_LANE_CHANGE: "MERGE_LANE_CHANGE",
            SCENE_CROSS: "CROSS",
            SCENE_OTHER: "OTHER"
        }
        return scene_map.get(scene, "UNKNOWN")

    def dump_state_action_value(self):
        """记录每个memory中state对应的action value到文件中"""
        print("Dumping state action values...")

        # 打开文件
        output_file = "data/state_action_values_py.txt"
        with open(output_file, 'w') as f:
            # 遍历所有memory中的state
            print(f"Total frames: {len(self.agent.replay_memory)}")
            for frame in range(len(self.agent.replay_memory)):
                # 获取state
                state = self.agent.replay_memory.get_frame(frame).state

                # 确保state是正确的形状
                if not isinstance(state, np.ndarray):
                    state = np.array(state)
                if state.ndim == 3:
                    state = state[np.newaxis, ...]  # 添加batch维度

                # 转换为tensor
                state_tensor = torch.from_numpy(state).float().to(self.agent.device)

                # 获取所有动作的Q值
                with torch.no_grad():
                    q_values = self.agent.evaluate_net(state_tensor)
                    q_values = q_values.cpu().numpy().flatten()

                # 写入文件
                q_values_str = ", ".join([f"{v:.6f}" for v in q_values])
                f.write(f"{q_values_str}\n")

                if frame % 1000 == 0:
                    print(f"Processed frame: {frame}")

        print(f"State action values have been saved to {output_file}")

def main():
    file_path = sys.argv[1]
    test_case = TestCase(file_path)
    test_case.dump_state_action_value()

    # 运行所有测试用例
    #test_case.test_velocity_cases()
    # test_case.test_distance_cases()
    # test_case.test_scene_cases()

if __name__ == "__main__":
    main()