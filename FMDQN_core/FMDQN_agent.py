import argparse
import logging
import random
import sys
import warnings
from pathlib import Path
from typing import Final
from planner_base import PlanerBase
from FMDQN import DQNAgent, DQNReplayer, play_qlearning
import numpy as np


logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

import gymnasium as gym

SMARTS_REPO_PATH = Path(__file__).parents[2].absolute()

sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.sstudio.scenario_construction import build_scenarios

AGENT_ID: Final[str] = "Agent"

# define the agent
class KeepLaneAgent(Agent):
    def act(self, obs, **kwargs):
        return 2

class SpeedPlanAgent(Agent):
    def __init__(self):
        super().__init__()
        self.planerbase = PlanerBase(100.0, 100.0)

    def convert_observation(self, observation):
        ego_pos = observation["ego_vehicle_state"]["position"]
        ego_heading = observation["ego_vehicle_state"]["heading"]
        ego_dist_to_cli = self.planerbase.calc_dist_to_cli_pt(ego_pos,\
                ego_heading)
        ego_vel = observation["ego_vehicle_state"]["speed"]
        obj_dist_to_cli, obj_vel = \
                self.planerbase.get_nearest_obj_info(observation)

        return [ego_vel, ego_dist_to_cli, obj_vel, obj_dist_to_cli]

    def act (self, obs, **kwargs):
        state = self.convert_observation(obs)
        print(state)

        return 12.0, 0

def main(scenarios, headless, num_episodes_training, num_episodes_testing, max_episode_steps=None):
    # This interface must match the action returned by the agent
    agent_interface = AgentInterface.from_type(
        AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps,
        neighborhood_vehicle_states=True
    )

    env_training = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=True,
    )
    env_training = SingleAgent(env_training)

    replaymemory = DQNReplayer(capacity=100000)
    episode_rewards = []
    ego_vels = []
    net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.0005}
    gamma=0.99
    epsilon=0.06
    batch_size=400
    agent = DQNAgent(net_kwargs, gamma, epsilon, batch_size,\
                     observation_dim=(3, 51, 101), action_size=3)
    train = False

    if train:
        for episode in episodes(n=num_episodes_training):

            observation, _ = env_training.reset()

            #episode.record_scenario(env.unwrapped.scenario_log)

            episode_reward, ego_vel = play_qlearning(env_training, agent, replaymemory, episode, train, render=True)
            episode_rewards.append(episode_reward)
            ego_vels.append(ego_vel)

        print('Training->平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                len(episode_rewards), np.mean(episode_rewards)))
        env_training.close()
        # save nn model
        if train:
            agent.save_model_params()

        agent.plot_reward(episode_rewards, ego_vels, gamma, epsilon, batch_size, net_kwargs)
    else:
        agent.load_model_params()

    # 测试
    agent.epsilon = 0.0
    env_testing = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=False,
    )
    env_testing = SingleAgent(env_testing)
    episode_rewards = []
    ego_vels = []
    for episode in episodes(n=num_episodes_testing):

        observation, _ = env_testing.reset()

        #episode.record_scenario(env.unwrapped.scenario_log)

        episode_reward, ego_vel = play_qlearning(env_testing, agent, replaymemory, episode, train=False, render=True)
        episode_rewards.append(episode_reward)
        ego_vels.append(ego_vel)

    print('Testing->平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
            len(episode_rewards), np.mean(episode_rewards)))
    env_testing.close()

    agent.plot_reward(episode_rewards, ego_vels, gamma, epsilon, batch_size, net_kwargs)

if __name__ == "__main__":
    parser = minimal_argument_parser(Path(__file__).stem)
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "loop"),
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "figure_eight"),
        ]

    build_scenarios(scenarios=args.scenarios)
    args.episodes = 100 # training episodes
    episodes_testing= 50 # testing episodes

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes_training=args.episodes,
        num_episodes_testing=episodes_testing,
        max_episode_steps=max(args.max_episode_steps, 120),
    )
