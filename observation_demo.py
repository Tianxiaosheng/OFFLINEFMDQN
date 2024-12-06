#! /usr/bin/env python3

import sys
import numpy as np
from pathlib import Path
from FMDQN_core.FMDQN import DQNAgent, DQNReplayer

# SMARTS_REPO_PATH = Path(__file__).parents[2].absolute()
# sys.path.insert(0, str(SMARTS_REPO_PATH))

def main():
    # DQN parameters
    file_path = sys.argv[1]
    net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.0005}
    gamma=0.99
    epsilon=0.00
    batch_size=400
    train = False
    # Initialize agent
    agent = DQNAgent(net_kwargs, gamma, epsilon, batch_size,\
                     observation_dim=(3, 51, 101), action_size=3,
                     offline_RL_data_path=file_path)

    # Load UOS's deserialization data to replay memory
    agent.load_replay_memory()

    for frame, replay_memory in enumerate(agent.replay_memory.memory):
        if (frame > 600 and frame < 2000):
            reward = agent.update_reward(agent.deserialization.get_lon_decision_input_by_frame(frame))
            agent.deserialization.dump_ego_info_by_frame(frame)
            agent.deserialization.dump_obj_info_by_frame(frame)
            print("action: {}, reward: {}".format(replay_memory.action, reward))
            # agent.ogm.dump_ogm_graphs(replay_memory.state)
            # agent.ogm.dump_ogm_graphs(replay_memory.next_state)
            # agent.replay_memory.print_frame(frame)

    print("len(replay_memory): ", len(agent.replay_memory))



if __name__ == "__main__":
    main()