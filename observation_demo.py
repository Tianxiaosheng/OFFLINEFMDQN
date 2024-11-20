#! /usr/bin/env python3

import sys
import numpy as np
from pathlib import Path
from FMDQN_core.FMDQN import DQNAgent, DQNReplayer

# SMARTS_REPO_PATH = Path(__file__).parents[2].absolute()
# sys.path.insert(0, str(SMARTS_REPO_PATH))

def main():
    # replay memory
    replay_memory = DQNReplayer(capacity=100000)

    # DQN parameters
    file_path = sys.argv[1]
    net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.0005}
    gamma=0.99
    epsilon=0.06
    batch_size=400
    agent = DQNAgent(net_kwargs, gamma, epsilon, batch_size,\
                     observation_dim=(4, 51, 101), action_size=3,
                     offline_RL_data_path=file_path)
    if agent.deserialization.get_lon_decision_inputs_size() > 297:
        agent.get_observation_from_lon_decision_input(agent.deserialization.get_lon_decision_input_by_frame(296))
        agent.ogm.dump_ogm_graphs()
    else:
        print("No lon decision inputs found")

if __name__ == "__main__":
    main()