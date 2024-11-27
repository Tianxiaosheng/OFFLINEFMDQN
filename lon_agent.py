import sys
import time
import numpy as np
from pathlib import Path
from FMDQN_core.FMDQN import DQNAgent, DQNReplayer, play_qlearning, replay_from_memory

# SMARTS_REPO_PATH = Path(__file__).parents[2].absolute()
# sys.path.insert(0, str(SMARTS_REPO_PATH))

def main(num_epochs_training, train=False):
    # DQN parameters
    file_path = sys.argv[1]
    net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.0005}
    gamma=0.99
    epsilon=0.00
    batch_size=400
    agent = DQNAgent(net_kwargs, gamma, epsilon, batch_size,\
                     observation_dim=(4, 51, 101), action_size=3,
                     offline_RL_data_path=file_path)

    agent.load_replay_memory()

    print("replay_memory accuracy Before training:")
    replay_from_memory(agent)
    if train:
        for epoch in range(num_epochs_training):
            start_time = time.time()
            play_qlearning(agent, train, render=True)
            end_time = time.time()
            print("epoch {} train time: {}".format(epoch, end_time - start_time))
            # save nn model
            if (epoch % 100 == 0):
                agent.save_model_params()

    else:
        agent.load_model_params()

    print("replay_memory accuracy After training:")
    replay_from_memory(agent)

if __name__ == "__main__":
    main(num_epochs_training=400, train=True)
