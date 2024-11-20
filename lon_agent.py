import sys
import numpy as np
from pathlib import Path
from FMDQN import DQNAgent, DQNReplayer, play_qlearning, fill_replay_memory

SMARTS_REPO_PATH = Path(__file__).parents[2].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))

def main():
    # replay memory
    replay_memory = DQNReplayer(capacity=100000)
    fill_replay_memory(replay_memory)
    # DQN parameters
    net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.0005}
    gamma=0.99
    epsilon=0.06
    batch_size=400
    agent = DQNAgent(net_kwargs, gamma, epsilon, batch_size,\
                     observation_dim=(3, 51, 101), action_size=3)
    train = False

    if train:
        play_qlearning(agent, replay_memory, episode, train, render=True)
        agent.save_model_params()
    else:
        agent.load_model_params()

if __name__ == "__main__":
    main()