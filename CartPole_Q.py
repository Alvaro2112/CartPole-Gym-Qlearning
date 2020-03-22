import numpy as np
import gym
import Matplotlib.pyplot as plt
import math


# Import and initialize Mountain Car Environment
env = gym.make('CartPole-v0')
env.reset()


# Determine size of discretized state space
NUMBER_STATES = (1, 1, 6, 12)

def plot_rewards(rewards):
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards.jpg')     
    plt.show()

def act(epsilon , current_state , Q_table ):
    if np.random.random() <= 1 - epsilon:
        return np.argmax(Q_table[current_state]) 
    else:
        return env.action_space.sample()

def get_epsilon(x):
        return max(0.01, min(1, 1 - math.log10((x + 1) / 25)))

# Update learning rate
def get_alpha(x):
        return max(0.1, min(1, 1 - math.log10((x + 1) / 25)))

def discretize(obs):
        upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
        lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((NUMBER_STATES[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(NUMBER_STATES[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

# Define Q-learning function
def QLearning(env, discount, epsilon, min_eps, episodes):

    # Initialize Q table
    Q_table = np.zeros(NUMBER_STATES + (env.action_space.n,))
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []

    # Run Q learning algorithm
    for i in range(episodes):

        # Initialize parameters
        tot_reward, reward = 0,0
        state = env.reset()
        alpha = get_alpha(i)
        epsilon = get_epsilon(i)
        done = False

        # Discretize state
        current_state = discretize(state)

        while not done:   
                
            # Determine next action - epsilon greedy strategy
            action = act(epsilon, current_state, Q_table)

            # Get next state and reward
            next_state, reward, done , _ = env.step(action) 
            
            # Discretize state2
            next_state = discretize(next_state)

            # Adjust Q value for current state
            Q_table[current_state][action] += alpha * (reward + discount * np.max(Q_table[next_state]) - Q_table[current_state][action])
                                     
            # Update variables
            tot_reward += reward
            current_state = next_state
        
        # Track rewards
        reward_list.append(tot_reward)
        
        # Print average reward for last 100 episodes
        if (i + 1) % 100 == 0:  
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []  
            print('Episode {} Average Reward: {}'.format(i + 1, ave_reward))

    env.close()
    
    return ave_reward_list 

# Run Q-learning algorithm
rewards = QLearning(env = env ,discount = 0.99 ,epsilon = 1 ,min_eps = 0.01 ,episodes = 2000)

# Plot Rewards
plot_rewards(rewards)
