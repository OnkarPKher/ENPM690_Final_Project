import numpy as np
import gym
import random
import matplotlib.pyplot as plt

env = gym.make("SpaceInvaders-v0")
env.render()

total_test_episodes = 15     # Total number of test episodes used for testing
max_steps = 10000                # Maximum number of steps per episode for testing

qtable = np.load('Qtable.npy')  # Values are stored in this Q-table.npy file  

env.reset()
rewards = []
# Testing learning measurement variables
current_step = 0
step_number = []
accumulated_reward = []
episodes = []

for episode in range(total_test_episodes):
    state = env.reset()
    all_rewards = 0
    state = 0
    step = 0
    done = False
    total_rewards = 0
    #print("****************************************************")
    #print("EPISODE ", episode)

    for step in range(max_steps):
        env.render()
        # Taking the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])

        # Retrieving the new state, reward, and game status from the action
        new_state_array, reward, done, info = env.step(action)
        
        new_state = 0
        # Finding the new state that the spaceship is at
        for i in range(159):
            if new_state_array[185][i][0] == 50:
                new_state = i
                break
        # Appending all the learning measurement data
        all_rewards += reward
        accumulated_reward.append(all_rewards)
        step_number.append(current_step)
        # Incrementing the current step number for learning measurement graph
        current_step += .001    

        total_rewards += reward

        # Checking if the game is finished
        if done:
            rewards.append(total_rewards)
            episodes.append(episode)
            print ("Score", total_rewards)
            break
        state = new_state
    
env.close()
print ("Average Score: ",  (sum(rewards))/total_test_episodes)

# Test - Total Reward v/s Step Number graph
plt.plot(step_number, accumulated_reward, linewidth=1.0)
plt.title('Reward per Game vs Step Number for Q-Learning Test')
plt.ylabel('Accumulated Reward')
plt.xlabel('Numbers of Steps (in thousands)')
plt.show()

# Test - Number of episodes v/s Rewards gained
plt.bar(episodes, rewards)
plt.title('Reward per Game for Q-Learning Test')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()
