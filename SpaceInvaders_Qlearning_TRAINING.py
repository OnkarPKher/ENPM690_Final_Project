import numpy as np
import gym
import random
import matplotlib.pyplot as plt 

env = gym.make("SpaceInvaders-v0")
env.render()


action_size = env.action_space.n
print("Action size ", action_size)

# state_size = env.observation_space
state_size = 1000
print("State size ", state_size)

qtable = np.zeros((state_size, action_size))


total_episodes = 5000            # Total number episodes
max_steps = 100000               # Maximum number of steps per episode

learning_rate = 0.25             # Learning rate
gamma = 0.75                     # Discounting rate

# Hyper-parameters for exploration
epsilon = 1.0                   # Rate of Exploration
max_epsilon = 1.0               # Probability of exploration at the start
min_epsilon = 0.01              # Minimum exploration probability 
decay_rate = 0.001              # Exponential decay rate for exploration prob

# Training learning measurement variables
# Initializing some data structures for Q-learning  
current_step = 0
step_number = []
accumulated_reward = []
accumulated_avg = []
all_rewards = 0
rewards = []
episodes = []

# Step 2: For life or until learning is stopped
for episode in range(total_episodes):   
    print('Episode ', episode+1,'/',total_episodes)
    # Resetting the environment
    state = env.reset()
    # Initialize the state to zero
    state = 0
    step = 0
    avg_rewards = 0
    total_rewards = 0
    done = False
    
    for step in range(max_steps):
        # Step 3: Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0,1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
            
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state_array, reward, done, info = env.step(action)
 
        # Append all the learning measurement data
        all_rewards += reward
        avg_rewards += reward
        total_rewards += reward
        accumulated_reward.append(all_rewards)
        accumulated_avg.append(avg_rewards)
        step_number.append(current_step)
        # Increment the current step number for learning measurement graph
        current_step += .001
        
        # Finding the new location of the new state
        new_state = 0
        for i in range(159):
            if new_state_array[185][i][0] == 50:
                new_state = i
                break
           
            
            
        # Update the Q-values by using the Bellman Equation given by:
        # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * 
                                    np.max(qtable[new_state, :]) - qtable[state, action])
                
        # Our new state is equal to the state
        state = new_state
        
        # If done : finish the episode
        if done == True:
            rewards.append(total_rewards)
            episodes.append(episode)
            break
    
    # Reduce the epsilon (because we need less and less exploration as the game proceeds)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

# Save the Q-table with the training info
np.save('Qtable', qtable)

# Training - Total Reward v/s Step Number graph
plt.plot(step_number, accumulated_reward, linewidth=1.0)
plt.title('Total Reward vs Step Number for Q-Learning Training')
plt.ylabel('Accumulated Reward')
plt.xlabel('Numbers of Steps (in thousands)')
plt.show()

# Training - Total Reward v/s Step Number graph
plt.plot(step_number, accumulated_avg, linewidth=1.0)
plt.title('Single Game Score vs Step Number for Q-Learning Training')
plt.ylabel('Accumulated Reward')
plt.xlabel('Numbers of Steps (in thousands)')
plt.show()

# Training - Score v/s Episode Number
plt.bar(episodes, rewards, linewidth=1.0)
plt.title('Reward per Game for Q-Learning Training')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()
