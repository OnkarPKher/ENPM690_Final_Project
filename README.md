# ENPM690_Final_Project

Done by 
</p> Kiran Palavlasa 119469804
</p>Onkar Prasanna Kher 120407062 

## Dependencies

The following dependencies must be installed. There are a lot of them, and it might be a pain to install some of them.

1. python3.5 or above 
2. numpy 
3. collections
4. tensorflow
5. random
6. skimage
7. gym
8. retro
9. keras

To import the dependencies above, use:

```
pip install (dependency name)
```

## Q-Learrning Training and Testing

<p align="center">
  <img src="GIF/graph.png">
  <br><b>Fig 1: Q-Learning Reward vs Number of Steps (500 Episodes) </b><br>
</p>

<p align="center">
  <img src="GIF/Space_invader_scores.gif">
  <br><b>Fig 2: Q-Learning playing Space Invaders with 500 Episodes of training</b><br>
</p>

# How to run
Run the code (.py) file on any IDE(VSC recommended) with all the dependencies installed 

# Q-Learning Training and Testing
To begin training and testing your Q-learning model, follow the steps outlined below:

# Training
Configure Your Environment: Set up your learning environment using the gym and retro libraries. You can choose from a variety of environments like Space Invaders.
1) Initialize the Q-table: Create a Q-table that will store the values for each (state, action) pair. If the state space is too large, consider using a neural network as a function approximator.
2) Set the Learning Parameters: Define the learning rate (alpha), discount factor (gamma), and exploration rate (epsilon). These will control the learning process.
# Start Training: For each episode:
Initialize the environment.
# For each step in the episode:
Choose an action using an ε-greedy policy: select a random action with probability ε or the action with the highest Q-value with probability 1-ε.
Execute the action in the environment.
Observe the new state and reward.
Update the Q-table using the Bellman equation.
If the episode ends (e.g., the game is over), break from the loop.
Repeat for a Desired Number of Episodes: The number of episodes depends on your specific problem but start with at least 500 for basic convergence.
# Testing
Test the Trained Model: After training, test the model by running it through the environment without ε-greedy exploration (set ε to 0) to see how well it performs.
Record the Outcomes: Keep track of the total rewards and the number of steps taken in each episode.
