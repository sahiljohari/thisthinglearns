# Playing Snake using Deep Reinforcement Learning

In this project, I have implemented a deep Q-network (DQN) to play a simple game of Snake. The goal is to maximize the score of game in a 10x10 space and modify the DQN to stretch out the performance as much as possible.

*Find full explanation [here](https://github.com/sahiljohari/thisthinglearns/blob/master/DeepQLearning_Snake_v1.ipynb)*
## Q-learning and Deep Q-Networks
There are two main approaches to solving RL problems:
- Methods based on value functions
- Methods based on policy search

There is also a hybrid actor-critic approach that employs both value functions and policy search. As far as the scope of this project is concerned, I have used the _value functions_ approach to play the Snake game.

In Q-learning, which is a value functions approach, we use a Q-table that maps environment states with actions taken on them. For each (state, action) pair, there is a reward that the agent achieves. The idea is to pick a value that maximizes the cumulative reward by the end of an episode. To do this, we use an equation called the Bellman equation, which is shown below:
![Bellman equation](https://github.com/sahiljohari/thisthinglearns/blob/master/Bellman_equation.PNG)

## Creating the game
The first step is to create an environment for our agent to operate on. This has been implemented in a simple way using Python coroutines and will be rendered (or displayed) using Pyplot (from Matplotlib). It can also be implemented in a way similar to OpenAI Gym environments, but I specifically want my own environment to control all the aspects of its execution behavior.

## Modeling an Agent
The agents takes the parameters and uses a Convolutional Neural Network to build a DQN. It then obtains an instance of the game environment and begins training on it. The training process comprises of the following steps:

- For half of the nb_epochs, take random actions and capture the game states and rewards - referred as Experience Replay.
- While the first half of the training consists of random actions (exploration), the other half is all about the model taking actions, known as exploitation.
- The set of experiences are sub-sampled into a batches of size batch_size.
- The batches are iterated through and a set of target Q-values are calculated using the Bellman equation.
- These Q-values are mapped to the states, which gives us kind of a Q-table that is fed to the neural network for training.

## Let's play!
![output](https://github.com/sahiljohari/thisthinglearns/blob/master/output_WX5pMq.gif)

After training for about 30,000 training epochs and testing for 100 episodes, the agent could achieve a highest score of 7.

## Conclusion
In this project, I have applied the concept of deep reinforcement learning on the classic game - Snake. I used the approach called Q-learning, which is based on value functions that estimates the expected return of being in a given state. I extended this approach to deep Q-network and used a convolutional neural network to implement it. Using this approach, the maximum score I could achieve was 7 (seven).

## Future Work
This project can be further extended using concepts like Policy search and Actor-Critic method. An interesting implementation would be to incorporate Genetic algorithm to create a population of agents and filter out the best through various generations.

## References
- [Basic Reinforcement Learning by Víctor Mayoral Vilches](https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial6/examples/Snake/snake.py)
- [Interactive Python Notebook for RL in Catch](https://gist.github.com/cadurosar/bd54c723c1d6335a43c8)
- [Introduction to reinforcement learning and OpenAI Gym](https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym)
