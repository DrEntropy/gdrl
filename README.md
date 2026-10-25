# Grokking Deep Reinforcement Learning

This is a personal fork of [mimoralea/gdrl](https://github.com/mimoralea/gdrl), for my own use. It is not the official book repository.

Chapter 6 runs locally with [uv](https://docs.astral.sh/uv/) and current Gymnasium. Other chapters still expect the original notebook APIs.

## Running chapter 6

```bash
uv sync
uv run jupyter lab notebooks/chapter_06/chapter-06.ipynb
```

## About the book

### Book's website

https://www.manning.com/books/grokking-deep-reinforcement-learning

### Table of content

  1. [Introduction to deep reinforcement learning](#1-introduction-to-deep-reinforcement-learning)
  2. [Mathematical foundations of reinforcement learning](#2-mathematical-foundations-of-reinforcement-learning)
  3. [Balancing immediate and long-term goals](#3-balancing-immediate-and-long-term-goals)
  4. [Balancing the gathering and utilization of information](#4-balancing-the-gathering-and-utilization-of-information)
  5. [Evaluating agents' behaviors](#5-evaluating-agents-behaviors)
  6. [Improving agents' behaviors](#6-improving-agents-behaviors)
  7. [Achieving goals more effectively and efficiently](#7-achieving-goals-more-effectively-and-efficiently)
  8. [Introduction to value-based deep reinforcement learning](#8-introduction-to-value-based-deep-reinforcement-learning)
  9. [More stable value-based methods](#9-more-stable-value-based-methods)
  10. [Sample-efficient value-based methods](#10-sample-efficient-value-based-methods)
  11. [Policy-gradient and actor-critic methods](#11-policy-gradient-and-actor-critic-methods)
  12. [Advanced actor-critic methods](#12-advanced-actor-critic-methods)
  13. [Towards artificial general intelligence](#13-towards-artificial-general-intelligence)

### Detailed table of content

#### 1. Introduction to deep reinforcement learning
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-1)\)
- \(No Notebook\)
      
#### 2. Mathematical foundations of reinforcement learning
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-2)\)
- \([Notebook](/notebooks/chapter_02/chapter-02.ipynb)\)
  - Implementations of several MDPs: 
    - Bandit Walk
    - Bandit Slippery Walk
    - Slippery Walk Three
    - Random Walk
    - Russell and Norvig's Gridworld from AIMA
    - FrozenLake
    - FrozenLake8x8
#### 3. Balancing immediate and long-term goals
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-3)\)
- \([Notebook](/notebooks/chapter_03/chapter-03.ipynb)\) 
  - Implementations of methods for finding optimal policies:
    - Policy Evaluation
    - Policy Improvement
    - Policy Iteration
    - Value Iteration
#### 4. Balancing the gathering and utilization of information
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-4)\)
- \([Notebook](/notebooks/chapter_04/chapter-04.ipynb)\)
  - Implementations of exploration strategies for bandit problems:
    - Random
    - Greedy
    - E-greedy
    - E-greedy with linearly decaying epsilon
    - E-greedy with exponentially decaying epsilon
    - Optimistic initialization
    - SoftMax
    - Upper Confidence Bound
    - Bayesian
#### 5. Evaluating agents' behaviors
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-5)\)
- \([Notebook](/notebooks/chapter_05/chapter-05.ipynb)\)
  - Implementation of algorithms that solve the prediction problem (policy estimation):
    - On-policy first-visit Monte-Carlo prediction
    - On-policy every-visit Monte-Carlo prediction
    - Temporal-Difference prediction (TD)
    - n-step Temporal-Difference prediction (n-step TD)
    - TD(λ)
#### 6. Improving agents' behaviors
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-6)\)
- \([Notebook](/notebooks/chapter_06/chapter-06.ipynb)\)
  - Implementation of algorithms that solve the control problem (policy improvement):
    - On-policy first-visit Monte-Carlo control
    - On-policy every-visit Monte-Carlo control
    - On-policy TD control: SARSA
    - Off-policy TD control: Q-Learning
    - Double Q-Learning
#### 7. Achieving goals more effectively and efficiently
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-7)\)
- \([Notebook](/notebooks/chapter_07/chapter-07.ipynb)\)
  - Implementation of more effective and efficient reinforcement learning algorithms:
    - SARSA(λ) with replacing traces
    - SARSA(λ) with accumulating traces
    - Q(λ) with replacing traces
    - Q(λ) with accumulating traces
    - Dyna-Q
    - Trajectory Sampling
#### 8. Introduction to value-based deep reinforcement learning
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-8)\)
- \([Notebook](/notebooks/chapter_08/chapter-08.ipynb)\)
  - Implementation of a value-based deep reinforcement learning baseline:
    - Neural Fitted Q-iteration (NFQ)
#### 9. More stable value-based methods
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-9)\)
- \([Notebook](/notebooks/chapter_09/chapter-09.ipynb)\)
  - Implementation of "classic" value-based deep reinforcement learning methods:
    - Deep Q-Networks (DQN)
    - Double Deep Q-Networks (DDQN)
#### 10. Sample-efficient value-based methods
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-10)\)
- \([Notebook](/notebooks/chapter_10/chapter-10.ipynb)\)
  - Implementation of main improvements for value-based deep reinforcement learning methods:
    - Dueling Deep Q-Networks (Dueling DQN)
    - Prioritized Experience Replay (PER)
#### 11. Policy-gradient and actor-critic methods
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-11)\)
- \([Notebook](/notebooks/chapter_11/chapter-11.ipynb)\)
  - Implementation of classic policy-based and actor-critic deep reinforcement learning methods:
    - Policy Gradients without value function and Monte-Carlo returns (REINFORCE)
    - Policy Gradients with value function baseline trained with Monte-Carlo returns (VPG)  
    - Asynchronous Advantage Actor-Critic (A3C)
    - Generalized Advantage Estimation (GAE)
    - \[Synchronous\] Advantage Actor-Critic (A2C)
#### 12. Advanced actor-critic methods
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-12)\)
- \([Notebook](/notebooks/chapter_12/chapter-12.ipynb)\)
  - Implementation of advanced actor-critic methods:
    - Deep Deterministic Policy Gradient (DDPG)
    - Twin Delayed Deep Deterministic Policy Gradient (TD3)
    - Soft Actor-Critic (SAC)
    - Proximal Policy Optimization (PPO)
#### 13. Towards artificial general intelligence
- \([Livebook](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-13)\)
- \(No Notebook\)
