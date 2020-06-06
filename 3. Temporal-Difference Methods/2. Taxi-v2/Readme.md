# [Open AI's Taxi-v2 Problem](https://gym.openai.com/envs/Taxi-v2/) 

## Problem Statement

This task was introduced in [Dietterich2000] to illustrate some issues in hierarchical reinforcement learning. There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.

## Results and Observation
According to Open AI, the best average reward for taxi-v2 problem should be greater than 9.1.<br><br>


| S.No | Hyper-parameter                                                                                                            | Best Average Reward |
|------|----------------------------------------------------------------------------------------------------------------------------|---------------------|
| 1.   | 1. Expected SARSA<br> 2. alpha = 0.01<br> 3. gamma = 1<br> 4. epsilon = 1/num_episodes<br> 5. epsilon_greedy_policy<br>    | -31.01              |
| 2.   | 1. SARSAMAX<br> 2. alpha = 0.01<br> 3. gamma = 1<br> 4. epsilon = 1/num_episodes<br> 5. epsilon_greedy_policy<br>          | -27.19              |
| 3.   | 1. SARSAMAX<br> 2. alpha = 0.01<br> 3. gamma = 0.1<br> 4. epsilon = 1/num_episodes<br> 5. epsilon_greedy_policy<br>        | -12.51              |
| 4.   | 1. Expected SARSA<br> 2. alpha = 0.04<br> 3. gamma = 0.8<br> 4. epsilon = 1/num_episodes<br> 5. epsilon_greedy_policy <br> | 9.2                 |
| 5.   | 1. SARSA<br> 2. alpha = 0.04<br> 3. gamma = 0.8<br> 4. epsilon = 1/num_episodes<br> 5. epsilon_greedy_policy <br> | 9.38 (fastest convergence)                 |

