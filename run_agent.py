import gym
from Agents import QAgent
import time
import numpy as np
from crawler_env import CrawlingRobotEnv

##Todo: Build a robot that can learn to crawl 

gamma = 0.9

env = CrawlingRobotEnv(render=False)
agent = QAgent(env,gamma)

current_state = env.reset()
cumulative_rewards = 0 
avg_reward = 0

i = 0
while i < 1000000:
    
    i = i + 1
    # Q-learning
    # Step 1: Initialize Q(S,A)=0 for all states and take the intial/first action
    action = agent.choose_action(current_state)
    next_state, reward, done, info = env.step(action)      
    
    # Step 2: The agent gets the next state and reward. Using this info update the previous Q-value
    agent.learn(current_state,action,reward,next_state)
    current_state = next_state                                # Update the new state
    cumulative_rewards += reward
    #time.sleep(0.05)
    if(i%5000 == 0):
        print('Average rewards over last ',i,' iterations:',cumulative_rewards/i)
        if(cumulative_rewards/i) > 1.3:
            break
        env.render = True
        
env = CrawlingRobotEnv(render=True)
current_state = env.reset()
cumulative_rewards = 0
agent.eps = 0                  # Select only greedy actions
i = 0

while True:
    
    i = i + 1
    # Q-learning
    # Step 1: Initialize Q(S,A)=0 for all states and take the intial/first action
    action = agent.choose_action(current_state)
    next_state, reward, done, info = env.step(action)      
        
    # Step 2: The agent gets the next state and reward. Using this info update the previous Q-value
    agent.learn(current_state,action,reward,next_state)
    current_state = next_state                                # Update the new state
    cumulative_rewards += reward
    time.sleep(0.1)
    if(i%5000 == 0):
        print('Average rewards over last 5000 iterations:',cumulative_rewards/i)
        if(cumulative_rewards/i) > 1.3:
            break
        