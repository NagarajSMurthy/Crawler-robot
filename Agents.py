import random
from collections import defaultdict
import numpy as np
class QAgent():
    def __init__(self,env,gamma):
        
        self.gamma = gamma
        self.env = env
        self.q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        self.alpha = 0.4
        self.eps = 0.5

    def choose_action(self,state):
        ##Todo: Epsilon-Greedy Action Selector
        if random.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_vals[state])

        return action 
        
    def learn(self,cur_state,action,reward,next_state):
        ##Todo: Learn From Your Experience
        # Update the Q-value
        max_q = np.max(self.q_vals[next_state])
        new_val = reward + (self.gamma*max_q) 
        oldv = self.q_vals[cur_state][action]
        self.q_vals[cur_state][action] = oldv + self.alpha*(new_val - oldv)
        
     
