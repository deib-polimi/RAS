from .controller import Controller
import numpy as np
import random
import pickle
import os
import time


class RLController(Controller):

    def __init__(self, period, init_cores, min_cores=1, max_cores=1000, st=0.8, name=None, train=True):
        
        super().__init__(period, init_cores, st, name=name)
        self.min_cores = min_cores
        self.max_cores = max_cores
        self.log_path = f"./logs/trace_rlcontroller-{time.time()}.log"
        self.q_table_path = "./controllers/rlcontroller.pkl"
        # Initialize Q-table as a dictionary for sparse representation
        self.Q = self.load_q_table()
        self.train = train

        # Hyperparameters
        self.alpha = 0.05  # Learning rate
        self.gamma = 0.75  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.alarm_state = 0.02 # +0.02 over set point states are defined with the "ALARM-" prefix

        # states are defined as (error, qn_ct_core_diff, user_slope) such metrics are quantized as follows
        self.error_quantization = 0.05 # states are defined by chunks of 0.05s
        self.user_quantitization = 10 # states are defined by chunks of 10 users

        self.scale_reward_if_violation = 10 # how large is the penalty if the rt > setpoint

        # Possible actions
        self.actions = list(np.arange(-2, 3, 1))
        
        self.state = None

    def control(self, t):
        action = self.learn(t)
        self.cores += action
        self.cores = min(max(self.min_cores, self.cores), self.max_cores)

    def save_q_table(self):
        with open(self.q_table_path, 'wb') as f:
            pickle.dump(self.Q, f)
        print(f"Q-table saved to {self.q_table_path}")

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'rb') as f:
                Q = pickle.load(f)
            print(f"Q-table loaded from {self.q_table_path}")
        else:
            Q = {}
            print("No Q-table found, starting with an empty Q-table")
        return Q

    def reward(self):
        error = self.monitoring.getRT()-self.setpoint
        return -1 * (self.scale_reward_if_violation*abs(error) if error > 0 else abs(error))
    
    def get_state(self):
        error = self.monitoring.getRT() - self.setpoint
        quantized_error = max(-10, min(10, int(round(error, 2) / self.error_quantization)))
        quantized_error = f"{'ALARM-' if error > self.alarm_state else ''}{quantized_error}"
        
        try:
            users = self.monitoring.getAllUsers()
            quantized_user_difference = int((users[-1] - users[-3])/self.user_quantitization)
        except:
            quantized_user_difference  = 0
        

        return quantized_error, quantized_user_difference
    
    def feasible_actions(self):
        return [a for a in self.actions if self.cores+a >= self.min_cores and self.cores+a <= self.max_cores]

    def learn(self, t):
        new_state = self.get_state()
        
        reward = 0
        
        if new_state not in self.Q:
            self.Q[new_state] = np.zeros(len(self.actions))
        
        # update Q table before the next step (not executed in the first step)
        if self.state != None:
            reward = self.reward()
            # Update Q-value
           
            self.Q[self.state][self.action_index] = self.Q[self.state][self.action_index] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state]) - self.Q[self.state][self.action_index])

        self.state = new_state    
        feasible_actions = self.feasible_actions()
        feasible_indices = [self.actions.index(action) for action in feasible_actions]

        if random.uniform(0, 1) < self.epsilon and self.train:
            self.action_index = random.choice(feasible_indices)
        else:
            self.action_index = max(feasible_indices, key=lambda index: self.Q[new_state][index])
            
        
        action = self.actions[self.action_index]

        log = f"RL {self.generator.name} - t:, {t} state: {new_state}, rt: {self.monitoring.getRT():.2f}, reward: {reward:.2f}, action: {action})"
        print(log)
        with open(self.log_path, 'a') as log_file:
            log_file.write(log + '\n')

        if self.train:
            self.save_q_table()
        return action
    

    def setTrain(self, train): self.train = train