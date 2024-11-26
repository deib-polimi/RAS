
from controllers import OPTCTRL, CTControllerScaleXJoint
import numpy as np
import random
import pickle
import os
import time


class JointController(OPTCTRL):
    def __init__(self, period, init_cores, range=(0.9, 1.6), min_cores=1, max_cores=1000, st=0.8, name=None):
        super().__init__(period, init_cores, min_cores, max_cores, st, name=name)
        self.scalex = CTControllerScaleXJoint(period, init_cores, st=st, max_cores=max_cores, min_cores=min_cores)
        self.qn_cores = 0
        self.range = range
        self.minRange = 0.5
        self.maxRange = 5
        self.rl = None

    def control(self, t):
        super().control(t)
        self.qn_cores = self.cores
        self.ct_cores = self.scalex.tick(t)
        
        if self.rl:
            action = self.rl.learn(t)
            self.range=(max(self.minRange, min(1, self.range[0]+action[0])), min(self.maxRange, max(1, self.range[1]+action[1])))
        
        m = max(self.min_cores, self.qn_cores*self.range[0])
        M = self.qn_cores*self.range[1]

        self.min_cores = m
        self.max_cores = M
        #self.cores = min(max(self.ct_cores, m), M)
        self.cores=self.ct_cores
        #self.cores=self.qn_cores
        #self.cores=self.ct_cores

    def setRL(self, rl):
        if rl:
            self.rl = RL(self)
        else:
            self.rl = None

    def setMonitoring(self,monitoring):
        super().setMonitoring(monitoring)
        self.scalex.setMonitoring(self.monitoring)

    def setSLA(self,sla):
        super().setSLA(sla)
        self.scalex.setSLA(sla)

    def reset(self):
        super().reset()
        self.scalex.reset()

    def setRange(self, range):
        self.range = range

    def __str__(self):
        return super().__str__() + " BC: %.2f DC: %.2f " % (self.BC, self.DC)
    

class RL:

    def __init__(self, controller):
        self.log_path = f"./logs/trace_rl-{time.time()}.log"
        self.q_table_path = "./controllers/jointcontroller.pkl"
        # Initialize Q-table as a dictionary for sparse representation
        self.Q = self.load_q_table()
        self.controller = controller
        # Hyperparameters
        self.alpha = 0.05  # Learning rate
        self.gamma = 0.75  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.alarm_state = 0.02 # +0.02 over set point states are defined with the "ALARM-" prefix

        # states are defined as (error, qn_ct_core_diff, user_slope) such metrics are quantized as follows
        self.error_quantization = 0.05 # states are defined by chunks of 0.05s
        self.user_quantitization = 10 # states are defined by chunks of 10 users
        self.core_quantization = 0.5 # states are defined by chunks of 0.5 cores 

        self.scale_reward_if_violation = 100 # how large is the penalty if the rt > setpoint

        # Possible actions
        self.actions = [(-0.2, +0.2), (-0.2, +0), (-0.2, -0.2), (+0.2, -0.2), 
                (+0.2, +0), (+0.2, +0.2), (0, -0.2), (0, 0), (0, -0.2)]
        
        self.state = None

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
        error = self.controller.monitoring.getRT()-self.controller.scalex.setpoint
        return -1 * (self.scale_reward_if_violation*abs(error) if error > 0 else abs(error))
    
    def get_state(self):
        error = self.controller.monitoring.getRT() - self.controller.scalex.setpoint
        quantized_error = int(round(error, 2) / self.error_quantization)
        quantized_error = f"{'ALARM-' if error > self.alarm_state else ''}{quantized_error}"
        
        try:
            users = self.controller.monitoring.getAllUsers()
            quantized_user_difference = int((users[-1] - users[-3])/self.user_quantitization)

        except:
            quantized_user_difference  = 0

        core_diff = self.controller.qn_cores - self.controller.ct_cores
        quantized_core_diff = round(core_diff, 1)
        quantized_core_diff = int(core_diff/self.core_quantization)

        return quantized_error, quantized_core_diff, quantized_user_difference
    
    def feasible_actions(self):
        rl, rr = self.controller.range
        m = self.controller.minRange
        M = self.controller.maxRange
        return [(al, ar) for al, ar in self.actions if rl+al >= m and rl+al <= 1 and rr+ar <= M and rr+ar >= 1]

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

        if random.uniform(0, 1) < self.epsilon:
            self.action_index = random.choice(feasible_indices)
        else:
            self.action_index = max(feasible_indices, key=lambda index: self.Q[new_state][index])
            
        
        action = self.actions[self.action_index]

        log = f"RL {self.controller.generator.name} - t:, {t} state: {new_state}, rt: {self.controller.monitoring.getRT():.2f}, reward: {reward:.2f}, action: {action}, prev_range: ({self.controller.range[0]:.1f}, {self.controller.range[1]:.1f}), qn_cores: {self.controller.qn_cores:.2f}, ct_cores: {self.controller.ct_cores:.2f})"
        print(log)
        with open(self.log_path, 'a') as log_file:
            log_file.write(log + '\n')

        self.save_q_table()
        return action
    
        
      
        
        
       
       
    
