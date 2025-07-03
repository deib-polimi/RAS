import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from collections import deque
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
from .ppocontroller import PPOController, _ActorCritic, _RolloutBuf

class GPPPOController(PPOController):
    """PPO autoscaler with GP-based compensation and PID training data generation."""

    # GP parameters
    _GP_TRAIN_START = 100
    _GP_MIN_SAMPLES = 300
    _GP_TRAIN_FREQ = 50
    _GP_INPUT_DIM = 3  # [RL action, num_users, response_time]
    _GP_MAX_BUFFER_SIZE = 300
    _GP_PERCENTILE = 95  # Use 75th percentile for predictions

    def __init__(self, period, init_cores, *,
                 min_cores=1, max_cores=1000, st=0.8, name=None,
                 train=True, burst_mode="none", burst_threshold_q=20,
                 burst_threshold_r=30, burst_extra=4, trend_features=False,
                 enable_log=True, log_dir="./logs",
                 kp=100, ki=0.01):
                 #kp=0.5, ki=0.05):  # PID parameters
        super().__init__(period, init_cores, min_cores=min_cores,
                        max_cores=max_cores, st=st, name=name,
                        train=train, burst_mode=burst_mode,
                        burst_threshold_q=burst_threshold_q,
                        burst_threshold_r=burst_threshold_r,
                        burst_extra=burst_extra,
                        trend_features=trend_features,
                        enable_log=enable_log, log_dir=log_dir)

        # PID parameters
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain

        # Initialize GP
        kernel = Matern(length_scale=np.ones(self._GP_INPUT_DIM), nu=2.5) + \
                WhiteKernel(noise_level=0.1)
        self.gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.gp_data_buffer = deque(maxlen=self._GP_MAX_BUFFER_SIZE)
        self.gp_train_counter = 0

        # Initialize PID state
        self.integral_error = 0

        # Additional logging
        if self.enable_log:
            self.gp_log_path = os.path.join(log_dir, f"gpppo-gp-{time.time()}.log")

    def _calculate_pid_compensation(self, current_rt):
        """Calculate PID compensation based on response time error using standard PID parameters."""
        # Calculate error (same as CTControllerScaleX)
        e = current_rt - self.setpoint  # RT too high -> positive error -> add resources
        
        # Update integral term
        self.integral_error += e
        
        # Calculate PID output
        compensation = self.kp * e + self.ki * self.integral_error

        print(f"sla: {self.setpoint} rt: {current_rt} e: {e} integral_error: {self.integral_error} compensation: {compensation}")
        
        return compensation

    def _train_gp(self):
        """Train the GP model if enough data is available."""
        print(f"Training GP with {len(self.gp_data_buffer)} samples")
        if len(self.gp_data_buffer) < self._GP_MIN_SAMPLES:
            return

        X = np.array([x for x, _ in self.gp_data_buffer])
        y = np.array([y for _, y in self.gp_data_buffer])
        self.gpr.fit(X, y)
        self.gp_train_counter = 0

        if self.enable_log:
            with open(self.gp_log_path, "a") as f:
                f.write(f"GP trained with {len(X)} samples\n")

    def _get_gp_prediction(self, X_pred, current_rt):
        """Get GP prediction using the specified percentile, considering error direction."""
        # Get mean and standard deviation of the prediction
        mean, std = self.gpr.predict(X_pred, return_std=True)
        
        # Determine which percentile to use based on error direction
        error = current_rt - self.setpoint  # RT too high -> positive error -> add resources
        if error > 0:  # RT is too high, we want to scale up
            percentile = self._GP_PERCENTILE  # Use lower percentile to be conservative when adding resources
        else:  # RT is too low, we want to scale down
            percentile = 100 - self._GP_PERCENTILE  # Use higher percentile to be conservative when removing resources
        
        # Calculate the percentile value
        percentile_value = norm.ppf(percentile / 100.0, loc=mean, scale=std)
        
        #return float(percentile_value.item())
        return float(mean)

    def control(self, t):
        # Get base PPO action
        if self.burst_mode in ("guard", "hybrid") and self._burst():
            self.cores = min(self.cores + self.burst_extra, self.max_cores)
            if self.burst_mode == "guard":
                self._log(t, None, self.burst_extra, guard=True)
                return

        state = self._state()
        logits, val = self.ac(torch.tensor(state, dtype=torch.float32, device=self.device))
        dist = torch.distributions.Categorical(logits=logits)
        a_idx = int(dist.sample().item())
        logp = float(dist.log_prob(torch.tensor(a_idx)))
        val = float(val)

        # Get base RL action
        delta = int(self.actions[a_idx])
        delta = max(self.min_cores - self.cores, min(delta, self.max_cores - self.cores))
        action_base_rl = delta

        # Get current metrics for GP input
        current_rt = self.monitoring.getRT()
        num_users = self.monitoring.users[-1]

        # Calculate PID compensation for GP training
        pid_compensation = self._calculate_pid_compensation(current_rt)

        # Store data for GP training
        gp_input = np.array([action_base_rl, num_users, current_rt])
        print(f"GP input: {gp_input}")
        self.gp_data_buffer.append((gp_input, pid_compensation))

        # Get GP compensation if trained
        gp_compensation = 0
        if len(self.gp_data_buffer) >= self._GP_MIN_SAMPLES:
            print(f"Predicting GP with {len(self.gp_data_buffer)} samples") 
            # Reshape input for prediction and handle single prediction
            X_pred = gp_input.reshape(1, -1)
            gp_compensation = self._get_gp_prediction(X_pred, current_rt)
            print(f"GP compensation: {gp_compensation}")

        if t >= self._GP_TRAIN_START:
            final_delta = action_base_rl + gp_compensation
        else:
            final_delta = action_base_rl

        final_delta = max(self.min_cores - self.cores, min(final_delta, self.max_cores - self.cores))
        print(f"Final delta: {final_delta}, action_base_rl: {action_base_rl}, gp_compensation: {gp_compensation}")
        self.cores += final_delta

        # Update PPO training
        if self.train and self.prev_state is not None:
            self.buf.store(self.prev_state, self.prev_act, self.prev_logp,
                          self._reward(), False, self.prev_val)
            if len(self.buf) >= self._ROLLOUT:
                self._update()
                self.buf.reset()

        # Update GP training counter
        self.gp_train_counter += 1
        if self.gp_train_counter >= self._GP_TRAIN_FREQ:
            self._train_gp()

        # Update state tracking
        self.prev_state = state
        self.prev_act = a_idx
        self.prev_logp = logp
        self.prev_val = val
        self.step_cnt += 1

        # Log
        if self.enable_log:
            rt = self.monitoring.getRT()
            line = (f"{t:.1f}s lat={rt:.2f} cores={self.cores} "
                   f"Δ={final_delta:.2f} (RL:{action_base_rl:.2f} GP:{gp_compensation:.2f}) "
                   f"PID:{pid_compensation:.2f} rew={self.prev_reward:.2f}")
            print(line)
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def reset(self):
        """Reset controller state."""
        super().reset()
        self.integral_error = 0
        self.gp_data_buffer.clear()
        self.gp_train_counter = 0

    def set_pid_params(self, kp, ki):
        """Update PID parameters."""
        self.kp = kp
        self.ki = ki 