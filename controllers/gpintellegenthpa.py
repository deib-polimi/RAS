#!/usr/bin/env python3
"""
GP-enhanced Intelligent HPA Controller

Integrates Gaussian Process and PID compensation into the neural network-based
autoscaling controller. Combines:
- Neural Network predictions for strategic scaling decisions
- PID feedback for immediate response time error corrections  
- Gaussian Process learning for adaptive compensation patterns

Architecture:
Neural Network → Base Action → GP Compensation → PID Training → Final Action
"""

import os
import time
import numpy as np
from datetime import datetime
from collections import deque
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm

# Handle both relative and absolute imports
try:
    from .intellegentHPA import intellegentHPA, AutoscalingPredictor
except ImportError:
    try:
        from intellegentHPA import intellegentHPA, AutoscalingPredictor
    except ImportError:
        # Fallback to direct import
        import sys
        sys.path.append('.')
        from intellegentHPA import intellegentHPA, AutoscalingPredictor


class GPintellegentHPA(intellegentHPA):
    """
    Enhanced Neural Network autoscaler with GP-based compensation and PID training data generation.
    
    Combines:
    - Neural Network: Strategic predictions based on historical patterns
    - PID Controller: Immediate feedback correction for response time errors
    - Gaussian Process: Learns compensation patterns to improve over time
    """

    # GP parameters
    _GP_TRAIN_START = 150
    _GP_MIN_SAMPLES = 150
    _GP_TRAIN_FREQ = 50
    _GP_INPUT_DIM = 3  # [NN action delta, num_users, response_time]
    _GP_MAX_BUFFER_SIZE = 300
    _GP_PERCENTILE = 99  # Use 95th percentile for predictions

    def __init__(self, period, init_cores, st=0.8, name=None,
                 model_path: str = 'models/model_response_time.h5',
                 max_cores: int = 300, window_size: int = 5,
                 kp=100, ki=0.01, enable_log=True, log_dir="./logs"):
        """
        Initialize the GP-enhanced Neural Network Controller.
        
        Args:
            period: Control period (how often to make scaling decisions)
            init_cores: Initial number of cores
            st: Safety threshold multiplier for SLA
            name: Controller name
            model_path: Path to the trained neural network model
            max_cores: Maximum number of cores available
            window_size: Size of historical data window for NN predictions
            kp: PID proportional gain
            ki: PID integral gain  
            enable_log: Enable detailed logging
            log_dir: Directory for log files
        """
        super().__init__(period, init_cores, st, name, model_path, max_cores, window_size)

        # PID parameters
        self.kp = kp
        self.ki = ki

        # Initialize GP
        kernel = Matern(length_scale=np.ones(self._GP_INPUT_DIM), nu=2.5) + \
                WhiteKernel(noise_level=0.1)
        self.gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.gp_data_buffer = deque(maxlen=self._GP_MAX_BUFFER_SIZE)
        self.gp_train_counter = 0

        # Initialize PID state for NN error correction
        self.integral_error = 0
        
        # State tracking for control effectiveness feedback
        self.previous_actual_rt = None  # RT risultante dal controllo precedente

        # Logging
        self.enable_log = enable_log
        self.log_dir = log_dir
        if self.enable_log:
            os.makedirs(log_dir, exist_ok=True)
            self.gp_log_path = os.path.join(log_dir, f"gpnn-gp-{time.time():.0f}.log")
            self.main_log_path = os.path.join(log_dir, f"gpnn-main-{time.time():.0f}.log")

        # Track step counter for GP training start
        self.step_cnt = 0

    def _calculate_pid_compensation(self, control_error):
        """
        Calculate PID compensation based on control effectiveness error.
        
        Args:
            control_error: Error between actual RT and target RT from previous step
                         (positive = RT too high, need more control/cores)
                         (negative = RT too low, had too much control/cores)
        
        Returns:
            PID compensation in cores (positive = add cores, negative = remove cores)
        """
        # PID works on control effectiveness error (actual_rt - target_rt)
        e = control_error
        
        # Update integral term
        self.integral_error += e
        
        # Calculate PID output (in cores)
        # If RT too high (positive error), add more cores
        # If RT too low (negative error), reduce cores
        compensation = self.kp * e + self.ki * self.integral_error

        if self.enable_log:
            print(f"PID-Control-Correction: rt_error={e:.3f} integral={self.integral_error:.3f} "
                  f"compensation={compensation:.3f} cores")
        
        return compensation

    def _train_gp(self):
        """Train the GP model if enough data is available."""
        if len(self.gp_data_buffer) < self._GP_MIN_SAMPLES:
            return

        X = np.array([x for x, _ in self.gp_data_buffer])
        y = np.array([y for _, y in self.gp_data_buffer])
        
        if self.enable_log:
            print(f"Training GP with {len(X)} samples")
            
        self.gpr.fit(X, y)
        self.gp_train_counter = 0

        if self.enable_log:
            with open(self.gp_log_path, "a") as f:
                f.write(f"t={time.time():.1f}: GP trained with {len(X)} samples\n")

    def _get_gp_prediction(self, X_pred, current_rt):
        """
        Get GP prediction using the specified percentile for conservative compensation.
        
        Args:
            X_pred: Input features for GP prediction
            current_rt: Current response time (for potential directional logic)
            
        Returns:
            Conservative compensation prediction using _GP_PERCENTILE
        """
        # Get mean and standard deviation of the prediction
        mean, std = self.gpr.predict(X_pred, return_std=True)
        
        # Extract scalar values
        mean_val = float(mean.item() if hasattr(mean, 'item') else mean)
        std_val = float(std.item() if hasattr(std, 'item') else std)
        
        # Calculate conservative percentile prediction
        # Using upper percentile for more conservative (higher) compensation
        percentile_value = norm.ppf(self._GP_PERCENTILE / 100.0, loc=mean_val, scale=std_val)
        
        if self.enable_log:
            print(f"GP-Percentile: mean={mean_val:.3f}, std={std_val:.3f}, "
                  f"{self._GP_PERCENTILE}th percentile={percentile_value:.3f}")
        
        return float(percentile_value)

    def control(self, t):
        """
        Enhanced control logic with GP-learned PID compensation.
        
        Flow:
        1. Get NN recommendation for current step
        2. Calculate PID compensation based on control effectiveness from previous step
        3. Store PID compensation for GP learning  
        4. Use GP prediction (if trained) or direct PID compensation
        5. Combine NN + compensation for final action
        6. Store actual RT for next step feedback
        
        Evolution:
        - Phase 1: NN + PID (collecting data)
        - Phase 2: NN + GP (GP predicts PID compensations)
        
        The GP learns to predict when and how much PID compensation is needed,
        making the system more proactive rather than reactive.
        """
        if not self.monitoring or not self.generator:
            return

        try:
            # Get current metrics
            current_metrics = self._get_current_metrics(t)
            if current_metrics is None:
                return

            current_users, current_response_time = current_metrics

            # === PHASE 1: Get Neural Network recommendation ===
            historical_data = np.array([
                [45, 2, 0.85],  # t-4
                [48, 2, 0.92],  # t-3  
                [52, 2, 1.05],  # t-2
                [55, 2, 1.15],  # t-1
                [58, 2, 1.25],  # t (current)
            ])

            # Get NN recommendation
            recommendation = self.predictor.get_scaling_recommendation(
                historical_data=historical_data,
                current_users=current_users,
                current_cores=self.cores,
                return_full_analysis=False
            )

            # Calculate base action delta from NN
            nn_recommended_cores = recommendation['recommended_cores']
            nn_action_delta = nn_recommended_cores - self.cores
            
            # === PHASE 2: Calculate PID compensation based on previous control effectiveness ===
            pid_compensation = 0
            if self.previous_actual_rt is not None:
                # Error between actual RT and target RT from previous control action
                control_error = self.previous_actual_rt - self.setpoint
                pid_compensation = self._calculate_pid_compensation(control_error)

                if(pid_compensation<0):
                    #voglio usare il pid solo nel caso di under provisoning
                    pid_compensation=0
                
                if self.enable_log:
                    print(f"Control-Feedback: actual_rt={self.previous_actual_rt:.3f}, "
                          f"target_rt={self.setpoint:.3f}, "
                          f"control_error={control_error:.3f}, pid_comp={pid_compensation:.3f}")

            # === PHASE 3: Store data for GP training ===
            if pid_compensation != 0:  # Only store successful PID compensations (under-provisioning cases)
                gp_input = np.array([nn_action_delta, current_users, current_response_time])
                self.gp_data_buffer.append((gp_input, pid_compensation))
                
                if self.enable_log:
                    print(f"GP-Data-Store: input=[{nn_action_delta:.2f}, {current_users}, {current_response_time:.3f}] → pid_comp={pid_compensation:.3f}")

            # === PHASE 4: Get GP compensation if available ===
            gp_compensation = 0
            if len(self.gp_data_buffer) >= self._GP_MIN_SAMPLES and self.step_cnt >= self._GP_TRAIN_START:
                gp_input = np.array([nn_action_delta, current_users, current_response_time])
                X_pred = gp_input.reshape(1, -1)
                gp_compensation = self._get_gp_prediction(X_pred, current_response_time)
                
                if self.enable_log:
                    print(f"GP-Prediction: input=[{nn_action_delta:.2f}, {current_users}, {current_response_time:.3f}] → gp_comp={gp_compensation:.3f}")

            # === PHASE 5: Combine NN + PID + GP ===
            if self.step_cnt >= self._GP_TRAIN_START and len(self.gp_data_buffer) >= self._GP_MIN_SAMPLES:
                # Use GP prediction instead of PID (GP has learned PID patterns)
                final_delta = nn_action_delta + max(0, gp_compensation)  # Only positive GP compensation
                compensation_source = "GP"
                actual_compensation = max(0, gp_compensation)
            else:
                # Use direct PID compensation
                final_delta = nn_action_delta + pid_compensation
                compensation_source = "PID"
                actual_compensation = pid_compensation

            # Apply bounds
            new_cores = self.cores + final_delta
            new_cores = max(1, min(new_cores, self.predictor.max_cores))

            # === PHASE 6: Update GP training ===
            self.gp_train_counter += 1
            if self.gp_train_counter >= self._GP_TRAIN_FREQ and len(self.gp_data_buffer) >= self._GP_MIN_SAMPLES:
                self._train_gp()

            # === PHASE 7: Apply action ===
            old_cores = self.cores
            self.cores = new_cores
            self.step_cnt += 1

            # === PHASE 8: Store current actual RT for next step control feedback ===
            self.previous_actual_rt = current_response_time

            # === PHASE 9: Logging ===
            rt_ratio = current_response_time / self.setpoint
            gp_status = f"GP:{len(self.gp_data_buffer)}/{self._GP_MIN_SAMPLES}" if len(self.gp_data_buffer) < self._GP_MIN_SAMPLES else "GP:ACTIVE"
            
            log_line = (f"t={t:.1f}: Users={current_users}, RT={current_response_time:.3f}s "
                       f"(ratio={rt_ratio:.2f}), Target={self.setpoint:.3f}s, "
                       f"Cores: {old_cores} -> {self.cores} "
                       f"(NN:{nn_action_delta:.2f} {compensation_source}:{actual_compensation:.2f}) {gp_status}")
            
            print(log_line)
            
            if self.enable_log:
                with open(self.main_log_path, "a") as f:
                    f.write(log_line + "\n")

        except Exception as e:
            print(f"ERROR t={t}: Control error: {e}")
            import traceback
            traceback.print_exc()

    def reset(self):
        """Reset controller state."""
        super().reset()
        self.integral_error = 0
        self.gp_data_buffer.clear()
        self.gp_train_counter = 0
        self.step_cnt = 0
        
        # Reset control effectiveness tracking
        self.previous_actual_rt = None

    def set_pid_params(self, kp, ki):
        """Update PID parameters."""
        self.kp = kp
        self.ki = ki

    def get_gp_stats(self):
        """Get GP training statistics for monitoring."""
        return {
            'gp_buffer_size': len(self.gp_data_buffer),
            'gp_trained': len(self.gp_data_buffer) >= self._GP_MIN_SAMPLES,
            'gp_active': self.step_cnt >= self._GP_TRAIN_START,
            'step_count': self.step_cnt,
            'integral_error': self.integral_error
        }

    def __str__(self):
        base_str = super().__str__()
        return (f"{base_str} kp={self.kp} ki={self.ki} "
                f"gp_samples={len(self.gp_data_buffer)}/{self._GP_MIN_SAMPLES}")


# Test function for command line usage
def main():
    """Test the GP-enhanced Neural Network controller."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GP-enhanced Neural Network Autoscaler")
    parser.add_argument('--model-path', default='models/model_response_time.h5',
                       help="Path to neural network model")
    parser.add_argument('--target-rt', type=float, default=0.4,
                       help="Target response time")
    parser.add_argument('--kp', type=float, default=100, help="PID proportional gain")
    parser.add_argument('--ki', type=float, default=0.01, help="PID integral gain")
    
    args = parser.parse_args()
    
    # Create controller instance
    controller = GPintellegentHPA(
        period=30,
        init_cores=2,
        st=0.8,
        name="GP-NN-Test",
        model_path=args.model_path,
        kp=args.kp,
        ki=args.ki
    )
    
    # Set SLA
    controller.setSLA(args.target_rt)
    
    print(f"Initialized GP-enhanced Neural Network controller:")
    print(f"  Target RT: {args.target_rt}s")
    print(f"  PID params: kp={args.kp}, ki={args.ki}")
    print(f"  GP params: min_samples={controller._GP_MIN_SAMPLES}, "
          f"train_start={controller._GP_TRAIN_START}")
    print(f"  Controller: {controller}")
    
    # Show GP stats
    stats = controller.get_gp_stats()
    print(f"  GP Stats: {stats}")


if __name__ == "__main__":
    main() 