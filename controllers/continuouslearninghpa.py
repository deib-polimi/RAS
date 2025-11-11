#!/usr/bin/env python3
"""
Continuous Learning HPA Controller

Controller intelligente che evolve automaticamente attraverso 3 fasi di apprendimento:

Fase 1: NN Solo + Statistical Monitoring
Fase 2: NN + PID + Raccolta Campioni per GP
Fase 3: NN + GP + Continuous Learning (con loop back alla Fase 2)

Il controller monitora continuamente le prestazioni e adatta automaticamente 
la sua strategia di controllo per gestire concept drift e cambiamenti applicativi.
"""

import os
import time
import numpy as np
from datetime import datetime
from collections import deque
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
from enum import Enum

# Handle both relative and absolute imports
try:
    from .gpintellegenthpa import GPintellegentHPA
except ImportError:
    try:
        from gpintellegenthpa import GPintellegentHPA
    except ImportError:
        # Fallback to direct import
        import sys
        sys.path.append('.')
        from gpintellegenthpa import GPintellegentHPA


class ControlPhase(Enum):
    """Enum per le fasi del controller"""
    NN_MONITORING = "NN_Monitoring"          # Fase 1: Solo NN + monitoring
    NN_PID_LEARNING = "NN_PID_Learning"      # Fase 2: NN + PID + raccolta dati
    NN_GP_PREDICTION = "NN_GP_Prediction"    # Fase 3: NN + GP predittivo


class ContinuousLearningHPA(GPintellegentHPA):
    """
    Controller con apprendimento continuo che evolve automaticamente tra 3 fasi:
    
    Fase 1 (NN_MONITORING): Solo Neural Network con monitoring statistico
    Fase 2 (NN_PID_LEARNING): NN + PID con raccolta campioni per GP training
    Fase 3 (NN_GP_PREDICTION): NN + GP predittivo con continuous learning
    
    Il sistema monitora continuamente le prestazioni e switcha automaticamente
    tra le fasi per adattarsi a concept drift e cambiamenti applicativi.
    """

    # Parametri per il monitoring statistico e transitions
    _BASELINE_WINDOW_SIZE = 100      # Finestra per calcolare baseline performance
    _RECENT_WINDOW_SIZE = 30         # Finestra per performance recenti  
    _DEGRADATION_THRESHOLD = 1.3     # Soglia per rilevare degradazione (1.3 = 30% worse)
    _STABILITY_THRESHOLD = 1.1       # Soglia per considerare stabile (1.1 = 10% worse max)
    _MIN_MONITORING_STEPS = 50       # Minimo step prima di valutare transitions
    
    # Parametri per GP retraining
    _GP_RETRAIN_THRESHOLD = 100      # Nuovi campioni prima di retrain GP
    _GP_PERFORMANCE_CHECK_FREQ = 25  # Frequenza check performance GP
    
    def __init__(self, period, init_cores, st=0.8, name=None,
                 model_path: str = 'models/model_response_time.h5',
                 max_cores: int = 300, window_size: int = 5,
                 kp=100, ki=0.01, enable_log=True, log_dir="./logs"):
        """
        Initialize the Continuous Learning HPA Controller.
        
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
        super().__init__(period, init_cores, st, name, model_path, max_cores, 
                        window_size, kp, ki, enable_log, log_dir)

        # Fase corrente del controller
        self.current_phase = ControlPhase.NN_MONITORING
        
        # Buffers per statistical monitoring
        self.baseline_errors = deque(maxlen=self._BASELINE_WINDOW_SIZE)
        self.recent_errors = deque(maxlen=self._RECENT_WINDOW_SIZE)
        
        # Contatori per transitions
        self.steps_in_current_phase = 0
        self.total_steps = 0
        self.last_performance_check = 0
        
        # Statistiche per decision making
        self.baseline_p50_error = None
        self.recent_p95_error = None
        
        # GP learning state
        self.new_samples_since_retrain = 0
        self.gp_retrain_count = 0
        
        # Phase transition history per debugging
        self.phase_history = []
        
        # Override logging paths per includere phase info
        if self.enable_log:
            timestamp = time.time()
            self.main_log_path = os.path.join(log_dir, f"continuous-learning-{timestamp:.0f}.log")
            self.phase_log_path = os.path.join(log_dir, f"continuous-learning-phases-{timestamp:.0f}.log")
            self.stats_log_path = os.path.join(log_dir, f"continuous-learning-stats-{timestamp:.0f}.log")

        print(f"🔄 Initialized Continuous Learning HPA Controller")
        print(f"   Initial Phase: {self.current_phase.value}")
        print(f"   Monitoring Windows: baseline={self._BASELINE_WINDOW_SIZE}, recent={self._RECENT_WINDOW_SIZE}")
        print(f"   Degradation Threshold: {self._DEGRADATION_THRESHOLD:.1f}x")

    def _calculate_control_error(self, actual_rt, target_rt):
        """
        Calculate normalized control error for statistical monitoring.
        
        Returns:
            Normalized error (0 = perfect, 1 = 100% over target)
        """
        if target_rt <= 0:
            return 0
        return max(0, (actual_rt - target_rt) / target_rt)

    def _update_statistical_monitoring(self, control_error):
        """
        Update statistical monitoring buffers and calculate performance metrics.
        
        Args:
            control_error: Normalized control error (0 = perfect)
        """
        # Add to both baseline and recent
        self.baseline_errors.append(control_error)
        self.recent_errors.append(control_error)
        
        # Calculate statistics if we have enough data
        if len(self.baseline_errors) >= 20:  # Need minimum data for percentiles
            self.baseline_p50_error = np.percentile(list(self.baseline_errors), 50)
        
        if len(self.recent_errors) >= 10:   # Need minimum data for percentiles
            self.recent_p95_error = np.percentile(list(self.recent_errors), 95)

    def _should_activate_pid_learning(self):
        """
        Determine if we should switch to PID learning phase.
        
        Returns:
            bool: True if degradation detected and should switch to PID learning
        """
        if self.total_steps < self._MIN_MONITORING_STEPS:
            return False
            
        if self.baseline_p50_error is None or self.recent_p95_error is None:
            return False
            
        # Check if recent 95th percentile is significantly worse than baseline 50th percentile
        degradation_ratio = self.recent_p95_error / (self.baseline_p50_error + 1e-6)
        
        return degradation_ratio > self._DEGRADATION_THRESHOLD

    def _should_activate_gp_prediction(self):
        """
        Determine if we should switch to GP prediction phase.
        
        Returns:
            bool: True if we have enough GP samples and performance is stable
        """
        # Need enough GP samples
        if len(self.gp_data_buffer) < self._GP_MIN_SAMPLES:
            return False
            
        # Check if we've been in PID learning long enough
        if self.steps_in_current_phase < 50:  # Minimum time in PID phase
            return False
            
        # Check if recent performance is stable (not too bad)
        if self.baseline_p50_error is None or self.recent_p95_error is None:
            return False
            
        stability_ratio = self.recent_p95_error / (self.baseline_p50_error + 1e-6)
        
        return stability_ratio <= self._STABILITY_THRESHOLD

    def _should_retrain_gp(self):
        """
        Determine if GP should be retrained with new samples.
        
        Returns:
            bool: True if GP should be retrained
        """
        return (self.new_samples_since_retrain >= self._GP_RETRAIN_THRESHOLD and
                len(self.gp_data_buffer) >= self._GP_MIN_SAMPLES)

    def _perform_phase_transition(self, new_phase: ControlPhase, reason: str):
        """
        Perform transition to a new control phase.
        
        Args:
            new_phase: Target phase to transition to
            reason: Reason for the transition (for logging)
        """
        old_phase = self.current_phase
        self.current_phase = new_phase
        
        # Reset phase-specific counters
        self.steps_in_current_phase = 0
        
        # Log transition
        transition = {
            'timestamp': time.time(),
            'total_steps': self.total_steps,
            'from_phase': old_phase.value,
            'to_phase': new_phase.value,
            'reason': reason,
            'gp_samples': len(self.gp_data_buffer),
            'baseline_p50': self.baseline_p50_error,
            'recent_p95': self.recent_p95_error
        }
        
        self.phase_history.append(transition)
        
        print(f"🔄 PHASE TRANSITION [{self.total_steps}]: {old_phase.value} → {new_phase.value}")
        print(f"   Reason: {reason}")
        print(f"   GP Samples: {len(self.gp_data_buffer)}")
        
        if self.enable_log:
            with open(self.phase_log_path, "a") as f:
                f.write(f"t={time.time():.1f}: {old_phase.value} → {new_phase.value} | {reason}\n")

    def _get_compensation_based_on_phase(self, nn_action_delta, current_users, current_response_time, pid_compensation):
        """
        Get compensation based on current control phase.
        
        Returns:
            (compensation_value, compensation_source)
        """
        if self.current_phase == ControlPhase.NN_MONITORING:
            # Fase 1: Solo NN, nessuna compensazione
            return 0, "NONE"
            
        elif self.current_phase == ControlPhase.NN_PID_LEARNING:
            # Fase 2: Usa PID e colleziona dati per GP
            if pid_compensation > 0:  # Solo under-provisioning cases
                # Store data for GP learning
                gp_input = np.array([nn_action_delta, current_users, current_response_time])
                self.gp_data_buffer.append((gp_input, pid_compensation))
                self.new_samples_since_retrain += 1
                
                if self.enable_log:
                    print(f"GP-Data-Collected: [{nn_action_delta:.2f}, {current_users}, {current_response_time:.3f}] → {pid_compensation:.3f}")
            
            return pid_compensation, "PID"
            
        elif self.current_phase == ControlPhase.NN_GP_PREDICTION:
            # Fase 3: Usa GP per predire compensazioni
            if len(self.gp_data_buffer) >= self._GP_MIN_SAMPLES:
                gp_input = np.array([nn_action_delta, current_users, current_response_time])
                X_pred = gp_input.reshape(1, -1)
                gp_compensation = self._get_gp_prediction(X_pred, current_response_time)
                
                # Continue collecting PID data for future GP retraining
                if pid_compensation > 0:
                    gp_input_for_storage = np.array([nn_action_delta, current_users, current_response_time])
                    self.gp_data_buffer.append((gp_input_for_storage, pid_compensation))
                    self.new_samples_since_retrain += 1
                
                return max(0, gp_compensation), "GP"
            else:
                # Fallback to PID if GP not ready
                return pid_compensation, "PID-FALLBACK"
                
        return 0, "UNKNOWN"

    def control(self, t):
        """
        Enhanced control logic with continuous learning and automatic phase transitions.
        
        Architecture Flow:
        1. Execute base control (NN + compensation based on current phase)
        2. Update statistical monitoring 
        3. Evaluate phase transitions
        4. Log comprehensive metrics
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
            
            # === PHASE 2: Calculate PID compensation (always calculated for monitoring) ===
            pid_compensation = 0
            if self.previous_actual_rt is not None:
                control_error = self.previous_actual_rt - self.setpoint
                pid_compensation = self._calculate_pid_compensation(control_error)
                
                if pid_compensation < 0:
                    pid_compensation = 0  # Only positive compensations
            
            # === PHASE 3: Get compensation based on current phase ===
            compensation, compensation_source = self._get_compensation_based_on_phase(
                nn_action_delta, current_users, current_response_time, pid_compensation)
            
            # === PHASE 4: Combine NN + compensation ===
            final_delta = nn_action_delta + compensation
            new_cores = max(1, min(self.cores + final_delta, self.predictor.max_cores))
            
            # === PHASE 5: Update statistical monitoring ===
            control_error = self._calculate_control_error(current_response_time, self.setpoint)
            self._update_statistical_monitoring(control_error)
            
            # === PHASE 6: Evaluate phase transitions ===
            self._evaluate_phase_transitions()
            
            # === PHASE 7: Handle GP retraining if needed ===
            if self._should_retrain_gp():
                self._train_gp()
                self.new_samples_since_retrain = 0
                self.gp_retrain_count += 1
                
                if self.enable_log:
                    print(f"🔄 GP RETRAINED #{self.gp_retrain_count} with {len(self.gp_data_buffer)} samples")

            # === PHASE 8: Apply action ===
            old_cores = self.cores
            self.cores = new_cores
            self.steps_in_current_phase += 1
            self.total_steps += 1

            # === PHASE 9: Store current actual RT for next step feedback ===
            self.previous_actual_rt = current_response_time

            # === PHASE 10: Comprehensive logging ===
            self._log_control_action(t, current_users, current_response_time, old_cores,
                                   nn_action_delta, compensation, compensation_source, control_error)

        except Exception as e:
            print(f"ERROR t={t}: Control error: {e}")
            import traceback
            traceback.print_exc()

    def _evaluate_phase_transitions(self):
        """Evaluate and perform phase transitions based on current conditions."""
        
        if self.current_phase == ControlPhase.NN_MONITORING:
            # Check if we should start PID learning due to degradation
            if self._should_activate_pid_learning():
                self._perform_phase_transition(
                    ControlPhase.NN_PID_LEARNING,
                    f"Performance degradation detected (recent_p95={self.recent_p95_error:.3f} vs baseline_p50={self.baseline_p50_error:.3f})"
                )
                
        elif self.current_phase == ControlPhase.NN_PID_LEARNING:
            # Check if we can switch to GP prediction
            if self._should_activate_gp_prediction():
                self._perform_phase_transition(
                    ControlPhase.NN_GP_PREDICTION,
                    f"Sufficient GP samples ({len(self.gp_data_buffer)}) and stable performance"
                )
                
        elif self.current_phase == ControlPhase.NN_GP_PREDICTION:
            # Check if we should go back to PID learning due to degradation
            if self._should_activate_pid_learning():
                self._perform_phase_transition(
                    ControlPhase.NN_PID_LEARNING,
                    f"GP performance degradation detected (recent_p95={self.recent_p95_error:.3f} vs baseline_p50={self.baseline_p50_error:.3f})"
                )

    def _log_control_action(self, t, current_users, current_response_time, old_cores,
                          nn_action_delta, compensation, compensation_source, control_error):
        """Log comprehensive control action with phase and statistical info."""
        
        rt_ratio = current_response_time / self.setpoint
        phase_info = f"{self.current_phase.value}[{self.steps_in_current_phase}]"
        
        # Statistical info
        stats_info = ""
        if self.baseline_p50_error is not None and self.recent_p95_error is not None:
            degradation_ratio = self.recent_p95_error / (self.baseline_p50_error + 1e-6)
            stats_info = f"stats[b50={self.baseline_p50_error:.3f},r95={self.recent_p95_error:.3f},ratio={degradation_ratio:.2f}]"
        
        # GP info
        gp_info = f"GP[{len(self.gp_data_buffer)}/{self._GP_MIN_SAMPLES},retrains={self.gp_retrain_count}]"
        
        log_line = (f"t={t:.1f}: Users={current_users}, RT={current_response_time:.3f}s "
                   f"(ratio={rt_ratio:.2f},err={control_error:.3f}), Target={self.setpoint:.3f}s, "
                   f"Cores: {old_cores} -> {self.cores} "
                   f"(NN:{nn_action_delta:.2f} {compensation_source}:{compensation:.2f}) "
                   f"{phase_info} {stats_info} {gp_info}")
        
        print(log_line)
        
        if self.enable_log:
            with open(self.main_log_path, "a") as f:
                f.write(log_line + "\n")
            
            # Log detailed statistics
            with open(self.stats_log_path, "a") as f:
                f.write(f"t={t:.1f},phase={self.current_phase.value},rt={current_response_time:.3f},"
                       f"error={control_error:.3f},baseline_p50={self.baseline_p50_error or 0:.3f},"
                       f"recent_p95={self.recent_p95_error or 0:.3f},cores={self.cores},"
                       f"gp_samples={len(self.gp_data_buffer)}\n")

    def reset(self):
        """Reset controller state including continuous learning components."""
        super().reset()
        
        # Reset phase and monitoring
        self.current_phase = ControlPhase.NN_MONITORING
        self.baseline_errors.clear()
        self.recent_errors.clear()
        
        # Reset counters
        self.steps_in_current_phase = 0
        self.total_steps = 0
        self.last_performance_check = 0
        self.new_samples_since_retrain = 0
        self.gp_retrain_count = 0
        
        # Reset statistics
        self.baseline_p50_error = None
        self.recent_p95_error = None
        
        # Reset history
        self.phase_history.clear()

    def get_continuous_learning_stats(self):
        """Get comprehensive statistics for monitoring and debugging."""
        return {
            'current_phase': self.current_phase.value,
            'steps_in_phase': self.steps_in_current_phase,
            'total_steps': self.total_steps,
            'baseline_p50_error': self.baseline_p50_error,
            'recent_p95_error': self.recent_p95_error,
            'degradation_ratio': (self.recent_p95_error / (self.baseline_p50_error + 1e-6)) 
                                if (self.baseline_p50_error and self.recent_p95_error) else None,
            'gp_samples': len(self.gp_data_buffer),
            'gp_retrains': self.gp_retrain_count,
            'new_samples_since_retrain': self.new_samples_since_retrain,
            'phase_transitions': len(self.phase_history),
            'last_transition': self.phase_history[-1] if self.phase_history else None
        }

    def set_degradation_threshold(self, threshold: float):
        """Set the degradation threshold for phase transitions."""
        self._DEGRADATION_THRESHOLD = threshold
        
    def set_stability_threshold(self, threshold: float):
        """Set the stability threshold for phase transitions."""
        self._STABILITY_THRESHOLD = threshold

    def __str__(self):
        base_str = super().__str__()
        phase_info = f"phase={self.current_phase.value}[{self.steps_in_current_phase}]"
        stats_info = f"transitions={len(self.phase_history)}"
        return f"{base_str} {phase_info} {stats_info}"


# Test function for command line usage
def main():
    """Test the Continuous Learning HPA controller."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Learning HPA Controller")
    parser.add_argument('--model-path', default='models/model_response_time.h5',
                       help="Path to neural network model")
    parser.add_argument('--target-rt', type=float, default=0.4,
                       help="Target response time")
    parser.add_argument('--kp', type=float, default=100, help="PID proportional gain")
    parser.add_argument('--ki', type=float, default=0.01, help="PID integral gain")
    parser.add_argument('--degradation-threshold', type=float, default=1.3,
                       help="Degradation threshold for phase transitions")
    
    args = parser.parse_args()
    
    # Create controller instance
    controller = ContinuousLearningHPA(
        period=30,
        init_cores=2,
        st=0.8,
        name="ContinuousLearning-Test",
        model_path=args.model_path,
        kp=args.kp,
        ki=args.ki
    )
    
    # Set SLA and thresholds
    controller.setSLA(args.target_rt)
    controller.set_degradation_threshold(args.degradation_threshold)
    
    print(f"Initialized Continuous Learning HPA controller:")
    print(f"  Target RT: {args.target_rt}s")
    print(f"  PID params: kp={args.kp}, ki={args.ki}")
    print(f"  Degradation threshold: {args.degradation_threshold}")
    print(f"  Controller: {controller}")
    
    # Show comprehensive stats
    stats = controller.get_continuous_learning_stats()
    print(f"  Continuous Learning Stats:")
    for key, value in stats.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main() 