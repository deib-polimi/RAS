#!/usr/bin/env python3
"""
Data Collection Controller

Stub controller specifically designed for collecting training data.
Randomly varies the number of cores to stimulate the system under different
conditions and generate diverse datasets for neural network training.

Compatible with EvolutionStrategy_simple.py and generate_training_data.py
"""

import random
import time
import os
import numpy as np

# Handle both relative and absolute imports for standalone execution
try:
    from .controller import Controller
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from controllers.controller import Controller


class DataCollectionController(Controller):
    """
    Stub controller for training data collection.
    
    This controller randomly allocates cores within specified bounds to:
    1. Generate diverse system responses under varying loads
    2. Collect data points with different user/core combinations
    3. Provide ground truth for neural network training
    
    The controller logs detailed metrics in CSV format compatible with
    the training pipeline.
    """
    
    def __init__(self, period, init_cores, min_cores=1, max_cores=100, 
                 st=0.8, name=None, enable_log=True, log_dir="./logs",
                 change_frequency=5, exploration_strategy="random"):
        """
        Initialize the Data Collection Controller.
        
        Args:
            period: Control period (how often to make decisions)
            init_cores: Initial number of cores
            min_cores: Minimum cores to allocate
            max_cores: Maximum cores to allocate
            st: Safety threshold (compatibility)
            name: Controller name
            enable_log: Enable detailed CSV logging
            log_dir: Directory for log files
            change_frequency: How often to change cores (every N periods)
            exploration_strategy: Strategy for core selection ("random", "sweep", "adaptive")
        """
        super().__init__(period, init_cores, st, name or "DataCollection")
        
        self.min_cores = min_cores
        self.max_cores = max_cores
        self.enable_log = enable_log
        self.change_frequency = change_frequency
        self.exploration_strategy = exploration_strategy
        
        # Internal state
        self.step_count = 0
        self.last_change_step = 0
        self.current_exploration_target = init_cores
        
        # For sweep strategy
        self.sweep_direction = 1
        self.sweep_step_size = max(1, (max_cores - min_cores) // 20)
        
        # For adaptive strategy
        self.performance_history = []
        self.recent_rt_samples = []
        
        # Setup logging
        if self.enable_log:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = int(time.time())
            self.log_path = os.path.join(log_dir, f"datacollection-{timestamp}.csv")
            self._write_log_header()
        
        print(f"🎯 Data Collection Controller Initialized")
        print(f"   Strategy: {exploration_strategy}")
        print(f"   Core Range: [{min_cores}, {max_cores}]")
        print(f"   Change Frequency: every {change_frequency} periods")
        if self.enable_log:
            print(f"   Logging to: {self.log_path}")

    def _write_log_header(self):
        """Write CSV header for training data compatibility."""
        if self.enable_log:
            with open(self.log_path, "w") as f:
                f.write("timestamp,step,users,cores,response_time,target_rt,sla_violation,exploration_target\n")

    def _should_change_cores(self):
        """Determine if cores should be changed based on frequency."""
        return (self.step_count - self.last_change_step) >= self.change_frequency

    def _get_random_cores(self):
        """Get random core allocation within bounds."""
        return random.randint(self.min_cores, self.max_cores)

    def _get_sweep_cores(self):
        """Get cores using systematic sweep strategy."""
        if self.current_exploration_target >= self.max_cores:
            self.sweep_direction = -1
        elif self.current_exploration_target <= self.min_cores:
            self.sweep_direction = 1
        
        next_cores = self.current_exploration_target + (self.sweep_direction * self.sweep_step_size)
        return max(self.min_cores, min(self.max_cores, next_cores))

    def _get_adaptive_cores(self):
        """Get cores using adaptive strategy based on recent performance."""
        if len(self.performance_history) < 3:
            return self._get_random_cores()
        
        # Analyze recent performance to find interesting regions
        recent_violations = sum(1 for rt, target in self.performance_history[-5:] if rt > target)
        
        if recent_violations > 3:
            # High violation rate, try higher cores
            return min(self.max_cores, self.current_exploration_target + random.randint(5, 15))
        elif recent_violations == 0:
            # No violations, try lower cores for cost efficiency exploration
            return max(self.min_cores, self.current_exploration_target - random.randint(1, 10))
        else:
            # Mixed performance, random exploration around current value
            delta = random.randint(-10, 10)
            return max(self.min_cores, min(self.max_cores, self.current_exploration_target + delta))

    def _get_next_core_allocation(self):
        """Get next core allocation based on exploration strategy."""
        if self.exploration_strategy == "random":
            return self._get_random_cores()
        elif self.exploration_strategy == "sweep":
            return self._get_sweep_cores()
        elif self.exploration_strategy == "adaptive":
            return self._get_adaptive_cores()
        else:
            return self._get_random_cores()

    def control(self, t):
        """
        Main control logic: randomly allocate cores and log system response.
        
        This method:
        1. Periodically changes core allocation based on exploration strategy
        2. Collects current system metrics (users, response time)
        3. Logs detailed data for training pipeline
        """
        if not self.monitoring or not self.generator:
            return

        try:
            # Get current system metrics
            current_users = self.generator.f(t) if self.generator else 0
            current_rt = self.monitoring.getRT() if hasattr(self.monitoring, 'getRT') else 0.0
            
            # Handle potential list returns
            if isinstance(current_rt, list):
                current_rt = current_rt[0] if current_rt else 0.0
            if isinstance(current_users, list):
                current_users = current_users[0] if current_users else 0
                
            # Convert to basic types
            current_rt = float(current_rt)
            current_users = int(current_users)
            
            # Decide if we should change cores
            if self._should_change_cores() or self.step_count == 0:
                new_target = self._get_next_core_allocation()
                self.current_exploration_target = new_target
                self.cores = new_target
                self.last_change_step = self.step_count
                
                print(f"📊 t={t:.1f}: Changed cores to {self.cores} (strategy: {self.exploration_strategy})")

            # Update performance history for adaptive strategy
            if hasattr(self, 'setpoint') and self.setpoint:
                target_rt = self.setpoint
                sla_violation = 1 if current_rt > target_rt else 0
                self.performance_history.append((current_rt, target_rt))
                
                # Keep only recent history
                if len(self.performance_history) > 20:
                    self.performance_history.pop(0)
            else:
                target_rt = 0.4  # Default target
                sla_violation = 1 if current_rt > target_rt else 0

            # Log data point for training
            self._log_training_data(t, current_users, current_rt, target_rt, sla_violation)
            
            self.step_count += 1

        except Exception as e:
            print(f"ERROR in DataCollectionController t={t}: {e}")
            import traceback
            traceback.print_exc()

    def _log_training_data(self, t, users, response_time, target_rt, sla_violation):
        """Log training data in CSV format."""
        if self.enable_log:
            try:
                with open(self.log_path, "a") as f:
                    f.write(f"{t:.1f},{self.step_count},{users},{self.cores},{response_time:.4f},"
                           f"{target_rt:.4f},{sla_violation},{self.current_exploration_target}\n")
            except Exception as e:
                print(f"Logging error: {e}")

        # Console output for monitoring
        rt_ratio = response_time / target_rt if target_rt > 0 else 0
        violation_status = "VIOLATION" if sla_violation else "OK"
        
        print(f"📈 t={t:.1f}: Users={users}, Cores={self.cores}, "
              f"RT={response_time:.3f}s (ratio={rt_ratio:.2f}), "
              f"Target={target_rt:.3f}s, Status={violation_status}")

    def reset(self):
        """Reset controller state."""
        super().reset()
        self.step_count = 0
        self.last_change_step = 0
        self.current_exploration_target = self.init_cores
        self.performance_history.clear()
        self.recent_rt_samples.clear()
        
        if self.exploration_strategy == "sweep":
            self.sweep_direction = 1

    def set_exploration_strategy(self, strategy):
        """Change exploration strategy during runtime."""
        valid_strategies = ["random", "sweep", "adaptive"]
        if strategy in valid_strategies:
            self.exploration_strategy = strategy
            print(f"🔄 Changed exploration strategy to: {strategy}")
        else:
            print(f"❌ Invalid strategy. Valid options: {valid_strategies}")

    def set_change_frequency(self, frequency):
        """Change the frequency of core allocation changes."""
        self.change_frequency = max(1, frequency)
        print(f"🔄 Changed frequency to: every {self.change_frequency} periods")

    def get_collection_stats(self):
        """Get statistics about data collection progress."""
        return {
            'total_steps': self.step_count,
            'exploration_strategy': self.exploration_strategy,
            'current_cores': self.cores,
            'exploration_target': self.current_exploration_target,
            'core_range': (self.min_cores, self.max_cores),
            'change_frequency': self.change_frequency,
            'performance_samples': len(self.performance_history),
            'log_enabled': self.enable_log,
            'log_path': getattr(self, 'log_path', None)
        }

    def __str__(self):
        base_str = super().__str__()
        strategy_info = f"strategy={self.exploration_strategy}"
        range_info = f"range=[{self.min_cores}-{self.max_cores}]"
        freq_info = f"freq={self.change_frequency}"
        return f"{base_str} {strategy_info} {range_info} {freq_info}"


# Test function for command line usage
def main():
    """Test the Data Collection controller."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Collection Controller")
    parser.add_argument('--min-cores', type=int, default=1, help="Minimum cores")
    parser.add_argument('--max-cores', type=int, default=50, help="Maximum cores")
    parser.add_argument('--strategy', choices=['random', 'sweep', 'adaptive'], 
                       default='random', help="Exploration strategy")
    parser.add_argument('--frequency', type=int, default=5, 
                       help="Change frequency (periods)")
    parser.add_argument('--target-rt', type=float, default=0.4, 
                       help="Target response time")
    
    args = parser.parse_args()
    
    # Create controller instance
    controller = DataCollectionController(
        period=30,
        init_cores=10,
        min_cores=args.min_cores,
        max_cores=args.max_cores,
        exploration_strategy=args.strategy,
        change_frequency=args.frequency,
        name="DataCollection-Test"
    )
    
    # Set SLA
    controller.setSLA(args.target_rt)
    
    print(f"Initialized Data Collection Controller:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Core Range: [{args.min_cores}, {args.max_cores}]")
    print(f"  Change Frequency: every {args.frequency} periods")
    print(f"  Target RT: {args.target_rt}s")
    print(f"  Controller: {controller}")
    
    # Show collection stats
    stats = controller.get_collection_stats()
    print(f"  Collection Stats:")
    for key, value in stats.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main() 