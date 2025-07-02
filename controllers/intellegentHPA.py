#!/usr/bin/env python3
"""
Autoscaling Predictor - Uses the trained Double Tower model to predict response times
and make intelligent scaling decisions based on a target response time threshold.

Updated to work with 3-feature model: users, cores, response_time
Organized as a class with singleton model loading and simplified API

Now includes Controller interface compliance for integration with simulation frameworks.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from pathlib import Path
import argparse
import json
from typing import List, Tuple, Dict, Optional
from collections import deque

# Handle both relative and absolute imports
try:
    from .controller import Controller
except ImportError:
    # When run as main script, try absolute import
    try:
        from controller import Controller
    except ImportError:
        # If that fails too, create a minimal Controller class for standalone use
        class Controller:
            def __init__(self, period, init_cores, st=0.8, name=None):
                self.period = period
                self.init_cores = init_cores
                self.cores = self.init_cores
                self.st = st
                self.name = name if name else type(self).__name__
                
            def setName(self, name):
                self.name = name

            def setSLA(self, sla):
                self.sla = sla
                self.setpoint = sla*self.st

            def setMonitoring(self, monitoring):
                self.monitoring = monitoring
            
            def setGenerator(self, generator):
                self.generator = generator

            def tick(self, t):
                if not t:
                    self.reset()

                if t and not (t % self.period):
                    self.control(t)

                return self.cores

            def control(self, t):
                pass

            def reset(self):
                self.cores = self.init_cores

            def __str__(self):
                return "%s - period: %d init_cores: %.2f" % (self.name, self.period, self.init_cores)

from pathlib import Path

class AutoscalingPredictor:
    # Class variable to store the loaded model (singleton pattern)
    _loaded_models = {}
    
    def __init__(self, model_path: str = 'models/model_response_time.h5', 
                 target_response_time: float = 2.0, 
                 max_cores: int = 16,
                 window_size: int = 5):
        """
        Initialize the autoscaling predictor.
        
        Args:
            model_path: Path to the trained model
            target_response_time: Target response time threshold in seconds
            max_cores: Maximum number of cores to test (will test 1 to max_cores)
            window_size: Size of the historical data window
        """
        self.model_path = model_path
        self.target_response_time = target_response_time
        self.max_cores = max_cores
        self.window_size = window_size

        print(f"🔄 Initializing autoscaling predictor with model path: {model_path}")
        print(f"🔄 Maximum cores: {max_cores}")
        print(f"🔄 Window size: {window_size}")
        
        # Generate available cores from 1 to max_cores
        self.available_cores = list(range(1, max_cores + 1))
        
        # Load model using singleton pattern
        self.model = self._load_model_singleton(model_path)
        
    @classmethod
    def _load_model_singleton(cls, model_path: str):
        """
        Load model using singleton pattern - only loads once per model path.
        """
        if model_path not in cls._loaded_models:
            print(f"🔄 Loading model from {model_path}...")
            cls._loaded_models[model_path] = load_model(model_path)
            print(f"✅ Model loaded and cached")
        else:
            print(f"✅ Using cached model from {model_path}")
        
        return cls._loaded_models[model_path]
    
    def predict_response_time(self, historical_data: np.ndarray, current_users: int, current_cores: int) -> float:
        """
        Predict response time given historical data and current configuration.
        
        Args:
            historical_data: Array of shape (window_size, 3) with historical metrics [users, cores, response_time]
            current_users: Current number of users
            current_cores: Current number of cores
            
        Returns:
            Predicted response time in seconds
        """
        # Ensure historical data has correct shape
        if historical_data.shape != (self.window_size, 3):
            raise ValueError(f"Historical data must have shape ({self.window_size}, 3), got {historical_data.shape}")
        
        # Create a copy of historical data to avoid modifying the original
        modified_historical_data = historical_data.copy()
        
        # Update the last entry (current timestep) with the new configuration
        modified_historical_data[-1, 0] = current_users  # Update users
        modified_historical_data[-1, 1] = current_cores  # Update cores
        
        # Prepare recurrent input (modified historical data)
        X_recurrent = modified_historical_data.reshape(1, self.window_size, 3)
        
        # Prepare feedforward input (current state: users, cores) 
        X_feedforward = np.array([[current_users, current_cores]])
        
        # Make prediction
        prediction = self.model.predict([X_recurrent, X_feedforward], verbose=0)
        
        return float(prediction[0][0])
    
    def find_optimal_cores(self, historical_data: np.ndarray, current_users: int, 
                          current_cores: int) -> Dict:
        """
        Find the optimal number of cores to meet the target response time.
        
        Args:
            historical_data: Historical metrics data (shape: window_size, 3)
            current_users: Current number of users
            current_cores: Current number of cores
            
        Returns:
            Dictionary with scaling decision and analysis
        """
        # First, predict with current configuration
        current_prediction = self.predict_response_time(historical_data, current_users, current_cores)
        
        result = {
            'current_users': current_users,
            'current_cores': current_cores,
            'current_predicted_response_time': current_prediction,
            'target_response_time': self.target_response_time,
            'meets_target': current_prediction <= self.target_response_time,
            'scaling_needed': current_prediction > self.target_response_time,
            'predictions': {},
            'recommended_cores': current_cores,
            'scaling_action': 'none'
        }
        
        # Test all available core configurations
        for cores in self.available_cores:
            predicted_rt = self.predict_response_time(historical_data, current_users, cores)
            result['predictions'][cores] = predicted_rt
        
        # Find the minimum cores that meet the target
        #if result['scaling_needed']:
        for cores in sorted(self.available_cores):
            predicted_rt = result['predictions'][cores]
            if predicted_rt <= self.target_response_time:
                result['recommended_cores'] = cores
                result['scaling_action'] = 'scale_up' if cores > current_cores else 'scale_down'
                break
        else:
            # Even maximum cores don't meet target
            result['recommended_cores'] = max(self.available_cores)
            result['scaling_action'] = 'scale_up_max'
            result['warning'] = f"Even with {max(self.available_cores)} cores, target cannot be met"
        
        return result
    
    def get_scaling_recommendation(self, historical_data: np.ndarray, 
                                 current_users: int, current_cores: int,
                                 return_full_analysis: bool = False) -> Dict:
        """
        Main API function for getting scaling recommendations.
        This is the function to use when calling from other Python modules.
        
        Args:
            historical_data: Historical metrics data (shape: window_size, 3)
            current_users: Current number of users
            current_cores: Current number of cores
            return_full_analysis: If True, returns all predictions; if False, returns only the recommendation
            
        Returns:
            Dictionary with scaling recommendation and optionally full analysis
        """
        analysis = self.find_optimal_cores(historical_data, current_users, current_cores)
        
        # Simplified output for programmatic use
        recommendation = {
            'action': analysis['scaling_action'],
            'recommended_cores': analysis['recommended_cores'],
            'current_response_time': analysis['current_predicted_response_time'],
            'expected_response_time': analysis['predictions'][analysis['recommended_cores']],
            'meets_target': analysis['meets_target'],
            'target_response_time': analysis['target_response_time']
        }
        
        if 'warning' in analysis:
            recommendation['warning'] = analysis['warning']
        
        if return_full_analysis:
            recommendation['full_analysis'] = analysis
            
        return recommendation
    
    def generate_scaling_report(self, historical_data: np.ndarray, 
                              current_users: int, current_cores: int,
                              show_all_predictions: bool = False) -> str:
        """
        Generate human-readable scaling report for command-line use.
        
        Args:
            show_all_predictions: If True, shows predictions for all core configurations
        """
        analysis = self.find_optimal_cores(historical_data, current_users, current_cores)
        
        report = f"""
🔍 AUTOSCALING ANALYSIS REPORT
═══════════════════════════════

📊 Current Configuration:
   • Users: {analysis['current_users']}
   • Cores: {analysis['current_cores']}
   • Predicted Response Time: {analysis['current_predicted_response_time']:.3f}s
   • Target Response Time: {analysis['target_response_time']}s

🎯 Target Status: {'✅ MEETING TARGET' if analysis['meets_target'] else '❌ EXCEEDING TARGET'}

🚀 RECOMMENDATION:
   • Action: {analysis['scaling_action'].upper().replace('_', ' ')}
   • Recommended Cores: {analysis['recommended_cores']}
   • Expected Response Time: {analysis['predictions'][analysis['recommended_cores']]:.3f}s"""
        
        if 'warning' in analysis:
            report += f"\n   ⚠️  WARNING: {analysis['warning']}"
        
        if show_all_predictions:
            report += "\n\n📈 Predictions for all core configurations:"
            for cores in sorted(analysis['predictions'].keys()):
                rt = analysis['predictions'][cores]
                status = "✅" if rt <= self.target_response_time else "❌"
                current_marker = " (CURRENT)" if cores == current_cores else ""
                recommended_marker = " (RECOMMENDED)" if cores == analysis['recommended_cores'] and cores != current_cores else ""
                report += f"\n   • {cores} cores: {rt:.3f}s {status}{current_marker}{recommended_marker}"
        else:
            # Show only a summary of key predictions
            key_cores = [1, current_cores, analysis['recommended_cores'], self.max_cores]
            key_cores = sorted(list(set(key_cores)))  # Remove duplicates and sort
            
            report += "\n\n📈 Key predictions:"
            for cores in key_cores:
                if cores in analysis['predictions']:
                    rt = analysis['predictions'][cores]
                    status = "✅" if rt <= self.target_response_time else "❌"
                    markers = []
                    if cores == current_cores:
                        markers.append("CURRENT")
                    if cores == analysis['recommended_cores'] and cores != current_cores:
                        markers.append("RECOMMENDED")
                    marker_str = f" ({', '.join(markers)})" if markers else ""
                    report += f"\n   • {cores} cores: {rt:.3f}s {status}{marker_str}"
        
        return report


class intellegentHPA(Controller):
    """
    Controller implementation using the Double Tower neural network for autoscaling predictions.
    
    This class implements the Controller interface while using AutoscalingPredictor internally
    for making intelligent scaling decisions based on response time predictions.
    """
    
    def __init__(self, period, init_cores, st=0.8, name=None,
                 model_path: str = 'models/model_response_time.h5',
                 max_cores: int = 300, window_size: int = 5):
        """
        Initialize the Double Tower Controller.
        
        Args:
            period: Control period (how often to make scaling decisions)
            init_cores: Initial number of cores
            st: Safety threshold multiplier for SLA (default 0.8 means target = 80% of SLA)
            name: Controller name (optional)
            model_path: Path to the trained Double Tower model
            max_cores: Maximum number of cores available for scaling
            window_size: Size of historical data window for predictions
        """
        super().__init__(period, init_cores, st, name)
        
        # Initialize the predictor with a default target (will be updated when SLA is set)
        self.predictor = AutoscalingPredictor(
            model_path=model_path,
            target_response_time=2.0,  # Default, will be updated by setSLA
            max_cores=max_cores,
            window_size=window_size
        )
        
        # Historical data buffer using deque for efficient sliding window
        self.historical_buffer = deque(maxlen=window_size)
        
        # Initialize with default values (will be populated during simulation)
        for _ in range(window_size):
            self.historical_buffer.append([50, init_cores, 1.0])  # [users, cores, response_time]
        
        # Control state
        self.monitoring = None
        self.generator = None
        self.sla = None
        self.setpoint = None
        
    def setSLA(self, sla):
        """Set the SLA target and update the predictor's target response time."""
        super().setSLA(sla)
        # Update predictor's target response time based on SLA and safety threshold
        self.predictor.target_response_time = self.setpoint
        
    def control(self, t):
        """
        Main control logic called every period.
        
        Args:
            t: Current simulation time
        """
        if not self.monitoring or not self.generator:
            # Cannot make decisions without monitoring data
            return
        
        try:
            # Get current metrics from monitoring
            current_metrics = self._get_current_metrics(t)
            if current_metrics is None:
                return
            
            current_users, current_response_time = current_metrics
            
            # Update historical buffer with current state
            #self.historical_buffer.append([current_users, self.cores, current_response_time])
            
            # Convert buffer to numpy array for prediction
            #historical_data = np.array(list(self.historical_buffer))
            
            # # TEMPORARY: Use the same sample data as command line for debugging
            # if t <= 10:  # Only for first few timesteps to test
            historical_data = np.array([
                [45, 2, 0.85],  # t-4
                [48, 2, 0.92],  # t-3
                [52, 2, 1.05],  # t-2
                [55, 2, 1.15],  # t-1
                [58, 2, 1.25],  # t (current)
            ])
            #     print(f"DEBUG t={t}: Using SAMPLE data instead of real data")
            
            # Get scaling recommendation from the predictor
            recommendation = self.predictor.get_scaling_recommendation(
                historical_data=historical_data,
                current_users=current_users,
                current_cores=self.cores,
                return_full_analysis=False
            )
            
            # Use the model's recommendation directly
            new_cores = recommendation['recommended_cores']
            
            # Print recommendation for monitoring
            rt_ratio = current_response_time / self.setpoint
            print(f"t={t}: Users={current_users}, RT={current_response_time:.3f}s (ratio={rt_ratio:.2f}), "
                  f"Target={self.setpoint:.3f}s, Action={recommendation['action']}, "
                  f"Cores: {self.cores} -> {new_cores}")
            
            # Debug: Print historical data for the first few timesteps
            # if t <= 5:
            #     historical_data_debug = np.array(list(self.historical_buffer))
            #     print(f"DEBUG t={t}: Historical data shape: {historical_data_debug.shape}")
            #     print(f"DEBUG t={t}: Historical data:\n{historical_data_debug}")
            
            # Ensure we don't exceed bounds
            new_cores = max(1, min(new_cores, self.predictor.max_cores))
            
            # Update cores
            self.cores = new_cores
                
        except Exception as e:
            # In case of any error, maintain current cores
            print(f"ERROR t={t}: Control error: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_current_metrics(self, t):
        """
        Extract current metrics from monitoring system.
        
        Returns:
            Tuple of (current_users, current_response_time) or None if data unavailable
        """
        try:
            # Use the standard monitoring interface like other controllers
            current_response_time = self.monitoring.getRT()  # or getRTp95() depending on preference
            current_users = self.monitoring.getUsers()  # This represents load/users
            
            return int(current_users), float(current_response_time)
            
        except Exception as e:
            if hasattr(self, '_debug') and self._debug:
                print(f"Error getting metrics at t={t}: {e}")
            return None
    
    def reset(self):
        """Reset the controller to initial state."""
        super().reset()
        
        # Reset historical buffer with default values
        self.historical_buffer.clear()
        for _ in range(self.predictor.window_size):
            self.historical_buffer.append([50, self.init_cores, 1.0])  # [users, cores, response_time]
    
    def set_debug(self, debug: bool):
        """Enable/disable debug logging."""
        self._debug = debug
    
    def get_prediction_analysis(self, return_full: bool = False):
        """
        Get current prediction analysis (useful for debugging/monitoring).
        
        Returns:
            Dictionary with current prediction analysis or None if not enough data
        """
        if len(self.historical_buffer) < self.predictor.window_size:
            return None
        
        try:
            historical_data = np.array(list(self.historical_buffer))
            current_users = int(historical_data[-1, 0])
            
            return self.predictor.get_scaling_recommendation(
                historical_data=historical_data,
                current_users=current_users,
                current_cores=self.cores,
                return_full_analysis=return_full
            )
        except Exception:
            return None
    
    def __str__(self):
        base_str = super().__str__()
        return f"{base_str} target_rt: {self.predictor.target_response_time:.2f}s max_cores: {self.predictor.max_cores}"


def load_sample_historical_data() -> np.ndarray:
    """
    Load sample historical data for testing.
    In a real scenario, this would come from your monitoring system.
    
    Returns array of shape (5, 3) with features: [users, cores, response_time]
    """
    # Sample historical data: [users, cores, response_time]
    # This represents the last 5 time steps
    sample_data = np.array([
        [45, 2, 0.85],  # t-4
        [48, 2, 0.92],  # t-3
        [52, 2, 1.05],  # t-2
        [55, 2, 1.15],  # t-1
        [58, 2, 1.25],  # t (current)
    ])
    return sample_data

def main():
    """
    Command-line interface for the autoscaling predictor.
    """
    parser = argparse.ArgumentParser(description="Autoscaling Predictor for Response Time Optimization")
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/model_response_time.h5',
        help="Path to the trained model"
    )
    parser.add_argument(
        '--target-rt',
        type=float,
        default=2.0,
        help="Target response time threshold in seconds"
    )
    parser.add_argument(
        '--current-users',
        type=int,
        default=60,
        help="Current number of users"
    )
    parser.add_argument(
        '--current-cores',
        type=int,
        default=2,
        help="Current number of cores"
    )
    parser.add_argument(
        '--max-cores',
        type=int,
        default=16,
        help="Maximum number of cores to test (will test 1 to max-cores)"
    )
    parser.add_argument(
        '--historical-data',
        type=str,
        help="Path to CSV file with historical data (optional, uses sample data if not provided)"
    )
    parser.add_argument(
        '--show-all-predictions',
        action='store_true',
        help="Show predictions for all core configurations (default: show only key predictions)"
    )
    parser.add_argument(
        '--save-analysis',
        action='store_true',
        help="Save detailed analysis to JSON file"
    )
    
    args = parser.parse_args()
    
    print(f"🔧 Testing core configurations: 1 to {args.max_cores}")
    
    # Initialize predictor
    try:
        predictor = AutoscalingPredictor(
            model_path=args.model_path,
            target_response_time=args.target_rt,
            max_cores=args.max_cores
        )
        print(f"✓ Predictor initialized with target response time: {args.target_rt}s")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return
    
    # Load historical data
    if args.historical_data:
        try:
            df = pd.read_csv(args.historical_data)
            # Assume CSV has columns: users, cores, response_time (3 features only)
            required_columns = ['users', 'cores', 'response_time']
            if not all(col in df.columns for col in required_columns):
                print(f"❌ CSV must contain columns: {required_columns}")
                print(f"Found columns: {list(df.columns)}")
                return
            
            historical_data = df[required_columns].tail(5).values
            
            if historical_data.shape[0] < 5:
                print(f"❌ Need at least 5 historical data points, got {historical_data.shape[0]}")
                return
                
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            print("Using sample data instead...")
            historical_data = load_sample_historical_data()
    else:
        print("📝 Using sample historical data for demonstration")
        historical_data = load_sample_historical_data()
    
    # Validate historical data shape
    if historical_data.shape != (5, 3):
        print(f"❌ Historical data must have shape (5, 3), got {historical_data.shape}")
        return
    
    # Generate report
    report = predictor.generate_scaling_report(
        historical_data, 
        args.current_users, 
        args.current_cores,
        show_all_predictions=args.show_all_predictions
    )
    
    print(report)
    
    # Save detailed analysis if requested
    if args.save_analysis:
        analysis = predictor.find_optimal_cores(historical_data, args.current_users, args.current_cores)
        
        output_file = "autoscaling_analysis.json"
        with open(output_file, 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            json_analysis = {}
            for key, value in analysis.items():
                if isinstance(value, np.floating):
                    json_analysis[key] = float(value)
                elif isinstance(value, np.integer):
                    json_analysis[key] = int(value)
                else:
                    json_analysis[key] = value
            
            json.dump(json_analysis, f, indent=2)
        
        print(f"\n💾 Detailed analysis saved to {output_file}")

# Example usage when imported as a module
def quick_prediction_example():
    """
    Example of how to use this module programmatically.
    """
    # Initialize predictor (model loaded once and cached)
    predictor = AutoscalingPredictor(target_response_time=1.5, max_cores=8)
    
    # Sample historical data
    historical_data = load_sample_historical_data()
    
    # Get recommendation
    recommendation = predictor.get_scaling_recommendation(
        historical_data=historical_data,
        current_users=80,
        current_cores=2
    )
    
    print("Example programmatic usage:")
    print(f"Action: {recommendation['action']}")
    print(f"Recommended cores: {recommendation['recommended_cores']}")
    print(f"Expected response time: {recommendation['expected_response_time']:.3f}s")
    
    return recommendation

if __name__ == "__main__":
    main() 