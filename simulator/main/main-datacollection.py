#!/usr/bin/env python3
"""
Main script for Training Data Collection

This script uses the DataCollectionController within the RAS framework
to collect training data by randomly varying core allocations and
observing system responses under different workload conditions.

The collected data will be compatible with the neural network training
pipeline (EvolutionStrategy_simple.py and generate_training_data.py).

Usage:
    python simulator/main/main-datacollection.py
"""

from .main import Main
from ..utils import commons as C

# Configure the Data Collection Controller
DATA_COLLECTION = C.DATA_COLLECTION

# Set SLA for the data collection
DATA_COLLECTION.setSLA(C.APP_SLA)  # 0.4s target response time

# Configure different exploration strategies for variety
# You can change this to "sweep" or "adaptive" for different data patterns
DATA_COLLECTION.set_exploration_strategy("random")
DATA_COLLECTION.set_change_frequency(5)  # Change cores every 5 periods

print("🎯 Starting Data Collection for Neural Network Training")
print("=" * 60)
print(f"Controller: {DATA_COLLECTION}")
print(f"Target SLA: {C.APP_SLA}s")
print(f"Core Range: [{C.MIN_CORES}, {C.MAX_CORES}]")
print(f"Application: {C.APPLICATION_1}")
print(f"Generators: {len(C.GEN_TRAIN_SET)} workload patterns")
print("=" * 60)

controllers = [
    DATA_COLLECTION
]

# Run multiple experiments with different workload patterns for diversity
print("\n🔄 Running Data Collection Experiments...")

# Run the experiment
main = Main(
    name=f"DataCollection-Exp",
    controllers=controllers,
    generators=C.GEN_TRAIN_SET,
    horizon=C.HORIZON,
    monitoringWindow=C.MONITORING_WINDOW,
    app=C.APPLICATION_2
)

main.start()

# Show progress
stats = DATA_COLLECTION.get_collection_stats()
print(f"   ✅ Collected {stats['total_steps']} data points")

print("\n🎉 Data Collection Complete!")
print("=" * 60)

# Final statistics
final_stats = DATA_COLLECTION.get_collection_stats()
print("📊 Final Collection Statistics:")
for key, value in final_stats.items():
    print(f"   {key}: {value}")

print(f"\n💾 Training data logged to: {final_stats.get('log_path', 'N/A')}")

print("\n💡 Next Steps:")
print("1. Review the generated CSV log file")
print("2. Convert data format if needed for generate_training_data.py")
print("3. Use the data to train neural networks with EvolutionStrategy_simple.py")
print("\n✅ Data collection ready for neural network training!") 