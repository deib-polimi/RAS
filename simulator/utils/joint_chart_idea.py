import matplotlib.pyplot as plt
import numpy as np

# Simulated data
time = np.arange(0, 100, 1)
qn_cores = 5 + np.sin(time / 10)  # Example QN cores estimation
X = 1
Y = 1
# Compute CT allocations based on QN estimations
np.random.seed(1)  # For reproducibility
ct_cores_raw = qn_cores + 0.7 * np.random.randn(len(time))  # Example CT cores allocation

# Apply a moving average to smooth the ct_cores line
window_size = 5
ct_cores = np.convolve(ct_cores_raw, np.ones(window_size)/window_size, mode='same')

ct_allocations_min = -X + qn_cores
ct_allocations_max = +Y + qn_cores

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time[2:-2], ct_cores[2:-2], label='CT', color='black')
plt.fill_between(time, ct_allocations_min, ct_allocations_max, color='red', alpha=0.3, label='Allowed Range (X * QN, Y * QN)')
plt.xlabel('Time (s)')
plt.ylabel('Cores')
plt.title('CT Cores Allocation with QN Estimated Range')
plt.legend()
plt.grid(True)
plt.show()
