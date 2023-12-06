import matplotlib.pyplot as plt
import numpy as np

# Sample data
models = ['Perfect', 'Distractor', 'No']
component2_data = [0.8491837779906065, 0.8055059016188796, 0.6197599613959817]  # Random data for component 1
component1_data = [0.36, 0.23, 0.04]   # Random data for component 2

# Bar width
bar_width = 0.35

# Set up figure and axis
fig, ax = plt.subplots()

# Plot the bars
bar1 = ax.bar(np.arange(len(models)), component1_data, width=bar_width, label='Baseline')
bar2 = ax.bar(np.arange(len(models)) + bar_width, component2_data, width=bar_width, label='FERMI')

# Set labels, title, and legend
ax.set_xlabel('Question Type')
ax.set_ylabel('FP Score')
ax.set_xticks(np.arange(len(models)) + bar_width / 2)
ax.set_xticklabels(models)
ax.legend()

# Show the plot
plt.savefig("plot.png")