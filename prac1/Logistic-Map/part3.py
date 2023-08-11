import torch
import matplotlib.pyplot as plt
import numpy as np

def logistic_map(x, y):
    '''
    Speciefies the logistic map function
    '''
    y_next = y * x * (1 - y)
    x_next = x + 0.000001
    yield x_next, y_next

# Parameters
num_points = 30000000  # Number of points in the x-axis

# Generate an array of r values and fractal data
r_values = torch.zeros(num_points + 1)
r_values[0] = 1
fractal_data = torch.zeros(num_points + 1)
fractal_data[0] = 0.5

# Enumerate
for i in range(num_points):
    if r_values[i] > 4:
        break

    x_next, y_next = next(logistic_map(r_values[i], fractal_data[i]))
    r_values[i + 1] = x_next
    fractal_data[i + 1] = y_next

# Plot fractal using matplot lib
plt.style.use('dark_background')
plt.figure(figsize=(10, 10))
plt.plot(r_values.numpy(), fractal_data.numpy(), '^', color='white', alpha=0.4, markersize = 0.013)
plt.xlim(1, 4)
plt.axis('on')
plt.show()
