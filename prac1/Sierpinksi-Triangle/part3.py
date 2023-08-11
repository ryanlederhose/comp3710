import torch
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize points
points = torch.tensor([[0.5, 0], [1, 1], [0, 1]], device=device)
point = torch.tensor([0.5, 0.5], device=device)  # Initial point

# Number of iterations
num_iterations = 1000000

# Initialize arrays to store x and y coordinates
x_coords = torch.empty(num_iterations, device=device)
y_coords = torch.empty(num_iterations, device=device)

# Iterate to generate the Sierpinski triangle
for i in range(num_iterations):
    # Randomly select one of the three vertices
    chosen_vertex = points[np.random.choice(3)]
    
    # Calculate the midpoint between the chosen vertex and the current point
    point = (point + chosen_vertex) / 2.0
    
    # Store the coordinates
    x_coords[i] = point[0]
    y_coords[i] = point[1]

# Plot the points of the Sierpinski triangle
plt.scatter(points[:, 0].cpu(), points[:, 1].cpu(), color='red')
plt.scatter(x_coords.cpu(), y_coords.cpu(), color='blue', s=0.1)  # Highlight the generated points
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


