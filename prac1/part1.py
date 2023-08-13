import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

# Sine parameters
A = 100
phase = 0
freq = 1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# Load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# Transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute equations
z1 = torch.exp(-(x**2 + y**2) / 2.0)    # Gaussian
z2 = A * torch.sin(freq * (x + y) + phase) # 2d Sine
z = z1 * z2                             # Multiplied outputs

# Plot z1
plt.imshow(z1.numpy())
plt.tight_layout()
plt.title("Gaussian")
plt.show()

# Plot z2
plt.imshow(z2.numpy())
plt.tight_layout()
plt.title("2d Sine")
plt.show()

# Plot z
plt.imshow(z.numpy())
plt.tight_layout()
plt.title("Multiplied Gaussian and 2d Sine")
plt.show()