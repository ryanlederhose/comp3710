import torch
import numpy as np
import matplotlib.pyplot as plt

def processFractal(a):
    '''
    Display an array of iteration counts as a colourful picture of a fractal
    '''
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([10 + 20 * np.cos(a_cyclic), 
                          30 + 50 * np.sin(a_cyclic),
                          155 - 80 * np.cos(a_cyclic)], 2)
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Y1, X1 = np.mgrid[-0.772:-0.76:0.00001, -0.272:-0.26:0.00001]   # zoomed mgrid

# Load Mandelbrot PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y)
zs = z.clone()
ns = torch.zeros_like(z)
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

# Load Mandelbrot zoomed PyTorch tensors
x1 = torch.Tensor(X1)
y1 = torch.Tensor(Y1)
z1 = torch.complex(x1, y1)
zs1 = z1.clone()
ns1 = torch.zeros_like(z1)
z1 = z1.to(device)
zs1 = zs1.to(device)
ns1 = ns1.to(device)

# Load Julia PyTorch tensors
x2 = torch.Tensor(X)
y2 = torch.Tensor(Y)
z2 = torch.complex(x2, y2)
zs2 = z2.clone()
ns2 = torch.zeros_like(z2)
c = complex(-0.7, 0.27015)  # Julia set constant
z2 = z2.to(device)
zs2 = zs2.to(device)
ns2 = ns2.to(device)


for i in range(200):
    # Compute the new values of z: z^2 + x
    zs_ = zs * zs + z       # Mandelbrot set
    zs1_ = zs1 * zs1 + z1       # Mandelbrot set zoomed
    zs2_ = zs2 * zs2 + c       # Julia set
    
    '''
    For Mandelbrot Set
    '''
    # Have we diverged with this new value?
    not_diverged = torch.abs(zs_) < 4.0

    # Update variables to compute
    ns += not_diverged
    zs = zs_

    '''
    For Mandelbrot zoomed set
    '''
    # Have we diverged with this new value?
    not_diverged = torch.abs(zs1_) < 4.0

    # Update variables to compute
    ns1 += not_diverged
    zs1 = zs1_

    '''
    For Julia set
    '''
    # Have we diverged with this new value?
    not_diverged = torch.abs(zs2_) < 4.0

    # Update variables to compute
    ns2 += not_diverged
    zs2 = zs2_

# Mandelbrot set
fig = plt.figure(figsize=(16,10))
plt.imshow(processFractal(ns.cpu().numpy()))
plt.title('Mandelbrot Set')
plt.tight_layout(pad=0)
plt.show()

# Mandelbrot zoomed
fig = plt.figure(figsize=(16,10))
plt.imshow(processFractal(ns1.cpu().numpy()))
plt.title('Mandelbrot Set Zoomed')
plt.tight_layout(pad=0)
plt.show()

# Julia set
fig = plt.figure(figsize=(16,10))
plt.imshow(processFractal(ns2.cpu().numpy()))
plt.title('Julia Set')
plt.tight_layout(pad=0)
plt.show()