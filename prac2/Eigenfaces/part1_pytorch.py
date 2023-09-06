from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Introspect the images arras to find the shaes (for plotting)
n_samples, h, w = lfw_people.images.shape

# For machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# The label to predict is the ID of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes %d" % n_classes)

# Split into a training set and a test set using stratified k fold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Load into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float64).to(device)
X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
y_train = torch.tensor(y_train, dtype=torch.float64).to(device)
y_test = torch.tensor(y_test, dtype=torch.float64).to(device)

# Compute a PCA (eigenfaces) on the face dataset (teated as unalabelled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

# Center data
mean = torch.mean(X_train, axis=0)
X_train -= mean
X_test -= mean

# Eigen-decomposition
U, S, V = torch.linalg.svd(X_train, full_matrices=False)
components = V[:n_components]
eigenfaces = components.reshape((n_components, h, w))

# Project into a PCA subspace
X_transformed = torch.matmul(X_train, components.T).to(device)
X_test_transformed = torch.matmul(X_test, components.T).to(device)

# Qualitative evalation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    '''Helper function to plot a gallery of portraits'''
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenfaces_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(torch.Tensor(eigenfaces).cpu(), eigenfaces_titles, h, w)

plt.show()

explained_variance = (S ** 2) / (n_samples - 1)
total_var = explained_variance.sum()
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = torch.cumsum(explained_variance_ratio, dim=0)
eigenvalueCount = torch.arange(n_components)

plt.plot(torch.Tensor(eigenvalueCount).cpu(), torch.Tensor(ratio_cumsum[:n_components]).cpu())
plt.title('Compactness')
plt.show()