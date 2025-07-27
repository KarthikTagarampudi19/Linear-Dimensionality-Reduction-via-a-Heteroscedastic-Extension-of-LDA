import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import linalg
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

# Clean start
plt.close('all')  # Close any open figures

# Load the Iris dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
#iris_df = pd.read_csv(url, header=None, names=column_names)

file_path = "./iris.data" 
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(file_path, header=None, names=column_names)


# Extract features and class labels
X = iris_df.iloc[:, 0:4].values  # Features (150x4)
labels = iris_df.iloc[:, 4].values  # Class labels as strings
unique_labels = np.unique(labels)
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(labels) + 1  # Convert to numeric classes 1, 2, 3 (matching MATLAB indexing)

# Convert to NumPy for processing
X_np = X.copy()

# Center the data
X_mean = np.mean(X_np, axis=0)
X_centered = X_np - X_mean

# Covariance matrix
n_samples = X_centered.shape[0]
cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

# Eigen decomposition - using eigh for symmetric matrix
eig_values, eig_vectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eig_values)[::-1]  # Descending order
eig_values = eig_values[idx]
eig_vectors = eig_vectors[:, idx]

# Flip the sign of the first eigenvector
#eig_vectors[:, 0] *= -1

# Retain enough components to cover 95% variance
explained_variance = np.cumsum(eig_values) / np.sum(eig_values)
k = np.where(explained_variance >= 0.95)[0][0] + 1  # +1 because indices are 0-based

# PCA projection to k dimensions
PCA_basis = eig_vectors[:, :k]
#PCA_basis = -1 * PCA_basis #To flip Vector direction
X_reduced = np.dot(X_centered, PCA_basis)  # 150 x k

# For PCA
print("PCA Eigendecomposition:")
if not np.isreal(eig_values).all():
    print("Complex eigenvalues detected in PCA, taking real parts")
    print(f"Complex components magnitude: {np.max(np.imag(eig_values))}")
    eig_values = np.real(eig_values)

if not np.isreal(eig_vectors).all():
    print("Complex eigenvectors detected in PCA, taking real parts")
    print(f"Complex components magnitude: {np.max(np.abs(np.imag(eig_vectors)))}")
    eig_vectors = np.real(eig_vectors)

# Print first few eigenvalues for comparison
print("First 3 eigenvalues:", np.round(eig_values[:3],4))
first_eigenvector_display = -1 * eig_vectors[:, 0]
print("First eigenvector:", np.round(first_eigenvector_display,4))


# Flexible Fisher LDA implementation that allows tweaking the spread
def hlda_sldr(X, Y, target_dim, spread_factor=1.0):
    """
    Implementation of Fisher's LDA with parameter to control the spread of points
    """
    n_samples, n_features = X.shape
    classes = np.unique(Y)
    n_classes = len(classes)
    
    
    # Calculate overall mean
    mean_overall = np.mean(X, axis=0)
    
    # Calculate between-class scatter matrix
    S_b = np.zeros((n_features, n_features))
    for c in classes:
        X_c = X[Y == c]
        mean_c = np.mean(X_c, axis=0)
        n_c = X_c.shape[0]
        
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        S_b += n_c * (mean_diff @ mean_diff.T)
    
    # Calculate within-class scatter matrix
    S_w = np.zeros((n_features, n_features))
    for c in classes:
        X_c = X[Y == c]
        mean_c = np.mean(X_c, axis=0)
        
        for i in range(X_c.shape[0]):
            x_diff = (X_c[i] - mean_c).reshape(n_features, 1)
            S_w += np.dot(x_diff, x_diff.T)
    
    # Apply spread factor to within-class scatter
    S_w = S_w / spread_factor

    # Add small regularization
    S_w += 1e-6 * np.eye(n_features)
    
    # Use scipy's generalized eigenvalue solver which is stable
    eig_vals, eig_vecs = linalg.eig(S_b, S_w)
    
    # Make sure eigenvalues and eigenvectors are real
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)

    # Sort eigenvalues in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    W = eig_vecs[:, idx[:target_dim]]
    Z = X @ W

    return W, Z

# Apply LDA with increased spread to match MATLAB plot
target_dim = 2
spread_factor = 1.0  # Adjust this value to control spread (lower = more spread)
_, Z = hlda_sldr(X_reduced, Y, target_dim, spread_factor)



print("Checking HLDA results:")
print(f"HLDA data dimensions: {Z.shape}")
print("First 3 data points after HLDA:")
print(np.round(Z[:3],4))

# Create plot with MATLAB-like appearance and similar spread
plt.figure(figsize=(10, 8))
colors = ['r', [0, 0.5, 0], 'b']  # Dark green for Iris-versicolor
markers = ['o', 'o', 'o']

# Configure plot with MATLAB-like properties
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 8),
    'figure.dpi': 100,
})

# Plot with smaller marker size to match MATLAB plot
for i, label in enumerate(unique_labels):
    idx = (Y == i+1)  # +1 because we adjusted Y to start from 1
    plt.scatter(Z[idx, 0], Z[idx, 1], 
            color=colors[i], 
            marker=markers[i],
            s=50,  # Smaller marker size to match MATLAB plot
            linewidth=1.0,
            edgecolors=colors[i],
            label=label)

plt.xlabel('HLDA Dimension 1')
plt.ylabel('HLDA Dimension 2')
plt.title('HLDA Reduced Iris Data (2D)')
plt.grid(True)
plt.legend()

# Set axis limits to match MATLAB plot more closely
plt.xlim(-3, 4)
plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()