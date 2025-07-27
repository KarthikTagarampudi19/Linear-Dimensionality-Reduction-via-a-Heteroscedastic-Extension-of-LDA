import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from hlda_sldr import hlda_sldr  # PyTorch HLDA 

# ----------------------------
# Manual LDA Classifier
# ----------------------------
def manual_lda_classifier(X_train, Y_train, X_test):
    classes = torch.unique(Y_train)
    d = X_train.shape[1]
    pooled_cov = torch.zeros((d, d), dtype=X_train.dtype)
    means = []
    priors = []
    
    # Compute class means, priors, and pooled covariance
    for c in classes:
        Xc = X_train[Y_train == c]
        priors.append(Xc.shape[0] / X_train.shape[0])
        means.append(Xc.mean(dim=0))
        cov_c = torch.cov(Xc.T)
        pooled_cov += cov_c * (Xc.shape[0] - 1)
    pooled_cov /= (X_train.shape[0] - len(classes))
    inv_cov = torch.linalg.inv(pooled_cov)
    
    # Compute class scores
    scores = []
    for i, mu in enumerate(means):
        mu = mu.view(-1, 1)
        score = X_test @ inv_cov @ mu - 0.5 * mu.T @ inv_cov @ mu + np.log(priors[i])
        scores.append(score.squeeze())
    scores = torch.stack(scores, dim=1)
    preds = classes[scores.argmax(dim=1)]
    return preds

# ----------------------------
# Manual QDA Classifier
# ----------------------------
def manual_qda_classifier(X_train, Y_train, X_test):
    classes = torch.unique(Y_train)
    scores = []
    priors = []
    means = []
    covs = []
    
    # Compute separate covariance per class
    for c in classes:
        Xc = X_train[Y_train == c]
        priors.append(Xc.shape[0] / X_train.shape[0])
        means.append(Xc.mean(dim=0))
        covs.append(torch.cov(Xc.T))
    
    # Score each test point for each class
    for i in range(len(classes)):
        mu = means[i]
        cov = covs[i]
        inv_cov = torch.linalg.inv(cov)
        log_det = torch.logdet(cov)
        s = []
        for x in X_test:
            diff = x - mu
            s.append(-0.5 * diff @ inv_cov @ diff - 0.5 * log_det + np.log(priors[i]))
        scores.append(torch.tensor(s))
    scores = torch.stack(scores, dim=1)
    preds = classes[scores.argmax(dim=1)]
    return preds

# ----------------------------
# Scatter Plot Helper
# ----------------------------
def scatter_plot(Z, labels, title, xlim=None, ylim=None, xticks=None, yticks=None):
    plt.figure()
    unique_labels = torch.unique(labels)
    colors = [
        [0.8, 0.1, 0.1],   # Red
        [0.1, 0.8, 0.1],   # Green
        [0.1, 0.1, 0.8],   # Blue
        [0.8, 0.8, 0.1],   # Yellow
        [0.8, 0.1, 0.8],   # Magenta
        [0.1, 0.8, 0.8],   # Cyan
        [0.5, 0.5, 0.5]    # Gray
    ]
    for i, label in enumerate(unique_labels):
        idx = (labels == label)
        color = colors[i % len(colors)]
        plt.scatter(Z[idx, 0], Z[idx, 1], label=f'Class {int(label)}',
                    color=color, s=40,marker='o')
    plt.title(title, fontsize=13)
    plt.xlabel("HLDA Dimension 1")
    plt.ylabel("HLDA Dimension 2")
    plt.legend()
    plt.box(True)
    plt.grid(True)
    plt.legend(title="Classes", loc="best", fontsize=9, title_fontsize=10)

    if xlim:
        plt.xlim(xlim)
        if xticks:
            plt.xticks(np.arange(xlim[0], xlim[1]+1, xticks))
    if ylim:
        plt.ylim(ylim)
        if yticks:
            plt.yticks(np.arange(ylim[0], ylim[1]+1, yticks))

    plt.tight_layout()

# ----------------------------
# Main Run
# ----------------------------

# Load Glass dataset
glass = np.loadtxt("glass.data", delimiter=",")
X_raw = glass[:, 1:-1]  # features (columns 2 to 10)
Y = glass[:, -1].astype(int)  # class labels

# Standardize the full dataset
scaler = StandardScaler()
X_raw = scaler.fit_transform(X_raw)

# Convert to torch
X_raw = torch.tensor(X_raw, dtype=torch.float64)
Y = torch.tensor(Y, dtype=torch.int64)

n_runs = 100
linear_errors = []
quadratic_errors = []

# These will be set in the last run for plotting
Z_train_final = Z_test_final = Z_all = None
Y_train_final = Y_test_final = None

target_dim = 3

for run in range(n_runs):
    # 90:10 split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_raw, Y, test_size=0.1, stratify=Y, random_state=run
    )

    # Center training data
    mu_train = X_train.mean(dim=0, keepdim=True)
    X_train_centered = X_train - mu_train
    X_test_centered = X_test - mu_train

    # PCA on training data
    U, S, Vh = torch.linalg.svd(X_train_centered, full_matrices=False)
    latent = S**2 / (X_train_centered.shape[0] - 1)

    sorted_vals, sort_idx = torch.sort(latent, descending=True)
    sorted_vecs = Vh.T[:, sort_idx]

    #sorted_vecs = -sorted_vecs

    if run == 0:

        print("PCA Eigendecomposition:")
        print("First 3 eigenvalues:", sorted_vals[:3].numpy())
        print("First eigenvector:", sorted_vecs[:, 0].numpy())
        np.savetxt("pca_first3_eigenvalues.csv", sorted_vals[:3].numpy(), delimiter=",")
        np.savetxt("pca_first_eigenvector.csv", sorted_vecs[:, 0].numpy().reshape(-1, 1), delimiter=",")

    pca_thresh = 1e-6 * latent.sum()
    valid_dims = latent > pca_thresh
    V = Vh.T[:, valid_dims]

    # Project using PCA
    X_train_pca = X_train_centered @ V
    X_test_pca = X_test_centered @ V

    # HLDA
    para, Z_train = hlda_sldr(X_train_pca, Y_train, target_dim)
    Z_test = X_test_pca @ para['W'] - para['mb'] @ para['W']

    if run == 0:
        print("Checking HLDA results:")
        print("HLDA data dimensions:", Z_train.shape)
        print("First 3 data points after HLDA:\n", Z_train[:3].numpy())
        np.savetxt("hlda_first3_data_points.csv", Z_train[:3].numpy(), delimiter=",")
        
    if run == n_runs - 1:
        # Apply PCA and HLDA to full dataset
        X_all_centered = X_raw - mu_train
        X_all_pca = X_all_centered @ V
        Z_all = X_all_pca @ para['W'] - para['mb'] @ para['W']

        Z_train_final = Z_train
        Z_test_final = Z_test
        Y_train_final = Y_train
        Y_test_final = Y_test

    # Manual LDA/QDA classification
    pred_linear = manual_lda_classifier(Z_train, Y_train, Z_test)
    pred_quad = manual_qda_classifier(Z_train, Y_train, Z_test)

    linear_errors.append((pred_linear != Y_test).float().mean().item())
    quadratic_errors.append((pred_quad != Y_test).float().mean().item())

# === PLOTS ===
scatter_plot(Z_all, Y, "Full HLDA Projection (Glass Dataset)", xlim=[-6, 9], ylim=[-2, 6], xticks=2, yticks=1)
scatter_plot(Z_train_final, Y_train_final, "HLDA Projection - Training Set", xlim=[-4, 9], ylim=[-2, 6], xticks=2, yticks=1)
scatter_plot(Z_test_final, Y_test_final, "HLDA Projection - Test Set", xlim=[-5, 2], ylim=[-1, 0.4], xticks=1, yticks=0.2)

# Classification Error Bar Plot
plt.figure()
plt.bar(['Linear MCE', 'Quadratic MCE'], [np.mean(linear_errors), np.mean(quadratic_errors)])
plt.ylabel("Classification Error")
plt.title("Mean Classification Error (100 Runs)")
plt.grid(True)

plt.show()

# Print results
print(f"Average Linear Classification Error: {np.mean(linear_errors):.2f}")
print(f"Average Quadratic Classification Error: {np.mean(quadratic_errors):.2f}")
