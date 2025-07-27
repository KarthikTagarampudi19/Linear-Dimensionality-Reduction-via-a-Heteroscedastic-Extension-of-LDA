import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hlda_sldr import hlda_sldr # Custom HLDA module

# --- Manual LDA Classifier ---
def manual_lda_classifier(X_train, Y_train, X_test):
    classes = torch.unique(Y_train)
    d = X_train.shape[1]
    pooled_cov = torch.zeros((d, d), dtype=X_train.dtype)
    means, priors = [], []
    for c in classes:
        Xc = X_train[Y_train == c]
        priors.append(Xc.shape[0] / X_train.shape[0])
        means.append(Xc.mean(dim=0))
        pooled_cov += torch.cov(Xc.T) * (Xc.shape[0] - 1)
    pooled_cov /= (X_train.shape[0] - len(classes))
    inv_cov = torch.linalg.inv(pooled_cov)
    
    # Compute LDA decision scores
    scores = []
    for i, mu in enumerate(means):
        mu = mu.view(-1, 1)
        score = X_test @ inv_cov @ mu - 0.5 * mu.T @ inv_cov @ mu + np.log(priors[i])
        scores.append(score.squeeze())
    scores = torch.stack(scores, dim=1)
    preds = classes[scores.argmax(dim=1)]
    return preds

# --- Manual QDA Classifier ---
def manual_qda_classifier(X_train, Y_train, X_test):
    classes = torch.unique(Y_train)
    scores = []
    priors, means, covs = [], [], []

    # Compute class means, priors, and pooled covariance
    for c in classes:
        Xc = X_train[Y_train == c]
        priors.append(Xc.shape[0] / X_train.shape[0])
        means.append(Xc.mean(dim=0))
        covs.append(torch.cov(Xc.T))
    
    # QDA scoring for each test sample
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

# --- Plotting ---
def plot_scatter_plain(Z, labels, title_str):
    plt.figure()
    unique_labels = torch.unique(labels)
    colors = [
        [0.8, 0.1, 0.1],   # Class 1 - red
        [0.1, 0.8, 0.1],   # Class 2 - green
        [0.1, 0.1, 0.8],   # Class 3 - blue
        [0.8, 0.8, 0.1],   # Class 4 - yellow/orange
        [0.8, 0.1, 0.8],   # Class 5 - purple
        [0.1, 0.8, 0.8],   # Class 6 - cyan
        [0.5, 0.5, 0.5],   # Class 7 - gray
    ]
    for i, label in enumerate(unique_labels):
        idx = labels == label
        color = colors[i % len(colors)]
        plt.scatter(Z[idx, 0], Z[idx, 1], label=f'Class {int(label)}',
                    color=color, s=35)
    
    # Axis settings
    plt.xlim([-6, 11])
    plt.ylim([-9, 10])
    plt.xticks(np.arange(-6, 12, 2))
    plt.yticks(np.arange(-9, 11, 2))
    plt.title(title_str)
    plt.xlabel("HLDA Dimension 1"); plt.ylabel("HLDA Dimension 2")
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.tight_layout()

# ==================
# MAIN EXPERIMENT
# ==================

# Load LandSat data
X_trn = np.loadtxt("sat.trn")
X_tst = np.loadtxt("sat.tst")

X_raw = np.vstack((X_trn[:, :36], X_tst[:, :36]))
Y = np.hstack((X_trn[:, 36], X_tst[:, 36])).astype(int)

# Normalize
scaler = StandardScaler()
X_raw = scaler.fit_transform(X_raw)

X_raw = torch.tensor(X_raw, dtype=torch.float64)
Y = torch.tensor(Y, dtype=torch.int64)

n_runs = 100
target_dim = 3
linear_errors = []
quadratic_errors = []

Z_train_final = Z_test_final = Z_all = None
Y_train_final = Y_test_final = None
mu_last = coeff_last = W_last = None

for run in range(n_runs):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_raw, Y, test_size=0.1, stratify=Y, random_state=run
    )

    mu_train = X_train.mean(dim=0, keepdim=True)
    X_train_centered = X_train - mu_train
    X_test_centered = X_test - mu_train

    U, S, Vh = torch.linalg.svd(X_train_centered, full_matrices=False)
    latent = S**2 / (X_train_centered.shape[0] - 1)

    sorted_vals, sort_idx = torch.sort(latent, descending=True)
    sorted_vecs = Vh.T[:, sort_idx]

    #sorted_vecs = -sorted_vecs

    if run == 0:
        # Log PCA statistics
        print("PCA Eigendecomposition:")
        print("First 3 eigenvalues:", sorted_vals[:3].numpy())
        print("First eigenvector:", sorted_vecs[:, 0].numpy())
        np.savetxt("pca_first3_eigenvalues_LandSat.csv", sorted_vals[:3].numpy(), delimiter=",")
        np.savetxt("pca_first_eigenvector_LandSat.csv", sorted_vecs[:, 0].numpy().reshape(-1, 1), delimiter=",")
    
    # Select dimensions above numerical threshold
    valid_dims = latent > (1e-6 * latent.sum())
    V = Vh.T[:, valid_dims]
    
    # Project to PCA space
    X_train_pca = X_train_centered @ V
    X_test_pca = X_test_centered @ V
    
    # HLDA Projection
    para, Z_train = hlda_sldr(X_train_pca, Y_train, target_dim)
    Z_test = X_test_pca @ para['W'] - para['mb'] @ para['W']

    if run == 0:
        print("Checking HLDA results:")
        print("HLDA data dimensions:", Z_train.shape)
        print("First 3 data points after HLDA:\n", Z_train[:3].numpy())
        np.savetxt("hlda_first3_data_points_LandSat.csv", Z_train[:3].numpy(), delimiter=",")
    
    # Save final transformation
    if run == n_runs - 1:
        mu_last = mu_train
        coeff_last = V
        W_last = para['W']
        Z_train_final = Z_train
        Z_test_final = Z_test
        Y_train_final = Y_train
        Y_test_final = Y_test
    
    # Classification and error computation
    pred_linear = manual_lda_classifier(Z_train, Y_train, Z_test)
    pred_quad = manual_qda_classifier(Z_train, Y_train, Z_test)

    linear_errors.append((pred_linear != Y_test).float().mean().item())
    quadratic_errors.append((pred_quad != Y_test).float().mean().item())

# === Final full projection
X_centered_full = X_raw - mu_last
X_pca_full = X_centered_full @ coeff_last
Z_all = X_pca_full @ W_last

# === Plots
plot_scatter_plain(Z_all, Y, "Full HLDA Projection (LandSat Dataset)")
plot_scatter_plain(Z_train_final, Y_train_final, "HLDA Projection - Training Set")
plot_scatter_plain(Z_test_final, Y_test_final, "HLDA Projection - Test Set")

# === MCE Plot
plt.figure()
plt.bar(['Linear MCE', 'Quadratic MCE'], [np.mean(linear_errors), np.mean(quadratic_errors)])
plt.ylabel("Classification Error")
plt.title("Mean Classification Error (100 Runs)")
plt.grid(True)
plt.show()

print(f"Average Linear Classification Error: {np.mean(linear_errors):.2f}")
print(f"Average Quadratic Classification Error: {np.mean(quadratic_errors):.2f}")
