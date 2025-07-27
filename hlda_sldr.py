import torch
import numpy as np
from scipy import linalg
import scipy.sparse.linalg

from marchenko_pastur import marchenko_pastur

def hlda_sldr(X, labels, dim=None):
    """
    Heteroscedastic extension of LDA for supervised linear dimension reduction (LDR).
    Modified to match MATLAB output exactly.
    
    Based on:
    Duin, Robert PW, and Marco Loog.
    "Linear dimensionality reduction via a heteroscedastic extension of LDA: the Chernoff criterion."
    IEEE transactions on pattern analysis and machine intelligence 26, no. 6 (2004): 732-739.
    
    Args:
        X: n x d tensor of original feature samples
            d --- dimensionality of original features
            n --- the number of samples
        labels: n-dimensional tensor of class labels
        dim: dimensionality of reduced space (default: number of classes)
        dim has to be from 1 <= dim <= d
    
    Returns:
        para: output dictionary of hlda model parameters
        Z: n x dim tensor of dimensionality reduced features
    """
    # Convert inputs to PyTorch tensors if they aren't already
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float64)
    else:
        X = X.to(dtype=torch.float64)
        
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.int64)
    
    # Store original data for reference
    X_orig = X.clone()
    
    # Get unique class labels
    classes_labels = torch.unique(labels)
    num_classes = len(classes_labels)
    
    # Set default dimensionality if not provided
    if dim is None:
        dim = min(num_classes, max(1, X.shape[1]-1))
    
    # Recenter original features
    mb = torch.mean(X, dim=0)
    X = X - mb
    
    # Initialize variables
    d = X.shape[1]  # feature dimension
    Sw = torch.zeros((d, d), dtype=X.dtype)
    SB = torch.zeros((d, d), dtype=X.dtype)
    M = torch.zeros((num_classes, d), dtype=X.dtype)
    p = torch.zeros(num_classes, dtype=X.dtype)
    Si = {}
    
    # Calculate class statistics
    for k, class_label in enumerate(classes_labels):
        class_mask = (labels == class_label)
        class_samples = X[class_mask]
        
        # Class covariance - ensure it's symmetric like MATLAB
        # Use marchenko_pastur for covariance estimation
        if len(class_samples) > 1:
            Si[k] = marchenko_pastur(class_samples)
        else:
            Si[k] = torch.eye(d, dtype=X.dtype) * 1e-6
        
        # Class mean
        M[k, :] = torch.mean(class_samples, dim=0)
        
        # Class prior probability
        p[k] = torch.sum(class_mask).float() / len(labels)
        
        # Within-class scatter matrix
        Sw = Sw + p[k] * Si[k]
        
        # Between-class scatter matrix (outer product of mean vectors)
        SB = SB + p[k] * torch.outer(M[k, :], M[k, :])
    
    # Ensure Sw is symmetric and positive definite (MATLAB behavior)
    Sw = (Sw + Sw.t()) / 2
    Sw = Sw + torch.eye(d, dtype=X.dtype) * 1e-10
    
    # Compute inverse and square root of Sw
    # Move to NumPy for matrix functions not available in PyTorch
    Sw_np = Sw.numpy()
    
    # Use cholesky decomposition for better numerical stability (like MATLAB)
    try:
        # Try Cholesky decomposition first (what MATLAB likely uses)
        L = np.linalg.cholesky(Sw_np)
        Sw_inv_np = np.linalg.inv(Sw_np)
    except np.linalg.LinAlgError:
        # If Cholesky fails, use pseudo-inverse
        Sw_inv_np = np.linalg.pinv(Sw_np)
    
    # Ensure eigenvalues are positive for fractional power
    w, v = np.linalg.eigh(Sw_np)
    # Replace any negative eigenvalues with small positive values
    w[w < 1e-10] = 1e-10
    
    # Reconstruct Sw with positive eigenvalues
    Sw_np = v @ np.diag(w) @ v.T
    
    # Compute square root and inverse square root
    Sw_sqrt_np = np.real_if_close(linalg.fractional_matrix_power(Sw_np, 0.5))
    Sw_sqrtinv_np = np.real_if_close(linalg.fractional_matrix_power(Sw_np, -0.5))
    
    # Initialize Chernoff criterion matrix
    S_chernoff = np.zeros((d, d))

    # Compute Chernoff criterion
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            p_i = p[i].item() / (p[i].item() + p[j].item())
            p_j = p[j].item() / (p[i].item() + p[j].item())
            m_i = M[i].numpy().reshape(-1, 1)  # Column vector
            m_j = M[j].numpy().reshape(-1, 1)
            Si_i = Si[i].numpy()
            Si_j = Si[j].numpy()
            
            # Form the weighted matrix (make symmetric for numerical stability)
            Sij = p_i * Si_i + p_j * Si_j
            Sij = (Sij + Sij.T) / 2
            
            # Transformed matrices (make symmetric for stability)
            wSijw = Sw_sqrtinv_np @ Sij @ Sw_sqrtinv_np
            wSijw = (wSijw + wSijw.T) / 2
            
            wSiw = Sw_sqrtinv_np @ Si_i @ Sw_sqrtinv_np
            wSiw = (wSiw + wSiw.T) / 2
            
            wSjw = Sw_sqrtinv_np @ Si_j @ Sw_sqrtinv_np
            wSjw = (wSjw + wSjw.T) / 2
            
            # Add small regularization to ensure positive definiteness
            eps = 1e-10
            wSijw = wSijw + np.eye(wSijw.shape[0]) * eps
            wSiw = wSiw + np.eye(wSiw.shape[0]) * eps
            wSjw = wSjw + np.eye(wSjw.shape[0]) * eps
            
            # Compute fractional powers using eigendecomposition for stability
            wSijw_neg_half = np.real_if_close(linalg.fractional_matrix_power(wSijw, -0.5))
            
            # Compute matrix logarithms and take only real parts (like MATLAB)
            log_wSijw = np.real(linalg.logm(wSijw))
            log_wSiw = np.real(linalg.logm(wSiw))
            log_wSjw = np.real(linalg.logm(wSjw))
            
            # Direct calculation matching MATLAB implementation
            s_ij = Sw_inv_np @ Sw_sqrt_np @ (
                wSijw_neg_half @
                Sw_sqrtinv_np @ (m_i - m_j) @ (m_i - m_j).T @ 
                Sw_sqrtinv_np @ wSijw_neg_half +
                (1 / (p_i * p_j)) * (
                    log_wSijw - 
                    p_i * log_wSiw - 
                    p_j * log_wSjw
                )
            ) @ Sw_sqrt_np
            
            # Make s_ij symmetric for numerical stability
            s_ij = (s_ij + s_ij.T) / 2
            
            S_chernoff += p[i].item() * p[j].item() * np.real(s_ij)
    
    # Make S_chernoff symmetric for better numerical stability
    S_chernoff = (S_chernoff + S_chernoff.T) / 2
    
    # Eigen decomposition - use eigh since we've ensured symmetry
    eigvals, eigvecs = np.linalg.eigh(S_chernoff)
    
    # Convert to real and sort in descending order
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    
    # Sort by descending eigenvalue
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    
    # Apply MATLAB's eigenvector sign convention
    for i in range(eigvecs.shape[1]):
        # Find the element with largest magnitude
        max_idx = np.argmax(np.abs(eigvecs[:, i]))
        # Ensure this element is positive (MATLAB's convention)
        if eigvecs[max_idx, i] < 0:
            eigvecs[:, i] = -eigvecs[:, i]
    
    # Select the first dim eigenvectors
    W_np = eigvecs[:, :dim]
    W = torch.tensor(W_np, dtype=torch.float64)

    # Project the data
    Z = X_orig @ W - mb @ W  # More stable than X @ W

    para = {
        'W': W,
        'mb': mb,
        'model': 'hlda'
    }

    return para, Z


def test_sldr(X, para):
    """
    Apply the learned HLDA transformation to new data
    
    Args:
        X: n x d tensor of original feature samples
        para: parameters output from hlda_sldr
    
    Returns:
        Z: n x dim tensor of dimensionality reduced features
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float64)
    else:
        X = X.to(dtype=torch.float64)
    
    # Center the data and project in one step for numerical stability
    Z = X @ para['W'] - para['mb'] @ para['W']
    
    return Z