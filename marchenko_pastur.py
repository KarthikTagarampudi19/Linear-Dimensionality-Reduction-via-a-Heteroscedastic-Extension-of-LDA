import torch
import numpy as np

from fit_MarchenkoPastur import fit_MarchenkoPastur

def fit_MarchenkoPastur(data, lambda_values):
    """
    Fits Marchenko-Pastur distribution and finds lambda_max for denoising
    
    Args:
        data: Input data matrix
        lambda_values: Eigenvalues of the covariance matrix
    
    Returns:
        lambda_max: Maximum eigenvalue for denoising
    """
    # In a full implementation, this would fit the Marchenko-Pastur distribution
    # to determine the maximum eigenvalue that represents noise
    
    # For simplicity, a common approach is to use a threshold based on the 
    # Marchenko-Pastur distribution parameters
    n, m = data.shape  # n samples, m features
    q = m / n
    
    # Theoretical maximum eigenvalue for pure noise according to M-P distribution
    lambda_max = (1 + np.sqrt(q))**2
    
    # Scale by the mean of the eigenvalues to adapt to the data scale
    lambda_max *= np.mean(lambda_values.cpu().numpy() if isinstance(lambda_values, torch.Tensor) else lambda_values)
    
    return lambda_max


def marchenko_pastur(data_in):
    """
    Implements Marchenko-Pastur distribution for denoising covariance matrices
    to avoid matrix singularity in computations.
    
    Args:
        data_in: Input data matrix
    
    Returns:
        Denoised covariance matrix
    """
    # Calculate asset covariance
    AssetCovar = data_in.T @ data_in / data_in.shape[0]
    
    # Make sure the covariance matrix is symmetric
    sigma_trg = (AssetCovar + AssetCovar.T) / 2
    
    # Perform SVD
    # PyTorch's SVD returns U, S, V where V is already transposed (V*)
    U, S, V = torch.svd(sigma_trg)
    lambda_values = S ** 2 # Singular values
    
    # Fit Marchenko-Pastur distribution and find lambda_max for denoising
    lambda_max = fit_MarchenkoPastur(data_in, lambda_values)
    
    # Find eigenvalues below lambda_max (noise)
    noise_ind = lambda_values <= lambda_max
    
    # Compute mean of noisy eigenvalues
    if torch.any(noise_ind):
        lambda_j = torch.mean(lambda_values[noise_ind])
    else:
        lambda_j = 0.0
    
    # Create denoised eigenvalues
    lambda_den = lambda_values.clone()
    lambda_den[noise_ind] = lambda_j
    
    # Reconstruct denoised covariance matrix
    # Note: PyTorch's V is already transposed, so we use V directly
    sigma_trg_den = U @ torch.diag(lambda_den) @ V.T
    
    return sigma_trg_den