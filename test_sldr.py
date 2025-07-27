import torch
import numpy as np


def test_sldr(X, para):
    """
    Apply a learned dimension reduction transformation to new data.
    
    Args:
        X: n x d tensor of original feature samples
            d --- dimensionality of original features
            n --- the number of samples
        para: dictionary of model parameters
    
    Returns:
        Z: n x dim tensor of dimensionality reduced features
    """
    # Convert to PyTorch tensor if not already
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    
    # Get mean vector from parameters
    mb = para['mb']
    
    # Recenter original features
    X = X - mb
    
    # Apply the appropriate transformation based on model type
    if para['model'] == 'plsda':
        mu = para['mu_pca']
        coeff = para['W']
        beta_coef = para['beta_coef']
        
        n = X.shape[0]
        # Add column of ones for intercept term
        X_with_intercept = torch.cat([torch.ones(n, 1, dtype=X.dtype), X], dim=1)
        
        # Compute predictions
        yhat = X_with_intercept @ beta_coef
        
        # Apply PCA transformation
        Z = (yhat - mu) @ coeff
    else:
        # For LDA, HLDA, etc.
        W = para['W']
        Z = X @ W
    
    return Z

# The original code had commented sections for different model types, 
# but the active code only differentiated between 'plsda' and other models.
# If you need the functionality for 'mmda' or other specific model types in the 
# commented section, they can be implemented as additional conditions.