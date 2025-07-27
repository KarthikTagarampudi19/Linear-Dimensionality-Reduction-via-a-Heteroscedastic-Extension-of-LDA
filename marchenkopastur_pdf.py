import torch
import numpy as np
import matplotlib.pyplot as plt

def marchenkopastur_pdf(lambda_vals, s, c):
    """
    Compute the Marchenko-Pastur probability density function
    
    Parameters:
    lambda_vals (torch.Tensor): Points at which to evaluate the PDF
    s (float): Scale parameter (sigma)
    c (float): Aspect ratio (n/p)
    
    Returns:
    torch.Tensor: PDF values at points lambda_vals
    """
    # Define the edges of the support
    lambda_p = s**2 * (1 + torch.sqrt(torch.tensor(c)))**2
    lambda_m = s**2 * (1 - torch.sqrt(torch.tensor(c)))**2
    
    # Initialize PDF values with zeros
    pdf_vals = torch.zeros_like(lambda_vals, dtype=torch.float32)
    
    # Apply the condition from the original MATLAB function
    if lambda_m > 0 and lambda_p < lambda_m * min(len(lambda_vals), 100):
        # Calculate PDF values using the MP formula
        pdf_vals = (1 / (2 * np.pi * lambda_vals * c * s**2)) * torch.sqrt((lambda_p - lambda_vals) * (lambda_vals - lambda_m))
        
        # Set values to zero where conditions are not met
        pdf_vals[pdf_vals < 0] = 0
        pdf_vals[lambda_vals > lambda_p] = 0
        pdf_vals[lambda_vals < lambda_m] = 0
    
    return pdf_vals

def fit_MarchenkoPastur(Data, lambda_vals):
    """
    Fit the Marchenko-Pastur distribution to eigenvalues
    
    Parameters:
    Data (torch.Tensor or numpy.ndarray): Input data matrix
    lambda_vals (torch.Tensor or numpy.ndarray): Eigenvalues to fit
    
    Returns:
    float: Upper edge of the MP distribution (lambda_max)
    dict: Additional parameters and fitting information
    """
    # Convert inputs to PyTorch tensors if they aren't already
    if not isinstance(Data, torch.Tensor):
        Data = torch.tensor(Data, dtype=torch.float32)
    if not isinstance(lambda_vals, torch.Tensor):
        lambda_vals = torch.tensor(lambda_vals, dtype=torch.float32)
    
    # Calculate standard deviation omitting NaN values
    s = torch.std(Data[~torch.isnan(Data)])
    
    # Calculate ratio
    r = Data.shape[1] / Data.shape[0]
    
    # Initialize counters and arrays for grid search
    counter = 0
    ll_vals = torch.zeros(50 * 30)
    L = torch.zeros(50 * 30)
    params = torch.zeros((50 * 30, 2))
    
    # Grid search for best parameters
    for s_var in range(1, 51):
        for r_var in range(1, 31):
            counter += 1
            
            # Choose parameters in [sigma/5, 2*sigma] and [ratio/3, 20*ratio]
            s_mp = (5 + s_var) / 30 * s
            r_mp = r_var * r / 10
            
            # Estimate MP likelihood
            pdf_vals = marchenkopastur_pdf(lambda_vals, s_mp, r_mp)
            
            # Filter out zeros
            pdf_vals_filtered = pdf_vals[pdf_vals > 0]
            
            # Save parameters for choosing next
            if len(pdf_vals_filtered) > 0:
                ll_vals[counter-1] = torch.sum(torch.log(pdf_vals_filtered))
                L[counter-1] = len(pdf_vals_filtered)
            params[counter-1, 0] = s_var
            params[counter-1, 1] = r_var
    
    # Choose parameters with max log-likelihood sums
    mask = L > (1/3) * len(lambda_vals)
    weighted_ll = ll_vals * mask
    ind_max = torch.argmax(weighted_ll)
    
    s_var = params[ind_max, 0].item()
    r_var = params[ind_max, 1].item()
    
    # Retrieve parameters
    s_mp = (5 + s_var) / 30 * s
    r_mp = r_var * r / 10
    
    # Adjusting plot intervals
    n = 100
    lambda_p = s_mp**2 * (1 + 10 * torch.sqrt(torch.tensor(r_mp)))**2  # max lambda
    lambda_m = s_mp**2 * (1 - 2 * torch.sqrt(torch.tensor(r_mp)))**2   # min lambda
    space_lambda = torch.linspace(lambda_m.item(), lambda_p.item(), n)
    
    # Calculate PDF values
    pdf_vals = marchenkopastur_pdf(space_lambda, s_mp, r_mp)
    pdf_vals = pdf_vals / torch.sum(pdf_vals)  # normalizing
    
    # Final normalizing (reproducing MATLAB's hist function)
    hist_output = torch.histc(lambda_vals, bins=n, min=lambda_m.item(), max=lambda_p.item())
    lambda_h = space_lambda  # equivalent to the bin centers
    
    # Calculate lambda_max and lambda_min for normalizing
    lambda_max = s_mp**2 * (1 + torch.sqrt(torch.tensor(r_mp)))**2
    lambda_min = s_mp**2 * (1 - torch.sqrt(torch.tensor(r_mp)))**2
    
    # Filter lambda values based on range
    lambda_range = lambda_vals[(lambda_vals <= lambda_max) & (lambda_vals >= lambda_min)]
    
    # Normalize histogram
    f = hist_output / torch.sum(hist_output) * len(lambda_vals) / len(lambda_range)
    
    return lambda_max.item(), {
        'lambda_min': lambda_min.item(),
        'sigma': s_mp.item(),
        'ratio': r_mp.item(),
        'pdf_x': space_lambda.numpy(),
        'pdf_y': pdf_vals.numpy(),
        'hist_values': f.numpy(),
        'hist_bins': lambda_h.numpy()
    }

def plot_marchenko_pastur_fit(lambda_vals, lambda_max_val, result_info):
    """
    Plot the histogram of eigenvalues with the fitted MP distribution
    
    Parameters:
    lambda_vals (torch.Tensor or numpy.ndarray): Eigenvalues 
    lambda_max_val (float): Estimated upper bound of the distribution
    result_info (dict): Result information from fit_MarchenkoPastur
    """
    plt.figure(figsize=(10, 6))
    
    # Convert to numpy if tensor
    if isinstance(lambda_vals, torch.Tensor):
        lambda_vals = lambda_vals.numpy()
    
    # Plot histogram
    plt.hist(lambda_vals, bins=50, density=True, alpha=0.5, label='Eigenvalue Histogram')
    
    # Plot fitted distribution
    plt.plot(result_info['pdf_x'], result_info['pdf_y'], 'r-', linewidth=2, label='Fitted MP Distribution')
    
    # Add vertical lines for bounds
    plt.axvline(x=result_info['lambda_min'], color='g', linestyle='--', label='λ_min')
    plt.axvline(x=lambda_max_val, color='b', linestyle='--', label='λ_max')
    
    plt.title('Marchenko-Pastur Distribution Fit')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
