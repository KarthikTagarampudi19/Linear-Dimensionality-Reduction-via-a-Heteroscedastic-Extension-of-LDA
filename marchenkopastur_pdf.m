function pdf_vals = marchenkopastur_pdf(lambda,s,c)
% Computes the theoretical Marchenko–Pastur Probability Density Function (PDF)
% Inputs:
%   lambda - vector of eigenvalues
%   s      - standard deviation (scale parameter)
%   c      - aspect ratio (n_samples / n_features)
% Output:
%   pdf_vals - vector of PDF values evaluated at each lambda

% Compute the theoretical maximum and minimum eigenvalues
lambda_p = s^2 * (1+sqrt(c))^2;
lambda_m = s^2 * (1-sqrt(c))^2;

% Check if valid range (to avoid invalid square roots)
if lambda_m>0 && lambda_p < lambda_m*(min(length(lambda),100))
    
    % Apply Marchenko–Pastur formula for PDF within [lambda_m, lambda_p]
    pdf_vals = (1./(2*pi*lambda*c*s^(2))).*sqrt((lambda_p-lambda).*(lambda-lambda_m));
    pdf_vals(pdf_vals<0) =0;
    pdf_vals(lambda>lambda_p) = 0;
    pdf_vals(lambda<lambda_m) = 0;
    
else
    % If invalid parameters, return zero array
    pdf_vals = 0*lambda;
end