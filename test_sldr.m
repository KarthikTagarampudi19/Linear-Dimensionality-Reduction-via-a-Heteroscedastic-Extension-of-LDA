function  Z = test_sldr(X, para)

% Z = test_sldr(X, para)
% Input:
%    X:      n x d matrix of original feature samples
%            d --- dimensionality of original features
%            n --- the number of samples
%    para:   structure of the model
% Output:
%    Z:      n x dim matrix of dimensionality reduced features

% Applies a trained SLDR (Supervised Linear Dimensionality Reduction) model to test data
% Inputs:
%   X    - (n x d) matrix of input samples
%   para - structure containing projection matrix and model parameters
% Output:
%   Z    - (n x dim) matrix of dimensionality-reduced features

% Retrieve the global mean from training data

mb = para.mb;

% recentering original feature
X = X - mb;

% Check which dimensionality reduction model was used
if  strcmp(para.model , 'plsda')
    % Partial Least Squares Discriminant Analysis path
    mu = para.mu_pca ;
    coeff = para.W;
    beta_coef = para.beta_coef;
    n = size(X,1);
    yhat = [ones(n,1) X]*beta_coef;
    Z = (yhat-mu)*coeff;
else
    W = para.W;
    Z = X*W;
end


% if strcmp(para.model , 'lda') || strcmp(para.model , 'hlda')
%
%     W = para.W;
%     mb = para.mb;
%     X = X - mb;
%     Z = X*W;
%
% elseif strcmp(para.model , 'mmda')
%
%     W = para.W;
%     mb = para.mb;
%     Sw_sqrtinv = para.Sw_sqrtinv;
%
%     X = X - mb;
%     X = X * Sw_sqrtinv ;
%     Z = X*W;
%
% end