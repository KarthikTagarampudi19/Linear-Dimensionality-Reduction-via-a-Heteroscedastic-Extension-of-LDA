 function [para, Z] = hlda_sldr(X, labels, dim)

% Heteroscedastic extension of LDA for supervised linear dimension reduction (LDR).
% Duin, Robert PW, and Marco Loog.
% "Linear dimensionality reduction via a heteroscedastic extension of LDA: the Chernoff criterion."
% IEEE transactions on pattern analysis and machine intelligence 26, no. 6 (2004): 732-739.

%[para,W] = hlda_sldr(X, labels, dim) , where dim values by default is Number of classes: C
% Input:
%    X:      n x d matrix of original feature samples
%            d --- dimensionality of original features
%            n --- the number of samples
%    labels: n --- dimensional vector of class labels
%    dim:    ----- dimensionality of reduced space (default:C)
%            dim has to be from 1<=dim<=d
% Output:
%    para:   output structure of hlda model for input of test_sldr.m function
%    Z:      n x dim matrix of dimensionality reduced features

% Get unique class labels and number of classes
classes_labels = unique(labels);
num_classes = length(classes_labels);


% Set default target dimension if not provided
if(nargin==2)
    dim= min(num_classes,max(1,size(X,2)-1));
end

% recentering original feature
mb = mean(X,'omitnan');
X = X - mb;

% Initialize within-class scatter and between-class scatter matrices
Sw = 0;
SB = 0;

% Compute class statistics: mean, prior, covariance using Marchenko–Pastur denoising
for k = 1:num_classes
    
    % Regularized class covariance using Marchenko–Pastur
    %Si{k}= cov( X(labels==classes_labels(k),:) ,1, 'omitrows' );
    Si{k}= marchenko_pastur( X(labels==classes_labels(k),:));% To avoid matrix singularity in computations, we employ Marchenko–Pastur for denoising covariance matrices.
    M(k,:) = mean(X(labels==classes_labels(k),:),'omitnan');
    p(k) = sum(labels==classes_labels(k))/length(labels);
    Sw = Sw + p(k)*Si{k};
    SB = SB + p(k)*M(k,:)' * M(k,:);

end

% Compute inverse and square roots of within-class scatter matrix
Sw_inv = inv(Sw);
Sw_sqrt = (Sw)^0.5;
Sw_sqrtinv = inv(Sw_sqrt);

S_chernoff = 0;

% Loop over all class pairs to compute the Chernoff criterion
for i=1:num_classes
    for j=i+1:num_classes
        p_i = p(i)/(p(i)+p(j));
        p_j = p(j)/(p(i)+p(j));

        m_i = M(i,:)';
        m_j = M(j,:)';

        Sij = p_i*Si{i} + p_j*Si{j};
        wSijw = Sw_sqrtinv*Sij*Sw_sqrtinv;
        wSiw = Sw_sqrtinv*Si{i}*Sw_sqrtinv;
        wSjw = Sw_sqrtinv*Si{j}*Sw_sqrtinv;

        S_chernoff = S_chernoff + p(i)*p(j)*Sw_inv*Sw_sqrt*(wSijw^-0.5...
            *Sw_sqrtinv*(m_i-m_j)*(m_i-m_j)'...
            *Sw_sqrtinv*wSijw^-0.5...
            +1/(p_i*p_j)*(logm(wSijw)-p_i*logm(wSiw)-p_j*logm(wSjw)))*Sw_sqrt;
    end
end

% selecting dim eigenvectors associated with the dim largest eigenvalues
[V,D] = eig(S_chernoff);
D = real(diag(D));
V = real(V);

% Sort eigenvectors by descending eigenvalue
[~,sort_index]=sort(D,'descend');
W =  V(:,sort_index(1:dim));


% Z has the dimentional reduced data sample X.
Z = X*W;

% Output parameters
para.W = W; % Projection matrix
para.mb = mb; % Data mean used in centering
para.model = 'hlda'; % Identifier for model type

end