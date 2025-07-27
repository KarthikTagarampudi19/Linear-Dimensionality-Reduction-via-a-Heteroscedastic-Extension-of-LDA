% Clean start
close all;      % Close any open figures
clc; clear;     % Clear console and variables

% Load the Iris dataset
opts = detectImportOptions('iris.data', 'FileType', 'text');
opts = setvartype(opts, 5, 'string');  % Make sure labels are strings
T = readtable('iris.data', opts);


% Extract features and class labels
X = table2array(T(:, 1:4));     % Features (150x4)
labels = T{:, 5};               % Class labels as strings
[unique_labels, ~, Y] = unique(labels);  % Convert to numeric classes 1, 2, 3

% Center the data
X_mean = mean(X, 1);
X_centered = X - X_mean;

% Covariance matrix
cov_matrix = cov(X_centered);

% Eigen decomposition
[eig_vectors, eig_values_matrix] = eig(cov_matrix);
eig_values = diag(eig_values_matrix);

% Sort eigenvalues and eigenvectors in descending order
[sorted_eig_values, sort_idx] = sort(eig_values, 'descend');
sorted_eig_vectors = eig_vectors(:, sort_idx);

% Retain enough components to cover 95% variance
explained_variance = cumsum(sorted_eig_values) / sum(sorted_eig_values);
k = find(explained_variance >= 0.95, 1);  % Smallest k that gives ≥ 95% variance

% PCA projection to k dimensions
PCA_basis = sorted_eig_vectors(:, 1:k);
X_reduced = X_centered * PCA_basis;  % 150 x k

% For PCA
disp('PCA Eigendecomposition:');
if ~isreal(eig_values)
    disp('Complex eigenvalues detected in PCA, taking real parts');
    disp(['Complex components magnitude: ', num2str(max(abs(imag(eig_values))))]);
    eig_values = real(eig_values);
end

if ~isreal(eig_vectors)
    disp('Complex eigenvectors detected in PCA, taking real parts');
    disp(['Complex components magnitude: ', num2str(max(abs(imag(eig_vectors(:)))))]);
    eig_vectors = real(eig_vectors);
end

% Print first few eigenvalues for comparison
disp('First 3 eigenvalues:');
disp(sorted_eig_values(1:3)');
disp('First eigenvector:');
disp(sorted_eig_vectors(:, 1)');


% HLDA projection to target dimension
target_dim = 2;
[para, Z] = hlda_sldr(X_reduced, Y, target_dim);

% Close any unexpected figure created by hlda_sldr
close(gcf);

% After calling hlda_sldr, check the intermediate results:
disp('Checking HLDA results:');
disp(['HLDA data dimensions: ', num2str(size(Z))]);
disp('First 3 data points after HLDA:');
disp(Z(1:3,:));

% Manual scatter plot with bold, filled markers (toolbox-free)
colors = {'r', '[0, 0.5, 0]', 'b'};
markers = {'o', 'o', 'o'};

figure;
hold on;
for i = 1:length(unique_labels)
    idx = (Y == i);
    plot(Z(idx,1), Z(idx,2), ...
         markers{i}, ...
         'Color', colors{i}, ...
         'MarkerFaceColor', colors{i}, ...  % Filled for bold appearance
         'DisplayName', unique_labels(i), ...
         'MarkerSize', 8, ...
         'LineWidth', 1.3);
end
xlabel('HLDA Dimension 1');
ylabel('HLDA Dimension 2');
title('HLDA Reduced Iris Data (2D)');
legend(cellstr(unique_labels), 'Location', 'best');
grid on;
hold off;