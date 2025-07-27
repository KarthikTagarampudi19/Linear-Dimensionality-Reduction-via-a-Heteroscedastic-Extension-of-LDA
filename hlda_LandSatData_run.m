% Clean start
close all; clc; clear;

% Load and preprocess LandSat data
train_data = load('sat.trn');
test_data = load('sat.tst');

% Concatenate all data into one array
X_raw = [train_data(:,1:36); test_data(:,1:36)];
Y = [train_data(:,37); test_data(:,37)];

% Normalize entire dataset (Z-score)
X_raw = (X_raw - mean(X_raw)) ./ std(X_raw);

% Preallocate errors
n_runs = 100;
linear_errors = zeros(n_runs,1);
quadratic_errors = zeros(n_runs,1);

target_dim = 3; % HLDA output dimensionality

% Final run storage
Z_train_final = [];
Z_test_final = [];
Y_train_final = [];
Y_test_final = [];

% for full dataset projection (captured after last run)
mu_last = [];
coeff_last = [];
W_last = [];

for run = 1:n_runs
    % Manual 90:10 split
    n = size(X_raw, 1);
    idx = randperm(n);
    n_train = round(0.9 * n);
    train_idx = idx(1:n_train);
    test_idx = idx(n_train+1:end);

    X_train = X_raw(train_idx, :);
    Y_train = Y(train_idx);
    X_test = X_raw(test_idx, :);
    Y_test = Y(test_idx);

    % Center training data
    mu_train = mean(X_train,1);
    X_train_centered = X_train - mu_train;
    X_test_centered = X_test - mu_train;

    % PCA on training data
    [~, S, V] = svd(X_train_centered, 'econ');
    latent = diag(S).^2 / (size(X_train_centered,1) - 1);
    PCA_threshold = 1e-6 * sum(latent);
    valid_dims = find(latent > PCA_threshold);
    coeff = V(:, valid_dims);

    % Project via PCA
    X_train_pca = X_train_centered * coeff;
    X_test_pca = X_test_centered * coeff;
    
    if run == 1
        % Sort latent values and eigenvectors for display
        [sorted_eig_values, sort_idx] = sort(latent, 'descend');
        sorted_eig_vectors = V(:, sort_idx);

        % Print PCA results
        disp('PCA Eigendecomposition:');
        disp('First 3 eigenvalues:');
        disp(sorted_eig_values(1:3)');
        writematrix(sorted_eig_values(1:3), 'pca_first3_eigenvalues_LandSat.csv');
        disp('First eigenvector:');
        disp(sorted_eig_vectors(:, 1)');
        writematrix(sorted_eig_vectors(:, 1), 'pca_first_eigenvector_LandSat.csv');
    end

    % HLDA
    [para, Z_train] = hlda_sldr(X_train_pca, Y_train, target_dim);
    Z_test = X_test_pca * para.W;
    
    if run == 1 
        disp('Checking HLDA results:');
        disp(['HLDA data dimensions: ', num2str(size(Z_train))]);
        disp('First 3 data points after HLDA:');
        disp(Z_train(1:3,:));
    end

    % Store transform from last run
    if run == n_runs
        mu_last = mu_train;
        coeff_last = coeff;
        W_last = para.W;

        Z_train_final = Z_train;
        Z_test_final = Z_test;
        Y_train_final = Y_train;
        Y_test_final = Y_test;
    end

    writematrix(Z_train(1:3,:), 'hlda_first3_data_points_LandSat.csv');


    % Classification
    pred_linear = manual_lda_classifier(Z_train, Y_train, Z_test);
    pred_quad = manual_qda_classifier(Z_train, Y_train, Z_test);

    linear_errors(run) = mean(pred_linear ~= Y_test);
    quadratic_errors(run) = mean(pred_quad ~= Y_test);
end

% ==== Final Full Dataset Projection ====
X_centered_full = X_raw - mu_last;
X_pca_full = X_centered_full * coeff_last;
Z_all = X_pca_full * W_last;

% ==== PLOTS ====
plot_scatter_plain(Z_all, Y, 'Full HLDA Projection (LandSat Dataset)');
legend({'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'}, 'Location', 'best');
plot_scatter_plain(Z_train_final, Y_train_final, 'HLDA Projection - Training Set');
legend({'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'}, 'Location', 'best');
plot_scatter_plain(Z_test_final, Y_test_final, 'HLDA Projection - Test Set');
legend({'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'}, 'Location', 'best');

% MCE plot
figure;
bar([mean(linear_errors), mean(quadratic_errors)]);
set(gca, 'XTickLabel', {'Linear MCE', 'Quadratic MCE'});
ylabel('Classification Error');
title('Mean Classification Error (100 Runs)');
grid on;

% Output
fprintf('Average Linear Classification Error: %.2f\n', mean(linear_errors));
fprintf('Average Quadratic Classification Error: %.2f\n', mean(quadratic_errors));

% ==== Dependencies ====

function preds = manual_lda_classifier(X_train, Y_train, X_test)
    % Implements Linear Discriminant Classifier manually
    classes = unique(Y_train);
    num_classes = numel(classes);
    n_features = size(X_train, 2);
    pooled_cov = zeros(n_features);
    means = zeros(num_classes, n_features);
    priors = zeros(num_classes, 1);
    for i = 1:num_classes
        Xi = X_train(Y_train == classes(i), :);
        priors(i) = size(Xi,1) / size(X_train,1);
        means(i,:) = mean(Xi, 1);
        pooled_cov = pooled_cov + cov(Xi, 1) * (size(Xi,1) - 1);
    end
    pooled_cov = pooled_cov / (size(X_train,1) - num_classes);
    Sigma_inv = inv(pooled_cov);
    scores = zeros(size(X_test,1), num_classes);
    for i = 1:num_classes
        mu = means(i,:)';
        scores(:,i) = X_test * Sigma_inv * mu - 0.5 * mu' * Sigma_inv * mu + log(priors(i));
    end
    [~, idx] = max(scores, [], 2);
    preds = classes(idx);
end

function preds = manual_qda_classifier(X_train, Y_train, X_test)
    % Implements Quadratic Discriminant Classifier manually
    classes = unique(Y_train);
    num_classes = numel(classes);
    n_features = size(X_train, 2);
    means = zeros(num_classes, n_features);
    priors = zeros(num_classes, 1);
    covs = cell(num_classes, 1);
    for i = 1:num_classes
        Xi = X_train(Y_train == classes(i), :);
        priors(i) = size(Xi,1) / size(X_train,1);
        means(i,:) = mean(Xi, 1);
        covs{i} = cov(Xi, 1);
    end
    scores = zeros(size(X_test,1), num_classes);
    for i = 1:num_classes
        mu = means(i,:)';
        Sigma = covs{i};
        Sigma_inv = inv(Sigma);
        log_det = log(det(Sigma));
        for j = 1:size(X_test,1)
            x = X_test(j,:)';
            scores(j,i) = -0.5 * (x - mu)' * Sigma_inv * (x - mu) ...
                          - 0.5 * log_det + log(priors(i));
        end
    end
    [~, idx] = max(scores, [], 2);
    preds = classes(idx);
end

% Scatter plot of 2D projection with class coloring
function plot_scatter_plain(Z, labels, plot_title)
    figure; hold on;
    uniq = unique(labels);
    num_classes = length(uniq);
    colors = [
        0.8 0.1 0.1;
        0.1 0.8 0.1;
        0.1 0.1 0.8;
        0.8 0.8 0.1;
        0.8 0.1 0.8;
        0.1 0.8 0.8;
        0.5 0.5 0.5;
        0.2 0.6 0.2;
        0.6 0.2 0.6
    ];
    for i = 1:num_classes
        idx = labels == uniq(i);
        c = colors(mod(i-1, size(colors,1)) + 1, :);
        plot(Z(idx,1), Z(idx,2), 'o', ...
            'MarkerSize', 6, ...
            'MarkerEdgeColor', c, ...
            'MarkerFaceColor', c);
    end
    title(plot_title);
    xlabel('HLDA Dimension 1'); ylabel('HLDA Dimension 2');
    box on; grid on;
end
