clc;
clear all;
close all;

rng(42);

% Knobs
n   = 1000;             % # samples
T   = [3, 4, 5];        % # periods for each signal
SNR = 50;               % Signal SNR
d   = 3;                % # mixed observations
r   = 3;                % # independent/principal components

% Generate ground truth
t        = @(n,T) linspace(0,1,n) * 2 * pi * T;
Ztrue(1,:) = sin(t(n,T(1)));            % Sinusoid
Ztrue(2,:) = sign(sin(t(n,T(2))));      % Square
Ztrue(3,:) = sawtooth(t(n,T(3)));       % Sawtooth

% Add noise
sigma    = @(SNR,X) exp(-SNR / 20) * (norm(X(:)) / sqrt(numel(X)));
Znoisy = Ztrue + sigma(SNR,Ztrue) * randn(size(Ztrue));

% Generate mixed signals
normRows = @(X) bsxfun(@rdivide,X,sum(X,2));
A = normRows(rand(d,3));
Zmixed = A * Znoisy;

% Perform Fast ICA
Zfica = fastICA(Zmixed,r);

% Perform max-kurtosis ICA
Zkica = kICA(Zmixed,r);

% Perform PCA
Zpca = PCA(Zmixed,r);

% Perform Kalman filter
% Replace the following lines with your Kalman filter implementation
% Zkalman = kalmanFilter(Zmixed);
Zkalman = zeros(size(Zmixed));  % Replace with your Kalman filter output

% Calculate evaluation metrics
mse = @(X, Y) mean((X(:) - Y(:)).^2);
mae = @(X, Y) mean(abs(X(:) - Y(:)));
rmse = @(X, Y) sqrt(mean((X(:) - Y(:)).^2));
psnr = @(X, Y) 20 * log10(max(X(:)) / sqrt(mse(X, Y)));

% Calculate metrics for Fast ICA
MSE_fica = mse(Ztrue, Zfica);
MAE_fica = mae(Ztrue, Zfica);
RMSE_fica = rmse(Ztrue, Zfica);
PSNR_fica = psnr(Ztrue, Zfica);

% Calculate metrics for max-kurtosis ICA
MSE_kica = mse(Ztrue, Zkica);
MAE_kica = mae(Ztrue, Zkica);
RMSE_kica = rmse(Ztrue, Zkica);
PSNR_kica = psnr(Ztrue, Zkica);

% Calculate metrics for PCA
MSE_pca = mse(Ztrue, Zpca);
MAE_pca = mae(Ztrue, Zpca);
RMSE_pca = rmse(Ztrue, Zpca);
PSNR_pca = psnr(Ztrue, Zpca);

% Calculate metrics for Kalman filter
MSE_kalman = mse(Ztrue, Zkalman);
MAE_kalman = mae(Ztrue, Zkalman);
RMSE_kalman = rmse(Ztrue, Zkalman);
PSNR_kalman = psnr(Ztrue, Zkalman);

disp('Evaluation Metrics:');
disp('-------------------');
disp('Fast ICA:');
disp(['MSE: ', num2str(MSE_fica)]);
disp(['MAE: ', num2str(MAE_fica)]);
disp(['RMSE: ', num2str(RMSE_fica)]);
disp(['PSNR: ', num2str(PSNR_fica)]);
disp('-------------------');
disp('Max-kurtosis ICA:');
disp(['MSE: ', num2str(MSE_kica)]);
disp(['MAE: ', num2str(MAE_kica)]);
disp(['RMSE: ', num2str(RMSE_kica)]);
disp(['PSNR: ', num2str(PSNR_kica)]);
disp('-------------------');
disp('PCA:');
disp(['MSE: ', num2str(MSE_pca)]);
disp(['MAE: ', num2str(MAE_pca)]);
disp(['RMSE: ', num2str(RMSE_pca)]);
disp(['PSNR: ', num2str(PSNR_pca)]);
disp('-------------------');
disp('Kalman Filter:');
disp(['MSE: ', num2str(MSE_kalman)]);
disp(['MAE: ', num2str(MAE_kalman)]);
disp(['RMSE: ', num2str(RMSE_kalman)]);
disp(['PSNR: ', num2str(PSNR_kalman)]);
