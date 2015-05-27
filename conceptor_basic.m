%% A basic conceptor with multiple input signals!

% clc;
clear;

train_time = 1500;
washout_time = 500;
test_time = 50;
learn_time = train_time - washout_time;

patterns = {@(n) sin(n / 4), @(n) sin(n/2), @(n) sin(n/3), @(n) sin(n/5)};
num_patterns = numel(patterns);

% figure();
% plot(d(1:100));
% title('Target function');

%% Initialize parameters
disp('Creating reservoir');

% Set seed for random numbers
rng(100);

% Scalings
scale_w = 1.5;
scale_w_in = 1.5;
scale_b = 0.2;

N = 100;
alpha = 10;

bias = randn(N, 1) * scale_b;

W = randn(N, N);
spectral_radius = max(abs(eig(W)));
W = W / spectral_radius;
W = scale_w * W;

W_in = (randn(N, 1)) * scale_w_in;

% Regulizers
lambda_out = 1e-2;
lambda_w = 1e-4;

n_disp = 4;
% ind = randperm(N, n_disp);
ind = [1 5 16 20];

%% Driving!
disp('Driving reservoir with training pattern');

% All patterns
XAll = zeros(N, num_patterns * learn_time);
PAll = zeros(1, num_patterns * learn_time);
XoldAll = zeros(N, num_patterns * learn_time);

for p = 1:num_patterns    
    % N * L
    X = zeros(N, learn_time);
    Xold = zeros(N, learn_time); % the one step delayed
    x = zeros(N, 1);
    % P - 1 * L
    P = zeros(1, learn_time);

    d = patterns{p};
    for i = 1:train_time
        x_ = x;
        x = tanh(W * x + W_in * d(i) + bias); % x(i) = tanh(W*x(i-1) + Win * d(i) + b)
        if i > washout_time
           X(:, i - washout_time) = x;
           Xold(:, i - washout_time) = x_;
           P(i - washout_time) = d(i);
        end
    end
    
    XAll(:, (p-1)*learn_time + 1: p*learn_time) = X;
    XoldAll(:, (p-1)*learn_time + 1: p*learn_time) = Xold;
    PAll(:, (p-1)*learn_time + 1: p*learn_time) = P;
end

% figure();
% for i = 1:n_disp
%     subplot(2,4,i);
%     fig_name = sprintf('Node %d', ind(i));
%     plot(X(ind(i), end-test_time:end));
%     title(fig_name);
% end
% 
% subplot(2,4,5);
% plot(d(train_time - test_time :train_time));
% title('Target Function');
% 
% suptitle('Dyanmics of internal nodes during training');

%% Computations
disp('Computations');

% W_out - 1 * N

W_out = (inv(XAll * XAll' + lambda_out * eye(N)) * XAll * PAll')';
rmse_out = sqrt(mean((W_out * XAll - PAll).^2));
fprintf('RMSE for W_out is %f\n', rmse_out);

% W_trained
% temp = atanh(XAll) - repmat(bias, 1, num_patterns * learn_time);
temp = W * XoldAll + W_in * PAll;
W_trained = (pinv(XoldAll * XoldAll' + lambda_w * eye(N)) * XoldAll * (temp)')';

rmse_w = sqrt(mean( mean((temp - W_trained * XoldAll).^ 2, 1)));
fprintf('RMSE for W is %f\n', mean(rmse_w));

% Conceptor matrix
C = zeros(num_patterns, N, N);

for p = 1:num_patterns
    X = XAll(:, (p-1) * learn_time + 1 : p * learn_time);
    R = X * X' / (learn_time);
    
    [U, E, V] = svd(R);
    S = E * pinv(E + alpha^(-2) * eye(N));
    C(p, :, :) = U * S * U';
end

%% Exploitation
YAll = zeros(num_patterns, test_time);
XTestAll = zeros(num_patterns, test_time, N);
PAll = zeros(num_patterns, test_time, 1);

for p = 1:num_patterns
    y = zeros(test_time, 1);
    XTest = zeros(test_time, N);
    x = 0.5 * randn(N, 1);
    P = zeros(test_time, 1);
    C_p = reshape(C(p, :, :), [N, N]);
    d = patterns{p};

    for t=1:washout_time + test_time
        x = tanh(W_trained * x + bias);
        x = C_p * x;
        if t>washout_time
            y(t - washout_time) = W_out * x;
            XTest(t - washout_time, :) = x;
            P(t - washout_time) = d(t);
        end
    end
    YAll(p, :) = y;
    XTestAll(p, :, :) = XTest;
    PAll(p, :, :) = P;
end

% Plot the node dynamics during exploitation
% figure();
% for i = 1:n_disp
%     subplot(2,4,i);
%     fig_name = sprintf('Node %d', ind(i));
%     plot(XTest(:, ind(i)));
%     title(fig_name);
% end
% 
% subplot(2,4,5);
% plot(P);
% title('Target Function');
% suptitle('Dyanmics of internal nodes during testing');

% Target .vs. Obtained
figure();
for p = 1:num_patterns
    subplot(2, num_patterns, p);
    plot(PAll(p, :));
    title(sprintf('Pattern %d', p));
    subplot(2, num_patterns, num_patterns + p);
    plot(YAll(p, :));
    title('Obtained Pattern');
end
suptitle('Testing Phase');