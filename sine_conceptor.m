%% A basic sine wave generator using conceptor logic!

% clc;
clear;

train_time = 1500;
washout_time = 500;
test_time = 50;
learn_time = train_time - washout_time;

sin_func = @(n) sin(n / 4);
d = sin_func([1: train_time+test_time]');
disp('Target function');
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

% N * L
X = zeros(N, learn_time);
Xold = zeros(N, learn_time); % the one step delayed
x = zeros(N, 1);

% P - 1 * L
P = zeros(1, learn_time);

for i = 1:train_time
    x_ = x;
    x = tanh(W * x + W_in * d(i) + bias); % x(i) = tanh(W*x(i-1) + Win * d(i) + b)
    if i > washout_time
       X(:, i - washout_time) = x;
       Xold(:, i - washout_time) = x_;
       P(i - washout_time) = d(i);
    end
end

figure();
for i = 1:n_disp
    subplot(2,4,i);
    fig_name = sprintf('Node %d', ind(i));
    plot(X(ind(i), end-test_time:end));
    title(fig_name);
end

subplot(2,4,5);
plot(d(train_time - test_time :train_time));
title('Target Function');

suptitle('Dyanmics of internal nodes during training');

%% Computations
disp('Computations');

% W_out - 1 * N

W_out = (inv(X * X' + lambda_out * eye(N)) * X * P')';
rmse_out = sqrt(mean((W_out * X - P).^2));
fprintf('RMSE for W_out is %f\n', rmse_out);

% W_trained
% temp = atanh(X) - repmat(bias, 1, learn_time);
temp = W * Xold + W_in * P;
W_trained = (pinv(Xold * Xold' + lambda_w * eye(N)) * Xold * (temp)')';

rmse_w = sqrt(mean( mean((temp - W_trained * Xold).^ 2, 1)));
fprintf('RMSE for W is %f\n', mean(rmse_w));

%% Exploitation
y = zeros(test_time, 1);
X_test = zeros(test_time, N);
x = 0.5 * randn(N, 1);
P = zeros(test_time, 1);

for t=1:washout_time + test_time
    x = tanh(W_trained * x + bias);
    if t>washout_time
        y(t - washout_time) = W_out * x;
        X_test(t - washout_time, :) = x;
        P(t - washout_time) = sin_func(t);
    end
end

% Plot the node dynamics during exploitation
figure();
for i = 1:n_disp
    subplot(2,4,i);
    fig_name = sprintf('Node %d', ind(i));
    plot(X_test(:, ind(i)));
    title(fig_name);
end

subplot(2,4,5);
plot(P);
title('Target Function');
suptitle('Dyanmics of internal nodes during testing');

% Target .vs. Obtained
figure();
subplot(2,1,1);
plot(P);
title('Target Pattern');
subplot(2,1,2);
plot(y);
ylim([-1, 1]);
title('Obtained Pattern');
suptitle('Testing Phase');