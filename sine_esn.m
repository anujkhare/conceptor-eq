%% An attempt at a sinewave generating echo state network

clc;
clear;

disp ('Displaying the teaching signal');
train_time = 300;
washout_time = 100;
test_time = 50;

sin_func = @(n) 1/2 .* sin(n/2);

b = 0.1;
f_act = @(x) tanh(x + b);
f_out = @(x) x;
f_out_inv = @(x) x;

% A train_time timestep sine teaching signal
n = [1: train_time+1]';

d = sin_func(n);
plot(n, d);
title('Target function');

%% Initialize reservoir
disp ('Creating reservoir');
reservoir_size = 30;
output_size = 1;

% Weights
W = (rand(reservoir_size, reservoir_size) - 0.5);
spectral_radius = max(abs(eig(W)));
W = W / spectral_radius;

alpha = 1.5;
W = alpha * W;

W_fb = (rand(reservoir_size, output_size) - 0.5) * 3;

n_disp = 4;
ind = randperm(reservoir_size);
ind = ind(1:n_disp);
%% Testing the self dynamics of the reservoir for random start states

x = rand(reservoir_size,1) - 0.5;
X = zeros(washout_time,reservoir_size);

for i = 1:washout_time
    X(i, :) = x;
    x = f_act(W * x);    % x(i+1)
end

p = figure();
for i=1:n_disp
    subplot(2,4,i);
    fig_name = sprintf('Node %d', ind(i));
    plot(X(:,ind(i)));
    title(fig_name);
end
suptitle ('Self dynamics of the internal nodes');
% saveas(p, ['Self dynamics of the internal nodes', '.png'], 'png');

%% Sampling Stage
disp ('Sampling Stage');
M = zeros(train_time - washout_time, reservoir_size);
T = zeros(train_time - washout_time, output_size);
x = zeros(reservoir_size, 1);

X = zeros(train_time, reservoir_size);

% here x(1) will be 0 (x(0) = d(0) = 0)
% We want to store x(n) with d(n), hence I have reversed the order
for i = 1:train_time
    x = f_act(W * x + W_fb * d(i));    % x(i+1)
    X(i, :) = x;
    if (i > washout_time)
        M(i - washout_time, :) = x;
        T(i - washout_time, :) = f_out_inv(d(i+1));
    end

end

p = figure();
for i=1:n_disp
    subplot(2,4,i);
    fig_name = sprintf('Node %d', ind(i));
    plot(X(:,ind(i)));
    title(fig_name);
end

subplot(2,4,5);
plot(d(1:300));
title('Target function');

suptitle ('Dynamics of the internal nodes during training');
% saveas(p, ['Dynamics of the internal nodes during training', '.png'], 'png');

%% Computations
disp ('Computation Stage and Testing')

% lambda = 5e-8;
lambda = 0;

W_out = pinv(M' * M + lambda * eye(reservoir_size)) * M' * T;
W_out = W_out';

fprintf('Mean absolute size of W_out: %f\n', mean(abs(W_out)));

mse = mean((T - M * W_out').^2);
fprintf('Training MSE: %f\n', mse);

%% Testing
y = zeros(test_time, output_size);
% x = M(end, :)'; % internal state at the end of teacher forcing
y(1) = f_out(W_out * x);

X_test = zeros(test_time, reservoir_size);
X_test(1, :) = x;


for t = 2: test_time
    x = f_act(W * x + W_fb * y(t-1));
    y(t) = f_out(W_out * x);
    X_test(t, :) = x;
end

% Plots
n_test = [train_time + 1 : train_time + test_time]';
d_test = sin_func(n_test);

figure();
subplot(2,1,1);
plot (n_test, d_test);
title('Target Pattern');
subplot(2,1,2);
plot (n_test, y);
title('Obtained Patter');
suptitle('Testing Phase');


p = figure();
for i=1:n_disp
    subplot(2,4,i);
    fig_name = sprintf('Node %d', ind(i));
    plot(X_test(:,i));
    title(fig_name);
end

subplot(2,4,5);
plot(d_test);
title('Target function');
suptitle ('Dynamics of the internal nodes during testing');
% saveas(p, ['Dynamics of the internal nodes during testing', '.png'], 'png');