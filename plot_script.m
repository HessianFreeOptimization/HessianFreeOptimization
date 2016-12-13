close all; clear all; clc;
%%
close all; clc;
fig1 = figure(1);
algorithms = {'gradient descent', 'gradient descent with backtracking', 'nesterov accelerated gradient','momentum',...
    'adagrad','RMSprop','adadelta','adam',...
    'lbfgs', 'hessian-free', 'adamAndLbfgs'};
lines =  {'-', '-', '-', ...
    '-', '-', '-', '-', '-', ...
    '--', '--', '-.'};
trials = [1, 1, 1, 1, ...
    1, 1, 1, 1, ...
    1, 1, 1];
base_iters = 6000;
iters = [base_iters, base_iters, base_iters, base_iters, ...
    base_iters, base_iters, base_iters, base_iters, ...
    base_iters, 3000, base_iters];
max1 = 0;
algorithms_plot = {};
algorithms_plot_count = 0;
plots1 = [];
plots2 = [];
colors = distinguishable_colors(length(algorithms));

labelx = 'epoch'; labelx_ind = 1;
% labelx = 'evals of objective'; labelx_ind = 2;
% labelx = 'evals of gradient'; labelx_ind = 3;

for al_iter = 1 : length(algorithms)
    algorithm = algorithms{al_iter};
    file_name = sprintf('./saved/single_%s-for-%d_iters-%d_trials.mat', algorithm, iters(al_iter), trials(al_iter));
    if exist(file_name, 'file') == 2
        load(file_name);
        [max1, plots1, plots2] = plot_curve(false, records, fig1, max1, plots1, plots2, colors(al_iter, :), lines{al_iter}, labelx_ind);
        hold on;
        algorithms_plot_count = algorithms_plot_count + 1;
        algorithms_plot{algorithms_plot_count} = algorithm;
    end
end

switch labelx_ind
    case 1
        subplot(1, 2 ,1);
    case 2
        subplot(2, 1 ,1);
    case 3
        subplot(2, 1 ,1);
end
grid on;
legend(plots1, algorithms_plot);
ylim([0, max1]);
xlabel(labelx);
ylabel('objective value');
title('objective');

switch labelx_ind
    case 1
        subplot(1, 2 ,2);
    case 2
        subplot(2, 1 ,2);
    case 3
        subplot(2, 1 ,2);
end
grid on;
legend(plots2, algorithms_plot);
ylim([0 1]);
xlabel(labelx);
ylabel('classification error');
title('training error');