close all; clear all; clc;
%%
close all; clc;
fig1 = figure(1);
% fig1.Position = [100, 100, 310, 240];
fig1.Position = [100, 100, 400, 300];
fig2 = figure(2);
fig2.Position = [100, 100, 400, 300];
algorithms = {'gradient descent', 'gradient descent with backtracking', 'momentum', 'nesterov accelerated gradient',...
    'adagrad','RMSprop','adadelta','adam',...
    'lbfgs', 'hessian-free', 'lbfgs-mom', ...
    'hessian-free-Damp-BK', 'hessian-free-noBK'};
algorithms_names = {'gradient descent', 'gradient descent with backtracking', 'gradient descent with momentum', 'Nesterov accelerated gradient descent',...
    'AdaGrad','RMSprop','AdaDelta','Adam',...
    'L-BFGS', 'Hessian-free', 'L-BFHS with momentum', ...
    'Hessian-free(Damp, BT)', 'Hessian-free(Damp, noBT)'};
lines =  {'-', '-', '-', ...
    '-', '-', '-', '-', '-', ...
    '--', '--', '--', ...
    '--', '--'};
trials = [10, 5, 5, 5, ...
    5, 5, 5, 5, ...
    10, 5, 1, ...
    1, 1];
base_iters = 6000;
iters = [base_iters, base_iters, base_iters, base_iters, ...
    base_iters, base_iters, base_iters, base_iters, ...
    base_iters, base_iters, 1000, ...
    base_iters, base_iters];
max1 = 0;
min1 = Inf;
algorithms_plot = {};
algorithms_plot_count = 0;
plots1 = [];
plots2 = [];
colors = distinguishable_colors(length(algorithms));

% labelx = 'epoch'; labelx_ind = 1;
labelx = 'evals of objective'; labelx_ind = 2;
% labelx = 'evals of gradient'; labelx_ind = 3;

% selected = [1, 2, 9, 10];
% selected = [1, 3, 4, 5, 6, 7, 8, 9, 10];
% selected = [9, 10];

selected = [10, 12, 13]; % hf

for index = 1 : length(selected)
    al_iter = selected(index);
    algorithm = algorithms{al_iter};
    file_name = sprintf('./saved/single_%s-for-%d_iters-%d_trials.mat', algorithm, iters(al_iter), trials(al_iter));
    if exist(file_name, 'file') == 2
        load(file_name);
        [max1, min1, plots1, plots2] = plot_curve(false, false, records, fig1, fig2, max1, min1, plots1, plots2, colors(al_iter, :), lines{al_iter}, labelx_ind);
    end
end

for index = 1 : length(selected)
    al_iter = selected(index);
    algorithm = algorithms{al_iter};
    file_name = sprintf('./saved/single_%s-for-%d_iters-%d_trials.mat', algorithm, iters(al_iter), trials(al_iter));
    if exist(file_name, 'file') == 2
        load(file_name);
        [max1, min1, plots1, plots2] = plot_curve(true, false, records, fig1, fig2, max1, min1, plots1, plots2, colors(al_iter, :), lines{al_iter}, labelx_ind);
        hold on;
        algorithms_plot_count = algorithms_plot_count + 1;
        algorithms_plot{algorithms_plot_count} = algorithms_names{al_iter};;
    end
end

% switch labelx_ind
%     case 1
%         subplot(1, 2 ,1);
%     case 2
%         subplot(2, 1 ,1);
%     case 3
%         subplot(2, 1 ,1);
% end
figure(1);
grid on;
hLegend = legend(plots1, algorithms_plot);
set(hLegend.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));

% ylim([min1, max1]);
% xlim([0 1000]);
xlabel(labelx);
ylabel('$f - \hat{f^*}$', 'Interpreter','LaTex');
switch labelx_ind
%     case 1
%         subplot(1, 2 ,1);
    case 2
        xlim([0 1.6e5]);
    case 3
        xlim([0 1.2e5]);
end

figure(2);
grid on;
hLegend = legend(plots2, algorithms_plot);
set(hLegend.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));
ylim([0 1]);
switch labelx_ind
%     case 1
%         subplot(1, 2 ,1);
    case 2
        xlim([0 1.6e5]);
    case 3
        xlim([0 1.2e5]);
end
xlabel(labelx);
ylabel('classification error');
% title('training/test error');