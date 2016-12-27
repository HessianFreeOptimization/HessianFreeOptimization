close all; clear; clc;

fig = figure('color',[1 1 1]);
fig.Position = [1800,10,1000,400];
% fig1 = figure(1);
% fig1.Position = [100, 100, 400, 300];
% fig2 = figure(2);
% fig2.Position = [100, 100, 400, 300];
algorithms = {'gradient descent', 'fixstep-lbfgs', 'momentum-lbfgs'};
algorithms_names = {'gradient descent', 'fixstep L-BFGS', 'momentum L-BFGS'};
lines =  {'-', '--', '--'};
trials = [1,10,10];
base_iters = 6000;
iters = [base_iters, base_iters, base_iters];
max1 = 0;
min1 = Inf;
algorithms_plot = {};

plots1 = [];
plots2 = [];
colors = distinguishable_colors(length(algorithms));

selected = [2, 3];

for index = 1 : length(selected)
    al_iter = selected(index);
    algorithm = algorithms{al_iter};
    file_name = sprintf('./saved/%s-for-%d_iters-%d_trials.mat', algorithm, iters(al_iter), trials(al_iter));
    %if exist(file_name, 'file') == 2
    load(file_name);
    
    
    for i = 1:trials(al_iter)
        ll = -records{i,1}.llrecord(:,1);
        %trial_curve = plot(ll, 'Color', colors(al_iter,:), 'LineWidth', 1);
        min1 = min(min1,min(ll));
        %hold on;
    end
end

subplot(1,2,1);
alpha = 0.3;

for index = 1 : length(selected)
    al_iter = selected(index);
    algorithm = algorithms{al_iter};
    file_name = sprintf('./saved/%s-for-%d_iters-%d_trials.mat', algorithm, iters(al_iter), trials(al_iter));

    load(file_name);
    
    llrecord_mean = zeros(size(records{1,1}.llrecord(:,1)));
    
    %subplot(1,2,1);
    for i = 1:trials(al_iter)
        ll = -records{i}.llrecord(:,1) - min1;
        trial_curve = semilogy(0:iters(al_iter), ll, 'Color', colors(al_iter,:), 'LineWidth', 1);
        trial_curve.LineStyle = '-';
        trial_curve.Color(4) = alpha;
        %min1 = min(min1,min(ll));
        hold on;
        llrecord_mean = llrecord_mean + records{i,1}.llrecord(:,1);
    end
    llrecord_mean = -llrecord_mean/trials(al_iter);
    trial_curve = semilogy(0:iters(al_iter), ll, 'Color', colors(al_iter,:), 'LineWidth', 2.5);
    trial_curve.LineStyle = '--';
    plots1 = [plots1, trial_curve];
    algorithms_plot = [algorithms_plot, algorithms_names(al_iter)];
end


%

grid on;
hLegend = legend(plots1, algorithms_plot);
set(hLegend.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));
xlabel('epoch');
ylabel('$f - \hat{f^*}$', 'Interpreter','LaTex');

%%

for index = 1 : length(selected)
    al_iter = selected(index);
    algorithm = algorithms{al_iter};
    file_name = sprintf('./saved/%s-for-%d_iters-%d_trials.mat', algorithm, iters(al_iter), trials(al_iter));
    if exist(file_name, 'file') == 2
        load(file_name);
        [max1, plots1, plots2] = plot_curve(true, false, false, records, fig1, fig2, max1, Inf, plots1, plots2, colors(al_iter, :), lines{al_iter}, 1);
        hold on;
        algorithms_plot_count = algorithms_plot_count + 1;
        algorithms_plot{algorithms_plot_count} = algorithms_names{al_iter};
    end
end




figure(1);
grid on;
hLegend = legend(plots1, algorithms_plot);
set(hLegend.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));
xlabel('epoch');
ylabel('$f - \hat{f^*}$', 'Interpreter','LaTex');


figure(2);
grid on;
hLegend = legend(plots2, algorithms_plot);
set(hLegend.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));
ylim([0 1]);

xlabel('epoch');
ylabel('classification error');
%EOF.
