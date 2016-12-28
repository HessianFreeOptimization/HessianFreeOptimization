close all; clear; clc;

fig = figure('color',[1 1 1]);
%fig.Position = [1800,10,1000,400];
fig.Position = [10,10,1500,600];

algorithms = {'gradient descent', 'fixstep-lbfgs', 'momentum-lbfgs'};
algorithms_names = {'gradient descent', 'fixstep L-BFGS', 'momentum L-BFGS'};
lines =  {'-', '--', '--'};
trials = [1,2,2];
base_iters = 6000;
iters = [base_iters, 10, 10];
max1 = 0;
min1 = Inf;
algorithms_plot = {};

plots1 = [];
plots2 = [];
colors = distinguishable_colors(length(algorithms));

selected = [2, 3];


% count minimum f value.
for index = 1 : length(selected)
    al_iter = selected(index);
    algorithm = algorithms{al_iter};
    file_name = sprintf('./saved/%s-for-%d_iters-%d_trials-0.0100_eta.mat', algorithm, iters(al_iter), trials(al_iter));
    load(file_name);
    
    for i = 1:trials(al_iter)
        ll = -records{i,1}.llrecord(:,1);
        min1 = min(min1,min(ll));
    end
end

subplot(1,2,1);
alpha = 0.3;

% plot f - f^\star
for index = 1 : length(selected)
    al_iter = selected(index);
    algorithm = algorithms{al_iter};
    file_name = sprintf('./saved/%s-for-%d_iters-%d_trials-0.0100_eta.mat', algorithm, iters(al_iter), trials(al_iter));
    load(file_name);
    
    llrecord_mean = zeros(size(records{1,1}.llrecord(:,1)));
    
    for i = 1:trials(al_iter)
        ll = -records{i,1}.llrecord(:,1) - min1;
        trial_curve = semilogy(0:iters(al_iter), ll, 'Color', colors(al_iter,:), 'LineWidth', 1);
        trial_curve.LineStyle = '-';
        trial_curve.Color(4) = alpha;
        hold on;
        llrecord_mean = llrecord_mean + ll;
    end
    llrecord_mean = llrecord_mean/trials(al_iter);
    trial_curve = semilogy(0:iters(al_iter), llrecord_mean, 'Color', colors(al_iter,:), 'LineWidth', 2.5);
    hold on;
    trial_curve.LineStyle = '--';
    plots1 = [plots1, trial_curve];
    algorithms_plot = [algorithms_plot, algorithms_names(al_iter)];
end

grid on;
hLegend = legend(plots1, algorithms_plot);
set(hLegend.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));
xlabel('epoch');
ylabel('$f - \hat{f^*}$', 'Interpreter','LaTex');




% count minimum step_size value.
min2 = Inf;
for index = 1 : length(selected)
    al_iter = selected(index);
    algorithm = algorithms{al_iter};
    file_name = sprintf('./saved/%s-for-%d_iters-%d_trials-0.0100_eta.mat', algorithm, iters(al_iter), trials(al_iter));
    load(file_name);
    
    for i = 1:trials(al_iter)
        step_size = records{i,1}.step_size(1:end,1);
        min2 = min(min2,min(step_size));
    end
end

subplot(1,2,2);

% plot step_size - step_size^\star
for index = 1 : length(selected)
    al_iter = selected(index);
    algorithm = algorithms{al_iter};
    file_name = sprintf('./saved/%s-for-%d_iters-%d_trials-0.0100_eta.mat', algorithm, iters(al_iter), trials(al_iter));
    load(file_name);
    
    step_mean = zeros(size(records{1,1}.step_size(:,1)));
    
    for i = 1:trials(al_iter)
        step_size = records{i,1}.step_size(:,1) - min2;
        if true
        trial_curve = semilogy(0:iters(al_iter), step_size, 'Color', colors(al_iter,:), 'LineWidth', 1);
        trial_curve.LineStyle = '-';
        trial_curve.Color(4) = alpha;
        hold on;
        end
        step_mean = step_mean + step_size;
    end
    step_mean = step_mean/trials(al_iter);
    trial_curve = semilogy(0:iters(al_iter), step_mean, 'Color', colors(al_iter,:), 'LineWidth', 2.5);
    hold on;
    trial_curve.LineStyle = '--';
    plots2 = [plots2, trial_curve];
    %algorithms_plot = [algorithms_plot, algorithms_names(al_iter)];
end

grid on;
hLegend = legend(plots2, algorithms_plot);
set(hLegend.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));
xlabel('epoch');
ylabel('$stepsize - \hat{stepsize^*}$', 'Interpreter','LaTex');




%%

%figure('Color',[1 1 1]);

grid on;
hLegend = legend(plots2, algorithms_plot);
set(hLegend.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));
ylim([0 1]);

xlabel('epoch');
ylabel('classification error');
%EOF.
