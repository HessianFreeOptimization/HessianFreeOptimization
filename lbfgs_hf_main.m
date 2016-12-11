clc; clear; close all;
maxIter = 100;
% network structure
layersizes = [25 30];
layertypes = {'logistic', 'logistic', 'softmax'};

[llrecord, errrecord, params] = gd_train('adam', 150, layersizes, layertypes);
[llrecord2, errrecord2, ~] = lbfgs_train(60, layersizes, layertypes, params);
% [llrecord2, errrecord2, ~] = hf_train(60, layersizes, layertypes, params);
llrecord = [llrecord; llrecord2];
errrecord = [errrecord; errrecord2];
%%
iters = size(llrecord, 1) - 1;
fig = figure('color', [1 1 1]);
subplot(1,2,1);
semilogy(0:iters, -llrecord(:,1),'rx-');
ylim([0 max(-llrecord(:,1))]);
hold on;
semilogy(0:iters, -llrecord(:,2),'bx-');
ylim([0 max(-llrecord(:,2))]);
title('log-likelihood');

subplot(1,2,2);
plot(0:iters, errrecord(:,1),'rx-');
hold on;
plot(0:iters, errrecord(:,2),'bx-');
title('error');
ylim([0 1]);
set(fig, 'Position', [10, 10, 1000, 400]);
