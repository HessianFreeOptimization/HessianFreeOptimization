clc; clear; close all;
% network structure
params.layersizes = [100 30];
params.layertypes = {'logistic', 'logistic', 'softmax'};
params.weight_decay = 2e-4;

% load datasets.
indata = loadMNISTImages('raw_data/train-images.idx3-ubyte');
y = loadMNISTLabels('raw_data/train-labels.idx1-ubyte');
y(y==0) = 10;
intest = loadMNISTImages('raw_data/t10k-images.idx3-ubyte');
yt = loadMNISTLabels('raw_data/t10k-labels.idx1-ubyte');
yt(yt==0) = 10;
outdata = zeros(10,size(indata,2));
for i = 1:10
	outdata(i,:) = (y == i);
end
outtest = zeros(10,size(intest,2));
for i = 1:10
	outtest(i,:) = (yt == i);
end
params.indata = indata(:, 1:5000);
params.outdata = outdata(:, 1:5000);
params.intest = intest(:, 1:1000);
params.outtest = outtest(:, 1:1000);
% params.indata = indata;
% params.outdata = outdata;
% params.intest = intest;
% params.outtest = outtest;

% fig1 = figure('color', [1 1 1]);
% set(fig1, 'Position', [10, 10, 1000, 400]);
% training
iters = 1000;
trials = 1;

global eval_f;
global eval_g;

records = cell(trials, 1);
for trial = 1:trials
    global eval_f;
    eval_f = 0;
    global eval_g;
    eval_g = 0;
    [llrecord, errrecord, weights, eval_fs, eval_gs] = lbfgs_mom_train(iters, params);
    records{trial}.llrecord = llrecord;
    records{trial}.errrecord = errrecord;
    records{trial}.weights = weights;
    records{trial}.eval_fs = eval_fs;
    records{trial}.eval_gs = eval_gs;
end
save(sprintf('./saved/single_lbfgs-mom-for-%d_iters-%d_trials.mat', iters, trials), 'records');


%%
load('saved/single_lbfgs-mom-for-1000_iters-1_trials.mat');
errrecord = records{1,1}.errrecord;
llrecord = -records{1,1}.llrecord;
fig = figure('color',[1 1 1]);
subplot(1,2,1);
plot(llrecord(:,1),'r');
hold on;
plot(llrecord(:,2),'b');
ylim([0 6]);
title('loss');

subplot(1,2,2);
plot(errrecord(:,1),'r');
hold on;
plot(errrecord(:,2),'b');
ylim([0 1]);
title('error');

set(fig, 'Position', [10, 10, 1000, 400]);

