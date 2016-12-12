clc; clear; close all;
%% network structure
params.layersizes = [200 30];
params.layertypes = {'logistic', 'logistic', 'softmax'};
params.weight_decay = 2e-4;

%% load datasets.
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
% params.indata = indata(:, 1000);
% params.outdata = outdata(:, 1000);
% params.intest = intest(:, 1000);
% params.outtest = outtest(:, 1000);
params.indata = indata;
params.outdata = outdata;
params.intest = intest;
params.outtest = outtest;

% fig1 = figure('color', [1 1 1]);
% set(fig1, 'Position', [10, 10, 1000, 400]);
%% training
iters = 3000;
trials = 20;

global eval_f;
global eval_g;

records = cell(trials, 1);
for trial = 1:trials
    eval_f = 0;
    eval_g = 0;
    [llrecord, errrecord, weights, eval_fs, eval_gs] = hf_train(iters, params);
    records{trial}.llrecord = llrecord;
    records{trial}.errrecord = errrecord;
    records{trial}.weights = weights;
    records{trial}.eval_fs = eval_fs;
    records{trial}.eval_gs = eval_gs;
end
save(sprintf('./saved/single_hf_for_%d_iter_test.mat', iters), 'records');
% [llrecord, errrecord, weights, eval_fs, eval_gs] = hf_train(iters, params, weights);
fig1 = figure(1);
plot_curve(records, fig1);

records = cell(trials, 1);
for trial = 1:trials
    global eval_f;
    eval_f = 0;
    global eval_g;
    eval_g = 0;
    [llrecord, errrecord, weights, eval_fs, eval_gs] = lbfgs_train(iters, params);
    records{trial}.llrecord = llrecord;
    records{trial}.errrecord = errrecord;
    records{trial}.weights = weights;
    records{trial}.eval_fs = eval_fs;
    records{trial}.eval_gs = eval_gs;
end
save(sprintf('./saved/single_lbfgs_for_%d_iter_test.mat', iters), 'records');
% [llrecord, errrecord, weights, eval_fs, eval_gs] = hf_train(iters, params, weights);
fig1 = figure(1);
plot_curve(records, fig1);
%% plotting
% close all
% fig1 = figure(1);
% plot_curve(records, fig1);