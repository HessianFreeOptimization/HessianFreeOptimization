clc; clear; close all;
%% network structure
params.layersizes = [100 30];
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
%% training
% iters = 6000;
iters1 = 500;
iters2 = 5500;
algorithms = {'gradient descent','momentum','nesterov accelerated gradient',...
    'adagrad','RMSprop','adadelta','adam', 'gradient descent with backtracking'};
global eval_f;
global eval_g;
al_iter = 7;
algorithm = algorithms{al_iter};
records = cell(1, 1);

trial = 1;
eval_f = 0;
eval_g = 0;
[llrecord, errrecord, weights, eval_fs, eval_gs] = lbfgs_train(iters1, params);
records{trial}.llrecord = llrecord;
records{trial}.errrecord = errrecord;
records{trial}.weights = weights;
records{trial}.eval_fs = eval_fs;
records{trial}.eval_gs = eval_gs;


eval_f = records{trial}.eval_fs(end, 1);
eval_g = records{trial}.eval_gs(end, 1);
[llrecord, errrecord, weights, eval_fs, eval_gs] = gd_train(algorithm, iters2, params, weights);
records{trial}.llrecord = [records{trial}.llrecord; llrecord];
records{trial}.errrecord = [records{trial}.errrecord; errrecord];
records{trial}.weights = weights;
records{trial}.eval_fs = [records{trial}.eval_fs; eval_fs];
records{trial}.eval_gs = [records{trial}.eval_gs; eval_gs];

save(sprintf('./saved/single_%sAndLbfgs-for-%d_iters-%d_trials.mat', algorithm, iters1 + iters2, trial), 'records');