clc; clear; close all;
% network structure
% TODO:
%params.layersizes = [100 30];
params.layersizes = [30];
%params.layertypes = {'logistic', 'logistic', 'softmax'};
params.layertypes = {'logistic', 'softmax'};
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

% training
iters = 300;
trials = 1;

global eval_f;
global eval_g;

records = cell(trials, 1);
for trial = 1:trials
    global eval_f;
    eval_f = 0;
    global eval_g;
    eval_g = 0;
    params.trial = trial;
    [llrecord, errrecord, weights, eval_fs, eval_gs] = fixstep_lbfgs(iters, params);
    records{trial}.llrecord = llrecord;
    records{trial}.errrecord = errrecord;
    records{trial}.weights = weights;
    records{trial}.eval_fs = eval_fs;
    records{trial}.eval_gs = eval_gs;
end

save(sprintf('./saved/fixstep-lbfgs-for-%d_iters-%d_trials.mat', iters, trials), 'records');
