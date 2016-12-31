clc; clear; close all;
% network structure
% TODO:
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
params.intest = intest(:, 1:5000);
params.outtest = outtest(:, 1:5000);

% training
% iters = 5000;
% trials = 10;
iters = 5000;
trials = 10;
global eval_f;
global eval_g;

algorithms = {'gradient descent', 'fixstep-lbfgs', 'momentum-lbfgs', 'lbfgs'};
algorithm = algorithms{2};

%for eta = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
for eta = [0.001, 0.01, 0.1]
    records = cell(trials, 1);
    if strcmp(algorithm, 'gradient descent')
        logfile = sprintf('./saved/gd-for-%d_iters-%d_trials-%.4f_eta.log', iters, trials, eta);
    elseif strcmp(algorithm, 'fixstep-lbfgs')
        logfile = sprintf('./saved/fixstep-lbfgs-for-%d_iters-%d_trials-%.4f_eta.log', iters, trials, eta);
    elseif strcmp(algorithm, 'lbfgs')
        logfile = sprintf('./saved/lbfgs-for-%d_iters-%d_trials-%.4f_eta.log', iters, trials, eta);
    elseif strcmp(algorithm, 'momentum-lbfgs')
        logfile = sprintf('./saved/momentum-lbfgs-for-%d_iters-%d_trials-%.4f_eta.log', iters, trials, eta);
    end
    fid = fopen(logfile, 'w');
    for trial = 1:trials
        eval_f = 0;
        eval_g = 0;
        params.trial = trial;       
        [llrecord, errrecord, weights, eval_fs, eval_gs, step_size] = optimization_method(algorithm, eta, iters, fid, params);
        records{trial}.llrecord = llrecord;
        records{trial}.errrecord = errrecord;
        records{trial}.weights = weights;
        records{trial}.eval_fs = eval_fs;
        records{trial}.eval_gs = eval_gs;
        records{trial}.step_size = step_size;
    end
    fclose(fid);
    if strcmp(algorithm, 'gradient descent')
        save(sprintf('./saved/gd-for-%d_iters-%d_trials-%.4f_eta.mat', iters, trials, eta), 'records');
    elseif strcmp(algorithm, 'fixstep-lbfgs')
        save(sprintf('./saved/fixstep-lbfgs-for-%d_iters-%d_trials-%.4f_eta.mat', iters, trials, eta), 'records');
    elseif strcmp(algorithm, 'momentum-lbfgs')
        save(sprintf('./saved/momentum-lbfgs-for-%d_iters-%d_trials-%.4f_eta.mat', iters, trials, eta), 'records');
    elseif strcmp(algorithm, 'lbfgs')
        save(sprintf('./saved/lbfgs-for-%d_iters-%d_trials-%.4f_eta.mat', iters, trials, eta), 'records');
    end
end

%EOF.
