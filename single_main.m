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
params.indata = indata;
params.outdata = outdata;
params.intest = intest;
params.outtest = outtest;
global eval_f;
eval_f = 0;
global eval_g;
eval_g = 0;

%% training
iters = 4000;
[llrecord, errrecord, weights, eval_fs, eval_gs] = gd_train('adam', iters, params);
% [llrecord, errrecord, weights, eval_fs, eval_gs] = hf_train(iters, params, weights);
%%
plot_curve(llrecord,  errrecord, eval_fs);
