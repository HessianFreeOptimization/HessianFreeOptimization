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

%% training
gd_iters = 4;
[llrecord, errrecord, weights] = gd_train('adam', gd_iters, params);
save(sprintf('adam-%d.mat', gd_iters), 'llrecord', 'errrecord', 'weights');
hf_iters = 5;
[llrecord2, errrecord2, weights2] = hf_train(hf_iters, params, weights);
% [llrecord2, errrecord2, ~] = hf_train(60, layersizes, layertypes, params);
save(sprintf('hf-%d-%d.mat', gd_iters, hf_iters), 'llrecord2', 'errrecord2', 'weights2');
llrecord = [llrecord; llrecord2];
errrecord = [errrecord; errrecord2];
%%
plot_curve(llrecord,  errrecord);