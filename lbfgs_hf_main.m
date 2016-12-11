clc; clear; close all;
maxIter = 100;
%% network structure
layersizes = [25 30];
layertypes = {'logistic', 'logistic', 'softmax'};

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

%% training
[llrecord, errrecord, params] = gd_train('adam', 150, layersizes, layertypes, indata, outdata, intest, outtest);
[llrecord2, errrecord2, ~] = lbfgs_train(60, layersizes, layertypes, indata, outdata, intest, outtest, params);
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
