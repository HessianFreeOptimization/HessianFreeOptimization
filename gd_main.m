clc; clear; close all;
maxIter = 100;

% network structure
layersizes = [25 30];
layertypes = {'logistic', 'logistic', 'softmax'};
algorithm = {'gradient descent','momentum','nesterov accelerated gradient',...
    'adagrad','RMSprop','adadelta','adam'};
algorithm = algorithm{7};

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

[llrecord, errrecord, params] = gd_train(algorithm, maxIter, layersizes, layertypes, indata, outdata, intest, outtest);

%%
plot_curve(llrecord,  errrecord);
