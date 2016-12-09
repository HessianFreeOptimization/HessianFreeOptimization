% SGD_mnist.m
% Load Training Data
clc; clear; close all;

X = loadMNISTImages('raw_data/train-images.idx3-ubyte');
y = loadMNISTLabels('raw_data/train-labels.idx1-ubyte');
Xt = loadMNISTImages('raw_data/t10k-images.idx3-ubyte');
yt = loadMNISTLabels('raw_data/t10k-labels.idx1-ubyte');

X = Xt';
X = X(1:6000,:);
y = yt;
y = y(1:6000,:);
%%
clc; clear; close all;
load('ex4data1.mat');
% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)



%  Initialize the weights of the neural network
%  (randInitializeWeights.m)

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%  To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc".
fprintf('Training Neural Network... \n')

%options = optimset('MaxIter', 10);
options.MaxIter = 100;
options.HessUpdate = 'lbfgs';
%options.HessUpdate = 'steepdesc';
options.Display = 'iter';
options.GradObj = 'on';

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

                               
[nn_params, cost] = fminlbfgs( costFunction, initial_nn_params, options);
                               
                               
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
%[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);




% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% prediction
pred = predict(Theta1, Theta2, X);

fprintf('Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);
loss = cost;
save('loss_lbfgs.mat', 'loss');


%%
load('loss_lbfgs.mat');
figure('color', [1 1 1]);
plot(0:length(loss)-1,loss);



