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



algname = 'Adadelta';



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
%options.HessUpdate = 'lbfgs';
options.HessUpdate = 'steepdesc';
options.Display = 'iter';
options.GradObj = 'on';

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);


eta = 0.01;
gamma = 0.9;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;                               
if algname == 'Adadelta'
    x = initial_nn_params;
    m = size(x,1);

%     cost = [];
%     for i = 1:300
%         [fval, grad] = costFunction(x);
%         cost = [cost, fval];
%         fprintf('%d\t%.5f\n',i-1,fval)
%         [xn,Eg2,Et2] = gradupdate(grad, x,xoo,Eg2,Et2,gamma,epsilon);
%         xoo = x;
%         x = xn;
%     end
    xoo = x;
    v = zeros(m,1);
    diagG = zeros(m,1);
    Eg2 = zeros(m,1);
    Et2 = zeros(m,1);
    mt = zeros(m,1);
    vt = zeros(m,1);
    strategy = 5;
    cost = [];
    for i = 1:1000
        [fval, grad] = costFunction(x);
        cost = [cost, fval];
        fprintf('%d\t%.5f\n',i-1,fval)
        [xn,v,mt,vt,diagG,Eg2,Et2] = gradupdate(strategy,grad,x,xoo,v,diagG,Eg2,Et2,...
            mt,vt,i,eta,gamma,beta1,beta2,epsilon);

        
        xoo = x;
        x = xn;
    end
%     eta = 1;
%     cost = fval;
%     for iter = 1:10
%        nn_params = nn_params - eta*grad;
%        [fval,grad] = costFunction(nn_params);
%        cost = [cost, fval];
%     end

end
nn_params = x;
plot(cost);



% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% prediction
pred = predict(Theta1, Theta2, X);

fprintf('Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);
loss = cost;
%save('loss_gd.mat', 'loss');


%%
load('loss_gd.mat');
figure('color', [1 1 1]);
plot(0:length(loss)-1,loss);



