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
options.MaxIter = 300;
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

                               
%[nn_params, cost] = fminlbfgs( costFunction, initial_nn_params, options);

x = initial_nn_params;
m = 7;
l = size(x,1);

cost = [];

s = [];
y = [];
[fval, grad] = costFunction(x);

for k = 1:100
    cost = [cost, fval];
    fprintf('%d\t%.5f\n',k-1,fval)
    %[xn,v,mt,vt,diagG,Eg2,Et2] = gradupdate(strategy,grad,x,xoo,v,diagG,Eg2,Et2,...
    %    mt,vt,i,eta,gamma,beta1,beta2,epsilon);
    
    % s: l x m, Delta_position
    % y: l x m, Delta_grad
	% pk = L_BFGS(g,s,y,m,Hdiag);
    
    q = -grad;
    p = q;
    if k ~= 1
        alpha = zeros(1,k);
        for i = size(s,2):-1:1
%             if s(:,i)'*q == 0
%                 alpha(i) = 0;
%             else
                alpha(i) = s(:,i)'*q / (y(:,i)'*s(:,i));
%             end
            q = q - alpha(i)*y(:,i);
        end
%         if y(:,k-1)'*y(:,k-1) == 0
%         	H0 = zeros(l);
%         else
            H0 = y(:,end)'*s(:,end) / (y(:,end)'*y(:,end)) * eye(l);
%         end
        p = H0*q;
        %max(q)
        %max(p)
        for i = 1:size(s,2)
%             if y(:,i)'*p == 0
%                 beta = 0;
%             else
                beta = y(:,i)'*p / (y(:,i)'*s(:,i));
%             end
            p = p + (alpha(i) - beta)*s(:,i);
        end
        %max(p)
    end

    t = 1;
    while costFunction(x+t*p) > costFunction(x) + 0.0001*t*grad'*p
       t = 0.9*t; 
    end
    
    s = [s, t*p];
    x = x + t*p;
    grado = grad;
    [fval, grad] = costFunction(x);
    y = [y, grad - grado];
    
    if size(s,2) > m
        s = s(:,2:end);
        y = y(:,2:end);
    end
end
nn_params = x;
plot(cost);








% nn_params = initial_nn_params;
% [fval, grad] = costFunction(nn_params);
% eta = 2;
% cost = fval;
% fprintf('%d\t%.5f\n',0,fval)
% for iter = 1:100
%    nn_params = nn_params - eta*grad;
%    [fval,grad] = costFunction(nn_params);
%    fprintf('%d\t%.5f\n',iter,fval)
%    cost = [cost, fval];
% end
% 
% 
% 
% plot(cost);
figure('color', [1 1 1]);
plot(0:length(cost)-1,cost);


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



