clc; clear; close all;
maxIter = 100;
algorithm = 'hf';
%algorithm = 'lbfgs';
[llrecord, errrecord] = hf_train(algorithm, maxIter);

figure('color', [1 1 1]);
plot(-llrecord(:,1),'rx-');
hold on;
plot(-llrecord(:,2),'bx-');



