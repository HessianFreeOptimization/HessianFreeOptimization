clc; clear; close all;
maxIter = 50;
algorithm = 'hf';
algorithm = 'lbfgs';
[llrecord, errrecord] = hf_train(algorithm, maxIter);

figure('color', [1 1 1]);
plot(-llrecord(:,1),'rx-');
hold on;
plot(-llrecord(:,2),'bx-');
%%
figure('color', [1 1 1]);
plot(errrecord(:,1),'rx-');
hold on;
plot(errrecord(:,2),'bx-');

