clc; clear; close all;
maxIter = 100;
[llrecord, errrecord] = hf_train(maxIter);
%
fig = figure('color', [1 1 1]);
subplot(1,2,1)
plot(0:maxIter, -llrecord(:,1),'rx-');
hold on;
plot(0:maxIter, -llrecord(:,2),'bx-');
subplot(1,2,2)
%figure('color', [1 1 1]);
plot(0:maxIter, errrecord(:,1),'rx-');
hold on;
plot(0:maxIter, errrecord(:,2),'bx-');
set(fig, 'Position', [10, 10, 1000, 400]);
