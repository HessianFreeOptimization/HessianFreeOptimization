clc; clear; close all;
maxepoch = 30;
[llrecord, errrecord] = hf_train(maxepoch);

figure('color', [1 1 1]);
plot(-llrecord(:,1),'rx-');
hold on;
plot(-llrecord(:,2),'bx-');



