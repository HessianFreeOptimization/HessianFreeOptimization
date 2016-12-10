clc; clear; close all;


%you will NEVER need more than a few hundred epochs unless you are doing
%something very wrong.  Here 'epoch' means parameter update, not 'pass over
%the training set'.
maxepoch = 500;


tmp = load('digs3pts_1.mat');
indata = tmp.bdata';
intest = tmp.bdatatest';
clear tmp

perm = randperm(size(indata,2));
indata = indata( :, perm );

%it's an auto-encoder so output is input
outdata = indata;
outtest = intest;

runDesc = '';

%next try using autodamp = 0 for rho computation.  both for version 6 and
%versions with rho and cg-backtrack computed on the training set


numchunks = 4
numchunks_test = 4;

%mattype = 'gn'; %Curvature matrix: Gauss-Newton.

%decay = 1.0;
decay = 0.95;

errtype = 'L2'; %report the L2-norm error (in addition to the quantity actually being optimized, i.e. the log-likelihood)

%standard L_2 weight-decay:
weightcost = 2e-5


nnet_train_2(maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, errtype, weightcost, decay);



