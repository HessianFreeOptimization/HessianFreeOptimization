function [llrecord, errrecord, paramsp, eval_fs, eval_gs, step_size] = fixstep_momentum_lbfgs(algorithm, eta, maxIter, params, paramsinit)
% variables
llrecord = zeros(maxIter+1,2);
errrecord = zeros(maxIter+1,2);
eval_fs = zeros(maxIter+1,1);
eval_gs = zeros(maxIter+1,1);
step_size = zeros(maxIter,1);
global eval_f;
global eval_g;

%standard L_2 weight-decay and params:
weight_decay = params.weight_decay;
layersizes = params.layersizes;
layertypes = params.layertypes;
indata = params.indata;
outdata = params.outdata;
intest = params.intest;
outtest = params.outtest;


layersizes = [size(indata,1) layersizes size(outdata,1)];
numlayers = size(layersizes,2) - 1;

[indims, numcases] = size(indata);
outputString(['input size:' num2str(indims) 'x' num2str(numcases)])

y = cell(1, numlayers+1);

psize = layersizes(1,2:(numlayers+1))*layersizes(1,1:numlayers)' + sum(layersizes(2:(numlayers+1)));

% pack all the parameters into a single vector
function M = pack(W,b)
    M = zeros( psize, 1 );
    cur = 0;
    for i = 1:numlayers
        M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1) = vec( W{i} );
        cur = cur + layersizes(i)*layersizes(i+1);
        M((cur+1):(cur + layersizes(i+1)), 1) = vec( b{i} );
        cur = cur + layersizes(i+1);
    end
end

% unpack parameters from a vector
function [W,b] = unpack(M)
    W = cell( numlayers, 1 );
    b = cell( numlayers, 1 );
    cur = 0;
    for i = 1:numlayers
        W{i} = reshape( M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1), [layersizes(i+1) layersizes(i)] );
        cur = cur + layersizes(i)*layersizes(i+1);
        b{i} = reshape( M((cur+1):(cur + layersizes(i+1)), 1), [layersizes(i+1) 1] );
        cur = cur + layersizes(i+1);
    end
end

function [ll, err] = computeLL(params, in, out)
    [W,b] = unpack(params);
    schunk = size(in,2);
    
    yi = in(:, 1:schunk );
    outc = out(:, 1:schunk );
    for i = 1:numlayers
        xi = W{i}*yi + repmat(b{i}, [1 schunk]);
        if strcmp(layertypes{i}, 'logistic')
            yi = 1./(1 + exp(-xi));
        elseif strcmp(layertypes{i}, 'softmax' )
            tmp = exp(xi);
            yi = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );   
        end
    end
    eval_f = eval_f + 1;
        
    ll = 0;
    if strcmp( layertypes{numlayers}, 'softmax' )
        ll = sum(sum(outc.*log(yi)));
    elseif strcmp( layertypes{numlayers}, 'logistic' )
        ll = sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0)))));
    end
    
    err = sum( sum(outc.*yi,1) ~= max(yi,[],1) ) / size(in,2);

    ll = ll / size(in,2);
    ll = ll - 0.5*weight_decay*params'*params;
end

times = zeros(maxIter,1);

if nargin == 4
    % initialization of params.
    paramsp = zeros(psize,1);
    [Wtmp,btmp] = unpack(paramsp);
    numconn = 15;
    for i = 1:numlayers
        initcoeff = 1;
        for j = 1:layersizes(i+1)
            idx = ceil(layersizes(i)*rand(1,numconn));
            Wtmp{i}(j,idx) = randn(numconn,1)*initcoeff;
        end
    end
    paramsp = pack(Wtmp, btmp);
else
    paramsp = paramsinit;
end

function grad = calcu_grad(paramsp)
    [Wu, bu] = unpack(paramsp);
    y = cell(1, numlayers+1);
    %ll = 0;
    %forward prop:
    y{1, 1} = indata(:, 1:numcases );
    yip1 =  y{1, 1} ;
    dEdW = cell(numlayers, 1);
    dEdb = cell(numlayers, 1);

    for i = 1:numlayers
        yi = yip1;
        xi = Wu{i}*yi + repmat(bu{i}, [1 numcases]);
        if strcmp(layertypes{i}, 'logistic')
            yip1 = 1./(1 + exp(-xi));
        elseif strcmp( layertypes{i}, 'softmax' )
            tmp = exp(xi);
            yip1 = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );
        end
        y{1, i+1} = yip1;
    end
    eval_f = eval_f + 1;
    outc = outdata(:, 1:numcases );
    
    for i = numlayers:-1:1
        if i < numlayers
            if strcmp(layertypes{i}, 'logistic')
                dEdxi = dEdyip1.*yip1.*(1-yip1);
            end
        else
            dEdxi = outc - yip1; %simplified due to canonical link
        end
        dEdyi = Wu{i}'*dEdxi;

        yi = y{1, i};

        %standard gradient comp:
        dEdW{i} = dEdxi*yi';
        dEdb{i} = sum(dEdxi,2);

        dEdyip1 = dEdyi;
        yip1 = yi;
    end
    eval_g = eval_g + 1;

    % psize x 1
    grad = pack(dEdW, dEdb);
    grad = grad / numcases;
    grad = grad - weight_decay*(paramsp);
end

outputString(sprintf('================ Start Training for %d iters... ================', maxIter))
% Main part: train and test.
gamma = 0.9;

m = size(paramsp,1);
v = zeros(m,1);

[ll, err] = computeLL(paramsp, indata, outdata);
llrecord(1,1) = ll;
errrecord(1,1) = err;
eval_gs(1,1) = eval_g;
eval_fs(1,1) = eval_f;
step_size(1,1) = 0;

outputString( ['Train Log likelihood: ' num2str(ll) ', error rate: ' num2str(err)] );

[ll_test, err_test] = computeLL(paramsp, intest, outtest);
llrecord(1,2) = ll_test;
errrecord(1,2) = err_test;
outputString( ['Test Log likelihood: ' num2str(ll_test) ', error rate: ' num2str(err_test)] );
outputString( '' );

grad = calcu_grad(paramsp);

lbfgs_m = 7;
bfgs_s = [];
bfgs_y = [];
for epoch = 1:maxIter
    tic
    bfgs_q = -grad;
    bfgs_p = bfgs_q;
    if epoch ~= 1
        alpha = zeros(1,lbfgs_m);
        for i = size(bfgs_s,2):-1:1
            alpha(i) = bfgs_s(:,i)'*bfgs_q / (bfgs_y(:,i)'*bfgs_s(:,i));
            bfgs_q = bfgs_q - alpha(i)*bfgs_y(:,i);
        end
        H0 = bfgs_y(:,end)'*bfgs_s(:,end) / (bfgs_y(:,end)'*bfgs_y(:,end));
        bfgs_p = H0*bfgs_q;
        for i = 1:size(bfgs_s,2)
            beta = bfgs_y(:,i)'*bfgs_p / (bfgs_y(:,i)'*bfgs_s(:,i));
            bfgs_p = bfgs_p + (alpha(i) - beta)*bfgs_s(:,i);
        end
    end
    
    %if isMomentum % momentum lbfgs
    if strcmp(algorithm, 'gradient descent')
        % backtracking...
        outputString(['GD w/ backtracking, trial: ', num2str(params.trial), ', epoch: ',num2str(epoch)]);
        v = -eta*grad;
        paramsp = paramsp - v;
    elseif strcmp(algorithm, 'fixstep-lbfgs') % fixstep lbfgs
        outputString(['Fixstep L-BFGS, trial: ', num2str(params.trial), ', epoch: ',num2str(epoch)]);
        % +/- bug?
        v = eta*bfgs_p;
        paramsp = paramsp - v;
    elseif strcmp(algorithm, 'momentum-lbfgs')
        outputString(['Momentum L-BFGS, trial: ', num2str(params.trial), ', epoch: ',num2str(epoch)]);
        v = gamma*v + eta*bfgs_p;
        paramsp = paramsp - v;
    end

    [ll, err] = computeLL(paramsp, indata, outdata);
    
    bfgs_s = [bfgs_s, v];
    
    gradold = grad;
    grad = calcu_grad(paramsp);
    outputString( ['No backtracking, step size: ' num2str(eta)] );

    bfgs_y = [bfgs_y, grad - gradold];

    if size(bfgs_s,2) > lbfgs_m
        bfgs_s = bfgs_s(:,2:end);
        bfgs_y = bfgs_y(:,2:end);
    end

    %Parameter update:
    llrecord(epoch+1,1) = ll;
    errrecord(epoch+1,1) = err;
    eval_gs(epoch+1,1) = eval_g;
    eval_fs(epoch+1,1) = eval_f;
    step_size(epoch+1,1) = norm(v);
    outputString( ['Train Log likelihood: ' num2str(ll) ', error rate: ' num2str(err)] );

    [ll_test, err_test] = computeLL(paramsp, intest, outtest);

    llrecord(epoch+1,2) = ll_test;
    errrecord(epoch+1,2) = err_test;
    outputString( ['Test Log likelihood: ' num2str(ll_test) ', error rate: ' num2str(err_test)] );
    outputString( '' );

    times(epoch) = toc;
end

outputString( ['Total time: ' num2str(sum(times)) ] );
end

function outputString(s)
    fprintf( '%s\n', s );
end

function v = vec(A)
    v = A(:);
end
%EOF.
