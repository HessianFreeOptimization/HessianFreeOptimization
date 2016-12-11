function [llrecord, errrecord, paramsp] = hf_train(maxIter, layersizes, layertypes, paramsinit)

% logging
llrecord = zeros(maxIter+1,2);
errrecord = zeros(maxIter+1,2);

%standard L_2 weight-decay:
weight_decay = 2e-5;

% params for damping
autodamp = 1;
drop = 2/3;
boost = 1/drop;

% the amount to decay the previous search direction for the
% purposes of initializing the next run of CG.
decay = 0.95; % Should be 0.95

% network structure
layersizes = [25 30];
layertypes = {'logistic', 'logistic', 'softmax'};
% layersizes = [25];
% layertypes = {'logistic', 'logistic'};
% load datasets.
tmp = load('ex4data1.mat');

indata = tmp.X';
outdata = zeros(10,length(tmp.y));
for i = 1:length(tmp.y)
    outdata(tmp.y(i),i) = 1;
end
perm = randperm(size(indata,2));
intmp = indata( :, perm );
outtmp = outdata(:, perm);

% training data
indata = intmp(:, 1:3000);
outdata = outtmp(:, 1:3000);

% test data
intest = intmp(:, 3001:5000);
outtest = outtmp(:, 3001:5000);

% next try using autodamp = 0 for rho computation.  both for version 6 and
% versions with rho and cg-backtrack computed on the training set
% mattype = 'gn'; %Curvature matrix: Gauss-Newton.

% IMPORTANT NOTES:  The most important variables to tweak are `initlambda' (easy) and
% `maxiters' (harder).  Also, if your particular application is still not working the next 
% most likely way to fix it is tweaking the variable `initcoeff' which controls
% overall magnitude of the initial random weights.  Please don't treat this code like a black-box,
% get a negative result, and then publish a paper about how the approach doesn't work :)  And if
% you are running into difficulties feel free to e-mail me at james.martens@gmail.com

%Fortunately after only 1 'epoch'
%you can often tell if you've made a bad choice.  The value of rho should lie
%somewhere between 0.75 and 0.95.  I could automate this part but I'm lazy
%and my code isn't designed to make such automation a natural thing to add.  Also
%note that 'lambda' is being added to the normalized curvature matrix (i.e.
%divided by the number of cases) while in the ICML paper I was adding it to
%the unnormalized curvature matrix.  This doesn't make any real
%difference to the optimization, but does make it somewhat easier to guage
%lambda and set its initial value since it will be 'independent' of the
%number of training cases in each mini-batch
initlambda = 45.0;

layersizes = [size(indata,1) layersizes size(outdata,1)];
numlayers = size(layersizes,2) - 1;

[indims numcases] = size(indata);
outputString(['==== Input size:' num2str(indims) 'x' num2str(numcases)])

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

% compute the vector-product with the Gauss-Newton matrix with the SMD
% approach;
% G is the Gaussian-Newton approximation to the Hessian matrix
function GV = computeGV(V)
    [VWu, Vbu] = unpack(V);
    %application of R operator
    rdEdy = cell(numlayers+1,1);
    rdEdx = cell(numlayers, 1);
    GVW = cell(numlayers,1);
    GVb = cell(numlayers,1);
    Rx = cell(numlayers,1);
    Ry = cell(numlayers,1);
    yip1 = y{1, 1};
    %forward prop:
    Ryip1 = zeros(layersizes(1), numcases);
    for i = 1:numlayers
        Ryi = Ryip1;
        yi = yip1;
        Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 numcases]);
        yip1 = y{1, i+1};
        if strcmp(layertypes{i}, 'logistic')
            Ryip1 = Rxi.*yip1.*(1-yip1);
        elseif strcmp( layertypes{i}, 'softmax' )
            Ryip1 = Rxi.*yip1 - yip1.* repmat( sum( Rxi.*yip1, 1 ), [layersizes(i+1) 1] );
        end
    end

    %Backwards pass.  This is where things start to differ from computeHV
    %note that the lower-case r notation doesn't really make sense.
    for i = numlayers:-1:1
        if i < numlayers
            if strcmp(layertypes{i}, 'logistic')
                rdEdx{i} = rdEdy{i+1}.*yip1.*(1-yip1);
            end
        else
            %assume canonical link functions:
            rdEdx{i} = -Ryip1;
            if strcmp(layertypes{i}, 'linear')
                rdEdx{i} = 2*rdEdx{i};
            end
        end
        rdEdy{i} = Wu{i}'*rdEdx{i};
        yi = y{1, i};
        GVW{i} = rdEdx{i}*yi';
        GVb{i} = sum(rdEdx{i},2);

        yip1 = yi;
    end
    % psize x 1
    GV = pack(GVW, GVb);
    GV = GV / numcases;
    GV = GV - weight_decay*(V);
    if autodamp
%         fprintf('Auto-damping enabled! \n');
        GV = GV - lambda*V;
    end
end

% forward and get the log-likelihood loss
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

% init for the change (decrement) vector to all variables
ch = zeros(psize, 1);
lambda = initlambda;

% logging
lambdarecord = zeros(maxIter,1);
times = zeros(maxIter,1);
totalpasses = 0;

% initialization of params: random init
if nargin == 3
    paramsp = zeros(psize,1);
    [Wtmp,btmp] = unpack(paramsp);
    numconn = 15;
    for layer = 1:numlayers
        initcoeff = 1;
        for j = 1:layersizes(layer+1)
            idx = ceil(layersizes(layer)*rand(1,numconn));
            Wtmp{layer}(j,idx) = randn(numconn,1)*initcoeff;
        end
    end
    paramsp = pack(Wtmp, btmp);
else
    paramsp = paramsinit;
end

outputString('================ Start HF Training... ================')
% Main part: train and test.

[ll, err] = computeLL(paramsp, indata, outdata);
llrecord(1,1) = ll;
errrecord(1,1) = err;
outputString( ['-- Init Train Log likelihood: ' num2str(ll) ', error rate: ' num2str(err)] );

[ll_test, err_test] = computeLL(paramsp, intest, outtest);
llrecord(1,2) = ll_test;
errrecord(1,2) = err_test;
outputString( ['-- Init Test Log likelihood: ' num2str(ll_test) ', error rate: ' num2str(err_test)] );
outputString( '' );

for epoch = 1:maxIter
    tic
    outputString(['-- Epoch: ' num2str(epoch)])
    [Wu, bu] = unpack(paramsp);
    y = cell(1, numlayers+1);
    ll = 0;

    % ===> forward prop: compute activations and grads, as well as loss:
    y{1, 1} = indata(:, 1:numcases );
    yip1 =  y{1, 1} ;
    dEdW = cell(numlayers, 1);
    dEdb = cell(numlayers, 1);
    dEdW2 = cell(numlayers, 1);
    dEdb2 = cell(numlayers, 1);

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
    outc = outdata(:, 1:numcases );
    if strcmp( layertypes{numlayers}, 'softmax' )
        ll = ll + sum(sum(outc.*log(yip1)));
    elseif strcmp( layertypes{numlayers}, 'logistic' )
        ll = ll + sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0)))));                
    end
    
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
        %gradient squared comp:
        dEdW2{i} = (dEdxi.^2)*(yi.^2)';
        dEdb2{i} = sum(dEdxi.^2,2);
        dEdyip1 = dEdyi;
        yip1 = yi;
    end
    % psize x 1
    grad = pack(dEdW, dEdb);
    grad2 = pack(dEdW2, dEdb2);
    grad = grad / numcases;
    grad = grad - weight_decay*(paramsp);
    grad2 = grad2 / numcases;

    ll = ll / numcases;
    ll = ll - 0.5*weight_decay*paramsp'*(paramsp);
    oldll = ll;

    %slightly decay the previous change vector before using it as an
    %initialization.  This is something I didn't mention in the paper.
    ch = decay*ch;

    %maxiters is the most important variable that you should try
    %tweaking.  While the ICML paper had maxiters=250 for everything
    %I've since found out that this wasn't optimal.
    maxiters = 250;
    miniters = 1;

    %TODO: preconditioning vector.  Feel free to experiment with this.
%     precon = (grad2 + ones(psize,1)*lambda + weight_decay).^(3/4);
    precon = ones(psize,1);
    
    % ==> conjugate grad descent and back-tracking
    [chs, iterses] = conjgrad_1( @(V)-computeGV(V), grad, ch, ceil(maxiters), ceil(miniters), precon );

    ch = chs{end};
    iters = iterses(end);
    totalpasses = totalpasses + iters;
    p = ch;
    outputString( ['CG steps used: ' num2str(iters) ', total is: ' num2str(totalpasses) ', ch magnitude : ' num2str(norm(ch))] );

    j = length(chs);
    %"CG-backtracking":
    %full training set version:
    [ll, err] = computeLL(paramsp + p, indata, outdata);
    for j = (length(chs)-1):-1:1
        [lowll, lowerr] = computeLL(paramsp + chs{j}, indata, outdata);
        if ll > lowll
            j = j+1;
            break;
        end
        ll = lowll;
        err = lowerr;
    end
    if isempty(j)
        j = 1;
    end
    p = chs{j};

    [ll_chunk, err_chunk] = computeLL(paramsp + chs{j}, indata, outdata);
    [oldll_chunk, olderr_chunk] = computeLL(paramsp, indata, outdata);

    %disabling damping when computing rho is something I'm not 100% sure
    autodamp = 0;
    denom = -0.5*chs{j}'*computeGV(chs{j}) - grad'*chs{j};
    autodamp = 1;
    rho = (oldll_chunk - ll_chunk)/denom;
    if oldll_chunk - ll_chunk > 0
        rho = -Inf;
    end
    outputString( ['Chose iters : ' num2str(iterses(j)) ' ,rho = ' num2str(rho)] );

    % back-tracking line-search
    step = 1.0;
    c = 10^(-2);
    j = 0;
    while j < 60
        if ll >= oldll + c*step*grad'*p
            break;
        else
            step = 0.8*step;
            j = j + 1;
        end
        [ll, err] = computeLL(paramsp + step*p, indata, outdata);
    end
    % reject
    if j == 60
        j = Inf;
        step = 0.0;
        ll = oldll;
    end

    % damping heuristic
    if rho < 0.25 || isnan(rho)
        lambda = lambda*boost;
    elseif rho > 0.75
        lambda = lambda*drop;
    end

    outputString( ['#backtracking: ' num2str(j) ', step size: ' num2str(step) ', New lambda: ' num2str(lambda)] );

    %Parameter update:
    paramsp = paramsp + step*p;
    lambdarecord(epoch,1) = lambda;
    llrecord(epoch+1,1) = ll;
    errrecord(epoch+1,1) = err;
    outputString( ['---- Train Log likelihood: ' num2str(ll) ', error rate: ' num2str(err)] );

    [ll_test, err_test] = computeLL(paramsp, intest, outtest);
    llrecord(epoch+1,2) = ll_test;
    errrecord(epoch+1,2) = err_test;
    outputString( ['---- Test Log likelihood: ' num2str(ll_test) ', error rate: ' num2str(err_test)] );
    outputString( '' );

    times(epoch) = toc;
end

outputString( ['Total time: ' num2str(sum(times)) ] );
end

function outputString( s )
    fprintf( '%s\n', s );
end

function v = vec(A)
    v = A(:);
end