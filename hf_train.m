function hf_train()
%you will NEVER need more than a few hundred epochs unless you are doing
%something very wrong.  Here 'epoch' means parameter update, not 'pass over
%the training set'.

%next try using autodamp = 0 for rho computation.  both for version 6 and
%versions with rho and cg-backtrack computed on the training set

%mattype = 'gn'; %Curvature matrix: Gauss-Newton.
tmp = load('digs3pts_1.mat');
indata = tmp.bdata';
intest = tmp.bdatatest';
clear tmp

perm = randperm(size(indata,2));
indata = indata( :, perm );

%it's an auto-encoder so output is input
outdata = indata;
outtest = intest;



tmp = load('digs3pts_1.mat');
indata = tmp.bdata';
intest = tmp.bdatatest';
clear tmp

perm = randperm(size(indata,2));
indata = indata( :, perm );

%it's an auto-encoder so output is input
outdata = indata;
outtest = intest;



decay = 0.95;

errtype = 'L2'; %report the L2-norm error (in addition to the quantity actually being optimized, i.e. the log-likelihood)

maxepoch = 10;

paramsp = [];

layersizes = [40 6 40];
%Note that the code layer uses linear units
layertypes = {'logistic', 'linear', 'logistic', 'logistic'};

numchunks = 4;
numchunks_test = 4;

%standard L_2 weight-decay:
weightcost = 2e-5;

% IMPORTANT NOTES:  The most important variables to tweak are `initlambda' (easy) and
% `maxiters' (harder).  Also, if your particular application is still not working the next 
% most likely way to fix it is tweaking the variable `initcoeff' which controls
% overall magnitude of the initial random weights.  Please don't treat this code like a black-box,
% get a negative result, and then publish a paper about how the approach doesn't work :)  And if
% you are running into difficulties feel free to e-mail me at james.martens@gmail.com
%
% paramsp - initial parameters in the form of a vector (can be []).  If
% this, or the arguments Win,bin are empty, the 'sparse initialization'
% technique is used
%
% Win, bin - initial parameters in their matrix forms (can be [])
%
% maxepoch - maximum number of 'epochs' (outer iteration of HF).  There is no termination condition
% for the optimizer and usually I just stop it by issuing a break command
%
% indata/outdata - input/output training data for the net (each case is a
% column).  Make sure that the training cases are randomly permuted when you invoke
% this function as it won't do it for you.
%
% numchunks - number of mini-batches used to partition the training set.
% During each epoch, a single mini-batch is used to compute the
% matrix-vector products, after which it gets cycled to the back of the
% last and is used again numchunk epochs later. Note that the gradient and 
% line-search are still computed using the full training set.  This of
% course is not practical for very large datasets, but in general you can
% afford to use a lot more data to compute the gradient than the
% matrix-vector products, since you only have to do the former once per iteration
% of the outer loop.
%
% intest/outtest -  test data
%
% numchunks_test - while the test set isn't used for matrix-vector
% products, you still may want to partition it so that it can be processed
% in pieces on the GPU instead of all at once.
%
%
% rms - by default we use the canonical error function for
% each output unit type.  e.g. square error for linear units and
% cross-entropy error for logistics.  Setting this to 1 (instead of 0) overrides 
% the default and forces it to use squared-error.  Note that even if what you
% care about is minimizing squared error it's sometimes still better
% to run on the optimizer with the canonical error
%
% errtype - in addition to displaying the objective function (log-likelihood) you may also
% want to keep track of another metric like squared error when you train
% deep auto-encoders.  This can be 'L2' for squared error, 'class' for
% classification error, or 'none' for nothing.  It should be easy enough to
% add your own type of error should you need one
%
% weightcost - the strength of the l_2 prior on the weights
%
% decay - the amount to decay the previous search direction for the
% purposes of initializing the next run of CG.  Should be 0.95









%rec_constants = {'layersizes', 'weightcost', 'autodamp', 'initlambda', 'drop', 'boost', 'numchunks', 'errtype', 'decay'};

autodamp = 1;

drop = 2/3;

boost = 1/drop;

%In addition to maxiters the variable below is something you should manually
%adjust.  It is quite problem specific.  Fortunately after only 1 'epoch'
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



% computeBV = computeGV
storeD = 0;





%use singles (this can make cpu code go faster):

mones = @(varargin)ones(varargin{:}, 'single');
mzeros = @(varargin)zeros(varargin{:}, 'single');
%conv = @(x)x;
conv = @single;


%use doubles:
%{
mones = @ones;
mzeros = @zeros;
%conv = @(x)x;
conv = @double;
%}


%if hybridmode
store = conv; %cache activities on the gpu



layersizes = [size(indata,1) layersizes size(outdata,1)];
numlayers = size(layersizes,2) - 1;

[indims numcases] = size(indata);
[tmp numtest] = size(intest);

if mod( numcases, numchunks ) ~= 0
    error( 'Number of chunks doesn''t divide number of training cases!' );
end

sizechunk = numcases/numchunks;
sizechunk_test = numtest/numchunks_test;


if numcases >= 512*64
    disp( 'jacket issues possible!' );
end


y = cell(numchunks, numlayers+1);
if storeD
    dEdy = cell(numchunks, numlayers+1);
    dEdx = cell(numchunks, numlayers);
end



function v = vec(A)
    v = A(:);
end


psize = layersizes(1,2:(numlayers+1))*layersizes(1,1:numlayers)' + sum(layersizes(2:(numlayers+1)));

%pack all the parameters into a single vector for easy manipulation
function M = pack(W,b)
    
    M = mzeros( psize, 1 );
    
    cur = 0;
    for i = 1:numlayers
        M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1) = vec( W{i} );
        cur = cur + layersizes(i)*layersizes(i+1);
        
        M((cur+1):(cur + layersizes(i+1)), 1) = vec( b{i} );
        cur = cur + layersizes(i+1);
    end
    
end

%unpack parameters from a vector so they can be used in various neural-net
%computations
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


%compute the vector-product with the Gauss-Newton matrix
function GV = computeGV(V)

    [VWu, Vbu] = unpack(V);
    
    GV = mzeros(psize,1);
    
    %if hybridmode
    chunkrange = targetchunk; %set outside

    for chunk = chunkrange
        
        %application of R operator
        rdEdy = cell(numlayers+1,1);
        rdEdx = cell(numlayers, 1);

        GVW = cell(numlayers,1);
        GVb = cell(numlayers,1);
        
        Rx = cell(numlayers,1);
        Ry = cell(numlayers,1);

        yip1 = conv(y{chunk, 1});

        %forward prop:
        Ryip1 = mzeros(layersizes(1), sizechunk);
            
        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 sizechunk]);
            %Rx{i} = store(Rxi);

            yip1 = conv(y{chunk, i+1});

            if strcmp(layertypes{i}, 'logistic')
                Ryip1 = Rxi.*yip1.*(1-yip1);
            elseif strcmp(layertypes{i}, 'linear')
                Ryip1 = Rxi;
            else
                error( 'Unknown/unsupported layer type' );
            end
            
            Rxi = [];

        end
        
        %Backwards pass.  This is where things start to differ from computeHV  Please note that the lower-case r 
        %notation doesn't really make sense so don't bother trying to decode it.  Instead there is a much better
        %way of thinkin about the GV computation, with its own notation, which I talk about in my more recent paper: 
        %"Learning Recurrent Neural Networks with Hessian-Free Optimization"
        for i = numlayers:-1:1

            if i < numlayers
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    rdEdx{i} = rdEdy{i+1}.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    rdEdx{i} = rdEdy{i+1};
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                %assume canonical link functions:
                rdEdx{i} = -Ryip1;

                if strcmp(layertypes{i}, 'linear')
                    rdEdx{i} = 2*rdEdx{i};
                end
                Ryip1 = [];

            end
            rdEdy{i+1} = [];
            
            rdEdy{i} = Wu{i}'*rdEdx{i};

            yi = conv(y{chunk, i});

            GVW{i} = rdEdx{i}*yi';
            GVb{i} = sum(rdEdx{i},2);

            rdEdx{i} = [];

            yip1 = yi;
            yi = [];
        end
        yip1 = [];
        rdEdy{1} = [];

        GV = GV + pack(GVW, GVb);
        
    end
    
    GV = GV / conv(numcases);
    
    %if hybridmode
    GV = GV * conv(numchunks);
    
    
    GV = GV - conv(weightcost)*(maskp.*V);

    if autodamp
        GV = GV - conv(lambda)*V;
    end
    
end
    

function [ll, err] = computeLL(params, in, out, nchunks, tchunk)

    ll = 0;
    
    err = 0;
    
    [W,b] = unpack(params);
    
    if mod( size(in,2), nchunks ) ~= 0
        error( 'Number of chunks doesn''t divide number of cases!' );
    end    
    
    schunk = size(in,2)/nchunks;
    
    if nargin > 4
        chunkrange = tchunk;
    else
        chunkrange = 1:nchunks;
    end
    
    for chunk = chunkrange
    
        yi = conv(in(:, ((chunk-1)*schunk+1):(chunk*schunk) ));
        outc = conv(out(:, ((chunk-1)*schunk+1):(chunk*schunk) ));

        for i = 1:numlayers
            xi = W{i}*yi + repmat(b{i}, [1 schunk]);

            if strcmp(layertypes{i}, 'logistic')
                yi = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'linear')
                yi = xi;
            end

        end

        if strcmp( layertypes{numlayers}, 'linear' )
            ll = ll + double( -sum(sum((outc - yi).^2)) );
        elseif strcmp( layertypes{numlayers}, 'logistic' )
                %this version is more stable:
                ll = ll + double(sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0))))));
        end
        xi = [];

        if strcmp( errtype, 'class' )
            %err = 1 - double(sum( sum(outc.*yi,1) == max(yi,[],1) ) )/size(in,2);
            err = err + double(sum( sum(outc.*yi,1) ~= max(yi,[],1) ) ) / size(in,2);
        elseif strcmp( errtype, 'L2' )
            err = err + double(sum(sum((yi - outc).^2, 1))) / size(in,2);
        elseif strcmp( errtype, 'none')
            %do nothing
        else
            error( 'Unrecognized error type' );
        end        
        outc = [];
        yi = [];
    end

    ll = ll / size(in,2);
    
    if nargin > 4
        ll = ll*nchunks;
        err = err*nchunks;
    end
    
    ll = ll - 0.5*weightcost*double(params'*(maskp.*params));

end


function yi = computePred(params, in) %for checking G computation using finite differences
    
    [W, b] = unpack(params);
    
    yi = in;
        
    for i = 1:numlayers
        xi = W{i}*yi + repmat(b{i}, [1 size(in,2)]);
        
        if i < numlayers
            if strcmp(layertypes{i}, 'logistic')
                yi = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'linear')
                yi = xi;
            end
        else
            yi = xi;
        end
        
    end
end



maskp = mones(psize,1);
[maskW, maskb] = unpack(maskp);
disp('not masking out the weight-decay for biases');
for i = 1:length(maskb)
    %maskb{i}(:) = 0; %uncomment this line apply the l_2 only to the connection weights and not the biases
end
maskp = pack(maskW,maskb);


indata = single(indata);
outdata = single(outdata);
intest = single(intest);
outtest = single(outtest);


function outputString( s )
    fprintf( 1, '%s\n', s );
end

ch = mzeros(psize, 1);


    
lambda = initlambda;

llrecord = zeros(maxepoch,2);
errrecord = zeros(maxepoch,2);
lambdarecord = zeros(maxepoch,1);
times = zeros(maxepoch,1);

totalpasses = 0;
epoch = 1;
    


%if isempty(paramsp)
%SPARSE INIT:
paramsp = zeros(psize,1); %not mzeros

[Wtmp,btmp] = unpack(paramsp);

numconn = 15;

for i = 1:numlayers

    initcoeff = 1;
    %incoming
    for j = 1:layersizes(i+1)
        idx = ceil(layersizes(i)*rand(1,numconn));
        Wtmp{i}(j,idx) = randn(numconn,1)*initcoeff;
    end

end

paramsp = pack(Wtmp, btmp);

clear Wtmp btmp



% outputString( 'Initial constant values:' );
% for i = 1:length(rec_constants)
%     outputString( [rec_constants{i} ': ' num2str(eval( rec_constants{i} )) ] );
% end

outputString( '=================================================' );

for epoch = epoch:maxepoch
    tic

    targetchunk = mod(epoch-1, numchunks)+1;
    
    [Wu, bu] = unpack(paramsp);


    y = cell(numchunks, numlayers+1);
    x = cell(numchunks, numlayers+1);
    
    if storeD
        dEdy = cell(numchunks, numlayers+1);
        dEdx = cell(numchunks, numlayers);
    end

    grad = mzeros(psize,1);
    grad2 = mzeros(psize,1);
    
    ll = 0;

    %forward prop:
    %index transition takes place at nonlinearity
    for chunk = 1:numchunks
        
        y{chunk, 1} = store(indata(:, ((chunk-1)*sizechunk+1):(chunk*sizechunk) ));
        yip1 = conv( y{chunk, 1} );

        dEdW = cell(numlayers, 1);
        dEdb = cell(numlayers, 1);

        dEdW2 = cell(numlayers, 1);
        dEdb2 = cell(numlayers, 1);

        for i = 1:numlayers

            yi = yip1;
            yip1 = [];
            xi = Wu{i}*yi + repmat(bu{i}, [1 sizechunk]);
            yi = [];

            if strcmp(layertypes{i}, 'logistic')
                yip1 = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'linear')
                yip1 = xi;
            else
                error( 'Unknown/unsupported layer type' );
            end
            
            y{chunk, i+1} = store(yip1);
        end

        %back prop:
        %cross-entropy for logistics:
        %dEdy{numlayers+1} = outdata./y{numlayers+1} - (1-outdata)./(1-y{numlayers+1});
        %cross-entropy for softmax:
        %dEdy{numlayers+1} = outdata./y{numlayers+1};

        if chunk ~= targetchunk
            y{chunk, numlayers+1} = []; %save memory
        end

        outc = conv(outdata(:, ((chunk-1)*sizechunk+1):(chunk*sizechunk) ));
        
        if strcmp( layertypes{numlayers}, 'linear' )
            ll = ll + double( -sum(sum((outc - yip1).^2)) );
        elseif strcmp( layertypes{numlayers}, 'logistic' )
            %more stable:
            ll = ll + sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0)))));                
        end
        xi = [];
        
        
        for i = numlayers:-1:1

            if i < numlayers
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    dEdxi = dEdyip1.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    dEdxi = dEdyip1;
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                dEdxi = outc - yip1; %simplified due to canonical link

                if strcmp(layertypes{i}, 'linear')
                    dEdxi = 2*dEdxi;  %the convention is to use the doubled version of the squared-error objective
                end

                outc = [];
            end
            dEdyi = Wu{i}'*dEdxi;

            if storeD && (chunk == targetchunk)
                dEdx{chunk, i} = store(dEdxi);
                dEdy{chunk, i} = store(dEdyi);
            end

            yi = conv(y{chunk, i});

            if chunk ~= targetchunk
                y{chunk, i} = []; %save memory
            end

            %standard gradient comp:
            dEdW{i} = dEdxi*yi';
            dEdb{i} = sum(dEdxi,2);

            %gradient squared comp:
            dEdW2{i} = (dEdxi.^2)*(yi.^2)';
            dEdb2{i} = sum(dEdxi.^2,2);

            dEdxi = [];

            dEdyip1 = dEdyi;
            dEdyi = [];

            yip1 = yi;
            yi = [];
        end
        yip1 = [];  dEdyip1 = [];

        if chunk == targetchunk
            gradchunk = pack(dEdW, dEdb);
            grad2chunk = pack(dEdW2, dEdb2);
        end

        grad = grad + pack(dEdW, dEdb);

        grad2 = grad2 + pack(dEdW2, dEdb2);

        %for checking F:
        %gradouter = gradouter + pack(dEdW, dEdb)*pack(dEdW, dEdb)';

        dEdW = []; dEdb = []; dEdW2 = []; dEdb2 = [];
    end
    
    grad = grad / conv(numcases);
    grad = grad - conv(weightcost)*(maskp.*paramsp);
    
    grad2 = grad2 / conv(numcases);
    
    gradchunk = gradchunk/conv(sizechunk) - conv(weightcost)*(maskp.*paramsp);
    grad2chunk = grad2chunk/conv(sizechunk);
    
    ll = ll / numcases;
    
    ll = ll - 0.5*weightcost*double(paramsp'*(maskp.*paramsp));
    
    
    oldll = ll;
    ll = [];

  
    %slightly decay the previous change vector before using it as an
    %initialization.  This is something I didn't mention in the paper,
    %and it's not overly important but it can help a lot in some situations 
    %so you should probably use it
    ch = conv(decay)*ch;

    %maxiters is the most important variable that you should try
    %tweaking.  While the ICML paper had maxiters=250 for everything
    %I've since found out that this wasn't optimal.  For example, with
    %pre-trained weights for CURVES, maxiters=150 is better.  And for
    %the FACES dataset you should use something like maxiters=100.
    %Setting it too small or large can be bad to various degrees.
    %Currently I'm trying to automate"this choice, but it's quite hard
    %to come up with a *robust* heuristic for doing this.

    maxiters = 250;
    miniters = 1;
    outputString(['maxiters = ' num2str(maxiters) '; miniters = ' num2str(miniters)]);

    %preconditioning vector.  Feel free to experiment with this.  For
    %some problems (like the RNNs) this style of diaognal precondition
    %doesn't seem to be beneficial.  Probably because the parameters don't
    %exibit any obvious "axis-aligned" scaling issues like they do with
    %standard deep neural nets
    precon = (grad2 + mones(psize,1)*conv(lambda) + maskp*conv(weightcost)).^(3/4);
    %precon = mones(psize,1);

    [chs, iterses] = conjgrad_1( @(V)-computeGV(V), grad, ch, ceil(maxiters), ceil(miniters), precon );

    ch = chs{end};
    iters = iterses(end);

    totalpasses = totalpasses + iters;
    outputString(['CG steps used: ' num2str(iters) ', total is: ' num2str(totalpasses) ]);

    p = ch;
    outputString( ['ch magnitude : ' num2str(double(norm(ch)))] );

    j = length(chs);
    

    %"CG-backtracking":
    %full training set version:
    [ll, err] = computeLL(paramsp + p, indata, outdata, numchunks);
    for j = (length(chs)-1):-1:1
        [lowll, lowerr] = computeLL(paramsp + chs{j}, indata, outdata, numchunks);

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
    outputString( ['Chose iters : ' num2str(iterses(j))] );


    [ll_chunk, err_chunk] = computeLL(paramsp + chs{j}, indata, outdata, numchunks, targetchunk);
    [oldll_chunk, olderr_chunk] = computeLL(paramsp, indata, outdata, numchunks, targetchunk);

    %disabling the damping when computing rho is something I'm not 100% sure
    %about.  It probably doesn't make a huge difference either way.  Also this
    %computation could probably be done on a different subset of the training data
    %or even the whole thing
    autodamp = 0;
    denom = -0.5*double(chs{j}'*computeGV(chs{j})) - double(grad'*chs{j});
    autodamp = 1;
    rho = (oldll_chunk - ll_chunk)/denom;
    if oldll_chunk - ll_chunk > 0
        rho = -Inf;
    end

    outputString( ['rho = ' num2str(rho)] );

    chs = [];


    %bog-standard back-tracking line-search implementation:
    rate = 1.0;

    c = 10^(-2);
    j = 0;
    while j < 60

        if ll >= oldll + c*rate*double(grad'*p)
            break;
        else
            rate = 0.8*rate;
            j = j + 1;
            %outputString('#');
        end

        %this is computed on the whole dataset.  If this is not possible you can
        %use another set such the test set or a seperate validation set
        [ll, err] = computeLL(paramsp + conv(rate)*p, indata, outdata, numchunks);
    end

    if j == 60
        %completely reject the step
        j = Inf;
        rate = 0.0;
        ll = oldll;
    end

    outputString( ['Number of reductions : ' num2str(j) ', chosen rate: ' num2str(rate)] );


    %the damping heuristic (also very standard in optimization):
    if autodamp
        if rho < 0.25 || isnan(rho)
            lambda = lambda*boost;
        elseif rho > 0.75
            lambda = lambda*drop;
        end
        outputString(['New lambda: ' num2str(lambda)]);
    end
        

    %Parameter update:
    paramsp = paramsp + conv(rate)*p;

    lambdarecord(epoch,1) = lambda;

    llrecord(epoch,1) = ll;
    errrecord(epoch,1) = err;
    times(epoch) = toc;
    outputString( ['epoch: ' num2str(epoch) ', Log likelihood: ' num2str(ll) ', error rate: ' num2str(err) ] );

    [ll_test, err_test] = computeLL(paramsp, intest, outtest, numchunks_test);
    llrecord(epoch,2) = ll_test;
    errrecord(epoch,2) = err_test;
    outputString( ['TEST Log likelihood: ' num2str(ll_test) ', error rate: ' num2str(err_test) ] );
    
    outputString( ['Error rate difference (test - train): ' num2str(err_test-err)] );
    
    outputString( '' );

    pause(0)
    drawnow
    
    tmp = paramsp;
    paramsp = single(paramsp);
    tmp2 = ch;
    ch = single(ch);
    paramsp = tmp;
    ch = tmp2;

    clear tmp tmp2

end

paramsp = double(paramsp);

outputString( ['Total time: ' num2str(sum(times)) ] );

end