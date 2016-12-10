function outdata = hf_train()
tmp = load('ex4data1.mat');

indata = tmp.X';
outdata = zeros(10,length(tmp.y));
for i = 1:length(tmp.y)
    outdata(tmp.y(i),i) = 1;
end

%next try using autodamp = 0 for rho computation.  both for version 6 and
%versions with rho and cg-backtrack computed on the training set

%mattype = 'gn'; %Curvature matrix: Gauss-Newton.

% tmp = load('digs3pts_1.mat');
% indata = tmp.bdata';
% intest = tmp.bdatatest';

perm = randperm(size(indata,2));
intmp = indata( :, perm );
outtmp = outdata(:, perm);

% indata = intmp;
% outdata = outtmp;

indata = intmp(:, 1:2500);
outdata = outtmp(:, 1:2500);

intest = intmp(:, 2501:5000);
outtest = outtmp(:, 2501:5000);
%it's an auto-encoder so output is input
% outtest = outdata;
% intest = indata;

% decay - the amount to decay the previous search direction for the
% purposes of initializing the next run of CG.
decay = 0.95; % Should be 0.95

maxepoch = 100;

paramsp = [];

layersizes = [25 30];
layertypes = {'logistic', 'logistic', 'logistic'};

%standard L_2 weight-decay:
weightcost = 2e-5;
weightcost = 2e-4;
%weightcost = 1;

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



autodamp = 1;

drop = 2/3;

boost = 1/drop;

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
outputString(['input size:' num2str(indims) 'x' num2str(numcases)])

y = cell(1, numlayers+1);

psize = layersizes(1,2:(numlayers+1))*layersizes(1,1:numlayers)' + sum(layersizes(2:(numlayers+1)));

%pack all the parameters into a single vector for easy manipulation
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

%unpack parameters from a vector
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
    
    GV = zeros(psize,1);
    
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

        yip1 = y{chunk, 1};

        %forward prop:
        Ryip1 = zeros(layersizes(1), numcases);
            
        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 numcases]);

            yip1 = y{chunk, i+1};

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

            yi = y{chunk, i};

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
    
    GV = GV / numcases;
    
    
    
    GV = GV - weightcost*(maskp.*V);

    if autodamp
        GV = GV - lambda*V;
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
    
        yi = in(:, ((chunk-1)*schunk+1):(chunk*schunk) );
        outc = out(:, ((chunk-1)*schunk+1):(chunk*schunk) );

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
   
        err = err + double(sum( sum(outc.*yi,1) ~= max(yi,[],1) ) ) / size(in,2);
        
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



maskp = ones(psize,1);
[maskW, maskb] = unpack(maskp);
disp('not masking out the weight-decay for biases');
maskp = pack(maskW,maskb);


indata = single(indata);
outdata = single(outdata);
intest = single(intest);
outtest = single(outtest);




ch = zeros(psize, 1);


    
lambda = initlambda;

llrecord = zeros(maxepoch,2);
errrecord = zeros(maxepoch,2);
lambdarecord = zeros(maxepoch,1);
times = zeros(maxepoch,1);

totalpasses = 0;
epoch = 1;
    


%if isempty(paramsp)
%SPARSE INIT:
paramsp = zeros(psize,1); %not zeros

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


outputString( '=================================================' );

for epoch = epoch:maxepoch
    tic

    targetchunk = mod(epoch-1, 1)+1;
    
    [Wu, bu] = unpack(paramsp);


    y = cell(1, numlayers+1);
    x = cell(1, numlayers+1);

    grad = zeros(psize,1);
    grad2 = zeros(psize,1);
    
    ll = 0;

    %forward prop:
    %index transition takes place at nonlinearity
    for chunk = 1:1
        
        y{chunk, 1} = indata(:, ((chunk-1)*numcases+1):(chunk*numcases) );
        yip1 =  y{chunk, 1} ;

        dEdW = cell(numlayers, 1);
        dEdb = cell(numlayers, 1);

        dEdW2 = cell(numlayers, 1);
        dEdb2 = cell(numlayers, 1);

        for i = 1:numlayers

            yi = yip1;
            yip1 = [];
            xi = Wu{i}*yi + repmat(bu{i}, [1 numcases]);
            yi = [];

            if strcmp(layertypes{i}, 'logistic')
                yip1 = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'linear')
                yip1 = xi;
            else
                error( 'Unknown/unsupported layer type' );
            end
            
            y{chunk, i+1} = yip1;
        end

        %back prop:
        %cross-entropy for logistics:
        %dEdy{numlayers+1} = outdata./y{numlayers+1} - (1-outdata)./(1-y{numlayers+1});
        %cross-entropy for softmax:
        %dEdy{numlayers+1} = outdata./y{numlayers+1};

        if chunk ~= targetchunk
            y{chunk, numlayers+1} = []; %save memory
        end

        outc = outdata(:, ((chunk-1)*numcases+1):(chunk*numcases) );
        
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

            yi = y{chunk, i};

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
    
    grad = grad / numcases;
    grad = grad - weightcost*(maskp.*paramsp);
    
    grad2 = grad2 / numcases;
    
    gradchunk = gradchunk/numcases - weightcost*(maskp.*paramsp);
    grad2chunk = grad2chunk/numcases;
    
    ll = ll / numcases;
    
    ll = ll - 0.5*weightcost*double(paramsp'*(maskp.*paramsp));
    
    
    oldll = ll;
    ll = [];

  
    %slightly decay the previous change vector before using it as an
    %initialization.  This is something I didn't mention in the paper,
    %and it's not overly important but it can help a lot in some situations 
    %so you should probably use it
    ch = decay*ch;

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
    precon = (grad2 + ones(psize,1)*lambda + maskp*weightcost).^(3/4);
    %precon = ones(psize,1);

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
    [ll, err] = computeLL(paramsp + p, indata, outdata, 1);
    for j = (length(chs)-1):-1:1
        [lowll, lowerr] = computeLL(paramsp + chs{j}, indata, outdata, 1);

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


    [ll_chunk, err_chunk] = computeLL(paramsp + chs{j}, indata, outdata, 1, targetchunk);
    [oldll_chunk, olderr_chunk] = computeLL(paramsp, indata, outdata, 1, targetchunk);

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
        [ll, err] = computeLL(paramsp + rate*p, indata, outdata, 1);
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
    paramsp = paramsp + rate*p;

    lambdarecord(epoch,1) = lambda;

    llrecord(epoch,1) = ll;
    errrecord(epoch,1) = err;
    times(epoch) = toc;
    outputString( ['epoch: ' num2str(epoch) ', Log likelihood: ' num2str(ll) ', error rate: ' num2str(err)] );

    [ll_test, err_test] = computeLL(paramsp, intest, outtest, 1);
    llrecord(epoch,2) = ll_test;
    errrecord(epoch,2) = err_test;
    outputString( ['TEST Log likelihood: ' num2str(ll_test) ', error rate: ' num2str(err_test)] );
    outputString( '' );

end

outputString( ['Total time: ' num2str(sum(times)) ] );

end

function outputString( s )
    fprintf( 1, '%s\n', s );
end

function v = vec(A)
    v = A(:);
end