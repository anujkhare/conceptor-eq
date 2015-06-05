%%%% demo: being invariant to immediate signal transformations. Here we
%%%% consider linear shift, scaling, and exponentiation of a white noise
%%%% pattern.
%%%% Here we compensate only Wout by inverting the energy spectrum to the
%%%% reference spectrum before passing signals through Wout.
%%%% In addition we
%%%% use feedback of some time-lagged input reconstructions, and mix into
%%%% the input a fraction of reservoir-predicted next input. The number of
%%%% nets can be freely chosen.

%%%% Note: MismatchRatios{nNet} = Rref{nNet} ./ Ezsqr{nNet}; works better
%%%% than sqrt(MismatchRatios{nNet} = Rref{nNet} ./ Ezsqr{nNet}); although
%%%% the latter would be more mathematically appealing

function [energyErrs, autoCorrErrs] = working(no_plots, NNets)

set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment basic setup
randstate = 1; newNets = 1; newSystemScalings = 1;
newData = 1;

%%% System parameters
NMultiplier = ones(1, NNets); % length of this gives Nr of nets
% showNets = [1 length(NMultiplier)]; % which nets are to be diagnostic-plotted
showNets = [1]; % which nets are to be diagnostic-plotted
N = 50;  % network size
M = 200;  % RF space size
Nfb = 2; % number of feedbacks
SR = 1 ;  % spectral radius
WinScaling = .4 ;
WfbScaling = 0. ;
BiasScaling = 0. ;

%%% learning controls
washoutLength = 100;
learnLength = 1000;
COinitLength = 1000;
COadaptLength = 2000;
testLength = 1000;
TychWouts = 0.05 * NMultiplier; % regularizers for Wout
LRR = 0.005; % leaking rate for R estimation
% set aperture1 = Inf if no conceptors are to be inserted
apertures = Inf * NMultiplier;

mismatchExp = 1; % a value of 1/2 would be mathematically indicated
% larger, over-compensating values work better

% set filter function.
dataType = 1; % 1: NARMA data with NARMA coefficient filtering
% 2: rand data with shift, scale, exponentiation

if dataType == 1
    Filter = @(y, ym1, ym2, a, b, c, d, shift) ...
        tanh( a * y + b * (ym1) +...
        c * (ym2) * (ym1) + d *(rand - 0.5));
    filterWashout = 100;
    
    baselineParams = [2 -1 -2 0 ]; pattScaling = 1; pattShift = 1;
    
    
elseif dataType == 2
    Filter = @(y, a, b, c) a + b * (sign(y) * abs(y).^c);
    filterWashout = 100;
    baselineParams = [-2 .5 1];
end


%%% plotting specs
signalPlotLength = 40;
plotN = 8;
maxLag = 49; % for autocorrelation plots

%%% Initializations
randn('state', randstate);
rand('twister', randstate);


% Create raw weights
if newNets
    NNets = length(NMultiplier);
    WinRaw = randn(N, 1);
    WfbRaw = randn(N, Nfb);
    biasRaw = randn(N, 1);
    FRaw = randn(M,N);
    FRawRowNorms = sqrt(sum(FRaw.^2, 2));
    FRaw = diag(1 ./ FRawRowNorms) * FRaw;
    GstarRaw = randn(N,M);
    GF = full(GstarRaw * FRaw);
    specrad = max(abs(eig(GF)));
    FstarRaw = FRaw;
    GstarRaw = GstarRaw / specrad;
end

% Scale raw weights and initialize weights
if newSystemScalings
    
        F = FstarRaw;
        G = GstarRaw * SR;
        Win = WinScaling * WinRaw;
        Wfb = WfbScaling * WfbRaw;
        bias = BiasScaling * biasRaw;
    
end


% create training and testing data
if newData
    L = washoutLength + COinitLength + COadaptLength + learnLength;
    if dataType == 1
        trainPatt = 0.5*(sin(2 * pi * (1:L) / 8) + ...
            sin(2 * pi * (1:L) / 5.03));
        a = baselineParams(1);
        b = baselineParams(2);
        c = baselineParams(3);
        d = baselineParams(4);
        testPattProto = trainPatt; 
        testPatt = trainPatt;
        for n = 3:L
            testPatt(n) = ...
                Filter(trainPatt(n),testPatt(n-1),testPatt(n-2),...
                a, b, c, d);
        end
        testPatt = pattScaling * testPatt + pattShift;
    elseif dataType == 2
        trainPatt = rand(1, ...
            washoutLength + COinitLength + COadaptLength + learnLength);
        a = baselineParams(1);
        b = baselineParams(2);
        c = baselineParams(3);
        testPattProto = trainPatt;
        
        testPatt = trainPatt;
        for n = 1:L
            testPatt(n) = Filter(trainPatt(n), a, b, c);
        end
    end
    
end

figNr = 0;

if 0
    figNr = figNr + 1;
    figure(figNr); clf;
    hold on;
    plot((1:signalPlotLength) + (p-1)*signalPlotLength, ...
        trainPatt(1,end - signalPlotLength + 1:end));
    
    hold off;
    title('train pattern');
end

%% 2-module modeling
%% Compute Conceptor
zCollector = zeros(M, learnLength );
z = zeros(M, 1);


for n = Nfb+1:washoutLength + learnLength
    r = tanh(G * z + Win * trainPatt(1,n)...
        + Wfb * trainPatt(1,n-Nfb:n-1)' + bias);
    z = F * r;
    
    if n > washoutLength
        zCollector(:,n - washoutLength) = z;
    end
    
end

R = diag(zCollector * zCollector') / learnLength;
C{1} = R ./ (R + apertures(1)^(-2));
% replicate conceptor FOR ALL NETS
for nNet = 2:NNets
    C{nNet} = C{1};
end

%% Learn Rref,  Wout, and collect r state plot data
zCollector = zeros(M, learnLength );
pCollector = zeros(Nfb, learnLength);
predCollector = zeros(1, learnLength);
z = zeros(M,1);
% Here we are doing the same thing, but restricting the dynamics using
% learnt conceptor
for n = Nfb+1:washoutLength + learnLength
    
    r = tanh(G * z + Win * trainPatt(1,n)...
        + Wfb * trainPatt(1,n-Nfb:n-1)' + bias);
    z = C{nNet} .* (F * r);
    
    if n > washoutLength
        zCollector(:,n - washoutLength) = z;
        
        pCollector(:, n - washoutLength) = ...
            trainPatt(1,n-Nfb+1:n)';
        predCollector(1, n - washoutLength) = ...
            trainPatt(1,n+1);
        
    end
end

Rref = diag(zCollector * zCollector') / learnLength;  % This represents how the correlation matrix should look when C acts on ref. signal!
% Also note that R represents E[z^2] (ie, energy spectrum of the signal!)
args = zCollector;
targs = pCollector;
WoutAll = (pinv(args * args' / learnLength + ...
    TychWouts(nNet) * eye(M)) * args * targs' / learnLength)' ;     % why??
Wout = WoutAll(end,:);
ytrainNRMSE = nrmse(Wout * args, targs(end,:));

%% Testing
for nNet = 1:NNets
    yCollectortest{nNet} = zeros(1, testLength );   % Test Output from each net
    MismatchRatiosTest{nNet} = zeros(M);            % Test MR for each net
end
pCollectortest = zeros(1, testLength);
uCollectortest = zeros(1, testLength);

for nNet = 1:NNets
    rs{nNet} = zeros(N,1);
    zs{nNet} = zeros(M,1);
    yAll{nNet} = zeros(Nfb,1);
end
for n = 1:washoutLength             % just the WASHOUT period
    rs{1} = tanh(G * zs{1} + Win * testPatt(1,n)... % For the first setup
        + Wfb * yAll{1} + bias);
    zs{1} = C{1} .* (F * rs{1});
    yAll{1} = WoutAll * zs{1};
    for nNet = 2:NNets                              % For subsequent cascades
        rs{nNet} = tanh(G * zs{nNet} + ...
            Win * yAll{nNet-1}(end,1) + ...
            Wfb * yAll{nNet} + bias);
        zs{nNet} = C{nNet} .* (F * rs{nNet});
        yAll{nNet} = WoutAll * zs{nNet};
    end
end
% Initialize the average (expected) signal energy vector E[z.^2], called
% Ezsqr, for all nets in the cascade,
% by driving the cascade for COinitLength steps with input signal
shift = washoutLength;
for nNet = 1:NNets
    zColl{nNet} = zeros(M, COinitLength);
    Ezsqr{nNet} = zeros(M, 1);
    MismatchRatios{nNet} = zeros(M, 1);
end

for n = 1:COinitLength
    rs{1} = tanh(G * zs{1} + Win * testPatt(1,n+shift)...
        + Wfb * yAll{1} + bias);
    zs{1} = C{1} .* (F * rs{1});
    yAll{1} = WoutAll * zs{1};
    zColl{1}(:,n) = zs{1};
    for nNet = 2:NNets
        rs{nNet} = tanh(G * zs{nNet} + ...
            Win * yAll{nNet-1}(end,1) + ...
            Wfb * yAll{nNet} + bias);
        zs{nNet} = C{nNet} .* (F * rs{nNet});
        yAll{nNet} = WoutAll * zs{nNet};
        zColl{nNet}(:,n) = zs{nNet};
    end
end
for nNet = 1:NNets
    Ezsqr{nNet} = diag(zColl{nNet} * zColl{nNet}') / COinitLength;
    % the mismatch ratios will function as the "ERROR" term that is used to pull
    % the actual signal energies of the z vectors toward the reference z
    % vectors known from the "clean" training input
    MismatchRatios{nNet} = (Rref ./ Ezsqr{nNet}).^mismatchExp;
end

%% Adapt forward through nets for COadaptLength
shift = washoutLength + COinitLength;

% plotInd = 0;
for n = 1:COadaptLength
    rs{1} = tanh(G * zs{1} + Win * ...
        testPatt(1,n+shift) ...
        + Wfb * yAll{1} + bias);
    zs{1} = C{1} .* (F * rs{1});
    % in the next two lines, the core adaptation is done, by re-shaping the
    % z vector with the help of the mismatch ratios which pull it toward
    % the reference z signal energy profile known from training

    zs{1} = MismatchRatios{1} .* zs{1};
    yAll{1} = WoutAll * (zs{1});
%    yAll{1} = WoutAll * (MismatchRatios{1} .* zs{1});
    
    % the following updates the estimate of Ezsqr and the mismatch ratio
    Ezsqr{1} = (1-LRR) * Ezsqr{1} + LRR * zs{1}.^2;
    MismatchRatios{1} = (Rref ./ Ezsqr{1}).^mismatchExp;
    for nNet = 2:NNets
        rs{nNet} = tanh(G * zs{nNet} + ...
            Win * ...
            yAll{nNet-1}(end,1) +...
            Wfb * yAll{nNet} + bias);
        zs{nNet} = C{nNet} .* (F * rs{nNet});
   
        zs{nNet} = MismatchRatios{nNet} .* zs{nNet};
        yAll{nNet} = WoutAll * (zs{nNet});
%         yAll{nNet} = WoutAll * (MismatchRatios{nNet} .* zs{nNet});
        
        Ezsqr{nNet} = (1-LRR) * Ezsqr{nNet} + LRR * zs{nNet}.^2;
        MismatchRatios{nNet} = (Rref ./ Ezsqr{nNet}).^mismatchExp;
    end
end

%% Finally, stop adapting, stay in the last adapted configuaration
% and collect data for plotting and error diagnostics
shift = washoutLength + COinitLength + COadaptLength;
for n = 1:testLength
    u = testPatt(1,n+shift);
    rs{1} = tanh(G * zs{1} + Win * ...
        u + Wfb * yAll{1} + bias);
    zs{1} = C{1} .* (F * rs{1});
    yAll{1} = WoutAll * (MismatchRatios{1} .* zs{1});
    yCollectortest{1}(:,n) = yAll{1}(end,1);
    for nNet = 2:NNets
        rs{nNet} = tanh(G * zs{nNet} + ...
            Win * ...
            yAll{nNet-1}(end,1)  +...
            Wfb * yAll{nNet} + bias);
        zs{nNet} = C{nNet} .* (F * rs{nNet});
        yAll{nNet} = WoutAll * (MismatchRatios{nNet} .* zs{nNet});
        yCollectortest{nNet}(:,n) = yAll{nNet}(end,1);
    end
    
    pCollectortest(:,n) = ...
        testPattProto(1,n + shift);
    uCollectortest(:,n) = u;
end
NNets=1; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate errors
for nNet = 1:NNets
    ytestNRMSE{nNet} = nrmse(yCollectortest{nNet}, ...
        pCollectortest);
    EngyRatios{nNet} = Rref ./ Ezsqr{nNet};
end

rawNRMSE = nrmse(testPattProto, testPatt);  %% why? what??


autoCorrP = autocorr(pCollectortest, maxLag);
for nNet = 1:NNets
    autoCorry{nNet} = autocorr(yCollectortest{nNet}, maxLag);
end

energyErrs = zeros(1,NNets);
for nNet = 1:NNets
    energyErrs(1,nNet) = ...
        norm((Rref - Ezsqr{nNet}) / norm(Rref))^2;
end

%% Calculations

testNRMSEs = zeros(1,NNets);
autoCorrErrs = zeros(1,NNets);
for nNet = 1:NNets
    testNRMSEs(nNet) = ytestNRMSE{nNet};
    autoCorrErrs(nNet) = ...
        norm((autoCorrP - autoCorry{nNet}) / norm(autoCorrP) )^2 ;
end

disp('***************************');
fprintf('raw NRMSE = %0.3g\n',  rawNRMSE);
fprintf('meanabs Wout = %0.3g\n',  mean(abs(Wout)));
fprintf('train NRMSEs = %0.3g\n', ytrainNRMSE);
disp(['test  NRMSEs = ' num2str(testNRMSEs, ' %0.3g')]);
disp(['energyErrs = ' num2str(energyErrs, ' %0.3g')]);
disp(['autoCorrErrs = ' num2str(autoCorrErrs, ' %0.3g')]);

%% Plots

if no_plots == 1
    return
end

% Energy Ratios
for nNet = showNets
    figNr = figNr + 1;
    figure(figNr); clf;
    hold on;
    plot(EngyRatios{nNet});
    title(sprintf('Energy ratios (unsorted) in %g', nNet));
%     plot(sort(EngyRatios{nNet}, 'descend'), '.');
    hold off;
%     title(sprintf('Energy ratios in %g', nNet));
    
end

% Autocorrelations
for nNet = showNets
    figNr = figNr + 1;
    figure(figNr); clf;
    hold on;
    plot(autoCorrP, 'r', 'LineWidth',2);
    plot(autoCorry{nNet}, 'b', 'LineWidth',2);
    hold off;
    title(sprintf('Autocorrs in %g (r=orig)', nNet));
    
end

% Signals
maxy = -10; miny = 10;
for nNet = 1:NNets

    raws{nNet} = uCollectortest(1, ...
        end - signalPlotLength + 1 : end);
    maxy = max(maxy, max(raws{nNet}));
    miny = min(miny, min(raws{nNet}));
    targets{nNet} = pCollectortest(1, ...
        end - signalPlotLength + 1 : end);
    maxy = max(maxy, max(targets{nNet}));
    miny = min(miny, min(targets{nNet}));
    effectives{nNet} = yCollectortest{nNet}(1, ...
        end - signalPlotLength + 1 : end);
    maxy = max(maxy, max(effectives{nNet}));
    miny = min(miny, min(effectives{nNet}));
    
end
for nNet = showNets
    figNr = figNr + 1;
    figure(figNr); clf;
    hold on;
    plot(1:signalPlotLength, raws{nNet},  ...
        'Color',0.75* [1 1 1],'LineWidth',3);
    
    plot(1:signalPlotLength, targets{nNet}, ...
        'r','LineWidth',2);
    plot(1:signalPlotLength, effectives{nNet}, ...
        'b','LineWidth',2);
    
    
    hold off;
    set(gca, 'YLim',[miny-0.2 maxy+0.2], 'XLim', [1 signalPlotLength]); % why?
    title(sprintf('y (%g) test out vs target (r)', nNet));
end

% Energy error and autocorr error plot
figNr = figNr + 1;
figure(figNr); clf;
hold on;
plot(log10(energyErrs),'bx-', 'LineWidth',2);
plot(log10(autoCorrErrs),'gx-', 'LineWidth',2);
hold off;
title('log10 energyErrs(b) autoCErrs(g)');
