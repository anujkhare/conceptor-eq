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



set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment basic setup
randstate = 1; newNets = 1; newSystemScalings = 1;
learnModel = 1;
newData = 1;
showSlidePics = 1;


%%% System parameters
NMultiplier = [1 1 1 1 1 ]; % length of this gives Nr of nets
showNets = [1 3 length(NMultiplier)]; % which nets are to be diagnostic-plotted
Ns = 50*NMultiplier;  % network sizes
Ms = 200*NMultiplier;  % RF space size
Nfb = 2; % number of feedbacks
SRs = 1 * NMultiplier;  % spectral radii
WinScalings = .4 * NMultiplier;
WfbScalings = 0. * NMultiplier;
BiasScalings = 0. * NMultiplier;
PredFracs =  0. * NMultiplier; % how much of the input is replaced
% by prediction feedback in testing

%%% learning controls
washoutLength = 100;
learnLength = 1000;
COinitLength = 1000;
COadaptLength = 2000;
testLength = 1000;
TychWouts = 0.05 * NMultiplier;
TychWpreds = 0.05 * NMultiplier;
% set TychEqui1 = 0 if no equilibration is to be done:
TychEquis = 0.00000000001 * NMultiplier;
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
    
    baselineParams = [1 0 0 0]; pattScaling = 5; pattShift = 5;
    baselineParams = [1 0 0 0]; pattScaling = .1; pattShift = 1;
    baselineParams = [.5 .6 1 0 ]; pattScaling = 1; pattShift = 1;
    %baselineParams = [1 0 0 1 ]; pattScaling = 1; pattShift = 1;
    baselineParams = [2 -1 -2 0 ]; pattScaling = 1; pattShift = 1;
    %baselineParams = [.2 1 -.3 0 ]; pattScaling = 1; pattShift = 0;
    %baselineParams = [3 0 3 0 ]; pattScaling = 1; pattShift = -1;
    %baselineParams = [1 0 0 2]; pattScaling = 1; pattShift = 0;
    
    
    
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
    for n = 1:NNets
        N = Ns(n); M = Ms(n);
        WinRaw{n} = randn(N, 1);
        WfbRaw{n} = randn(N, Nfb);
        biasRaw{n} = randn(N, 1);
        FRaw = randn(M,N);
        FRawRowNorms = sqrt(sum(FRaw.^2,2));
        FRaw = diag(1 ./ FRawRowNorms) * FRaw;
        GstarRaw{n} = randn(N,M);
        GF = full(GstarRaw{n} * FRaw);
        specrad = max(abs(eig(GF)));
        FstarRaw{n} = FRaw;
        GstarRaw{n} = GstarRaw{n} / specrad;
    end
end

% Scale raw weights and initialize weights
if newSystemScalings
    for nNet = 1:NNets
        Fstar{nNet} = FstarRaw{nNet};
        Gstar{nNet} = GstarRaw{nNet} * SRs(nNet);
        Win{nNet} = WinScalings(nNet) * WinRaw{nNet};
        Wfb{nNet} = WfbScalings(nNet) * WfbRaw{nNet};
        bias{nNet} = BiasScalings(nNet) * biasRaw{nNet};
    end
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

%%% 2-module modeling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if learnModel
% compute conceptors
for nNet = 1:NNets
    zCollectors{nNet} = zeros(Ms(nNet), learnLength );
    z{nNet} = zeros(Ms(nNet), 1);
end


for n = Nfb+1:washoutLength + learnLength
    for nNet = 1:NNets
        r{nNet} = tanh(Gstar{nNet} * z{nNet} + Win{nNet} * trainPatt(1,n)...
            + Wfb{nNet} * trainPatt(1,n-Nfb:n-1)' + bias{nNet});
        z{nNet} = Fstar{nNet} * r{nNet};
        
        if n > washoutLength
            zCollectors{nNet}(:,n - washoutLength) = z{nNet};
        end
    end
end
for nNet = 1:NNets
    R = diag(zCollectors{nNet} * zCollectors{nNet}') / learnLength;
    C{nNet} = R ./ (R + apertures(nNet)^(-2));
end


% equilibrate nets 
if TychEquis(1) == 0
    for nNet = 1:NNets
        G{nNet} = Gstar{nNet};
        F{nNet} = Fstar{nNet};
    end
else
    for nNet = 1:NNets
        zCollector{nNet} = zeros(Ms(nNet), learnLength );
        rCollector{nNet} = zeros(Ns(nNet), learnLength );
    end
    uCollector = zeros(1, learnLength);
    uMinus1Collector = zeros(Nfb, learnLength);
    for nNet = 1:NNets
        z{nNet} = zeros(Ms(nNet),1);
    end
    for n = Nfb+1:washoutLength + learnLength
        for nNet = 1:NNets
            r{nNet} = tanh(Gstar{nNet} * z{nNet} + Win{nNet} * trainPatt(1,n)...
                + Wfb{nNet} * trainPatt(1,n-Nfb:n-1)' + bias{nNet});
            z{nNet} = C{nNet} .* (Fstar{nNet} * r{nNet});
            
            if n > washoutLength
                zCollector{nNet}(:,n - washoutLength) = z{nNet};
                rCollector{nNet}(:,n - washoutLength) = r{nNet};
                if nNet == 1
                    uCollector(:,n - washoutLength) = trainPatt(1,n);
                    uMinus1Collector(:,n - washoutLength) = ...
                        trainPatt(1,n-Nfb:n-1);
                end
            end
        end
    end
    
    for nNet = 1:NNets
        args = [zCollector{nNet}; uCollector; uMinus1Collector];
        targs = Gstar{nNet} * zCollector{nNet} + Win{nNet} * uCollector ...
            + Wfb{nNet} * uMinus1Collector ;
        GWinWfb = (inv(args * args' / learnLength + ...
            TychEquis(nNet) * eye(Ms(nNet)+Nfb+1)) * args * targs' / learnLength)' ;
        WinOriginal{nNet} = Win{nNet}; WfbOriginal{nNet} = Wfb{nNet};
        G{nNet} = GWinWfb(:,1:Ms(nNet));
        Win{nNet} = GWinWfb(:,Ms(nNet)+1);
        Wfb{nNet} = GWinWfb(:,Ms(nNet)+2:end);
        
        args = rCollector{nNet}; targs = Fstar{nNet} * rCollector{nNet};
        F{nNet} = (inv(args * args' / learnLength + ...
            TychEquis(nNet) * eye(Ns(nNet))) * args * targs' / learnLength)' ;
    end
    
    
end

% learn Rref,  Wout, Wpred, and collect r state plot data
for nNet = 1:NNets
    rPL{nNet} = zeros(plotN, learnLength );
    zCollector{nNet} = zeros(Ms(nNet), learnLength );
end
pCollector = zeros(Nfb, learnLength);
predCollector = zeros(1, learnLength);
for nNet = 1:NNets
    z{nNet} = zeros(Ms(nNet),1);
end

for n = Nfb+1:washoutLength + learnLength
    for nNet = 1:NNets
        r{nNet} = tanh(G{nNet} * z{nNet} + Win{nNet} * trainPatt(1,n)...
            + Wfb{nNet} * trainPatt(1,n-Nfb:n-1)' + bias{nNet});
        z{nNet} = C{nNet} .* (F{nNet} * r{nNet});
        
        if n > washoutLength
            rPL{nNet}(:, n - washoutLength) = r{nNet}(1:plotN,1);
            zCollector{nNet}(:,n - washoutLength) = z{nNet};
            if nNet == 1
                pCollector(:, n - washoutLength) = ...
                    trainPatt(1,n-Nfb+1:n)';
                predCollector(1, n - washoutLength) = ...
                    trainPatt(1,n+1);
            end
        end
    end
end
for nNet = 1:NNets
    Rref{nNet} = diag(zCollector{nNet} * zCollector{nNet}') / learnLength;
    
    args = zCollector{nNet};
    targs = pCollector;
    WoutAll{nNet} = (inv(args * args' / learnLength + ...
        TychWouts(nNet) * eye(Ms(nNet))) * args * targs' / learnLength)' ;
    Wout{nNet} = WoutAll{nNet}(end,:);
    ytrainNRMSE{nNet} = nrmse(Wout{nNet} * args, targs(end,:));
    
    args = zCollector{nNet};
    targs = predCollector;
    Wpred{nNet} = (inv(args * args' / learnLength + ...
        TychWpreds(nNet) * eye(Ms(nNet))) * args * targs' / learnLength)' ;
    ypredNRMSE{nNet} = nrmse(Wpred{nNet} * args, targs);
end
end

%%
% testing
for nNet = 1:NNets
    yCollectortest{nNet} = zeros(1, testLength );
    MismatchRatiosTest{nNet} = zeros(Ms(nNet));
end
pCollectortest = zeros(1, testLength);
uCollectortest = zeros(1, testLength);



for nNet = 1:NNets
    z{nNet} = zeros(Ms(nNet),1);
    yAll{nNet} = zeros(Nfb,1);
end
for n = 1:washoutLength
    r{1} = tanh(G{1} * z{1} + Win{1} * testPatt(1,n)...
        + Wfb{1} * yAll{1} + bias{1});
    z{1} = C{1} .* (F{1} * r{1});
    yAll{1} = WoutAll{1} * z{1};
    for nNet = 2:NNets
        r{nNet} = tanh(G{nNet} * z{nNet} + ...
            Win{nNet} * yAll{nNet-1}(end,1) + ...
            Wfb{nNet} * yAll{nNet} + bias{nNet});
        z{nNet} = C{nNet} .* (F{nNet} * r{nNet});
        yAll{nNet} = WoutAll{nNet} * z{nNet};
    end
end
% initialize R
shift = washoutLength;
for nNet = 1:NNets
    zColl{nNet} = zeros(Ms(nNet), COinitLength);
end

for n = 1:COinitLength
    r{1} = tanh(G{1} * z{1} + Win{1} * testPatt(1,n+shift)...
        + Wfb{1} * yAll{1} + bias{1});
    z{1} = C{1} .* (F{1} * r{1});
    yAll{1} = WoutAll{1} * z{1};
    yPred{1} = Wpred{1} * z{1};
    zColl{1}(:,n) = z{1};
    for nNet = 2:NNets
        r{nNet} = tanh(G{nNet} * z{nNet} + ...
            Win{nNet} * yAll{nNet-1}(end,1) + ...
            Wfb{nNet} * yAll{nNet} + bias{nNet});
        z{nNet} = C{nNet} .* (F{nNet} * r{nNet});
        yAll{nNet} = WoutAll{nNet} * z{nNet};
        yPred{nNet} = Wpred{nNet} * z{nNet};
        zColl{nNet}(:,n) = z{nNet};
    end
end
for nNet = 1:NNets
    Ezsqr{nNet} = diag(zColl{nNet} * zColl{nNet}') / COinitLength;
    MismatchRatios{nNet} = (Rref{nNet} ./ Ezsqr{nNet}).^mismatchExp;
end



% adapt forward through nets
shift = washoutLength + COinitLength;

plotInd = 0;
for n = 1:COadaptLength
    r{1} = tanh(G{1} * z{1} + Win{1} * ...
        ((1-PredFracs(1))* testPatt(1,n+shift) + ...
        PredFracs(1) * yPred{1})...
        + Wfb{1} * yAll{1} + bias{1});
    z{1} = C{1} .* (F{1} * r{1});
    yAll{1} = WoutAll{1} * (MismatchRatios{1} .* z{1});
    yPred{1} = Wpred{1} * (MismatchRatios{1} .* z{1});
    Ezsqr{1} = (1-LRR) * Ezsqr{1} + LRR * z{1}.^2;
    MismatchRatios{1} = (Rref{1} ./ Ezsqr{1}).^mismatchExp;
    for nNet = 2:NNets
        r{nNet} = tanh(G{nNet} * z{nNet} + ...
            Win{nNet} * ...
            ((1-PredFracs(1))* yAll{nNet-1}(end,1) + ...
            PredFracs(nNet) * yPred{nNet}) +...
            Wfb{nNet} * yAll{nNet} + bias{nNet});
        z{nNet} = C{nNet} .* (F{nNet} * r{nNet});
        yAll{nNet} = WoutAll{nNet} * (MismatchRatios{nNet} .* z{nNet});
        yPred{nNet} = Wpred{nNet} * (MismatchRatios{nNet} .* z{nNet});
        Ezsqr{nNet} = (1-LRR) * Ezsqr{nNet} + LRR * z{nNet}.^2;
        MismatchRatios{nNet} = (Rref{nNet} ./ Ezsqr{nNet}).^mismatchExp;
    end
end

shift = washoutLength + COinitLength + COadaptLength;
for n = 1:testLength
    u = testPatt(1,n+shift);
    r{1} = tanh(G{1} * z{1} + Win{1} * ...
        ((1-PredFracs(1))* u + ...
        PredFracs(1) * yPred{1})...
        + Wfb{1} * yAll{1} + bias{1});
    z{1} = C{1} .* (F{1} * r{1});
    yAll{1} = WoutAll{1} * (MismatchRatios{1} .* z{1});
    yPred{1} = Wpred{1} * (MismatchRatios{1} .* z{1});
    yCollectortest{1}(:,n) = yAll{1}(end,1);
    for nNet = 2:NNets
        r{nNet} = tanh(G{nNet} * z{nNet} + ...
            Win{nNet} * ...
            ((1-PredFracs(1))* yAll{nNet-1}(end,1) + ...
            PredFracs(nNet) * yPred{nNet}) +...
            Wfb{nNet} * yAll{nNet} + bias{nNet});
        z{nNet} = C{nNet} .* (F{nNet} * r{nNet});
        yAll{nNet} = WoutAll{nNet} * (MismatchRatios{nNet} .* z{nNet});
        yPred{nNet} = Wpred{nNet} * (MismatchRatios{nNet} .* z{nNet});
        yCollectortest{nNet}(:,n) = yAll{nNet}(end,1);
    end
    
    pCollectortest(:,n) = ...
        testPattProto(1,n + shift);
    uCollectortest(:,n) = u;
end


for nNet = 1:NNets
    ytestNRMSE{nNet} = nrmse(yCollectortest{nNet}, ...
        pCollectortest);
    EngyRatios{nNet} = Rref{nNet} ./ Ezsqr{nNet};
end

rawNRMSE = nrmse(testPattProto, testPatt);


autoCorrP = autocorr(pCollectortest, maxLag);
for nNet = 1:NNets
    autoCorry{nNet} = autocorr(yCollectortest{nNet}, maxLag);
end

engyErrs = zeros(1,NNets);
for nNet = 1:NNets
    engyErrs(1,nNet) = ...
        norm((Rref{nNet} - Ezsqr{nNet}) / norm(Rref{nNet}))^2;
end

for nNet = showNets
    figNr = figNr + 1;
    figure(figNr); clf;
    hold on;
    plot(sort(EngyRatios{nNet}, 'descend'), '.');
    hold off;
    title(sprintf('%g engy ratios', nNet));
    
end


for nNet = showNets
    figNr = figNr + 1;
    figure(figNr); clf;
    hold on;
    plot(autoCorrP, 'r', 'LineWidth',2);
    plot(autoCorry{nNet}, 'b', 'LineWidth',2);
    hold off;
    title(sprintf('%g autocorrs (r=orig)', nNet));
    
end


if 0
    for nNet = showNets
        figNr = figNr + 1;
        figure(figNr); clf;
        plot(rPL{nNet}(:,end - signalPlotLength + 1 : end)');
        set(gca, 'YLim',[-1 1]);
        title(sprintf('net %g r states', nNet));
    end
end


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
    set(gca, 'YLim',[miny-0.2 maxy+0.2], 'XLim', [1 signalPlotLength]);
    title(sprintf('y %g test out vs target (r)', nNet));
end


%%

trainNRMSEs = zeros(1,NNets);
predNRMSEs = zeros(1,NNets);
testNRMSEs = zeros(1,NNets);
autoCorrErrs = zeros(1,NNets);
meanabsGWinWfbOriginal = zeros(1,NNets);
meanabsGWinWfb = zeros(1,NNets);
meanabsWout = zeros(1,NNets);
for nNet = 1:NNets
    meanabsGWinWfbOriginal(nNet) = mean(mean(abs([Gstar{nNet}...
        WinOriginal{nNet} WfbOriginal{nNet}])));
    meanabsGWinWfb(nNet) = mean(mean(abs([G{nNet}...
        Win{nNet} Wfb{nNet}])));
    trainNRMSEs(nNet) = ytrainNRMSE{nNet};
    predNRMSEs(nNet) = ypredNRMSE{nNet};
    testNRMSEs(nNet) = ytestNRMSE{nNet};
    meanabsWout(nNet) = mean(abs(Wout{nNet}));
    autoCorrErrs(nNet) = ...
        norm((autoCorrP - autoCorry{nNet}) / norm(autoCorrP) )^2 ;
end

% figNr = figNr + 1;
% figure(figNr); clf;
% hold on;
% plot(log10(engyErrs),'b', 'LineWidth',2);
% plot(log10(autoCorrErrs),'g', 'LineWidth',2);
% hold off;
% title('log10 engyErrs(b) autoCErrs(g)');


disp('************* W V **************');
disp(sprintf('raw NRMSE = %0.3g',  rawNRMSE));
disp(['meanabs GWinWfbOriginal = ' num2str(meanabsGWinWfbOriginal, ' %0.2g')]);
disp(['meanabs GWinWfb         = ' num2str(meanabsGWinWfb, ' %0.2g')]);
disp(['meanabs Wout         = ' num2str(meanabsWout, ' %0.2g')]);
disp(['pred NRMSEs  = ' num2str(predNRMSEs, ' %0.3g')]);
disp(['train NRMSEs = ' num2str(trainNRMSEs, ' %0.3g')]);
disp(['test  NRMSEs = ' num2str(testNRMSEs, ' %0.3g')]);
disp(['engyErrs = ' num2str(engyErrs, ' %0.3g')]);
disp(['autoCorrErrs = ' num2str(autoCorrErrs, ' %0.3g')]);

%% pics for slides
if showSlidePics
% %     figure(20); clf;
% %     set(gcf, 'WindowStyle','normal');
% %     set(gcf, 'Position',[850 200 200 150]);
% %     plot(1:signalPlotLength, targets{nNet}, ...
% %         'r','LineWidth',2);
% %     set(gca, 'FontSize',14);
% %     %%
% %     figure(21); clf;
% %     set(gcf, 'WindowStyle','normal');
% %     set(gcf, 'Position',[850 200 200 150]);
% %     plot(1:signalPlotLength, targets{nNet}, ...
% %         'r','LineWidth',2); hold on;
% %     plot(1:signalPlotLength, raws{nNet},  ...
% %         'Color',0.75* [1 1 1],'LineWidth',3);
% %     hold off;
% %     set(gca, 'FontSize',14);
% %     %%
% %     figure(22); clf;
% %     set(gcf, 'WindowStyle','normal');
% %     set(gcf, 'Position',[850 200 200 150]);
% %     plot(1:signalPlotLength, targets{nNet}, ...
% %         'r','LineWidth',2); hold on;
% %     plot(1:signalPlotLength, raws{nNet},  ...
% %         'Color',0.75* [1 1 1],'LineWidth',3);
% %     plot(1:signalPlotLength, effectives{5}, ...
% %         'b','LineWidth',2);
% %     hold off;
% %     set(gca, 'FontSize',14);
%     %%
%     figure(23); clf;
%     set(gcf, 'WindowStyle','normal');
%     set(gcf, 'Position',[850 200 500 150]);
%     subplot(1,2,1);
%     hold on;
%     plot(1:signalPlotLength, raws{NNets},  ...
%         'Color',0.6* [1 1 1],'LineWidth',3);
%     plot(1:signalPlotLength, targets{NNets}, ...
%         'r','LineWidth',2); 
%     
%     plot(1:signalPlotLength, effectives{5}, ...
%         'b','LineWidth',2);
%     hold off;
%     set(gca, 'FontSize',16, 'Box', 'on');
%     subplot(1,2,2);
%     hold on;
%     plot(autoCorrP, 'r', 'LineWidth',2);
%     plot(autoCorry{NNets}, 'b', 'LineWidth',2);
%     hold off;
%     set(gca, 'FontSize',16, 'Box', 'on');
    
    
end
% % 
% %     figure(20); clf;
% %     set(gcf, 'WindowStyle','normal');
% %     set(gcf, 'Position',[750 700 400 200]);
% %     hold on;
% % plot(log10(engyErrs),'b', 'LineWidth',3);
% % plot(log10(autoCorrErrs),'g', 'LineWidth',3);
% % hold off;
% % title('log10 engyErrs(b) autoCErrs(g)', 'FontSize',16);
% % set(gca, 'FontSize',16);
% %     
% % figure(21); clf;
% %     set(gcf, 'WindowStyle','normal');
% %     set(gcf, 'Position',[750 200 450 400]);
% %     hold on;
% %     plot(1:signalPlotLength, raws{nNet},  ...
% %         'Color',0.6* [1 1 1],'LineWidth',5);
% %     plot(1:signalPlotLength, targets{nNet}, ...
% %         'r','LineWidth',3); 
% %     plot(1:signalPlotLength, effectives{5}, ...
% %         'b','LineWidth',3);
% %     hold off;
% %     title('input(gray) reference(r) out(blue)','FontSize',16);
% %     set(gca, 'FontSize',16);
% %     
% % 
% % figure(22); clf;
% %     set(gcf, 'WindowStyle','normal');
% %     
% %     set(gcf, 'Position',[1200 200 450 400]);
% %     hold on;
% %     plot(autoCorrP, 'r', 'LineWidth',2);
% %     plot(autoCorry{nNet}, 'b', 'LineWidth',2);
% %     hold off;
% %     title(sprintf('autocorrelations ref(red) out(blue)'), 'FontSize', 16);
% %     set(gca, 'FontSize',16);


    
