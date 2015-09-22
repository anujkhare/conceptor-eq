#     disp('***************************');
#     fprintf('raw NRMSE = #0.3g\n',  rawNRMSE);
#     fprintf('meanabs Wout = #0.3g\n',  mean(abs(Wout)));
#     fprintf('train NRMSEs = #0.3g\n', ytrainNRMSE);
#     disp(['test  NRMSEs = ' num2str(testNRMSEs, ' #0.3g')]);
#     disp(['energyErrs = ' num2str(energyErrs, ' #0.3g')]);
#     disp(['autoCorrErrs = ' num2str(autoCorrErrs, ' #0.3g')]);

    ## Plots

    if no_plots == 1:
        return

    # Autocorrelations
    for nNet in param.showNets:
        figNr = figNr + 1;
        figure(figNr); clf;
        hold on;
        plot(autoCorrP, 'r', 'LineWidth',2);
        plot(autoCorry{nNet}, 'b', 'LineWidth',2);
        hold off;
        title(sprintf('Autocorrs in #g (r=orig)', nNet));

    # Signals
    maxy = -10; miny = 10;
    for nNet in xrange(param.NNets):

        raws{nNet} = uCollectortest(1, ...
             - signalPlotLength + 1 : );
        maxy = max(maxy, max(raws{nNet}));
        miny = min(miny, min(raws{nNet}));
        targets{nNet} = pCollectortest(1, ...
             - signalPlotLength + 1 : );
        maxy = max(maxy, max(targets{nNet}));
        miny = min(miny, min(targets{nNet}));
        effectives{nNet} = yCollectortest{nNet}(1, ...
             - signalPlotLength + 1 : );
        maxy = max(maxy, max(effectives{nNet}));
        miny = min(miny, min(effectives{nNet}));

    for nNet in param.showNets:
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
        #set(gca, 'YLim',[miny-0.2 maxy+0.2], 'XLim', [1 signalPlotLength]); # why?
        title(sprintf('y (#g) test out vs target (r)', nNet));

    # Energy Ratios
    for nNet in param.showNets:
        figNr = figNr + 1;
        figure(figNr); clf;
        hold on;
        plot(EngyRatios{nNet});
        title(sprintf('Energy ratios (unsorted) in #g', nNet));
    #     plot(sort(EngyRatios{nNet}, 'desc'), '.');
        hold off;
    #     title(sprintf('Energy ratios in #g', nNet));

    

    # Energy error and autocorr error plot
    figNr = figNr + 1;
    figure(figNr); clf;
    hold on;
    plot(log10(energyErrs),'bx-', 'LineWidth',2);
    plot(log10(autoCorrErrs),'gx-', 'LineWidth',2);
    hold off;
    title('log10 energyErrs(b) autoCErrs(g)');
