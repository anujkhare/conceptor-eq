# Testing
def test_eq(Rref, C, WoutAll,
            trainPatt, #only for plotting!, REMOVE LATER
            testPattProto,
            testPatt,
            param = ExpParam(),
            resProto = None,):
    
    param.InSize = testPatt.shape[0]
    if param.InSize != WoutAll.shape[0]:
        raise TypeError("Shape mismatch - WoutAll: " + str(WoutAll.shape[0]) + " , InSize: " + str(param.InSize))
    ####################################################################################################
    print "Washout..."
    
    if resProto is None:
        resProto = Reservoir(InSize=param.InSize)
    
    res = [deepcopy(resProto) for i in xrange(param.NNets)]
#     res = []  # List of all reservoirs in the cascade
    
#     for i in xrange(param.NNets):
#         res.append(Reservoir(InSize=param.InSize))
#         res[i].initialize_system()
    yCollectortest = np.zeros((param.NNets, param.InSize, param.testLength))   # Test Output from each net
    pCollectortest = np.zeros((param.InSize, param.testLength));
    uCollectortest = np.zeros((param.InSize, param.testLength));

    yAll = np.zeros((param.NNets, res[0].InSize,1))
        
    for n in xrange(param.washoutLength):             # just the WASHOUT period
        res[0].update_state(testPatt[:, n], C)
        yAll[0] = np.dot(WoutAll, res[0].z)

        for nNet in xrange(1, param.NNets):                              # For subsequent cascades
            res[nNet].update_state(yAll[nNet-1], C)
            yAll[nNet] = np.dot(WoutAll, res[nNet].z)

    Rref = np.reshape(Rref, (res[0].M, 1))
    ####################################################################################################
    # Initialize the average (expected) signal energy vector E[z.^2], called
    # Ezsqr, for all nets in the cascade,
    # by driving the cascade for COinitLength steps with input signal
    print "Initialize Ezsqr..."
    
    shift = param.washoutLength
    MismatchRatios = []; zColl = []; Ezsqr = [];
    for nNet in xrange(param.NNets):
        zColl.append(np.zeros((res[nNet].M, param.COinitLength)))
#         Ezsqr.append(np.zeros((res[nNet].M, 1)))
        MismatchRatios.append(np.zeros((res[i].M, 1)))            # Test MR for each net
  
    for n in xrange(param.COinitLength):
        res[0].update_state(testPatt[:, n+shift], C)
        yAll[0] = np.dot(WoutAll, res[0].z)
        zColl[0][:, n] = res[0].z[:, 0];

        for nNet in xrange(1, param.NNets):                       # For subsequent cascades
            res[nNet].update_state(yAll[nNet-1], C)
            yAll[nNet] = np.dot(WoutAll, res[nNet].z)
            zColl[nNet][:, n] = res[0].z[:, 0];

    zColl[nNet]
    for nNet in xrange(param.NNets):
        Ezsqr.append(np.diag(np.dot(zColl[nNet], np.transpose(zColl[nNet]))) / param.COinitLength)
        Ezsqr[nNet] = np.reshape(Ezsqr[nNet], (res[nNet].M, 1))
        # the mismatch ratios will function as the "ERROR" term that is used to pull
        # the actual signal energies of the z vectors toward the reference z
        # vectors known from the "clean" training input
        MismatchRatios[nNet] = (Rref / Ezsqr[nNet]) ** param.mismatchExp;

    ####################################################################################################
    ## Adapt forward through nets for COadaptLength
    print "Adapting MismatchRatio..."
    
    shift = param.washoutLength + param.COinitLength;
    y_co_adapt = np.zeros((param.InSize, param.COadaptLength))
    
    for n in xrange(param.COadaptLength):
        res[0].update_state(testPatt[:, n+shift], C)
        # in the next two lines, the core adaptation is done, by re-shaping the
        # z vector with the help of the mismatch ratios which pull it toward
        # the reference z signal energy profile known from training
        yAll[0] = np.dot(WoutAll, MismatchRatios[0] * res[0].z)

        for nNet in xrange(1, param.NNets):                              # For subsequent cascades
            res[nNet].update_state(yAll[nNet-1], C)
            yAll[nNet] = np.dot(WoutAll, MismatchRatios[nNet] * res[nNet].z)

        # the following updates the estimate of Ezsqr and the mismatch ratio
        for nNet in xrange(param.NNets):
            Ezsqr[nNet] = (1-param.LRR) * Ezsqr[nNet] + param.LRR * res[nNet].z ** 2;
            MismatchRatios[nNet] = (Rref / Ezsqr[nNet]) ** param.mismatchExp;

        y_co_adapt[:, n] = yAll[param.NNets - 1][:, 0]
    
    if param.noPlots == False:
        plt.figure();
        xs = range(param.COadaptLength - 100, param.COadaptLength)
        plt.plot([y_co_adapt[0, i] for i in xs], 'b', linewidth=1.5);
        plt.plot([trainPatt[0, shift+i] for i in xs], 'r', linewidth= 1.5);
        plt.title('y[0] during COadapt vs trainPatt (red)');
    
    ####################################################################################################  
    ## Finally, stop adapting, stay in the last adapted configuaration
    # and collect data for plotting and error diagnostics
    print "Testing..."
    
    shift = param.washoutLength + param.COinitLength + param.COadaptLength;
    
    for n in xrange(param.testLength):
        u = testPatt[:, n+shift]
        res[0].update_state(u, C)
        yAll[0] = np.dot(WoutAll, MismatchRatios[0] * res[0].z)
        yCollectortest[0][:, n] = yAll[0][:, 0]

        for nNet in xrange(1, param.NNets):                              # For subsequent cascades
            res[nNet].update_state(yAll[nNet-1], C)
            yAll[nNet] = np.dot(WoutAll, MismatchRatios[nNet] * res[nNet].z)
            yCollectortest[nNet][:, n] = yAll[nNet][:, 0]

        pCollectortest[:,n] = testPattProto[:, n + shift]
        uCollectortest[:,n] = u  
            
    ####################################################################################################  
    # Calculate errors
    print "Calculating Errors..."
    ytestNRMSE = []; EngyRatios = [];
    
    for nNet in xrange(param.NNets):
        ytestNRMSE.append(np.mean([i ** 2 for i in yCollectortest[nNet] - pCollectortest]))
        EngyRatios.append(Rref / Ezsqr[nNet])

    rawNRMSE = nrmse(testPattProto, testPatt)

#     autoCorrP = autocorr(pCollectortest, maxLag);
#     for nNet in xrange(param.NNets):
#         autoCorry[nNet] = autocorr(yCollectortest[nNet], maxLag);

    energyErrs = np.zeros((1, param.NNets));
    for nNet in xrange(param.NNets):
        energyErrs[0, nNet] = np.linalg.norm((Rref - Ezsqr[nNet]) / np.linalg.norm(Rref)) ** 2;
    
    print '-' * 20
    print('raw NRMSE = %0.3g' %  (np.mean(rawNRMSE)));
    print('meanabs Wout = %0.3g' %  np.mean(abs(WoutAll)));
    print('test  NRMSEs = %s' % ', '.join(map(str, ytestNRMSE)))
    print('energyErrs = %s' % ', '.join(map(str, energyErrs)))
#     print('autoCorrErrs = %0.3g' % autoCorrErrs);

    ####################################################################################################  
    ## Plots

    if param.noPlots == 0:
#         # Autocorrelations
#         for nNet in param.showNets:
#             figNr = figNr + 1;
#             plt.figure(figNr); clf;
#             hold('on')
#             plot(autoCorrP, 'r', 'LineWidth',2);
#             plot(autoCorry{nNet}, 'b', 'LineWidth',2);
#             hold off;
#             title(sprintf('Autocorrs in #g (r=orig)', nNet));
#
        # Signals
        maxy = -10; miny = 10; 
        raws = []; targets = []; effectives = [];
        for nNet in xrange(param.NNets):
            raws.append(uCollectortest[0, - param.signalPlotLength + 1 : ])
            plt.maxy = max(maxy, max(raws[nNet]));
            plt.miny = min(miny, min(raws[nNet]));
            targets.append(pCollectortest[0, - param.signalPlotLength + 1 : ])
            plt.maxy = max(maxy, max(targets[nNet]));
            plt.miny = min(miny, min(targets[nNet]));
            effectives.append(yCollectortest[nNet][0, - param.signalPlotLength + 1 : ])
            plt.maxy = max(maxy, max(effectives[nNet]));
            plt.miny = min(miny, min(effectives[nNet]));

        for nNet in param.showNets:
            plt.figure()
            plt.plot(raws[nNet], color='0.75', linewidth=3)
            plt.plot(targets[nNet], 'r', linewidth=2)
            plt.plot(effectives[nNet], 'b', linewidth=2, alpha=0.75)

            #set(gca, 'YLim',[miny-0.2 maxy+0.2], 'XLim', [1 signalPlotLength]); # why?
            plt.title('y' + str(nNet) + 'test out vs target (r)');

        # Energy Ratios
        for nNet in param.showNets:
            plt.figure()
            plt.plot(EngyRatios[nNet]);
            plt.title('Energy ratios (unsorted) in ' + str(nNet));

        # Energy error and autocorr error plot
        plt.figure();
#         plt.plot(log10(energyErrs),'bx-', linewidth=2);
#         plot(log10(autoCorrErrs),'gx-', linewidth=2);
        plt.title('log10 energyErrs(b) autoCErrs(g)');
    
    return yCollectortest[param.NNets - 1], ytestNRMSE, energyErrs
