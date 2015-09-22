# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# This code is almost completely adapted from the matlab code.

# <markdowncell>

# %load_ext vimception

# <codecell>

import numpy as np
from IPython.core.debugger import Tracer
import matplotlib.pyplot as plt
from copy import deepcopy
from math import *

# <codecell>

def generate_data(dataType = 1, p = ExpParam()):
    # filter function.
    # 1: NARMA data with NARMA coefficient filtering
    # 2: rand data with shift, scale, exponentiation

    L = p.washoutLength + p.COinitLength + p.COadaptLength + p.learnLength;
    trainPatt = []; testPatt = []; testPattProto = [];
    
    if dataType == 0:
        Filter = lambda y, ym1, ym2, a, b, c, d: np.tanh( a * y + b *\
                            (ym1) + c * (ym2) * (ym1) + d *(np.random.randn(1)))
        filterWashout = 100;
        
        baselineParams = [2, -1, -2, 0];
        pattScaling = 1; pattShift = 1;
    
    if dataType == 1:
        Filter = lambda y, ym1, ym2, a, b, c, d: np.tanh( a * y + b *\
                                    (ym1) + c * (ym2) * (ym1) + d *(np.random.randn(1)))
        filterWashout = 100;
        
        baselineParams = [2, -1, -2, 0];
        pattScaling = 1; pattShift = 1;
        
    elif dataType == 2:
        Filter = lambda y, a, b, c: a + b * (sign(y) * abs(y) ** c);
        filterWashout = 100;
        baselineParams = [-2, .5, 1];

    if dataType == 0:
        trainPatt = np.array([0.5 * (sin(2 * pi * i / 5)) for i in range(L)])
    elif dataType == 1:
        trainPatt = np.array([0.5 * (sin(2 * pi * i / 8) + sin(2 * pi * i / 5.03)) for i in range(L)])
    
    
    if dataType == 0 or dataType == 1:
        a = baselineParams[0];
        b = baselineParams[1];
        c = baselineParams[2];
        d = baselineParams[3];
        testPatt = trainPatt.copy();
        for n in xrange(2, L):
            testPatt[n] = Filter(testPatt[n],testPatt[n-1],testPatt[n-2],
                                  a, b, c, d);
        
        testPatt = pattScaling * testPatt + pattShift
    
        trainPatt = np.reshape(trainPatt, (1, L))
        testPatt = np.reshape(testPatt, (1, L))
        testPattProto = trainPatt;    
        
    elif dataType == 2:
        trainPatt = np.random.rand(1, L);
        a = baselineParams[0];
        b = baselineParams[1];
        c = baselineParams[2];
        testPattProto = trainPatt;

        testPatt = trainPatt.copy();
        for n in xrange(L):
            testPatt[n] = Filter(trainPatt[n], a, b, c);
            
    return trainPatt, testPatt, testPattProto

# <codecell>

class ExpParam:
    def __init__(self,
                 NNets = 5, InSize = 12, noPlots = 0,
                 washoutLength = 100, learnLength = 1000,
                 COinitLength = 1000, COadaptLength = 2000,
                 testLength = 1000, TychWouts = 0.05,
                 LRR = 0.005, # leaking rate for R estimation
                 delta = 0,   # timesteps delayed predictions
                 apertures = np.Inf, # Inf if no conceptors are to be inserted
                 showNets = [], # which nets are to be diagnostic-plotted
                 mismatchExp = 1,
                 signalPlotLength = 40, plotN = 8, maxLag = 49):
        # Experiment control params
        self.NNets = NNets
        self.noPlots = noPlots
        self.InSize = InSize
        self.TychWouts = TychWouts, # regularizers for Wout
        self.COinitLength = COinitLength
        self.COadaptLength = COadaptLength
        self.testLength = testLength
        self.LRR = LRR
        self.delta = delta
        self.apertures = apertures
        self.mismatchExp = mismatchExp # a value of 1/2 would be mathematically indicated
                                  # larger, over-compensating values work better

        self.washoutLength = washoutLength
        self.learnLength = learnLength
        
        # plotting specs
        self.showNets = [0, 2, NNets-1]
        self.signalPlotLength = signalPlotLength
        self.plotN = plotN
        self.maxLag = maxLag

# <codecell>

class Reservoir:
    def __init__(self,
                 InSize = 12,
                 N = 50,  # network size
                 M = 200,  # RF space size
                 Nfb = 2, # number of feedbacks
                 SR = 1 ,  # spectral radius
                 WinScaling = .4 ,
                 WfbScaling = 0. ,
                 BiasScaling = 0.):
        self.InSize = InSize
        #showNets = showNets
        self.N = N
        self.M = M
        self.Nfb = Nfb
        self.SR = SR
        self.WinScaling = WinScaling
        self.WfbScaling = WfbScaling
        self.BiasScaling = BiasScaling
        
        self.initialize_system()

    def initialize_system(self):
        np.random.seed(100) # ALL RES created are THE SAME
        # Create raw weights
        WinRaw = np.random.randn(self.N, self.InSize);
        #WfbRaw = np.random.randn(self.N, self.Nfb);
        biasRaw = np.random.randn(self.N, 1);
        FRaw = np.random.randn(self.M,self.N);
        FRawRowNorms = np.sqrt(np.sum(FRaw ** 2, 1));
        FRaw = np.dot(np.diag(1 / FRawRowNorms), FRaw)
        GstarRaw = np.random.randn(self.N,self.M);
        GF = np.dot(GstarRaw, FRaw);
        specrad = max(np.abs(np.linalg.eig(GF)[0]));
        FstarRaw = FRaw;
        GstarRaw = GstarRaw / specrad;

        # Scale raw weights and initialize weights
        self.F    = FstarRaw;
        self.G    = GstarRaw * self.SR;
        self.Win  = self.WinScaling * WinRaw;
#         self.Wfb  = self.WfbScaling * WfbRaw;
        self.bias = self.BiasScaling * biasRaw;

        self.r = np.zeros((self.N, 1), dtype=float)
        self.z = np.zeros((self.M, 1), dtype=float)
#         self.y = np.zeros((self.InSize, 1), dtype=float)
        
    def update_state(self, in_patt, conceptor=None):
        if in_patt.shape[0] != self.InSize:
#             print self.InSize, in_patt.shape[0]
            raise ValueError("Input size is incorrect")

#         print np.dot(self.Win, in_patt).reshape((50,1))
        self.r = np.tanh(np.dot(self.G, self.z) + np.dot(self.Win, in_patt).reshape((self.N, 1)) + self.bias)
        self.z = np.dot(self.F, self.r)
        if conceptor is not None:
            self.z = np.reshape(conceptor,(self.M, 1)) * self.z

# <codecell>

def rmse(patt1, patt2):
    err = patt1 - patt2
    return np.sqrt(np.mean(err ** 2, axis=1))

# <codecell>

def nrmse(patt1, patt2):
    err = patt1 - patt2
    combinedVar = 0.5 * (np.var(patt1, axis=1, ddof=1) + np.var(patt2, axis=1, ddof=1))
    return np.sqrt(np.mean(err ** 2, axis=1) / combinedVar)

# <markdowncell>

# def autocorr(patt, maxLag):
#     # Returns autocorrs for a scalar series on lags 1, 2 ... maxLag
#     # input must be a numpy array in either column or row format
#     
#     if len(patt) <= maxLag
#         raise ValueError('timeseries too short for computing requested autocorrs')
# 
#     # get the timeseries in col form
#     if patt.shape[0] == 1:
#         patt = np.transpose(patt)
#         
#     L = patt.shape[0]
#     
#     dataLagMat = np.zeros((L-maxLag, maxLag+1))
#     dataLagMat[:,0] = patt[0:L-maxLag, 0]
#     tsCol(1:L-maxLag,1)
#     
#     for lag = 1:maxLag
#         dataLagMat(:,lag+1) = tsCol(1+lag:L-maxLag+lag, 1);
#     
#     dataMat = repmat(tsCol(1:L-maxLag,1),1, maxLag+1);
#     
#     autocorrPlotData = diag(dataLagMat' * dataMat) / (L-maxLag);

# <codecell>

def train_eq(trainPatt, res = None, param = ExpParam()):
    
    param.InSize = trainPatt.shape[0]
    if res is None:
        print "Generating new reservoir"
        res = Reservoir(InSize=trainPatt.shape[0])
    ####################################################################################################
    # 2-module modeling - Compute Conceptor
    print "Learning Conceptor..."
    
    zCollector = np.zeros((res.M, param.learnLength))

    for n in xrange(param.washoutLength + param.learnLength):
        res.update_state(trainPatt[:, n]) 

        if n >= param.washoutLength:
            zCollector[:, n - param.washoutLength] = res.z[:, 0];
    
    R = np.diag(np.dot(zCollector, np.transpose(zCollector))) / param.learnLength;
    
    if param.apertures == np.Inf:
        C = np.ones(R.shape, dtype=float)
    else:
        C = R / (R + param.apertures**(-2));
    
    ####################################################################################################
    ## Learn Rref,  Wout, and collect r state plot data
    print "Learning Rref, Wout..."
    
    zCollector = np.zeros((res.M, param.learnLength))
    pCollector = np.zeros((res.InSize, param.learnLength))
    predCollector = np.zeros((res.InSize, param.learnLength))
    
    # Here we are doing the same thing, but restricting the dynamics using learnt conceptor
    for n in xrange(param.delta, param.washoutLength + param.learnLength):
        res.update_state(trainPatt[:, n], conceptor=C)
        
        if n >= param.washoutLength:
            zCollector[:, n - param.washoutLength] = res.z[:, 0]
            pCollector[:, n - param.washoutLength] = trainPatt[:, n - param.delta]
            predCollector[:, n - param.washoutLength] = trainPatt[:, n+1 - param.delta]


    # Reference R - represents E[z^2] for the ideal case
    Rref = np.diag(np.dot(zCollector, np.transpose(zCollector))) / param.learnLength;
    args = zCollector
    targs = pCollector
    # Note that WoutAll learns to predict y(n-delta) not y(n)
    WoutAll = np.transpose(np.dot(np.linalg.pinv(np.dot(args, np.transpose(args)) / param.learnLength + \
                                                 param.TychWouts[0] * np.eye(res.M)
                                                 ),
                                  np.dot(args, np.transpose(targs)) / param.learnLength
                                  )
                           )

    ytrainNRMSE = nrmse(np.dot(WoutAll, args), targs)
    print('Training NRMSE: %f' % np.mean(ytrainNRMSE))
    
    return Rref, C, WoutAll#, ytrainNRMSE

# <codecell>

# Testing
def test_eq(Rref, C, WoutAll,
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
    
    res = [deepcopy(resProto) for i in xrange(param.NNets)]     # create NNets reservoirs using prototype

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
            zColl[nNet][:, n] = res[nNet].z[:, 0];
        
    for nNet in xrange(param.NNets):
        Ezsqr.append(np.diag(np.dot(zColl[nNet], np.transpose(zColl[nNet]))) / param.COinitLength)
        Ezsqr[nNet] = np.reshape(Ezsqr[nNet], (res[nNet].M, 1))
        # the mismatch ratios will function as the "ERROR" term that is used to pull
        # the actual signal energies of the z vectors toward the reference z
        # vectors known from the "clean" training input
        MismatchRatios[nNet] = (Rref / Ezsqr[nNet]) ** param.mismatchExp;

#     print ', '.join(map(str, [np.mean(MismatchRatios[i]) for i in range(param.NNets)]))
    ####################################################################################################
    ## Adapt forward through nets for COadaptLength
    print "Adapting MismatchRatio..."
    
    shift = param.washoutLength + param.COinitLength;
#     y_co_adapt = np.zeros((param.InSize, param.COadaptLength))

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

#         y_co_adapt[:, n] = yAll[param.NNets - 1][:, 0]
        
#     if param.noPlots == False:
#         plt.figure();
#         xs = range(param.COadaptLength - 100, param.COadaptLength)
#         plt.plot([y_co_adapt[0, i] for i in xs], 'b', linewidth=1.5);
#         plt.plot([trainPatt[0, shift+i] for i in xs], 'r', linewidth= 1.5);
#         plt.title('y[0] during COadapt vs trainPatt (red)');
    
    ####################################################################################################  
    ## Finally, stop adapting, stay in the last adapted configuaration
    # and collect data for plotting and error diagnostics
    print "Testing..."
    
    shift = param.washoutLength + param.COinitLength + param.COadaptLength;
    
    for n in xrange(param.testLength):
        u = testPatt[:, n + shift]
        res[0].update_state(u, C)
        yAll[0] = np.dot(WoutAll, MismatchRatios[0] * res[0].z)
        yCollectortest[0][:, n] = yAll[0][:, 0]

        for nNet in xrange(1, param.NNets):                              # For subsequent cascades
            res[nNet].update_state(yAll[nNet-1], C)
            yAll[nNet] = np.dot(WoutAll, MismatchRatios[nNet] * res[nNet].z)
            yCollectortest[nNet][:, n] = yAll[nNet][:, 0]
        
        pCollectortest[:,n] = testPattProto[:, n + shift - param.NNets * param.delta]
        uCollectortest[:,n] = testPatt[:, n + shift - param.NNets * param.delta]

    ####################################################################################################  
    # Calculate errors
    print "Calculating Errors..."
    ytestNRMSE = []; EngyRatios = []; energyErrs = []; autoCorry = [];
    
    rawNRMSE = nrmse(testPattProto, testPatt)
#     autoCorrP = autocorr(pCollectortest, maxLag)

    for nNet in xrange(param.NNets):
        ytestNRMSE.append(nrmse(yCollectortest[nNet], pCollectortest)[0])
        EngyRatios.append(Rref / Ezsqr[nNet])
#         autoCorry.append(autocorr(yCollectortest[nNet], maxLag))
        energyErrs.append(np.linalg.norm((Rref - Ezsqr[nNet]) / np.linalg.norm(Rref)) ** 2)
    
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
        plt.plot(np.log10(energyErrs),'bx-', linewidth=2);
#         plot(np.log10(autoCorrErrs),'gx-', linewidth=2);
        plt.title('log10 energyErrs(b) autoCErrs(g)');
    
    return yCollectortest[param.NNets - 1], pCollectortest, \
           uCollectortest, ytestNRMSE, energyErrs

# <markdowncell>

# %pylab inline
# exp = ExpParam()
# train1, test1, testProto1 = generate_data()
# r2, c2, w2 = train_eq(train1, param=exp)
# test_eq(r2, c2, w2, testProto1, test1, param=exp);

