# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# %load_ext vimception

# <codecell>

import numpy as np
from IPython.core.debugger import Tracer
import matplotlib.pyplot as plt
from math import *
from copy import deepcopy

# <codecell>

class ExpParam:
    def __init__(self,
                 NNets = 5, InSize = 12, noPlots = 0,
                 washoutLength = 100, learnLength = 1000,
                 COinitLength = 1000, COadaptLength = 2000,
                 testLength = 1000, TychWouts = 0.05,
                 windowSize = 1,
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
        self.windowSize = windowSize
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

class ReservoirMat:
    def __init__(self,
                 InSize = 12,
                 N = 200,  # network size
                 Nfb = 2, # number of feedbacks
                 windowSize = 1,
                 SR = 1 ,  # spectral radius
                 WinScaling = .4 ,
                 WfbScaling = 0. ,
                 BiasScaling = 0.):

        self.InSize = InSize * windowSize
        #showNets = showNets
        self.N = N
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
        
        WRaw = np.random.randn(self.N, self.N)
        specrad = max(np.abs(np.linalg.eig(WRaw)[0]))
        WRaw = WRaw / specrad

        # Scale raw weights and initialize weights
        self.W = WRaw * self.SR
        self.Win  = self.WinScaling * WinRaw;
#         self.Wfb  = self.WfbScaling * WfbRaw;
        self.bias = self.BiasScaling * biasRaw;

        self.x = np.ones((self.N, 1), dtype=float)
        self.Wscalings = np.ones((self.N, self.N))
        
    def update_state(self, in_patt, conceptor=None):
        if in_patt.shape[0] != self.InSize:
#             print self.InSize, in_patt.shape[0]
            raise ValueError("Input size is incorrect")
    
        x = np.dot(self.W * self.Wscalings, self.x)
        self.x = np.tanh(x + np.dot(self.Win, in_patt).reshape((self.N, 1)) + self.bias)

        if conceptor is not None:
            self.x = np.reshape(conceptor,(self.N, 1)) * self.x

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
        trainPatt = rand(1, L);
        a = baselineParams[0];
        b = baselineParams[1];
        c = baselineParams[2];
        testPattProto = trainPatt;

        testPatt = trainPatt.copy();
        for n in xrange(L):
            testPatt[n] = Filter(trainPatt[n], a, b, c);
            
    return trainPatt, testPatt, testPattProto

# <codecell>

def rmse(patt1, patt2):
    err = patt1 - patt2
    return np.sqrt(np.mean(err ** 2, axis=1))

# <codecell>

def nrmse(patt1, patt2):
    err = patt1 - patt2
    combinedVar = 0.5 * (np.var(patt1, axis=1, ddof=1) + np.var(patt2, axis=1, ddof=1))
    return np.sqrt(np.mean(err ** 2, axis=1) / combinedVar)

# <codecell>

len(a.shape)

# <codecell>

def autocorr(patt, maxLag):
    '''
    Returns autocorrs for a scalar series on lags 1, 2 ... maxLag
    input must be a numpy array in either column or row format
    
    patt: Timeseries should be in column form
    '''
    
    L = patt.shape[0]
#     print patt.shape, L

    if (len(patt.shape) > 1):
#         print "hey"
        rval = np.zeros((patt.shape[1], maxLag+1))
        for i in xrange(patt.shape[1]):
            rval[i, :] = autocorr(patt[:, i], maxLag)
        return rval

    if L <= maxLag:
        raise ValueError('timeseries too short for computing requested autocorrs')
    
    dataLagMat = np.zeros((L-maxLag, maxLag+1))
    dataLagMat[:, maxLag] = patt[maxLag:]
    
    for lag in xrange(0, maxLag):
        dataLagMat[:,lag] = patt[lag:-maxLag+lag];
#     print dataLagMat

    dataMat = np.transpose(np.tile(patt[:-maxLag], (maxLag+1, 1)))
#     print dataMat
#     dataMat = repmat(tsCol(1:L-maxLag,1),1, maxLag+1);
    
    autocorrPlotData = np.diag(np.dot(np.transpose(dataLagMat), dataMat)) / (L-maxLag);
    return autocorrPlotData

# <codecell>

def getPattWindow(patt = np.random.randint(0, 10, size=(3, 10)), windowSize = 2):
    if windowSize < 1:
        raise ValueError("window size can't be less than 1")
    if windowSize == 1:
        return patt

    l = patt[:, 0:-(windowSize - 1)]
    for i in xrange(1, windowSize):
        endInd = windowSize - 1 - i
        if endInd == 0:
            l = np.append(l, patt[:, i:], axis=0)
        else:
            l = np.append(l, patt[:, i:endInd], axis=0)
#     print l.shape
#     print patt
#     print l
    return l

# <codecell>

def train_eq_mat(trainPatt, res = None, param = ExpParam()):
    
    trainPatt = getPattWindow(trainPatt, param.windowSize)
    
    param.InSize = trainPatt.shape[0]
    if res is None:
        print "Generating new reservoir"
        res = ReservoirMat(InSize=trainPatt.shape[0])
    ####################################################################################################
    # 2-module modeling - Compute Conceptor
    print "Learning Conceptor..."
    
    xCollector = np.zeros((res.N, param.learnLength))

    for n in xrange(param.washoutLength + param.learnLength):
        res.update_state(trainPatt[:, n]) 

        if n >= param.washoutLength:
            xCollector[:, n - param.washoutLength] = res.x[:, 0];
    
    R = np.diag(np.dot(xCollector, np.transpose(xCollector))) / param.learnLength;
    
    if param.apertures == np.Inf:
        C = np.ones(R.shape, dtype=float)
    else:
        C = R / (R + param.apertures**(-2));
#         U, E, V = np.linalg.svd(R);    # TRY THIS WAY!
#         S = np.dot(E, np.linalg.pinv(E + param.apertures ** (-2) * np.eye(E.shape[0])))
#         C = np.dot(np.dot(U, S), np.transpose(U))
    ####################################################################################################
    ## Learn Rref,  Wout, and collect r state plot data
    print "Learning Rref, Wout..."
    
    xCollector = np.zeros((res.N, param.learnLength))
    pCollector = np.zeros((res.InSize, param.learnLength))
    predCollector = np.zeros((res.InSize, param.learnLength))
    
    # Here we are doing the same thing, but restricting the dynamics using learnt conceptor
    for n in xrange(param.delta, param.washoutLength + param.learnLength):
        res.update_state(trainPatt[:, n], conceptor=C)
        
        if n >= param.washoutLength:
            xCollector[:, n - param.washoutLength] = res.x[:, 0]
            pCollector[:, n - param.washoutLength] = trainPatt[:, n - param.delta]
            predCollector[:, n - param.washoutLength] = trainPatt[:, n+1 - param.delta]


    # Reference R - represents E[z^2] for the ideal case
    Rref = np.dot(xCollector, np.transpose(xCollector)) / param.learnLength;
    args = xCollector
    targs = pCollector
    # Note that WoutAll learns to predict y(n-delta) not y(n)
    WoutAll = np.transpose(np.dot(np.linalg.pinv(np.dot(args, np.transpose(args)) / param.learnLength + \
                                                 param.TychWouts[0] * np.eye(res.N)
                                                 ),
                                  np.dot(args, np.transpose(targs)) / param.learnLength
                                  )
                           )

    ytrainNRMSE = nrmse(np.dot(WoutAll, args), targs)
    print('Training NRMSE: %f' % np.mean(ytrainNRMSE))
    
    return Rref, C, WoutAll#, ytrainNRMSE

# <codecell>

def cumMean(ts1, ts2, t=0, l0=0):
    l = len(ts1)
    if len(ts2) != l:
        raise ValueError("Dimensions don't match!")

    rval = np.zeros((l, 1))
    for i in xrange(l):
        t = t + ts1[i] * ts2[i]
        rval[i] = t / (l0 + i + 1.0)
    return rval

# <codecell>

a = np.random.randint(1, 10, (5, 5))
print a
a[a<3]=a[a<3]-1
print a

# <codecell>

# Testing
def test_eq_mat(Rref, C, WoutAll,
            testPattProto,
            testPatt,
            param = ExpParam(),
            resProto = None,):
    
    testPatt = getPattWindow(testPatt, param.windowSize)
    testPattProto = getPattWindow(testPattProto, param.windowSize)
    
    param.InSize = testPatt.shape[0]
    if param.InSize != WoutAll.shape[0] * param.windowSize:
        raise TypeError("Shape mismatch - WoutAll: " + str(WoutAll.shape[0]) + " , InSize: " + str(param.InSize))
    ####################################################################################################
    print "Washout..."
    
    if resProto is None:
        resProto = ReservoirMat(InSize=param.InSize)
    
    res = [deepcopy(resProto) for i in xrange(param.NNets)]     # create NNets reservoirs using prototype

    yCollectortest = np.zeros((param.NNets, param.InSize, param.testLength))   # Test Output from each net
    pCollectortest = np.zeros((param.InSize, param.testLength));
    uCollectortest = np.zeros((param.InSize, param.testLength));
    yAll = np.zeros((param.NNets, res[0].InSize,1))

    for n in xrange(param.washoutLength):             # just the WASHOUT period
        res[0].update_state(testPatt[:, n], C)
        yAll[0] = np.dot(WoutAll, res[0].x)

        for nNet in xrange(1, param.NNets):                              # For subsequent cascades
            res[nNet].update_state(yAll[nNet-1], C)
            yAll[nNet] = np.dot(WoutAll, res[nNet].x)

    Rcopy = np.copy(Rref)
    Rref = np.reshape(np.diag(Rref), (res[0].N, 1))
    ####################################################################################################
    # Initialize the average (expected) signal energy vector E[z.^2], called
    # Ezsqr, for all nets in the cascade,
    # by driving the cascade for COinitLength steps with input signal
    print "Initialize Ezsqr..."
    
    shift = param.washoutLength
    MismatchRatios = []; zColl = []; Ezsqr = []; R = []; sig= [];
    for nNet in xrange(param.NNets):
#         zColl.append(np.zeros((res[nNet].N, param.COinitLength)))
        zColl.append(np.zeros((res[nNet].N, param.COinitLength + param.COadaptLength + \
                               param.testLength)))
#         Ezsqr.append(np.zeros((res[nNet].N, 1)))
        MismatchRatios.append(np.zeros((res[i].N, 1)))            # Test MR for each net
  
    for n in xrange(param.COinitLength):
        res[0].update_state(testPatt[:, n+shift], C)
        yAll[0] = np.dot(WoutAll, res[0].x)
        zColl[0][:, n] = res[0].x[:, 0];

        for nNet in xrange(1, param.NNets):                       # For subsequent cascades
            res[nNet].update_state(yAll[nNet-1], C)
            yAll[nNet] = np.dot(WoutAll, res[nNet].x)
            zColl[nNet][:, n] = res[nNet].x[:, 0]
        
    for nNet in xrange(param.NNets):
        R.append(np.dot(zColl[nNet], np.transpose(zColl[nNet])) / param.COinitLength)
        Ezsqr.append(np.diag(R[nNet]))
        Ezsqr[nNet] = np.reshape(Ezsqr[nNet], (res[nNet].N, 1))
        # the mismatch ratios will function as the "ERROR" term that is used to pull
        # the actual signal energies of the z vectors toward the reference z
        # vectors known from the "clean" training input
        MismatchRatios[nNet] = (Rref / Ezsqr[nNet]) ** param.mismatchExp;
        tmp = np.ones((resProto.N, resProto.N))
        tmp[np.sign(R[nNet]) != np.sign(Rcopy)] = -1
        res[nNet].Wscalings = res[nNet].Wscalings * tmp
        sig.append(tmp);

#         Rdiff = Rcopy - R[nNet]
#         res[nNet].Wscalings[Rdiff < 0] = 0
#         for i in xrange(res[nNet].N):
#             res[nNet].Wscalings[i, i] = 1 
    
    
    pairs = [(30, 50), (10, 80), (15, 40)]
    figure();
    for k, (i, j) in enumerate(pairs):
        subplot(2, 3, k + 1)
        l0 = param.COinitLength - 200
        t = np.dot(zColl[0][i, :l0], np.transpose(zColl[0][j, :l0]))
        corr = cumMean(zColl[0][i, l0:l0+200], zColl[0][j ,l0:l0+200], t, l0)
        plot(range(200), corr, color='blue')
        plot(range(200), np.tile(Rcopy[i, j], (200, 1)), color='red')
        
        subplot(2, 3, k+4)
        plot(zColl[0][i, l0:l0+200])
        
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
        yAll[0] = np.dot(WoutAll, MismatchRatios[0] * res[0].x)
        zColl[0][:, n + shift - param.washoutLength] = res[0].x[:, 0]

        for nNet in xrange(1, param.NNets):                              # For subsequent cascades
            res[nNet].update_state(yAll[nNet-1], C)
            yAll[nNet] = np.dot(WoutAll, MismatchRatios[nNet] * res[nNet].x)
            zColl[nNet][:, n + shift - param.washoutLength] = res[nNet].x[:, 0]
            
        # the following updates the estimate of Ezsqr and the mismatch ratio
        for nNet in xrange(param.NNets):
            Ezsqr[nNet] = (1-param.LRR) * Ezsqr[nNet] + param.LRR * res[nNet].x ** 2;
            MismatchRatios[nNet] = (Rref / Ezsqr[nNet]) ** param.mismatchExp;
            
            R[nNet] = np.dot(zColl[nNet], np.transpose(zColl[nNet])) / (param.COinitLength + n)
#             R[nNet] = R[nNet] * (1-param.LRR) + param.LRR * np.dot(res[nNet].x, np.transpose(res[nNet].x))
                 
            Rdiff = np.abs(Rcopy) - np.abs(R[nNet])
            res[nNet].Wscalings[Rdiff <= 0] = 0
            res[nNet].Wscalings[Rdiff > 0] = sig[nNet][Rdiff>0] * 2.0
#             for i in xrange(res[nNet].N):
#                 res[nNet].Wscalings[i, i] = 1 
 
    figure()
    for k, (i, j) in enumerate(pairs):
        subplot(2, 3, k+1)
        l0 = param.COinitLength + param.COadaptLength - 201
        t = np.dot(zColl[0][i, :l0], np.transpose(zColl[0][j, :l0]))
        corr = cumMean(zColl[0][i, l0:l0+200], zColl[0][j ,l0:l0+200], t, l0)
        plot(range(200), corr, color='blue')
        plot(range(200), np.tile(Rcopy[i, j], (200, 1)), color='red')
        
        subplot(2, 3, k+4)
        plot(zColl[0][i, l0:l0+200])

    ####################################################################################################  
    ## Finally, stop adapting, stay in the last adapted configuaration
    # and collect data for plotting and error diagnostics
    print "Testing..."
    
    shift = param.washoutLength + param.COinitLength + param.COadaptLength;
    
    for n in xrange(param.testLength):
        u = testPatt[:, n + shift]
        res[0].update_state(u, C)
        yAll[0] = np.dot(WoutAll, MismatchRatios[0] * res[0].x)
        yCollectortest[0][:, n] = yAll[0][:, 0]
        zColl[0][:, n + shift - param.washoutLength] = res[0].x[:, 0]
        
        for nNet in xrange(1, param.NNets):                              # For subsequent cascades
            res[nNet].update_state(yAll[nNet-1], C)
            yAll[nNet] = np.dot(WoutAll, MismatchRatios[nNet] * res[nNet].x)
            yCollectortest[nNet][:, n] = yAll[nNet][:, 0]
            zColl[nNet][:, n + shift - param.washoutLength] = res[nNet].x[:, 0]
        
        pCollectortest[:,n] = testPattProto[:, n + shift - param.NNets * param.delta]
        uCollectortest[:,n] = testPatt[:, n + shift - param.NNets * param.delta]

    ####################################################################################################  
    # Calculate errors
    print "Calculating Errors..."
    ytestNRMSE = []; EngyRatios = []; energyErrs = []; autoCorrErrs = [];
    
    rawNRMSE = nrmse(testPattProto, testPatt)
    autoCorrP = autocorr(np.transpose(pCollectortest), param.maxLag)

    for nNet in xrange(param.NNets):
        ytestNRMSE.append(nrmse(yCollectortest[nNet], pCollectortest)[0])
        EngyRatios.append(Rref / Ezsqr[nNet])
        autoCorry = (autocorr(np.transpose(yCollectortest[nNet]), param.maxLag))
        autoCorrErrs.append((np.linalg.norm(autoCorrP - autoCorry) / np.linalg.norm(autoCorrP)) ** 2)
        energyErrs.append(np.linalg.norm((Rref - Ezsqr[nNet]) / np.linalg.norm(Rref)) ** 2)
    
    print '-' * 20
    print('raw NRMSE = %0.3g' %  (np.mean(rawNRMSE)));
    print('meanabs Wout = %0.3g' %  np.mean(abs(WoutAll)));
    print('test  NRMSEs = %s' % ', '.join(map(str, ytestNRMSE)))
    print('energyErrs = %s' % ', '.join(map(str, energyErrs)))
    print('autoCorrErrs = %s' % ', '.join(map(str, autoCorrErrs)));

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

# <codecell>

exp = ExpParam(noPlots=0, COadaptLength=500)
train1, test1, testProto1 = generate_data()
r2, c2, w2 = train_eq_mat(train1, param=exp)
test_eq_mat(r2, c2, w2, testProto1, test1, param=exp);

