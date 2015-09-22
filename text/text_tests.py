# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ## TODO
# - thresholding inside the network? (bad idea prob)
# - bundle the test image code
# - image results don't match up with matlab code
# - feedback

# <markdowncell>

# ### Tests
# 
# I'll do the initial tests on text here

# <codecell>

%reset

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

from skimage import io
from skimage import transform as tf
%pylab inline

# <codecell>

from generate_text_cairo import *
TD = TextDescription 

# <codecell>

from conceptor_equalization import *

# <codecell>

from conceptor_eq_mat import *

# <codecell>

# mat = io.loadmat('vars')
# exp_mat = ExpParam(NNets=5, signalPlotLength=100, noPlots=0, delta=0)

# testPattProto_mat = mat['testPattProto']
# testPatt_mat = mat['testPatt']
# trainPatt_mat = mat['trainPatt']

# res_mat = Reservoir(InSize=1)
# res_mat.initialize_system()
# res_mat.Win = mat['Win']
# res_mat.Wfb = mat['Wfb']
# res_mat.G = mat['G']
# res_mat.F = mat['F']
# res_mat.GF = mat['GF']
# res_mat.bias = mat['bias']

# res_mat.r = np.zeros((res_mat.N, 1), dtype=float)
# res_mat.z = np.zeros((res_mat.M, 1), dtype=float)
# res_mat.InSize = 1

# # exp_mat.apertures = 1000000000
# exp_mat.mismatchExp = 0

# Rref_mat, C_mat, Wout_mat = train_eq(trainPatt_mat, res=res_mat, param = exp_mat)
# test_eq(Rref_mat, C_mat, Wout_mat, testPattProto_mat, testPatt_mat, exp_mat, res_mat);

# <codecell>

from math import *

# <codecell>

trainPatt, testPatt, testPattProto = generate_data(dataType=1)
Rref, C, Wout = train_eq(trainPatt=trainPatt)
param=ExpParam(signalPlotLength=200, delta=0, noPlots=True)
test_eq(Rref, C, Wout, testPattProto, testPatt, param=param);

# <codecell>

trainPatt, testPatt, testPattProto = generate_data(dataType=1)
Rref, C, Wout = train_eq_mat(trainPatt=trainPatt)
param=ExpParam(signalPlotLength=200, delta=0, noPlots=True, windowSize=1)
test_eq_mat(Rref, C, Wout, testPattProto, testPatt, param=param);

# <codecell>

size_y = 12; text = 'bc'; reps=1400; prob = 1;

## Images
trainImage = get_matrix(TD(text, rand=1, reps=reps, prob=prob, font_family=font_list[152], italic=0, size_y=size_y, force_new=1))
imshow(trainImage[:, 1:100], cmap=cm.Greys_r)

testText = getMC(text, reps, prob=prob)
# testImage = get_matrix(TD(testText, rand=0, italic=0, font_family = font_list[146],
#                           size_y=size_y, file_name='images/'+text+'DiRand.png', force_new=1))
testImage = get_matrix(TD(testText, rand=0, italic=1, size_y=size_y, font_family=font_list[152],
                          file_name='images/'+text+'ItRand.png', force_new=1))
testImageProto = get_matrix(TD(testText, rand=0, italic=0, size_y=size_y, font_family=font_list[152],
                               file_name='images/'+text+'NoRand.png', force_new=1))

# <codecell>

param_im_mat = ExpParam(noPlots=1, signalPlotLength=200, NNets=3, delta=0, washoutLength=400, learnLength=1000,
                    COadaptLength=2000, COinitLength=1000, TychWouts=0.1)
res_im_mat = ReservoirMat(InSize=size_y, N=250, SR=1, WinScaling=0.5, BiasScaling=0.5)


Rref_im, C_im, Wout_im = train_eq_mat(trainPatt = trainImage, res=res_im_mat, param=param_im_mat)
y_im, p_im, u_im = test_eq_mat(Rref_im, C_im, Wout_im, testImageProto, testImage, param_im_mat, res_im_mat)[0:3];
# y_im, p_im, u_im = test_eq(Rref_im, C_im, Wout_im, testImage, modifiedImage, param_im, res_im)[0:3];

shift = param_im.washoutLength + param_im.COinitLength + param_im.COadaptLength;
subplot(4, 1, 1); io.imshow(p_im[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Proto')
subplot(4, 1, 2); io.imshow(y_im[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Output')
subplot(4, 1, 3); io.imshow(u_im[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Test')
subplot(4, 1, 4); io.imshow(trainImage[:, shift: shift+param_im.signalPlotLength], cmap=cm.Greys_r);
title('Train')

# <codecell>

param_im = ExpParam(noPlots=1, signalPlotLength=200, NNets=2, delta=0, washoutLength=400, learnLength=1000,
                    COadaptLength=2000, COinitLength=1000, TychWouts=0.1)
res_im = Reservoir(InSize=size_y, N=50, SR=1, WinScaling=0.5, BiasScaling=0.5)

Rref_im, C_im, Wout_im = train_eq(trainPatt = trainImage, res=res_im, param=param_im)
y_im, p_im, u_im = test_eq(Rref_im, C_im, Wout_im, testImageProto, testImage, param_im, res_im)[0:3];
# y_im, p_im, u_im = test_eq(Rref_im, C_im, Wout_im, testImage, modifiedImage, param_im, res_im)[0:3];

shift = param_im.washoutLength + param_im.COinitLength + param_im.COadaptLength;
subplot(4, 1, 1); io.imshow(p_im[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Proto')
subplot(4, 1, 2); io.imshow(y_im[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Output')
subplot(4, 1, 3); io.imshow(u_im[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Test')
subplot(4, 1, 4); io.imshow(trainImage[:, shift: shift+param_im.signalPlotLength], cmap=cm.Greys_r);
title('Train')

# <markdowncell>

# ### using rfc, the output was as follows:
# - meanabs Wout = 1.47
# - test  NRMSEs = 1.47714551344, 2.3798529696, 2.90798794955
# - energyErrs = 0.0206287031692, 0.0114282848889, 0.0142103772382
# 
# ### using matrix,
# - meanabs Wout = 1.44
# - test  NRMSEs = 1.70346378604, 2.3919606912, 2.33698647086
# - energyErrs = 0.000561479768643, 0.000190579045444, 0.000290005330996
# 
# ## By imposing MC distribution with switching prob
# - Training NRMSE: 0.984242
# 
# ### 0.5 (as before):
# - meanabs Wout = 1.69
# - test  NRMSEs = 1.41474066446, 1.4496651504, 1.56696324803
# - energyErrs = 0.0120069547554, 0.0343069686163, 0.0226487606975
# 
# ### 0.7:
# - meanabs Wout = 1.69
# - test  NRMSEs = 1.41803488005, 1.42653904864, 1.41369975742
# - energyErrs = 0.00835859211198, 0.0128014215231, 0.0143401872031
# 
# ### 0.9:
# meanabs Wout = 1.69
# test  NRMSEs = 1.41396407858, 1.47953375242, 1.44840201402
# energyErrs = 0.00947644848132, 0.0158975072272, 0.0170710003971
# 
# ### Even with alternating b's and c's it is just as bad, making the window very important.. I wonder how I missed this earlier...

# <codecell>

text="bc"
testText = getRandSeq(text, reps)
testImage = get_matrix(TD(testText, rand=0, italic=1, size_y=size_y, file_name='images/'+text+'ItRand.png', force_new=0))
testImageProto = get_matrix(TD(testText, rand=0, italic=0, size_y=size_y, file_name='images/'+text+'NoRand.png', force_new=0))

y_im, p_im, u_im = test_eq(Rref_im, C_im, Wout_im, testImageProto, testImage, param_im, res_im)[0:3];
# y_im, p_im, u_im = test_eq(Rref_im, C_im, Wout_im, testImage, modifiedImage, param_im, res_im)[0:3];

shift = param_im.washoutLength + param_im.COinitLength + param_im.COadaptLength;

subplot(4, 1, 1); io.imshow(p_im[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Proto')
subplot(4, 1, 2); io.imshow(y_im[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Output')
subplot(4, 1, 3); io.imshow(u_im[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Test')
subplot(4, 1, 4); io.imshow(trainImage[:, shift: shift+param_im.signalPlotLength], cmap=cm.Greys_r); title('Train')

# <codecell>

T = 185
p1 = p_im.copy(); p1[p1<T] = 0; p1[p1!=0] = 255
y1 = y_im.copy(); y1[y1<T] = 0; y1[y1!=0] = 255
u1 = u_im.copy(); u1[u1<T] = 0; u1[u1!=0] = 255

print np.mean(rmse(y1, p1)), np.mean(rmse(u1, y1)), np.mean(rmse(u1, p1))
subplot(3, 1, 1); io.imshow(p1[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Proto')
subplot(3, 1, 2); io.imshow(y1[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Output')
subplot(3, 1, 3); io.imshow(u1[:, :param_im.signalPlotLength], cmap=cm.Greys_r); title('Test')

# <codecell>

bc2 = display_text(TD("bc", rand=1, reps=10, italic=0, size_y=28, force_new=False))
print bc2

