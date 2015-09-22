# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ###TODO
# - Figure out a way around the temp surface
# - figure out a way to get matrix directly from surface! .
# 
# The font sizes vary according to the font style/properties, so as to occupy as much spce as possible for a given y-size of the surface."

# <codecell>

import cairo
import numpy as np
from scipy import misc
#%pylab inline
from skimage import transform as tf
from IPython.html.widgets import interact
from skimage import io
from math import *
import matplotlib.pyplot as plt
import os

# <codecell>

if os.path.isdir('images') is False:
    print 'Created dir "Images"'
    os.system('mkdir Images')
else:
    print '"Images" already exists'

# <codecell>

# Get the font names in the system
# File generated using pangocairo
f = open('fonts', 'r')
font_list =[]
for line in f:
    txt = line.strip()
    font_list = font_list + [txt]
len(font_list)

# <codecell>

def getRandSeq(t, reps):
    l = len(t)
    seq = (np.random.randint(0, l, (1, l * reps)))
    text = ''.join([t[i] for i in seq[0, :]])
    return text

# <codecell>

# reps = 1000; acount = 0.; bcount = 0.;
# text = getMC('abc', reps)
# print text.count('a') / 2.0 / reps
# # print text
# print text.count('ab') * 1.0 / text.count('a'), text.count('ba') * 1.0 / text.count('b')

# <codecell>

def getMC(text, reps, prob=0.7, skip=1):
    l = len(text)
    
    prevInd = np.random.randint(0, l)
    seq = text[prevInd]
    
    for i in xrange(1, l * reps):
        # with prob, use the successive letter, or draw randomly from others with equal probabilty
        sup = (prevInd + skip) % l
        if np.random.rand() <= prob: 
            prevInd = sup
        else:
            t = np.random.randint(0, l-1)
            if t>=sup:
                t = t+1
            prevInd = t
        
        seq = seq + text[prevInd]
    
#     string = ''.join([t[i] for i in seq[0, :]])
    return seq

# <markdowncell>

# display_text(TextDescription('ab', reps=10, force_new=1))

# <codecell>

# size_x is not used unless explicitly defined in the parameters, and the surface just fits in the x direction.
class TextDescription:
    def __init__(self, text = 'A', size_x = None, size_y = 12, x_pad = 4, y_pad =4,
                 reps = 1, font_family = font_list[152], italic = False, bold = False,
                 file_name = None, force_new = True, rand=False, verbose=False,
                 prob = 0.7, skip = 1):
                
        if rand==True:
            self.text = getMC(text, reps, prob, skip)
#             self.text = getRandSeq(text, reps)
        else:
            self.text = text * reps
#         print 'td', len(self.text), len(text)
        self.size_y = size_y
        self.x_pad = self.y_pad = x_pad
        self.font_family = font_family
        self.file_name = file_name
        self.verbose = verbose

        if italic == True:
            self.font_type=cairo.FONT_SLANT_ITALIC
        else:
            self.font_type=cairo.FONT_SLANT_NORMAL
        
        if bold == True:
            self.font_weight=cairo.FONT_WEIGHT_BOLD
        else:
            self.font_weight=cairo.FONT_WEIGHT_NORMAL
        
        if size_x is not None:
            self.size_x = size_x

            
        if file_name == None:
            itText = ['No', 'It']
            rText = ['', 'Rand'] 
                     
            file_name = 'images/' + text + itText[italic] + rText[rand] + \
                        str(font_list.index(font_family)) + '.png';
            
        self.file_name = file_name
        
        self.force_new = force_new
            
def export_text_png(arg = TextDescription()):    
    # Temp drawing to determine the apt x-size for the surface. Couldn't think of a better way!
    # The baseline depends on font properties.. But will be consistent for all chars in a font.
    surface_temp = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
    cr = cairo.Context(surface_temp)
    cr.set_font_size(arg.size_y - arg.y_pad)
    cr.select_font_face(arg.font_family, arg.font_type, arg.font_weight)
    x_bearing, y_bearing, width, height = cr.text_extents('THE QUICK BROWN FOX JUMPED OVER A YA \
                                          HIGH FENCE the quick brown fox jumped over a ya high fence')[:4]
    baseline = 0.5*arg.size_y - height/2 - y_bearing
    
    if not hasattr(arg, 'size_x'):
        x_bearing, y_bearing, width, height = cr.text_extents(arg.text)[:4]
        arg.size_x = int(ceil(width)) + arg.x_pad

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, arg.size_x, arg.size_y)
    cr = cairo.Context(surface)

    cr.set_source_rgb(1, 1, 1)
    cr.paint_with_alpha(1)

    cr.set_source_rgb(0, 0.0, 0.0)
    cr.select_font_face(arg.font_family, arg.font_type, arg.font_weight)
    cr.set_font_size(arg.size_y - arg.y_pad)
    
    x_bearing, y_bearing, width, height = cr.text_extents(arg.text)[:4]
    cr.move_to(0.5*arg.size_x - width/2 - x_bearing, baseline)
#     print cr.font_extents()
#     print cr.text_extents(arg.text)
    
    cr.show_text(arg.text)
    surface.write_to_png(arg.file_name)
    
def get_matrix(arg=TextDescription()):
    if arg.force_new == True or os.path.isfile(arg.file_name) == False:
            print "Generating new image: " + arg.file_name
            export_text_png(arg)
#     else:
#         print "Using existing image: " + arg.file_name
    
    im = io.imread(arg.file_name)[:, :, 1]
    return im

def display_text(arg):    
    im = get_matrix(arg)
    py.imshow(im, cmap=py.cm.Greys_r)
    py.axis('off')
    return im
    
# Generate A-Z and a-z
def plot_char_set(arg=TextDescription()):
    for i in xrange(26):
        py.subplot(10,6,i+1)
        arg.text=chr(i+65)
        display_text(arg)
    for i in xrange(26):
        py.subplot(10,6,31+i)
        arg.text=chr(i+97)
        display_text(arg)
        
def plot_all_fonts(text='ab', rand=False):
    py.subplots(16, 10, figsize=(28,28))
    for i, font in enumerate(font_list):
        py.subplot(16,10,i+1)
        display_text(TextDescription(text=text, font_family=font, force_new=False))

# <markdowncell>

# mat = get_matrix(TextDescription("abc", reps=15, size_y=30, font_family=font_list[1]))
# afine_tf = tf.AffineTransform(shear=0.3)
# modifiedImage = tf.warp(mat, afine_tf) * 255      # Apply affine transform to image
# io.imshow(modifiedImage, cmap=cm.Greys_r)

