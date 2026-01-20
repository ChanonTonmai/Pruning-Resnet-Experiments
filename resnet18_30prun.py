#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 17:12:04 2025

@author: khongpra
"""
manual_conv_1 = [
    1062125568, # conv1
    4161798144, # layer1.0.conv1
    4161798144, # layer1.0.conv2
    4161798144, # layer1.1.conv1
    4161798144, # layer1.1.conv2
    2080899072, # layer2.0.conv1
    4161798144, # layer2.0.conv2
    57802752,   # layer2.0.downsample.0
    1040449536, # layer2.1.conv1
    1040449536, # layer2.1.conv2
    520224768,  # layer3.0.conv1
    1040449536, # layer3.0.conv2
    14450688,   # layer3.0.downsample.0
    260112384,  # layer3.1.conv1
    260112384,  # layer3.1.conv2
    169869312,  # layer4.0.conv1
    339738624,  # layer4.0.conv2
    4718592,    # layer4.0.downsample.0
    84934656,   # layer4.1.conv1
    84934656    # layer4.1.conv2
]
prunned_conv = [
    861736136,  # conv1
    3168573742, # layer1.0.conv1
    3538869756, # layer1.0.conv2
    3569478438, # layer1.1.conv1
    3547572104, # layer1.1.conv2
    1747387332, # layer2.0.conv1
    3361511650, # layer2.0.conv2
    51160704,   # layer2.0.downsample.0
    829402696,  # layer2.1.conv1
    822597572,  # layer2.1.conv2
    403610344,  # layer3.0.conv1
    762266956,  # layer3.0.conv2
    11845456,   # layer3.0.downsample.0
    174993610,  # layer3.1.conv1
    171473528,  # layer3.1.conv2
    102342618,  # layer4.0.conv1
    197988486,  # layer4.0.conv2
    3889632,    # layer4.0.downsample.0
    40614714,   # layer4.1.conv1
    34974198    # layer4.1.conv2
]

speedups = [
    0.18866830630707498, # conv1
    0.23865270914974968, # layer1.0.conv1
    0.14967770334994893, # layer1.0.conv2
    0.14232302613088965, # layer1.1.conv1
    0.14758669660265003, # layer1.1.conv2
    0.1602729053454064,  # layer2.0.conv1
    0.1922934429565645,  # layer2.0.conv2
    0.11490885416666667, # layer2.0.downsample.0
    0.20284197618211058, # layer2.1.conv1
    0.20938253751117056, # layer2.1.conv2
    0.2241616146965152,  # layer3.0.conv1
    0.26736768134807454, # layer3.0.conv2
    0.18028428819444445, # layer3.0.downsample.0
    0.3272384524375433,  # layer3.1.conv1
    0.3407713798048154,  # layer3.1.conv2
    0.3975214428371853,  # layer4.0.conv1
    0.4172329196223506,  # layer4.0.conv2
    0.17567952473958334, # layer4.0.downsample.0
    0.5218122270372179,  # layer4.1.conv1
    0.5882222917344835   # layer4.1.conv2
]


import numpy as np

print(np.mean(speedups))