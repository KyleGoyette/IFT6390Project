#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:24:38 2017

@author: kyle
"""

import numpy as np
import pandas as pd


filename = "creditcard.csv"
data = pd.DataFrame.from_csv(filename,sep=',')

data = data.values
data = np.array(data)
print data.shape[0]

np.random.shuffle(data)
n = data.shape[0]
train_split = int(0.6*n)
train_val = int(0.8*n)


train_data = data[:train_split,:]
validation_data = data[train_split:train_val,:]
test_data = data[train_val:]

print train_data.shape
print validation_data.shape
print test_data.shape

print train_data.shape[0] + validation_data.shape[0] +test_data.shape[0]
np.savetxt("cc_default_train.csv", train_data, delimiter=',')
np.savetxt("cc_default_validation.csv", validation_data, delimiter=',')
np.savetxt("cc_default_test.csv", test_data, delimiter=',')