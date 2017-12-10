#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 03:34:09 2017

@author: kyle
"""
import numpy as np
import pandas as pd

filename="../creditcard.csv"

types = collections.OrderedDict([
        ("Time", type((0))),
        ("V1",type(0.0)),
        ("V2",type(0.0)),
        ("V3",type(0.0)),
        ("V4",type(0.0)),
        ("V5",type(0.0)),
        ("V6",type(0.0)),
        ("V7",type(0.0)),
        ("V8",type(0.0)),
        ("V9",type(0.0)),
        ("V10",type(0.0)),
        ("V11",type(0.0)),
        ("V12",type(0.0)),
        ("V13",type(0.0)),
        ("V14",type(0.0)),
        ("V15",type(0.0)),
        ("V16",type(0.0)),
        ("V17",type(0.0)),
        ("V18",type(0.0)),
        ("V19",type(0.0)),
        ("V20",type(0.0)),
        ("V21",type(0.0)),
        ("V22",type(0.0)),
        ("V23",type(0.0)),
        ("V24",type(0.0)),
        ("V25",type(0.0)),
        ("V26",type(0.0)),
        ("V27",type(0.0)),
        ("V28",type(0.0)),
        ("Amount",type(0.0)),
        ("Class",type(""))
        ])

def upsample(data):
    data[:,-1] = data[:,-1].astype(np.int)
    train_data, validation_data, test_data = split_data(data)
    positive_samples_base = train_data[train_data[:,-1]==1,:]
    if positive_samples_base.shape[0]==0:
        print 'no positive samples'
        exit
    positive_samples = np.copy(positive_samples_base)
    
    negative_samples = train_data[train_data[:,-1]==0,:]
    n = len(positive_samples)
    print n
    print negative_samples.shape[0]
    
    while (positive_samples.shape[0])<negative_samples.shape[0]:
        positive_samples = np.concatenate((positive_samples,positive_samples_base),axis=0)
        n+=1
        if (n%100==0):
            print positive_samples.shape
    train_data = np.concatenate((train_data,positive_samples))
    return train_data, validation_data,test_data

def split_data(data,train=0.6,validation=0.2):
    test = 1-train-validation
    np.random.shuffle(data)
    train_split = int(train*data.shape[0])
    val_split = int((train+validation)*data.shape[0])
    train_data = data[:train_split,:]
    val_data = data[train_split:val_split,:]
    test_data = data[val_split:,:]
    return train_data,val_data,test_data




data = pd.read_csv(filename, names = types.keys(), dtype=types,
                       header=0)


data = data.values

print data.shape
train_balanced,validation,test = upsample(data)
print data_balanced.shape
formats=[]
for i in range(30):
    formats.append('%f')
formats.append('%i')
np.savetxt("../balanced_creditcard_train.csv",train_balanced,delimiter=',',fmt=formats)
np.savetxt("../balanced_creditcard_val.csv",validation,delimiter=',',fmt=formats)
np.savetxt("../balanced_creditcard_test.csv",test,delimiter=',',fmt=formats)