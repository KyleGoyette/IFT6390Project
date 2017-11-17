#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:50:17 2017

@author: kyle
"""
import tensorflow as tf
import numpy as np
import pandas as ps
import NN


from tensorflow.contrib.data import Dataset, Iterator
NUMEPOCHS = 100
BATCHSIZE = 50
learning_rate = 0.00001

filename="../creditcard.csv"

def load_data(path,y_name = 'Class', train_fraction=0.6,seed=None):
    data = pd.read_csv(path, names = types.keys(), dtype=types,
                       header=0)
    data = data.values
    n = data.shape[0]
    train_index = int(n*train_fraction)
    np.random.shuffle(data)
    train_features=data[:train_index,:-1]
    train_labels = data[:train_index,-1].astype(np.int)
    test_features= data[train_index:,-2]
    test_labels=data[train_index:,-1].astype(np.int)

    return (train_features, train_labels), (test_features,test_labels)
    
def placeholder_inputs(batchsize):
    features_placeholder = tf.placeholder(tf.float32, shape=(batchsize,29))
    labels_placeholder = tf.placeholder(tf.int32,shape=(batchsize))
    return features_placeholder, labels_placeholder

def run_training():
    
    (features, labels), (_,_) = load_data(filename)
    accuracy_history = [0]*NUMEPOCHS
    
    with tf.Graph().as_default():
        features_placeholder, labels_placeholder = placeholder_inputs(BATCHSIZE)

        logits, reg_loss = NN.inference(features_placeholder)
        
        loss = NN.loss(logits,labels_placeholder,reg=0.0001,l2_loss=reg_loss)
        
        train_op = NN.training(loss,learning_rate)
        
        eval_correct = NN.evaluation(logits, labels_placeholder)
        
        summary = tf.summary.merge_all()
        
        init = tf.global_variables_initializer()
        
        saver = tf.train.Saver()
        
        sess = tf.Session()

        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
        
        sess.run(init)
        for epoch in range(1,NUMEPOCHS+1):
            num_correct_epoch = 0
            for step in range(0,features.shape[0]/BATCHSIZE):
                batch_features = features[(step*BATCHSIZE):(step+1)*BATCHSIZE,:]
                batch_labels = labels[step*BATCHSIZE:(step+1)*BATCHSIZE]
                _,total_loss,correct = sess.run([train_op,loss,eval_correct],
                                                feed_dict={features_placeholder:batch_features,
                                                labels_placeholder: batch_labels})
                num_correct_epoch += correct
                
                #if step%500==0:
                #    print total_loss,correct
            accuracy_history[epoch-1] = num_correct_epoch/float(step*BATCHSIZE)
            
            print "Epoch: " + repr(epoch)+ " Accuracy: " + repr(accuracy_history[epoch-1])
run_training()