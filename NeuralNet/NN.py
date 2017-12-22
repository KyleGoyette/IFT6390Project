#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:19:17 2017

@author: kyle
"""

import tensorflow as tf
network = [128,3*128,3*3*128,3*3*128,3*3*3*128,1]
def inference(dims,inputs, test = False):
    outsize= dims
    D = inputs.shape[1]
    W=[0]*len(network)
    b=[0]*len(network)
    hidden=[0]*len(network)
    for i in range(len(network)):
        with tf.variable_scope('activation'+repr(i+1)) as scope:
            if test:
                scope.reuse_variables()
            if (i==0):
                W[i] = tf.get_variable('affine'+repr(i+1),shape=[D,outsize[i]], initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
                b[i] = tf.get_variable('bias'+repr(i+1),shape=[outsize[i]], initializer = tf.constant_initializer(0.0))
                #hidden = tf.nn.relu(tf.matmul(inputs,W[i])+b[i])
                hidden = tf.matmul(inputs,W[i],name='weightmul1')
                hidden = tf.add(hidden,b[i],name='addbias1')
                
            
                
            else:
                W[i] = tf.get_variable('affine'+repr(i+1),shape=[outsize[i-1],outsize[i]], initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
                b[i] = tf.get_variable('bias'+repr(i+1),shape=[outsize[i]], initializer = tf.constant_initializer(0.0))
                hidden = tf.matmul(hidden,W[i],name="WeightMul"+repr(i+1))+b[i]

            tf.summary.histogram('weights'+repr(i+1),W[i])
            tf.summary.histogram('biases'+repr(i+1),b[i])
        
        
        
        with tf.variable_scope('hidden'+repr(i+1)) as scope:
            if test:
                    scope.reuse_variables()    
            if (i<len(network)-1):    
                hidden = tf.contrib.layers.batch_norm(hidden)
                hidden = tf.nn.relu(hidden,name="hidden" + repr(i+1)+ "relu")

    logits = hidden
    #logits = tf.sigmoid(hidden)   
        
    #reg_loss = reg_loss1+reg_loss2+reg_loss3+reg_loss4+reg_loss5+reg_loss6
    return logits#,reg_loss

def loss(logits,labels,batchsize):
    
    #weighted_logits = tf.multiply(logits,class_weights)
    #softmax_logits = tf.nn.softmax(logits,name='Softmax')
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name="CrossEntropy")
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(tf.reshape(labels,shape=(batchsize,1)),dtype=tf.float32),logits=logits,name='CrossEntropy')
    
    #scaled_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels,softmax_logits,weights=class_weights)
    loss = tf.reduce_mean(cross_entropy,name="CrossEntropy_mean")
    #loss += reg*l2_loss
    #loss += l2_loss
    tf.summary.scalar('loss',loss)
    #tf.summary.scalar('l2Loss',l2_loss)
    return loss

def training(loss, learning_rate,global_step):

    optimizer = tf.train.AdagradOptimizer(learning_rate)
    #tf.summary.scalar('learning_rate',learning_rate)
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits, labels):
    
    correct = tf.nn.in_top_k(logits,labels,1)
    return tf.reduce_sum(tf.cast(correct,tf.int32))

def metrics_eval(logits,labels):
    #sigmoid preds
    #predictions = tf.round(logits)
    
    #softmax
    
    predictions = tf.argmax(logits,1)
    zeros_like_labels = tf.zeros_like(labels)
    zeros_like_predictions = tf.zeros_like(predictions)
    ones_like_labels = tf.ones_like(labels)
    ones_like_predictions = tf.ones_like(predictions)
    
    true_positives_op = tf.reduce_sum(tf.cast(tf.logical_and(
            tf.equal(labels,ones_like_labels),
            tf.equal(predictions,ones_like_predictions)
            ), "float"))
    
    true_negatives_op = tf.reduce_sum(tf.cast(tf.logical_and(
            tf.equal(labels, zeros_like_labels),
            tf.equal(predictions,zeros_like_predictions)
            ),"float"))
    
    false_positives_op = tf.reduce_sum(tf.cast(tf.logical_and(
            tf.equal(labels,zeros_like_labels),
            tf.equal(predictions, ones_like_predictions)
            ), "float"))
    
    false_negatives_op = tf.reduce_sum(tf.cast(tf.logical_and(
            tf.equal(labels,ones_like_labels),
            tf.equal(predictions,zeros_like_predictions)
            ), "float"))
    
    return (true_positives_op,true_negatives_op,false_positives_op,false_negatives_op)
    

def metrics_eval_sigmoid(logits,labels):
    #sigmoid preds
    predictions = tf.cast(tf.round(logits),dtype=tf.int32)
    labels = tf.reshape(labels,shape=(29999,1))
    zeros_like_labels = tf.zeros_like(labels)
    zeros_like_predictions = tf.zeros_like(predictions)
    ones_like_labels = tf.ones_like(labels)
    ones_like_predictions = tf.ones_like(predictions)
    print ones_like_predictions
    print ones_like_labels
    
    true_positives_op = tf.reduce_sum(tf.cast(tf.logical_and(
            tf.equal(labels,ones_like_labels),
            tf.equal(predictions,ones_like_predictions)
            ), "float"))
    
    true_negatives_op = tf.reduce_sum(tf.cast(tf.logical_and(
            tf.equal(labels, zeros_like_labels),
            tf.equal(predictions,zeros_like_predictions)
            ),"float"))
    
    false_positives_op = tf.reduce_sum(tf.cast(tf.logical_and(
            tf.equal(labels,zeros_like_labels),
            tf.equal(predictions, ones_like_predictions)
            ), "float"))
    
    false_negatives_op = tf.reduce_sum(tf.cast(tf.logical_and(
            tf.equal(labels,ones_like_labels),
            tf.equal(predictions,zeros_like_predictions)
            ), "float"))
    
    return (true_positives_op,true_negatives_op,false_positives_op,false_negatives_op)    
