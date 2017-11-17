#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:19:17 2017

@author: kyle
"""

import tensorflow as tf

def inference(inputs, eva = False):
    outsize=[100,200,100,2]
    D = inputs.shape[1]
    
    with tf.variable_scope('hidden1') as scope:
        if eva:
            scope.reuse_variables()
        W1 = tf.get_variable('affine1',shape=[D,outsize[0]], initializer = tf.truncated_normal_initializer(stddev=0.3))
        b1 = tf.get_variable('bias1',shape=[outsize[0]], initializer = tf.constant_initializer(0.0))
        
        hidden1 = tf.nn.relu(tf.matmul(inputs,W1) +b1)
        reg_loss1 = tf.nn.l2_loss(W1)
    
    with tf.variable_scope('hidden2') as scope:
        if eva:
            scope.reuse_variables()
        W2 = tf.get_variable('affine2',shape=[outsize[0],outsize[1]], initializer = tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('bias2',shape=[outsize[1]], initializer = tf.constant_initializer(0.0))
       
        hidden2 = tf.nn.relu(tf.matmul(hidden1,W2)+b2)
        reg_loss2 = tf.nn.l2_loss(W2)
        
    with tf.variable_scope('hidden3') as scope:
        if eva:
            scope.reuse_variables()
        W3 = tf.get_variable('affine3',shape=[outsize[1],outsize[2]], initializer = tf.truncated_normal_initializer(stddev=0.1))
        b3 = tf.get_variable('bias3',shape=[outsize[2]], initializer = tf.constant_initializer(0.0))
       
        hidden3 = tf.nn.relu(tf.matmul(hidden2,W3)+b3)
        reg_loss3 = tf.nn.l2_loss(W3)
        
    with tf.variable_scope('hidden4') as scope:
        if eva:
            scope.reuse_variables()
        W4 = tf.get_variable('affine4',shape=[outsize[2],outsize[3]], initializer = tf.truncated_normal_initializer(stddev=0.1))
        b4 = tf.get_variable('bias4',shape=[outsize[3]], initializer = tf.constant_initializer(0.0))
       
        logits = tf.nn.softmax(tf.matmul(hidden3,W4)+b4)
        reg_loss4 = tf.nn.l2_loss(W4)
        
        
    reg_loss = reg_loss1+reg_loss2+reg_loss3+reg_loss4
    return logits,reg_loss

def loss(logits,labels, reg=0, l2_loss = 0):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name="CrossEntropy")
    loss = tf.reduce_mean(cross_entropy,name="CrossEntropy_mean")
    loss += l2_loss
    #('loss',loss)
    return loss

def training(loss, learning_rate):
    
    #tf.summary.scalar('loss',loss)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0,name='global_step',trainable=False)
    
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits,labels,1)
    return tf.reduce_sum(tf.cast(correct,tf.int32))



