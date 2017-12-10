#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:19:17 2017

@author: kyle
"""

import tensorflow as tf

def inference(inputs, eva = False):
    outsize=[200,300,200,2]
    D = inputs.shape[1]
    
    with tf.variable_scope('hidden1') as scope:
        if eva:
            scope.reuse_variables()
        W1 = tf.get_variable('affine1',shape=[D,outsize[0]], initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        b1 = tf.get_variable('bias1',shape=[outsize[0]], initializer = tf.constant_initializer(0.0))
        
        hidden1 = tf.nn.relu(tf.matmul(inputs,W1) +b1)
        reg_loss1 = tf.nn.l2_loss(W1)
    
    with tf.variable_scope('hidden2') as scope:
        if eva:
            scope.reuse_variables()
        W2 = tf.get_variable('affine2',shape=[outsize[0],outsize[1]], initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        b2 = tf.get_variable('bias2',shape=[outsize[1]], initializer = tf.constant_initializer(0.0))
       
        hidden2 = tf.nn.relu(tf.matmul(hidden1,W2)+b2)
        reg_loss2 = tf.nn.l2_loss(W2)
        
    with tf.variable_scope('hidden3') as scope:
        if eva:
            scope.reuse_variables()
        W3 = tf.get_variable('affine3',shape=[outsize[1],outsize[2]], initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        b3 = tf.get_variable('bias3',shape=[outsize[2]], initializer = tf.constant_initializer(0.0))
       
        hidden3 = tf.nn.relu(tf.matmul(hidden2,W3)+b3)
        reg_loss3 = tf.nn.l2_loss(W3)
        
    with tf.variable_scope('hidden4') as scope:
        if eva:
            scope.reuse_variables()
        W4 = tf.get_variable('affine4',shape=[outsize[2],outsize[3]], initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        b4 = tf.get_variable('bias4',shape=[outsize[3]], initializer = tf.constant_initializer(0.0))
       
        logits = (tf.matmul(hidden3,W4)+b4)
        reg_loss4 = tf.nn.l2_loss(W4)
        
        
    reg_loss = reg_loss1+reg_loss2+reg_loss3+reg_loss4
    return logits,reg_loss

def loss(logits,labels, reg=0, l2_loss = 0):
    labels = tf.to_int64(labels)
    #weighted_logits = tf.multiply(logits,class_weights)
    #softmax_logits = tf.nn.softmax(logits,name='Softmax')
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name="CrossEntropy")

    #scaled_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels,softmax_logits,weights=class_weights)
    loss = tf.reduce_mean(cross_entropy,name="CrossEntropy_mean")
    loss += l2_loss
    #loss += l2_loss
    tf.summary.scalar('loss',loss)
    #tf.summary.scalar('l2Loss',l2_loss)
    return loss

def training(loss, learning_rate,global_step):

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #tf.summary.scalar('learning_rate',learning_rate)
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits, labels):
    
    correct = tf.nn.in_top_k(logits,labels,1)
    return tf.reduce_sum(tf.cast(correct,tf.int32))

def metrics_eval(logits,labels):
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
    
    
