#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:50:17 2017

@author: kyle
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import collections
import NN

NUMEPOCHS = 1000
BATCHSIZE = 500
initial_learning_rate = 0.00001

filename="../balanced_creditcard_train.csv"

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

def load_data(paths,y_name = 'Class',seed=None):
    train_path = paths[0]
    if len(paths)>1:
        val_path = paths[1]
    if len(paths)>2:
        test_path = paths[2]
    
    train_data = pd.read_csv(train_path, names = types.keys(), dtype=types,
                       header=0)
    train_data = train_data.values
    
    n = train_data.shape[0]
    
    train_features=train_data[:,:-1]
    
    train_labels = train_data[:,-1].astype(np.int)



    return (train_features, train_labels)


    


def placeholder_inputs(batchsize):
    features_placeholder = tf.placeholder(tf.float32, shape=(batchsize,30))
    labels_placeholder = tf.placeholder(tf.int32,shape=(batchsize))
    return features_placeholder, labels_placeholder

def calc_ratio(labels):
    num_1_examples = np.sum(labels)
    return num_1_examples/float(labels.shape[0])

def run_training():
    
    (features, labels)= load_data([filename])
    n = features.shape[0]/BATCHSIZE
    ratio = calc_ratio(labels)
    print ratio
    
    accuracy_history = [0]*NUMEPOCHS
    
    with tf.Graph().as_default():
        
        global_step = tf.Variable(0,trainable=False)
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                           NUMEPOCHS*n, 0.99, staircase=True)
        features_placeholder, labels_placeholder = placeholder_inputs(BATCHSIZE)
        
        class_weight = tf.constant([ratio, 1.0-ratio])

        logits, reg_loss = NN.inference(features_placeholder)
        #onehotlabs = tf.one_hot(labels_placeholder,depth=2)
       #tf.one_hot())
        #class_weights = tf.matmul(onehotlabs,tf.reshape(class_weight,[2,1]))
        #print class_weights
        #class_weights = tf.multiply(tf.cast(tf.reshape(labels_placeholder,[BATCHSIZE,1]),dtype=tf.float32),tf.reshape(class_weight,[1,2]))
        #tf.multiply(tf.cast(tf.reshape(labels_placeholder,[BATCHSIZE,1]),dtype=tf.float32),tf.reshape(class_weight,[1,2]))
        loss = NN.loss(logits,labels_placeholder,reg=0.0000001,l2_loss=reg_loss)
        
        
        increment_global_step_op = tf.assign(global_step, global_step+1)
        train_op = NN.training(loss,learning_rate,global_step)
        
        eval_correct = NN.evaluation(logits, labels_placeholder)
        #precision_total,precision_batch = precision_eval(logits,labels_placeholder)
        #recall_total, recall_batch = recall_eval(logits,labels_placeholder)
        metrics_op = NN.metrics_eval(logits,labels_placeholder) 
        summary_op = tf.summary.merge_all()
        predictions_op = tf.argmax(logits,1)
        init = tf.global_variables_initializer()
        
        saver = tf.train.Saver()
        
        sess = tf.Session()

        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
        
        sess.run(init)
        print 'Starting training...'
        for epoch in range(1,NUMEPOCHS+1):
            num_correct_epoch = 0
            loss_average = 0
            false_positives_sum = 0
            false_negatives_sum = 0
            true_positives_sum = 0
            true_negatives_sum = 0
            for step in range(0,n):
                
                batch_features = features[(step*BATCHSIZE):(step+1)*BATCHSIZE,:]
                batch_labels = labels[step*BATCHSIZE:(step+1)*BATCHSIZE]

                #
                feed_dict={features_placeholder:batch_features, labels_placeholder: batch_labels}

                _,total_loss,correct, metrics, predictions = sess.run([train_op,loss,eval_correct,metrics_op, predictions_op],
                                                feed_dict=feed_dict)
                
                
                (true_positives, true_negatives, false_positives, false_negatives) = metrics
                false_positives_sum += false_positives
                false_negatives_sum += false_negatives
                true_positives_sum += true_positives
                true_negatives_sum += true_negatives
                loss_average+=total_loss
                num_correct_epoch += correct
            loss_average /= float(n)
            #recall = true_positives_sum/(true_positives_sum + false_negatives_sum)
            #precision = true_positives_sum/(true_positives_sum+false_positives_sum)
            #fscore = 2*(precision*recall)/(precision+recall)
            accuracy_history[epoch-1] = num_correct_epoch/float(step*BATCHSIZE)
            print "Epoch: " + repr(epoch)+" Loss: " + repr(loss_average) +" Accuracy: " + repr(accuracy_history[epoch-1])
            #" Fscore: " + repr(fscore)
            print "False negatives: " + repr(false_negatives_sum)
            print "True negatives:" + repr(true_negatives_sum)
            print "False Positives: " + repr(false_positives_sum)
            print "True Positives: " + repr(true_positives_sum)
            sess.run(increment_global_step_op) 
run_training()