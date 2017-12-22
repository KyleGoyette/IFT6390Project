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
from sklearn.model_selection import train_test_split
#hyperparams


class DefaultNN:
    #Default training classs for the RNN
    
    def __init__(self,hidden_dims,summary_path,trainpath,valpath=None,batchNorm=False):
        self.types=collections.OrderedDict([
                ("ID",type((0.0))),
                ("LIMIT_BAL",type((0.0))),
                ("SEX",type((0.0))),
                ("EDUCATION",type((0.0))),
                ("MARRIAGE",type((0.0))),
                ("AGE",type((0.0))),
                ("PAY_0",type((0.0))),
                ("PAY_2",type((0.0))),
                ("PAY_3",type((0.0))),
                ("PAY_4",type((0.0))),
                ("PAY_5",type((0.0))),
                ("PAY_6",type((0.0))),
                ("BILL_AMT1",type((0.0))),
                ("BILL_AMT2",type((0.0))),
                ("BILL_AMT3",type((0.0))),
                ("BILL_AMT4",type((0.0))),
                ("BILL_AMT5",type((0.0))),
                ("BILL_AMT6",type((0.0))),
                ("PAY_AMT1",type((0.0))),
                ("PAY_AMT2",type((0.0))),
                ("PAY_AMT3",type((0.0))),
                ("PAY_AMT4",type((0.0))),
                ("PAY_AMT5",type((0.0))),
                ("PAY_AMT6",type((0.0))),
                ("default.payment.next.month",type((0))),
                ])
        self.hidden_dims=hidden_dims
        self.trainpath = trainpath
        self.summary_path = summary_path
        self.valpath = valpath
        self.batchNorm = batchNorm
    
    def load_train_data(self,y_name = 'Class',seed=None):
        train_path = self.trainpath
        val_path = self.valpath
        
        
        train_data = pd.read_csv(train_path, names = self.types.keys(), dtype=self.types,
                                    header=1)
        train_feats = train_data[['AGE','LIMIT_BAL','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5',
                          'PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4',
                          'BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3',
                          'PAY_AMT4','PAY_AMT5','PAY_AMT6']]
        
        onehot_features = pd.get_dummies(train_data[['SEX','MARRIAGE','EDUCATION']],
                                         columns=['SEX','MARRIAGE','EDUCATION'])
        train_features = [train_feats,onehot_features]
        train_features = pd.concat(train_features,axis=1)
        
        features = train_feats.columns.values
        
        self.mean = {}
        self.std = {}
        for feature in features:
            self.mean[feature] = train_features[feature].mean()
            self.std[feature] = train_features[feature].std()
            train_features.loc[:,feature] = (train_features[feature]-self.mean[feature])/self.std[feature]
        
        train_features = train_features.values
        
        train_labels = train_data['default.payment.next.month'].astype(np.int)
        train_labels = train_labels.values
        
        if (type(val_path)==type(None)):
            return (train_features, train_labels), (None,None)
        else:
            val_data = pd.read_csv(val_path, names = self.types.keys(), dtype=self.types,
                           header=0)
            for feature in features:
                val_data.loc[:,feature] = (val_data[feature]-self.mean[feature])/self.std[feature]
                
            val_data = val_data.values
            val_features = val_data[:,:-1]
            val_labels = val_data[:,-1].astype(np.int)

    
        return (train_features, train_labels), (val_features,val_labels)
    
    def placeholder_inputs(self,batchsize):
        features_placeholder = tf.placeholder(tf.float32, shape=(batchsize,33))
        labels_placeholder = tf.placeholder(tf.int32,shape=(batchsize),name="TrainLabels")
        return features_placeholder, labels_placeholder
    
    def calc_ratio(self,labels):
        num_1_examples = np.sum(labels)
        return num_1_examples/float(labels.shape[0])
    
    
    def save_to_log(self,save_file,train_tp,train_fp,train_tn,train_fn,val_tp,val_fp,val_tn,val_fn,loss):
        num_epochs = len(train_tp)
        f = open(save_file,'w')
        f.write("Epoch, Train TP, Train Fp, Train Tn, Train Fn, Val Tp, Val Fp, Val Tn, Val Fn, Loss\n")
        for ep in range(num_epochs):
            f.write("{:d}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}\n".format(ep,train_tp[ep],train_fp[ep], train_tn[ep],train_fn[ep],val_tp[ep],val_fp[ep],val_tn[ep],val_fn[ep],loss[ep]))
        f.close()
        
    def inference(self,inputs, test = False):
        outsize= self.hidden_dims
        D = inputs.shape[1]
        W=[0]*len(self.hidden_dims)
        b=[0]*len(self.hidden_dims)
        hidden=[0]*len(self.hidden_dims)
        for i in range(len(self.hidden_dims)):
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
                if (i<len(self.hidden_dims)-1):    
                    hidden = tf.contrib.layers.batch_norm(hidden)
                    hidden = tf.nn.relu(hidden,name="hidden" + repr(i+1)+ "relu")

        logits = hidden
        
        #reg_loss = reg_loss1+reg_loss2+reg_loss3+reg_loss4+reg_loss5+reg_loss6
        return logits#,reg_loss

    def loss(self,logits,labels,batchsize):
    
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
    
    def training(self,loss, learning_rate,global_step):

        optimizer = tf.train.AdagradOptimizer(learning_rate)
        #tf.summary.scalar('learning_rate',learning_rate)
        train_op = optimizer.minimize(loss,global_step=global_step)
        return train_op
    
    
    def metrics_eval_sigmoid(self,logits,labels,batchsize):
        #sigmoid preds
        predictions = tf.cast(tf.round(logits),dtype=tf.int32)
        labels = tf.reshape(labels,shape=(batchsize,1))
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
    
    def train(self,initial_learning_rate,num_epochs,batchsize,reg,savepath):
        
        (features, labels),(val_features,val_labels) = self.load_train_data()
        self.num_examples = features.shape[0]
        n = self.num_examples/batchsize
        ratio = calc_ratio(labels)
        print ratio
        
        
        
        with tf.Graph().as_default():
            
            global_step = tf.Variable(0,trainable=False)
            learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               10*self.num_examples/batchsize, 0.96, staircase=True)
            features_placeholder, labels_placeholder = placeholder_inputs(batchsize)
            if (type(val_features) != type(None)):
                self.valsize = val_features.shape[0]
                val_features_placeholder = tf.placeholder(tf.float32,shape=(self.valsize,30))
                val_labels_placeholder = tf.placeholder(tf.float32,shape=(self.valsize))
            #class_weight = tf.constant([1.0-ratio, ratio])
            logits = self.inference(features_placeholder)
            #weighted_logits = tf.multiply(logits,class_weight)
            loss = NN.loss(logits,labels_placeholder,batchsize)
            train_op = NN.training(loss,learning_rate,global_step)
            
            #auc calc
            reset_op = tf.local_variables_initializer()
            auc_features_placeholder = tf.placeholder(tf.float32,shape=(self.num_examples,33))
            auc_labels_placeholder = tf.placeholder(tf.int32,shape=(self.num_examples))
            auc_logits = NN.inference(auc_features_placeholder,test=True)
            probs = tf.sigmoid(auc_logits)
            metrics_op=NN.metrics_eval_sigmoid(probs,auc_labels_placeholder,self.num_examples)
            
            auc_op,update_op = tf.metrics.auc(labels=auc_labels_placeholder,predictions=probs,curve='PR')
            tf.summary.scalar("AUC",auc_op)
            
            if (type(val_features) != type(None)):
                val_preds = NN.inference(val_features_placeholder,eva=True)
                val_metrics_op = NN.metrics_eval(val_preds,val_labels_placeholder)
            
            increment_global_step_op = tf.assign(global_step, global_step+1)
            tf.summary.scalar('learningrate',learning_rate)
    
            tf.summary.scalar('globalstep',global_step)
            #precision_total,precision_batch = precision_eval(logits,labels_placeholder)
            #recall_total, recall_batch = recall_eval(logits,labels_placeholder)
            #metrics_op = NN.metrics_eval_sigmoid(logits,labels_placeholder) 
            summary_op = tf.summary.merge_all()
            predictions_op = tf.argmax(logits,1)
            
            
            init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            
            saver = tf.train.Saver()
            
            sess = tf.Session()
    
            summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)
            
            sess.run(init)
            
            loss_history = [0]*num_epochs
            false_positives = [0]*num_epochs
            false_negatives = [0]*num_epochs
            true_positives = [0]*num_epochs
            true_negatives = [0]*num_epochs
            auc_history = [0]*num_epochs
            
            val_auc = [0]*num_epochs
            tp_val = [0]*num_epochs
            tn_val = [0]*num_epochs
            fp_val = [0]*num_epochs   
            fn_val = [0]*num_epochs       
            print 'Starting training...'
            for epoch in range(0,num_epochs):
                loss_average = 0
                for step in range(0,n):
                    batch_lower_bound = (step*batchsize)%self.num_examples
                    batch_upper_bound = ((step+1)*batchsize)%self.num_examples
    
                    batch_features = features[batch_lower_bound:batch_upper_bound,:]
                    batch_labels = labels[batch_lower_bound:batch_upper_bound]
    
                    feed_dict={features_placeholder:batch_features, labels_placeholder: batch_labels}
    
                    _,batch_loss,  predictions = sess.run([train_op,loss, predictions_op],
                                                    feed_dict=feed_dict)
                    loss_average+=batch_loss
                
                #Epoch Checks - Calculate AUC-PR for train data
                auc_feed_dict={
                        auc_features_placeholder: features[:,:],
                        auc_labels_placeholder: labels[:]
                        }
                sess.run([reset_op])
                _,auc,logitsy,probsy,metrics = sess.run([auc_op,update_op,auc_logits,probs,metrics_op],feed_dict=auc_feed_dict)
                auc_history[epoch] = auc
                (true_positives[epoch], true_negatives[epoch],
                 false_positives[epoch], false_negatives[epoch]) = metrics
                #update tensorboard
                summary = sess.run(summary_op,feed_dict=feed_dict)
                summary_writer.add_summary(summary,epoch)
                #save epoch loss to loss history
                loss_history[epoch] = loss_average
                if (type(val_features) != type(None)):
                    val_feed_dict = {
                        val_features_placeholder:val_features,
                        val_labels_placeholder: val_labels
                    }
                    sess.run([reset_op])
                    val_auc, val_metrics = sess.run([update_op,metrics_op],feed_dict=val_feed_dict)
                    tp_valy,tn_valy,fp_valy,fn_valy = val_metrics
                    tp_val[epoch] = tp_valy
                    tn_val[epoch] = tn_valy
                    fp_val[epoch] = fp_valy
                    fn_val[epoch] = fn_valy
                #val_acc = (tp_valy+tn_valy)/float(tp_valy+fp_valy+tn_valy+fn_valy)
                print "Epoch: " + repr(epoch+1)+" Loss: " + repr(loss_history[epoch]) +" AUC: " + repr(auc_history[epoch])# + ", Val acc: " + repr(val_acc)
                #" Fscore: " + repr(fscore)
                print "False negatives: " + repr(false_negatives[epoch])
                print "True negatives: " + repr(true_negatives[epoch])
                print "False Positives: " + repr(false_positives[epoch])
                print "True Positives: " + repr(true_positives[epoch])
                sess.run(increment_global_step_op)
            #save_to_log("../Curves4.csv",true_positives_sum,false_positives_sum,true_negatives_sum,false_negatives_sum, tp_val,fp_val,tn_val,fn_val,loss_history)
            save_net_path = saver.save(sess,savepath)#"../Net_saves/model.ckpt")
    #(features, labels),(val_features,val_labels),(test_features,test_labels)= load_data([filename_train])
    
dims=[128,3*128,3*3*128,3*3*128,3*3*3*128,1]
NUMEPOCHS = 10000
BATCHSIZE = 1000
initial_learning_rate = 0.001
num_epochs=10
batchsize=1000
reg=0
savepath='/home/kyle/Documents/IFT6390/Project/IFT6390Project/Net_saves/Default/boo'
REGU=0.0000000
SUMMARYDIR = '../Default/logs/'
filename_train="../CreditCardDefault_Train.csv"
#hidden_dims,summary_path,trainpath,valpath=None,batchNorm=False
NN = DefaultNN(dims,SUMMARYDIR,filename_train)
NN.train(initial_learning_rate,num_epochs,batchsize,reg,savepath)