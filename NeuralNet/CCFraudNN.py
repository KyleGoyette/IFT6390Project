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
import os
#from imblearn import under_sampling, over_sampling
#from imblearn.over_sampling import SMOTE
#hyperparams


class FraudNN:
    #Fraud Training Class
    
    def __init__(self,hidden_dims,summary_path,trainpath,logdir,valpath=None,batchNorm=True):
        self.types=collections.OrderedDict([
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
        self.hidden_dims=hidden_dims
        self.trainpath = trainpath
        self.summary_path = summary_path
        self.valpath = valpath
        self.batchNorm = batchNorm
        self.logdir=logdir
    
    def load_train_data(self,y_name = 'Class',seed=None):
        train_path = self.trainpath
        val_path = self.valpath
        
        
        train_data = pd.read_csv(train_path, names = self.types.keys(), dtype=self.types,
                                    header=1)
        train_features=train_data.values
        np.random.shuffle(train_features)
        self.mean_time = np.mean(train_features[:,1])
        self.std_time = np.std(train_features[:,1])
        train_features[:,1] = (train_features[:,1]-self.mean_time)/self.std_time
        
        self.mean_amount = np.mean(train_features[:,-2])
        self.std_amount = np.std(train_features[:,-2])
        train_features[:,-2] = (train_features[:,-2]-self.mean_amount)/self.std_amount
        train_features = train_features[:,:-1]
        
        train_labels = train_data.values
        train_labels = train_labels[:,-1].astype(np.int)
        print train_features.shape
        
        if (type(val_path)==type(None)):
            return (train_features, train_labels), (None,None)
        else:
            val_data = pd.read_csv(val_path, names = self.types.keys(), dtype=self.types,
                           header=0)
            
            val_features = val_data.values
            
            val_features[:,1] = (val_features[:,1]-self.mean_time)/self.std_time
            val_features[:,-2] = (val_features[:,-2]-self.mean_amount)/self.std_amount
            
            val_labels = val_features[:,-1].astype(np.int)
            print "Sum of positives in val: " + repr(np.sum(val_labels))
            val_features = val_features[:,:-1]
            print val_features.shape
            
        return (train_features, train_labels), (val_features,val_labels)
    
    def placeholder_inputs(self,batchsize):
        features_placeholder = tf.placeholder(tf.float32, shape=(batchsize,30))
        labels_placeholder = tf.placeholder(tf.int32,shape=(batchsize),name="TrainLabels")
        return features_placeholder, labels_placeholder
    
    def calc_ratio(self,labels):
        num_1_examples = np.sum(labels)
        return num_1_examples/float(labels.shape[0])
    
    
    def save_to_log(self,save_path,loss,accuracy,val_acc,train_auc,val_auc,fscore,fscore_val,precision,precision_val,recall,recall_val,train_tp,train_fp,train_tn,train_fn,val_tp,val_fp,val_tn,val_fn,hyperparams):
        num_epochs = len(train_tp)
        save_file = save_path+"stats.csv"
        f = open(save_file,'w')
        f.write("Epoch, Loss, Train Accuracy, Train Auc, Train Fscore, Train Precision, Train Recall, Validation Accuracy, Validation Auc, Validation Fscore, Validation Precision, Validation Recall, Train TP, Train Fp, Train Tn, Train Fn, Val Tp, Val Fp, Val Tn, Val Fn\n")
        for ep in range(num_epochs):
            f.write("{:d}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}\n".format(ep,loss[ep],accuracy[ep],train_auc[ep],fscore[ep],precision[ep],recall[ep],val_acc[ep],val_auc[ep],
                    fscore_val[ep],precision_val[ep],recall_val[ep],train_tp[ep],train_fp[ep], train_tn[ep],train_fn[ep],val_tp[ep],val_fp[ep],val_tn[ep],val_fn[ep]))
        f.close()
        
        fo = open(save_path+"HyperParams.txt", "w")
        for k, v in hyperparams.items():
            fo.write(str(k) + ' : '+ str(v) + '\n')
        fo.close()
        
    def inference(self,inputs, test = False):
        outsize= self.hidden_dims
        D = inputs.shape[1]
        self.W=[0]*len(self.hidden_dims)
        self.b=[0]*len(self.hidden_dims)
        hidden=[0]*len(self.hidden_dims)

        for i in range(len(self.hidden_dims)):
            with tf.variable_scope('activation'+repr(i+1)) as scope:
                if test:
                    scope.reuse_variables()
                if (i==0):
                    self.W[i] = tf.get_variable('affine'+repr(i+1),shape=[D,outsize[i]], initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
                    self.b[i] = tf.get_variable('bias'+repr(i+1),shape=[outsize[i]], initializer = tf.constant_initializer(0.0))
                        #hidden = tf.nn.relu(tf.matmul(inputs,W[i])+b[i])
                    hidden = tf.matmul(inputs,self.W[i],name='weightmul1')
                    hidden = tf.add(hidden,self.b[i],name='addbias1')
                        
                        

                else:
                    self.W[i] = tf.get_variable('affine'+repr(i+1),shape=[outsize[i-1],outsize[i]], initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
                    self.b[i] = tf.get_variable('bias'+repr(i+1),shape=[outsize[i]], initializer = tf.constant_initializer(0.0))
                    hidden = tf.matmul(hidden,self.W[i],name="WeightMul"+repr(i+1))+self.b[i]


            tf.summary.histogram('weights'+repr(i+1),self.W[i])
            tf.summary.histogram('biases'+repr(i+1),self.b[i])
        
        
        
            with tf.variable_scope('hidden'+repr(i+1)) as scope:
                if test:
                    scope.reuse_variables()    
                if (i<len(self.hidden_dims)-1):
                    if self.batchNorm:
                        hidden = tf.contrib.layers.batch_norm(hidden)
                    hidden = tf.nn.relu(hidden,name="hidden" + repr(i+1)+ "relu")

        logits = hidden
        

        return logits

    def loss(self,logits,labels,batchsize,weights=1.0):
    
        #weighted_logits = tf.multiply(logits,class_weights)
        #softmax_logits = tf.nn.softmax(logits,name='Softmax')
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name="CrossEntropy")
        cross_entropy = tf.losses.sigmoid_cross_entropy(tf.cast(tf.reshape(labels,shape=(batchsize,1)),dtype=tf.float32),weights=weights,logits=logits)
        
        #scaled_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels,softmax_logits,weights=class_weights)
        loss = tf.reduce_mean(cross_entropy,name="CrossEntropy_mean")
        l2loss=0
        for i in range(0,len(self.hidden_dims)):
            l2loss+= self.reg*tf.nn.l2_loss(self.W[i])
        loss+=l2loss
        #loss += l2_loss
        tf.summary.scalar('loss',loss)
        #tf.summary.scalar('l2Loss',l2_loss)
        return loss    
    
    def training(self,loss, learning_rate,global_step):

        optimizer = tf.train.AdagradOptimizer(learning_rate)
        #tf.summary.scalar('learning_rate',learning_rate)
        train_op = optimizer.minimize(loss,global_step=global_step)
        return train_op
    
    
    def metrics_eval_sigmoid(self,logits,labels,batchsize,val=False):
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
    
    def train(self,initial_learning_rate,num_epochs,batchsize,reg,savepath,weight_logits=True):
        
        (train_features, train_labels),(val_features,val_labels) = self.load_train_data()
        self.num_examples = train_features.shape[0]
        self.reg=reg
        n = self.num_examples/batchsize
        ratio = self.calc_ratio(train_labels)
        print ratio
        
        
        
        with tf.Graph().as_default():
            
            global_step = tf.Variable(0,trainable=False)
            learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               10*self.num_examples/batchsize, 0.96, staircase=True)
            features_placeholder, labels_placeholder = self.placeholder_inputs(batchsize)

            logits = self.inference(features_placeholder)
            if weight_logits:
                class_weights = tf.where(labels_placeholder==1,1.0-ratio,ratio)
                loss = self.loss(logits,labels_placeholder,batchsize,weights=class_weights)
            else:
                loss = self.loss(logits,labels_placeholder,batchsize)
            train_op = self.training(loss,learning_rate,global_step)
            
            #auc calc
            reset_op = tf.local_variables_initializer()
            auc_features_placeholder = tf.placeholder(tf.float32,shape=(self.num_examples,30),name="AUCFeatsPlaceholder")
            auc_labels_placeholder = tf.placeholder(tf.int32,shape=(self.num_examples),name="AUCLabelsPlaceholder")
            auc_logits = self.inference(auc_features_placeholder,test=True)
            probs = tf.sigmoid(auc_logits)
            metrics_op=self.metrics_eval_sigmoid(probs,auc_labels_placeholder,self.num_examples)
            
            auc_op,update_op = tf.metrics.auc(labels=auc_labels_placeholder,predictions=probs,curve='PR')
            tf.summary.scalar("TrainAUC",auc_op)
            
            if (type(val_features) != type(None)):
                reset_op_val = tf.local_variables_initializer()
                self.valsize = val_features.shape[0]
                print self.valsize
                val_features_placeholder = tf.placeholder(tf.float32,shape=(self.valsize,30),name="ValFeatsPlaceholder")
                val_labels_placeholder = tf.placeholder(tf.int32,shape=(self.valsize),name="ValLabelsPlaceholder")
                
                val_preds = self.inference(val_features_placeholder,test=True)
                val_probs = tf.sigmoid(val_preds)
                val_metrics_op = self.metrics_eval_sigmoid(val_probs,val_labels_placeholder,self.valsize)
                
                auc_val_op,val_update_op = tf.metrics.auc(labels=val_labels_placeholder,predictions=val_probs,curve='PR')
                tf.summary.scalar("ValAUC",auc_val_op)
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
            
            precision = [0]*num_epochs
            recall = [0]*num_epochs
            fscore = [0]*num_epochs
            accuracy = [0]*num_epochs
            precision_val = [0]*num_epochs
            recall_val = [0]*num_epochs
            fscore_val = [0]*num_epochs
            accuracy_val = [0]*num_epochs
            print 'Starting training...'
            for epoch in range(0,num_epochs):
                loss_average = 0
                for step in range(0,n):
                    batch_lower_bound = (step*batchsize)%self.num_examples
                    batch_upper_bound = ((step+1)*batchsize)%self.num_examples
    
                    batch_features = train_features[batch_lower_bound:batch_upper_bound,:]
                    batch_labels = train_labels[batch_lower_bound:batch_upper_bound]
    
                    feed_dict={features_placeholder:batch_features, labels_placeholder: batch_labels}
    
                    _,batch_loss,  predictions = sess.run([train_op,loss, predictions_op],
                                                    feed_dict=feed_dict)
                    loss_average+=batch_loss
                summary = sess.run(summary_op,feed_dict=feed_dict)
                summary_writer.add_summary(summary,epoch)
                #Epoch Checks - Calculate AUC-PR for train data
                auc_feed_dict={
                        auc_features_placeholder: train_features,
                        auc_labels_placeholder: train_labels
                        }
                sess.run([reset_op])
                _,auc,logitsy,probsy,metrics = sess.run([auc_op,update_op,auc_logits,probs,metrics_op],feed_dict=auc_feed_dict)
                auc_history[epoch] = auc
                (true_positives[epoch], true_negatives[epoch],
                 false_positives[epoch], false_negatives[epoch]) = metrics
                accuracy[epoch] = (true_positives[epoch]+true_negatives[epoch])/float(true_positives[epoch] + true_negatives[epoch] + false_positives[epoch] + false_negatives[epoch])
                precision[epoch] = true_positives[epoch]/float(true_positives[epoch]+false_positives[epoch])
                recall[epoch] = true_positives[epoch]/float(true_positives[epoch]+false_negatives[epoch])
                fscore[epoch] = 2*precision[epoch]*recall[epoch]/float(precision[epoch]+recall[epoch])
                #update tensorboard
                
                #save epoch loss to loss history
                loss_history[epoch] = loss_average
                if (type(val_features) != type(None)):

                    val_feed_dict = {
                        val_features_placeholder: val_features,
                        val_labels_placeholder: val_labels
                    }
                    sess.run([reset_op_val])
                    _,val_auc[epoch], val_metrics,vprobs = sess.run([auc_val_op,val_update_op,val_metrics_op,val_probs],feed_dict=val_feed_dict)

                    (tp_valy,tn_valy,fp_valy,fn_valy) = val_metrics

                    tp_val[epoch] = tp_valy
                    tn_val[epoch] = tn_valy
                    fp_val[epoch] = fp_valy
                    fn_val[epoch] = fn_valy
                    accuracy_val[epoch] = (tp_val[epoch]+tn_val[epoch])/float(tp_valy+tn_valy+fn_valy+fp_valy)
                    precision_val[epoch] = tp_valy/(float(tp_valy+fp_valy))
                    recall_val[epoch] = tp_valy/float(tp_valy+fn_valy)
                    fscore_val[epoch] = 2*precision_val[epoch]*recall_val[epoch]/float(precision_val[epoch]+recall_val[epoch])
                #val_acc = (tp_valy+tn_valy)/float(tp_valy+fp_valy+tn_valy+fn_valy)
                print "Epoch: " + repr(epoch+1)+" Loss: " + repr(loss_history[epoch]) +" Train AUC: " + repr(auc_history[epoch]) + " Val AUC: " +repr(val_auc[epoch])
                print "Train Set:"
                print "False negatives: " + repr(false_negatives[epoch])
                print "True negatives: " + repr(true_negatives[epoch])
                print "False Positives: " + repr(false_positives[epoch])
                print "True Positives: " + repr(true_positives[epoch])
                print "Val Set:"
                print "False negatives: " + repr(fn_val[epoch])
                print "True negatives: " + repr(tn_val[epoch])
                print "False Positives: " + repr(fp_val[epoch])
                print "True Positives: " + repr(tp_val[epoch])
                sess.run(increment_global_step_op)
                hparams={
                        'learning rate': initial_learning_rate,
                        'dims': self.hidden_dims,
                        'Regularization': self.reg,
                        'decay rate': 0.96,
                        'batchnorm:': self.batchNorm
                        }
            self.save_to_log(self.logdir,loss_history,accuracy,accuracy_val,auc_history,val_auc,fscore,fscore_val,precision,precision_val,recall,recall_val,true_positives,false_positives,true_negatives,false_negatives, tp_val,fp_val,tn_val,fn_val,hparams)
            save_net_path = saver.save(sess,savepath)
            return max(val_auc)
    #(features, labels),(val_features,val_labels),(test_features,test_labels)= load_data([filename_train])


hyperparams = {
        'dims':[[30,50,75,100,125,1],[40,60,90,135,200,1]],
        'learning_rate':[0.001,0.01],
        'reg': [0.0000001,0.0000003,0.000001],
        'batchNorm': True,
        'weight_classes': [True,False]
        }
NUMEPOCHS = 10000
BATCHSIZE = 1000
num_epochs=200
batchsize=1000
reg=0.000001
SUMMARYDIR = '../Fraud/logs/'
filename_train="../CC_fraud_Train.csv"
filename_val = "../CC_fraud_Val.csv"
#hidden_dims,summary_path,trainpath,valpath=None,batchNorm=False
NN_id = 0
aucs={}
best_val_auc=10000
statsdir = '/home/kyle/Documents/IFT6390/Project/IFT6390Project/Fraud/'
savepath = '/home/kyle/Documents/IFT6390/Project/IFT6390Project/Fraud/Net'
#NN = FraudNN([80,100,150,200,1],SUMMARYDIR,filename_train,statsdir,filename_val)
#NN.train(initial_learning_rate,num_epochs,BATCHSIZE,reg,savepath)
for dim in hyperparams['dims']:
    for lr in hyperparams['learning_rate']:
        for regu in hyperparams['reg']:
            for weight in hyperparams['weight_classes']:
                NN_id += 1
                logdir = '/home/kyle/Documents/IFT6390/Project/IFT6390Project/Fraud/'+ repr(NN_id)+'/'
                statsdir = logdir + 'Curves/'
                savepath = logdir+'Savednets'+repr(NN_id)
                SummaryDir = logdir+ 'Summaries'
                os.makedirs(logdir)
                os.makedirs(statsdir)
                NN = FraudNN(dims,SummaryDir,filename_train,statsdir,filename_val)
                val_auc = NN.train(learning_rate,num_epochs,batchsize,reg,savepath)
                aucs[NN_id] = val_auc
                if val_auc<best_val_auc:
                    bestID = NN_id
fo = open("/home/kyle/Documents/IFT6390/Project/IFT6390Project/Fraud/Aucs.txt", "w")
for k, v in aucs.items():
    fo.write(str(k) + ' : '+ str(v) + '\n')
fo.close()