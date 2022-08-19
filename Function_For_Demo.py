# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 18:05:23 2022

@author: fanyu
"""

import time
import os
import pandas as pd  
import numpy as np
import tensorflow as tf
import math
from lifelines.utils import concordance_index

###function###
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return (tf.nn.tanh(layer))

def x_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return (layer)

def random_mini_batches(X, Y, mini_batch_size = 30): 
    m = X.shape[0]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation , :]
    shuffled_Y = Y[permutation , ]

    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, ]
        mini_batch = (mini_batch_X , mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m , :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m , ]
        mini_batch = (mini_batch_X , mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def jieduan(v,tau):
    
    a = v
    comparison = tf.less(abs(a), tf.reduce_max(abs(a))*tf.constant(tau)) 
    b = tf.where(comparison, tf.zeros_like(a), a)
    return b

def negative_log_likelihood(y_true, y_pred,train_data):

    logL = 0
    cumsum_y_pred = tf.cumsum(y_pred)
    hazard_ratio = tf.exp(y_pred)
    cumsum_hazard_ratio = tf.cumsum(hazard_ratio)
    if train_data['ties'] == 'noties':
        log_risk = tf.log(cumsum_hazard_ratio)
        likelihood = y_pred - log_risk
        uncensored_likelihood = likelihood * y_true
        logL = -tf.reduce_sum(uncensored_likelihood)
    else:
        for t in train_data['failures']:                                                                       
            tfail = train_data['failures'][t]
            trisk = train_data['atrisk'][t]
            d = len(tfail)
            dr = len(trisk)

            logL += -cumsum_y_pred[tfail[-1]] + (0 if tfail[0] == 0 else cumsum_y_pred[tfail[0]-1])

            if train_data['ties'] == 'breslow':
                s = cumsum_hazard_ratio[trisk[-1]]
                logL += tf.log(s) * d
            elif train_data['ties'] == 'efron':
                s = cumsum_hazard_ratio[trisk[-1]]
                r = cumsum_hazard_ratio[tfail[-1]] - (0 if tfail[0] == 0 else cumsum_hazard_ratio[tfail[0]-1])
                for j in range(d):
                    logL += tf.log(s - j * r / d)
            else:
                raise NotImplementedError('tie breaking method not recognized')
    observations = tf.reduce_sum(y_true)
    return logL / observations

def metrics_ci(label_true, y_pred):
    hr_pred = -y_pred
    ci = concordance_index(label_true['t'], hr_pred, label_true['e'])
    return ci

def prepare_data(x, label):
    if isinstance(label, dict):
       e, t = label['e'], label['t']

    df1 = pd.DataFrame({'t': t, 'e': e})
    df1.sort_values(['t', 'e'], ascending=[False, True], inplace=True)
    sort_idx = list(df1.index)
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    return x, {'e': e, 't': t}

def parse_data(x, label):
    x, label = prepare_data(x, label)
    e, t = label['e'], label['t']

    failures = {}
    atrisk = {}
    n, cnt = 0, 0

    for i in range(len(e)):
        if e[i]:
            if t[i] not in failures:
                failures[t[i]] = [i]
                n += 1
            else:
                cnt += 1
                failures[t[i]].append(i)

            if t[i] not in atrisk:
                atrisk[t[i]] = []
                for j in range(0, i+1):
                    atrisk[t[i]].append(j)
            else:
                atrisk[t[i]].append(i)
    if cnt >= n / 2:
        ties = 'efron'
    elif cnt > 0:
        ties = 'breslow'
    else:
        ties = 'noties'

    return x, e, t, failures, atrisk, ties

    
def linear_H(p,up,x,beta1,beta2,squ_beta,interact_beta):

    b = np.zeros((p,))
    b[0:int(up/2)] = 1*beta1
    b[int(up/2):up] = 1*beta2

    risk = np.dot(x, b) + \
    squ_beta * (x[:,(up)]*x[:,(up)] + x[:,(up+1)]*x[:,(up+1)]) +\
    interact_beta * 5 * (x[:,(up-2)]*x[:,(up-1)]+x[:,(up-2)]*x[:,(up-3)])
    return risk




def generate_data(p,up, N, beta1, beta2,squ_beta,interact_beta,
    cigtype = 1, censor_rate = 0.3,
    average_death = 5):

    if cigtype == 2:
        cig = np.eye(p)
    elif cigtype == 1:
        c1 = (range(1,p+1)*np.ones((p,p))).astype(int)
        cig = np.exp(-abs(c1.T-c1))
        

    data = np.random.multivariate_normal(mean=np.zeros(p), cov=cig, size=N)

    p_death = average_death * np.ones((N,1))
    risk = linear_H(p, up, data, beta1, beta2, squ_beta,interact_beta)
    risk = risk - np.mean(risk)
    death_time = np.zeros((N,1))
    for i in range(N):
        death_time[i] = np.random.exponential(p_death[i]) / np.exp(risk[i])
    death_time_sort = np.array(death_time)
    end_time = np.sort(death_time_sort,axis = 0)[int(N*(1-censor_rate))]
    censoring = np.ones((N,1))
    death_time[death_time > end_time] = end_time
    censoring[death_time == end_time] = 0
    death_time = np.squeeze(death_time)
    censoring = np.squeeze(censoring)

    dataset = {
        'x' : data.astype(np.float32),
        'e' : censoring.astype(np.int32),
        't' : death_time.astype(np.float32),
        'hr' : risk.astype(np.float32)
    }

    return dataset

def CoxNnTgdr(x_train,y_train,train_data_dic,test_data_dic,n_hidden_1,
              x_test,y_test,max_epoch = 500,initial_learning_rate = 0.1,
              decay_rate = 1,tau = 0.8,batch_size = 100):
    
    n_input = x_train.shape[1]
    X = tf.placeholder("float32", [None, n_input] , name = 'X')
    Y = tf.placeholder("float32", [None, 1] , name = 'Y')
    ### Create train parameters ###
    w_1_name = "w_1_{it}".format(it = tau)
    w_2_name = "w_2_{it}".format(it = tau)
    b_1_name = "b_1_{it}".format(it = tau)
    weights = {
            'W_1': tf.Variable(tf.zeros([n_input, n_hidden_1],dtype=tf.float32) , name = w_1_name),
            'W_2': tf.Variable(tf.zeros([n_hidden_1, 1],dtype=tf.float32) , name = w_2_name), 
            }
    biases = {
            'b_1': tf.Variable(tf.random_normal([n_hidden_1] ,mean=0.0, stddev=0.01, seed=None) , name = b_1_name),
            }
    
    ### create the layer ###
    layer_1 = tf.nn.dropout(fully_connected(X, weights['W_1'], biases['b_1']),1)
    final_output = tf.matmul(layer_1, weights['W_2'])    
    
    loss1 = negative_log_likelihood(Y, final_output,train_data_dic)
    loss2 = negative_log_likelihood(Y, final_output,test_data_dic)
    
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                               global_step = max_epoch,
                               decay_steps = 25,
                               decay_rate = decay_rate,
                               staircase = False)
    useoptimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    grads = useoptimizer.compute_gradients(loss1)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (jieduan(g,tau), v)
    optimizers = useoptimizer.apply_gradients(grads)
    
    W = weights
    Wlist = []
    loss_train_list = []
    loss_test_list = []
    CI_train_list = []
    CI_test_list = []
    
    N = train_data_dic['E'].shape[0]
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # Train steps
        for i in range(max_epoch):
            _, output_y, loss_value = sess.run([optimizers, final_output, loss1],
                                              feed_dict = {X:  train_data_dic['X'] , Y:train_data_dic['E'].reshape((N, 1))})
            # Record information
            loss_train_list.append(loss_value)
            label = {
                't': train_data_dic['T'],
                'e': train_data_dic['E']
            }

            CI = metrics_ci(label, output_y)
            CI_train_list.append(CI)
            # Print evaluation on test set
            if ((i+1) % 50 == 0):
                risks, loss_test_value, w_l = sess.run([final_output,loss2,W],feed_dict = {X: x_test,Y: y_test['e'].reshape(len(y_test['e']), 1)})
                risks = np.squeeze(risks)
                CI_test = metrics_ci(y_test, risks)
                Wlist.append(w_l)
                loss_test_list.append(loss_test_value)
                CI_test_list.append(CI_test)

    return loss_train_list,loss_test_list,CI_train_list,risks,CI_test_list,Wlist


def CV_CoxNnTgdr(x,tau_list,n_hidden_1,
                 max_epoch = 200,k = 2,
                 initial_learning_rate = 0.01,decay_rate = 0.99,
                 reaptimes = 50,batch_size = 100):
    
    loss_test1_100 = {}
    loss_test2_100 = {}
    loss_compare_mat = []
    for i in range(len(tau_list)):
        tf.reset_default_graph()
        loss_test1_100[i] = []
        loss_test2_100[i] = []
        for j in range(reaptimes):
            
            permutation = list(np.random.permutation(x['x'].shape[0]))
            X1_ind = permutation[0:int(len(permutation)/k)]
            X2_ind = permutation[int(len(permutation)/k):len(permutation)]
            
        
            train_data1_x = x['x'][X1_ind,]
            train_data1_y = {'e':x['e'][X1_ind,],'t':x['t'][X1_ind,]}
            train_data2_x = x['x'][X2_ind,]
            train_data2_y = {'e':x['e'][X2_ind,],'t':x['t'][X2_ind,]}
            
            train_data1_dic = dict()
            train_data1_dic['X'], train_data1_dic['E'], \
            train_data1_dic['T'], train_data1_dic['failures'], \
            train_data1_dic['atrisk'], train_data1_dic['ties'] = parse_data(train_data1_x, train_data1_y)
            
            train_data2_dic = dict()
            train_data2_dic['X'], train_data2_dic['E'], \
            train_data2_dic['T'], train_data2_dic['failures'], \
            train_data2_dic['atrisk'], train_data2_dic['ties'] = parse_data(train_data2_x, train_data2_y)    
            start_time = time.clock()
            loss_train1_list,loss_test1_list,CI_train1_list,risk1,CI_test1_list,Wlist1 = CoxNnTgdr(train_data1_x,train_data1_y,
                                                                                                   train_data1_dic,train_data2_dic,
                                                                                             n_hidden_1,train_data2_x,train_data2_y,
                                                                                             max_epoch,initial_learning_rate = 0.01,
                                                                                             decay_rate = 0.99, tau = tau_list[i],
                                                                                             batch_size = 100)
            loss_test1_100[i].append(CI_test1_list)
            loss_train2_list,loss_test2_list,CI_train2_list,risk2,CI_test2_list,Wlist2 = CoxNnTgdr(train_data2_x,train_data2_y,
                                                                                                   train_data2_dic,train_data1_dic,
                                                                                             n_hidden_1,train_data1_x,train_data1_y,
                                                                                             max_epoch,initial_learning_rate = 0.01,
                                                                                             decay_rate = 0.99, tau = tau_list[i],
                                                                                             batch_size = 100)
            loss_test2_100[i].append(CI_test2_list)
            end_time = time.clock()
            print('Taulist = '+str(tau_list[i])+'; Repeat = '+str(j+1)+';执行时间 = ' + str(end_time - start_time) + ' 秒')
            
        loss1_mat = np.matrix(loss_test1_100[i])
        loss2_mat = np.matrix(loss_test2_100[i])
        loss_mat = np.mean((loss1_mat + loss2_mat)/2,axis = 0)
        if (len(loss_compare_mat) == 0):
            loss_compare_mat = loss_mat
        else:
            loss_compare_mat = np.vstack((loss_compare_mat ,loss_mat))
        
    loss_compare_mat = np.matrix(loss_compare_mat)
    best_index = np.where(loss_compare_mat == np.max(loss_compare_mat))
    itelist = np.arange(50,50*int(max_epoch/50)+1,50)
    best_tau = tau_list[int(best_index[0][0])]
    best_ite = itelist[int(best_index[1][0])]
    model = {'Best_tau':best_tau,'Best_ite':best_ite,'CI':loss_compare_mat}
    return model



def CoxNn(x_train,y_train,train_data_dic,test_data_dic,n_hidden_1,
              x_test,y_test,max_epoch = 500,initial_learning_rate = 0.1,
              decay_rate = 1,lamb1 = 0.001):
    
    n_input = x_train.shape[1]
    X = tf.placeholder("float32", [None, n_input] , name = 'X')
    Y = tf.placeholder("float32", [None, 1] , name = 'Y')
    ### Create train parameters ###
    w_1_name = "w_1_{it}".format(it = lamb1)
    w_2_name = "w_2_{it}".format(it = lamb1)
    b_1_name = "b_1_{it}".format(it = lamb1)
    weights = {
            'W_1': tf.Variable(tf.random_normal([n_input, n_hidden_1],mean=0.0, stddev=0.001,dtype=tf.float32) , name = w_1_name),
            'W_2': tf.Variable(tf.random_normal([n_hidden_1, 1],mean=0.0, stddev=0.001,dtype=tf.float32) , name = w_2_name),    
            }
    biases = {
            'b_1': tf.Variable(tf.random_normal([n_hidden_1] ,mean=0.0, stddev=0.001, seed=None) , name = b_1_name),
            }
    
    ### create the layer ###
    layer_1 = fully_connected(X, weights['W_1'], biases['b_1'])
    final_output = tf.matmul(layer_1, weights['W_2'])    
    
    reg = tf.contrib.layers.l1_regularizer(lamb1)(weights['W_1']) + tf.contrib.layers.l1_regularizer(lamb1)(weights['W_2'])
    cost1 = negative_log_likelihood(Y, final_output, train_data_dic)
    cost2 = negative_log_likelihood(Y, final_output, test_data_dic)
    loss1 = cost1 + reg
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                               global_step = max_epoch,
                               decay_steps = 50,
                               decay_rate = decay_rate,
                               staircase = False)
    useoptimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    grads = useoptimizer.compute_gradients(loss1)
    optimizers = useoptimizer.apply_gradients(grads)
    
    W = weights
    Wlist = []
    loss_train_list = []
    loss_test_list = []
    CI_train_list = []
    CI_test_list = []
    
    N = train_data_dic['E'].shape[0]
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # Train steps
        for i in range(max_epoch):
            _, output_y, loss_value = sess.run([optimizers, final_output, cost1],
                                                feed_dict = {X:  train_data_dic['X'],Y: train_data_dic['E'].reshape((N, 1))})
            # Record information
            loss_train_list.append(loss_value)
            label = {
                't': train_data_dic['T'],
                'e': train_data_dic['E']
            }
            CI = metrics_ci(label, output_y)
            CI_train_list.append(CI)
            # Print evaluation on test set
            if ((i+1) % 50 == 0):
                risk, loss_test_value, w_l = sess.run([final_output,cost2,W],feed_dict = {X: x_test,Y: y_test['e'].reshape(len(y_test['e']), 1)})
                risk = np.squeeze(risk)
                CI_test = metrics_ci(y_test, risk)
                Wlist.append(w_l)
                loss_test_list.append(loss_test_value)
                CI_test_list.append(CI_test)

    return loss_train_list,loss_test_list,CI_train_list,risk,CI_test_list,Wlist


def CV_CoxNn(x,beta_list,n_hidden_1,
                 max_epoch = 500,k = 2,
                 initial_learning_rate = 0.01,decay_rate = 0.99,
                 reaptimes = 50):
    
    loss_test1_100 = {}
    loss_test2_100 = {}
    loss_compare_mat = []
    for i in range(len(beta_list)):
        tf.reset_default_graph()
        loss_test1_100[i] = []
        loss_test2_100[i] = []
        for j in range(reaptimes):
            
            permutation = list(np.random.permutation(x['x'].shape[0]))
            X1_ind = permutation[0:int(len(permutation)/k)]
            X2_ind = permutation[int(len(permutation)/k):len(permutation)]
            
        
            train_data1_x = x['x'][X1_ind,]
            train_data1_y = {'e':x['e'][X1_ind,],'t':x['t'][X1_ind,]}
            train_data2_x = x['x'][X2_ind,]
            train_data2_y = {'e':x['e'][X2_ind,],'t':x['t'][X2_ind,]}
            
            train_data1_dic = dict()
            train_data1_dic['X'], train_data1_dic['E'], \
            train_data1_dic['T'], train_data1_dic['failures'], \
            train_data1_dic['atrisk'], train_data1_dic['ties'] = parse_data(train_data1_x, train_data1_y)
            
            train_data2_dic = dict()
            train_data2_dic['X'], train_data2_dic['E'], \
            train_data2_dic['T'], train_data2_dic['failures'], \
            train_data2_dic['atrisk'], train_data2_dic['ties'] = parse_data(train_data2_x, train_data2_y)    
            start_time = time.clock()
            loss_train1_list,loss_test1_list,CI_train1_list,risk1,CI_test1_list,Wlist1 = CoxNn(train_data1_x,train_data1_y,
                                                                                               train_data1_dic,train_data2_dic,
                                                                                             n_hidden_1,train_data2_x,train_data2_y,
                                                                                             max_epoch,initial_learning_rate = 0.01,
                                                                                             decay_rate = 0.99, lamb1 = beta_list[i])
            loss_test1_100[i].append(CI_test1_list)
            loss_train2_list,loss_test2_list,CI_train2_list,risk2,CI_test2_list,Wlist2 = CoxNn(train_data2_x,train_data2_y,
                                                                                               train_data2_dic,train_data1_dic,
                                                                                             n_hidden_1,train_data1_x,train_data1_y,
                                                                                             max_epoch,initial_learning_rate = 0.01,
                                                                                             decay_rate = 0.99, lamb1 = beta_list[i])
            loss_test2_100[i].append(CI_test2_list)
            end_time = time.clock()
            print('Betalist = '+str(beta_list[i])+'; Repeat = '+str(j+1)+';执行时间 = ' + str(end_time - start_time) + ' 秒')
            
        loss1_mat = np.matrix(loss_test1_100[i])
        loss2_mat = np.matrix(loss_test2_100[i])
        loss_mat = np.mean((loss1_mat + loss2_mat)/2,axis = 0)
        if (len(loss_compare_mat) == 0):
            loss_compare_mat = loss_mat
        else:
            loss_compare_mat = np.vstack((loss_compare_mat ,loss_mat))
        
    loss_compare_mat = np.matrix(loss_compare_mat)
    best_index = np.where(loss_compare_mat == np.max(loss_compare_mat))
    itelist = np.arange(50,50*int(max_epoch/50)+1,50)
    best_tau = beta_list[int(best_index[0])]
    best_ite = itelist[int(best_index[1])]
    model = {'Best_beta':best_tau,'Best_ite':best_ite,'CI':loss_compare_mat}
    return model


def CoxregTgdr(x_train,y_train,train_data_dic,test_data_dic,
              x_test,y_test,max_epoch = 500,initial_learning_rate = 0.1,
              decay_rate = 0.99,tau = 0.8,batch_size = 100):
    
    n_input = x_train.shape[1]
    X = tf.placeholder("float32", [None, n_input] , name = 'X')
    Y = tf.placeholder("float32", [None, 1] , name = 'Y')
    w_1_name = "w_1_{it}".format(it = tau)
    weights = {
            'W_1': tf.Variable(tf.zeros([n_input, 1],dtype=tf.float32) , name = w_1_name),   
            }
    
    final_output = tf.matmul(X, weights['W_1'])    
    
    loss1 = negative_log_likelihood(Y, final_output,train_data_dic)
    loss2 = negative_log_likelihood(Y, final_output,test_data_dic)
    
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                               global_step = max_epoch,
                               decay_steps = 50,
                               decay_rate = decay_rate,
                               staircase = False)
    useoptimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    grads = useoptimizer.compute_gradients(loss1)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (jieduan(g,tau), v)
    optimizers = useoptimizer.apply_gradients(grads)
    
    W = weights
    Wlist = []
    loss_train_list = []
    loss_test_list = []
    CI_train_list = []
    CI_test_list = []
    
    N = train_data_dic['E'].shape[0]
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # Train steps
        for i in range(max_epoch):
            _, output_y, loss_value = sess.run([optimizers, final_output, loss1],
                                              feed_dict = {X:  train_data_dic['X'] , Y:train_data_dic['E'].reshape((N, 1))})
            # Record information
            loss_train_list.append(loss_value)
            label = {
                't': train_data_dic['T'],
                'e': train_data_dic['E']
            }

            CI = metrics_ci(label, output_y)
            CI_train_list.append(CI)
            # Print evaluation on test set
            if ((i+1) % 50 == 0):
                risks, loss_test_value, w_l = sess.run([final_output,loss2,W],feed_dict = {X: x_test,Y: y_test['e'].reshape(len(y_test['e']), 1)})
                risks = np.squeeze(risks)
                CI_test = metrics_ci(y_test, risks)
                Wlist.append(w_l)
                loss_test_list.append(loss_test_value)
                CI_test_list.append(CI_test)

    return loss_train_list,loss_test_list,CI_train_list,risks,CI_test_list,Wlist



def CV_CoxregTgdr(x,tau_list,
                 max_epoch = 200,k = 2,
                 initial_learning_rate = 0.01,decay_rate = 0.99,
                 reaptimes = 50,batch_size = 100):
    
    loss_test1_100 = {}
    loss_test2_100 = {}
    loss_compare_mat = []
    for i in range(len(tau_list)):
        tf.reset_default_graph()
        loss_test1_100[i] = []
        loss_test2_100[i] = []
        for j in range(reaptimes):
            
            permutation = list(np.random.permutation(x['x'].shape[0]))
            X1_ind = permutation[0:int(len(permutation)/k)]
            X2_ind = permutation[int(len(permutation)/k):len(permutation)]
            
        
            train_data1_x = x['x'][X1_ind,]
            train_data1_y = {'e':x['e'][X1_ind,],'t':x['t'][X1_ind,]}
            train_data2_x = x['x'][X2_ind,]
            train_data2_y = {'e':x['e'][X2_ind,],'t':x['t'][X2_ind,]}
            
            train_data1_dic = dict()
            train_data1_dic['X'], train_data1_dic['E'], \
            train_data1_dic['T'], train_data1_dic['failures'], \
            train_data1_dic['atrisk'], train_data1_dic['ties'] = parse_data(train_data1_x, train_data1_y)
            
            train_data2_dic = dict()
            train_data2_dic['X'], train_data2_dic['E'], \
            train_data2_dic['T'], train_data2_dic['failures'], \
            train_data2_dic['atrisk'], train_data2_dic['ties'] = parse_data(train_data2_x, train_data2_y)    
            start_time = time.clock()
            loss_train1_list,loss_test1_list,CI_train1_list,risk1,CI_test1_list,Wlist1 = CoxregTgdr(train_data1_x,train_data1_y,
                                                                                             train_data1_dic,train_data2_dic,
                                                                                             train_data2_x,train_data2_y,
                                                                                             max_epoch,initial_learning_rate = 0.01,
                                                                                             decay_rate = 0.99, tau = tau_list[i],
                                                                                             batch_size = 100)
            loss_test1_100[i].append(CI_test1_list)
            loss_train2_list,loss_test2_list,CI_train2_list,risk2,CI_test2_list,Wlist2 = CoxregTgdr(train_data2_x,train_data2_y,
                                                                                             train_data2_dic,train_data1_dic,
                                                                                             train_data1_x,train_data1_y,
                                                                                             max_epoch,initial_learning_rate = 0.01,
                                                                                             decay_rate = 0.99, tau = tau_list[i],
                                                                                             batch_size = 100)
            loss_test2_100[i].append(CI_test2_list)
            end_time = time.clock()
            print('Taulist = '+str(tau_list[i])+'; Repeat = '+str(j+1)+';执行时间 = ' + str(end_time - start_time) + ' 秒')
            
        loss1_mat = np.matrix(loss_test1_100[i])
        loss2_mat = np.matrix(loss_test2_100[i])
        loss_mat = np.mean((loss1_mat + loss2_mat)/2,axis = 0)
        if (len(loss_compare_mat) == 0):
            loss_compare_mat = loss_mat
        else:
            loss_compare_mat = np.vstack((loss_compare_mat ,loss_mat))
        
    loss_compare_mat = np.matrix(loss_compare_mat)
    best_index = np.where(loss_compare_mat == np.max(loss_compare_mat))
    itelist = np.arange(50,50*int(max_epoch/50)+1,50)
    best_tau = tau_list[int(best_index[0])]
    best_ite = itelist[int(best_index[1])]
    model = {'Best_tau':best_tau,'Best_ite':best_ite,'CI':loss_compare_mat}
    return model