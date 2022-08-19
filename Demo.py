# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 18:28:40 2022

@author: fanyu
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

### demo for p = 1000, up = 10, n = 2000 of Simulation setting 1 Scenario 1 ###
p = 1000
up = 10
n = 2000
max_epoch = 200
taulist = [0.5,0.6,0.7,0.8,0.9,0.95]

train_data = generate_data(p,up,n,beta1 = 1, beta2 = 1,squ_beta = 0, interact_beta = 0, cigtype = 1,censor_rate = 0.3)

permutation = list(np.random.permutation(train_data['x'].shape[0]))

train = {}
train['x'] = train_data['x'][permutation[0:int(len(permutation)/2)],]
train['e'] = train_data['e'][permutation[0:int(len(permutation)/2)],]
train['t'] = train_data['t'][permutation[0:int(len(permutation)/2)],]
train['hr'] = train_data['hr'][permutation[0:int(len(permutation)/2)],]

test = {}
test['x'] = train_data['x'][permutation[int(len(permutation)/2):int(len(permutation))],]
test['e'] = train_data['e'][permutation[int(len(permutation)/2):int(len(permutation))],]
test['t'] = train_data['t'][permutation[int(len(permutation)/2):int(len(permutation))],]
test['hr'] = train_data['hr'][permutation[int(len(permutation)/2):int(len(permutation))],]

mm = CV_CoxNnTgdr(train,taulist,
                 10,max_epoch = max_epoch ,k = 2,
                 initial_learning_rate = 0.01,decay_rate = 0.99,
                 reaptimes = 1)

train_X = train['x']
train_y = {'e': train['e'], 't': train['t']}
test_X = test['x']
test_y = {'e': test['e'], 't': test['t']}

train_data_dic = dict()
train_data_dic['X'], train_data_dic['E'], \
train_data_dic['T'], train_data_dic['failures'], \
train_data_dic['atrisk'], train_data_dic['ties'] = parse_data(train_X, train_y)

test_data_dic = dict()
test_data_dic['X'], test_data_dic['E'], \
test_data_dic['T'], test_data_dic['failures'], \
test_data_dic['atrisk'], test_data_dic['ties'] = parse_data(test_X, test_y)

loss_train_list,loss_test_list,CI_train_list,risks,CI_test_list,Wlist = CoxNnTgdr(train_X,train_y,
                                                                                  train_data_dic,test_data_dic,10,
                                                                                 test_X,test_y,max_epoch = mm['Best_ite'],
                                                                                 initial_learning_rate = 0.01,
                                                                                 decay_rate = 0.99, tau = mm['Best_tau'])

k = int(mm['Best_ite']/50)
w1 = Wlist[k-1]['W_1']
w2 = Wlist[k-1]['W_2']
w12 = w1.dot(w2)
rea_fea = np.zeros(p)
rea_fea[0:up] = 1
pre_fea = np.array(abs(w12)>0*max(abs(w12))).astype(int)
TN,FP,FN,TP = confusion_matrix(rea_fea,pre_fea).ravel()
fea_precision = TP/(TP+FP)
fea_recall = TP/(TP+FN)
fea_Fmeasure = 2*fea_precision*fea_recall/(fea_precision+fea_recall)
auc = roc_auc_score(rea_fea, abs(w12))





