
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
import re
import os


from models.beta_irt import Beta_IRT
import visualization.plots as vs

import pandas as pd
import argparse

from hsvi.tensorflow.distributions import Normal,Beta

def str2bool(x):
    if x.lower() == 'false':
        return False
    else:
        return True

parser = argparse.ArgumentParser()

name_fmt = 'Need input data file with the name in format: irt_data_[dataset]_s[data_size]_f[noise_fraction percentile]_sd[random_seed].csv'


parser.add_argument('-df','--IRT_dfile', default='./irt_data_moons_s400_f20_sd42_m12.csv', type=str, help='file path of IRT data')
parser.add_argument('-rp','--result_path', default='./results/', type=str, help='result path')
parser.add_argument('-am','--a_prior_mean', default=1., type=float, help='prior mean of discrimination')
parser.add_argument('-as','--a_prior_std', default=1., type=float, help='prior std dev of discrimination')
parser.add_argument('-fa','--fixed_a', default=False, type=str2bool, help='if use fixed discrimination, set to True')
parser.add_argument('-plts','--plot_scatter', default=False, type=str2bool, help='plot scatter figures for 2D data')
parser.add_argument('-sd','--seed', default=42, type=int, help='random seed')
parser.add_argument('-itr','--niter', default=1000, type=int, help='number of iterations')
parser.add_argument('-litr','--local_iter', default=10, type=int, help='number of local iterations')


args = parser.parse_args()
print('seed',args.seed)
tf.set_random_seed(args.seed)
np.random.seed(args.seed)



file_name = args.IRT_dfile

# read file name #
in_f = file_name.split('/')[-1]
fpath = file_name[:-len(in_f)]
in_f = re.split('_|\.',in_f)
if len(in_f) < 7:   
    print('Wrong format of the name of data file')
    print(name_fmt)
    sys.exit()
    

irt_data = pd.read_csv(file_name)

dataset = in_f[2]
result_path = os.path.join(args.result_path,dataset)
if not os.path.exists(result_path):
    os.mkdir(result_path)

niter = args.niter



partial_name = str.join('_',in_f[2:-1])
xtest = pd.read_csv(fpath+'xtest_'+partial_name+'.csv') # read original data

if args.fixed_a:
    partial_save_name = partial_name +'_fixed_am'+str(args.a_prior_mean).replace('.','@')
else:
    partial_save_name = partial_name +'_am'+str(args.a_prior_mean).replace('.','@')+'_as'+str(args.a_prior_std).replace('.','@')



# setup Beta IRT model #

M = irt_data.shape[0] #number of items
C = irt_data.shape[1] #number of classifiers


theta = Beta(tf.ones([C]),tf.ones([C]),sample_shape=[M],name='theta')
a,b = np.sum(1-irt_data,axis=1).astype(np.float32),np.sum(irt_data,axis=1).astype(np.float32)
delta = Beta(a,b,sample_shape=[C],name='delta')
if args.fixed_a:
    a = tf.ones(M)*args.a_prior_mean
else:
    a = Normal(tf.ones(M)*args.a_prior_mean,tf.ones([M])*args.a_prior_std,sample_shape=[C],name='a')

model = Beta_IRT(M,C,theta,delta,a)

D = np.float32(irt_data.values)

model.init_inference(data=D)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
model.fit(n_iter=niter,local_iter=args.local_iter,sess=sess)

# generate output files #

# output ability
ability = pd.DataFrame(index=irt_data.columns)
ability['ability'] = tf.nn.sigmoid(model.qtheta.distribution.loc).eval(session=sess)
ability.loc['stddev'] = ability.ability.std()
ability.to_csv(result_path+'/irt_ability_vi_'+partial_save_name+'.csv') 

# output difficulty and discrimination
if args.fixed_a:
    discrimination = a.eval(session=sess)
else:   
    discrimination = model.qa.loc.eval(session=sess)

difficulty = tf.nn.sigmoid(model.qdelta.distribution.loc).eval(session=sess)
if args.plot_scatter:
    fig = vs.plot_parameters(xtest.values[:,:-1], difficulty, discrimination)
    fig.savefig(result_path+'/irt_parameters_vi_'+partial_save_name+'.pdf') 

parameters = pd.DataFrame(index=irt_data.index)
parameters['difficulty'] = difficulty
parameters['discrimination'] = discrimination
parameters.to_csv(result_path+'/irt_parameters_vi_'+partial_save_name+'.csv',index=False)

# visualize correlation between difficulty and response
irt_prob_avg = irt_data.mean(axis=1)
if args.fixed_a:
    fig = vs.plot_item_parameters_corr(irt_prob_avg,difficulty,xtest.noise)
    
else:
    fig = vs.plot_item_parameters_corr(irt_prob_avg,difficulty,xtest.noise,discrimination)

fig.savefig(result_path+'/irt_itemparam_corr_'+partial_save_name+'.pdf')

# output performance of detected noisy points
if not args.fixed_a:
    if args.plot_scatter:
        fig = vs.plot_noisy_points(xtest,discrimination)
        fig.savefig(result_path+'/dnoise_visual_'+partial_save_name+'.pdf')
    #print(xtest.loc[xtest.noise>0].index)
    correct_noise_sum = xtest.loc[discrimination<0,'noise'].sum()
    true_noise_sum = xtest['noise'].sum()
    predict_noise_sum = (discrimination<0).sum()
    if predict_noise_sum < 1:
        print('None noise is found!')
        precision = 0.
    else:   
        precision = 1.*correct_noise_sum/predict_noise_sum

    if true_noise_sum < 1:
        print('None noise is injected!')
        recall = 0.
    else:   
        recall = 1.*correct_noise_sum/true_noise_sum
    print('precision', precision, 'recall',recall)
    with open(result_path+'/dnoise_performance_'+partial_save_name+'.txt', 'w') as pfile:
        pfile.write('precision = '+str(precision)+'\n'+'recall = '+str(recall))








