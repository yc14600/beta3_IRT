from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import edward as ed
import six
import os
import sys
import re
import time

from hsvi.Hierarchi_klqp import Hierarchi_klqp

from edward.models import Normal,Beta,Gamma,TransformedDistribution,InverseGamma,RandomVariable

ds = tf.contrib.distributions

class Beta_IRT:

    def __init__(self,M,C,theta_prior,delta_prior,a_prior):

        self.M = M
        self.C = C
        self.theta_prior = theta_prior # prior of ability
        self.delta_prior = delta_prior # prior of difficulty
        self.a_prior = a_prior  # prior of discrimination
     

        if isinstance(a_prior,ed.RandomVariable):
            # variational posterior of discrimination
            self.qa = Normal(loc=tf.Variable(tf.ones([M])), scale=tf.nn.softplus(tf.Variable(tf.ones([M])*.5)),name='qa')
        else:
            self.qa = a_prior

        with tf.variable_scope('local'):
            # variational posterior of ability
            if isinstance(self.theta_prior,RandomVariable):
                self.qtheta = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([C])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([C])))),\
                                                           bijector=ds.bijectors.Sigmoid(), sample_shape=[M],name='qtheta')
            else: 
                self.qtheta = self.theta_prior
            # variational posterior of difficulty
            self.qdelta = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([M])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([M])))), \
                                                            bijector=ds.bijectors.Sigmoid(), sample_shape=[C],name='qdelta')

        alpha = (tf.transpose(self.qtheta)/self.qdelta)**self.qa

        beta = ((1. - tf.transpose(self.qtheta))/(1. - self.qdelta))**self.qa

        # observed variable
        self.x = Beta(tf.transpose(alpha),tf.transpose(beta))



    def init_inference(self, data, n_iter=1000, n_print=100):
        
        # for discrimination a is latent variable
        if isinstance(self.a_prior,RandomVariable):  
            if isinstance(self.theta_prior, RandomVariable):      
                self.inference = Hierarchi_klqp(latent_vars={self.a_prior:self.qa}, data={self.x:data}, \
                            local_vars={self.theta_prior:self.qtheta,self.delta_prior:self.qdelta},local_data={self.x:data})
            else:
                self.inference = Hierarchi_klqp(latent_vars={self.a_prior:self.qa}, data={self.x:data}, \
                            local_vars={self.delta_prior:self.qdelta},local_data={self.x:data})
        # for discrimination a is constant
        else:      
            self.inference = Hierarchi_klqp(latent_vars={self.theta_prior:self.qtheta,self.delta_prior:self.qdelta},data={self.x:data})
        
        self.inference.initialize(auto_transform=False,n_iter=n_iter,n_print=n_print)

    
    def fit(self,local_iter=50):
        
        tf.global_variables_initializer().run()

        for jj in range(self.inference.n_iter):  
            if isinstance(self.a_prior,ed.RandomVariable):
                for _ in range(local_iter):
                    self.inference.update(scope='local')
            
            info_dict = self.inference.update(scope='global')
            self.inference.print_progress(info_dict)
        

