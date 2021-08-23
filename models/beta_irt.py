from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


from hsvi.tensorflow import Hierarchy_SVI

from hsvi.tensorflow.distributions import Normal,Beta,TransformedDistribution,RandomVariable

ds = tf.contrib.distributions

class Beta_IRT:

    def __init__(self,M,C,theta_prior,delta_prior,a_prior):

        self.M = M
        self.C = C
        self.theta_prior = theta_prior # prior of ability
        self.delta_prior = delta_prior # prior of difficulty
        self.a_prior = a_prior  # prior of discrimination
     

        if isinstance(a_prior,RandomVariable):
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
                self.inference = Hierarchy_SVI(latent_vars={self.a_prior:self.qa}, data={self.x:data}, \
                            local_vars={self.theta_prior:self.qtheta,self.delta_prior:self.qdelta},local_data={self.x:data})
            else:
                self.inference = Hierarchy_SVI(latent_vars={self.a_prior:self.qa}, data={self.x:data}, \
                            local_vars={self.delta_prior:self.qdelta},local_data={self.x:data})
        # for discrimination a is constant
        else:      
            self.inference = Hierarchy_SVI(latent_vars={self.theta_prior:self.qtheta,self.delta_prior:self.qdelta},data={self.x:data})
        
    
    def fit(self,local_iter=50):
        
        tf.global_variables_initializer().run()

        for jj in range(self.inference.n_iter):  
            if isinstance(self.a_prior,RandomVariable):
                for _ in range(local_iter):
                    self.inference.update(scope='local')
            
            info_dict = self.inference.update(scope='global')
            
        

