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
            with tf.variable_scope('global'):
                self.qa = Normal(loc=tf.Variable(tf.ones([M])), scale=tf.nn.softplus(tf.Variable(tf.ones([M])*.5)),name='qa')
        else:
            self.qa = a_prior

        with tf.variable_scope('local'):
            # variational posterior of ability
            if isinstance(self.theta_prior,RandomVariable):
                self.qtheta = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([C],stddev=0.01)), scale=tf.nn.softplus(tf.Variable(tf.random_normal([C],stddev=0.01)))),\
                                                           bijector=ds.bijectors.Sigmoid(), sample_shape=[M],name='qtheta')
            else: 
                self.qtheta = self.theta_prior
            # variational posterior of difficulty
            self.qdelta = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([M],stddev=0.01)), scale=tf.nn.softplus(tf.Variable(tf.random_normal([M],stddev=0.01)))), \
                                                            bijector=ds.bijectors.Sigmoid(), sample_shape=[C],name='qdelta')

        alpha = (tf.transpose(self.qtheta)/self.qdelta)**self.qa

        beta = ((1. - tf.transpose(self.qtheta))/(1. - self.qdelta))**self.qa

        # observed variable
        self.x = Beta(tf.transpose(alpha),tf.transpose(beta))



    def init_inference(self, data):
        
        # for discrimination a is latent variable
        if isinstance(self.a_prior,RandomVariable):  
            if isinstance(self.theta_prior, RandomVariable):      
                self.inference = Hierarchy_SVI(latent_vars={'global':{self.a_prior:self.qa},'local':{self.theta_prior:self.qtheta,self.delta_prior:self.qdelta}}, data={'global':{self.x:data},'local':{self.x:data}})
            else:
                self.inference = Hierarchy_SVI(latent_vars={'global':{self.a_prior:self.qa},'local':{self.delta_prior:self.qdelta}}, data={'global':{self.x:data},'local':{self.x:data}})
        # for discrimination a is constant
        else:      
            self.inference = Hierarchy_SVI(latent_vars={'local':{self.theta_prior:self.qtheta,self.delta_prior:self.qdelta}},data={'local':{self.x:data}})
        
    
    def fit(self,n_iter=1000,local_iter=10,sess=None):
        if sess is None:
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        tf.global_variables_initializer().run(session=sess)


        for jj in range(n_iter):  
            if isinstance(self.a_prior,RandomVariable):
                for _ in range(local_iter):
                    self.inference.update(scope='local',sess=sess)
            
            info_dict = self.inference.update(scope='global',sess=sess)

            if (jj+1)%100==0 or jj==0:
                print(info_dict['loss'])
            
        

