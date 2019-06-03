from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import six
import os

#from edward.util import check_data, check_latent_vars
from edward.inferences import Inference


class Hierarchi_klqp(Inference):
    def __init__(self,latent_vars={},data={},local_data={},local_vars=None,*args,**kwargs): 

        
        super(Hierarchi_klqp,self).__init__(*args,**kwargs)
        self.latent_vars = latent_vars
        self.data = data

        self.local_vars = local_vars
        self.local_data = local_data
        
        
    def initialize(self,scale={},optimizer={}, clipping={}, constraints=None, *args, **kwargs):
        self.scale = scale
        self.optimizer = optimizer   
        self.clipping = clipping     
        self.constraints = constraints

        var_list = set()
        for v in tf.trainable_variables():
            var_list.add(v) 
        if not self.local_vars is None:
            local_var_list = set()
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="local"):
                local_var_list.add(v)
            var_list.difference_update(local_var_list)
            local_var_list = list(local_var_list)
        else:
            local_var_list = None
            
        var_list = list(var_list)
        self.loss, grads_and_vars, self.local_loss, grads_and_vars_local = self.build_loss_and_gradients(var_list,local_var_list)
            
        self.config_optimizer()

        self.train = self.optimizer['global'][0].apply_gradients(grads_and_vars,global_step=self.optimizer['global'][1])
        
        if not local_var_list is None:
            self.config_optimizer(scope='local')
            self.train_local = self.optimizer['local'][0].apply_gradients(grads_and_vars_local, global_step=self.optimizer['local'][1])
        
        super(Hierarchi_klqp,self).initialize(*args,**kwargs)


    def config_optimizer(self,scope='global'):
        # Need default optimizer
        if not scope in self.optimizer:
            if scope=='local':
                decay = (10000,0.999) 
                with tf.variable_scope('local'):
                    global_step = tf.Variable(0, trainable=False, name="local_step")
            else:
                decay = (100,0.9)
                global_step = tf.Variable(0, trainable=False, name="global_step")

            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                        global_step,
                                                        decay[0], decay[1], staircase=True)
            self.optimizer[scope] = (tf.train.AdamOptimizer(learning_rate),global_step)

        # Need default global_step for train
        elif len(self.optimizer[scope])==1:
            if scope=='local':
                with tf.variable_scope('local'):
                    global_step = tf.Variable(0, trainable=False, name="local_step")
            else:
                global_step = tf.Variable(np.int64(0), trainable=False, name="global_step")

            self.optimizer[scope].append(global_step)

        return
        
        
    def build_loss_and_gradients(self,var_list,local_var_list):
    
        loss, grads_and_vars = self.build_reparam_ELBO_and_grads(var_list)

        if not local_var_list is None:
            local_loss, grads_and_vars_local = self.build_reparam_ELBO_and_grads(local_var_list,scope='local')
        else:
            local_loss = 0. 
            grads_and_vars_local = None
             
        return loss, grads_and_vars, local_loss, grads_and_vars_local


    def build_reparam_ELBO_and_grads(self,var_list,scope='global'):
            ll = 0.
            kl = 0.
            if scope == 'global':
                data = self.data
                vars = self.latent_vars
            else:
                data = self.local_data
                vars = self.local_vars

            for x, qx in six.iteritems(data):
                ll += tf.reduce_mean(self.scale.get(x,1.)*x.log_prob(qx))
            
            for z,qz in six.iteritems(vars):
                kl += tf.reduce_mean(qz.log_prob(qz))-tf.reduce_mean(z.log_prob(qz))

            if not self.constraints is None:
                closs = 0.
                for qz in six.iterkeys(self.constraints):
                    if qz in self.vars.values():
                        closs +=  self.constraints[qz]
                        #print(closs)
                kl += closs
            
            loss = kl - ll 
            grads = tf.gradients(loss, var_list)

            if scope in self.clipping:
                grads = [tf.clip_by_value(grd,self.clipping[scope][0],self.clipping[scope][1]) for grd in grads]

            grads_and_vars = list(zip(grads, var_list))
            
            return loss, grads_and_vars
        
    def update(self,feed_dict=None,scope='global'):
        if feed_dict is None:
          feed_dict = {}

        for key, value in six.iteritems(self.local_data):
          if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
            feed_dict[key] = value
        
        sess = ed.get_session()
        
        if scope=='global':
            _,t, loss = sess.run([self.train, self.increment_t, self.loss], feed_dict)
            return {'t':t,'loss':loss}

        if scope=='local':
            _,local_loss = sess.run([self.train_local,self.local_loss], feed_dict)
            return {'t':self.increment_t,'loss':local_loss}

    def print_progress(self, info_dict):
        """Print progress to output."""
        if self.n_print != 0:
            t = info_dict['t']
            if t == 1 or t % self.n_print == 0:
                self.progbar.update(t, {'Loss': info_dict['loss']})

    
    
    
