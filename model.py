# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:30:03 2019

@author: nrdas
"""
#Code without Keras


import numpy as np
import tensorflow as tf
import os

class Net(tf.keras.Model):
    def __init__(self, input_dimensions, use_conv, conv_shape, hidden_shape, activation, learning_rate, ckpt_dir):
        '''
        input_dimensions: a tuple containg the 2D dimensions of the input (height, width).
        use_conv: boolean whether to use convolutional layerrs or not
        conv_shape: tuple for the shape of the convolutional kernel.
        hidden_shape: an int for the size of the hidden layer.
        activation: the activation function desired for the hidden layers. leaky_relu, relu, or tanh.
        learning rate: the LR desired for the network.
        ckpt_dir: string location of directory to store model checkpoints.
        '''
        super(Net, self).__init__()

        self.sess = tf.InteractiveSession()
        
        self.learning_rate = learning_rate
        
        flattened_size = input_dimensions[0] * input_dimensions[1]
        
        if use_conv:
            self.input_layer =  tf.placeholder(tf.float32, [1, input_dimensions[0], input_dimensions[1], 1])
        else:
            self.input_layer =  tf.placeholder(tf.float32, [None, flattened_size])
        
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        
        self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')
        
        #engineer the network
        if activation == 'relu':
            func = tf.nn.relu
        elif activation == 'tanh':
            func = tf.nn.tanh
        else:
            func = tf.nn.leaky_relu
        
        if use_conv:
            
            convolution = tf.layers.conv2d(self.input_layer,
                                           filters=64,
                                           kernel_size=3,
                                           activation=func,
                                           padding='same')
            hidden = tf.layers.dense(convolution,
                                     units=hidden_shape,
                                     activation=func,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.aprob = tf.layers.dense(hidden,
                                         units=1,
                                         activation=tf.sigmoid,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
            
        else:
            
            hidden = tf.layers.dense(self.input_layer,
                                     units=hidden_shape,
                                     activation=func,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.aprob = tf.layers.dense(
                    hidden,
                    units=1,
                    activation=tf.sigmoid,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        

        
        self.loss = tf.losses.log_loss(
            labels=self.sampled_actions,
            predictions=self.aprob,
            weights=self.reward)
            
            
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.loss)
    
        tf.global_variables_initializer().run()
            
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(ckpt_dir,
                                                'policy_network.ckpt')
        
        
    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def forward_pass(self, observations):
        
        up_probability = self.sess.run(
            self.aprob,
            feed_dict={self.input_layer: observations.reshape([1, -1])})
        return up_probability
        
        
    def train(self, batch_tup):
        print("Training")
    
        states, actions, rewards = zip(*batch_tup)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        
        feed_dict = {
            self.input_layer: states,
            self.sampled_actions: actions,
            self.reward: rewards
        }
        
        self.sess.run(self.trainer, feed_dict)
        
    def closeSess():
        self.sess.close()

'''
if __name__ == '__main__':
    path = 'C:\\Users\\nrdas\\Downloads\\SADE_AI\\TFRL\\checks'
    model1 = Net((80,80), False, None, 200, 'tanh', 0.005, path)
'''            
            
