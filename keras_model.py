# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:30:03 2019

@author: nrdas
"""

#This script was made for the purpose of runninng the algorithm with 'tf.keras'. It is still a work in progress.

#import here
import numpy as np
import tensorflow as tf
import os
#from keras.models import load_model

#create a class for building a model.
class Net(tf.keras.Model):
    def __init__(self, input_dimensions, hidden_shape, activation, learning_rate, ckpt_dir):
        '''
        input_dimensions: a tuple containg the 2D dimensions of the input (height, width).
        hidden_shape: an int for the size of the hidden layer.
        activation: the activation function desired for the hidden layers. leaky_relu, relu, or tanh.
        learning rate: the LR desired for the network.
        ckpt_dir: string location of directory to store model checkpoints.
        '''
        super(Net, self).__init__()

        #make the tf session
        self.sess = tf.InteractiveSession()
        
        self.learning_rate = learning_rate
        
        flattened_size = input_dimensions[0] * input_dimensions[1]
                
        #set the input layer of the network. This is a Keras Tensor
        self.input_layer =  tf.keras.Input([flattened_size,])
        
        #This is a placeholder for the actions that were taken
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        
        #A placeholder for the rewards 
        self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')
        
        #decide the activation function
        if activation == 'relu':
            func = tf.nn.relu
        elif activation == 'tanh':
            func = tf.nn.tanh
        else:
            func = tf.nn.leaky_relu

        #make the first layer
        self.layer1 = tf.keras.layers.Dense(
                hidden_shape,
                activation=func)
        
        #pass  input placeholder through the first layer object
        x = self.layer1(self.input_layer)
        
        
        #Make output layer
        self.aprob = tf.keras.layers.Dense(
                1, 
                activation=tf.nn.sigmoid)
        
        #pass intermediate state into output layer to get output
        output = self.aprob(x)
        
        #Match input and output to declare a functioning Keras model.
        self.neural_network = tf.keras.Model(inputs=self.input_layer, outputs=output)
        
        
        #Make a loss with the placeholders and outputs
        self.loss = tf.losses.log_loss(
            labels=self.sampled_actions,
            predictions=output,
            weights=self.reward)
        
        #opt=tf.keras.optimizers.Adam(lr=self.learning_rate)
        #Make an optimizer
        opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        #Use this optimizer to minimize loss
        self.trainer = opt.minimize(self.loss)
        
        #Run
        tf.global_variables_initializer().run()
        
        #Set the checkpoint file name
        self.checkpoint_file = os.path.join(ckpt_dir,
                                            'policy_network.h5')
        
            
    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.neural_network = tf.keras.models.load_model(self.checkpoint_file, compile=False)
        
    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.neural_network.save(self.checkpoint_file, overwrite=True)

    def forward_pass(self, observations): 
        inp = observations.reshape([1, -1])
        out = self.neural_network.predict(inp)
        return out
        
    def train(self, batch_tup):
        print("Training")
        
        #stack up the states, actions, and rewards
        states, actions, rewards = zip(*batch_tup)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        
        #make a dictionary to feed in the placeholders
        feed_dict = {
            self.input_layer: states,
            self.sampled_actions: actions,
            self.reward: rewards
        }
        
        #run the graph
        self.sess.run(self.trainer, feed_dict)
        
    def closeSess(self):
        sess.close()
        
'''
if __name__ == '__main__':
    path = 'C:\\Users\\nrdas\\Downloads\\SADE_AI\\TFRL\\checks'
    model1 = Net((80,80), 200, 'tanh', 0.005, path)
'''    
            
