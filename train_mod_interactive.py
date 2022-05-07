"""
train_mod_interactive.py:  Training of the channel model

This program trains both the link state predictor
and path GAN models from the ray tracing data.  

Example to train for Beijing:
    python train_mod_interactive.py --model_dir models/Beijing_test1 --nepochs_path 10 
"""

import numpy as np
import pickle
import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K
import argparse
from mmwchanmod.learn.models import ChanMod


"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Trains the channel model')
parser.add_argument('--data_dir',action='store',default= 'data/', 
    help='directory to store data')
parser.add_argument('--test_size',action='store',default=0.20,type=int,
    help='size of test set')
parser.add_argument('--nlatent',action='store',default=20,type=int,
    help='number of latent variables')
parser.add_argument('--model_dir',action='store',default= 'models/Beijing', 
    help='directory to store models')
parser.add_argument('--lr_path',action='store',default=1e-4,type=float,
    help='learning rate for the path model') 
parser.add_argument('--nepochs_path',action='store',default=3000,type=int,
    help='number of epochs for training the path model')
parser.add_argument('--batch_size_path',action='store',default=1024,type=float,
    help='batch size for the path model') 
parser.add_argument('--checkpoint_period',action='store',default=500,type=int,
    help='Period in epochs for storing checkpoint. A value of 0 indicates no checkpoints')  

args = parser.parse_args()
data_dir = args.data_dir
test_size = args.test_size
nlatent = args.nlatent
model_dir = args.model_dir
lr_path = args.lr_path
nepochs_path = args.nepochs_path
batch_size_path = args.batch_size_path
checkpoint_period = args.checkpoint_period

"""
Load the data
"""
# Load pre_processed data (.p format)
with open(data_dir+"train_data.p", 'rb') as handle:
    train_d_ = pickle.load(handle)
with open(data_dir+"test_data.p", 'rb') as handle:
    test_d_ = pickle.load(handle)

# nlink_train = len(all_data['gan_scale_data'])
# data = np.reshape(all_data['gan_scale_data'],(nlink,170))
# ncluster_ls = np.vstack((all_data['ncluster_ls'], all_data['ncluster_f1_ls'], all_data['ncluster_f2_ls'])).T
# dvec = raw_data['dvec']
# link_state = raw_data['link_state']
# data = np.append(ncluster_ls, data, axis=1)

# # Train test split
train_data = train_d_
test_data = test_d_

# nts = int(nlink * test_size) # number of test samples
# ntr = nlink - nts # number of train samples
# I = np.random.permutation(nlink)
nlink_train = train_d_['gan_scale_data'].shape[0]
ncluster_ls = np.vstack((train_d_['ncluster_ls'], train_d_['ncluster_f1_ls'], train_d_['ncluster_f2_ls'])).T
train_data['data'] = np.append(ncluster_ls, np.reshape(train_d_['gan_scale_data'], (nlink_train, 180)), axis=1)
train_data['dvec'] = train_d_['dvec']
train_data['link_state'] = train_d_['link_state_f1']

nlink_test = test_d_['gan_scale_data'].shape[0]
ncluster_ls = np.vstack((test_d_['ncluster_ls'], test_d_['ncluster_f1_ls'], test_d_['ncluster_f2_ls'])).T
test_data['data'] = np.append(ncluster_ls, np.reshape(test_d_['gan_scale_data'], (nlink_test, 180)), axis=1)
test_data['dvec'] = test_d_['dvec']
test_data['link_state'] = test_d_['link_state_f1']

# """
# Build and train the model
# """
# # Train the path generator
K.clear_session()

chan_mod = ChanMod(nlatent = nlatent, model_dir = model_dir) 

chan_mod.build_path_mod()

chan_mod.fit_path_mod(train_data, test_data, lr=lr_path,\
                          epochs=nepochs_path,\
                          batch_size=batch_size_path,\
                          checkpoint_period=checkpoint_period)  
    
# Save train and test data
with open(model_dir+'/train_data.p', 'wb') as handle:
    pickle.dump(train_data, handle)

with open(model_dir+'/test_data.p', 'wb') as handle:
    pickle.dump(test_data, handle)