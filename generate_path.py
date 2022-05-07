"""
generate_path.py: Genereate paths from the pre-trained model

Example to generate data:
    python generate_path.py --model_dir models/Beijing_test1 --result_dir results --data_fn gen_path.p
"""
import pickle
import os
import argparse

import tensorflow.keras.backend as K
    
from mmwchanmod.learn.models import load_model 

    
"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Generate paths from pre-trained model')    
parser.add_argument('--model_dir',action='store',default= 'models/Beijing', 
    help='directory to store models')
parser.add_argument(\
    '--result_dir',action='store',\
    default='results', help='directory for the generated data')    
parser.add_argument(\
    '--data_fn',action='store',\
    default='gen_path.p', help='generated data file name')        
    
args = parser.parse_args()
model_dir = args.model_dir
model_name = model_dir.split('/')[-1]
result_dir = args.result_dir
data_fn = args.data_fn

"""
load data
"""
# Load test data (.p format)
with open(model_dir+'/test_data.p', 'rb') as handle:
    test_data = pickle.load(handle)

"""
Load the pre-trained model
"""
# Construct and load the channel model object
print('Loading pre-trained model %s' % model_name)
K.clear_session()
chan_mod = load_model(model_name)
    
"""
Generate data
"""
X = chan_mod.sample_path(test_data['dvec'])

if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    print('Created directory %s' % result_dir)
with open(os.path.join(result_dir, data_fn), 'wb') as handle:
    pickle.dump(X, handle)