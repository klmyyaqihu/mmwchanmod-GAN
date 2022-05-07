"""
Test the function of the mmwchanmod.learn.kmeans.py

The results is used to train and test the GAN,
train.p and test.p

"""

from mmwchanmod.learn.kmeans import KMeansCluster
import pickle

data_28_dir = "./data/dict_scale_whole_28.p"
data_140_dir = "./data/dict_scale_whole_140.p"

# Load the scaled ray tracing data dictionarys
with open(data_28_dir, 'rb') as file:
    data_28 = pickle.load(file)
with open(data_140_dir, 'rb') as file:
    data_140 = pickle.load(file)
    
kmeans_class = KMeansCluster(data_28, data_140)
kmeans_class.fit_KMeans()
kmeans_class.store_KMeans_results()
train_data, test_data = kmeans_class.scale_data_for_GAN() # two te
with open ("./data/train_data.p", "wb") as file:
    pickle.dump(train_data, file)
with open ("./data/test_data.p", "wb") as file:
    pickle.dump(test_data, file)
