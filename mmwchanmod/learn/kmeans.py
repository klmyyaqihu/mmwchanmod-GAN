"""
KMeans Cluster Class

Using the ray tracing data as the input,

"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import sys

'''
Function of computing rms excess spread
'''
def rms_ex_spread(data, power):
    """
    Input: data: array (npath,)
           data_mean: float
           power: array (npath,) in mW
    Return: rms excess spread
    """
    
    rms_mean = np.sum(power * data) / np.sum(power)
    
    numerator = np.sum(power*(data-rms_mean)**2)
    denominator = np.sum(power)     
    
    return np.sqrt(numerator/denominator)

class KMeansCluster():
    
    def __init__(self, data_f1, data_f2, test_size=0.2, max_ncluster = 10):

        self.max_ncluster = max_ncluster
        self.test_size = test_size
        self.data_f1 = data_f1
        self.data_f2 = data_f2
        self.npath_f1_ls = data_f1['npath']
        self.npath_f2_ls = data_f2['npath']
        
        # conver dBm to mW
        # Power
        power_f1 = np.copy(np.squeeze(data_f1['path_data'][:,:,5])) # (nlink, 20) in dBm
        power_f2 = np.copy(np.squeeze(data_f2['path_data'][:,:,5])) # (nlink, 20) in dBm
        self.power_f1 = 10**(0.1*power_f1) # in mW
        self.power_f2 = 10**(0.1*power_f2) # in mW
        
        self.path_data = np.append(data_f1['path_data'][:,:,:5], 
                              data_f2['path_data'][:,:,:5], axis = 1)
        self.nlink = len(self.path_data)
        
    def fit_KMeans(self):
        # Scale angles to same range
        self.d_scale = np.copy(self.path_data) # copy out, path_data is not scaled
        self.d_scale[:,:,1] = self.d_scale[:,:,1]/180 # inclination
        self.d_scale[:,:,3] = self.d_scale[:,:,3]/180
        self.d_scale[:,:,0] = self.d_scale[:,:,0]/360 # azimuth
        self.d_scale[:,:,2] = self.d_scale[:,:,2]/360

        # Initialize result array nlink*max_npath
        nlink = self.nlink
        max_npath = 20 * 2 # two freq * 20
        self.results = np.zeros([nlink, max_npath]) - 1
        self.ncluster_ls = np.zeros([nlink,], dtype = int)

        # Loop by links
        for ilink in range(nlink):
            if ilink%100 == 0:
                print(f'LINK: {ilink}/{nlink}')
            npath_f1 = int(self.npath_f1_ls[ilink])
            npath_f2 = int(self.npath_f2_ls[ilink])
            
            # if we don't have any path in both freq, go next link
            if (npath_f1 + npath_f2) == 0 : continue 

            # ilink data
            x = np.squeeze(self.d_scale[ilink,:,:]) # 40 * 5 array
            
            # delete empyty paths by npath
            x = np.delete(x, np.where(x[:,4]==0)[0], axis = 0) # dly=0 means no path there
            npath_af_del = len(x)
            # if npath_af_del != (npath_f1 + npath_f2): sys.exit() # check delete
           
            # if npath_af_del <= 2 we don't need to use the K-Means
            if npath_af_del <= 2:
        #TODO        # Put them in the same cluster: label 0 ?
                self.results[ilink, :npath_af_del] = 0 # label 0
            else: # npath_af_del >= 3 => K-Means
                # scale the dly by MinMaxScaler
                x[:,4] = (x[:,4] - np.min(x[:,4])) / (np.max(x[:,4]) - np.min(x[:,4]))
                # '''
                # Silhouette Score
                # '''
                # K-Means params: at most the number of npath_af_del/2 clusters or 10
                kmax = min(int(np.ceil(npath_af_del/2)), 10)
                K = range(2, kmax+1) # at least 2 clusters 
                # Dissimilarity would not be defined for a single cluster,
                # thus, minimum number of clusters should be 2
                sil = [] # reset the sil ls
                for k in K:
                  kmeans = KMeans(n_clusters = k).fit(x) # fit K-Means
                  labels = kmeans.labels_
                  sil.append(silhouette_score(x, labels, metric = 'euclidean'))  
                # Select the best num of cluster
                n_cluster = K[np.argmax(sil)]
                self.ncluster_ls[ilink] = n_cluster
                kmeans_model = KMeans(n_clusters = n_cluster)
                labels = kmeans_model.fit_predict(x)
                self.results[ilink, :npath_f1] = labels[:npath_f1]
                self.results[ilink, 20:(20+npath_f2)] = labels[npath_f1:]
    
    
    def store_KMeans_results(self):
        n_freq = 2
        self.gan_data = np.zeros((self.nlink, self.max_ncluster, 6+6*n_freq))-1
        self.ncluster_f1 = np.zeros([self.nlink,], dtype=(int))
        self.ncluster_f2 = np.zeros([self.nlink,], dtype=(int))
        
        d_scale = self.d_scale
        power_f1 = self.power_f1
        power_f2 = self.power_f2
        
        print("Save Data ++++++++++++++++++++++++++++++++")
        for ilink in range(self.nlink):
            if ilink%100 == 0:
                print(f'LINK: {ilink}/{self.nlink}')
            i_ncluster = self.ncluster_ls[ilink] # number of cluster in ilink
            
            if i_ncluster == 0: continue # outage
            
            d_ = np.zeros((i_ncluster, 6+6*n_freq))-1
            for icluster in range(i_ncluster):
                '''
                Fisrt five params
                0: az-aod, 
                1: inc-aod, 
                2: az-aoa, 
                3: inc-aoa,
                4: min_dly
                '''
                I = np.where(self.results[ilink,:]==icluster)[0] # take paths belong to clus
                d_[icluster,:4] = np.mean(d_scale[ilink, I, :4], axis=0)
                d_[icluster,4] = np.min(d_scale[ilink, I, 4])
                
                '''
                Six params for each freq (f1: 5-10; f2: 11-16)
                5/11: az-aod (scale back to -180~180) rms spread
                6/12: inc-aod (scale back to 0~180) rms spread
                7/13: az-aoa (scale back to -180~180) rms spread
                8/14: inc-aoa (scale back to 0~180) rms spread
                9/15: (excess LOS) delay rms spread 
                10/16: total power in mW
                '''
                I_f1 = np.where(self.results[ilink,:20]==icluster)[0]
                if len(I_f1)>0:
                    self.ncluster_f1[ilink] += 1
                    d_[icluster,5] = rms_ex_spread(d_scale[ilink, I_f1, 0] * 360, 
                                                            power_f1[ilink, I_f1])
                    d_[icluster,6] = rms_ex_spread(d_scale[ilink, I_f1, 1] * 180, 
                                                            power_f1[ilink, I_f1])
                    d_[icluster,7] = rms_ex_spread(d_scale[ilink, I_f1, 2] * 360, 
                                                            power_f1[ilink, I_f1])
                    d_[icluster,8] = rms_ex_spread(d_scale[ilink, I_f1, 3] * 180, 
                                                            power_f1[ilink, I_f1])
                    d_[icluster,9] = rms_ex_spread(d_scale[ilink, I_f1, 4], 
                                                            power_f1[ilink, I_f1])
                    d_[icluster,10] = np.sum(power_f1[ilink, I_f1]) # linear
         
                I_f2 = np.where(self.results[ilink,20:]==icluster)[0]+20
                if len(I_f2)>0:
                    self.ncluster_f2[ilink] += 1
                    d_[icluster,11] = rms_ex_spread(d_scale[ilink, I_f2, 0] * 360, 
                                                            power_f2[ilink, I_f2-20])
                    d_[icluster,12] = rms_ex_spread(d_scale[ilink, I_f2, 1] * 180, 
                                                            power_f2[ilink, I_f2-20])
                    d_[icluster,13] = rms_ex_spread(d_scale[ilink, I_f2, 2] * 360, 
                                                            power_f2[ilink, I_f2-20])
                    d_[icluster,14] = rms_ex_spread(d_scale[ilink, I_f2, 3] * 180, 
                                                            power_f2[ilink, I_f2-20])
                    d_[icluster,15] = rms_ex_spread(d_scale[ilink, I_f2, 4], 
                                                            power_f2[ilink, I_f2-20])
                    d_[icluster,16] = np.sum(power_f2[ilink, I_f2-20]) # linear
                
                # difference of total power in 28 GHz - 140 GHz
                if len(I_f1)>0 and len(I_f2)>0: # both freqshs have paths
                    d_[icluster, 17] = 10*np.log10(d_[icluster,10]) - 10*np.log10(d_[icluster,16]) 
                                        # dBm - dBm
                    
            # Sort by the freq1 clusters' total power
            self.gan_data[ilink,:i_ncluster,:] = d_[d_[:,10].argsort()[::-1]]
            
            
    def scale_data_for_GAN(self):
    
        gan_d = np.copy(self.gan_data)
        
        nlink = self.nlink
        ncluster_ls = self.ncluster_ls
        ncluster_f1_ls = self.ncluster_f1
        ncluster_f2_ls = self.ncluster_f2
        
        nts = int(nlink * self.test_size) # number of test samples
        ntr = nlink - nts # number of train samples
        I_random = np.random.permutation(nlink)
        I_train = I_random[:ntr]
        I_test = I_random[ntr:]
        
        """
        Scale min_delay [:,:,4]
        Excess LOS min dly and then divide by mean 
        """
        dvec = self.data_f1['dvec']
        dist = np.maximum(np.sqrt(np.sum(dvec**2, axis = 1)), 0.01) 
        light_sp = 2.996955055081703e8 # remcom 
        los_dly = dist/light_sp
        store_dly_ls = []
        for ilink in range(nlink):
            icluster = ncluster_ls[ilink]
            if icluster == 0: continue
            # Excess min_delay
            gan_d[ilink,:icluster,4] = np.maximum(
                (gan_d[ilink,:icluster,4] - los_dly[ilink]), 0.0)
            if ilink in I_train:
                store_dly_ls.extend(gan_d[ilink,:icluster,4])

        # Get mean of store_excess_dly
        ex_dly_mean = np.mean(store_dly_ls)
        for ilink in range(nlink): # divide by mean
            icluster = ncluster_ls[ilink]
            if icluster == 0: continue
            gan_d[ilink,:icluster,4] = gan_d[ilink,:icluster,4] / ex_dly_mean
            
        """
        Scale Difference of Total Power [:,:,17] 
        This item is in dBm
        Get Min/Max for two freq and then use MaxMinScaler
        """
        store_diff_power_ls = []
        for ilink in range(nlink):
            icluster = ncluster_ls[ilink]
            if icluster == 0: continue
            if ilink in I_train:
                I = np.where(gan_d[ilink,:,17] != -1)[0]
                store_diff_power_ls.extend(gan_d[ilink,I,17])

        # min and max in dBm
        max_diff_power = np.max(store_diff_power_ls)
        min_diff_power = np.min(store_diff_power_ls)

        for ilink in range(nlink):
            icluster = ncluster_ls[ilink]
            if icluster == 0: continue
            I = np.where(gan_d[ilink,:,17] != -1)[0]
            gan_d[ilink,I,17] = (
                (gan_d[ilink,I,17] - min_diff_power)/
                (max_diff_power - min_diff_power))
        

        """
        Scale Total Power [:,:,10] and [:,:,16]
        Covert to dBm first
        Get Min/Max for two freq and then use MaxMinScaler
        """
        store_f1_power_ls = []
        store_f2_power_ls = []
        for ilink in range(nlink):
            icluster = ncluster_ls[ilink]
            icluster_f1 = ncluster_f1_ls[ilink]
            icluster_f2 = ncluster_f2_ls[ilink]
            if icluster == 0: continue
            # linear to dBm
            if icluster_f1 > 0:
                gan_d[ilink,:icluster_f1,10] = 10*np.log10(
                    gan_d[ilink,:icluster_f1,10])
                if ilink in I_train:
                    store_f1_power_ls.extend(gan_d[ilink,:icluster_f1,10]) # f1
            if icluster_f2 > 0:
                # maybe discrete for example c1 and c3 have but no path in c2
                # find the index that total power != -1
                I = np.where(gan_d[ilink,:,16] != -1)[0]
                gan_d[ilink,I,16] = 10*np.log10(gan_d[ilink,I,16])
                if ilink in I_train:
                    store_f2_power_ls.extend(gan_d[ilink,I,16]) # f2

        # freq 1 min and max
        min_power_f1 = np.min(store_f1_power_ls)
        max_power_f1 = np.max(store_f1_power_ls)
        # freq 2 min and max
        min_power_f2 = np.min(store_f2_power_ls)
        max_power_f2 = np.max(store_f2_power_ls)

        for ilink in range(nlink):
            icluster = ncluster_ls[ilink]
            icluster_f1 = ncluster_f1_ls[ilink]
            icluster_f2 = ncluster_f2_ls[ilink]
            if icluster == 0: continue
            if icluster_f1 > 0:
                gan_d[ilink,:icluster_f1,10] = (
                    (gan_d[ilink,:icluster_f1,10] - min_power_f1)/
                    (max_power_f1 - min_power_f1)) 
            if icluster_f2 > 0:
                # maybe discrete for example c1 and c3 have but no path in c2
                # find the index that total power != -1
                I = np.where(gan_d[ilink,:,16] != -1)[0]
                gan_d[ilink,I,16] = (
                    (gan_d[ilink,I,16] - min_power_f2)/
                    (max_power_f2 - min_power_f2))


        """
        Scale RMS dly [:,:,9] and [:,:,15]
        Get Min/Max for two freq and then use MaxMinScaler
        """
        store_f1_9_ls = []
        store_f2_15_ls = []
        for ilink in range(nlink):
            if ilink in I_train:
                icluster = ncluster_ls[ilink]
                icluster_f1 = ncluster_f1_ls[ilink]
                icluster_f2 = ncluster_f2_ls[ilink]
                if icluster == 0: continue
                if icluster_f1 > 0:
                    store_f1_9_ls.extend(gan_d[ilink,:icluster_f1,9]) # f1
                if icluster_f2 > 0:
                    # maybe discrete for example c1 and c3 have but no path in c2
                    # find the index that total power != -1
                    I = np.where(gan_d[ilink,:,15] != -1)[0]
                    store_f2_15_ls.extend(gan_d[ilink,I,15]) # f2

        # freq 1 min and max
        min_9_f1 = np.min(store_f1_9_ls)
        max_9_f1 = np.max(store_f1_9_ls)
        # freq 2 min and max
        min_15_f2 = np.min(store_f2_15_ls)
        max_15_f2 = np.max(store_f2_15_ls)

        for ilink in range(nlink):
            icluster = ncluster_ls[ilink]
            icluster_f1 = ncluster_f1_ls[ilink]
            icluster_f2 = ncluster_f2_ls[ilink]
            if icluster == 0: continue
            if icluster_f1 > 0:
                gan_d[ilink,:icluster_f1,9] = (
                    (gan_d[ilink,:icluster_f1,9] - min_9_f1)/
                    (max_9_f1 - min_9_f1)) 
            if icluster_f2 > 0:
                # maybe discrete for example c1 and c3 have but no path in c2
                # find the index that total power != -1
                I = np.where(gan_d[ilink,:,15] != -1)[0]
                gan_d[ilink,I,15] = (
                    (gan_d[ilink,I,15] - min_15_f2)/
                    (max_15_f2 - min_15_f2))

        """
        Scale RMS spread AoD Az [:,:,5] and [:,:,11]
        Get Min/Max for two freq and then use MaxMinScaler
        """
        store_f1_5_ls = []
        store_f2_11_ls = []
        for ilink in range(nlink):
            if ilink in I_train:
                icluster = ncluster_ls[ilink]
                icluster_f1 = ncluster_f1_ls[ilink]
                icluster_f2 = ncluster_f2_ls[ilink]
                if icluster == 0: continue
                if icluster_f1 > 0:
                    store_f1_5_ls.extend(gan_d[ilink,:icluster_f1,5]) # f1
                if icluster_f2 > 0:
                    # maybe discrete for example c1 and c3 have but no path in c2
                    # find the index that total power != -1
                    I = np.where(gan_d[ilink,:,11] != -1)[0]
                    store_f2_11_ls.extend(gan_d[ilink,I,11]) # f2

        # freq 1 min and max
        min_5_f1 = np.min(store_f1_5_ls)
        max_5_f1 = np.max(store_f1_5_ls)
        # freq 2 min and max
        min_11_f2 = np.min(store_f2_11_ls)
        max_11_f2 = np.max(store_f2_11_ls)

        for ilink in range(nlink):
            icluster = ncluster_ls[ilink]
            icluster_f1 = ncluster_f1_ls[ilink]
            icluster_f2 = ncluster_f2_ls[ilink]
            if icluster == 0: continue
            if icluster_f1 > 0:
                gan_d[ilink,:icluster_f1,5] = (
                    (gan_d[ilink,:icluster_f1,5] - min_5_f1)/
                    (max_5_f1 - min_5_f1)) 
            if icluster_f2 > 0:
                # maybe discrete for example c1 and c3 have but no path in c2
                # find the index that total power != -1
                I = np.where(gan_d[ilink,:,11] != -1)[0]
                gan_d[ilink,I,11] = (
                    (gan_d[ilink,I,11] - min_11_f2)/
                    (max_11_f2 - min_11_f2))

        """
        Scale RMS spread AoD Inc [:,:,6] and [:,:,12]
        Covert to dBm first
        Get Min/Max for two freq and then use MaxMinScaler
        """
        store_f1_6_ls = []
        store_f2_12_ls = []
        for ilink in range(nlink):
            if ilink in I_train:
                icluster = ncluster_ls[ilink]
                icluster_f1 = ncluster_f1_ls[ilink]
                icluster_f2 = ncluster_f2_ls[ilink]
                if icluster == 0: continue
                if icluster_f1 > 0:
                    store_f1_6_ls.extend(gan_d[ilink,:icluster_f1,6]) # f1
                if icluster_f2 > 0:
                    # maybe discrete for example c1 and c3 have but no path in c2
                    # find the index that total power != -1
                    I = np.where(gan_d[ilink,:,12] != -1)[0]
                    store_f2_12_ls.extend(gan_d[ilink,I,12]) # f2

        # freq 1 min and max
        min_6_f1 = np.min(store_f1_6_ls)
        max_6_f1 = np.max(store_f1_6_ls)
        # freq 2 min and max
        min_12_f2 = np.min(store_f2_12_ls)
        max_12_f2 = np.max(store_f2_12_ls)

        for ilink in range(nlink):
            icluster = ncluster_ls[ilink]
            icluster_f1 = ncluster_f1_ls[ilink]
            icluster_f2 = ncluster_f2_ls[ilink]
            if icluster == 0: continue
            if icluster_f1 > 0:
                gan_d[ilink,:icluster_f1,6] = (
                    (gan_d[ilink,:icluster_f1,6] - min_6_f1)/
                    (max_6_f1 - min_6_f1)) 
            if icluster_f2 > 0:
                # maybe discrete for example c1 and c3 have but no path in c2
                # find the index that total power != -1
                I = np.where(gan_d[ilink,:,12] != -1)[0]
                gan_d[ilink,I,12] = (
                    (gan_d[ilink,I,12] - min_12_f2)/
                    (max_12_f2 - min_12_f2))
                
                
        """
        Scale RMS spread AoA Az [:,:,7] and [:,:,13]
        Covert to dBm first
        Get Min/Max for two freq and then use MaxMinScaler
        """
        store_f1_7_ls = []
        store_f2_13_ls = []
        for ilink in range(nlink):
            if ilink in I_train:
                icluster = ncluster_ls[ilink]
                icluster_f1 = ncluster_f1_ls[ilink]
                icluster_f2 = ncluster_f2_ls[ilink]
                if icluster == 0: continue
                if icluster_f1 > 0:
                    store_f1_7_ls.extend(gan_d[ilink,:icluster_f1,7]) # f1
                if icluster_f2 > 0:
                    # maybe discrete for example c1 and c3 have but no path in c2
                    # find the index that total power != -1
                    I = np.where(gan_d[ilink,:,13] != -1)[0]
                    store_f2_13_ls.extend(gan_d[ilink,I,13]) # f2

        # freq 1 min and max
        min_7_f1 = np.min(store_f1_7_ls)
        max_7_f1 = np.max(store_f1_7_ls)
        # freq 2 min and max
        min_13_f2 = np.min(store_f2_13_ls)
        max_13_f2 = np.max(store_f2_13_ls)

        for ilink in range(nlink):
            icluster = ncluster_ls[ilink]
            icluster_f1 = ncluster_f1_ls[ilink]
            icluster_f2 = ncluster_f2_ls[ilink]
            if icluster == 0: continue
            # linear to dBm
            if icluster_f1 > 0:
                gan_d[ilink,:icluster_f1,7] = (
                    (gan_d[ilink,:icluster_f1,7] - min_7_f1)/
                    (max_7_f1 - min_7_f1)) 
            if icluster_f2 > 0:
                # maybe discrete for example c1 and c3 have but no path in c2
                # find the index that total power != -1
                I = np.where(gan_d[ilink,:,13] != -1)[0]
                gan_d[ilink,I,13] = (
                    (gan_d[ilink,I,13] - min_13_f2)/
                    (max_13_f2 - min_13_f2))

        """
        Scale RMS spread AoA Inc [:,:,8] and [:,:,14]
        Covert to dBm first
        Get Min/Max for two freq and then use MaxMinScaler
        """
        store_f1_8_ls = []
        store_f2_14_ls = []
        for ilink in range(nlink):
            if ilink in I_train:
                icluster = ncluster_ls[ilink]
                icluster_f1 = ncluster_f1_ls[ilink]
                icluster_f2 = ncluster_f2_ls[ilink]
                if icluster == 0: continue
                if icluster_f1 > 0:
                    store_f1_8_ls.extend(gan_d[ilink,:icluster_f1,8]) # f1
                if icluster_f2 > 0:
                    # maybe discrete for example c1 and c3 have but no path in c2
                    # find the index that total power != -1
                    I = np.where(gan_d[ilink,:,14] != -1)[0]
                    store_f2_14_ls.extend(gan_d[ilink,I,14]) # f2

        # freq 1 min and max
        min_8_f1 = np.min(store_f1_8_ls)
        max_8_f1 = np.max(store_f1_8_ls)
        # freq 2 min and max
        min_14_f2 = np.min(store_f2_14_ls)
        max_14_f2 = np.max(store_f2_14_ls)

        for ilink in range(nlink):
            icluster = ncluster_ls[ilink]
            icluster_f1 = ncluster_f1_ls[ilink]
            icluster_f2 = ncluster_f2_ls[ilink]
            if icluster == 0: continue
            # linear to dBm
            if icluster_f1 > 0:
                gan_d[ilink,:icluster_f1,8] = (
                    (gan_d[ilink,:icluster_f1,8] - min_8_f1)/
                    (max_8_f1 - min_8_f1)) 
            if icluster_f2 > 0:
                # maybe discrete for example c1 and c3 have but no path in c2
                # find the index that total power != -1
                I = np.where(gan_d[ilink,:,14] != -1)[0]
                gan_d[ilink,I,14] = (
                    (gan_d[ilink,I,14] - min_14_f2)/
                    (max_14_f2 - min_14_f2))
        
        train_data = dict(dvec = dvec[I_train],
                fspl_f1 = self.data_f1['fspl'][I_train],
                fspl_f2 = self.data_f2['fspl'][I_train],
                npath_f1 = self.data_f1['npath'][I_train],
                npath_f2 = self.data_f2['npath'][I_train],
                ncluster_ls = ncluster_ls[I_train], 
                link_state_f1 = self.data_f1['link_state'][I_train],
                link_state_f2 = self.data_f2['link_state'][I_train],
                gan_scale_data = gan_d[I_train],
                ncluster_f1_ls = ncluster_f1_ls[I_train], 
                ncluster_f2_ls = ncluster_f2_ls[I_train],
                ex_dly_mean = ex_dly_mean,
                los_dly = los_dly[I_train],
                max_ls = [max_5_f1, max_6_f1, max_7_f1, max_8_f1,
                          max_9_f1, max_power_f1,
                          max_11_f2, max_12_f2, max_13_f2, max_14_f2,
                          max_15_f2, max_power_f2, max_diff_power],
                min_ls = [min_5_f1, min_6_f1, min_7_f1, min_8_f1,
                            min_9_f1, min_power_f1,
                            min_11_f2, min_12_f2, min_13_f2, min_14_f2,
                            min_15_f2, min_power_f2, min_diff_power],
                light_speed = light_sp)
        
        test_data = dict(dvec = dvec[I_test],
                fspl_f1 = self.data_f1['fspl'][I_test],
                fspl_f2 = self.data_f2['fspl'][I_test],
                npath_f1 = self.data_f1['npath'][I_test],
                npath_f2 = self.data_f2['npath'][I_test],
                ncluster_ls = ncluster_ls[I_test], 
                link_state_f1 = self.data_f1['link_state'][I_test],
                link_state_f2 = self.data_f2['link_state'][I_test],
                gan_scale_data = gan_d[I_test],
                ncluster_f1_ls = ncluster_f1_ls[I_test], 
                ncluster_f2_ls = ncluster_f2_ls[I_test],
                ex_dly_mean = ex_dly_mean,
                los_dly = los_dly[I_test],
                max_ls = [max_5_f1, max_6_f1, max_7_f1, max_8_f1,
                          max_9_f1, max_power_f1,
                          max_11_f2, max_12_f2, max_13_f2, max_14_f2,
                          max_15_f2, max_power_f2, max_diff_power],
                min_ls = [min_5_f1, min_6_f1, min_7_f1, min_8_f1,
                            min_9_f1, min_power_f1,
                            min_11_f2, min_12_f2, min_13_f2, min_14_f2,
                            min_15_f2, min_power_f2, min_diff_power],
                light_speed = light_sp)
        
        return train_data, test_data
    
    
    