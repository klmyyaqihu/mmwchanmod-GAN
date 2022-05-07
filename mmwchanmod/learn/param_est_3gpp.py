"""
param_est_3gpp.py:  classes for parameterizing 3gpp model

Yaqi Hu, Mingsheng Yin 
Modified for Multi-frequency channel modeling

"""

import tensorflow as tf
import numpy as np
import pickle
from datetime import date 

from mmwchanmod.common.constants import LinkState, PhyConst
from mmwchanmod.learn.datastats import data_to_mpchan
from mmwchanmod.common.constants import DataConfig

class ParamEst3gppUAS(object):
    def __init__(self, param, fc, data_dir, cell = 1, hbs = 2, use_outage=True):
        """
        3gpp UAS parameter modeling class

        Parameters
        ----------
        param : str
            parameter to estimate. Options:
                - pLOS
        data_dir : str
            data_dir of data to use for re-parameterization
        cell : int
            gNB type to use:
                - 0 for aerial
                - 1 for terrestrial
        hbs : int
            the height of the BS in the scenario
        """
        self.param = param
        self.data_dir = data_dir
        self.rx_type = cell
        self.hbs = hbs
        self.use_outage = use_outage
        #self.linkState
        
        self.fc = fc # recording freq
        self.dat_tr = None
        self.dvec = None
        self.dx_tr = None
        self.dz_tr = None
        self.hut = None
        self.xtr, self.ytr = None, None
        self.app_range = None
        self.ind = None
        self.cfg = None
        self.xtr_pl, self.ytr_pl = None, None

    def load_data(self, max_ex_pl = 80, tx_pow_dbm = 23, npaths_max = 20,\
                    freq_set = [2.3e9,28e9]):
        """
        Loads the data set and assigns it to object variables.

        Returns
        -------
        None.

        """
        # add
        # Load pre_processed data (.p format)
        with open(self.data_dir, 'rb') as handle:
            data = pickle.load(handle)
            print(f"load model from: {self.data_dir}")

        # set cfg
        cfg = DataConfig()
        cfg.date_created = date.today().strftime("%d-%m-%Y")  
        cfg.max_ex_pl = max_ex_pl
        cfg.tx_pow_dbm = tx_pow_dbm
        cfg.npaths_max = npaths_max
        cfg.freq_set = freq_set
        cfg.nfreq = len(freq_set)
        self.cfg = cfg

        # set data
        nlink = data['dvec'].shape[0]
        data['rx_type'] = np.ones((nlink,))
        self.dat_tr = data
        self.dvec = self.dat_tr['dvec']

    def transform_data(self, dvec):
        """
        Format the training data xtrain = [log10(h_UT), d2D, h_UT, h_BS]
        
        h_UT = height of user
        h_BS = height of BS
        """
        # Get horizontal and vertical distances
        dx = np.sqrt(dvec[:, 0] ** 2 + dvec[:, 1] ** 2)
        dz = dvec[:, 2]

        # Move relative to ground
        hut = dz + self.hbs
        self.dz_tr = dz
        self.hut = hut
        self.dx_tr = dx
        hut = np.maximum(1, hut)

        # App_range = hbs
        app_range = np.tile(self.hbs, dx.shape)

        # Set x
        x = np.column_stack((np.log10(hut), dx, hut, app_range))
        return x

    def transform_data_pl(self, dvec):
        """
        Format the training data xtrain = [log10(h_UT), d2D, h_UT, h_BS]

        """
        # Get horizontal and vertical distances
        dx = np.sqrt(dvec[:, 0] ** 2 + dvec[:, 1] ** 2)
        dz = dvec[:, 2]

        # Move relative to ground
        hut = dz + self.hbs
        hut = np.maximum(1, hut)

        # App_range = hbs
        app_range = np.tile(self.hbs, dx.shape)

        # Set x
        x = np.column_stack((np.log10(hut), dx, hut, app_range))
        return x
    
    def format_data_dly(self):
        """
        Get the training data for estimating the delay spread
        
        TODO: move the power data calcuation to its own function.
            
        Returns
        -------
        None.

        """
        
        #nDatSamp = len(self.dat_tr['nlos_dly'])
        nPathsSamp = 24

        # Get the data sample (true) abosolute propagation delay data
        dly_dat_nlos = self.dat_tr['nlos_dly'][:,:nPathsSamp]
        
        # Get the sample path loss data
        pl_dat_los = self.dat_tr['los_pl']
        pl_dat_nlos = self.dat_tr['nlos_pl'][:,:nPathsSamp]
        
        link_state = self.dat_tr['link_state']
        
        include_los = np.where((self.dat_tr['rx_type'] == self.rx_type) & \
                                link_state == LinkState.los_link)[0]
        include_nlos = np.where((self.dat_tr['rx_type'] == self.rx_type) & \
                                (link_state == LinkState.nlos_link))[0]
        
        self.dly_dat_los = dly_dat_nlos[include_los,:]
        
        self.dly_dat_nlos = dly_dat_nlos[include_nlos,:19]
        self.power_dat_los = np.concatenate((pl_dat_los[include_los,None],\
                        pl_dat_nlos[include_los,:]),1)
        self.power_dat_nlos = pl_dat_nlos[include_nlos,:]
        
        self.Ilos = include_los
        self.Inlos = include_nlos
    
    def format_data(self):
        """
        Format the training data xtrain = [log10(h_UT), d2D, h_UT, h_BS]

        Returns
        -------
        None.

        """
        # Get the link state
        link_state = self.dat_tr['link_state']

        # If using the outage points, include all links.
        # Else, remove the outage points
        if self.use_outage:
            include_link = np.tile(True, link_state.shape)
        else:
            include_link = (link_state == LinkState.los_link) | \
                           (link_state == LinkState.nlos_link)

        # Get horizontal and vertical distances
        dx = np.sqrt(self.dvec[:, 0] ** 2 + self.dvec[:, 1] ** 2)
        dz = self.dvec[:, 2]

        # Get labels for where a link exists
        self.ind = np.where((self.dat_tr['rx_type'] == self.rx_type) & include_link)[0]

        # Index the distance arrays
        self.dx_tr = dx[self.ind]
        self.dz_tr = dz[self.ind]

        # Compute UT height (3GPP standard)
        # self.hut = self.dz_tr + np.abs(min(self.dz_tr)) + self.hbs
        ntr = np.shape(self.dz_tr)[0]
        self.hut = self.dz_tr + self.hbs

        # SR:  There is some inconsistency in the data.  For some samples,
        # dz + hbs < 0, which means we are going below ground?  Is something
        # wrong
        self.hut = np.maximum(1, self.hut)

        self.app_range = np.tile(self.hbs, ntr).astype(np.float32)

        # Form datasets
        self.xtr = np.column_stack((np.log10(self.hut), self.dx_tr, self.hut, self.app_range))
        self.ytr = (link_state[self.ind] == LinkState.los_link).astype(int)
        

    def format_data_path_loss(self,train_linkState):
        """
        Format the training data xtrain = [log10(h_UT), log10(d3D), h_UT, h_BS, ls, d2D]

        Returns
        -------
        None.
        """
        # Get the link state
        link_state = self.dat_tr['link_state']
        print(f'++++++++++++++++++{len(link_state)}')
        
        if train_linkState == LinkState.los_link:
            include_link = (link_state == LinkState.los_link)
        elif train_linkState == LinkState.nlos_link:
            include_link = (link_state == LinkState.nlos_link)
        else:
            include_link = (link_state == LinkState.los_link) | \
                           (link_state == LinkState.nlos_link)


        chan_list, _ = data_to_mpchan(self.dat_tr, self.cfg)
        n = len(chan_list)
        pl_omni = np.zeros(n)
        for i, chan in enumerate(chan_list):
            if chan.link_state != LinkState.no_link:
                if self.fc == self.cfg.freq_set[0]:
                    pl_omni[i] = chan.comp_omni_path_loss()[0]
                elif self.fc == self.cfg.freq_set[1]:
                    pl_omni[i] = chan.comp_omni_path_loss()[1]
                else:
                    print("Wrong!!")

        dx = np.sqrt(self.dvec[:, 0] ** 2 + self.dvec[:, 1] ** 2)
        dz = self.dvec[:, 2]

        # Get labels for where a link exists
        self.ind = np.where((self.dat_tr['rx_type'] == self.rx_type) & include_link)[0]
        print(f'++++++++++++++++++{len(self.ind)}')
        # Index the distance arrays
        self.dx_tr = dx[self.ind]
        self.dz_tr = dz[self.ind]

        # Compute UT height (3GPP standard)
        # self.hut = self.dz_tr + np.abs(min(self.dz_tr)) + self.hbs
        ntr = np.shape(self.dz_tr)[0]
        self.hut = self.dz_tr + self.hbs
        self.hut = np.maximum(1, self.hut)
        self.app_range = np.tile(self.hbs, ntr).astype(np.float32)
        d3d = np.sqrt(self.dx_tr**2 + self.dz_tr**2)

        # Form Path loss datasets
        self.xtr_pl = np.column_stack((np.log10(self.hut), np.log10(d3d), self.hut,
                                       self.app_range, link_state[self.ind], self.dx_tr))
        self.ytr_pl = pl_omni[self.ind]   
        

class BoundCallback(tf.keras.callbacks.Callback):
    """
    Bounds the variables
    """

    def __init__(self, layer, low=None, high=None):
        """
        Parameters:
        ----------
        layer:  TF Layer
            layer to apply the bounding to
        low, high:  np.arrays or None
            lower and upper bounds on the parameters
        """
        super().__init__()
        self.layer = layer
        self.low = low
        self.high = high

    def on_train_batch_end(self, batch, logs=None):

        # Get the current weights of the model and apply the bound
        new_w = []
        for w in self.layer.get_weights():
            # Apply the bounds
            if not (self.low is None):
                w = np.maximum(self.low, w)
            if not (self.high is None):
                w = np.minimum(self.high, w)
            new_w.append(w)

        # Reset the weights
        self.layer.set_weights(new_w)


class Custom3gppPathLossLayer(tf.keras.layers.Layer):
    """
    Custom TF layer for training the PATH LOSS parameters in the 3GPP aerial model

    The inputs are X = [log10(h_UT), log10(d3D), d2D, h_UT, h_BS, ls]

    if (h_UT < h_BS):

        p1 = param[1]

    if (d2D < d1):
        plos = 1
    else
        plos = d1/d2d + exp(-d2d/p1)*(1-d1/d2d)
    """
    def __init__(self, param_los_nom=None, param_nlos_nom=None, fc=28e9, **kwargs):
        super(Custom3gppPathLossLayer, self).__init__(**kwargs)

        self.weight_los = None
        self.weight_nlos = None
        self.input_dim = None
        if param_los_nom is None:
            # self.param_los_nom = [32.4, 21.0, 20.0, 32.4, 40.0, 20.0, -9.5, 4.0]
            self.param_los_nom = [32.4, 21.0, 20.0, 32.4, 40.0, 20.0, -9.5]
        else:
            self.param_los_nom = param_los_nom
        if param_nlos_nom is None:
            # self.param_nlos_nom = [35.3, 22.4, 21.3, -0.3, 7.82]
            self.param_nlos_nom = [35.3, 22.4, 21.3, -0.3]
        else:
            self.param_nlos_nom = param_nlos_nom
        self.fc = tf.Variable(initial_value=fc, trainable=False, dtype='float32')
        # print(f"fc in Custom3gppPathLossLayer = {self.fc}" )
        self.c = PhyConst.light_speed

        tot_params = len(self.param_los_nom) + len(self.param_nlos_nom)
        print(f"The total number of trainable params for this layer is {tot_params}")

    def build(self, input_dim):
        self.input_dim = input_dim

        # Set the initial scaling to 1, which set the parameters
        # equal to the 3GPP values
        weight_init = np.ones(len(self.param_los_nom))
        self.weight_los = tf.Variable(initial_value=weight_init,
                                      dtype='float32',
                                      trainable=True,
                                      name='weight_los')
        weight_init_ = np.ones(len(self.param_nlos_nom))
        self.weight_nlos = tf.Variable(initial_value=weight_init_,
                                       dtype='float32',
                                       trainable=True,
                                       name='weight_nlos')

    def call(self, inputs, training=None):
        # The inputs are [log10(h_UT), log10(d3D), h_UT, h_BS, ls, d2D]
        # # compute free space path loss
        # free_space_pl = 20.0*tf.experimental.numpy.log10(inputs[:, 1]) + \
        #                 20.0*tf.experimental.numpy.log10(tf.divide(self.fc, 1e6)) - 27.55
        # break_point_d = 4.0*(inputs[:, 3]-1.0)*(inputs[:, 2])*(self.fc/self.c)
        break_point_d = 4.0*(inputs[:, 3])*(inputs[:, 2])*(self.fc/self.c)

        # Scale the weights to get the parameters
        param_los = self.weight_los * self.param_los_nom
        param_nlos = self.weight_nlos * self.param_nlos_nom

        pl_umi_los = tf.where(inputs[:, 5] <= break_point_d,
                              param_los[0] + param_los[1]*inputs[:, 1] + param_los[2]*tf.experimental.numpy.log10(tf.divide(self.fc, 1e9)),
                              param_los[3] + param_los[4]*inputs[:, 1] + param_los[5]*tf.experimental.numpy.log10(tf.divide(self.fc, 1e9))
                              + param_los[6]*tf.experimental.numpy.log10(break_point_d**2 + (inputs[:, 3] - inputs[:, 2])**2))

        # pl_umi_av_los = param_los[7]+(param_los[8]+param_los[9]*inputs[:, 0])*inputs[:, 1] + \
        #                 param_los[10]*tf.experimental.numpy.log10(tf.divide(self.fc, 1e9))

        path_loss_v = tf.where( inputs[:, 4] == LinkState.los_link,
                                # LOS
                                pl_umi_los + tf.random.normal([1], 0, 4.0),
                                # NLOS
                                tf.math.maximum(pl_umi_los, param_nlos[0] * inputs[:, 1] + param_nlos[1] +
                                                param_nlos[2] * tf.experimental.numpy.log10(tf.divide(self.fc, 1e9)) +
                                                param_nlos[3] * (inputs[:, 2] - 1.5)) + \
                                                tf.random.normal([1], 0, 7.82)
                               ) 
        return path_loss_v

    def get_config(self):
        base_config = super(Custom3gppPathLossLayer, self).get_config()
        config = {'input_dim': self.input_dim}
        return dict(list(base_config.items()) + list(config.items()))


class Custom3gppPlosLayer(tf.keras.layers.Layer):
    """
    Custom TF layer for training PLOS in the 3GPP aerial model

    The inputs are X = [log10(h_UT), d2D, h_UT, h_BS]

    if (h_UT < h_BS):
        d1 = param[0]
        p1 = param[1]

    if (d2D < d1):
        plos = 1
    else
        plos = d1/d2d + exp(-d2d/p1)*(1-d1/d2d)
    """
    def __init__(self, param_nom=None, **kwargs):
        super(Custom3gppPlosLayer, self).__init__(**kwargs)

        self.weight = None
        self.input_dim = None
        if param_nom is None:
            self.param_nom = np.array([18.0, 36.0])
        else:
            self.param_nom = param_nom

        tot_params = len(self.param_nom)
        print(f"The total number of trainable params for this layer is {tot_params}")

    def build(self, input_dim):
        self.input_dim = input_dim

        # Set the initial scaling to 1, which set the parameters
        # equal to the 3GPP values
        weight_init = np.ones(len(self.param_nom))
        self.weight = tf.Variable(initial_value=weight_init,
                                  dtype='float32',
                                  trainable=True)

    def call(self, inputs, training=None):

        # Scale the weights to get the parameters
        param = self.weight * self.param_nom

        d1 = param[0]
        p1 = param[1]
        plos = tf.where(inputs[:, 1] <= d1,
                        1.0, d1/inputs[:, 1] + tf.exp(-inputs[:, 1]/p1)*(1-d1/inputs[:, 1]))

        # clip the PLOS to avoid unexpected behaviors in training
        if training:
            plos = tf.where(plos <= 0.001, 0.001, plos)
            plos = tf.where(plos >= 0.999, 0.999, plos)
        return plos

    def get_config(self):
        base_config = super(Custom3gppPlosLayer, self).get_config()
        config = {'input_dim': self.input_dim}
        return dict(list(base_config.items()) + list(config.items()))
