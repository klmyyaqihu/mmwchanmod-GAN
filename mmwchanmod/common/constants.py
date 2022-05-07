"""
constants.py:  Constants and data structures used in the package
"""
import os

class PhyConst(object):
    """
    Physical constants
    """
    light_speed = 2.99792458e8
    kT =  -174
    
class AngleFormat(object):
    """
    Constants for formatting angles in each link
    """
    
    # Indices for the angle data
    aoa_phi_ind = 0
    aoa_theta_ind = 1
    aod_phi_ind = 2
    aod_theta_ind = 3
    nangle = 4
    ang_name = ['AoA_Phi', 'AoA_theta', 'AoD_phi', 'AoD_theta']
    
class LinkState(object):
    """
    Static class with link states
    """
    no_link = 0
    los_link = 1
    nlos_link = 2
    name = ['NoLink', 'LOS', 'NLOS']
    nlink_state = 3
    
class DataConfig(object):
    """
    Meta data on ray tracing data
    """
    def __init__(self):

        self.freq_set = []
        self.nfreq = 0

        self.date_created = 'unknown'
        self.desc = 'data set'
        self.rx_types = ['RX0']
        self.max_ex_pl = 80
        self.tx_pow_dbm = 23
        self.npaths_max = 20 
        
        
    def __str__(self):
        string =  ('Description:   %s' % self.desc) + os.linesep
        string += ('Date created:  %s' % self.date_created) + os.linesep
        string += ('fc:            %12.4e Hz' % self.fc) + os.linesep
        string += ('max excess path loss: %5.1f' % self.max_ex_pl) + os.linesep
        string += ('max num paths: %d' % self.npaths_max) + os.linesep
        string += ('RX types:      %s' % str(self.rx_types))
        return string
    
    def summary(self):
        """
        Prints a summary of the configuration
        """
        print(str(self))    

           