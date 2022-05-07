"""
datastats.py:  Methods for basic statistics on datasets
"""

import numpy as np
from mmwchanmod.common.constants import LinkState   
from mmwchanmod.sim.chanmod import MPChan   

def combine_nlos_los(data, cfg):
    """
    Combines the LOS and NLOS components in the data
    
    Parameters
    ----------
    data : dictionary
        Data dictionary 

    Returns
    -------
    pl : (nlink,npaths) array
        path losses for each link and each path
    pl2 : (nlink,npaths) array
    ang : (nlink,npaths,nangle) array
        angles for each link and each path
    dly : (nlink,npaths) array
        angles for each link and each path
    """
    # Copy the NLOS path losses and angles
    pl_ls = np.zeros((cfg.nfreq, data['nlos_pl'].shape[0],data['nlos_pl'].shape[1]))
    for ifreq in range(cfg.nfreq):
        if ifreq == 0:
            #10/03
            pl_ls[ifreq,:,:] = np.copy(data['nlos_pl'])
            # pl_ls[ifreq,:,:] = data['nlos_pl']
        else:
            #10/03
            pl_ls[ifreq,:,:] = np.copy(data['nlos_pl'+str(ifreq+1)])
            # pl_ls[ifreq,:,:] = data['nlos_pl'+str(ifreq+1)]

    # pl = np.copy(data['nlos_pl'])
    # pl2 = np.copy(data['nlos_pl2'])
    ang = np.copy(data['nlos_ang'])
    dly = np.copy(data['nlos_dly'])
    
    # On the links with LOS paths, move over the
    # the NLOS data and insert the NLOS paths
    Ilos = np.where(data['link_state'] == LinkState.los_link)[0]
    for ifreq in range(cfg.nfreq):
        if ifreq == 0:
            pl_ls[ifreq,Ilos,1:] = pl_ls[ifreq,Ilos,:-1]
            pl_ls[ifreq,Ilos,0] = data['los_pl'][Ilos]
        else:
            pl_ls[ifreq,Ilos,1:] = pl_ls[ifreq,Ilos,:-1]
            pl_ls[ifreq,Ilos,0] = data['los_pl'+str(ifreq+1)][Ilos]


    # pl[Ilos,1:] = pl[Ilos,:-1]
    # pl2[Ilos,1:] = pl2[Ilos,:-1]
    ang[Ilos,1:,:] = ang[Ilos,:-1,:]
    dly[Ilos,1:] = dly[Ilos,:-1]
    # pl[Ilos,0] = data['los_pl'][Ilos]
    # pl2[Ilos,0] = data['los_pl2'][Ilos]
    ang[Ilos,0,:] = data['los_ang'][Ilos,:]
    dly[Ilos,0] = data['los_dly'][Ilos]
    
    return pl_ls, ang, dly
    


def data_to_mpchan(data, cfg):
    """
    Converts a data dictionary to a list of MPChan
    
    Parameters
    ----------
    data:  Dictionary
        Dictionary with lists of items for each channel
    cfg: DataConfig
        Data configuration 
        
    Returns
    -------
    chan_list:  List of MPChan objects
        One object for each channel
    link_state:  np.array of ints
        Links states for each link.  This may be different than the
        data['link_state'] since occassionally some paths will be
        truncated.
    """
    
    # Combine the NLOS and LOS paths
    pl_ls, ang, dly = combine_nlos_los(data, cfg)
    fspl1 = data['fspl1']
    fspl2 = data['fspl2']
        
    # Loop over the channels and convert each to a MP channel structure
    chan_list = []
    n = ang.shape[0]
    
    link_state = data['link_state']
    
    for i in range(n):
        chan = MPChan(cfg = cfg)
        pl2 = pl_ls[1,i,:]
        pl = pl_ls[0,i,:]
        # valid_path_idx = np.where(pl2<np.max(pl2))[0]
        # valid_path_idx = np.where(pl!=np.max(pl))[0]
        # if len(valid_path_idx) == 0:
        #     npath = 0
        # else:
        #     npath = valid_path_idx[-1] + 1 

        # if (npath > 0):
        #     chan.dly = dly[i,:npath]
        #     chan.ang = ang[i,:npath,:]
        #     # chan.pl  = pl[i,:npath]
        #     # chan.pl2  = pl2[i,:npath]
        #     chan.pl_ls = pl_ls[:,i,:npath]
        #     chan.link_state = data['link_state'][i]
        # else:
        #     link_state[i] = LinkState.no_link
        # chan_list.append(chan)

        # 10/03
        # npath = np.where(pl2 != max(pl))[0][-1]
        # print(len(np.where(pl != max(pl))[0]))
        # npath = 20 - len(np.where(pl >= fspl1[i] + 75)[0])
        
        # 12/10
        npath = 20
        if data['link_state'][i] != LinkState.no_link:
            chan.dly = dly[i,:npath]
            chan.ang = ang[i,:npath,:]
            chan.pl_ls = pl_ls[:,i,:npath]
            chan.link_state = data['link_state'][i]
        else:
            chan.link_state = LinkState.no_link
        chan_list.append(chan)
        
    return chan_list, link_state    

def hist_mean(x,y,**kwargs):
    """
    Computes conditional empirical mean and histogram.
    
    Parameters
    ----------
    x:  (n,) array
        vector with conditioning values
    y:  (n,) array
        vector to compute the conditional mean
    **kwargs:  dictionary
        arguments to pass to np.histogram()
    
    Returns
    -------
    bin_edges:  (nbin+1,) array
        bin edges used in the histogram of  histogram of x
    xcnt:  (nbin,) array
        count of x values in each bin
    ymean:  (nbin,) array
        mean of y in each bin conditioned on x in each bin
    """
    xcnt, bin_edges = np.histogram(x, **kwargs)
    nbin = len(bin_edges)-1
    ymean = np.zeros(nbin)
    for i in range(nbin):
        I = np.where((x >= bin_edges[i]) & (x < bin_edges[i+1]))[0]
        if len(I) > 0:
            ymean[i] = np.mean(y[I])
            
    return bin_edges, xcnt, ymean

def hist_mean2d(x,y,z,**kwargs):
    """
    Computes conditional empirical mean and histogram with 2D data
    
    Parameters
    ----------
    x, y:  (n,) arrays
        vectors for the conditioning values
    z:  (n,) array
        vector to compute the conditional mean
    **kwargs:  dictionary
        arguments to pass to np.histogram2d()
    
    Returns
    -------
    xedges, yedges:  (nbinx+1,) and (nbiny*1,) array
        bin edges used in the histogram of  histogram of x and y
    xycnt:  (nbinx,nbiny) array
        count of x, y values in each bin
    zmean:  (nbinx,nbiny) array
        mean of z in each bin conditioned on x,y in each bin
    """
    xycnt, xedges, yedges = np.histogram2d(x,y,**kwargs)
    nbinx = len(xedges)-1
    nbiny = len(yedges)-1
    zmean = np.zeros((nbinx,nbiny))
    for i in range(nbinx):
        for j in range(nbiny):
            I = np.where((x >= xedges[i]) & (x < xedges[i+1]) &\
                         (y >= yedges[j]) & (y < yedges[j+1]))[0]
            if len(I) > 0:
                zmean[i,j] = np.mean(z[I])                
                
    return xedges, yedges, xycnt, zmean

    
