"""
Plots the results.
inv_transfomr function re-scaled the data

Inputs are 
    test.p : real data
    gen_path.p : GAN generated data
    

Create at 2022/4/21/16:25:12
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kde

"""
Load and inverse transform the data
"""
def inv_transform(X, max_ls, min_ls, los_dly, ex_dly_mean):
    
    """
    Inverse transform the data to nlink*10*18
    
    Max 10 clusters- each has 18 parameters
    
    """

    # ncluster_ls = X[:,:3]
    nlink = len(X)
    if len(X.shape) == 2: # nlink*173
        X_inv = np.reshape(X[:,3:], [nlink, 10, 18])
    
    for ilink in range(nlink):
    # for ilink in range(1):
        I_f1 = np.where(np.squeeze(X_inv[ilink, :, 10]) >= 0.0)[0]
        I_f1_outage = np.where(np.squeeze(X_inv[ilink, :, 10]) < 0.0)[0]
        I_f2 = np.where(np.squeeze(X_inv[ilink, :, 16]) >= 0.0)[0]
        I_f2_outage = np.where(np.squeeze(X_inv[ilink, :, 16]) < 0.0)[0]
        
        npath = max(len(I_f1), len(I_f2))
        if npath == 0:
            X_inv[ilink,:,:] = -1
        else:
            # inverse [0:5] (Mean angles + min delay)
            X_inv[ilink, npath:, 0:5] = -1
            
            for ipath in range(npath):
                if X_inv[ilink, ipath, 0] > 0.5:
                    X_inv[ilink, ipath, 0] = X_inv[ilink, ipath, 0] % 0.5
                if X_inv[ilink, ipath, 0] < -0.5:
                    X_inv[ilink, ipath, 0] = X_inv[ilink, ipath, 0] % (-0.5)
                if X_inv[ilink, ipath, 2] > 0.5:
                    X_inv[ilink, ipath, 2] = X_inv[ilink, ipath, 2] % 0.5
                if X_inv[ilink, ipath, 2] < -0.5:
                    X_inv[ilink, ipath, 2] = X_inv[ilink, ipath, 2] % (-0.5)
                    
                if X_inv[ilink, ipath, 1] > 1:
                    X_inv[ilink, ipath, 1] = X_inv[ilink, ipath, 1] % 1
                if X_inv[ilink, ipath, 1] < 0:
                    X_inv[ilink, ipath, 1] = -X_inv[ilink, ipath, 1] % (1)
                if X_inv[ilink, ipath, 3] > 1:
                    X_inv[ilink, ipath, 3] = X_inv[ilink, ipath, 3] % 1
                if X_inv[ilink, ipath, 3] < 0:
                    X_inv[ilink, ipath, 3] = -X_inv[ilink, ipath, 3] % (1)
                    
            # Inv- 0, 1, 2, 3 AoD and AoA
            X_inv[ilink, :npath, 0] = X_inv[ilink, :npath, 0] * 360
            X_inv[ilink, :npath, 1] = X_inv[ilink, :npath, 1] * 180
            X_inv[ilink, :npath, 2] = X_inv[ilink, :npath, 2] * 360
            X_inv[ilink, :npath, 3] = X_inv[ilink, :npath, 3] * 180
            # Inv- 4 mean dly
            X_inv[ilink, :npath, 4] = np.maximum(X_inv[ilink, :npath, 4], 0.0)
            X_inv[ilink, :npath, 4] = (X_inv[ilink, :npath, 4]*ex_dly_mean +
                                       los_dly[ilink])
        
            # inverse freq 1
            if len(I_f1) == 0:
                X_inv[ilink, :, 5:11] = -1
            else:
                X_inv[ilink, I_f1_outage, 5:11] = -1
                X_inv[ilink, I_f1, 5:11] = np.minimum(X_inv[ilink, I_f1, 5:11], 1.0)
                X_inv[ilink, I_f1, 5:11] = np.maximum(X_inv[ilink, I_f1, 5:11], 0.0)
                
                X_inv[ilink, I_f1_outage, 17] = -1
                X_inv[ilink, I_f1, 17] = np.minimum(X_inv[ilink, I_f1, 17], 1.0)
                X_inv[ilink, I_f1, 17] = np.maximum(X_inv[ilink, I_f1, 17], 0.0)
                
                # Inv- 5, 6, 7, 8, 9, 10 RMS aoa aod dly power
                X_inv[ilink, I_f1, 5] = (X_inv[ilink, I_f1, 5] * (
                                        max_ls[0] - min_ls[0])  + min_ls[0])
                X_inv[ilink, I_f1, 6] = (X_inv[ilink, I_f1, 6] * (
                                        max_ls[1] - min_ls[1])  + min_ls[1])
                X_inv[ilink, I_f1, 7] = (X_inv[ilink, I_f1, 7] * (
                                        max_ls[2] - min_ls[2])  + min_ls[2])
                X_inv[ilink, I_f1, 8] = (X_inv[ilink, I_f1, 8] * (
                                        max_ls[3] - min_ls[3])  + min_ls[3])
                X_inv[ilink, I_f1, 9] = (X_inv[ilink, I_f1, 9] * (
                                        max_ls[4] - min_ls[4])  + min_ls[4])
                X_inv[ilink, I_f1, 10] = (X_inv[ilink, I_f1, 10] * (
                                        max_ls[5] - min_ls[5])  + min_ls[5])
                
            # inverse freq 2
            if len(I_f2) == 0:
                X_inv[ilink, :, 11:17] = -1
            else:
                X_inv[ilink, I_f2_outage, 11:17] = -1
                X_inv[ilink, I_f2, 11:17] = np.minimum(X_inv[ilink, I_f2, 11:17], 1.0)
                X_inv[ilink, I_f2, 11:17] = np.maximum(X_inv[ilink, I_f2, 11:17], 0.0)
                
                X_inv[ilink, I_f2_outage, 17] = -1
                
                # Inv- 5, 6, 7, 8, 9, 10 RMS aoa aod dly power
                X_inv[ilink, I_f2, 11] = (X_inv[ilink, I_f2, 11] * (
                                        max_ls[6] - min_ls[6])  + min_ls[6])
                X_inv[ilink, I_f2, 12] = (X_inv[ilink, I_f2, 12] * (
                                        max_ls[7] - min_ls[7])  + min_ls[7])
                X_inv[ilink, I_f2, 13] = (X_inv[ilink, I_f2, 13] * (
                                        max_ls[8] - min_ls[8])  + min_ls[8])
                X_inv[ilink, I_f2, 14] = (X_inv[ilink, I_f2, 14] * (
                                        max_ls[9] - min_ls[9])  + min_ls[9])
                X_inv[ilink, I_f2, 15] = (X_inv[ilink, I_f2, 15] * (
                                        max_ls[10] - min_ls[10])  + min_ls[10])
                X_inv[ilink, I_f2, 16] = (X_inv[ilink, I_f2, 16] * (
                                        max_ls[11] - min_ls[11])  + min_ls[11])
            
            I_11 = np.where(np.squeeze(X_inv[ilink, :, 17]) != -1)[0]
            X_inv[ilink, I_11, 17] = (X_inv[ilink, I_11, 17] * (
                                     max_ls[12] - min_ls[12])  + min_ls[12])  
                
            
    return X_inv

with open("gen_data.p", "rb") as file:
    gen_d_ = pickle.load(file)

with open("test_data.p", "rb") as file:
    true_d_ = pickle.load(file)
    
true_data = true_d_['data']
max_ls = true_d_['max_ls']
min_ls = true_d_['min_ls']
los_dly = true_d_['los_dly']
ex_dly_mean = true_d_['ex_dly_mean']
true_data_inv = inv_transform(true_data, max_ls, min_ls, los_dly, ex_dly_mean)
gen_data_inv = inv_transform(gen_d_, max_ls, min_ls, los_dly, ex_dly_mean)

"""
Plots of power
"""
def plot_power_two_freq_vs(data, true_data, use_diff = False, save_fig = False):
    """
    Plot 28 vs 140 GHz scatter figure.
    
    Parameters
    ----------
    data : np.array of float64 (nlink, 10, 18)
        Inverse-data of GAN generated links.
    true_data : np.array of float64 (nlink, 10, 18)
        Inverse-data of true test links.
    use_diff : bool, optional
        If true, using the difference to compute 140GHz received power.
        [ilink, :, 17]
        The default is False.
    save_fig : bool, optional
        If true, save figure. The default is False.

    Returns
    -------
    None.

    """
    
    # plot GAN links
    power = data[:,:,[10,16]]
    diff_pwr = data[:,:,17]
    nlink = len(power)
    sum_p_f1_ls = []
    sum_p_f2_ls = []

    for ilink in range(nlink):
        I_f1 = np.where(np.squeeze(power[ilink,:,0]) != -1)[0]
        I_f2 = np.where(np.squeeze(power[ilink,:,1]) != -1)[0]
        if len(I_f1) == 0 or len(I_f2) == 0: continue # outage
        # MaxMinSclare back
        # freq 1
        p = power[ilink,I_f1,0]
        p_mW = 10**(0.1*p) #mW
        sum_p_mW = np.sum(p_mW)
        sum_p1_dBm = 10*np.log10(sum_p_mW)

        # freq 2
        # MaxMinSclare back
        if use_diff:
            # using difference of total power to compute 140GHz power
            sum_p_mW = 0
            for i in I_f2:
                if (power[ilink, i, 0] != -1 and diff_pwr[ilink, i] != -1):
                    p_f1 = power[ilink,i,0] # dBm
                    p_f2 = p_f1 - diff_pwr[ilink,i] # dBm
                    p_mW = 10 ** (0.1 * p_f2)
                    
                    p_train_out_mW = 10 ** (0.1 * power[ilink,i,1])
                    if p_mW >= p_train_out_mW: # P_f1 - diff >= train-out p_f2
                        p_mW = p_train_out_mW
                    
                    sum_p_mW += p_mW
                else:
                    p_f2 = power[ilink,i,1] # dBm
                    sum_p_mW += 10 ** (0.1 * p_f2)
                    
            sum_p2_dBm = 10 * np.log10(sum_p_mW)
        else: # use_diff = False
            p = power[ilink,I_f2,1] # dBm
            p_mW = 10**(0.1*p) #mW
            sum_p_mW = np.sum(p_mW)
            sum_p2_dBm = 10*np.log10(sum_p_mW)

    
        # if (sum_p1_dBm >= sum_p2_dBm - 10):
        sum_p_f1_ls.append(sum_p1_dBm)  
        sum_p_f2_ls.append(sum_p2_dBm)
    
    # plot
    plt.figure()
    I = np.where(np.array(sum_p_f2_ls)>=-230)[0]
    plt.scatter(np.array(sum_p_f1_ls)[I], np.array(sum_p_f2_ls)[I], c='r', s=10, alpha=0.1)
    plt.xlabel("28 GHz (dBm)")
    plt.ylabel("140 GHz (dBm)")
    plt.grid()
    # plt.title("Total Received Power Scatter")
    
    
    # plot true links
    true_power = true_data[:,:,[10,16]]
    nlink = len(true_power)
    sum_p_f1_ls = []
    sum_p_f2_ls = []

    for ilink in range(nlink):
        I_f1 = np.where(np.squeeze(true_power[ilink,:,0]) != -1)[0]
        I_f2 = np.where(np.squeeze(true_power[ilink,:,1]) != -1)[0]
        if len(I_f1) == 0 or len(I_f2) == 0: continue # outage
        # MaxMinSclare back
        # freq 1
        p = true_power[ilink,I_f1,0] # dBm
        p_mW = 10**(0.1*p) #mW
        sum_p_mW = np.sum(p_mW)
        sum_p1_dBm = 10*np.log10(sum_p_mW)
        sum_p_f1_ls.append(sum_p1_dBm)
        
        # freq 2
        p = true_power[ilink,I_f2,1] # dBm
        p_mW = 10**(0.1*p) #mW
        sum_p_mW = np.sum(p_mW)
        sum_p2_dBm = 10*np.log10(sum_p_mW)
        sum_p_f2_ls.append(sum_p2_dBm)
        
    I = np.where(np.array(sum_p_f2_ls)>=-230)[0]
    plt.scatter(np.array(sum_p_f1_ls)[I], np.array(sum_p_f2_ls)[I], c='b', s=10, alpha=0.2)
    plt.legend(["Generated", "True"])
    plt.xlim([-210, -45])
    if save_fig:
        plt.savefig("./plot/power_freq_vs_scatter.png", dpi = 400) 


def plot_power_cdf(data, true_data, use_diff = False, save_fig = False):
    """
    Plot CDF of received power.

    Parameters
    ----------
    data : np.array of float64 (nlink, 10, 18)
        Inverse-data of GAN generated links.
    true_data : np.array of float64 (nlink, 10, 18)
        Inverse-data of true test links.
    use_diff : bool, optional
        If true, using the difference to compute 140GHz received power.
        [ilink, :, 17]
        The default is False.
    save_fig : bool, optional
        If true, save figure. The default is False.

    Returns
    -------
    None.

    """
    
    power = data[:,:,[10,16]]
    diff_pwr = data[:,:,17]
    nlink = len(power)
    
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    
    sum_p_f1_ls = []
    # freq 1
    for ilink in range(nlink):
        I_f1 = np.where(np.squeeze(power[ilink,:,0]) != -1)[0]
        if len(I_f1) == 0: continue # outage in f1
        # MaxMinSclare back
        p = power[ilink,I_f1,0] # dBm
        p_mW = 10**(0.1*p) #mW
        sum_p_mW = np.sum(p_mW)
        sum_p_dBm = 10*np.log10(sum_p_mW)
        sum_p_f1_ls.append(sum_p_dBm)
    
    sum_p_f2_ls = []
    # freq 2
    for ilink in range(nlink):
        # freq 1
        I_f2 = np.where(np.squeeze(power[ilink,:,1]) != -1)[0]
        if len(I_f2) == 0: continue # outage in f2
        # MaxMinSclare back

        if use_diff:
            # using difference of total power to compute 140GHz power
            sum_p_mW = 0
            for i in I_f2:
                if (power[ilink, i, 0] != -1 and diff_pwr[ilink, i] != -1):
                    p_f1 = power[ilink,i,0] # dBm
                    p_f2 = p_f1 - diff_pwr[ilink,i] # dBm
                    p_mW = 10 ** (0.1 * p_f2)
                    sum_p_mW += p_mW
                else:
                    p_f2 = power[ilink,i,1] # dBm
                    sum_p_mW += 10 ** (0.1 * p_f2)
            sum_p_dBm = 10 * np.log10(sum_p_mW)
            sum_p_f2_ls.append(sum_p_dBm)
        else:
            p = power[ilink,I_f2,1] # dBm
            p_mW = 10**(0.1*p) #mW
            sum_p_mW = np.sum(p_mW)
            sum_p_dBm = 10*np.log10(sum_p_mW)
            sum_p_f2_ls.append(sum_p_dBm)
                    
                    
    I = np.arange(len(sum_p_f1_ls))/len(sum_p_f1_ls)
    axs[0].plot(np.sort(sum_p_f1_ls), I)
    I = np.arange(len(sum_p_f2_ls))/len(sum_p_f2_ls)
    axs[1].plot(np.sort(sum_p_f2_ls), I)
    
    true_power = true_data[:,:,[10,16]]
    sum_p_f1_ls = []
    # freq 1
    for ilink in range(nlink):
        I_f1 = np.where(np.squeeze(true_power[ilink,:,0]) != -1)[0]
        if len(I_f1) == 0: continue # outage in f1
        # MaxMinSclare back
        p = true_power[ilink,I_f1,0] # dBm
        p_mW = 10**(0.1*p) #mW
        sum_p_mW = np.sum(p_mW)
        sum_p_dBm = 10*np.log10(sum_p_mW)
        sum_p_f1_ls.append(sum_p_dBm)
    
    sum_p_f2_ls = []
    # freq 2
    for ilink in range(nlink):
        # freq 1
        I_f2 = np.where(np.squeeze(true_power[ilink,:,1]) != -1)[0]
        if len(I_f2) == 0: continue # outage in f1
        # MaxMinSclare back
        p = true_power[ilink,I_f2,1] # dBm
        p_mW = 10**(0.1*p) #mW
        sum_p_mW = np.sum(p_mW)
        sum_p_dBm = 10*np.log10(sum_p_mW)
        sum_p_f2_ls.append(sum_p_dBm)
    
    I = np.arange(len(sum_p_f1_ls))/len(sum_p_f1_ls)
    axs[0].plot(np.sort(sum_p_f1_ls), I)
    I = np.arange(len(sum_p_f2_ls))/len(sum_p_f2_ls)
    axs[1].plot(np.sort(sum_p_f2_ls), I)
    
    axs[0].grid()
    axs[1].grid()
    axs[0].title.set_text("28 GHz")
    axs[1].title.set_text("140 GHz")
    axs[0].set_ylabel("CDF")
    f.supxlabel("Total received power (dBm)")
    plt.legend(["Generated", "True"])
    if save_fig:
        plt.savefig("./plot/power_cdf.png", dpi = 400)
    
        # 
# plot_power_two_freq_vs(gen_data_inv, true_data_inv, True, save_fig=True)
# plot_power_cdf(gen_data_inv, true_data_inv,save_fig=True)

"""
Plots of power
"""
def plot_power_two_freq_vs_kde(data, true_data, use_diff = False, save_fig = False):
    """
    Plot 28 vs 140 GHz scatter figure.
    
    Parameters
    ----------
    data : np.array of float64 (nlink, 10, 18)
        Inverse-data of GAN generated links.
    true_data : np.array of float64 (nlink, 10, 18)
        Inverse-data of true test links.
    use_diff : bool, optional
        If true, using the difference to compute 140GHz received power.
        [ilink, :, 17]
        The default is False.
    save_fig : bool, optional
        If true, save figure. The default is False.

    Returns
    -------
    None.

    """
    
    # plot GAN links
    power = data[:,:,[10,16]]
    diff_pwr = data[:,:,17]
    nlink = len(power)
    sum_p_f1_ls = []
    sum_p_f2_ls = []

    for ilink in range(nlink):
        I_f1 = np.where(np.squeeze(power[ilink,:,0]) != -1)[0]
        I_f2 = np.where(np.squeeze(power[ilink,:,1]) != -1)[0]
        if len(I_f1) == 0 or len(I_f2) == 0: continue # outage
        # MaxMinSclare back
        # freq 1
        p = power[ilink,I_f1,0]
        p_mW = 10**(0.1*p) #mW
        sum_p_mW = np.sum(p_mW)
        sum_p1_dBm = 10*np.log10(sum_p_mW)

        # freq 2
        # MaxMinSclare back
        if use_diff:
            # using difference of total power to compute 140GHz power
            sum_p_mW = 0
            for i in I_f2:
                if (power[ilink, i, 0] != -1 and diff_pwr[ilink, i] != -1):
                    p_f1 = power[ilink,i,0] # dBm
                    p_f2 = p_f1 - diff_pwr[ilink,i] # dBm
                    p_mW = 10 ** (0.1 * p_f2)
                    
                    p_train_out_mW = 10 ** (0.1 * power[ilink,i,1])
                    if p_mW >= p_train_out_mW: # P_f1 - diff >= train-out p_f2
                        p_mW = p_train_out_mW
                    
                    sum_p_mW += p_mW
                else:
                    p_f2 = power[ilink,i,1] # dBm
                    sum_p_mW += 10 ** (0.1 * p_f2)
                    
            sum_p2_dBm = 10 * np.log10(sum_p_mW)
        else: # use_diff = False
            p = power[ilink,I_f2,1] # dBm
            p_mW = 10**(0.1*p) #mW
            sum_p_mW = np.sum(p_mW)
            sum_p2_dBm = 10*np.log10(sum_p_mW)

    
        # if (sum_p1_dBm >= sum_p2_dBm - 10):
        sum_p_f1_ls.append(sum_p1_dBm)  
        sum_p_f2_ls.append(sum_p2_dBm)
    
    # plot
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 4))
    sum_p_f1_ls = np.array(sum_p_f1_ls)
    sum_p_f2_ls = np.array(sum_p_f2_ls)
    
    I = np.where(sum_p_f2_ls>=-230)[0]

    sns.kdeplot(sum_p_f1_ls[I], sum_p_f2_ls[I], fill = True, ax=axs[0])
    sns.set_style("whitegrid")
    
    axs[0].set_xlabel("28 GHz (dBm)")
    axs[0].set_ylabel("140 GHz (dBm)")
    axs[0].set_title('Generated Links')
    axs[0].set_xlim([-210, -45])
    
    
    # plot true links
    true_power = true_data[:,:,[10,16]]
    nlink = len(true_power)
    sum_p_f1_ls = []
    sum_p_f2_ls = []

    for ilink in range(nlink):
        I_f1 = np.where(np.squeeze(true_power[ilink,:,0]) != -1)[0]
        I_f2 = np.where(np.squeeze(true_power[ilink,:,1]) != -1)[0]
        if len(I_f1) == 0 or len(I_f2) == 0: continue # outage
        # MaxMinSclare back
        # freq 1
        p = true_power[ilink,I_f1,0] # dBm
        p_mW = 10**(0.1*p) #mW
        sum_p_mW = np.sum(p_mW)
        sum_p1_dBm = 10*np.log10(sum_p_mW)
        sum_p_f1_ls.append(sum_p1_dBm)
        
        # freq 2
        p = true_power[ilink,I_f2,1] # dBm
        p_mW = 10**(0.1*p) #mW
        sum_p_mW = np.sum(p_mW)
        sum_p2_dBm = 10*np.log10(sum_p_mW)
        sum_p_f2_ls.append(sum_p2_dBm)
        
    sum_p_f1_ls = np.array(sum_p_f1_ls)
    sum_p_f2_ls = np.array(sum_p_f2_ls)
        
    I = np.where(sum_p_f2_ls>=-230)[0]

    sns.kdeplot(sum_p_f1_ls[I], sum_p_f2_ls[I], fill = True, ax=axs[1])
    sns.set_style("whitegrid")
    axs[1].set_xlabel("28 GHz (dBm)")
    axs[1].set_title('Ray Tracing Samples')

    if save_fig:
        plt.savefig("./plot/power_freq_vs_kde.png", dpi = 400) 

# plot_power_two_freq_vs_kde(gen_data_inv, true_data_inv, False, save_fig=True)


"""
Plots of Az vs Inc all 4x4
"""
def plot_ang_az_inc(data, true_data, loss_TH = int(160), save_fig = False):
    
    f, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
    # plot true data
    angs = true_data[:,:,[2,3]]
    power = true_data[:,:,10]
    az_plot = []
    inc_plot = []
    # Scale angles back
    for ilink in range(len(angs)):
        I = np.where(angs[ilink, :, 0] != -1)[0]
        if len(I) == 0: continue
        
        for i in I:
            p = power[ilink,i]
            if p >= 23 - loss_TH:
                inc_plot.append(angs[ilink, i, 1])
                az_plot.append(angs[ilink, i, 0])
            
    nbins=100
    k = kde.gaussian_kde([az_plot, inc_plot])
    xi, yi = np.mgrid[min(az_plot):max(az_plot):nbins*1j, 
                      min(inc_plot):max(inc_plot):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axs[0][0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.cividis)
    
    # plot true data
    angs = data[:,:,[2,3]]
    power = data[:,:,10]
    az_plot = []
    inc_plot = []
    # Scale angles back
    for ilink in range(len(angs)):
        I = np.where(angs[ilink, :, 0] != -1)[0]
        if len(I) == 0: continue
        
        for i in I:
            p = power[ilink,i]
            if p >= 23 - loss_TH:
                inc_plot.append(angs[ilink, i, 1])
                az_plot.append(angs[ilink, i, 0])
            
    nbins=100
    k = kde.gaussian_kde([az_plot,inc_plot])
    xi, yi = np.mgrid[min(az_plot):max(az_plot):nbins*1j, 
                      min(inc_plot):max(inc_plot):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axs[0][1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.cividis)
    axs[0][0].set_ylim([0,125])
    axs[0][0].title.set_text("Ray Tracing Samples")
    axs[0][1].title.set_text("Generated Clusters")
    axs[0][0].set_xlabel("Azimuth AoA")
    axs[0][1].set_xlabel("Azimuth AoA")
    axs[0][0].set_ylabel("Inclination AoA")
    
    
    angs = true_data[:,:,[0,1]]
    power = true_data[:,:,10]
    az_plot = []
    inc_plot = []
    # Scale angles back
    for ilink in range(len(angs)):
        I = np.where(angs[ilink, :, 0] != -1)[0]
        if len(I) == 0: continue
        
        for i in I:
            p = power[ilink,i]
            if p >= 23 - loss_TH:
                inc_plot.append(angs[ilink, i, 1])
                az_plot.append(angs[ilink, i, 0])
            
    nbins=100
    k = kde.gaussian_kde([az_plot,inc_plot])
    xi, yi = np.mgrid[min(az_plot):max(az_plot):nbins*1j, 
                      min(inc_plot):max(inc_plot):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axs[1][0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.cividis)
    
    # plot true data
    angs = data[:,:,[0,1]]
    power = data[:,:,10]
    az_plot = []
    inc_plot = []
    # Scale angles back
    for ilink in range(len(angs)):
        I = np.where(angs[ilink, :, 0] != -1)[0]
        if len(I) == 0: continue
        
        for i in I:
            p = power[ilink,i]
            if p >= 23 - loss_TH:
                inc_plot.append(angs[ilink, i, 1])
                az_plot.append(angs[ilink, i, 0])
            
    nbins=100
    k = kde.gaussian_kde([az_plot,inc_plot])
    xi, yi = np.mgrid[min(az_plot):max(az_plot):nbins*1j, 
                      min(inc_plot):max(inc_plot):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axs[1][1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.cividis)
    axs[1][0].set_xlabel("Azimuth AoD")
    axs[1][1].set_xlabel("Azimuth AoD")
    axs[1][0].set_ylabel("Inclination AoD")
    
    
    if save_fig:
        plt.savefig("./plot/az_vs_inc.png", dpi = 400)
    
plot_ang_az_inc(gen_data_inv, true_data_inv, save_fig=True)


"""
Plots of Az vs Inc AoA
"""
def plot_aoa_az_inc(data, true_data, loss_TH = int(160), save_fig = False):
    
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 3))
    # plot true data
    angs = true_data[:,:,[2,3]]
    power = true_data[:,:,10]
    az_plot = []
    inc_plot = []
    # Scale angles back
    for ilink in range(len(angs)):
        I = np.where(angs[ilink, :, 0] != -1)[0]
        if len(I) == 0: continue
        
        for i in I:
            p = power[ilink,i]
            if p >= 23 - loss_TH:
                inc_plot.append(angs[ilink, i, 1])
                az_plot.append(angs[ilink, i, 0])
            
    nbins=100
    k = kde.gaussian_kde([az_plot,inc_plot])
    xi, yi = np.mgrid[min(az_plot):max(az_plot):nbins*1j, 
                      min(inc_plot):max(inc_plot):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axs[0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.cividis)
    
    # plot true data
    angs = data[:,:,[2,3]]
    power = data[:,:,10]
    az_plot = []
    inc_plot = []
    # Scale angles back
    for ilink in range(len(angs)):
        I = np.where(angs[ilink, :, 0] != -1)[0]
        if len(I) == 0: continue
        
        for i in I:
            p = power[ilink,i]
            if p >= 23 - loss_TH:
                inc_plot.append(angs[ilink, i, 1])
                az_plot.append(angs[ilink, i, 0])
            
    nbins=100
    k = kde.gaussian_kde([az_plot,inc_plot])
    xi, yi = np.mgrid[min(az_plot):max(az_plot):nbins*1j, 
                      min(inc_plot):max(inc_plot):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axs[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.cividis)
    axs[0].set_ylim([0,125])
    axs[0].title.set_text("True Clusters")
    axs[1].title.set_text("Generated Clusters")
    f.supxlabel("Azimuth AoA")
    axs[0].set_ylabel("Inclination AoA")

    if save_fig:
        plt.savefig("./plot/aoa_az_vs_inc.png", dpi = 400)
    
# plot_aoa_az_inc(gen_data_inv, true_data_inv, save_fig=True)


"""
Plots of RMS AoA including zeros
"""
def plot_rms_aoa_inc_cdf(data, true_data, loss_TH = int(160), save_fig = False):
    
    # plot true data
    rms_f1 = true_data[:,:,8]
    rms_f2 = true_data[:,:,14]
    rms_f1_plot = []
    rms_f2_plot = []
    power_f1 = true_data[:,:,10]
    power_f2 = true_data[:,:,16]
    nlink = len(true_data)
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
        
        if len(I_f1) > 0 and len(I_f2) > 0:
            for idx in I_f1:
                if idx in I_f2:
                    # if rms_f1[ilink, idx] > 1e-8 and rms_f2[ilink, idx] > 1e-8:
                    rms_f1_plot.append(rms_f1[ilink, idx])
                    rms_f2_plot.append(rms_f2[ilink, idx])
        
        # if len(I_f1) > 0:
        #     for idx in I_f1:
        #         if rms_f1[ilink, idx] > 1e-8:
        #             rms_f1_plot.append(rms_f1[ilink, idx])
        
        # if len(I_f2) > 0:
        #     for idx in I_f2:
        #         if rms_f2[ilink, idx] > 1e-8:
        #             rms_f2_plot.append(rms_f2[ilink, idx])
    # plot CDF
    plt.figure()
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b-.')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r-.')
    
    
    # plot GAN data
    rms_f1 = data[:,:,8]
    rms_f2 = data[:,:,14]
    rms_f1_plot = []
    rms_f2_plot = []
    power_f1 = data[:,:,10]
    power_f2 = data[:,:,16]
    nlink = len(data)
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
        
        if len(I_f1) > 0 and len(I_f2) > 0:
            for idx in I_f1:
                if idx in I_f2:
                    # if rms_f1[ilink, idx] > 1e-8 and rms_f2[ilink, idx] > 1e-8:
                    rms_f1_plot.append(rms_f1[ilink, idx])
                    rms_f2_plot.append(rms_f2[ilink, idx])
    
    
    # for ilink in range(nlink):
    #     I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
    #     if len(I_f1) > 0:
    #         for idx in I_f1:
    #             if rms_f1[ilink, idx] > 1e-8:
    #                 rms_f1_plot.append(rms_f1[ilink, idx])
    #     I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
    #     if len(I_f2) > 0:
    #         for idx in I_f2:
    #             if rms_f2[ilink, idx] > 1e-8:
    #                 rms_f2_plot.append(rms_f2[ilink, idx])
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r')
    

    
    plt.legend(["28 GHz True", "140 GHz True",
                "28 GHz Generated", "140 GHz Generated"])
    plt.grid()
    plt.xlabel("RMS Spread of Inclination AoA")
    plt.ylabel("CDF")
    plt.ylim([0.8,1])
    plt.xlim([0,10])
    # plt.title(f"RMS Spread of Inclination AoA (path loss<{loss_TH} dB)")
    if save_fig:
        plt.savefig("./plot/aoa_inc_rms_spread.png", dpi = 400)    


# plot_rms_aoa_inc_cdf(gen_data_inv, true_data_inv, save_fig=True)

"""
Plots of RMS AoA Inc not including zeros
"""
def plot_rms_aoa_inc_cdf_no_zeros(data, true_data, loss_TH = int(160), save_fig = False):
    
    # plot true data
    rms_f1 = true_data[:,:,8]
    rms_f2 = true_data[:,:,14]
    rms_f1_plot = []
    rms_f2_plot = []
    power_f1 = true_data[:,:,10]
    power_f2 = true_data[:,:,16]
    nlink = len(true_data)
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
        
        if len(I_f1) > 0 and len(I_f2) > 0:
            for idx in I_f1:
                if idx in I_f2:
                    if rms_f1[ilink, idx] > 1e-8 and rms_f2[ilink, idx] > 1e-8:
                        rms_f1_plot.append(rms_f1[ilink, idx])
                        rms_f2_plot.append(rms_f2[ilink, idx])
        
    # plot CDF
    plt.figure()
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b-.')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r-.')
    
    
    # plot GAN data
    rms_f1 = data[:,:,8]
    rms_f2 = data[:,:,14]
    rms_f1_plot = []
    rms_f2_plot = []
    power_f1 = data[:,:,10]
    power_f2 = data[:,:,16]
    nlink = len(data)
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
        
        if len(I_f1) > 0 and len(I_f2) > 0:
            for idx in I_f1:
                if idx in I_f2:
                    if rms_f1[ilink, idx] > 1e-8 and rms_f2[ilink, idx] > 1e-8:
                        rms_f1_plot.append(rms_f1[ilink, idx])
                        rms_f2_plot.append(rms_f2[ilink, idx])

    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r')
    
    
    plt.legend(["28 GHz True", "140 GHz True",
                "28 GHz Generated", "140 GHz Generated"])
    plt.grid()
    plt.xlabel("RMS Spread of Inclination AoA")
    plt.ylabel("CDF")
    plt.xlim([0,10])
    # plt.title(f"RMS Spread of Inclination AoA (path loss<{loss_TH} dB)")
    if save_fig:
        plt.savefig("./plot/aoa_inc_rms_spread_no_zeros.png", dpi = 400)    


# plot_rms_aoa_inc_cdf_no_zeros(gen_data_inv, true_data_inv, 250, save_fig=False)

"""
Plots of RMS AoA Azimuth not including zeros
"""

def plot_rms_aoa_az_cdf_no_zeros(data, true_data, loss_TH = int(160), save_fig = False):
    
    # plot true data
    rms_f1 = true_data[:,:,7]
    rms_f2 = true_data[:,:,13]
    rms_f1_plot = []
    rms_f2_plot = []
    power_f1 = true_data[:,:,10]
    power_f2 = true_data[:,:,16]
    nlink = len(true_data)
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
        
        if len(I_f1) > 0:
            for idx in I_f1:
                if rms_f1[ilink, idx] > 1e-8:
                    rms_f1_plot.append(rms_f1[ilink, idx])
        if len(I_f2) > 0:
            for idx in I_f2:
                if rms_f2[ilink, idx] > 1e-8:
                    rms_f2_plot.append(rms_f2[ilink, idx])
        
    # plot CDF
    plt.figure()
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b-.')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r-.')
    
    
    # plot GAN data
    rms_f1 = data[:,:,7]
    rms_f2 = data[:,:,13]
    rms_f1_plot = []
    rms_f2_plot = []
    power_f1 = data[:,:,10]
    power_f2 = data[:,:,16]
    nlink = len(data)
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]

        if len(I_f1) > 0:
            for idx in I_f1:
                if rms_f1[ilink, idx] > 1e-8:
                    rms_f1_plot.append(rms_f1[ilink, idx])
        if len(I_f2) > 0:
            for idx in I_f2:
                if rms_f2[ilink, idx] > 1e-8:
                    rms_f2_plot.append(rms_f2[ilink, idx])

    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r')
    
    
    plt.legend(["28 GHz True", "140 GHz True",
                "28 GHz Generated", "140 GHz Generated"])
    plt.grid()
    plt.xlabel("RMS Azimuth Angle of Arrival Spread")
    plt.ylabel("CDF")
    plt.xlim([0,10])
    # plt.title(f"RMS Spread of Inclination AoA (path loss<{loss_TH} dB)")
    if save_fig:
        plt.savefig("./plot/aoa_az_rms_spread_no_zeros.png", dpi = 400)    


# plot_rms_aoa_az_cdf_no_zeros(gen_data_inv, true_data_inv, 160, save_fig=False)

"""
Plots of RMS AoA az zeros
"""
def plot_rms_aoa_az_cdf(data, true_data, loss_TH = int(160), save_fig = False):
    
    # plot true data
    rms_f1 = true_data[:,:,7]
    rms_f2 = true_data[:,:,13]
    rms_f1_plot = []
    rms_f2_plot = []
    power_f1 = true_data[:,:,10]
    power_f2 = true_data[:,:,16]
    nlink = len(true_data)
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
        
        
        if len(I_f1) > 0:
            for idx in I_f1:
                rms_f1_plot.append(rms_f1[ilink, idx])
        
        if len(I_f2) > 0:
            for idx in I_f2:
                rms_f2_plot.append(rms_f2[ilink, idx])
    # plot CDF
    plt.figure()
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b-.')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r-.')
    
    
    # plot GAN data
    rms_f1 = data[:,:,7]
    rms_f2 = data[:,:,13]
    rms_f1_plot = []
    rms_f2_plot = []
    power_f1 = data[:,:,10]
    power_f2 = data[:,:,16]
    nlink = len(data)
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
    
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        if len(I_f1) > 0:
            for idx in I_f1:
                rms_f1_plot.append(rms_f1[ilink, idx])
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
        if len(I_f2) > 0:
            for idx in I_f2:
                rms_f2_plot.append(rms_f2[ilink, idx])
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r')

    
    plt.legend(["28 GHz True", "140 GHz True",
                "28 GHz Generated", "140 GHz Generated"])
    plt.grid()
    plt.xlabel("RMS Azimuth AoA Spread")
    plt.ylabel("CDF")
    plt.ylim([0.75,1])
    plt.xlim([0,10])
    plt.title(f"RMS Azimuth AoA Spread (path loss<{loss_TH} dB)")
    if save_fig:
        plt.savefig("./plot/aoa_az_rms_spread.png", dpi = 400)    


# plot_rms_aoa_az_cdf(gen_data_inv, true_data_inv, 160, save_fig=False)
'''
Function of computing rms spread
'''
def rms_ex_spread(data, power):
    """
    Input: data: array (npath,)
           data_mean: float
           power: array (npath,) in mW
    Return: rms excess spread
    """
    
    rms_mean = np.sum(power * data) / np.sum(power)
    
    numerator = np.sum(power * ((data-rms_mean)**2))
    denominator = np.sum(power)   
    
    return np.sqrt(numerator/denominator)


"""
Plots of RMS AoA az zeros by links
"""
def plot_rms_aoa_az_cdf_links(data, true_data, loss_TH = int(160), save_fig = False):
    
    # Plot true data
    # compute f1
    rms_f1_plot = []
    nlink = len(true_data)
    for ilink in range(nlink):
        I = np.where((true_data[ilink,:,10] != -1)
                     & (true_data[ilink,:,10] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(true_data[ilink, I, 2])
            powers = np.squeeze(true_data[ilink, I, 10])
            powers = 10**(0.1 * powers)
            rms_f1_plot.append(rms_ex_spread(angs, powers))
    
    # compute f2
    rms_f2_plot = []
    nlink = len(true_data)
    for ilink in range(nlink):
        I = np.where((true_data[ilink,:,16] != -1)
                     & (true_data[ilink,:,16] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(true_data[ilink, I, 2])
            powers = np.squeeze(true_data[ilink, I, 16])
            powers = 10**(0.1 * powers)
            rms_f2_plot.append(rms_ex_spread(angs, powers))
        
    # plot CDF
    plt.figure()
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b-.')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r-.')
    
    # Plot generated data
    # compute f1
    rms_f1_plot = []
    nlink = len(data)
    for ilink in range(nlink):
        I = np.where((data[ilink,:,10] != -1)
                     & (data[ilink,:,10] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(data[ilink, I, 2])
            powers = np.squeeze(data[ilink, I, 10])
            powers = 10**(0.1 * powers)
            rms_f1_plot.append(rms_ex_spread(angs, powers))
    
    # compute f2
    rms_f2_plot = []
    nlink = len(data)
    for ilink in range(nlink):
        I = np.where((data[ilink,:,16] != -1) 
                     & (data[ilink,:,16] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(data[ilink, I, 2])
            powers = np.squeeze(data[ilink, I, 16])
            powers = 10**(0.1 * powers)
            rms_f2_plot.append(rms_ex_spread(angs, powers))
        
    # plot CDF
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r')

    
    plt.legend(["28 GHz True", "140 GHz True",
                "28 GHz Generated", "140 GHz Generated"], loc='lower right')
    plt.grid()
    plt.xlabel("RMS Azimuth AoA Spread")
    plt.title(f"RMS Azimuth AoA Spread per Link \n (considering clusters' path loss < {loss_TH} dB)")
    plt.ylabel("CDF")
    plt.ylim([0,1])
    plt.xlim([0,150])
    if save_fig:
        plt.savefig("./plot/aoa_az_rms_spread_links.png", dpi = 400)

# plot_rms_aoa_az_cdf_links(gen_data_inv, true_data_inv, 160, save_fig=True)


"""
Plots of RMS AoD az zeros by links
"""
def plot_rms_aod_az_cdf_links(data, true_data, loss_TH = int(160), save_fig = False):
    
    # Plot true data
    # compute f1
    rms_f1_plot = []
    nlink = len(true_data)
    for ilink in range(nlink):
        I = np.where((true_data[ilink,:,10] != -1)
                     & (true_data[ilink,:,10] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(true_data[ilink, I, 0])
            powers = np.squeeze(true_data[ilink, I, 10])
            powers = 10**(0.1 * powers)
            rms_f1_plot.append(rms_ex_spread(angs, powers))
    
    # compute f2
    rms_f2_plot = []
    nlink = len(true_data)
    for ilink in range(nlink):
        I = np.where((true_data[ilink,:,16] != -1)
                     & (true_data[ilink,:,16] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(true_data[ilink, I, 0])
            powers = np.squeeze(true_data[ilink, I, 16])
            powers = 10**(0.1 * powers)
            rms_f2_plot.append(rms_ex_spread(angs, powers))
        
    # plot CDF
    plt.figure()
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b-.')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r-.')
    
    # Plot generated data
    # compute f1
    rms_f1_plot = []
    nlink = len(data)
    for ilink in range(nlink):
        I = np.where((data[ilink,:,10] != -1)
                     & (data[ilink,:,10] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(data[ilink, I, 0])
            powers = np.squeeze(data[ilink, I, 10])
            powers = 10**(0.1 * powers)
            rms_f1_plot.append(rms_ex_spread(angs, powers))
    
    # compute f2
    rms_f2_plot = []
    nlink = len(data)
    for ilink in range(nlink):
        I = np.where((data[ilink,:,16] != -1) 
                     & (data[ilink,:,16] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(data[ilink, I, 0])
            powers = np.squeeze(data[ilink, I, 16])
            powers = 10**(0.1 * powers)
            rms_f2_plot.append(rms_ex_spread(angs, powers))
        
    # plot CDF
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r')

    
    plt.legend(["28 GHz True", "140 GHz True",
                "28 GHz Generated", "140 GHz Generated"], loc='lower right')
    plt.grid()
    plt.xlabel("RMS Azimuth AoD Spread")
    plt.title(f"RMS Azimuth AoD Spread per Link \n (considering clusters' path loss < {loss_TH} dB)")
    plt.ylabel("CDF")
    plt.ylim([0,1])
    plt.xlim([0,150])
    if save_fig:
        plt.savefig("./plot/aod_az_rms_spread_links.png", dpi = 400)

# plot_rms_aod_az_cdf_links(gen_data_inv, true_data_inv, 160, save_fig=True)

"""
Plots of RMS delay zeros per link
"""
def plot_rms_dly_cdf_links(data, true_data, loss_TH = int(160), save_fig = False):
    
    # Plot true data
    # compute f1
    rms_f1_plot = []
    nlink = len(true_data)
    for ilink in range(nlink):
        I = np.where((true_data[ilink,:,10] != -1)
                     & (true_data[ilink,:,10] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(true_data[ilink, I, 4])
            powers = np.squeeze(true_data[ilink, I, 10])
            powers = 10**(0.1 * powers)
            rms_f1_plot.append(rms_ex_spread(angs, powers))
    
    # compute f2
    rms_f2_plot = []
    nlink = len(true_data)
    for ilink in range(nlink):
        I = np.where((true_data[ilink,:,16] != -1)
                     & (true_data[ilink,:,16] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(true_data[ilink, I, 4])
            powers = np.squeeze(true_data[ilink, I, 16])
            powers = 10**(0.1 * powers)
            rms_f2_plot.append(rms_ex_spread(angs, powers))
        
    # plot CDF
    plt.figure()
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b-.')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r-.')
    
    # Plot generated data
    # compute f1
    rms_f1_plot = []
    nlink = len(data)
    for ilink in range(nlink):
        I = np.where((data[ilink,:,10] != -1)
                     & (data[ilink,:,10] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(data[ilink, I, 4])
            powers = np.squeeze(data[ilink, I, 10])
            powers = 10**(0.1 * powers)
            rms_f1_plot.append(rms_ex_spread(angs, powers))
    
    # compute f2
    rms_f2_plot = []
    nlink = len(data)
    for ilink in range(nlink):
        I = np.where((data[ilink,:,16] != -1) 
                     & (data[ilink,:,16] >= (23 - loss_TH)))[0]
        if len(I) > 0:
            angs = np.squeeze(data[ilink, I, 4])
            powers = np.squeeze(data[ilink, I, 16])
            powers = 10**(0.1 * powers)
            rms_f2_plot.append(rms_ex_spread(angs, powers))
        
    # plot CDF
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r')

    
    plt.legend(["28 GHz True", "140 GHz True",
                "28 GHz Generated", "140 GHz Generated"], loc='lower right')
    plt.grid()
    plt.xlabel("RMS Delay Spread")
    plt.title(f"RMS Delay Spread per Link \n (considering clusters' path loss < {loss_TH} dB)")
    plt.ylabel("CDF")
    plt.ylim([0,1])
    plt.xlim([0,2e-7])
    if save_fig:
        plt.savefig("./plot/dly_rms_spread_links.png", dpi = 400)

# plot_rms_dly_cdf_links(gen_data_inv, true_data_inv, 160, save_fig=True)





"""
Plot Delay
"""
def plot_dly_cdf(data, true_data, save_fig = False):
    
    # plot true data
    dly = true_data[:,:,4]
    nlink = len(true_data)
    dly_plot = []
    for ilink in range(nlink):
        I = np.where(dly[ilink,:] != -1)[0]
        dly_plot.extend(dly[ilink,I])
    plt.figure()
    
    nI = len(dly_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(dly_plot), I)
    
    dly = data[:,:,4]
    nlink = len(data)
    dly_plot = []
    for ilink in range(nlink):
        I = np.where(dly[ilink,:] != -1)[0]
        dly_plot.extend(dly[ilink,I])
    # plot CDF
    nI = len(dly_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(dly_plot), I)
    
    
    # plt.legend(["Generated", "True"])
    plt.legend(["True", "Generated"])
    plt.grid()
    plt.xlabel("Delay (second)")
    plt.ylabel("CDF")
    # plt.title("Path Mean Delay of Clusters")
    if save_fig:
        plt.savefig("./plot/delay_cdf.png", dpi = 400)


# plot_dly_cdf(gen_data_inv, true_data_inv, True)


"""
Plot RMS Delay
"""
def plot_rms_dly_cdf(data, true_data, loss_TH = int(160), save_fig = False):
    
    # plot true data
    rms_f1 = true_data[:,:,9]
    rms_f2 = true_data[:,:,15]
    rms_f1_plot = []
    rms_f2_plot = []
    power_f1 = true_data[:,:,10]
    power_f2 = true_data[:,:,16]
    nlink = len(true_data)
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
        
        if len(I_f1) > 0:
            for idx in I_f1:
                rms_f1_plot.append(rms_f1[ilink, idx])
        if len(I_f2) > 0:
            for idx in I_f2:
                rms_f2_plot.append(rms_f2[ilink, idx])
        
    # plot CDF
    plt.figure()
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b-.')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r-.')
    
    
    # plot GAN data
    rms_f1 = data[:,:,9]
    rms_f2 = data[:,:,15]
    rms_f1_plot = []
    rms_f2_plot = []
    power_f1 = data[:,:,10]
    power_f2 = data[:,:,16]
    nlink = len(data)
    for ilink in range(nlink):
        I_f1 = np.where(power_f1[ilink,:] >= 23 - loss_TH)[0]
        I_f2 = np.where(power_f2[ilink,:] >= 23 - loss_TH)[0]
        
        if len(I_f1) > 0:
            for idx in I_f1:
                rms_f1_plot.append(rms_f1[ilink, idx])
        if len(I_f2) > 0:
            for idx in I_f2:
                rms_f2_plot.append(rms_f2[ilink, idx])
    
    # 28 
    nI = len(rms_f1_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f1_plot), I, 'b')
    # 140 
    nI = len(rms_f2_plot)
    I = np.arange(nI)/nI
    plt.plot(np.sort(rms_f2_plot), I,'r')
    
    plt.legend(["28 GHz True", "140 GHz True",
                "28 GHz Generated", "140 GHz Generated"])
    plt.grid()
    plt.xlabel("RMS Delay Spread")
    plt.ylabel("CDF")
    plt.xlim([0,8e-8])
    plt.ylim([0.75,1.0])
    plt.title(f"RMS Delay Spread by Cluster (path loss<{loss_TH} dB)")    
    if save_fig:
        plt.savefig("./plot/delay_rms_cdf.png", dpi = 400)
    
# plot_rms_dly_cdf(gen_data_inv, true_data_inv, 160, save_fig=True)

























