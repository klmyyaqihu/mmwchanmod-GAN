# Multi-Freqeuncy Millimeter Wave Channel Modeling via Generative Adversarial Network
* Sundeep Rangan, Yaqi Hu, Mingsheng Yin, William Xia, Marco Mezzavilla (NYU)

The work is extended from [mmwchanmod](https://github.com/nyu-wireless/mmwchanmod), and the major differences are:
* Treat a links as clusters of paths
* Extend the model for multi-frequncies
* Use Generative Adversarial Network (GAN) instead of Variational Autoencoder (VAE)


## Data
'dict.p' (from Beijing map) stores all raw data required for GAN
'data.csv' can show the insights of 'dict.p'

## Training model from scratch	
Go to the current folder and run(e.g.):
```
python train_mod_interactive.py --model_dir models/model_name --nepochs_path 100 
```

'train_mod_interactive.py' has commands to change the number of epochs and model parameters.

## The folder 'models' stores a example trained model
'WGAN_GP'

## plot the path-loss cdf
Go to the current folder and run(e.g.):
```
python plot_path_loss_cdf.py  --model_dir models/model_name --plot_fn plot_name.png
```

## the folder 'nnArch' stores the visualization of current network structures
-- dsc.png is the structure of discriminator
-- gen.png is the structure of generator

## To modify the code for the network, please go to "mmwchanmod/learn/models.py". In "models.py": 
1. the class 'CondGAN' describes the structure of GAN
2. 'fit_path_mod' in class 'ChanMod' is where we train GAN

## plot the SNR
-- model: python plot_snr_cmp_model.py --model_dir models/WGAN-GP --plot_fn xxx.png
-- data: python plot_snr_cmp_data.py --model_dir models/WGAN-GP --plot_fn xxx.png
