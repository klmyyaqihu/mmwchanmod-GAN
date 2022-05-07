# Multi-Freqeuncy Millimeter Wave Channel Modeling via Generative Adversarial Network
* Sundeep Rangan, Yaqi Hu, Mingsheng Yin, William Xia, Marco Mezzavilla (NYU)

The work is extended from [mmwchanmod](https://github.com/nyu-wireless/mmwchanmod), and the major differences are:
* Treat a links as clusters of paths
* Extend the model for multi-frequncies
* Use Generative Adversarial Network (GAN) instead of Variational Autoencoder (VAE)


## Data
First, run kmean_on_paths.py' generete data for train and test clusters data.

## Training model from scratch	
Go to the current folder and run(e.g.):
```
python train_mod_interactive.py --model_dir models/Beijing_test1 --nepochs_path 1000 
```

'train_mod_interactive.py' has commands to change the number of epochs and model parameters.

## The folder 'models' stores a example trained model
'Beijing'

## The folder 'plots' has code for plotting results, and plots
'plots_result.py' is the code for plotting.
The folder 'plot' stores the plots.

## To modify the code for the network, please go to "mmwchanmod/learn/models.py". In "models.py": 
1. the class 'CondGAN' describes the structure of GAN
2. 'fit_path_mod' in class 'ChanMod' is where we train GAN
