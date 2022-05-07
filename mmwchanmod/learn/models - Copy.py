"""
models.py:  Classes for the modeling, partial code reference from 
    1) https://github.com/nyu-wireless/mmwchanmod
    2) https://keras.io/examples/generative/wgan_gp/

"""
import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import numpy as np
import sklearn.preprocessing
import pickle
import os

from mmwchanmod.learn.preproc_param import preproc_to_param, param_to_preproc
    

class CondGAN(object):
    '''
    Object for bulit GAN
    '''
    def __init__(self, nlatent, nparams, ncond,\
         nunits_dsc=(1120,560,280,), nunits_gen=(280,560,1120)):

        self.nlatent = nlatent
        self.nparams = nparams        
        self.ncond = ncond
        self.nunits_dsc = nunits_dsc
        self.nunits_gen = nunits_gen

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):

        # conditional input
        cond = tfkl.Input((self.ncond,), name='cond')
        n_nodes = self.nunits_dsc[-1]
        li = tfkl.Dense(n_nodes)(cond)
        li = tfkl.Reshape((n_nodes,1))(li)

        # channel parameter input
        in_lat = tfkl.Input((self.nlatent,), name='in_lat')
        gen = tfkl.Dense(n_nodes)(in_lat)
        gen = tfkl.Reshape((n_nodes,1))(gen)

        # merge channel gen and condition
        merge = tfkl.Concatenate(name='gen_cond')([gen, li])
        gen = tfkl.Flatten()(merge)
        gen = tfkl.LeakyReLU(alpha=0.2)(gen)
        gen = tfkl.BatchNormalization()(gen)
        layer_names = []
        for i in range(len(self.nunits_gen)):           
            gen = tfkl.Dense(self.nunits_gen[i],\
                           name='FC%d' % i)(gen)
            gen = tfkl.LeakyReLU(alpha=0.2)(gen)
            layer_names.append('FC%d' % i)

        # output
        out_layer = tfkl.Dense(self.nparams)(gen)
        # define model
        g_model = tfk.Model([in_lat, cond], out_layer)

        # save network architecture
        # dot_img_file = 'nnArch/gen.png'
        # tf.keras.utils.plot_model(g_model, to_file=dot_img_file, show_shapes=True)   

        return g_model

    def build_discriminator(self):
        # conditional input
        cond = tfkl.Input((self.ncond,), name='cond')

        # scale up dimensions with linear activation
        n_nodes = self.nparams
        li = tfkl.Dense(n_nodes)(cond)
        # reshape to additional channel
        li = tfkl.Reshape((self.nparams,1))(li)

        # channel parameter input
        x = tfkl.Input((self.nparams,1), name='x')

        # concat data and condition
        dat_cond = tfkl.Concatenate(name='dat_cond')([x, li])
        fe = tfkl.Flatten()(dat_cond)
        layer_names = []
        for i in range(len(self.nunits_dsc)):           
            fe = tfkl.Dense(self.nunits_dsc[i],
                           name='FC%d' % i)(fe)
            fe = tfkl.LeakyReLU(alpha=0.2)(fe)
            fe = tfkl.Dropout(0.3)(fe)
            layer_names.append('FC%d' % i)

        # output
        out_layer = tfkl.Dense(1, activation='linear')(fe)
        # define model
        d_model = tfkm.Model([x, cond], out_layer)
        
        # save network architecture
        # dot_img_file = 'nnArch/dsc.png'
        # tf.keras.utils.plot_model(d_model, to_file=dot_img_file, show_shapes=True) 

        return d_model
    
    
class ChanMod(object):
    """
    Object for modeling mmWave channel model data.
    
    There is one part in the model:
        * path_mod:  This predicts the other channel parameters (clusters) from the condition and link_state.
        
    Each model has a pre-processor on the data and conditions that is also
    trained.
          
    """  
    def __init__(self, nlatent=50, model_dir='models'):
        """
        Constructor

        Parameters
        ----------
        nlatent : int
            number of latent states in the GAN model 
        model_dir : string
            path to the directory for all the model files.
            if this path does not exist, it will be created 
        """        
        self.ndim = 3  # number of spatial dimensions
        self.model_dir = model_dir
        self.nlatent = nlatent

        # Arbitrary Freq
        self.nfreq = 2
        self.nparams = 10*(2*6+5)+3
        
        # File names
        self.loss_hist_fn = 'loss_hist.p'
        self.path_weights_fn='path_weights.h5'
        self.path_preproc_fn='path_preproc.p'
        
        # Version number for the pre-processing file format
        self.version = 0
    
    def build_path_mod(self):
        """
        Builds the GAN for the NLOS paths
        """
        
        # Number of data inputs in the transformed domain
        # For each sample and each path, there are ten clusters.
        # For each cluster, it includes:
        # * four average angles: az-aod, inc-aod, az-aoa, inc-aoa
        # * one minimum delay: min_dly
        # For each frequency inside a cluster, it has:
        # * our angles rms spread: az-aod, inc-aod, az-aoa, inc-aoa
        # * one delay rms spread
        # * one total power of in mW
        # for a total of 10*(5+6*2)=170 parameters

        
        # Number of condition variables
        #   * d3d
        #   * log10(d3d)

        self.ncond = 2

        self.path_mod = CondGAN(\
            nlatent=self.nlatent, ncond=self.ncond,\
            nparams=self.nparams)
    
    def fit_path_mod(self, train_data, test_data, epochs=50, lr=1e-4,\
                     checkpoint_period = 100, save_mod=True, batch_size=512,\
                     d_steps=5, gp_weight=10):
        """
        Trains the path model
        

        Parameters
        ----------
        train_data : dictionary
            training data dictionary.
        test_data : dictionary
            test data dictionary. 
        epochs: int
            number of training epochs
        lr: scalar
            learning rate
        checkpoint_period:  int
            period in epochs for saving the model checkpoints.  
            A value of 0 indicates that checkpoints are not be saved.
        save_mod:  boolean
            Indicates if model is to be saved
        batch_size: int
        d_steps: int
            train the discriminator for d_steps more steps 
            as compared to one step of the generator
        gp_weight: int
            gradient penalty weights
        """
        
        # Extract the links that are in LOS or NLOS
        # ls_tr = train_data['link_state']
        # Itr = np.where(ls_tr != LinkState.no_link)[0]
        
        # Fit and transform the condition data
        Utr = self.transform_cond(train_data['dvec'], fit=True)
        
        # Xtr : (nlink, nparams) array, training data
        Xtr = train_data['data']
        
        # Save the pre-processor
        if save_mod:
            self.save_path_preproc()
        # Create the file paths
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        weigths_path = self.model_dir
               
        # Build the discriminator and the generator
        discriminator = self.path_mod.discriminator
        generator = self.path_mod.generator

        generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0, beta_2=0.9)
        discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0, beta_2=0.9)
        
        # Train the discriminator and the generator
        gen_loss = []
        dsc_loss = []
        for epoch in range(epochs):

            I = np.random.permutation(Xtr.shape[0])

            nsteps = len(I)//batch_size

            # For each batch, we are going to perform the
            # following steps as laid out in the original paper:
            # 1. Train the generator and get the generator loss
            # 2. Train the discriminator and get the discriminator loss
            # 3. Calculate the gradient penalty
            # 4. Multiply this gradient penalty with a constant weight factor
            # 5. Add the gradient penalty to the discriminator loss
            # 6. Return the generator and discriminator losses as a loss dictionary

            # Train the discriminator first. The original paper recommends training
            # the discriminator for `x` more steps (typically 5) as compared to
            # one step of the generator.

            for i in range(nsteps):
                
                idx = I[i*batch_size:(i+1)*batch_size]
                Xtrain, labels = Xtr[idx], Utr[idx]

                for j in range(d_steps):
                    # Get the latent vector
                    z = np.random.normal(0, 1, size=(batch_size, self.nlatent))

                    with tf.GradientTape() as tape:
                        # Generate fake channels from the latent vector
                        fake_chans = generator([z, labels], training=True)
                        # Get the logits for the fake channels
                        fake_logits = discriminator([fake_chans, labels], training=True)
                        # Get the logits for the real channels
                        real_logits = discriminator([Xtrain, labels], training=True)

                        # Calculate the discriminator loss using the fake and real channel logits
                        d_cost = self.discriminator_loss(real_logits, fake_logits)
                        # Calculate the gradient penalty
                        gp = self.gradient_penalty(batch_size, Xtrain, fake_chans, labels)
                        # Add the gradient penalty to the original discriminator loss
                        d_loss = d_cost + gp * gp_weight

                    # Get the gradients w.r.t the discriminator loss
                    d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
                    # Update the weights of the discriminator using the discriminator optimizer
                    discriminator_optimizer.apply_gradients(zip(d_gradient, discriminator.trainable_variables))

                # Train the generator
                # Get the latent vector
                z = np.random.normal(0, 1, size=(batch_size, self.nlatent))
                with tf.GradientTape() as tape:
                    # Generate fake channels using the generator
                    generated_chans = generator([z, labels], training=True)
                    # Get the discriminator logits for fake channels
                    gen_chan_logits = discriminator([generated_chans, labels], training=True)
                    # Calculate the generator loss
                    g_loss = self.generator_loss(gen_chan_logits)

                # Get the gradients w.r.t the generator loss
                gen_gradient = tape.gradient(g_loss, generator.trainable_variables)
                # Update the weights of the generator using the generator optimizer
                generator_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))

                #save and print generator and discriminator loss
                gen_loss.append(g_loss.numpy())
                dsc_loss.append(d_loss.numpy())
            
            tf.print(f'Epoch:{epoch} G_loss: {g_loss} D_loss: {d_loss}')

            if epoch % checkpoint_period == 0:
                self.path_mod.generator.save(weigths_path+f'/generator-epochs-{epoch}.h5')
            
        # Save the weights model
        if save_mod:
            self.save_path_model()   

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Save loss
        loss_hist_path = os.path.join(self.model_dir, self.loss_hist_fn)
        with open(loss_hist_path,'wb') as loss_fp:
            pickle.dump(\
                [gen_loss, dsc_loss], loss_fp)
        
    def transform_cond(self, dvec, fit=False):
        """
        Pre-processing transform on the condition

        Parameters
        ----------
        dvec : (nlink,ndim) array
            vector from cell to UAV
        fit : boolean
            flag indicating if the transform should be fit

        Returns
        -------
        U : (nlink,ncond) array
            Transform conditioned features
        """                     
        
        # 3D distance 
        d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)
        d3d = np.round(d3d)
        
        # Transform the condition variables
        U0 = np.column_stack((d3d, np.log10(d3d)))
        self.ncond = U0.shape[1]
        
        # Transform the data with the scaler.
        # If fit is set, the transform is also learned
        if fit:
            self.cond_scaler = sklearn.preprocessing.StandardScaler()
            U = self.cond_scaler.fit_transform(U0)
        else:
            U = self.cond_scaler.transform(U0)
            
        return U  
    
    def save_path_preproc(self):
        """
        Saves path preprocessor
        """
        # Create the file paths
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        preproc_path = os.path.join(self.model_dir, self.path_preproc_fn)
        
        # Serializing sklearn objects may not be valid if the de-serializing
        # program has a different sklearn version.  So, we store parameters
        # of the pre-processors instead
        cond_param = preproc_to_param(self.cond_scaler, 'StandardScaler')
        
        # Save the pre-processors
        with open(preproc_path,'wb') as fp:
            pickle.dump([self.version, cond_param, self.nlatent], fp)
        
    def discriminator_loss(self, real_logits, fake_logits):
        # Define the loss functions for the discriminator,
        # which should be (fake_loss - real_loss).
        # We will add the gradient penalty later to this loss function.
        real_loss = tf.reduce_mean(real_logits)
        fake_loss = tf.reduce_mean(fake_logits)
        total_loss = fake_loss - real_loss
        return total_loss

    def generator_loss(self, fake_logits):
        return -tf.reduce_mean(fake_logits)

    def gradient_penalty(self, batch_size, real, fake, conds):
        epsilon =  tf.random.normal([batch_size,1], 0.0, 1.0)
        diff = real-fake
        interpolated = fake - epsilon*diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.path_mod.discriminator([interpolated, conds], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def save_path_model(self):
        """
        Saves model data to files

        """        
        
        # Create the file paths
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        weigths_path = os.path.join(self.model_dir, self.path_weights_fn)

            
        # Save the GAN weights
        self.path_mod.generator.save_weights(weigths_path, save_format='h5')
        
    def load_path_model(self, ckpt=None):
        """
        Load model data from files
        
        Parameters
        ----------
        ckpt : None or int
            If integer, loads a checkpoint file with the epoch number.

        """
        # Create the file paths
        preproc_path = os.path.join(self.model_dir, self.path_preproc_fn)
        if ckpt is None:
            fn = self.path_weights_fn
        else:
            fn = ('path_weights.%d.h5' % ckpt)
        weights_path = os.path.join(self.model_dir, fn)
        
        # Load the pre-processors
        with open(preproc_path,'rb') as fp:
            ver, cond_param, self.nlatent = pickle.load(fp)
                
        # Re-constructor the pre-processors
        self.cond_scaler = param_to_preproc(cond_param, 'StandardScaler')              
            
        # Build the path model
        self.build_path_mod()
            
        # Load the GAN weights
        self.path_mod.generator.load_weights(weights_path)
    
    
    def sample_path(self, dvec):
        """
        Generates random samples of the path data using the trained model

        Parameters
        ----------
        dvec : (nlink,ndim) array
            Vector from cell to UAV
        link_state:  (nlink,) array of {no_link, los_link, nlos_link}            
            A value of `None` indicates that the link state should be
            generated randomly from the link state predictor model
        return_dict:  boolean, default False
            If set, it will return a dictionary with all the values
            Otherwise it will return a channel list
   
        Returns
        -------
        X: (nlink, 173)
        """
        # Get dimensions
        nlink = dvec.shape[0]  
        
        # Get the condition variables and random noise
        U = self.transform_cond(dvec)
        Z = np.random.normal(0,1,(nlink,self.nlatent))
        
        # Run through the generator network
        X = self.path_mod.generator.predict([Z,U]) 

        return X
    
def load_model(mod_name):
    """
    Loads a pre-trained model from local directory
    
    Parameters
    ----------
    mod_name : string
        Model name to be downloaded. 
        
    Returns
    -------
    chan_mod:  ChanMod
        pre-trained channel model
    """    
        
    # Create the local data directory if needed    
    mod_root = os.path.join(os.path.dirname(__file__),'..','..','models')
    mod_root = os.path.abspath(mod_root)
    if not os.path.exists(mod_root):
        os.mkdir(mod_root)
        print('Creating directory %s' % mod_root)
        
    # Check if model directory exists
    mod_dir = os.path.join(mod_root, mod_name)
        
    # Check if model directory exists
    if not os.path.exists(mod_dir):
        raise ValueError('Cannot find model %s' % mod_dir)
        
    # Create the model
    chan_mod = ChanMod(model_dir=mod_dir)
    
    # Load the configuration and link classifier model
    chan_mod.load_path_model()        
    
    return chan_mod  