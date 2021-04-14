
# coding: utf-8

# In[4]:


#                 SAE, Artificial Neural Network
#
# Artitficial neural network is built in order to obtain statistical quantities
# from few molecular samples using Deep Autoencoder network
# 09/11/2018
#          				**Author**
#				     Alireza Moradzadeh
#			 University of Illinois at Urbana-Champaign
# 				   Mechanical Engineering
#				    moradza2@illinois.edu
#				     GitHub Repository
#					     moradza
#				       Version 1.0
#					      2018
#				    Single Bead model
import csv
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import timeit
from collections import OrderedDict
import pickle
import argparse
from dae import *
from utils import *

train_features = 303
hidden_width = [200,125, 75, 20, 75, 125, 200]
batch_size = 1

Den_True = True
Temp_True = True
shuffle = True

mode='predict' # predict # train

folder_path = os.getcwd()
cwd = folder_path
print(cwd)

save_dir = os.path.join(cwd, 'checkpoint/checkpoint.ckpt')
load_dir = os.path.join(cwd, 'checkpoint')
print( 'Current working directory:  ' , cwd)
if not tf.test.gpu_device_name():
    print('No GPU Find for training')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

if Den_True:
    quant_dens_Max, quant_dens_min = {}, {}
    quant_dens_Max, quant_dens_min = get_density_range(folder_path, 'Den.csv')
    print("Density quants : ")
    print(quant_dens_Max)
    print(quant_dens_min)

if Temp_True:
    quant_temp_Max, quant_temp_min = {}, {}
    quant_temp_Max, quant_temp_min = get_temperature_range(folder_path, 'Temp.csv')
    print("Temperature quants : ")
    print(quant_temp_Max)
    print(quant_temp_min)

mu = np.loadtxt('mean_rdf.txt')
std = np.loadtxt('std_rdf.txt')

# train function

def train(model, train_rdf_rd, train_temp_rd, train_dens_rd,val_rdf_rd, val_temp_rd, val_dens_rd ,  load_dir=load_dir,           learning_rate=0.0001, batch_size=batch_size, keep_prob =0.75,
          num_steps=100000, save_step=100, show_step=10, check_name='checkpoint.ckpt'):
    """Implements the training loop of mini-batch gradient descent.

    Performs mini-batch gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(VariationalAutoencoder): Initialized VAE model.
        dataset: dataset.
        learning_rate(float): Learning rate.
        batch_size(int): Batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    data_set_size = train_rdf_rd.n
    num_batch = int(train_rdf_rd.n/batch_size)
    print("number of batch: ", num_batch, "  , batch size: ", batch_size," , dataset size: ", train_rdf_rd.n )
    loss_train = []
    loss_val = []
    if os.path.exists(load_dir):
        os.chdir(load_dir)
        model.load(load_dir)
        
    if os.path.exists('iteration.dat'):
        cur_step = int(np.loadtxt('iteration.dat'))
    else:
        cur_step = 0
    for step in range(cur_step, num_steps):
        
        # Batch and Sample
        loss_train_batch = 0.0
        loss_val_batch = 0.0
        start = timeit.default_timer()
#         for batch in range(num_batch):
        batch_x_rdf, _ =  train_rdf_rd.next()
        batch_x_rdf = batch_x_rdf[:,:,50:]
        batch_x_temp, _ =  train_temp_rd.next()
        batch_x_dens, _ =  train_dens_rd.next()
        batch_x_all = combine3(batch_x_rdf, batch_x_temp, batch_x_dens)
        
        batch_x = batch_x_all[1:,:,:].reshape((-1, model._ndims), order='F')
        batch_y = np.array([batch_x_all[0,:,:] for i in range(batch_x_all.shape[0]-1)]).                                                        reshape((-1, model._ndims),order='F')
        
        model.session.run(model.update_op_tensor,feed_dict={model.x_placeholder: batch_x,                                                          model.learning_rate_placeholder: learning_rate,                                                        model.y_placeholder: batch_y, model.keep_prob:keep_prob})

        if step % show_step ==0:
            loss_train_batch = model.session.run(model.loss_tensor,feed_dict={model.y_placeholder: batch_y,                                                                                        model.x_placeholder: batch_x, model.keep_prob:1})
        if not os.path.exists(load_dir):
            os.makedirs(load_dir)
            
        if step % show_step ==0:
            data_set_size_val = val_rdf_rd.n
            num_batch_val = int(val_rdf_rd.n/batch_size)
#             for batch in range(num_batch_val):
            vbatch_x_rdf, _ =  val_rdf_rd.next()
            vbatch_x_rdf = vbatch_x_rdf[:,:,50:]
            vbatch_x_temp, _ =  val_temp_rd.next()
            vbatch_x_dens, _ =  val_dens_rd.next()
            vbatch_x_all = combine3(vbatch_x_rdf, vbatch_x_temp, vbatch_x_dens)
            
            vbatch_x = vbatch_x_all[1:,:,:].reshape((-1, model._ndims), order='F')
            vbatch_y = np.array([vbatch_x_all[0,:,:] for i in range(vbatch_x_all.shape[0]-1)]).                                                                reshape((-1, model._ndims),order='F')
           
            
            loss_val_batch = model.session.run(model.loss_tensor,feed_dict={model.y_placeholder: vbatch_y,                                                                                 model.x_placeholder: vbatch_x                                                                                 ,model.keep_prob:1.0})
                
            stop = timeit.default_timer()
            loss_train.append(loss_train_batch)
            loss_val.append(loss_val_batch)
            print("Progress: ", str(100*(step+1)/num_steps)," % .. Training loss: ", str(loss_train[-1]),                  " .. Validation loss: ", str(loss_val[-1])," .. Processing Time: ", str(stop - start), "  second. " )

        if step % save_step == 0:
            save_dir = os.path.join(load_dir, check_name)
            model.save( save_dir, step)
            os.chdir(load_dir)
            np.savetxt('iteration.dat', np.array([step]), fmt='%i')

    return np.array(loss_train), np.array(loss_val)
# Create Reader for Training and Validation Dataset and Testing Data set
train_rdf_rd  = rdf_reader(mu=mu, std=std, normalize=True,batch_size=batch_size)
train_temp_rd = temp_reader(quant_temp_Max=quant_temp_Max, quant_temp_min=quant_temp_min)
train_dens_rd = dens_reader(quant_dens_Max=quant_dens_Max, quant_dens_min=quant_dens_min)

# Create the Model
hidden_width = [250,200, 150, 100, 150, 200, 250]

model = Autoencoder(hidden_width=hidden_width)
if mode == 'train':
    losses_train, losses_validation = train(model, train_rdf_rd, train_temp_rd, train_dens_rd,val_rdf_rd, val_temp_rd, val_dens_rd, num_steps=args.num_step )
    # Save losses of dataset
    train_losses = pd.DataFrame(losses_train)
    val_losses = pd.DataFrame(losses_validation)
    with open('train_losses.dat', 'a') as f:
        np.savetxt(f, train_losses.values, fmt='%2.10f')
        f.close()
    with open('val_losses.dat', 'a') as f:
        np.savetxt(f, val_losses.values, fmt='%2.10f')
        f.close()
    print('Done with trianing of network and saving losses')
if mode == 'predict':
    model.load(load_dir,ct = 'checkpoint.ckpt-0.meta', step=6000000)
    vbatch_x_rdf, _ =  train_rdf_rd.next()
    vbatch_x_rdf = vbatch_x_rdf[:,:,50:]
    vbatch_x_temp, _ =  train_temp_rd.next()
    vbatch_x_dens, _ =  train_dens_rd.next()
    vbatch_x_all = combine3(vbatch_x_rdf, vbatch_x_temp, vbatch_x_dens)
    vbatch_x = vbatch_x_all[1:,:,:].reshape((-1, model._ndims), order='F')
    #vbatch_y = np.array([vbatch_x_all[0,:,:] for i in range(vbatch_x_all.shape[0]-1)]).                                                                reshape((-1, model._ndims),order='F')
    batch_x = vbatch_x[10,:].reshape((1,-1))
    #batch_y = vbatch_y[1000,:].reshape((1,-1))
    
    predicted_rdfs = model.generate_samples(batch_x)


