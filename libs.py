import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm  

import time


'''
    Name: Aws Al Jumaily 
    Student ID: 6459861
    COSC 5P77: Final Assignment 
    
    Note: This code is an extention of the code found at: https://github.com/bacnguyencong/rbm-pytorch/tree/master
'''



def save_image(img, file_name):
    '''
    Show and save the image.

    Args:
        img (Tensor): The image.
        file_name (Str): The destination.

    '''
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name

    plt.imsave(f, npimg)


def make_binary(data, threshold=0.5):
    '''
    Transforms non-binary input data into binary data

    Args:
        data (Tensor): Input data.
        threshold (Int): Value cut-off threshold.

    '''
    return (data >= threshold).float()



def train(model_type, model, dataset, training_log_file, n_epochs=3, lr=0.01):
   '''
   Trains the model. 

    Args:
        model_type (String): RBM or BM.
        model: The model.
        dataset (Tensor): Input data.
        training_log_file (Str): The destination of training log file.
        n_epochs (int): The number of epochs.
        lr (Float): The learning rate. 
        
        
        
    Returns:
        The trained model.

    '''
    train_op = optim.Adam(model.parameters(), lr)
    model.train()

    f = open(training_log_file, 'w')
    start_time = time.time()  
    for epoch in range(n_epochs):
        data_loader = tqdm(dataset, desc=f'Epoch {epoch}', leave=False)
        loss_ = []
        for _, (data, target) in enumerate(data_loader):
            binary_data = make_binary(data.view(-1, 784))
            if model_type == "RBM":
                x, x_gibbs = model(binary_data)    
                loss = model.free_energy(x) - model.free_energy(x_gibbs)
            else:
                x, x_gibbs, h_mu, h_mu_gibbs = model(binary_data)    
                loss = model.energy(x, h_mu) - model.energy(x_gibbs,h_mu_gibbs)         
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()

            end_time = time.time()
            elapsed_time = end_time - start_time
        epoch_info = f'Epoch {epoch}\t Loss={np.mean(loss_):.4f}\t Time={elapsed_time:.2f} seconds'
        print(epoch_info)
        f.write(epoch_info+'\n')


    return model
