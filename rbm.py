import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
'''
    Name: Aws Al Jumaily 
    Student ID: 6459861
    COSC 5P77: Final Assignment 
    
    Note: This code is an extention of the code found at: https://github.com/bacnguyencong/rbm-pytorch/tree/master

'''

class RBM(nn.Module):
    '''Restricted Boltzmann Machine.

    Args:s
        n_vis (int, optional): The size of visible layer. Defaults to 784.
        n_hid (int, optional): The size of hidden layer. Defaults to 128.
        k (int, optional): The number of Gibbs sampling. Defaults to 1.
    '''

    def __init__(self, n_vis=784, n_hid=128, k=1):
        '''Create a RBM.'''
        super(RBM, self).__init__()
        self.a = nn.Parameter(torch.randn(1, n_vis)) #visible bias
        self.b = nn.Parameter(torch.randn(1, n_hid)) #hidden bias
        self.W = nn.Parameter(torch.randn(n_hid, n_vis)) #hidden - visible weight matrix
        self.k = k #gibbs iteration 

    def sample_hidden(self, x):
        '''
        Samples hidden layer given visible input x.

        Args:
            x (Tensor): The visible input.

        Returns:
            sample (Tensor): The hidden sample.

        '''
        hidden = torch.sigmoid(torch.matmul(x,self.W.t())+self.b)
        sample = hidden.bernoulli()
        return sample

    def sample_visible(self, h, testing=False):
        '''
        Samples visible layer given hidden input h.

        Args:
            h (Tensor): The hidden input.
            testing (Boolean): Ensures the test output is smooth.

        Returns:
            sample (Tensor): The visible sample.

        '''
        visible = torch.sigmoid(torch.matmul(h,self.W)+self.a)
        if testing:
            sample = visible
        else:
           sample = visible.bernoulli()
        return sample
        
        

    def free_energy(self, x):
        '''
        Computes the free energy based on the visible input x.

        Args:
            x (Tensor): The visible input.

        Returns:
            (Int): The free energy

        '''   
        first_term = torch.matmul(x, self.a.t())
        second_term = torch.sum(F.softplus(torch.matmul(x,self.W.t())+self.b),dim=1)

        return torch.mean(-first_term - second_term)


    def forward(self, x, testing=False):
        '''
        Reconstructs examples based on k iterations of Gibbs sampling given visible input x.

        Args:
            x (Tensor): The visible input.
            testing (Boolean): Ensures the test output is smooth.

        Returns:
            x, x_gibb (Tensor, Tensor): The original x and reconstructed x.

        '''  
        h = self.sample_hidden(x)
        for i in range(self.k):
            if i == self.k-1 and testing:
                x_gibbs = self.sample_visible(h, testing=True)
            else:
                x_gibbs = self.sample_visible(h)
            h = self.sample_hidden(x_gibbs)
        return x, x_gibbs
