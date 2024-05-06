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

class Mean_Field_BM(nn.Module):
    '''
    Boltzmann Machine using mean field.

    Args:
        n_vis (int): The size of visible layer.
        n_hid (int): The size of hidden layer.
        k (int): The number of Gibbs sampling iterations.
    '''
    def __init__(self, n_vis=784, n_hid=128, k=10):
        super(Mean_Field_BM, self).__init__()
        self.k = k
        self.a = nn.Parameter(torch.randn(1, n_vis)) #visible bias
        self.b = nn.Parameter(torch.randn(1, n_hid)) #visible bias
        self.W = nn.Parameter(torch.randn(n_hid, n_vis)) #hidden - visible weight matrix
        self.U = torch.randn(n_vis, n_vis) #visible - visible weight matrix
        self.V = torch.randn(n_hid, n_hid) #hidden - hidden weight matrix

        with torch.no_grad():
            self.U.fill_diagonal_(0)
            self.V.fill_diagonal_(0)

        self.h_mu = torch.randn(1, n_hid) 
        self.x_mu = torch.randn(1, n_vis) 




    def update_h_mu(self, x):
        '''
        Updates global variable h_mu for approximation.

        Args:
            x (Tensor): The visible input.

        '''
        self.h_mu = torch.sigmoid(torch.mean(self.b+ torch.matmul(self.V, self.h_mu.t()),dim=1) + torch.sum(torch.matmul(self.W, x.t()),dim=1)).unsqueeze(0)



    def update_x_mu(self, h):
        '''
        Updates global variable x_mu for approximation.

        Args:
            h (Tensor): The hidden input.

        '''
        self.x_mu = torch.sigmoid(torch.mean(self.a+ torch.matmul(self.U, self.x_mu.t()),dim=1) + torch.sum(torch.matmul(self.W.t(), h.t()),dim=1)).unsqueeze(0)




    def sample_hidden(self, x):
        '''
        Samples hidden layer given visible input x.

        Args:
            x (Tensor): The visible input.

        Returns:
            sample (Tensor): The hidden sample.

        '''
        hidden_1 = torch.matmul(x,self.W.t())+torch.matmul(self.h_mu.clone().detach(), self.V)+self.b 
        hidden = torch.sigmoid(hidden_1) 
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
        visible_1 = torch.matmul(h,self.W)+torch.matmul(self.x_mu.clone().detach(), self.U)+self.a
        visible = torch.sigmoid(visible_1) 
        if testing:
            sample = visible
        else:
           sample = visible.bernoulli()
        return sample


    def energy(self, x, h_mu):
        '''
        Computes the energy based on the visible input x and h_mu.

        Args:
            x (Tensor): The visible input.
            h_mu (Tensor): h_mu associated with x.

        Returns:
            (Int): The energy

        '''  
        first_term = torch.matmul(self.a, x.t())
        second_term =torch.matmul(self.b, h_mu.t())
        third_term = torch.sum(torch.matmul(torch.matmul(x, self.U), x.t()), dim=1)
        fourth_term = torch.sum(torch.matmul(torch.matmul(h_mu, self.V), h_mu.t()), dim=1)
        fifth_term = torch.sum(torch.matmul(F.softplus(torch.matmul(x, self.W.t())), h_mu.t()),  dim=1)

        return -torch.mean(first_term + second_term + 0.5*third_term + 0.5*fourth_term + fifth_term)





    def get_h_functional_entropy(self):
        '''
        Computes the h functional entropy.

        Args:
            h_entropy (Tensor): h functional entropy.

        '''
        log_h_mu = torch.log(self.h_mu + 1e-9)
        h_entropy = -torch.sum((self.h_mu * log_h_mu) + ((1-self.h_mu)*log_h_mu))
        return h_entropy
    




    def get_x_functional_entropy(self):
        '''
        Computes the x functional entropy.

        Args:
            x_entropy (Tensor): x functional entropy.

        '''
        log_x_mu = torch.log(torch.abs(self.x_mu) + 1e-9) * torch.sign(self.x_mu)
        x_entropy = -torch.sum((self.x_mu * log_x_mu) + ((1-self.x_mu)*log_x_mu))
        return x_entropy




    def forward(self, x, testing=False):
        '''
        Reconstructs examples based on k iterations of Gibbs sampling given visible input x.

        Args:
            x (Tensor): The visible input.
            testing (Boolean): Ensures the test output is smooth.

        Returns:
            x, x_gibb, h_mu, h_mu_gibbs (Tensor, Tensor, Tensor, Tensor): The original x, reconstructed x, 
            h_mu at state x, h_mu at state x_gibbs

        '''  
        self.update_h_mu(x)
        h_mu = self.h_mu.clone().detach()
        h = self.sample_hidden(x)
        self.update_x_mu(h)
        for i in range(self.k):
            if i == self.k-1  and testing:
                x_gibbs = self.sample_visible(h, testing=True)
            else:
                x_gibbs = self.sample_visible(h)
            self.update_h_mu(x_gibbs)
            h = self.sample_hidden(x_gibbs)
            self.update_x_mu(h)

        h_mu_gibbs = self.h_mu.clone().detach()

        return x, x_gibbs, h_mu, h_mu_gibbs
