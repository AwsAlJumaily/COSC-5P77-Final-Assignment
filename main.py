import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from rbm import RBM
from mean_field_bm import Mean_Field_BM


from libs import train, save_image

'''
    Name: Aws Al Jumaily 
    Student ID: 6459861
    COSC 5P77: Final Assignment 
    
    Note: This code is an extention of the code found at: https://github.com/bacnguyencong/rbm-pytorch/tree/master

'''




batch_size = 64
n_epochs = 10
lr = 0.01
n_hid = 500
n_vis = 784 
gibbs_iteration=3



'''
Uncomment the dataset variable that you would like to use. 
'''
dataset = "MNIST"
#dataset = "Fashion_MNIST"


'''
Uncomment the model_type variable that you would like to use. 
'''
#model_type = "RBM"
model_type = "Mean_Field_BM"



if model_type == "RBM":
    model = RBM(n_vis=n_vis, n_hid=n_hid, k=gibbs_iteration)

elif model_type == "Mean_Field_BM":
    model = Mean_Field_BM(n_vis=n_vis, n_hid=n_hid, k=gibbs_iteration)





if dataset == "MNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size
    )


    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./datasets', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size
    )




elif dataset == "Fashion_MNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                   ])),
        batch_size=batch_size
    )


    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./datasets', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size
    )





print('Model: ', model_type)
print('Dataset: ', dataset)

training_log_file = f"outputs/{dataset}/{dataset}_{model_type}_training_logs.txt"

model = train(model_type, model, train_loader, training_log_file, n_epochs=n_epochs, lr=lr)

images = next(iter(test_loader))[0]
if model_type == "Mean_Field_BM":
    x, x_gibbs, h_mu, h_mu_gibbs = model(images.view(-1, 784), testing=True)
else:
    x, x_gibbs= model(images.view(-1, 784), testing=True)

save_image(make_grid(x.view(batch_size, 1, 28, 28).data), f"outputs/{dataset}/{dataset}_original_test")
save_image(make_grid(x_gibbs.view(batch_size, 1, 28, 28).data), f"outputs/{dataset}/{dataset}_{model_type}_generated_test")


