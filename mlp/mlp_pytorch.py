from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import trange
import data_utils as du
from torch import nn, optim
import torch.nn.functional as F

from test import test_model
from train import train_epoch

# Lets create a simple MLP network similar to the sine wave approximator
class Simple_MLP(nn.Module):
    def __init__(self, num_classes):
        super(Simple_MLP, self).__init__()
        # We will use 4 linear layers
        # The input to the model is 784 (28x28 - the image size)
        # and there should be num_classes outputs

        # hidden layer size 512
        self.fc1 = nn.Linear(784, 512)
        # hidden layer size 256
        self.fc2 = nn.Linear(512, 256)
        # hidden layer size 128
        self.fc3 = nn.Linear(256, 128)
        # output layer size num_classes
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        # The data we pass the model is a batch of single channel images
        # with shape BSx1x28x28 we need to flatten it to BSx784
        # To use it in a linear layer
        x = x.view(x.shape[0], -1) # For 28x28 images, this is equivalent to x.view(x.shape[0], 784)
        
        # We will use a relu activation function for this network! (F.relu)
        # NOTE F.relu is the "functional" version of the activation function!
        # nn.ReLU is a class constructor of a "ReLU" object
        
        # These two things are the same for MOST purposes!
        # TO DO, layer and then activation function
        x = F.relu(self.fc1(x))
        # TO DO, layer and then activation function
        x = F.relu(self.fc2(x))
        # TO DO, layer and then activation function
        x = F.relu(self.fc3(x))
        # TO DO, add layer
        x = self.fc4(x)

        return x


if __name__ == '__main__':
    GPU_indx = 0
    device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
    # Create our model
    model = Simple_MLP(10).to(device)
    # Create our loss function
    criterion = nn.CrossEntropyLoss()
    # Define our loss funcition and optimizer
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Number of Epochs
    n_epochs = 20
    # We can print out our model structure
    print(model)
    # Note: this is only the order in which the layers were defined NOT the path of the forward pass!
    train_loader, test_loader = du.load_mnist_data()

    train_loss = []
    test_loss  = []
    test_acc   = []

    for i in trange(n_epochs, desc="Epoch", leave=False):
        model, optimizer, train_loss = train_epoch(model, train_loader, criterion, optimizer, train_loss, device)
        test_loss, acc = test_model(model, test_loader, criterion, test_loss, device)
        test_acc.append(acc)
        
    print("Final Accuracy: %.2f%%" % acc)

    images, labels = next(iter(train_loader))
    x = model(images.to(device))
    plt.title("Scatterplot for images?")
    plt.scatter(np.arange(10), F.softmax(x, 1)[0].detach().cpu().numpy())
    plt.show()

    plt.title('Train Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.plot(train_loss)
    plt.show()

    plt.title('Test Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.plot(test_loss)

