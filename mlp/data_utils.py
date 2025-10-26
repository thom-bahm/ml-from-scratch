from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_mnist_data(batch_size=64):
    """
    Load and preprocess MNIST dataset.
    
    Returns:
        X_train: Training images 
        y_train: Training labels
        X_test: Test images
        y_test: Test labels
    
    TODO: Implement the data loading logic
    Think about:
    - How to download/load the dataset using torchvision
    - What transformations you need (ToTensor, normalization)
    - How to convert from PyTorch tensors to numpy arrays
    - Whether to flatten the images (28x28 -> 784)
    - How to handle the labels (keep as integers or convert to one-hot?)
    """

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    # print(train_dataset.data.shape)
    # print(train_dataset.targets.shape)
    # print("----")
    # print(test_dataset.data.shape)
    # print(test_dataset.targets.shape)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=0, pin_memory=False)

    return train_loader, test_loader

def preprocess_data(X, y):
    """
    Additional preprocessing steps.
    
    Args:
        X: Input images
        y: Labels
    
    Returns:
        Preprocessed X and y
    
    TODO: Consider what preprocessing you might need:
    - Normalization (0-255 -> 0-1 or standardization?)
    - Reshaping images
    - One-hot encoding labels
    """
    
    # Your implementation here
    pass

if __name__ == "__main__":
    # Test your data loading
    print("Testing data loading...")
    train_loader, test_loader = load_mnist_data()

    images, labels = next(iter(train_loader))
    print("The input data shape is :\n", images.shape)
    print("The target output data shape is :\n", labels.shape)

    plt.figure(figsize = (20,10))
    out = torchvision.utils.make_grid(images, 32)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.show()