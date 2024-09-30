# Make MNIST dataset with 90 degree rotations
# and reflections

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os


def make_mnist_dataset():
    data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    X, y = data.data, data.targets
    if len(X.shape) == 3:
        X = X.unsqueeze(1).float() / 255
    return X, y

def rotate_Xy(X, y):
    k = np.random.randint(0, 3, size=len(X))
    for i in range(len(X)):
        X[i] = torch.rot90(X[i], k=k[i], dims=[1, 2])

    return X, y

def reflect_Xy(X, y):
    for i in range(len(X)):
        if np.random.rand() > 0.5:
            X[i] = torch.flip(X[i], dims=[2])
    return X, y

def plot_sample(X, y, index):
    plt.imshow(X[index].squeeze(), cmap='gray')
    plt.colorbar()
    plt.title(y[index].item())
    plt.show()

if __name__ == "__main__":
    X, y = make_mnist_dataset()
    X, y = rotate_Xy(X, y)
    # Make 60, 20, 20 split
    N = len(X)
    X_train, y_train = X[:N//10*6], y[:N//10*6]
    X_val, y_val = X[N//10*6:N//10*8], y[N//10*6:N//10*8]
    X_test, y_test = X[N//10*8:], y[N//10*8:]

    # Store the data under data/MNIST/C4/
    os.makedirs('data/MNIST/C4/', exist_ok=True)
    torch.save((X_train, y_train), 'data/MNIST/C4/training.pt')
    torch.save((X_val, y_val), 'data/MNIST/C4/validation.pt')
    torch.save((X_test, y_test), 'data/MNIST/C4/test.pt')
    
    X_r, y_r = reflect_Xy(X, y)
    # Make 60, 20, 20 split
    N = len(X_r)
    X_train, y_train = X_r[:N//10*6], y_r[:N//10*6]
    X_val, y_val = X_r[N//10*6:N//10*8], y_r[N//10*6:N//10*8]
    X_test, y_test = X_r[N//10*8:], y_r[N//10*8:]

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape:", X_val.shape, y_val.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    # Store the data under data/MNIST/D4/
    os.makedirs('data/MNIST/D4/', exist_ok=True)
    torch.save((X_train, y_train), 'data/MNIST/D4/training.pt')
    torch.save((X_val, y_val), 'data/MNIST/D4/validation.pt')
    torch.save((X_test, y_test), 'data/MNIST/D4/test.pt')

    print("Data stored under data/MNIST/C4/ and data/MNIST/D4/")

    # Plot a sample
    plot_sample(X, y, 0)

    plot_sample(X_r, y_r, 30)

    print("Done!")