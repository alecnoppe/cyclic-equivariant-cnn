import torch
import torch.nn as nn
from src.models.ConvBlock import ConvBlock
from src.D4 import D4_Slice, D4_Pool, D4_Roll

class D4_CNN(nn.Module):
    def __init__(self, in_channels, block_sizes, activation, pooling_fn, input_dim=(28,28)):
        """
        CNN model with D4 operations

        Args:
            in_channels: int; number of input channels
            block_sizes: list of int; number of output channels for each ConvBlock
            activation: torch.nn.Module; activation function
            pooling_fn: torch.nn.Module; pooling function        
            input_dim: tuple of int; input dimensions
        """
        super().__init__()
        self.d4_slice = D4_Slice()
        self.d4_pool = D4_Pool()
        self.d4_roll = D4_Roll()

        self.blocks = nn.ModuleList()
        for i, block_size in enumerate(block_sizes):
            in_channels = in_channels if i == 0 else block_sizes[i-1]*8
            self.blocks.append(ConvBlock(in_channels, block_size, 3, 1, 1, activation, pooling_fn))
            self.blocks.append(self.d4_roll)
        
        self.out_dim = (input_dim[0]//(2**len(block_sizes)), input_dim[1]//(2**len(block_sizes)))

        self.fc = nn.Linear(block_sizes[-1]*8*self.out_dim[0]*self.out_dim[1], 10)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x: torch.Tensor; input tensor

        Returns:
            torch.Tensor; output tensor (predicted label probabilities)
        """
        x = self.d4_slice(x)
        
        for block in self.blocks:
            x = block(x)

        x = self.d4_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.final_activation(x)
        return x
