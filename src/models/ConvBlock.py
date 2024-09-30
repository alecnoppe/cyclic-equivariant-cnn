import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, pooling_fn):
        super().__init__()
        """
        Simple Convolutional block with activation and pooling function

        Input -> Conv -> Activation -> Conv -> Activation -> Pooling

        Use nn.Identity() for no activation function or no pooling function

        Args:
            in_channels: int; number of input channels
            out_channels: int; number of output channels
            kernel_size: int; size of the kernel
            stride: int; stride of the convolution
            padding: int; padding of the convolution
            activation: torch.nn.Module; activation function
            pooling_fn: torch.nn.Module; pooling function
        """
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
        self.pooling_fn = pooling_fn

    def forward(self, x):
        """
        Forward pass of the ConvBlock

        Args:
            x: torch.Tensor; input tensor

        Returns:
            torch.Tensor; output tensor
        """
        out = self.in_conv(x)
        out = self.activation(out)
        out = self.out_conv(out)
        out = self.activation(out)
        out = self.pooling_fn(out)
        return out