from torch import nn
import torch


class Accuracy(nn.Module):
    def __init__(self, n_classes):
        """Accuracy metric for classification tasks"""
        super().__init__()
        self.n_classes = n_classes

    def forward(self, pred_y, y):
        """
        Forward pass of the accuracy metric

        Args:
            pred_y: torch.Tensor; predicted labels
            y: torch.Tensor; actual labels

        Returns:
            torch.Tensor; accuracy (between 0 and 1)
        """
        # pred_y will be the class probabilities of different labels
        # y will be the actual label
        return torch.sum(torch.argmax(pred_y, dim=1) == y) / y.shape[0]