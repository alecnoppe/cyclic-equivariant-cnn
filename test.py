#!/usr/bin/env python

"""
test.py loads a pre-trained model and runs the Trainer.test() method, with a chosen loss function.

NOTE: Trainer.test() is averaged per batch, not per sample.
"""

from src.utils.MNIST_Dataset import MNIST_Dataset
from src.utils.Trainer import Trainer
from src.utils.Accuracy import Accuracy
from src.models.C4_CNN import C4_CNN
from src.models.D4_CNN import D4_CNN
from src.models.CNN import CNN

import argparse
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    # Add arguments for 'model', 'data', 'model-path', 'epochs', 
    # 'optimizer', 'batch-size' and 'loss' and 'loss-path'
    parser.add_argument('--model', type=str, default='c4', help='model to use')
    parser.add_argument('--data', type=str, default='c4', help='data to use')
    parser.add_argument('--model-path', type=str, default='models/c4_cnn.pt', help='path to save model')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--loss', type=str, default='cross', help='loss function to use')
    parser.add_argument('--loss-path', type=str, default='results/c4_cnn_loss.csv', help='path to save loss')
    args = parser.parse_args()

    DATA_DICT = {
        "c4": "data/MNIST/C4/",
        "d4": "data/MNIST/D4/"
    }

    OPT_DICT = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD
    }

    LOSS_DICT = {
        'cross': torch.nn.CrossEntropyLoss(),
        'accuracy':Accuracy(10)
    }

    MODEL_DICT = {
        'c4': C4_CNN(1, (32, ), torch.nn.SiLU(), torch.nn.MaxPool2d(2)),
        'd4': D4_CNN(1, (32, ), torch.nn.SiLU(), torch.nn.MaxPool2d(2)),
        'cnn': CNN(1, (32, ), torch.nn.SiLU(), torch.nn.MaxPool2d(2))
    }

    training_dataloader = DataLoader(MNIST_Dataset(DATA_DICT[args.data]+"training.pt"), batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(MNIST_Dataset(DATA_DICT[args.data]+"validation.pt"), batch_size=args.batch_size)
    test_dataloader = DataLoader(MNIST_Dataset(DATA_DICT[args.data]+"test.pt"), batch_size=8)
    
    model = MODEL_DICT[args.model]
    optimizer = OPT_DICT[args.optimizer](params=model.parameters(), lr=0.001, weight_decay=0.0001)
    loss = LOSS_DICT[args.loss]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(model, optimizer, loss, training_dataloader, validation_dataloader, test_dataloader, device)
    best_model = torch.load(args.model_path)
    best_model.eval()
    print("-"*20)
    print("TEST LOSS:\t", trainer.test(best_model))

if __name__ == '__main__':
    main()