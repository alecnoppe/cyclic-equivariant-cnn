Author: Alec Noppe @alecnoppe

# cyclic-equivariant-cnn
PyTorch implementation of the C4-Equivariant CNN, as proposed in [1]. Also includes an implementation of a D4-Equivariant CNN; an extension of [1]. 

If you take inspiration from this implementation, and/or copy code-snippets - make sure you to give credit to this repository. 


[1]: Dieleman, S., De Fauw, J., & Kavukcuoglu, K. (2016, June). Exploiting cyclic symmetry in convolutional neural networks. In International conference on machine learning (pp. 1889-1898). PMLR.
    https://arxiv.org/abs/1602.02660

## Running Instructions

First, run the `python make_datasets.py` script to download the original MNIST datasets and create the C4/D4 variants.

Afterwards, use `main.py` to train your chosen model $\text{<model>} \in {\text{cnn, c4, d4}}$  on your chosen dataset $\text{<dataset>} \in {\text{c4, d4}}$ as follows:

```
python main.py --model <model> --data <data> --model-path models/<model>-<data>.pt
```

$\textbf{All parameters}$:

    --model MODEL                       model to use
    --data DATA                         data to use
    --model-path MODEL_PATH             path to save model
    --epochs EPOCHS                     number of epochs
    --optimizer OPTIMIZER               optimizer to use
    --batch-size BATCH_SIZE             batch size
    --loss LOSS                         loss function to use
    --loss-path LOSS_PATH               path to save loss

## File structure

    data/                   - Directory for data files/directories used to train the model(s)
        MNIST/C4/               - C4 MNIST dataset

        MNIST/D4/               - D4 MNIST dataset

    figures/                - Directory for (evaluation) figures (.png/.svg/.jpg/.gif/.mp4)

    models/                 - Directory for saved model files (.pt) (checkpoints)

    notebooks/              - Directory for Jupyter notebooks (.ipynb)
        *.ipynb                 - Mainly used for debugging, shows how the operations work

    results/                - Directory for results (.csv)

    src/                    - Source code directory
        models/                 - Directory for model source code

            C4_CNN.py               - C4-Equivariant CNN, as introduced in [1]

            CNN.py                  - Standard CNN model

            ConvBlock.py            - Basic Convolutional blocks, used in all models.

            D4_CNN.py               - D4-Equivariant CNN, extension of [1]

        utils/                  - Directory for utility code

            Accuracy.py             - Class to compute the accuracy of predictions.

            MNIST_Dataset.py        - Dataset class for the (C4/D4) MNIST datasets.

            Trainer.py              - 'Trainer' class with standard methods for training PyTorch models

        C4.py                   - C4 operations as introduced in [1]

        D4.py                   - Extensions of the C4 (discrete cyclic group) operations introduced in [1], 
                                    to also be equivariant to reflections (D4 dihedral group)

    main.py                 - Main access point to the system. Used to train the models.

    make_datasets.py        - Download the MNIST datasets, and add C4 rotations and reflections (D4)

    test.py                 - Test a trained model, reports test accuracy.
