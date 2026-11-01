# Lab 4 · Image Classification with a Multi-Layer Perceptron (MNIST)

Classifying 28×28 handwritten-digit images from MNIST (60,000 training / 10,000 test images) into 10 classes with an MLP.

**File:** [`04_mlp_mnist_classification.py`](04_mlp_mnist_classification.py). It is a Colab notebook exported as a Python script, so the Markdown cells appear as comments and docstrings.

## What I worked on

- Loading MNIST with `torchvision.datasets` and `DataLoader`, and visualising samples
- Building the network as `nn.Sequential`: `Linear(784, 128)` → `ReLU` → `Linear(128, 10)` → `LogSoftmax`
- Training with RMSprop (lr = 1e-3, batch 64, 10 epochs) and the NLL loss, on GPU if available
- Writing the accuracy metric and the train/test loops
- Tracking and plotting training/test loss and accuracy per epoch

## Stack

Python · PyTorch · torchvision · Matplotlib · NumPy · Google Colab

## How to run

Open the script in [Google Colab](https://colab.research.google.com/) or run it locally:

```bash
pip install torch torchvision numpy matplotlib
python 04_mlp_mnist_classification.py
```

## Credits

Lab template created by Santiago Pascual (UPC School, 2019), based on a Keras version by Míriam Bellver and updated by Daniel Fojo and Xavier Giró-i-Nieto as course material. The original authors are credited at the top of the file. The exercise solutions are my own work.

---
Part of my [GDSA labs](https://github.com/Pedrom-GH/GDSA_PEDRO) · B.Sc. in Telecommunications Systems Engineering, UPC – ESEIAAT · Pedro Muñoz Álvarez
