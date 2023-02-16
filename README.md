### A very simple federated learning simulator

The code was run successfully in the following environment: Python 3.8, PyTorch 1.7, Torchvision 0.8.1

All configurations can be found in the `config.py` file.

FashionMNIST:
```
python3 main.py -data fashion
```

CIFAR-10:
```
python3 main.py -data cifar 
```

This code is a simplified version of https://github.com/PengchaoHan/EasyFL, only including the core functionalities and simple models.
