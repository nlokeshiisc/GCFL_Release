import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
import pickle as pkl
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = load_data()

with open("Federated/Data/cifar_10/cifar_10_train.pkl", "wb") as file:
    pkl.dump((x_train, y_train), file)
with open("Federated/Data/cifar_10/cifar_10_test.pkl", "wb") as file:
    pkl.dump((x_test, y_test), file)