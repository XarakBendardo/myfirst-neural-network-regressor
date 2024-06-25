import numpy as np
import random
from typing import Callable


def AF_ReLU(x: float) -> float:
    return max(x, 0)


def DF_ReLu(x: float) -> float:
    return 1 if x >= 0 else 0


def AF_sigmoid(x: float) -> float:
    return np.exp(x) / (1 + np.exp(x))


def DF_sigmoid(x: float) -> float:
    return np.exp(x) / ((1 + np.exp(x)) * (1 + np.exp(x)))


class UnknownActivationFunctionException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class TwoLayeredPerceptron:
    DEFAULT_NEURON_CAOUNT = 10

    weigths: list[list[list[float]]]
    biases: list[list[float]]
    activation_func: Callable
    derivative: Callable

    def __init__(self, hidden_layer_neuron_count: int = DEFAULT_NEURON_CAOUNT, activation_function: Callable = AF_ReLU) -> None:
        if activation_function != AF_ReLU and activation_function != AF_sigmoid:
            raise UnknownActivationFunctionException("Activation function must be either ReLU or sigmoid")

        self.activation_func = activation_function

        self.weigths = []
        self.biases = []

        # hidden layer
        self.weigths.append([[random.uniform(-1, 1)] for _ in range(hidden_layer_neuron_count)])
        self.biases.append([random.uniform(-1, 1) for _ in range(hidden_layer_neuron_count)])

        # output layer
        self.weigths.append([[random.uniform(-1, 1) for _ in range(hidden_layer_neuron_count)]])
        self.biases.append([random.uniform(-1, 1)])

