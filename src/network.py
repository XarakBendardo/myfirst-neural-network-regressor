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

    layers: list[list[list[float]]]
    biases: list[list[float]]
    activation_func: Callable
    derivative: Callable


    def __init__(self, hidden_layer_neuron_count: int = DEFAULT_NEURON_CAOUNT, activation_function: Callable = AF_sigmoid) -> None:
        if activation_function != AF_ReLU and activation_function != AF_sigmoid:
            raise UnknownActivationFunctionException("Activation function must be either ReLU or sigmoid")

        self.activation_func = activation_function

        self.layers = []
        self.biases = []

        # hidden layer
        self.layers.append([[random.uniform(-1, 1)] for _ in range(hidden_layer_neuron_count)])
        self.biases.append([random.uniform(-1, 1) for _ in range(hidden_layer_neuron_count)])

        # output layer
        self.layers.append([[random.uniform(-1, 1) for _ in range(hidden_layer_neuron_count)]])
        self.biases.append([random.uniform(-1, 1)])

    def feed_forward(self, x: float) -> float:
        hidden_layer_output = []
        for neuron_num in range(len(self.layers[0])):
            hidden_layer_output.append(self.activation_func(self.layers[0][neuron_num][0] * x + self.biases[0][neuron_num]))

        network_output = 0
        for neuron_weight_num in range(len(self.layers[1][0])):
            network_output += hidden_layer_output[neuron_weight_num] * self.layers[1][0][neuron_weight_num]

        #network_output = self.activation_func(network_output + self.biases[1][0])
        network_output = network_output + self.biases[1][0]
        return network_output