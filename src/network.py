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


def AF_arctan(x: float) -> float:
    return np.arctan(x)


def DF_arctan(x: float) -> float:
    return 1 / (1 + x * x)


class UnknownActivationFunctionException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class TwoLayeredPerceptron:
    DEFAULT_NEURON_CAOUNT = 10

    layers: list[list[list[float]]]
    biases: list[list[float]]
    n_hidden_neurons: int
    activations: list[list[float]]
    last_x: float
    activation_func: Callable
    activation_derivative: Callable


    def __init__(self, hidden_layer_neuron_count: int = DEFAULT_NEURON_CAOUNT) -> None:
        self.activation_func = AF_arctan
        self.activation_derivative = DF_arctan
        self.n_hidden_neurons = hidden_layer_neuron_count

        self.layers = []
        self.biases = []

        # hidden layer
        self.layers.append([[random.uniform(-1, 1)] for _ in range(hidden_layer_neuron_count)])
        self.biases.append([random.uniform(-1, 1) for _ in range(hidden_layer_neuron_count)])

        # output layer
        self.layers.append([[random.uniform(-1, 1) for _ in range(hidden_layer_neuron_count)]])
        self.biases.append([random.uniform(-1, 1)])

    def feed_forward(self, x: float) -> float:
        self.last_x = x
        hidden_layer_output = []
        self.activations = [[], []]
        for neuron_num in range(len(self.layers[0])):
            hidden_layer_output.append(self.activation_func(self.layers[0][neuron_num][0] * x + self.biases[0][neuron_num]))
            self.activations[0].append(self.layers[0][neuron_num][0] * x + self.biases[0][neuron_num])

        network_output = 0
        for neuron_weight_num in range(len(self.layers[1][0])):
            network_output += hidden_layer_output[neuron_weight_num] * self.layers[1][0][neuron_weight_num]

        #network_output = self.activation_func(network_output + self.biases[1][0])
        network_output = network_output + self.biases[1][0]
        self.activations[1].append(network_output)
        return network_output

    def propagate_error(self, y_true, y_predicted, learning_rate: float = 0.05) -> None:
        output_layer_error = (y_predicted - y_true) # linear actiavtion function in the output layer
        # output layer weights - only one output neuron
        self.layers[1][0] = [self.layers[1][0][weight_num] - learning_rate * output_layer_error * self.activation_func(self.activations[0][weight_num]) for weight_num in range(self.n_hidden_neurons)]

        # hidden layer weights - only one weight in each neuron (one dimentional input)
        for neuron_num in range(self.n_hidden_neurons):
            neuron_error = output_layer_error * self.layers[1][0][neuron_num] * self.activation_derivative(self.activations[0][neuron_num])
            self.layers[0][neuron_num] = [self.layers[0][neuron_num][0] - learning_rate * neuron_error * self.last_x]

    def feed(self, x: list[float], y: list[float]) -> None:
        pass