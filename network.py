import os.path

import numpy
import numpy as np


class Network:
    def __init__(self, hidden_size: int = 100, length: int = 50, alpha: float = 10) -> None:
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.length = length
        self.synapse0, self.synapse1, self.synapse2, self.synapse3 = self.__load_weight()

    def __load_weight(self) -> tuple:
        synapse0 = self.__load_synapse('weight0.txt', self.length, self.hidden_size)
        synapse1 = self.__load_synapse('weight1.txt', self.hidden_size, 10)
        synapse2 = self.__load_synapse('weight2.txt', 10, self.hidden_size // 2)
        synapse3 = self.__load_synapse('weight3.txt', self.hidden_size // 2, 1, True)
        return synapse0, synapse1, synapse2, synapse3

    def __load_synapse(self, file_name: str, length, hidden_size, is_t: bool = False):
        file_name = self.__create_file_path(file_name)
        if os.path.exists(file_name):
            synapse = np.loadtxt(file_name, dtype=numpy.float64)
            if is_t:
                return np.array([synapse]).T
            return synapse
        synapse = 2 * np.random.random((length, hidden_size)) - 1
        np.savetxt(file_name, synapse, fmt='%f')
        return synapse

    def __save_synapse(self, synapse, file_name: str):
        file_name = self.__create_file_path(file_name)
        np.savetxt(file_name, synapse, fmt='%f')

    def __create_file_path(self, file_name: str) -> str:
        if not os.path.isdir('weight'):
            os.mkdir('weight')
        return 'weight/' + file_name

    def __sigmoid(self, x):
        output = 1 / (1 + np.exp(-x))
        return output

    def __sigmoid_output_to_derivative(self, output):
        return output * (1 - output)

    def train(self, x, y, xrange: int = 1):
        for _ in range(xrange):
            layer_0 = x
            layer_1 = self.__sigmoid(np.dot(layer_0, self.synapse0))
            layer_2 = self.__sigmoid(np.dot(layer_1, self.synapse1))
            layer_3 = self.__sigmoid(np.dot(layer_2, self.synapse2))
            layer_4 = self.__sigmoid(np.dot(layer_3, self.synapse3))

            layer_4_error = layer_4 - y
            layer_4_delta = layer_4_error * self.__sigmoid_output_to_derivative(layer_4)

            layer_3_error = layer_4_delta.dot(self.synapse3.T)
            layer_3_delta = layer_3_error * self.__sigmoid_output_to_derivative(layer_3)

            layer_2_error = layer_3_delta.dot(self.synapse2.T)
            layer_2_delta = layer_2_error * self.__sigmoid_output_to_derivative(layer_2)

            layer_1_error = layer_2_delta.dot(self.synapse1.T)
            layer_1_delta = layer_1_error * self.__sigmoid_output_to_derivative(layer_1)

            self.synapse3 -= self.alpha * (layer_3.T.dot(layer_4_delta))
            self.synapse2 -= self.alpha * (layer_2.T.dot(layer_3_delta))
            self.synapse1 -= self.alpha * (layer_1.T.dot(layer_2_delta))
            self.synapse0 -= self.alpha * (layer_0.T.dot(layer_1_delta))

        self.__save_synapse(self.synapse3, 'weight3.txt')
        self.__save_synapse(self.synapse2, 'weight2.txt')
        self.__save_synapse(self.synapse1, 'weight1.txt')
        self.__save_synapse(self.synapse0, 'weight0.txt')

    def check(self, x):
        layer_0 = x
        layer_1 = self.__sigmoid(np.dot(layer_0, self.synapse0))
        layer_2 = self.__sigmoid(np.dot(layer_1, self.synapse1))
        layer_3 = self.__sigmoid(np.dot(layer_2, self.synapse2))
        layer_4 = self.__sigmoid(np.dot(layer_3, self.synapse3))
        return layer_4
