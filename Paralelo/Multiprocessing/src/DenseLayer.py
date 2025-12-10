import numpy as np
from Paralelo.Multiprocessing.src.Activations import Activations


class DenseLayer:
    """
    Representa una capa completamente conectada (Fully Connected).
    Maneja: Z = XW + B
    """

    def __init__(self, n_input, n_output, activation_type='sigmoid'):
        self.n_input = n_input
        self.n_output = n_output
        self.activation_type = activation_type

        # Inicialización de Pesos (Random pequeño)
        self.weights = np.random.randn(n_input, n_output) * 0.1
        self.biases = np.zeros((1, n_output))

        # Cache para Backprop
        self.input_cache = None  # X
        self.output_z = None  # Z
        self.output_a = None  # A (activación)

        # Gradientes
        self.d_weights = None
        self.d_biases = None

    def forward_prop(self, input_data):
        """
        Calcula Z = X . W + B y aplica activación.
        """
        self.input_cache = input_data
        self.output_z = np.dot(input_data, self.weights) + self.biases

        if self.activation_type == 'sigmoid':
            self.output_a = Activations.sigmoid(self.output_z)
        elif self.activation_type == 'softmax':
            self.output_a = Activations.softmax(self.output_z)
        else:
            self.output_a = self.output_z  # Linear

        return self.output_a

    def backward_prop(self, output_gradient, learning_rate):
        """
        Calcula gradientes y actualiza pesos.
        Retorna: gradiente de entrada para la capa anterior (dE/dX).
        """
        batch_size = self.input_cache.shape[0]

        # 1. Calcular dZ (Derivada de la activación)
        if self.activation_type == 'sigmoid':
            d_activation = Activations.sigmoid_derivative(self.output_z)
            dZ = output_gradient * d_activation
        elif self.activation_type == 'softmax':
            dZ = output_gradient
        else:
            dZ = output_gradient

        # 2. Calcular Gradientes de Pesos y Bias
        self.d_weights = np.dot(self.input_cache.T, dZ)
        self.d_biases = np.sum(dZ, axis=0, keepdims=True)

        # 3. Calcular gradiente para propagar hacia atrás (dX)
        input_gradient = np.dot(dZ, self.weights.T)

        # 4. Actualizar Pesos (SGD)
        self.weights -= learning_rate * (self.d_weights / batch_size)
        self.biases -= learning_rate * (self.d_biases / batch_size)

        return input_gradient
