import numpy as np

class Activations:
    """
    Contenedor estático para funciones de activación y sus derivadas.
    Matemática Vectorizada.
    """

    @staticmethod
    def sigmoid(x):
        # Evita overflow con np.clip si es necesario, pero estándar es:
        # 1 / (1 + e^-x)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        # f'(x) = f(x) * (1 - f(x))
        # Asumimos que 'x' ya es el resultado de sigmoid (A), no Z.
        # Si x es Z, la formula cambia. Aquí asumimos x = sigmoid(Z)
        s = 1 / (1 + np.exp(-x)) # Recalculamos por seguridad si entra Z
        return s * (1 - s)

    @staticmethod
    def softmax(x):
        # Estabilidad numérica: restar el máximo para evitar exp() gigantes
        # x shape: (batch_size, num_classes)
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
