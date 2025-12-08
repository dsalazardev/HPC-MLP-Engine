import numpy as np
from Secuencial.Python.src.DenseLayer import DenseLayer


class MLP:
    """
    Clase Principal: Perceptrón Multicapa.
    Maneja la lista de capas y el ciclo de entrenamiento.
    """

    def __init__(self, layer_structure, learning_rate=0.01):
        """
        layer_structure: lista de enteros [784, 128, 10]
        """
        self.learning_rate = learning_rate
        self.layers = []
        self.current_loss = 0.0

        # Construcción dinámica de capas
        for i in range(len(layer_structure) - 1):
            n_in = layer_structure[i]
            n_out = layer_structure[i + 1]

            # Última capa es Softmax, las demás Sigmoid (estándar MLP)
            if i == len(layer_structure) - 2:
                act_type = 'softmax'
            else:
                act_type = 'sigmoid'

            layer = DenseLayer(n_in, n_out, activation_type=act_type)
            self.layers.append(layer)

    def forward(self, X):
        """ Propaga la entrada a través de todas las capas """
        activation = X
        for layer in self.layers:
            activation = layer.forward_prop(activation)
        return activation

    def backward(self, X, y_true):
        """
        Backpropagation completo.
        Para Softmax + CrossEntropy, el gradiente inicial es (Pred - True).
        """
        # 1. Forward pass (ya se hizo para calcular caches, pero necesitamos la salida final)
        # Nota: En train() llamamos a forward primero, así que los caches están listos.
        # Solo necesitamos recuperar la última salida de la última capa.
        y_pred = self.layers[-1].output_a

        # 2. Calcular gradiente inicial (dLoss/dZ_out)
        # Para CrossEntropyLogLoss + Softmax: dZ = y_pred - y_true
        output_gradient = y_pred - y_true

        # 3. Propagar hacia atrás
        for layer in reversed(self.layers):
            output_gradient = layer.backward_prop(output_gradient, self.learning_rate)

    def update_weights(self):
        # Esta lógica ya está encapsulada dentro de layer.backward_prop
        # para cumplir con la firma del UML, pero podríamos invocarla explícitamente aquí
        pass

    def calculate_loss(self, y_pred, y_true):
        # Cross Entropy Loss
        # Se suma un epsilon pequeño para evitar log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def train(self, X_train, y_train, epochs, batch_size):
        n_samples = X_train.shape[0]

        print(f"Iniciando entrenamiento: {epochs} épocas, Batch size: {batch_size}")

        for epoch in range(epochs):
            # Shuffle del dataset
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            # Mini-batch loop
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_X = X_shuffled[start:end]
                batch_y = y_shuffled[start:end]

                # A. Forward
                y_pred = self.forward(batch_X)

                # B. Loss
                loss = self.calculate_loss(y_pred, batch_y)
                epoch_loss += loss

                # C. & D. Backward & Update
                self.backward(batch_X, batch_y)

            self.current_loss = epoch_loss / (n_samples / batch_size)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {self.current_loss:.4f}")

    def predict(self, X_test):
        probs = self.forward(X_test)
        # Retorna el índice de la clase con mayor probabilidad
        return np.argmax(probs, axis=1)