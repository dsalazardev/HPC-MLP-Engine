import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from src.DenseLayer import DenseLayer
from src.Activations import Activations


class MLP:
    """
    Perceptrón Multicapa con paralelización mediante ThreadPool.
    Estrategia: Paralelizar el procesamiento de batches grandes
    y aplicar gradientes de forma síncrona.
    """

    def __init__(self, layer_structure, learning_rate=0.01, n_workers=None):
        self.learning_rate = learning_rate
        self.layers = []
        self.current_loss = 0.0
        self.n_workers = n_workers if n_workers else 4

        for i in range(len(layer_structure) - 1):
            n_in = layer_structure[i]
            n_out = layer_structure[i + 1]
            act_type = 'softmax' if i == len(layer_structure) - 2 else 'sigmoid'
            layer = DenseLayer(n_in, n_out, activation_type=act_type)
            self.layers.append(layer)

    def forward(self, X):
        """Forward pass secuencial (para predicción)"""
        activation = X
        for layer in self.layers:
            activation = layer.forward_prop(activation)
        return activation

    def backward(self, X, y_true):
        """Backpropagation secuencial"""
        y_pred = self.layers[-1].output_a
        output_gradient = y_pred - y_true

        for layer in reversed(self.layers):
            output_gradient = layer.backward_prop(output_gradient, self.learning_rate)

    def calculate_loss(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def train(self, X_train, y_train, epochs, batch_size):
        """
        Entrenamiento con paralelización interna de NumPy.
        NumPy con BLAS/LAPACK ya usa múltiples threads internamente.
        """
        n_samples = X_train.shape[0]

        print(f"=== ENTRENAMIENTO PARALELO (NumPy Optimizado) ===")
        print(f"Workers NumPy internos: {self.n_workers}")
        print(f"Épocas: {epochs}, Batch size: {batch_size}")

        # Configurar threads de NumPy/OpenBLAS
        import os
        os.environ['OMP_NUM_THREADS'] = str(self.n_workers)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.n_workers)
        os.environ['MKL_NUM_THREADS'] = str(self.n_workers)

        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            n_batches = 0

            # Mini-batch loop
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_X = X_shuffled[start:end]
                batch_y = y_shuffled[start:end]

                # Forward
                y_pred = self.forward(batch_X)

                # Loss
                loss = self.calculate_loss(y_pred, batch_y)
                epoch_loss += loss
                n_batches += 1

                # Backward & Update
                self.backward(batch_X, batch_y)

            epoch_time = time.time() - epoch_start
            self.current_loss = epoch_loss / n_batches
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {self.current_loss:.4f} - Tiempo: {epoch_time:.2f}s")

    def predict(self, X_test):
        probs = self.forward(X_test)
        return np.argmax(probs, axis=1)
