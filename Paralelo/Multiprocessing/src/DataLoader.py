import numpy as np
import struct
import os


class DataLoader:
    """
    Clase encargada de cargar y preprocesar el dataset MNIST.
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_mnist(self):
        """
        Carga el dataset MNIST desde los archivos binarios.
        Retorna: ((X_train, y_train), (X_test, y_test))
        """
        # Archivos de entrenamiento
        train_images_path = os.path.join(self.dataset_path, 'train-images.idx3-ubyte')
        train_labels_path = os.path.join(self.dataset_path, 'train-labels.idx1-ubyte')
        
        # Archivos de test
        test_images_path = os.path.join(self.dataset_path, 't10k-images.idx3-ubyte')
        test_labels_path = os.path.join(self.dataset_path, 't10k-labels.idx1-ubyte')

        # Cargar datos
        X_train = self._load_images(train_images_path)
        y_train = self._load_labels(train_labels_path)
        X_test = self._load_images(test_images_path)
        y_test = self._load_labels(test_labels_path)

        return (X_train, y_train), (X_test, y_test)

    def _load_images(self, filepath):
        """
        Carga imágenes desde archivo idx3-ubyte.
        """
        with open(filepath, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows * cols)
            # Normalizar a [0, 1]
            images = images.astype(np.float32) / 255.0
        return images

    def _load_labels(self, filepath):
        """
        Carga etiquetas desde archivo idx1-ubyte.
        """
        with open(filepath, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def one_hot_encode(self, labels, num_classes=10):
        """
        Convierte etiquetas numéricas a one-hot encoding.
        """
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot
