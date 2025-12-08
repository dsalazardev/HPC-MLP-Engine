import numpy as np
import struct
import os


class DataLoader:
    """
    Maneja la carga de datos binarios MNIST (formato IDX) y preprocesamiento.
    """

    def __init__(self, base_path):
        """
        base_path: Ruta a la carpeta que contiene los archivos .ubyte
        Ejemplo: './Dataset/archive'
        """
        self.base_path = base_path

    def load_mnist(self):
        """
        Carga los 4 archivos binarios de MNIST.
        Retorna: (X_train, y_train), (X_test, y_test)
        """
        # Nombres estándar de los archivos en el dataset original
        train_img_fn = 'train-images.idx3-ubyte'
        train_lbl_fn = 'train-labels.idx1-ubyte'
        test_img_fn = 't10k-images.idx3-ubyte'  # A veces llamado t10k-images-idx3-ubyte
        test_lbl_fn = 't10k-labels.idx1-ubyte'

        # Verificar rutas completas (ajusta nombres si tu dataset tiene guiones o puntos diferentes)
        # Nota: He visto en tu 'ls' que los archivos se llaman con punto '.' (ej: train-images.idx3-ubyte)
        # pero están dentro de carpetas. Asegúrate de apuntar a los archivos, no a las carpetas.

        # Estrategia robusta: busca el archivo exacto en la ruta dada
        path_train_img = os.path.join(self.base_path, train_img_fn)
        path_train_lbl = os.path.join(self.base_path, train_lbl_fn)
        path_test_img = os.path.join(self.base_path, test_img_fn)
        path_test_lbl = os.path.join(self.base_path, test_lbl_fn)

        print(f"Cargando datos desde: {self.base_path}")

        X_train = self._read_images(path_train_img)
        y_train = self._read_labels(path_train_lbl)
        X_test = self._read_images(path_test_img)
        y_test = self._read_labels(path_test_lbl)

        print(f"Dimensiones Train: X={X_train.shape}, y={y_train.shape}")
        return (X_train, y_train), (X_test, y_test)

    def _read_labels(self, path):
        """ Lee archivo IDX1 (etiquetas) """
        with open(path, 'rb') as f:
            # Leer header: Magic Number y Número de items (2 enteros de 32bit big-endian)
            magic, num = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number inválido en etiquetas: {magic}")

            # Leer datos
            labels = np.fromfile(f, dtype=np.uint8)

        return labels

    def _read_images(self, path):
        """ Lee archivo IDX3 (imágenes) """
        with open(path, 'rb') as f:
            # Leer header: Magic, Num Images, Rows, Cols (4 enteros)
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number inválido en imágenes: {magic}")

            # Leer datos crudos
            images = np.fromfile(f, dtype=np.uint8)

            # Reshape a (N, 784)
            # IMPORTANTE: No usamos .T aquí porque queremos (Samples, Features)
            images = images.reshape(num, rows * cols)

            # Normalización a float32 [0, 1]
            images = images.astype(np.float32) / 255.0

        return images

    def one_hot_encode(self, y, num_classes=10):
        """ Convierte vector de etiquetas (N,) a matriz (N, 10) """
        m = y.shape[0]
        one_hot = np.zeros((m, num_classes), dtype=np.float32)
        # Indexado avanzado de NumPy
        one_hot[np.arange(m), y] = 1.0
        return one_hot