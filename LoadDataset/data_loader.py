import numpy as np
import struct
import os


class MNISTLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_data(self):

        train_img_path = os.path.join(self.base_path, 'train-images.idx3-ubyte')
        train_lbl_path = os.path.join(self.base_path, 'train-labels.idx1-ubyte')
        test_img_path = os.path.join(self.base_path, 't10k-images.idx3-ubyte')
        test_lbl_path = os.path.join(self.base_path, 't10k-labels.idx1-ubyte')

        X_train = self._read_images(train_img_path)
        y_train = self._read_labels(train_lbl_path)
        X_test = self._read_images(test_img_path)
        y_test = self._read_labels(test_lbl_path)

        return (X_train, y_train), (X_test, y_test)

    def _read_labels(self, path):
        with open(path, 'rb') as f:

            magic, num = struct.unpack(">II", f.read(8))

            labels = np.fromfile(f, dtype=np.uint8)

        return labels

    def _read_images(self, path):
        with open(path, 'rb') as f:

            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))

            images = np.fromfile(f, dtype=np.uint8)

            images = images.reshape(num, rows * cols)

            images = images.T.astype(np.float32) / 255.0

        return images