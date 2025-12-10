"""
MLP con Paralelización usando multiprocessing + SHARED MEMORY

Estrategia: Usar memoria compartida para EVITAR serialización de arrays.
- Los datos de entrenamiento se copian UNA vez a memoria compartida
- Los workers acceden directamente sin pickle/unpickle
- Solo se sincronizan los gradientes (pequeños comparado con datos)

Esto elimina el cuello de botella de IPC.
"""

import numpy as np
from multiprocessing import Pool, cpu_count, RawArray
import ctypes

# Variables globales para memoria compartida
_shared_X = None
_shared_y = None
_shared_X_shape = None
_shared_y_shape = None
_shared_indices = None


def _init_worker(shared_X, X_shape, shared_y, y_shape):
    """Inicializador de workers - recibe referencias a memoria compartida."""
    global _shared_X, _shared_y, _shared_X_shape, _shared_y_shape
    _shared_X = shared_X
    _shared_y = shared_y
    _shared_X_shape = X_shape
    _shared_y_shape = y_shape


def _get_shared_arrays():
    """Obtiene arrays numpy de la memoria compartida (vista, sin copia)."""
    X = np.frombuffer(_shared_X, dtype=np.float64).reshape(_shared_X_shape)
    y = np.frombuffer(_shared_y, dtype=np.float64).reshape(_shared_y_shape)
    return X, y


def _sigmoid(x):
    """Sigmoid vectorizada"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x):
    """Softmax con estabilidad numérica"""
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def _worker_compute_gradients(args):
    """
    Worker que computa gradientes para un rango de índices.
    
    Lee datos de MEMORIA COMPARTIDA (sin serialización).
    Solo recibe: índices, pesos (pequeños), configuración.
    """
    indices, weights, biases, activation_types = args
    
    # Obtener datos de memoria compartida (VISTA, sin copia)
    X_full, y_full = _get_shared_arrays()
    X_chunk = X_full[indices]
    y_chunk = y_full[indices]
    
    n_layers = len(weights)
    batch_size = X_chunk.shape[0]
    
    if batch_size == 0:
        return None
    
    # === FORWARD PASS ===
    layer_inputs = [X_chunk]
    Z_values = []
    A = X_chunk
    
    for i in range(n_layers):
        Z = np.dot(A, weights[i]) + biases[i]
        Z_values.append(Z)
        
        if activation_types[i] == 'sigmoid':
            A = _sigmoid(Z)
        elif activation_types[i] == 'softmax':
            A = _softmax(Z)
        
        layer_inputs.append(A)
    
    y_pred = A
    
    # === LOSS ===
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_chunk * np.log(y_pred_clipped)) / batch_size
    
    # === BACKWARD PASS ===
    dW_list = []
    db_list = []
    
    dZ = y_pred - y_chunk
    
    for i in range(n_layers - 1, -1, -1):
        A_prev = layer_inputs[i]
        
        dW = np.dot(A_prev.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        
        dW_list.insert(0, dW)
        db_list.insert(0, db)
        
        if i > 0:
            dA = np.dot(dZ, weights[i].T)
            if activation_types[i-1] == 'sigmoid':
                s = _sigmoid(Z_values[i-1])
                dZ = dA * s * (1 - s)
            else:
                dZ = dA
    
    return dW_list, db_list, loss, batch_size


class MLP:
    """
    MLP con multiprocessing y MEMORIA COMPARTIDA.
    
    Elimina el overhead de serialización usando RawArray.
    """

    def __init__(self, layer_structure, learning_rate=0.01, n_workers=None):
        self.learning_rate = learning_rate
        self.n_workers = n_workers if n_workers else cpu_count()
        self.current_loss = 0.0
        
        self.weights = []
        self.biases = []
        self.activation_types = []
        
        for i in range(len(layer_structure) - 1):
            n_in = layer_structure[i]
            n_out = layer_structure[i + 1]
            
            W = np.random.randn(n_in, n_out).astype(np.float64) * 0.1
            b = np.zeros((1, n_out), dtype=np.float64)
            
            self.weights.append(W)
            self.biases.append(b)
            
            if i == len(layer_structure) - 2:
                self.activation_types.append('softmax')
            else:
                self.activation_types.append('sigmoid')
        
        print(f"MLP inicializado con {len(self.weights)} capas")
        print(f"Usando {self.n_workers} workers con MEMORIA COMPARTIDA")

    def forward(self, X):
        """Forward pass secuencial"""
        A = X
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            if self.activation_types[i] == 'sigmoid':
                A = _sigmoid(Z)
            elif self.activation_types[i] == 'softmax':
                A = _softmax(Z)
        return A

    def calculate_loss(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def train(self, X_train, y_train, epochs, batch_size):
        """
        Entrenamiento con MEMORIA COMPARTIDA.
        
        1. Copia datos a RawArray UNA vez al inicio
        2. Workers acceden directamente (sin pickle de datos)
        3. Solo se transfieren índices y gradientes
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        print(f"\n=== ENTRENAMIENTO CON SHARED MEMORY ===")
        print(f"Workers: {self.n_workers}")
        print(f"Épocas: {epochs}, Batch size: {batch_size}")
        print(f"Samples: {n_samples}, Batches/época: {n_batches}")
        
        # Crear memoria compartida para X e y (UNA vez)
        X_flat = X_train.astype(np.float64).flatten()
        y_flat = y_train.astype(np.float64).flatten()
        
        shared_X = RawArray(ctypes.c_double, X_flat.size)
        shared_y = RawArray(ctypes.c_double, y_flat.size)
        
        # Copiar datos a memoria compartida
        np.frombuffer(shared_X, dtype=np.float64)[:] = X_flat
        np.frombuffer(shared_y, dtype=np.float64)[:] = y_flat
        
        print(f"Memoria compartida: {(X_flat.nbytes + y_flat.nbytes) / 1e6:.1f} MB")
        
        # Crear pool con inicializador (workers reciben referencia a memoria compartida)
        with Pool(
            processes=self.n_workers,
            initializer=_init_worker,
            initargs=(shared_X, X_train.shape, shared_y, y_train.shape)
        ) as pool:
            
            for epoch in range(epochs):
                # Shuffle índices
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                
                epoch_loss = 0.0
                batches_done = 0
                
                # Mini-batch loop
                for batch_start in range(0, n_samples, batch_size):
                    batch_end = min(batch_start + batch_size, n_samples)
                    batch_indices = indices[batch_start:batch_end]
                    current_batch_size = len(batch_indices)
                    
                    # Dividir ÍNDICES entre workers (no datos)
                    chunk_size = current_batch_size // self.n_workers
                    
                    worker_args = []
                    for w in range(self.n_workers):
                        w_start = w * chunk_size
                        w_end = current_batch_size if w == self.n_workers - 1 else (w + 1) * chunk_size
                        
                        # Solo enviamos ÍNDICES y pesos (pequeños)
                        worker_indices = batch_indices[w_start:w_end]
                        args = (worker_indices, self.weights, self.biases, self.activation_types)
                        worker_args.append(args)
                    
                    # Ejecutar en paralelo
                    results = pool.map(_worker_compute_gradients, worker_args)
                    
                    # Agregar gradientes
                    accumulated_dW = [np.zeros_like(W) for W in self.weights]
                    accumulated_db = [np.zeros_like(b) for b in self.biases]
                    total_loss = 0.0
                    total_samples = 0
                    
                    for result in results:
                        if result is None:
                            continue
                        dW_list, db_list, loss, n_chunk = result
                        total_loss += loss * n_chunk
                        total_samples += n_chunk
                        
                        for i in range(len(self.weights)):
                            accumulated_dW[i] += dW_list[i]
                            accumulated_db[i] += db_list[i]
                    
                    if total_samples > 0:
                        for i in range(len(self.weights)):
                            self.weights[i] -= self.learning_rate * (accumulated_dW[i] / total_samples)
                            self.biases[i] -= self.learning_rate * (accumulated_db[i] / total_samples)
                        
                        epoch_loss += total_loss / total_samples
                        batches_done += 1
                
                self.current_loss = epoch_loss / batches_done if batches_done > 0 else 0
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {self.current_loss:.4f}")

    def predict(self, X_test):
        probs = self.forward(X_test)
        return np.argmax(probs, axis=1)
