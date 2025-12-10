"""
MLP con Paralelización usando multiprocessing.Pool

Estrategia: PARALELIZAR FORWARD Y BACKWARD POR BATCH
- El batch se divide entre workers
- Cada worker procesa su porción del batch completa (forward+backward)
- Se agregan gradientes y se actualizan pesos

Esto usa procesos separados para el cómputo, evitando el GIL.
"""

import numpy as np
from multiprocessing import Pool, cpu_count


# =============================================================================
# FUNCIONES WORKER (nivel módulo para pickle)
# =============================================================================

def _sigmoid(x):
    """Sigmoid vectorizada"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x):
    """Softmax con estabilidad numérica"""
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def _worker_forward_backward(args):
    """
    Worker que procesa un sub-batch completo.
    
    Hace forward y backward para su porción de datos.
    Retorna los gradientes para agregar.
    """
    X_chunk, y_chunk, weights, biases, activations_types = args
    
    n_layers = len(weights)
    batch_size = X_chunk.shape[0]
    
    # === FORWARD PASS ===
    layer_inputs = [X_chunk]
    Z_values = []
    A = X_chunk
    
    for i in range(n_layers):
        Z = np.dot(A, weights[i]) + biases[i]
        Z_values.append(Z)
        
        if activations_types[i] == 'sigmoid':
            A = _sigmoid(Z)
        elif activations_types[i] == 'softmax':
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
    
    # Gradiente inicial (softmax + cross-entropy)
    dZ = y_pred - y_chunk
    
    for i in range(n_layers - 1, -1, -1):
        A_prev = layer_inputs[i]
        
        # Gradientes
        dW = np.dot(A_prev.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        
        dW_list.insert(0, dW)
        db_list.insert(0, db)
        
        # Propagar hacia atrás
        if i > 0:
            dA = np.dot(dZ, weights[i].T)
            if activations_types[i-1] == 'sigmoid':
                s = _sigmoid(Z_values[i-1])
                dZ = dA * s * (1 - s)
            else:
                dZ = dA
    
    return dW_list, db_list, loss, batch_size


# =============================================================================
# CLASE MLP
# =============================================================================

class MLP:
    """
    Perceptrón Multicapa con paralelización usando multiprocessing.Pool
    
    Divide cada batch entre workers para forward/backward en paralelo.
    """

    def __init__(self, layer_structure, learning_rate=0.01, n_workers=None):
        self.learning_rate = learning_rate
        self.n_workers = n_workers if n_workers else cpu_count()
        self.current_loss = 0.0
        
        # Inicializar pesos
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
        print(f"Usando {self.n_workers} workers")

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
        Entrenamiento con multiprocessing - MÍNIMO OVERHEAD.
        
        Estrategia: Procesar múltiples batches por worker en cada llamada.
        - Dividir época en N chunks (uno por worker)
        - Cada worker procesa TODOS sus batches secuencialmente
        - Sincronizar solo N_UPDATES veces por época
        """
        n_samples = X_train.shape[0]
        n_updates_per_epoch = 30  # Sincronizaciones por época (ajustable)
        
        # Cuántas muestras procesa cada worker antes de sincronizar
        samples_per_sync = n_samples // n_updates_per_epoch
        samples_per_worker = samples_per_sync // self.n_workers
        
        print(f"\n=== ENTRENAMIENTO CON MULTIPROCESSING ===")
        print(f"Workers: {self.n_workers}")
        print(f"Épocas: {epochs}, Batch size: {batch_size}")
        print(f"Samples: {n_samples}")
        print(f"Sincronizaciones por época: {n_updates_per_epoch}")
        print(f"Samples por worker por sync: {samples_per_worker}")
        
        with Pool(processes=self.n_workers) as pool:
            for epoch in range(epochs):
                # Shuffle
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                
                epoch_loss = 0.0
                n_updates = 0
                
                # Procesar en chunks grandes (N_UPDATES por época)
                for sync_idx in range(n_updates_per_epoch):
                    sync_start = sync_idx * samples_per_sync
                    sync_end = min((sync_idx + 1) * samples_per_sync, n_samples)
                    
                    X_sync = X_shuffled[sync_start:sync_end]
                    y_sync = y_shuffled[sync_start:sync_end]
                    
                    # Dividir entre workers
                    worker_args = []
                    chunk_size = X_sync.shape[0] // self.n_workers
                    
                    for w in range(self.n_workers):
                        w_start = w * chunk_size
                        w_end = X_sync.shape[0] if w == self.n_workers - 1 else (w + 1) * chunk_size
                        
                        args = (
                            X_sync[w_start:w_end],
                            y_sync[w_start:w_end],
                            self.weights,
                            self.biases,
                            self.activation_types
                        )
                        worker_args.append(args)
                    
                    # Ejecutar en paralelo (1 llamada a pool.map por sync)
                    results = pool.map(_worker_forward_backward, worker_args)
                    
                    # Agregar gradientes
                    accumulated_dW = [np.zeros_like(W) for W in self.weights]
                    accumulated_db = [np.zeros_like(b) for b in self.biases]
                    total_loss = 0.0
                    total_samples = 0
                    
                    for dW_list, db_list, loss, n_chunk in results:
                        total_loss += loss * n_chunk
                        total_samples += n_chunk
                        
                        for i in range(len(self.weights)):
                            accumulated_dW[i] += dW_list[i]
                            accumulated_db[i] += db_list[i]
                    
                    # Actualizar pesos
                    for i in range(len(self.weights)):
                        self.weights[i] -= self.learning_rate * (accumulated_dW[i] / total_samples)
                        self.biases[i] -= self.learning_rate * (accumulated_db[i] / total_samples)
                    
                    epoch_loss += total_loss / total_samples
                    n_updates += 1
                
                self.current_loss = epoch_loss / n_updates
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {self.current_loss:.4f}")

    def predict(self, X_test):
        probs = self.forward(X_test)
        return np.argmax(probs, axis=1)
