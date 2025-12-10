"""
MLP con Multiprocessing - Paralelismo de Operaciones de Red

Este módulo implementa una red neuronal MLP usando multiprocessing.Pool
para paralelizar las OPERACIONES INTERNAS de la red neuronal.

Estrategia: Paralelismo de Operaciones Matriciales
- Cada operación (matmul, activación, gradientes) se divide entre workers
- Cada worker procesa una porción de las filas del batch
- Los resultados se concatenan para formar la salida completa

Esto paraleliza el CÓMPUTO de la red, no solo la división de datos.
"""

import numpy as np
import time
from multiprocessing import cpu_count
from Paralelo.Multiprocessing.src.DataLoader import DataLoader
from Paralelo.Multiprocessing.src.MLP import MLP
from pathlib import Path


def main():
    # =================================================================
    # 1. CONFIGURACIÓN DE HIPERPARÁMETROS
    # =================================================================
    EPOCHS = 10
    BATCH_SIZE = 250  # Batch size normal (las operaciones se paralelizan internamente)
    LEARNING_RATE = 0.1
    LAYER_STRUCTURE = [784, 500, 10]
    N_WORKERS = 4  # Workers para paralelizar operaciones matriciales
    
    # Ruta al dataset
    DATASET_PATH = str(Path(__file__).resolve().parents[2] / 'Dataset' / 'archive')
    
    print("=" * 60)
    print("  MLP CON MULTIPROCESSING - OPERACIONES PARALELAS")
    print("=" * 60)
    print(f"Workers (procesos): {N_WORKERS}")
    print(f"CPUs detectados: {cpu_count()}")
    
    # =================================================================
    # 2. CARGA DE DATOS
    # =================================================================
    dataset_dir = Path(DATASET_PATH)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {dataset_dir}")
    
    loader = DataLoader(DATASET_PATH)
    
    try:
        (X_train, y_train_raw), (X_test, y_test_raw) = loader.load_mnist()
        print(f"\nDatos Cargados:")
        print(f"  -> Training Set: {X_train.shape}")
        print(f"  -> Test Set:     {X_test.shape}")
        
        # One-hot encoding para entrenamiento
        y_train_encoded = loader.one_hot_encode(y_train_raw, 10)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] No se encontraron archivos del dataset: {e}")
        return
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        return
    
    # =================================================================
    # 3. CREAR MODELO
    # =================================================================
    print(f"\nArquitectura: {LAYER_STRUCTURE}")
    mlp = MLP(LAYER_STRUCTURE, LEARNING_RATE, n_workers=N_WORKERS)
    
    # =================================================================
    # 4. ENTRENAMIENTO (con medición de tiempo)
    # =================================================================
    print(f"\n--- Iniciando Entrenamiento ({EPOCHS} épocas) ---")
    
    start_time = time.time()
    mlp.train(X_train, y_train_encoded, EPOCHS, BATCH_SIZE)
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"\n--- Entrenamiento finalizado en {total_time:.4f} segundos ---")
    
    # =================================================================
    # 5. EVALUACIÓN
    # =================================================================
    print("\nEvaluando modelo en conjunto de Test...")
    
    predictions = mlp.predict(X_test)
    accuracy = np.mean(predictions == y_test_raw)
    
    print(f"Precisión (Accuracy): {accuracy * 100:.2f}%")
    
    # =================================================================
    # 6. RESUMEN PARA INFORME TÉCNICO
    # =================================================================
    print("\n" + "=" * 60)
    print("RESUMEN PARA INFORME TÉCNICO")
    print("=" * 60)
    print(f"Escenario:    Python Paralelo (multiprocessing.Pool)")
    print(f"Workers:      {N_WORKERS} procesos")
    print(f"Batch Size:   {BATCH_SIZE}")
    print(f"Tiempo Total: {total_time:.4f} s")
    print(f"Accuracy:     {accuracy * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
