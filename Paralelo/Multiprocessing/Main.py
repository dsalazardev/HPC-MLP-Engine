import numpy as np
import time
from src.DataLoader import DataLoader
from src.MLP import MLP
from pathlib import Path


def main():

    # 1. Configuración de Hiperparámetros
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.1
    LAYER_STRUCTURE = [784, 500, 10]
    N_WORKERS = 4  # Número de procesos paralelos

    # Ruta al dataset
    DATASET_PATH = str(Path(__file__).resolve().parents[2] / 'Dataset' / 'archive')

    print(f"Cargando datos desde: {DATASET_PATH}")

    dataset_dir = Path(DATASET_PATH)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No se encontró la carpeta del dataset en: {dataset_dir}")

    print("=== ESCENARIO: RED NEURONAL PARALELA (Python Multiprocessing) ===")

    # 2. Carga y Preprocesamiento de Datos
    loader = DataLoader(DATASET_PATH)

    try:
        (X_train, y_train_raw), (X_test, y_test_raw) = loader.load_mnist()

        print(f"Datos Cargados Exitosamente:")
        print(f"  -> Training Set: {X_train.shape} imágenes")
        print(f"  -> Test Set:     {X_test.shape} imágenes")

        y_train_encoded = loader.one_hot_encode(y_train_raw, 10)

    except FileNotFoundError as e:
        print(f"\n[ERROR CRÍTICO] No se encontraron los archivos del dataset.")
        print(f"Detalle: {e}")
        return
    except Exception as e:
        print(f"\n[ERROR DESCONOCIDO]: {e}")
        return

    # 3. Instanciación del Modelo
    print("\nInicializando MLP con arquitectura:", LAYER_STRUCTURE)
    print(f"Workers paralelos: {N_WORKERS}")
    mlp = MLP(LAYER_STRUCTURE, LEARNING_RATE, n_workers=N_WORKERS)

    # 4. Entrenamiento (Benchmark de Tiempo)
    print(f"\n--- Iniciando Entrenamiento ({EPOCHS} épocas) ---")

    start_time = time.time()
    mlp.train(X_train, y_train_encoded, EPOCHS, BATCH_SIZE)
    end_time = time.time()
    
    total_time = end_time - start_time

    print(f"\n--- Entrenamiento finalizado en {total_time:.4f} segundos ---")

    # 5. Evaluación
    print("\nEvaluando modelo en el conjunto de Test...")
    predictions = mlp.predict(X_test)
    accuracy = np.mean(predictions == y_test_raw)

    print(f"Precisión Final (Accuracy): {accuracy * 100:.2f}%")

    # 6. Reporte
    print("\n" + "="*60)
    print("RESUMEN PARA INFORME TÉCNICO")
    print("="*60)
    print(f"Escenario: Python Paralelo (Multiprocessing)")
    print(f"Workers:   {N_WORKERS}")
    print(f"Tiempo Total: {total_time:.4f} s")
    print(f"Accuracy:     {accuracy * 100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
