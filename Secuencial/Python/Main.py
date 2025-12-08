import numpy as np
import time
from Secuencial.Python.src.DataLoader import DataLoader
from Secuencial.Python.src.MLP import MLP
from pathlib import Path


def main():

    # 1. Configuración de Hiperparámetros
    EPOCHS = 10  # Requerido por el poyecto
    BATCH_SIZE = 500 # Tamaño del bloque para actualización
    LEARNING_RATE = 0.1  # Tasa de aprendizaje
    LAYER_STRUCTURE = [784, 500, 10]  # 784 (Input), 128 (Hidden), 10 (Output)

    # Ruta al dataset (ajusta según tu estructura de carpetas)
    DATASET_PATH = str(Path(__file__).resolve().parents[2] / 'Dataset' / 'archive')

    print(f"Cargando datos desde: {DATASET_PATH}")

    dataset_dir = Path(DATASET_PATH)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No se encontró la carpeta del dataset en: {dataset_dir}")

    print("=== INICIO ESCENARIO 1: RED NEURONAL SECUENCIAL (PYTHON) ===")

    # 2. Carga y Preprocesamiento de Datos
    loader = DataLoader(DATASET_PATH)

    try:
        # Carga los datos crudos (imágenes normalizadas y etiquetas numéricas)
        # load_mnist ya devuelve la separación oficial 60k Train / 10k Test
        (X_train, y_train_raw), (X_test, y_test_raw) = loader.load_mnist()

        print(f"Datos Cargados Exitosamente:")
        print(f"  -> Training Set: {X_train.shape} imágenes")
        print(f"  -> Test Set:     {X_test.shape} imágenes")

        # Convertir etiquetas de ENTRENAMIENTO a One-Hot Encoding
        # Necesario para que la salida de la red (vector de 10) se compare con la etiqueta
        y_train_encoded = loader.one_hot_encode(y_train_raw, 10)

        # NOTA: No convertimos y_test_raw a one-hot.
        # Para medir precisión (accuracy), comparamos el índice predicho (0-9)
        # contra la etiqueta real (0-9).

    except FileNotFoundError as e:
        print(f"\n[ERROR CRÍTICO] No se encontraron los archivos del dataset.")
        print(f"Detalle: {e}")
        print(f"Verifica que la carpeta '{DATASET_PATH}' exista y contenga los 4 archivos .ubyte")
        return  # Detener ejecución
    except Exception as e:
        print(f"\n[ERROR DESCONOCIDO]: {e}")
        return

    # 3. Instanciación del Modelo
    print("\nInicializando MLP con arquitectura:", LAYER_STRUCTURE)
    mlp = MLP(LAYER_STRUCTURE, LEARNING_RATE)

    # 4. Entrenamiento (Benchmark de Tiempo)
    print(f"\n--- Iniciando Entrenamiento ({EPOCHS} épocas) ---")

    start_time = time.time()

    # Entrenamos solo con el set de train
    mlp.train(X_train, y_train_encoded, EPOCHS, BATCH_SIZE)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"--- Entrenamiento finalizado en {total_time:.4f} segundos ---")

    # 5. Evaluación y Métricas
    print("\nEvaluando modelo en el conjunto de Test (imágenes nunca vistas)...")

    # Realizar predicciones
    predictions = mlp.predict(X_test)  # Devuelve array de índices (ej: [7, 2, 1, ...])

    # Calcular precisión: (Predicciones Correctas / Total Ejemplos)
    accuracy = np.mean(predictions == y_test_raw)

    print(f"Precisión Final (Accuracy): {accuracy * 100:.2f}%")

    # 6. Reporte para el Informe
    print("\nRESUMEN PARA INFORME TÉCNICO")
    print(f"Escenario: 1a (Python Secuencial)")
    print(f"Tiempo Total: {total_time:.4f} s")
    print(f"Accuracy:     {accuracy * 100:.2f}%")
    print("=============================================================")


if __name__ == "__main__":
    main()