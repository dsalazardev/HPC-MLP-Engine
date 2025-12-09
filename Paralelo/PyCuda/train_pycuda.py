import os
import time
import math
import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt

# ============================================================================
# 1. FUNCIÓN PARA CARGAR DATOS
# ============================================================================
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28*28).astype(np.float32) / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    labels = np.asarray(labels).flatten()
    one_hot = np.zeros((labels.size, 10), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot

# ============================================================================
# 2. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================
def entrenar_mlp_pycuda(X_train, y_train, X_test, y_test, 
                        epochs=10, batch_size=32, 
                        hidden_neurons=256, learning_rate=0.1):
    """
    Entrena un MLP con PyCUDA
    
    Args:
        epochs: Número de épocas
        batch_size: Tamaño del lote (probar: 16, 32, 64, 128, 256, 512)
        hidden_neurons: Neuronas en capa oculta
        learning_rate: Tasa de aprendizaje
        
    Returns:
        dict con resultados del entrenamiento
    """
    
    # --- Hiperparámetros y arquitectura ---
    E = 784
    H = hidden_neurons   
    S = 10
    tiempos_kernels = []
    tiempos_host_to_device = []
    tiempos_device_to_host = []
    
    # --- Inicialización de pesos ---
    print(f"\n🔧 Inicializando red con:")
    print(f"   - Neuronas capa oculta: {H}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Épocas: {epochs}")
    
    np.random.seed(42)
    W1 = np.random.randn(E, H).astype(np.float32) * 0.01
    b1 = np.zeros((H,), dtype=np.float32)
    W2 = np.random.randn(H, S).astype(np.float32) * 0.01
    b2 = np.zeros((S,), dtype=np.float32)
    
    # --- Kernels CUDA ---
    print("   - Compilando kernels CUDA...")
    kernel_multi = """
    __global__ void multi(float *X, float *W, float *Z, float *b, int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += X[row * K + i] * W[i * N + col];
            }
            Z[row * N + col] = sum + b[col];
        }
    }
    """
    
    kernel_relu = """
    __global__ void relu(float *Z, float *A, int M, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            float v = Z[row * N + col];
            A[row * N + col] = (v > 0.0f) ? v : 0.0f;
        }
    }
    """
    
    kernel_softmax = """
    __global__ void softmax(float *Z, float *A, int M, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        if (row < M) {
            float max_val = -1e20f;
            for (int i = 0; i < N; ++i) {
                float v = Z[row * N + i];
                if (v > max_val) max_val = v;
            }
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                float ex = expf(Z[row * N + i] - max_val);
                A[row * N + i] = ex;
                sum += ex;
            }
            for (int i = 0; i < N; ++i) {
                A[row * N + i] /= sum;
            }
        }
    }
    """
    
    kernel_matmul = """
    __global__ void matmul(float *A, float *B, float *C, int M, int K, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """
    
    # Compilar módulo
    mod = SourceModule(kernel_multi + kernel_relu + kernel_softmax + kernel_matmul, 
                      options=["-allow-unsupported-compiler"])
    func_multi = mod.get_function("multi")
    func_relu = mod.get_function("relu")
    func_softmax = mod.get_function("softmax")
    func_matmul = mod.get_function("matmul")
    
    # --- Reservar memoria GPU ---
    print("   - Reservando memoria GPU...")
    
    # Buffers de forward (dependen de batch_size)

    X_GPU = drv.mem_alloc(batch_size * E * 4)  
    Z1_GPU = drv.mem_alloc(batch_size * H * 4)
    A1_GPU = drv.mem_alloc(batch_size * H * 4)
    Z2_GPU = drv.mem_alloc(batch_size * S * 4)
    A2_GPU = drv.mem_alloc(batch_size * S * 4)
    
    # Pesos y biases (tamaño fijo)
    W1_GPU = drv.mem_alloc(W1.nbytes)
    W2_GPU = drv.mem_alloc(W2.nbytes)
    b1_GPU = drv.mem_alloc(b1.nbytes)
    b2_GPU = drv.mem_alloc(b2.nbytes)
    

    start = drv.Event(); end = drv.Event()
    start.record()

    drv.memcpy_htod(W1_GPU, W1)
    drv.memcpy_htod(W2_GPU, W2)
    drv.memcpy_htod(b1_GPU, b1)
    drv.memcpy_htod(b2_GPU, b2)

    end.record(); end.synchronize()
    t_htod = start.time_till(end)
    tiempos_host_to_device.append(t_htod)
    
    # Buffers para backprop
    DZ2_GPU = drv.mem_alloc(batch_size * S * 4)
    A1T_GPU = drv.mem_alloc(H * batch_size * 4)    
    DW2_GPU = drv.mem_alloc(H * S * 4)
    DW1_GPU = drv.mem_alloc(E * H * 4)
    
    # Buffers en CPU
    z1 = np.zeros((batch_size, H), dtype=np.float32)
    a1 = np.zeros_like(z1)
    z2 = np.zeros((batch_size, S), dtype=np.float32)
    a2 = np.zeros_like(z2)
    
    # Configuración bloques/hilos
    block = (16, 16, 1)
    
    def grid_for(M, N):
        gx = (N + block[0] - 1) // block[0]
        gy = (M + block[1] - 1) // block[1]
        return (gx, gy, 1)
    
    # --- Ciclo de entrenamiento ---
    num_samples = X_train.shape[0]
    steps_per_epoch = math.ceil(num_samples / batch_size)
    
    # Para almacenar resultados
    tiempos_epoch = []
    accuracies_train = []
    accuracies_test = []
    
    print(f"   - Comenzando entrenamiento ({steps_per_epoch} pasos/epoch)...")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        # Mezclar datos
        perm = np.random.permutation(num_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_start = time.perf_counter()
        
        for step in range(steps_per_epoch):
            # Obtener batch actual
            start = step * batch_size
            end = min(start + batch_size, num_samples)
            cur_bs = end - start
            Xb = X_shuffled[start:end].astype(np.float32)
            yb = y_shuffled[start:end].astype(np.float32)
            
            # Padding si es necesario
            if cur_bs != batch_size:
                pad = batch_size - cur_bs
                Xb = np.vstack([Xb, np.zeros((pad, E), dtype=np.float32)])
                yb = np.vstack([yb, np.zeros((pad, S), dtype=np.float32)])
            
            # --- FORWARD PASS ---
            start = drv.Event(); end = drv.Event()
            start.record()

            drv.memcpy_htod(X_GPU, Xb)

            end.record(); end.synchronize()
            tiempos_host_to_device.append(start.time_till(end))
            
            start = drv.Event(); end = drv.Event()
            start.record()

            # Capa 1: z1 = X · W1 + b1
            func_multi(X_GPU, W1_GPU, Z1_GPU, b1_GPU, 
                      np.int32(batch_size), np.int32(H), np.int32(E),
                      block=block, grid=grid_for(batch_size, H))
            
            end.record(); end.synchronize()
            t_kernel = start.time_till(end) 
            tiempos_kernels.append(t_kernel)

            drv.Context.synchronize()
            
            # ReLU
            start = drv.Event(); end = drv.Event()
            start.record()

            func_relu(Z1_GPU, A1_GPU, np.int32(batch_size), np.int32(H),
                     block=block, grid=grid_for(batch_size, H))
            
            end.record(); end.synchronize()
            t_kernel = start.time_till(end) 
            tiempos_kernels.append(t_kernel)

            drv.Context.synchronize()
            
            # Capa 2: z2 = a1 · W2 + b2
            start = drv.Event(); end = drv.Event()
            start.record()

            func_multi(A1_GPU, W2_GPU, Z2_GPU, b2_GPU,
                      np.int32(batch_size), np.int32(S), np.int32(H),
                      block=block, grid=grid_for(batch_size, S))
            
            end.record(); end.synchronize()
            t_kernel = start.time_till(end) 
            tiempos_kernels.append(t_kernel)

            drv.Context.synchronize()
            
            # Softmax
            start = drv.Event(); end = drv.Event()
            start.record()

            func_softmax(Z2_GPU, A2_GPU, np.int32(batch_size), np.int32(S),
                        block=(1, 64, 1), grid=(1, (batch_size + 64 - 1)//64, 1))
            
            end.record(); end.synchronize()
            t_kernel = start.time_till(end)
            tiempos_kernels.append(t_kernel)

            drv.Context.synchronize()
            
            # Traer resultados a CPU
            start = drv.Event(); end = drv.Event()
            start.record()

            drv.memcpy_dtoh(a2, A2_GPU)
            drv.memcpy_dtoh(z1, Z1_GPU)

            end.record(); end.synchronize()
            tiempos_device_to_host.append(start.time_till(end))

            drv.Context.synchronize()
            
            # --- CÁLCULO DE PÉRDIDA Y ACCURACY ---
            batch_loss = -np.sum(yb[:cur_bs] * np.log(a2[:cur_bs] + 1e-8)) / cur_bs
            preds = np.argmax(a2[:cur_bs], axis=1)
            truths = np.argmax(yb[:cur_bs], axis=1)
            batch_acc = np.mean(preds == truths)
            
            epoch_loss += batch_loss * cur_bs
            epoch_correct += (preds == truths).sum()
            
            # --- BACKPROPAGATION ---
            dz2 = a2.copy()
            dz2[:cur_bs] -= yb[:cur_bs]
            dz2[cur_bs:] = 0.0
            
            start = drv.Event(); end = drv.Event()
            start.record()

            drv.memcpy_htod(DZ2_GPU, dz2)
            drv.memcpy_htod(A1T_GPU, a1.T.astype(np.float32))

            end.record(); end.synchronize()
            t_htod = start.time_till(end) 
            tiempos_host_to_device.append(t_htod)

            start = drv.Event(); end = drv.Event()
            start.record()

            drv.memcpy_dtoh(a1, A1_GPU)

            end.record(); end.synchronize()

            t_dtoh = start.time_till(end) 
            tiempos_device_to_host.append(t_dtoh)

            
            # dW2 = A1^T · dz2

            start = drv.Event(); end = drv.Event()
            start.record()

            func_matmul(A1T_GPU, DZ2_GPU, DW2_GPU, 
                       np.int32(H), np.int32(batch_size), np.int32(S),
                       block=block, grid=((S + block[0] - 1)//block[0], 
                                          (H + block[1] - 1)//block[1]))
            
            end.record(); end.synchronize()
            t_kernel = start.time_till(end) 
            tiempos_kernels.append(t_kernel)
            
            drv.Context.synchronize()
            DW2 = np.empty((H, S), dtype=np.float32)
            start = drv.Event(); end = drv.Event()
            start.record()

            drv.memcpy_dtoh(DW2, DW2_GPU)

            end.record(); end.synchronize()
            tiempos_device_to_host.append(start.time_till(end))

            
            # dW1 (en CPU por simplicidad)
            da1 = dz2[:batch_size].dot(W2.T)
            dz1 = da1 * (z1 > 0).astype(np.float32)
            dW1 = Xb.T.dot(dz1)
            db1 = np.sum(dz1, axis=0)
            
            # --- ACTUALIZACIÓN DE PESOS ---
            W2 -= learning_rate * (DW2 / batch_size) 
            b2 -= learning_rate * (np.sum(dz2, axis=0) / batch_size)
            W1 -= learning_rate * (dW1 / batch_size)
            b1 -= learning_rate * (db1 / batch_size)
            
            # Actualizar pesos en GPU

            start = drv.Event(); end = drv.Event()
            start.record()

            drv.memcpy_htod(W1_GPU, W1)
            drv.memcpy_htod(W2_GPU, W2)
            drv.memcpy_htod(b1_GPU, b1)
            drv.memcpy_htod(b2_GPU, b2)

            end.record(); end.synchronize()
            t_htod = start.time_till(end) 
            tiempos_host_to_device.append(t_htod)
        
        # --- FIN DE EPOCH ---
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        
        # Calcular métricas del epoch
        epoch_loss = epoch_loss / num_samples
        epoch_acc = epoch_correct / num_samples
        tiempos_epoch.append(epoch_time)
        accuracies_train.append(epoch_acc)
        
        # Evaluar en test set
        z1_test = X_test.dot(W1) + b1
        a1_test = np.maximum(0, z1_test)
        z2_test = a1_test.dot(W2) + b2
        expz = np.exp(z2_test - np.max(z2_test, axis=1, keepdims=True))
        a2_test = expz / np.sum(expz, axis=1, keepdims=True)
        test_preds = np.argmax(a2_test, axis=1)
        test_truth = np.argmax(y_test, axis=1)
        test_acc = (test_preds == test_truth).mean()
        accuracies_test.append(test_acc)
        
        # Mostrar progreso
        print(f"Epoch {epoch}/{epochs}  loss={epoch_loss:.4f}  train_acc={epoch_acc:.4f}  "
              f"test_acc={test_acc:.4f}  time={epoch_time:.2f}s")
        
    
    # --- PROFILING DETALLADO ---
    print("\n" + "=" * 60)
    print("📊 PROFILING DETALLADO - Batch Size =", batch_size)
    print("=" * 60)
    
    if tiempos_kernels and tiempos_host_to_device and tiempos_device_to_host:
        # Estadísticas básicas
        avg_kernel = np.mean(tiempos_kernels)
        avg_h2d = np.mean(tiempos_host_to_device)
        avg_d2h = np.mean(tiempos_device_to_host)
        total_avg = avg_kernel + avg_h2d + avg_d2h
        
        print(f"Transferencias CPU→GPU: {avg_h2d:.1f} ms ({avg_h2d/total_avg*100:.1f}%)")
        print(f"Ejecución de kernels:    {avg_kernel:.1f} ms ({avg_kernel/total_avg*100:.1f}%)")
        print(f"Transferencias GPU→CPU: {avg_d2h:.1f} ms ({avg_d2h/total_avg*100:.1f}%)")
        print(f"TOTAL por batch:        {total_avg:.1f} ms")
        print(f"Throughput estimado:    {batch_size/(total_avg/1000):.0f} imágenes/segundo")
        
        # Crear gráfica
        datos_profiling = crear_grafica_profiling_detallado(
            tiempos_kernels, tiempos_host_to_device, tiempos_device_to_host, batch_size
        )
    else:
        print("⚠️  No se pudieron medir todos los tiempos de profiling")
        datos_profiling = None
    
    # --- LIMPIAR MEMORIA GPU ---
    print("\n🧹 Limpiando memoria GPU...")
    X_GPU.free()
    Z1_GPU.free()
    A1_GPU.free()
    Z2_GPU.free()
    A2_GPU.free()
    W1_GPU.free()
    W2_GPU.free()
    b1_GPU.free()
    b2_GPU.free()
    DZ2_GPU.free()
    A1T_GPU.free()
    DW2_GPU.free()
    DW1_GPU.free()
    
    # --- RETORNAR RESULTADOS ---
    return {
        'batch_size': batch_size,
        'epochs': epochs,
        'hidden_neurons': hidden_neurons,
        'learning_rate': learning_rate,
        'tiempo_total': sum(tiempos_epoch),
        'tiempo_por_epoch': np.mean(tiempos_epoch),
        'accuracy_final_train': accuracies_train[-1],
        'accuracy_final_test': accuracies_test[-1],
        'tiempos_epoch': tiempos_epoch,
        'accuracies_train': accuracies_train,
        'accuracies_test': accuracies_test,
        'pesos_finales': (W1, b1, W2, b2),
        'profiling': datos_profiling,
        'tiempos_kernel': tiempos_kernels,
        'tiempos_host_to_device': tiempos_host_to_device,
        'tiempos_device_to_host': tiempos_device_to_host
    }

# ============================================================================
# 3. FUNCIÓN PARA ANÁLISIS DE BATCH SIZE (OBLIGATORIA)
# ============================================================================
def analizar_batch_sizes(X_train, y_train, X_test, y_test, 
                         batch_sizes=[16, 32, 64, 128, 256, 512], 
                         epochs=10, hidden_neurons=256):
    """
    Ejecuta el análisis de batch size requerido por el documento del proyecto
    
    Args:
        batch_sizes: Lista de batch sizes a probar
        epochs: Número de épocas (usar menos para pruebas rápidas)
        
    Returns:
        DataFrame con resultados
    """
    import pandas as pd
    
    print("=" * 70)
    print("ANÁLISIS DE BATCH SIZE PARA PyCUDA")
    print("=" * 70)
    print(f"Probando batch sizes: {batch_sizes}")
    print(f"Épocas por prueba: {epochs}")
    print(f"Neuronas capa oculta: {hidden_neurons}")
    print("=" * 70)
    
    resultados = []
    
    for i, bs in enumerate(batch_sizes):
        print(f"\n[{i+1}/{len(batch_sizes)}] 🔄 Ejecutando con batch_size = {bs}")
        print("-" * 50)
        
        try:
            # Ejecutar entrenamiento
            resultado = entrenar_mlp_pycuda(
                X_train, y_train, X_test, y_test,
                epochs=epochs,
                batch_size=bs,
                hidden_neurons=hidden_neurons,
                learning_rate=0.1
            )
            
            resultados.append(resultado)
            
            # Mostrar resumen
            print(f"   ✅ Tiempo por epoch: {resultado['tiempo_por_epoch']:.2f}s")
            print(f"   ✅ Accuracy test final: {resultado['accuracy_final_test']:.4f}")
            print(f"   ✅ Tiempo total: {resultado['tiempo_total']:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error con batch_size={bs}: {e}")
            resultados.append({
                'batch_size': bs,
                'tiempo_por_epoch': None,
                'accuracy_final_test': None,
                'tiempo_total': None,
                'error': str(e)
            })
    
    # Crear tabla de resultados
    print("\n" + "=" * 80)
    print("TABLA RESUMEN: IMPACTO DEL BATCH SIZE")
    print("=" * 80)
    
    tabla = []
    for res in resultados:
        if 'error' not in res:
            tabla.append([
                res['batch_size'],
                f"{res['tiempo_por_epoch']:.2f}s",
                f"{res['accuracy_final_test']:.4f}",
                f"{res['tiempo_total']:.2f}s"
            ])
    
    df = pd.DataFrame(tabla, columns=['Batch Size', 'Tiempo/Epoch', 'Accuracy Test', 'Tiempo Total'])
    print(df.to_string(index=False))
    
    return df, resultados

# ============================================================================
# 4. FUNCIÓN PARA COMPARACIÓN PRINCIPAL (10 EPOCHS)
# ============================================================================
def ejecutar_comparacion_principal():
    """
    Ejecuta la comparación principal con 10 epochs (como sugiere el documento)
    """
    # Cargar datos
    dataset_dir = r"Dataset\archive"
    
    print("📂 Cargando dataset MNIST...")
    X_train = load_mnist_images(os.path.join(dataset_dir, 'train-images.idx3-ubyte'))
    y_train = load_mnist_labels(os.path.join(dataset_dir, 'train-labels.idx1-ubyte'))
    X_test = load_mnist_images(os.path.join(dataset_dir, 't10k-images.idx3-ubyte'))
    y_test = load_mnist_labels(os.path.join(dataset_dir, 't10k-labels.idx1-ubyte'))
    
    print("✅ Datos cargados correctamente")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 1. Ejecutar entrenamiento principal (10 epochs, batch=32)
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO PRINCIPAL PyCUDA")
    print("=" * 70)
    print("Configuración: 10 epochs, batch_size=32, H=256, α=0.1")
    
    resultado_principal = entrenar_mlp_pycuda(
        X_train, y_train, X_test, y_test,
        epochs=10,  # Como sugiere el documento
        batch_size=32,  # Valor intermedio
        hidden_neurons=256,
        learning_rate=0.1
    )
    
    # 2. Análisis de batch
    print("\n" + "=" * 70)
    print("ANÁLISIS DE BATCH SIZE")
    print("=" * 70)
    
    # Para análisis rápido, usar menos epochs
    df_resultados, todos_resultados = analizar_batch_sizes(
        X_train, y_train, X_test, y_test,
        batch_sizes=[16, 32, 64, 128, 256, 512],
        epochs=10,  # Menos epochs para análisis rápido
        hidden_neurons=256
    )
    
    # 3. Guardar resultados
    guardar_resultados(resultado_principal, df_resultados)
    
    return resultado_principal, df_resultados

# ============================================================================
# 5. FUNCIÓN AUXILIAR: GUARDAR RESULTADOS
# ============================================================================
def guardar_resultados(resultado_principal, df_resultados):
    """Guarda resultados en archivos para el informe"""
    import json
    
    # Crear directorio de resultados si no existe
    os.makedirs("resultados", exist_ok=True)
    
    # Guardar resultados principales
    with open("resultados/resultado_principal.json", "w") as f:
        json.dump({
            'batch_size': resultado_principal['batch_size'],
            'epochs': resultado_principal['epochs'],
            'tiempo_total': resultado_principal['tiempo_total'],
            'accuracy_final_test': resultado_principal['accuracy_final_test'],
            'tiempo_por_epoch': resultado_principal['tiempo_por_epoch']
        }, f, indent=2)
    
    # Guardar tabla de batch sizes
    df_resultados.to_csv("resultados/analisis_batch_sizes.csv", index=False)
    
    # Crear gráfica de batch size vs tiempo
    if not df_resultados.empty:
        plt.figure(figsize=(10, 6))
        batch_sizes = df_resultados['Batch Size'].astype(str)
        tiempos = df_resultados['Tiempo/Epoch'].str.replace('s', '').astype(float)
        
        plt.bar(batch_sizes, tiempos)
        plt.xlabel('Batch Size')
        plt.ylabel('Tiempo por Epoch (segundos)')
        plt.title('Impacto del Batch Size en Rendimiento PyCUDA')
        plt.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for i, v in enumerate(tiempos):
            plt.text(i, v + 0.1, f'{v:.2f}s', ha='center')
        
        plt.tight_layout()
        plt.savefig("resultados/grafica_batch_size.png", dpi=150)
        print("\n📊 Gráfica guardada en: resultados/grafica_batch_size.png")
    
    print("\n💾 Resultados guardados en carpeta 'resultados/'")


# ============================================================================
# 6. COMPARACIÓN DE TIEMPOS
# ============================================================================
def crear_grafica_profiling_detallado(tiempos_kernels, tiempos_host_to_device, tiempos_device_to_host, batch_size):
    """
    Crea gráfica de profiling detallado
    """
    import matplotlib.pyplot as plt
    
    if not tiempos_kernels or not tiempos_host_to_device or not tiempos_device_to_host:
        print("⚠️  No hay datos suficientes para crear gráfica de profiling")
        return
    
    # Calcular promedios en ms
    avg_kernel = np.mean(tiempos_kernels)
    avg_h2d = np.mean(tiempos_host_to_device)
    avg_d2h = np.mean(tiempos_device_to_host)
    total_avg = avg_kernel + avg_h2d + avg_d2h
    
    # Datos para gráfica
    categorias = ['CPU→GPU', 'Kernels', 'GPU→CPU']
    tiempos_ms = [avg_h2d, avg_kernel, avg_d2h]
    porcentajes = [avg_h2d/total_avg*100, avg_kernel/total_avg*100, avg_d2h/total_avg*100]
    colores = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfica 1: Tiempos absolutos
    bars1 = ax1.bar(categorias, tiempos_ms, color=colores, alpha=0.8)
    ax1.set_ylabel('Tiempo (ms)', fontsize=12)
    ax1.set_title(f'Tiempos de Profiling - Batch Size = {batch_size}', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, tiempo in zip(bars1, tiempos_ms):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{tiempo:.1f} ms', ha='center', va='bottom')
    
    # Gráfica 2: Porcentajes (pie chart)
    ax2.pie(porcentajes, labels=categorias, autopct='%1.1f%%', 
            colors=colores, startangle=90, explode=(0.05, 0.05, 0.05))
    ax2.set_title('Distribución Porcentual', fontsize=14)
    
    # Información adicional
    plt.figtext(0.5, 0.01, 
                f'Total por batch: {total_avg:.1f} ms | '
                f'Throughput estimado: {batch_size/(total_avg/1000):.0f} imágenes/segundo',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Guardar
    os.makedirs("resultados/profiling", exist_ok=True)
    plt.savefig(f'resultados/profiling/profiling_batch_{batch_size}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Gráfica de profiling guardada: resultados/profiling/profiling_batch_{batch_size}.png")
    
    return {
        'batch_size': batch_size,
        'avg_h2d_ms': avg_h2d,
        'avg_kernel_ms': avg_kernel,
        'avg_d2h_ms': avg_d2h,
        'total_ms': total_avg,
        'porcentajes': {
            'h2d': avg_h2d/total_avg*100,
            'kernel': avg_kernel/total_avg*100,
            'd2h': avg_d2h/total_avg*100
        }
    }

# ============================================================================
# 7. EJECUCIÓN PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    print("🚀 PROYECTO: IMPLEMENTACIÓN Y PARALELIZACIÓN DE MLP CON PyCUDA")
    print("=" * 70)
    
    # Opción 1: Ejecutar comparación completa
    resultado_principal, df_resultados = ejecutar_comparacion_principal()
    
    # Opción 2: Ejecutar solo con parámetros específicos
    # dataset_dir = r"tu_ruta_aqui"
    # X_train, y_train, X_test, y_test = cargar_datos(dataset_dir)
    # resultado = entrenar_mlp_pycuda(X_train, y_train, X_test, y_test,
    #                                 epochs=10, batch_size=32)
    
    print("\n" + "=" * 70)
    print("✅ EJECUCIÓN COMPLETADA")
    print("=" * 70)