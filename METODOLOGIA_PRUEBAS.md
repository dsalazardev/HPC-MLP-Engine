# Metodología de Pruebas: Medición de Tiempos de Ejecución

## 1. Descripción del Hardware

### Configuración del Sistema de Pruebas
- **Procesador (CPU)**: [Especificar modelo y características]
- **Número de Núcleos/Threads**: [Especificar cantidad]
- **Memoria RAM**: [Especificar cantidad y tipo]
- **GPU** (si aplica): [Especificar modelo, memoria VRAM, arquitectura CUDA]
- **Sistema Operativo**: [Especificar versión]
- **Compilador**: [gcc/clang versión, nvcc versión para CUDA]

> Nota: Complete los campos anteriores con la información de su sistema.

---

## 2. Métodos de Medición de Tiempos

### 2.1 C Secuencial (Secuencial/C/main.c)

**Función de Medición**: `clock_t` (de `<time.h>`)

```c
#include <time.h>

clock_t start_total = clock();
// ... código a medir ...
clock_t end_total = clock();
double time_spent = (double)(end_total - start_total) / CLOCKS_PER_SEC;
```

**Características**:
- Usa `clock()` que mide tiempo de CPU consumido
- Convierte a segundos dividiendo entre `CLOCKS_PER_SEC`
- Se mide tanto el tiempo total de entrenamiento como el tiempo por época
- **Precisión**: Depende del sistema, típicamente milisegundos

**Limitaciones**:
- `clock()` mide tiempo de CPU, no tiempo de reloj pared (wall-clock time)
- En sistemas multiprocesador, puede contar múltiples núcleos

---

### 2.2 Python Secuencial (Secuencial/Python/Main.py)

**Función de Medición**: `time.time()`

```python
import time

start_time = time.time()
# ... código a medir ...
end_time = time.time()
total_time = end_time - start_time
```

**Características**:
- Usa `time.time()` que devuelve segundos desde epoch (1 de enero de 1970)
- Mide tiempo de reloj pared (wall-clock time)
- Se mide el tiempo total de entrenamiento (10 épocas)
- **Precisión**: Depende del sistema operativo, típicamente milisegundos

**Ventajas**:
- Es una función de alto nivel que es portable
- Mide el tiempo real transcurrido

---

### 2.3 Python Paralelo con Multiprocessing (Paralelo/Multiprocessing/Main.py)

**Función de Medición**: `time.time()`

```python
import time

start_time = time.time()
# ... código MLP paralelo ...
end_time = time.time()
total_time = end_time - start_time
```

**Características**:
- Usa `time.time()` igual que la versión secuencial de Python
- Mide tiempo de reloj pared incluyendo:
  - Creación de procesos workers
  - Comunicación entre procesos
  - Computación paralela
  - Sincronización
- **Número de Workers**: 4 (configurable en `N_WORKERS`)
- Se mide el tiempo total de entrenamiento completo

**Nota sobre Paralelismo**:
- Aunque se usan múltiples procesos, `time.time()` mide el tiempo real transcurrido
- El speedup se calcula comparando este tiempo contra la versión secuencial de Python

---

### 2.4 C Paralelo con OpenMP (Paralelo/OpenMP/main.c)

**Función de Medición**: `omp_get_wtime()` (de `<omp.h>`)

```c
#include <omp.h>

double start_total = omp_get_wtime();
// ... código paralelizado con OpenMP ...
double end_total = omp_get_wtime();
double time_spent = end_total - start_total;  // en segundos
```

**Características**:
- Usa `omp_get_wtime()` que mide tiempo de reloj pared de forma portátil
- Esta función es específicamente diseñada para aplicaciones paralelas
- Se mide tanto el tiempo total como el tiempo por época
- **Número de Threads OpenMP**: Determinado por `omp_get_max_threads()` (variable de entorno `OMP_NUM_THREADS`)
- **Precisión**: Alta (típicamente microsegundos)

**Ventajas sobre clock()**:
- `omp_get_wtime()` mide tiempo real transcurrido, no tiempo de CPU
- Es la función recomendada para medir performance de código OpenMP
- Permite comparaciones justas entre versiones secuencial y paralela

**Directivas OpenMP Utilizadas**:
- `#pragma omp parallel for` en operaciones de matrices
- Sincronización automática al final de regiones paralelas
- Distribución de iteraciones entre threads disponibles

---

### 2.5 PyCUDA (Paralelo/PyCuda/train_pycuda.py)

**Funciones de Medición**: 
- `time.perf_counter()` para tiempo total
- `pycuda.driver.Event` para eventos GPU

```python
import time
import pycuda.driver as drv

# Tiempo total de entrenamiento
epoch_start = time.perf_counter()
# ... código de entrenamiento ...
epoch_end = time.perf_counter()
epoch_time = epoch_end - epoch_start

# Tiempos específicos de GPU
start = drv.Event()
end = drv.Event()

start.record()
# ... kernel CUDA o transferencia de datos ...
end.record()
end.synchronize()
t_kernel = start.time_till(end)  # en milisegundos
```

**Características**:
- `time.perf_counter()`: Mide tiempo de reloj pared de alta resolución
- `pycuda.driver.Event`: Mide tiempos específicos en GPU con sincronización
- Se mide:
  - Tiempo total de cada época
  - Tiempos de transferencia Host-to-Device (CPU ↔ GPU)
  - Tiempos de kernels CUDA (computación en GPU)
  - Tiempos de transferencia Device-to-Host (GPU ↔ CPU)
- **Precisión**: Milisegundos para kernels, microsegundos para perf_counter

**GPU Utilizada**:
- Se auto-inicializa con `pycuda.autoinit`
- Usa la GPU disponible en el sistema
- Soporta lotes (batches) configurables: 16, 32, 64, 128, 256, 512

**Operaciones Cronometradas**:
1. Forward pass (computación en GPU)
2. Cálculo de pérdida (loss)
3. Backpropagation (computación en GPU)
4. Actualización de pesos (en GPU)
5. Transferencias de datos entre host y device

---

## 3. Configuración de Experimentos

### Dataset
- **Nombre**: MNIST (Modified National Institute of Standards and Technology)
- **Tamaño Entrenamiento**: 60,000 imágenes
- **Tamaño Test**: 10,000 imágenes
- **Dimensión**: 28×28 píxeles (784 características)
- **Clases**: 10 dígitos (0-9)

### Arquitectura de Red Neuronal
- **Capa de Entrada**: 784 neuronas (28×28)
- **Capa Oculta**: 500 neuronas
- **Capa de Salida**: 10 neuronas (una por clase)

### Hiperparámetros de Entrenamiento
- **Épocas**: 10
- **Tamaño de Batch**: 256
- **Tasa de Aprendizaje**: 0.1
- **Función de Activación**: ReLU (capas ocultas), Softmax (salida)
- **Función de Pérdida**: Cross-Entropy

---

## 4. Procedimiento de Medición

### Paso 1: Preparación
1. Limpiar cachés del sistema si es posible
2. Cerrar aplicaciones no esenciales
3. Configurar variables de entorno:
   - Para OpenMP: `OMP_NUM_THREADS` (e.g., 4, 8, 16)
   - Para CUDA: Verificar disponibilidad de GPU

### Paso 2: Ejecución
1. Ejecutar cada implementación múltiples veces (mínimo 3 ejecuciones)
2. Registrar el tiempo de ejecución reportado por cada programa
3. Descartar la primera ejecución (calentamiento de cachés)
4. Calcular promedio y desviación estándar de los tiempos

### Paso 3: Análisis
1. Calcular speedup: `Speedup = T_secuencial / T_paralelo`
2. Calcular eficiencia: `Eficiencia = Speedup / Número_Procesadores`
3. Comparar resultados accuracy entre versiones

---

## 5. Consideraciones Especiales

### Python vs C
- C secuencial usa `clock()` mientras que Python secuencial usa `time.time()`
- Para comparación justa, se debe tener en cuenta que:
  - C es compilado (más rápido)
  - Python es interpretado (más lento)
  - Las bibliotecas (NumPy) en Python usan código C optimizado

### OpenMP vs C Secuencial
- C secuencial usa `clock()` (tiempo de CPU)
- OpenMP usa `omp_get_wtime()` (tiempo de reloj pared)
- Para comparación, se debe convertir o usar la misma función

### PyCUDA
- Los tiempos de GPU incluyen overhead de lanzamiento de kernels
- Las transferencias de datos CPU-GPU son un cuello de botella
- El speedup depende del tamaño del batch y características de la GPU

---

## 6. Reproducibilidad

Para reproducir estos experimentos:

```bash
# C Secuencial
cd Secuencial/C
./build_run.sh
./main

# Python Secuencial
cd Secuencial/Python
python Main.py

# Python Paralelo (Multiprocessing)
cd Paralelo/Multiprocessing
python Main.py

# C con OpenMP
cd Paralelo/OpenMP
./build_run.sh
./main

# PyCUDA
cd Paralelo/PyCuda
python train_pycuda.py
```

---

## 7. Archivos de Salida

Los tiempos se registran en:
- `speedup_multiprocessing.csv`: Resultados del speedup de Multiprocessing
- `Paralelo/OpenMP/openmp_speedup.csv`: Resultados del speedup de OpenMP
- Salida estándar de cada programa con tiempos por época

---

**Fecha de Generación**: [Completar con fecha de ejecución]
**Responsable**: [Completar con nombre]
