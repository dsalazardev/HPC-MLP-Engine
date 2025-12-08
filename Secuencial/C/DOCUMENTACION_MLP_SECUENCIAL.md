# 🧠 Documentación Completa: Red Neuronal MLP en C (Secuencial)

## Índice
1. [Introducción](#introducción)
2. [Arquitectura General](#arquitectura-general)
3. [Estructura de Directorios](#estructura-de-directorios)
4. [Módulo de Álgebra Lineal (linalg)](#módulo-de-álgebra-lineal-linalg)
5. [Módulo de Carga de Datos (common/mnist)](#módulo-de-carga-de-datos-commonmnist)
6. [Módulo de Red Neuronal (network)](#módulo-de-red-neuronal-network)
7. [Programa Principal (main)](#programa-principal-main)
8. [Script de Compilación](#script-de-compilación)
9. [Flujo de Ejecución](#flujo-de-ejecución)
10. [Optimizaciones Implementadas](#optimizaciones-implementadas)

---

## Introducción

Este proyecto implementa una **Red Neuronal Perceptrón Multicapa (MLP)** completamente desde cero en lenguaje C, sin dependencias externas de bibliotecas de machine learning. El objetivo es entrenar un modelo para clasificar dígitos manuscritos del dataset **MNIST**.

### Características principales:
- ✅ Implementación secuencial optimizada
- ✅ Operaciones de álgebra lineal con optimización de caché
- ✅ Carga nativa de archivos MNIST en formato IDX
- ✅ Forward y Backward propagation completos
- ✅ Funciones de activación: Sigmoid y Softmax
- ✅ Entrenamiento por mini-batches

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROGRAMA PRINCIPAL                        │
│                           (main.c)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐   │
│  │   MNIST     │     │   NETWORK   │     │     LINALG      │   │
│  │  (common/)  │     │ (network/)  │     │    (linalg/)    │   │
│  │             │     │             │     │                 │   │
│  │ • Carga de  │────▶│ • Capas     │────▶│ • Matrices      │   │
│  │   imágenes  │     │ • Forward   │     │ • Multiplicación│   │
│  │ • Carga de  │     │ • Backward  │     │ • Activaciones  │   │
│  │   etiquetas │     │ • Update    │     │ • Operaciones   │   │
│  └─────────────┘     └─────────────┘     └─────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Topología de la Red Neural:
```
Entrada (784) ──▶ Capa Oculta (500) ──▶ Salida (10)
   │                    │                    │
   │                    │                    │
 Pixels             Sigmoid              Softmax
28x28=784          activación          probabilidades
```

---

## Estructura de Directorios

```
C/
├── main.c              # Punto de entrada del programa
├── main.h              # Header del programa principal
├── build_run.sh        # Script de compilación y ejecución
├── mlp_secuencial      # Ejecutable generado
│
├── common/             # Módulo de utilidades comunes
│   ├── mnist.c         # Implementación carga MNIST
│   └── mnist.h         # Declaraciones carga MNIST
│
├── linalg/             # Módulo de álgebra lineal
│   ├── linalg.c        # Implementación operaciones matriciales
│   └── linalg.h        # Declaraciones y estructura Matrix
│
└── network/            # Módulo de red neuronal
    ├── network.c       # Implementación de la red
    └── network.h       # Declaraciones Layer y Network
```

---

## Módulo de Álgebra Lineal (linalg)

Este módulo es el **corazón matemático** del proyecto. Implementa todas las operaciones con matrices necesarias para el funcionamiento de la red neuronal.

### Header: `linalg/linalg.h`

```c
#ifndef LINALG_H
#define LINALG_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// --- ESTRUCTURA MATRIX (Según UML) ---
typedef struct {
    float* data;  // Array plano 1D
    int rows;
    int cols;
} Matrix;

// --- PROTOTIPOS ---
Matrix* mat_init(int rows, int cols);
void mat_free(Matrix* m);
void mat_print(Matrix* m);
void mat_randomize(Matrix* m);
void mat_fill_zeros(Matrix* m);

// Operaciones Core
void mat_mul(const Matrix* restrict A, const Matrix* restrict B, Matrix* restrict C);
void mat_mul_AtB(const Matrix* restrict A, const Matrix* restrict B, Matrix* restrict C);
void mat_add(Matrix* restrict A, const Matrix* restrict B);
void mat_subtract(Matrix* restrict A, const Matrix* restrict B);
Matrix* mat_copy(Matrix* A);                // Deep copy

// Activaciones (In-place para ahorrar memoria)
void apply_sigmoid(Matrix* m);
void sigmoid_derivative(const Matrix* restrict m, Matrix* restrict dest);
void apply_softmax(Matrix* m);

#endif
```

### Implementación: `linalg/linalg.c`

```c
#include "linalg.h"
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

// === GESTIÓN DE MEMORIA ===

Matrix* mat_init(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (m == NULL) {
        fprintf(stderr, "Error: No se pudo asignar memoria para la estructura Matriz\n");
        exit(1);
    }
    m->rows = rows;
    m->cols = cols;
    // calloc inicializa en 0.0, vital para evitar basura y bugs numéricos
    m->data = (float*)calloc(rows * cols, sizeof(float));
    if (m->data == NULL) {
        fprintf(stderr, "Error: No se pudo asignar memoria para los datos de la Matriz\n");
        free(m);
        exit(1);
    }
    return m;
}

void mat_free(Matrix* m) {
    if (m) {
        if (m->data) free(m->data);
        free(m);
    }
}

void mat_randomize(Matrix* m) {
    // Inicialización He/Xavier simplificada para convergencia rápida
    float scale = sqrtf(2.0f / m->rows);
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

// === OPERACIONES CORE (OPTIMIZADAS PARA CACHÉ) ===

// 1. MULTIPLICACIÓN ESTÁNDAR (Forward Prop)
// C = A * B
// Optimizaciones:
// - Loop Interchange (i-k-j) para acceso lineal
// - 'restrict' para permitir vectorización SIMD
void mat_mul(const Matrix* restrict A, const Matrix* restrict B, Matrix* restrict C) {
    // Validación rápida de dimensiones
    if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
        fprintf(stderr, "Error Dim: Mul %dx%d * %dx%d -> %dx%d\n",
                A->rows, A->cols, B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    // Limpiamos C (memset es ultra rápido)
    memset(C->data, 0, C->rows * C->cols * sizeof(float));

    // Bucle optimizado i-k-j
    for (int i = 0; i < A->rows; i++) {
        for (int k = 0; k < A->cols; k++) {
            float r = A->data[i * A->cols + k]; // Cacheamos valor de A

            // Bucle interno: Vectorizable automáticamente (-O3 -march=native)
            for (int j = 0; j < B->cols; j++) {
                C->data[i * C->cols + j] += r * B->data[k * B->cols + j];
            }
        }
    }
}

// 2. MULTIPLICACIÓN CON TRANSPUESTA IMPLÍCITA (Backward Prop)
// C = A^T * B
// Optimizaciones:
// - Evita malloc/free de la matriz transpuesta
// - Lee A y B secuencialmente usando acumulación (k-i-j)
void mat_mul_AtB(const Matrix* restrict A, const Matrix* restrict B, Matrix* restrict C) {
    // Validación: A^T (cols x rows) * B (rows x cols_B)
    // Por tanto, A->rows debe ser igual a B->rows (dimensión de contracción, ej. Batch Size)
    if (A->rows != B->rows || C->rows != A->cols || C->cols != B->cols) {
        fprintf(stderr, "Error Dim: Mul_AtB A(%dx%d)^T * B(%dx%d) -> C(%dx%d)\n",
                A->rows, A->cols, B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    memset(C->data, 0, C->rows * C->cols * sizeof(float));

    // Estrategia k-i-j:
    // k: Recorre la dimensión común (filas de A y B, ej. Batch Size)
    // i: Recorre columnas de A (que son filas de A^T)
    // j: Recorre columnas de B
    for (int k = 0; k < A->rows; k++) {
        for (int i = 0; i < A->cols; i++) {
            float val_a = A->data[k * A->cols + i]; // Lectura secuencial de A

            // Bucle interno vectorizable
            for (int j = 0; j < B->cols; j++) {
                // Acumulamos en C
                C->data[i * C->cols + j] += val_a * B->data[k * B->cols + j];
            }
        }
    }
}

// === UTILIDADES MATEMÁTICAS ===

// A = A + B
void mat_add(Matrix* restrict A, const Matrix* restrict B) {
    int size = A->rows * A->cols;
    // El compilador vectorizará esto gracias a 'restrict' y al acceso lineal
    for (int i = 0; i < size; i++) {
        A->data[i] += B->data[i];
    }
}

// A = A - B
void mat_subtract(Matrix* restrict A, const Matrix* restrict B) {
    int size = A->rows * A->cols;
    for (int i = 0; i < size; i++) {
        A->data[i] -= B->data[i];
    }
}

// === FUNCIONES DE ACTIVACIÓN ===

void apply_sigmoid(Matrix* m) {
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        m->data[i] = 1.0f / (1.0f + expf(-m->data[i]));
    }
}

// Calcula f'(x) basada en la salida 'm' y guarda el resultado en 'dest'
void sigmoid_derivative(const Matrix* restrict m, Matrix* restrict dest) {
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        float val = m->data[i];
        dest->data[i] = val * (1.0f - val);
    }
}

void apply_softmax(Matrix* m) {
    // Softmax por fila (Batch processing)
    for (int i = 0; i < m->rows; i++) {
        float max_val = -1e9; // -Infinito
        int row_offset = i * m->cols;

        // 1. Encontrar maximo para estabilidad numérica
        for (int j = 0; j < m->cols; j++) {
            if (m->data[row_offset + j] > max_val) max_val = m->data[row_offset + j];
        }

        // 2. Exponencial y suma
        float sum = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            float val = expf(m->data[row_offset + j] - max_val);
            m->data[row_offset + j] = val;
            sum += val;
        }

        // 3. Normalización
        for (int j = 0; j < m->cols; j++) {
            m->data[row_offset + j] /= sum;
        }
    }
}

// Función auxiliar (ya casi no la usamos gracias a AtB, pero útil para debug)
Matrix* mat_transpose(Matrix* A) {
    Matrix* T = mat_init(A->cols, A->rows);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            T->data[j * T->cols + i] = A->data[i * A->cols + j];
        }
    }
    return T;
}
```

### Explicación de Conceptos Clave:

#### Estructura Matrix
```c
typedef struct {
    float* data;  // Array plano 1D (row-major order)
    int rows;     // Número de filas
    int cols;     // Número de columnas
} Matrix;
```

La matriz se almacena como un **array plano unidimensional** en formato **row-major** (fila por fila). El acceso a un elemento `[i][j]` se calcula como: `data[i * cols + j]`.

#### Optimización i-k-j
El orden de los bucles en la multiplicación de matrices afecta drásticamente el rendimiento debido al **cache locality**. El orden `i-k-j` asegura acceso secuencial a la memoria.

#### Keyword `restrict`
Indica al compilador que los punteros no se solapan, permitiendo **optimizaciones SIMD** automáticas.

---

## Módulo de Carga de Datos (common/mnist)

Este módulo se encarga de leer los archivos binarios del dataset MNIST y convertirlos a estructuras Matrix.

### Header: `common/mnist.h`

```c
#ifndef MNIST_H
#define MNIST_H

#include "../linalg/linalg.h" // Necesitamos saber qué es una Matrix

// Carga imágenes desde archivo binario IDX3 (.ubyte)
// Retorna una Matriz de (Num_Imgs x 784) normalizada (0.0 - 1.0)
Matrix* mnist_load_images(const char* filename);

// Carga etiquetas desde archivo binario IDX1 (.ubyte)
// Retorna una Matriz One-Hot (Num_Imgs x 10)
Matrix* mnist_load_labels(const char* filename);

#endif
```

### Implementación: `common/mnist.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "mnist.h"

// Macro de seguridad HPC: Verifica que fread lea exactamente los elementos pedidos.
// Si falla, el programa se detiene (Assertion Failure).
#define CHECK_READ(call, expected) assert((call) == (expected))

// Función auxiliar para voltear Endianness (Big Endian -> Little Endian)
// Los archivos MNIST vienen en formato Big Endian (no nativo para Intel/AMD).
uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0x000000ff) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val << 24) & 0xff000000);
}

Matrix* mnist_load_images(const char* filename) {
    FILE* f = fopen(filename, "rb"); // "rb" es vital para lectura binaria
    if (!f) {
        printf("ERROR CRITICO: No se pudo abrir el archivo de imagenes: %s\n", filename);
        exit(1);
    }

    uint32_t magic, num_imgs, rows, cols;

    // Leemos encabezados validando la lectura con CHECK_READ
    CHECK_READ(fread(&magic, sizeof(uint32_t), 1, f), 1);
    CHECK_READ(fread(&num_imgs, sizeof(uint32_t), 1, f), 1);
    CHECK_READ(fread(&rows, sizeof(uint32_t), 1, f), 1);
    CHECK_READ(fread(&cols, sizeof(uint32_t), 1, f), 1);

    // Corregimos Endianness
    magic = swap_endian(magic);
    num_imgs = swap_endian(num_imgs);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic != 2051) {
        printf("ERROR: Magic number incorrecto en imagenes. Leido: %d (Esperado: 2051)\n", magic);
        fclose(f);
        exit(1);
    }

    printf("Cargando %d imagenes de %dx%d...\n", num_imgs, rows, cols);

    int inputs_per_img = rows * cols; // 784 pixels
    Matrix* m = mat_init(num_imgs, inputs_per_img);

    // Buffer temporal para leer bytes (unsigned char 0-255)
    unsigned char* buffer = (unsigned char*)malloc(inputs_per_img * sizeof(unsigned char));
    if (!buffer) {
        printf("ERROR: Fallo asignacion de memoria para buffer.\n");
        exit(1);
    }

    for (int i = 0; i < num_imgs; i++) {
        // Leemos 784 bytes de una sola imagen de golpe
        CHECK_READ(fread(buffer, sizeof(unsigned char), inputs_per_img, f), (size_t)inputs_per_img);

        // Normalizamos a float 0.0-1.0 y guardamos en la matriz
        for (int j = 0; j < inputs_per_img; j++) {
            m->data[i * m->cols + j] = (float)buffer[j] / 255.0f;
        }
    }

    free(buffer);
    fclose(f);
    return m;
}

Matrix* mnist_load_labels(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("ERROR CRITICO: No se pudo abrir el archivo de etiquetas: %s\n", filename);
        exit(1);
    }

    uint32_t magic, num_labels;

    // Leemos encabezados validando lectura
    CHECK_READ(fread(&magic, sizeof(uint32_t), 1, f), 1);
    CHECK_READ(fread(&num_labels, sizeof(uint32_t), 1, f), 1);

    magic = swap_endian(magic);
    num_labels = swap_endian(num_labels);

    if (magic != 2049) {
        printf("ERROR: Magic number incorrecto en etiquetas. Leido: %d (Esperado: 2049)\n", magic);
        fclose(f);
        exit(1);
    }

    printf("Cargando %d etiquetas...\n", num_labels);

    // Creamos matriz para One-Hot Encoding (N x 10)
    // mat_init usa calloc, así que todo inicia en 0.0
    Matrix* m = mat_init(num_labels, 10);

    unsigned char label;
    for (int i = 0; i < num_labels; i++) {
        // Leemos 1 byte (la etiqueta 0-9)
        CHECK_READ(fread(&label, sizeof(unsigned char), 1, f), 1);
        
        // One-Hot: Ponemos un 1.0 en la columna correspondiente
        if (label < 10) {
            m->data[i * 10 + label] = 1.0f;
        } else {
            printf("Advertencia: Etiqueta %d fuera de rango en indice %d\n", label, i);
        }
    }

    fclose(f);
    return m;
}
```

### Formato de Archivos MNIST

#### Archivo de Imágenes (IDX3-UBYTE)
```
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
...
```

#### Archivo de Etiquetas (IDX1-UBYTE)
```
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
...
```

### One-Hot Encoding

Las etiquetas se convierten a vectores one-hot de 10 elementos:
```
Etiqueta 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Etiqueta 7 → [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

---

## Módulo de Red Neuronal (network)

Este módulo implementa la arquitectura de la red neuronal y los algoritmos de entrenamiento.

### Header: `network/network.h`

```c
#ifndef NETWORK_H
#define NETWORK_H

#include "../linalg/linalg.h"

// --- ESTRUCTURA LAYER (Según UML) ---
typedef struct {
    Matrix* W;       // Pesos
    Matrix* b;       // Sesgos

    // --- MEMORIA PRE-ASIGNADA (BUFFERS) ---
    Matrix* Z;       // Resultado de W*Input + b (Pre-activación)
    Matrix* A;       // Resultado de Activation(Z) (Salida de capa)
    Matrix* delta;   // Error de la capa (dZ) para Backprop
    // --------------------------------------

    Matrix* dW;      // Gradientes acumulados
    Matrix* db;
} Layer;

// --- ESTRUCTURA NETWORK ---
typedef struct {
    int num_layers;
    Layer** layers;  // Array de punteros a Layer
    float learning_rate;
} Network;

// --- NET CORE MODULE ---
Network* net_create(int* topology, int n_layers, float lr, int batch_size);
void net_free(Network* net);
Matrix* net_forward(Network* net, Matrix* input);
void net_backward(Network* net, Matrix* input, Matrix* target); // Backprop completo
void net_update(Network* net, int batch_size); // SGD Update

#endif
```

### Implementación: `network/network.c`

```c
#include "network.h"
#include <string.h> // Para memset y memcpy

Network* net_create(int* topology, int n_layers, float lr, int batch_size) {
    Network* net = (Network*)malloc(sizeof(Network));
    net->num_layers = n_layers - 1;
    net->learning_rate = lr;
    net->layers = (Layer**)malloc(net->num_layers * sizeof(Layer*));

    for (int i = 0; i < net->num_layers; i++) {
        net->layers[i] = (Layer*)malloc(sizeof(Layer));
        int n_in = topology[i];
        int n_out = topology[i+1];

        // 1. Pesos y Gradientes
        net->layers[i]->W = mat_init(n_in, n_out);
        mat_randomize(net->layers[i]->W);
        net->layers[i]->b = mat_init(1, n_out); // Init en 0 por calloc

        net->layers[i]->dW = mat_init(n_in, n_out);
        net->layers[i]->db = mat_init(1, n_out);

        // 2. BUFFERS PRE-ASIGNADOS (Zero-Malloc)
        net->layers[i]->Z = mat_init(batch_size, n_out);
        net->layers[i]->A = mat_init(batch_size, n_out);
        net->layers[i]->delta = mat_init(batch_size, n_out);
    }
    return net;
}

// Forward Pass sin mallocs (Usa buffers pre-asignados)
Matrix* net_forward(Network* net, Matrix* input) {
    Matrix* current_input = input;

    for (int i = 0; i < net->num_layers; i++) {
        Layer* l = net->layers[i];

        // 1. Z = X * W (Escribe directo en l->Z usando optimización linalg)
        mat_mul(current_input, l->W, l->Z);

        // 2. Sumar Bias (Optimización: Punteros directos)
        int rows = l->Z->rows;
        int cols = l->Z->cols;
        float* z_ptr = l->Z->data;
        float* b_ptr = l->b->data;

        for(int r=0; r < rows; r++) {
            for(int c=0; c < cols; c++) {
                z_ptr[r*cols + c] += b_ptr[c];
            }
        }

        // 3. Activación
        // Copiamos Z a A para aplicar activación in-place
        memcpy(l->A->data, l->Z->data, rows * cols * sizeof(float));

        if (i == net->num_layers - 1) {
            apply_softmax(l->A);
        } else {
            apply_sigmoid(l->A);
        }

        current_input = l->A; // El input de la siguiente es el A de la actual
    }
    return current_input; // Retorna puntero al último buffer A
}

// Backward Pass Optimizado (Cache Friendly)
void net_backward(Network* net, Matrix* input, Matrix* target) {

    for (int i = net->num_layers - 1; i >= 0; i--) {
        Layer* l = net->layers[i];
        Matrix* prev_A = (i == 0) ? input : net->layers[i-1]->A;

        // --- A. CALCULO DEL ERROR (delta) ---
        int size = l->delta->rows * l->delta->cols;

        if (i == net->num_layers - 1) {
            // Capa Salida: delta = A - Target
            for(int k=0; k < size; k++) {
                l->delta->data[k] = l->A->data[k] - target->data[k];
            }
        } else {
            // Capa Oculta: delta = (delta_next * W_next^T) .* f'(A)
            Layer* next_l = net->layers[i+1];

            // Paso 1: Propagar error hacia atrás (delta_next * W^T)
            // Aquí usamos un bucle manual cuidadoso para no transponer W físicamente
            Matrix* err = l->delta;
            Matrix* W = next_l->W;
            Matrix* D = next_l->delta;

            // Limpiamos buffer de error
            memset(err->data, 0, size * sizeof(float));

            // Multiplicación D * W^T
            // Recorremos D linealmente (r, k) y acumulamos en err
            for (int r = 0; r < D->rows; r++) { // Batch
                for (int k = 0; k < D->cols; k++) { // Neuronas Next
                    float d_val = D->data[r * D->cols + k];
                    for (int c = 0; c < W->rows; c++) { // Neuronas Curr
                        // err[r][c] += D[r][k] * W[c][k]
                        err->data[r * err->cols + c] += d_val * W->data[c * W->cols + k];
                    }
                }
            }

            // Paso 2: Multiplicar por derivada (Hadamard product)
            for(int k=0; k < size; k++) {
                float a_val = l->A->data[k];
                l->delta->data[k] *= (a_val * (1.0f - a_val));
            }
        }

        // --- B. CALCULO DE GRADIENTES (dW) ---
        // dW = prev_A^T * delta
        // AQUÍ ESTÁ LA MAGIA: Usamos la función optimizada de linalg
        // Esto reemplaza tus bucles lentos por acceso lineal k-i-j
        mat_mul_AtB(prev_A, l->delta, l->dW);

        // --- C. CALCULO DE GRADIENTES (db) ---
        // db = Sum(delta, axis=0)
        memset(l->db->data, 0, l->db->cols * sizeof(float));
        for(int r=0; r < l->delta->rows; r++) {
            for(int c=0; c < l->delta->cols; c++) {
                l->db->data[c] += l->delta->data[r * l->delta->cols + c];
            }
        }
    }
}

void net_update(Network* net, int batch_size) {
    for (int i = 0; i < net->num_layers; i++) {
        Layer* l = net->layers[i];
        float scalar = net->learning_rate / batch_size;

        int w_size = l->W->rows * l->W->cols;
        for (int j = 0; j < w_size; j++) {
            l->W->data[j] -= l->dW->data[j] * scalar;
        }

        int b_size = l->b->cols;
        for (int j = 0; j < b_size; j++) {
            l->b->data[j] -= l->db->data[j] * scalar;
        }
    }
}
```

### Explicación del Flujo de la Red

#### Forward Propagation
```
Para cada capa i:
    1. Z[i] = A[i-1] * W[i]      (Multiplicación matricial)
    2. Z[i] = Z[i] + b[i]       (Suma de bias)
    3. A[i] = activation(Z[i])  (Función de activación)
```

#### Backward Propagation
```
Para cada capa i (desde la última hasta la primera):
    
    Si es capa de salida:
        delta[i] = A[i] - target
    
    Si es capa oculta:
        delta[i] = (delta[i+1] * W[i+1]^T) ⊙ f'(A[i])
    
    Gradientes:
        dW[i] = A[i-1]^T * delta[i]
        db[i] = sum(delta[i], axis=0)
```

#### Update (SGD)
```
Para cada capa i:
    W[i] = W[i] - (learning_rate / batch_size) * dW[i]
    b[i] = b[i] - (learning_rate / batch_size) * db[i]
```

---

## Programa Principal (main)

### Header: `main.h`

```c
//
// Created by salazar on 8/12/25.
//

#ifndef PROYECTO_MAIN_H
#define PROYECTO_MAIN_H

#endif //PROYECTO_MAIN_H
```

### Implementación: `main.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network/network.h"
#include "common/mnist.h"

// Función auxiliar para copiar un pedazo de datos (Batch)
// Copia 'batch_size' filas desde 'src' (empezando en 'start_row') hacia 'dest'
void fill_batch(Matrix* dest, Matrix* src, int start_row, int batch_size) {
    // Validar que no nos salgamos del array
    if (start_row + batch_size > src->rows) {
        batch_size = src->rows - start_row; // Ajuste para el último batch si sobra
    }

    // Copiamos fila por fila
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < src->cols; j++) {
            // src index: (fila_actual + i) * ancho + col
            // dest index: i * ancho + col
            dest->data[i * dest->cols + j] = src->data[(start_row + i) * src->cols + j];
        }
    }
}

int main() {
    srand(time(NULL));

    printf("=== ESCENARIO 1b: C SECUENCIAL (MNIST REAL) ===\n");

    // 1. Cargar Datos Reales (MNIST)

    // Rutas relativas asumiendo que estás en /Secuencial/C
    const char* TRAIN_IMG = "../../Dataset/archive/train-images.idx3-ubyte";
    const char* TRAIN_LBL = "../../Dataset/archive/train-labels.idx1-ubyte";

    printf("Cargando dataset MNIST...\n");
    Matrix* X_train_full = mnist_load_images(TRAIN_IMG);
    Matrix* Y_train_full = mnist_load_labels(TRAIN_LBL);

    if (X_train_full == NULL || Y_train_full == NULL) {
        printf("Error: Fallo al cargar los datos. Revisa las rutas.\n");
        return 1;
    }

    printf("Datos cargados: %d imagenes\n", X_train_full->rows);

    // 2. Configuración de la Red

    int batch_size = 32;
    int epochs = 10;
    float learning_rate = 0.1f;
    int topology[] = {784, 500, 10}; // 784 Inputs -> 128 Hidden -> 10 Outputs

    printf("Inicializando red neural...\n");
    Network* net = net_create(topology, 3, learning_rate, batch_size);
    // Matrices temporales para el Batch (reutilizables)
    // Esto evita hacer malloc/free miles de veces dentro del bucle
    Matrix* X_batch = mat_init(batch_size, 784);
    Matrix* Y_batch = mat_init(batch_size, 10);

    // 3. Bucle de Entrenamiento

    printf("--- Iniciando Entrenamiento ---\n");
    clock_t start_total = clock();

    int total_samples = X_train_full->rows;
    int num_batches = total_samples / batch_size;

    for (int e = 0; e < epochs; e++) {
        float epoch_loss = 0.0f;
        clock_t start_epoch = clock();

        // Iterar sobre todo el dataset en bloques (batches)
        for (int b = 0; b < num_batches; b++) {
            int start_row = b * batch_size;

            // A. Llenar el batch actual con datos
            fill_batch(X_batch, X_train_full, start_row, batch_size);
            fill_batch(Y_batch, Y_train_full, start_row, batch_size);

            // B. Forward Pass
            Matrix* predictions = net_forward(net, X_batch);

            // C. Backward Pass (Calcula gradientes)
            net_backward(net, X_batch, Y_batch);

            // D. Update Weights (Aplica gradientes)
            net_update(net, batch_size);

            // E. Calcular Loss (MSE simple para monitoreo)
            // Sumamos el error cuadrado de este batch
            for(int k=0; k < batch_size * 10; k++) {
                float diff = predictions->data[k] - Y_batch->data[k];
                epoch_loss += diff * diff;
            }
        }

        // Promedio del loss por epoch
        epoch_loss /= total_samples;

        double time_epoch = (double)(clock() - start_epoch) / CLOCKS_PER_SEC;
        printf("Epoch %d/%d - Loss: %.6f - Tiempo: %.2fs\n",
               e+1, epochs, epoch_loss, time_epoch);
    }

    clock_t end_total = clock();
    double time_spent = (double)(end_total - start_total) / CLOCKS_PER_SEC;
    printf("\n=== ENTRENAMIENTO FINALIZADO ===\n");
    printf("Tiempo Total: %.4f segundos\n", time_spent);

    // 4. Limpieza de Memoria (Vital en C)

    mat_free(X_train_full);
    mat_free(Y_train_full);
    mat_free(X_batch);
    mat_free(Y_batch);
    // net_free(net); // Implementar si tienes la función, o dejar que el OS limpie al salir.
    
    return 0;
}
```

### Flujo del Programa Principal

```
┌─────────────────────────────────────────────────────────────┐
│                      INICIALIZACIÓN                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Inicializar semilla aleatoria (srand)                    │
│ 2. Cargar imágenes MNIST (60,000 x 784)                     │
│ 3. Cargar etiquetas MNIST (60,000 x 10) [One-Hot]           │
│ 4. Crear red neuronal (784 → 500 → 10)                      │
│ 5. Crear buffers de batch (32 x 784) y (32 x 10)            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    BUCLE DE ÉPOCAS                           │
│                    (10 iteraciones)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐ │
│  │              BUCLE DE BATCHES                          │ │
│  │              (1875 iteraciones)                        │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  1. fill_batch() - Copiar datos al buffer              │ │
│  │  2. net_forward() - Propagación hacia adelante         │ │
│  │  3. net_backward() - Calcular gradientes               │ │
│  │  4. net_update() - Actualizar pesos                    │ │
│  │  5. Calcular loss del batch                            │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Imprimir: Epoch, Loss, Tiempo                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      FINALIZACIÓN                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Imprimir tiempo total                                    │
│ 2. Liberar memoria (mat_free)                               │
│ 3. Retornar 0                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Script de Compilación

### `build_run.sh`

```bash
#!/bin/bash

rm -f mlp_secuencial


echo "Compilando..."
gcc main.c common/mnist.c network/network.c linalg/linalg.c -o mlp_secuencial -lm -O3 -Wall

if [ $? -eq 0 ]; then
    echo "Compilación exitosa. Ejecutando..."
    ./mlp_secuencial
else
    echo "Error en la compilación."
fi
```

### Opciones de Compilación Explicadas

| Flag | Descripción |
|------|-------------|
| `-o mlp_secuencial` | Nombre del ejecutable de salida |
| `-lm` | Enlazar con la biblioteca matemática (para `expf`, `sqrtf`) |
| `-O3` | Máximo nivel de optimización del compilador |
| `-Wall` | Mostrar todas las advertencias |

### Cómo Ejecutar

```bash
cd /ruta/a/Secuencial/C
chmod +x build_run.sh
./build_run.sh
```

---

## Flujo de Ejecución

### Diagrama de Secuencia Completo

```
┌──────┐    ┌───────┐    ┌────────┐    ┌────────┐    ┌───────┐
│ main │    │ mnist │    │network │    │ linalg │    │batches│
└──┬───┘    └───┬───┘    └───┬────┘    └───┬────┘    └───┬───┘
   │            │            │             │             │
   │ load_images│            │             │             │
   │───────────>│            │             │             │
   │            │ mat_init() │             │             │
   │            │───────────────────────────>             │
   │<───────────│            │             │             │
   │            │            │             │             │
   │ load_labels│            │             │             │
   │───────────>│            │             │             │
   │<───────────│            │             │             │
   │            │            │             │             │
   │ net_create │            │             │             │
   │──────────────────────────>             │             │
   │            │            │ mat_init()  │             │
   │            │            │─────────────>             │
   │            │            │ mat_randomize()           │
   │            │            │─────────────>             │
   │<────────────────────────│             │             │
   │            │            │             │             │
   │            │            │             │             │
   │ [LOOP: epochs]          │             │             │
   │──┐         │            │             │             │
   │  │ [LOOP: batches]      │             │             │
   │  │──┐      │            │             │             │
   │  │  │ fill_batch        │             │             │
   │  │  │──────────────────────────────────────────────>│
   │  │  │      │            │             │             │
   │  │  │ net_forward       │             │             │
   │  │  │────────────────────>            │             │
   │  │  │      │            │ mat_mul()   │             │
   │  │  │      │            │─────────────>             │
   │  │  │      │            │ apply_sigmoid()           │
   │  │  │      │            │─────────────>             │
   │  │  │<────────────────────            │             │
   │  │  │      │            │             │             │
   │  │  │ net_backward      │             │             │
   │  │  │────────────────────>            │             │
   │  │  │      │            │ mat_mul_AtB()             │
   │  │  │      │            │─────────────>             │
   │  │  │<────────────────────            │             │
   │  │  │      │            │             │             │
   │  │  │ net_update        │             │             │
   │  │  │────────────────────>            │             │
   │  │  │<────────────────────            │             │
   │  │<─┘      │            │             │             │
   │<─┘         │            │             │             │
   │            │            │             │             │
   │ mat_free() │            │             │             │
   │───────────────────────────>           │             │
   │            │            │             │             │
```

---

## Optimizaciones Implementadas

### 1. **Zero-Malloc Training**
Los buffers de las capas (`Z`, `A`, `delta`) se pre-asignan una sola vez al crear la red. Durante el entrenamiento, no se hace ningún `malloc`/`free`, evitando fragmentación de memoria y overhead.

### 2. **Loop Interchange (i-k-j)**
La multiplicación de matrices usa el orden de bucles `i-k-j` en lugar del tradicional `i-j-k`, mejorando la localidad de caché.

```c
// MALO (cache misses en B)
for (i) for (j) for (k) C[i][j] += A[i][k] * B[k][j];

// BUENO (acceso lineal)
for (i) for (k) for (j) C[i][j] += A[i][k] * B[k][j];
```

### 3. **Transpuesta Implícita**
La función `mat_mul_AtB()` calcula $A^T \cdot B$ sin crear una matriz transpuesta temporal, ahorrando memoria y tiempo.

### 4. **Keyword `restrict`**
Permite al compilador asumir que los punteros no se solapan, habilitando vectorización SIMD automática.

### 5. **Softmax Estable**
Se resta el máximo antes de calcular la exponencial para evitar overflow numérico:

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

### 6. **Inicialización He/Xavier**
Los pesos se inicializan con varianza controlada para acelerar la convergencia:

$$W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$$

---

## Parámetros de Entrenamiento

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `batch_size` | 32 | Muestras por iteración |
| `epochs` | 10 | Pasadas completas por el dataset |
| `learning_rate` | 0.1 | Tasa de aprendizaje |
| `topology` | [784, 500, 10] | Arquitectura de la red |

---

## Dependencias

- **Compilador:** GCC (o compatible)
- **Bibliotecas:** `libm` (matemáticas estándar)
- **Dataset:** MNIST en formato IDX (archivos `.ubyte`)

---

## Autor

Documentación generada para el proyecto de **Programación Concurrente**.

Fecha: 8 de diciembre de 2025
