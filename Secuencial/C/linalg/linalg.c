#include "linalg.h"
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

// === GESTIÓN DE MEMORIA ===

Matrix* mat_init(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;

    // ALINEACIÓN DE MEMORIA (32 bytes para AVX)
    // Reemplazamos calloc por posix_memalign + memset
    size_t size = rows * cols * sizeof(float);

    // Aseguramos que el tamaño sea múltiplo de 32 bytes para evitar errores de alineación al final
    // (Opcional, pero buena práctica en HPC estricto)

    if (posix_memalign((void**)&m->data, 32, size) != 0) {
        fprintf(stderr, "Error de memoria alineada\n");
        exit(1);
    }

    // posix_memalign contiene basura, hay que limpiar a 0
    memset(m->data, 0, size);

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