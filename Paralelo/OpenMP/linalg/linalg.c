#include "linalg.h"
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h> // Vital para OpenMP

// === GESTIÓN DE MEMORIA ===

Matrix* mat_init(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) exit(1);
    m->rows = rows;
    m->cols = cols;

    // ALINEACIÓN DE MEMORIA (32 bytes para AVX)
    // Usamos posix_memalign para que las direcciones de memoria sean amigables con AVX
    size_t size = rows * cols * sizeof(float);
    if (posix_memalign((void**)&m->data, 32, size) != 0) {
        fprintf(stderr, "Error de memoria alineada\n");
        exit(1);
    }

    // Limpiamos basura
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
    float scale = sqrtf(2.0f / m->rows);
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

// === OPERACIONES CORE (OPTIMIZADAS PARA OPENMP) ===

// 1. MULTIPLICACIÓN ESTÁNDAR (Forward Prop)
// C = A * B
void mat_mul(const Matrix* restrict A, const Matrix* restrict B, Matrix* restrict C) {
    // Limpieza rápida
    memset(C->data, 0, C->rows * C->cols * sizeof(float));

    // ESTRATEGIA: Paralelizar filas de A (Batch)
    // schedule(static) es mejor porque todas las filas cuestan lo mismo procesar.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A->rows; i++) {
        for (int k = 0; k < A->cols; k++) {
            float r = A->data[i * A->cols + k];

            // Bucle interno vectorizado por el compilador (AVX)
            // No paralelizamos aquí para evitar overhead excesivo
            for (int j = 0; j < B->cols; j++) {
                C->data[i * C->cols + j] += r * B->data[k * B->cols + j];
            }
        }
    }
}

// 2. MULTIPLICACIÓN A^T * B (Backward Prop - Gradientes)
// ESTA ES LA CLAVE DEL RENDIMIENTO PARALELO
void mat_mul_AtB(const Matrix* restrict A, const Matrix* restrict B, Matrix* restrict C) {
    // No necesitamos memset si sobrescribimos todo, pero por seguridad:
    // (En esta lógica calculamos la suma total y asignamos, así que memset es opcional pero seguro)
    // memset(C->data, 0, C->rows * C->cols * sizeof(float));

    // ESTRATEGIA: Paralelizar la SALIDA (i, j)
    // A diferencia del secuencial, aquí iteramos sobre cada celda de la matriz de gradientes C.
    // i: 0..Inputs (ej. 784)
    // j: 0..Outputs (ej. 500)
    // Total tareas: 392,000 -> Ideal para saturar 4, 8, 16 hilos.

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < A->cols; i++) {      // Filas de A^T
        for (int j = 0; j < B->cols; j++) {  // Columnas de B

            float sum = 0.0f; // Acumulador privado para cada hilo

            // Reducción sobre el Batch (k)
            // Cada hilo procesa las 32 muestras del batch para su celda (i,j)
            for (int k = 0; k < A->rows; k++) {
                sum += A->data[k * A->cols + i] * B->data[k * B->cols + j];
            }

            C->data[i * C->cols + j] = sum;
        }
    }
}

// === UTILIDADES MATEMÁTICAS (Paralelizadas) ===

void mat_add(Matrix* restrict A, const Matrix* restrict B) {
    int size = A->rows * A->cols;

    // Paralelismo simple de grano fino
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        A->data[i] += B->data[i];
    }
}

void mat_subtract(Matrix* restrict A, const Matrix* restrict B) {
    int size = A->rows * A->cols;

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        A->data[i] -= B->data[i];
    }
}

// === FUNCIONES DE ACTIVACIÓN (Paralelizadas) ===

void apply_sigmoid(Matrix* m) {
    int size = m->rows * m->cols;

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        m->data[i] = 1.0f / (1.0f + expf(-m->data[i]));
    }
}

void sigmoid_derivative(const Matrix* restrict m, Matrix* restrict dest) {
    int size = m->rows * m->cols;

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        float val = m->data[i];
        dest->data[i] = val * (1.0f - val);
    }
}

void apply_softmax(Matrix* m) {
    // Softmax tiene dependencias por fila (max y sum), es difícil de paralelizar eficientemente
    // con grano fino si el batch es pequeño.
    // Paralelizamos solo las filas (Batch).

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m->rows; i++) {
        float max_val = -1e9;
        int row_offset = i * m->cols;

        // 1. Max
        for (int j = 0; j < m->cols; j++) {
            if (m->data[row_offset + j] > max_val) max_val = m->data[row_offset + j];
        }

        // 2. Exp + Sum
        float sum = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            float val = expf(m->data[row_offset + j] - max_val);
            m->data[row_offset + j] = val;
            sum += val;
        }

        // 3. Div
        for (int j = 0; j < m->cols; j++) {
            m->data[row_offset + j] /= sum;
        }
    }
}

// Auxiliar (ya no usada en ruta crítica, pero la dejamos)
Matrix* mat_transpose(Matrix* A) {
    Matrix* T = mat_init(A->cols, A->rows);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            T->data[j * T->cols + i] = A->data[i * A->cols + j];
        }
    }
    return T;
}