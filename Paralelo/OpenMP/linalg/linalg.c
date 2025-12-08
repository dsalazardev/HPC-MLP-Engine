#include "linalg.h"
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// === GESTIÓN DE MEMORIA ===

Matrix* mat_init(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) exit(1);
    m->rows = rows;
    m->cols = cols;

    size_t size = rows * cols * sizeof(float);
    if (posix_memalign((void**)&m->data, 32, size) != 0) {
        fprintf(stderr, "Error de memoria alineada\n");
        exit(1);
    }
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

// === OPERACIONES CORE ===
// Estrategia: Paralelizar SOLO cuando hay suficiente trabajo
// Con batch >= 128, vale la pena usar threads

// 1. MULTIPLICACIÓN ESTÁNDAR C = A * B
void mat_mul(const Matrix* restrict A, const Matrix* restrict B, Matrix* restrict C) {
    const int rows_A = A->rows;
    const int cols_A = A->cols;
    const int cols_B = B->cols;
    const float* restrict a_data = A->data;
    const float* restrict b_data = B->data;
    float* restrict c_data = C->data;
    
    // Solo paralelizar si: batch >= 256 Y hay suficiente trabajo total
    const int work = rows_A * cols_A * cols_B;
    
    if (rows_A >= 256 && work >= 1000000) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows_A; i++) {
            float* restrict c_row = &c_data[i * cols_B];
            // Limpiar fila local
            for (int j = 0; j < cols_B; j++) c_row[j] = 0.0f;
            
            for (int k = 0; k < cols_A; k++) {
                const float a_ik = a_data[i * cols_A + k];
                const float* restrict b_row = &b_data[k * cols_B];
                #pragma omp simd
                for (int j = 0; j < cols_B; j++) {
                    c_row[j] += a_ik * b_row[j];
                }
            }
        }
    } else {
        // Versión secuencial optimizada para SIMD automático
        memset(c_data, 0, rows_A * cols_B * sizeof(float));
        for (int i = 0; i < rows_A; i++) {
            for (int k = 0; k < cols_A; k++) {
                const float a_ik = a_data[i * cols_A + k];
                const float* restrict b_row = &b_data[k * cols_B];
                float* restrict c_row = &c_data[i * cols_B];
                #pragma omp simd
                for (int j = 0; j < cols_B; j++) {
                    c_row[j] += a_ik * b_row[j];
                }
            }
        }
    }
}

// 2. MULTIPLICACIÓN A^T * B (para gradientes)
// Esta es la operación MÁS PESADA - aquí sí vale la pena paralelizar
void mat_mul_AtB(const Matrix* restrict A, const Matrix* restrict B, Matrix* restrict C) {
    const int rows_A = A->rows;   // Batch (256)
    const int cols_A = A->cols;   // Entrada (784 o 500)
    const int cols_B = B->cols;   // Salida (500 o 10)
    const float* restrict a_data = A->data;
    const float* restrict b_data = B->data;
    float* restrict c_data = C->data;
    
    // Paralelizar cuando cols_A es grande (784 neuronas de entrada)
    // Esta operación: 256 * 784 * 500 = 100M operaciones - ¡vale la pena!
    if (cols_A >= 500 && cols_B >= 10) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < cols_A; i++) {
            float* restrict c_row = &c_data[i * cols_B];
            
            // Inicializar fila local
            for (int j = 0; j < cols_B; j++) c_row[j] = 0.0f;
            
            // Acumular productos
            for (int k = 0; k < rows_A; k++) {
                const float a_ki = a_data[k * cols_A + i];
                const float* restrict b_row = &b_data[k * cols_B];
                #pragma omp simd
                for (int j = 0; j < cols_B; j++) {
                    c_row[j] += a_ki * b_row[j];
                }
            }
        }
    } else {
        // Versión secuencial con SIMD
        memset(c_data, 0, cols_A * cols_B * sizeof(float));
        for (int k = 0; k < rows_A; k++) {
            for (int i = 0; i < cols_A; i++) {
                const float a_ki = a_data[k * cols_A + i];
                const float* restrict b_row = &b_data[k * cols_B];
                float* restrict c_row = &c_data[i * cols_B];
                #pragma omp simd
                for (int j = 0; j < cols_B; j++) {
                    c_row[j] += a_ki * b_row[j];
                }
            }
        }
    }
}

// === OPERACIONES ELEMENTALES (Siempre secuenciales - muy pequeñas) ===

void mat_add(Matrix* restrict A, const Matrix* restrict B) {
    const int size = A->rows * A->cols;
    float* restrict a = A->data;
    const float* restrict b = B->data;
    
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void mat_subtract(Matrix* restrict A, const Matrix* restrict B) {
    const int size = A->rows * A->cols;
    float* restrict a = A->data;
    const float* restrict b = B->data;
    
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        a[i] -= b[i];
    }
}

// === FUNCIONES DE ACTIVACIÓN ===

void apply_sigmoid(Matrix* m) {
    const int size = m->rows * m->cols;
    float* restrict data = m->data;
    
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}

void sigmoid_derivative(const Matrix* restrict m, Matrix* restrict dest) {
    const int size = m->rows * m->cols;
    const float* restrict src = m->data;
    float* restrict dst = dest->data;
    
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        const float val = src[i];
        dst[i] = val * (1.0f - val);
    }
}

void apply_softmax(Matrix* m) {
    const int rows = m->rows;
    const int cols = m->cols;
    float* restrict data = m->data;
    
    for (int i = 0; i < rows; i++) {
        float* restrict row = &data[i * cols];
        
        float max_val = row[0];
        for (int j = 1; j < cols; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        
        const float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) {
            row[j] *= inv_sum;
        }
    }
}

Matrix* mat_transpose(Matrix* A) {
    Matrix* T = mat_init(A->cols, A->rows);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            T->data[j * T->cols + i] = A->data[i * A->cols + j];
        }
    }
    return T;
}