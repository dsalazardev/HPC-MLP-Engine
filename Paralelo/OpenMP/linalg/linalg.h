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