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