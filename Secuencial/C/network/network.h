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