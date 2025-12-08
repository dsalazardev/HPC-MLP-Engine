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
// Backward Pass Optimizado (Cache Friendly + Vectorización AVX)
void net_backward(Network* net, Matrix* input, Matrix* target) {

    // Recorremos las capas desde la última hacia atrás
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
            // Capa Oculta:
            // Queremos calcular: delta = (delta_next * W_next^T) .* f'(A)

            Layer* next_l = net->layers[i+1];
            Matrix* err = l->delta;     // Destino (delta actual)
            Matrix* W = next_l->W;      // Pesos siguiente capa
            Matrix* D = next_l->delta;  // Delta siguiente capa

            // --- OPTIMIZACIÓN CRÍTICA (r-c-k loop) ---
            // Cambiamos el orden de los bucles para acceder a W y D secuencialmente.
            // Esto permite que el compilador use instrucciones SIMD (AVX) para sumar 8 floats a la vez.

            for (int r = 0; r < D->rows; r++) {       // 1. Batch (filas)
                for (int c = 0; c < W->rows; c++) {   // 2. Neuronas Actuales (filas de W)

                    float sum = 0.0f; // Acumulador en registro de CPU (muy rápido)

                    // 3. Neuronas Siguientes (columnas de W / columnas de D)
                    // Aquí es donde ocurre la magia: W[c][k] y D[r][k] se leen secuencialmente.
                    // ¡Sin saltos de memoria!
                    for (int k = 0; k < D->cols; k++) {
                        sum += D->data[r * D->cols + k] * W->data[c * W->cols + k];
                    }

                    err->data[r * err->cols + c] = sum;
                }
            }

            // Paso 2: Multiplicar por derivada de Sigmoide (Element-wise)
            for(int k=0; k < size; k++) {
                float a_val = l->A->data[k];
                // f'(x) = f(x) * (1 - f(x))
                l->delta->data[k] *= (a_val * (1.0f - a_val));
            }
        }

        // --- B. CALCULO DE GRADIENTES (dW) ---
        // dW = prev_A^T * delta
        // Usamos nuestra función optimizada mat_mul_AtB de linalg.c
        // Esta función ya usa bloques y acceso lineal internamente.
        mat_mul_AtB(prev_A, l->delta, l->dW);

        // --- C. CALCULO DE GRADIENTES (db) ---
        // db = Sum(delta, axis=0) -> Sumar las columnas del batch

        // Limpiamos el gradiente de bias
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