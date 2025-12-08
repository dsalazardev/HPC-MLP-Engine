#include "network.h"
#include <string.h>
#include <omp.h> // <--- IMPORTANTE

Network* net_create(int* topology, int n_layers, float lr, int batch_size) {
    Network* net = (Network*)malloc(sizeof(Network));
    net->num_layers = n_layers - 1;
    net->learning_rate = lr;
    net->layers = (Layer**)malloc(net->num_layers * sizeof(Layer*));

    for (int i = 0; i < net->num_layers; i++) {
        net->layers[i] = (Layer*)malloc(sizeof(Layer));
        int n_in = topology[i];
        int n_out = topology[i+1];

        // 1. Pesos y Gradientes (Usamos mat_init de linalg que ya alinea memoria)
        net->layers[i]->W = mat_init(n_in, n_out);
        mat_randomize(net->layers[i]->W);
        net->layers[i]->b = mat_init(1, n_out);

        net->layers[i]->dW = mat_init(n_in, n_out);
        net->layers[i]->db = mat_init(1, n_out);

        // 2. Buffers
        net->layers[i]->Z = mat_init(batch_size, n_out);
        net->layers[i]->A = mat_init(batch_size, n_out);
        net->layers[i]->delta = mat_init(batch_size, n_out);
    }
    return net;
}

Matrix* net_forward(Network* net, Matrix* input) {
    Matrix* current_input = input;

    for (int i = 0; i < net->num_layers; i++) {
        Layer* l = net->layers[i];

        // 1. Z = X * W (Esto ya llama al mat_mul paralelizado de linalg.c)
        mat_mul(current_input, l->W, l->Z);

        // 2. Sumar Bias
        // ANTES: Secuencial
        // AHORA: Paralelo (Collapse fusiona filas y cols para tener más hilos trabajando)
        int rows = l->Z->rows;
        int cols = l->Z->cols;
        float* z_ptr = l->Z->data;
        float* b_ptr = l->b->data;

        #pragma omp parallel for collapse(2) schedule(static)
        for(int r=0; r < rows; r++) {
            for(int c=0; c < cols; c++) {
                z_ptr[r*cols + c] += b_ptr[c];
            }
        }

        // 3. Activación
        // Copia de memoria (paralelizable si es grande, pero memcpy suele ser mejor)
        memcpy(l->A->data, l->Z->data, rows * cols * sizeof(float));

        if (i == net->num_layers - 1) {
            apply_softmax(l->A); // Softmax en linalg.c ya tiene pragmas (o debería)
        } else {
            apply_sigmoid(l->A); // Sigmoid en linalg.c ya tiene pragmas
        }

        current_input = l->A;
    }
    return current_input;
}

void net_backward(Network* net, Matrix* input, Matrix* target) {

    for (int i = net->num_layers - 1; i >= 0; i--) {
        Layer* l = net->layers[i];
        Matrix* prev_A = (i == 0) ? input : net->layers[i-1]->A;

        // --- A. CALCULO DEL ERROR (delta) ---
        int size = l->delta->rows * l->delta->cols;

        if (i == net->num_layers - 1) {
            // Capa Salida: delta = A - Target
            // Paralelizamos este bucle simple
            #pragma omp parallel for schedule(static)
            for(int k=0; k < size; k++) {
                l->delta->data[k] = l->A->data[k] - target->data[k];
            }
        } else {
            // Capa Oculta
            Layer* next_l = net->layers[i+1];
            Matrix* err = l->delta;
            Matrix* W = next_l->W;
            Matrix* D = next_l->delta;

            // Limpiamos buffer
            memset(err->data, 0, size * sizeof(float));

            // Paso 1: Propagar error (Manual Matrix Mult)
            // ESTE ERA EL CUELLO DE BOTELLA SECUENCIAL
            // Ahora usamos collapse(2) para que OpenMP use todos los núcleos

            #pragma omp parallel for collapse(2) schedule(static)
            for (int r = 0; r < D->rows; r++) {       // Batch
                for (int c = 0; c < W->rows; c++) {   // Neuronas Curr
                    float sum = 0.0f;
                    // Bucle interno K es secuencial para vectorización SIMD dentro del hilo
                    for (int k = 0; k < D->cols; k++) { // Neuronas Next
                        sum += D->data[r * D->cols + k] * W->data[c * W->cols + k];
                    }
                    err->data[r * err->cols + c] = sum;
                }
            }

            // Paso 2: Derivada
            #pragma omp parallel for schedule(static)
            for(int k=0; k < size; k++) {
                float a_val = l->A->data[k];
                l->delta->data[k] *= (a_val * (1.0f - a_val));
            }
        }

        // --- B. CALCULO DE GRADIENTES (dW) ---
        // Llama a mat_mul_AtB de linalg.c que YA debe estar paralelizada
        mat_mul_AtB(prev_A, l->delta, l->dW);

        // --- C. CALCULO DE GRADIENTES (db) ---
        // Suma de columnas (Reducción)
        // Esto es delicado en paralelo por race condition en l->db.
        // Como db es pequeño (10 o 500 elementos), mejor hacerlo secuencial o
        // paralelizar por columnas (c)

        memset(l->db->data, 0, l->db->cols * sizeof(float));

        #pragma omp parallel for schedule(static)
        for(int c=0; c < l->delta->cols; c++) {
            float sum = 0.0f;
            for(int r=0; r < l->delta->rows; r++) {
                sum += l->delta->data[r * l->delta->cols + c];
            }
            l->db->data[c] = sum;
        }
    }
}

void net_update(Network* net, int batch_size) {
    // Paralelismo a nivel de capas no vale la pena (son pocas)
    // Paralelismo DENTRO de la actualización de pesos SI vale la pena (son muchos pesos)

    for (int i = 0; i < net->num_layers; i++) {
        Layer* l = net->layers[i];
        float scalar = net->learning_rate / batch_size;

        int w_size = l->W->rows * l->W->cols;

        // Actualizar Pesos en Paralelo
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < w_size; j++) {
            l->W->data[j] -= l->dW->data[j] * scalar;
        }

        int b_size = l->b->cols;
        // Bias es pequeño, el overhead de crear hilos puede no valer la pena,
        // pero 'omp parallel for' con static suele ser ligero.
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < b_size; j++) {
            l->b->data[j] -= l->db->data[j] * scalar;
        }
    }
}