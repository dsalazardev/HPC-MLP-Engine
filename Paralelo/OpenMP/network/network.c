#include "network.h"
#include <string.h>
#include <omp.h>

Network* net_create(int* topology, int n_layers, float lr, int batch_size) {
    Network* net = (Network*)malloc(sizeof(Network));
    net->num_layers = n_layers - 1;
    net->learning_rate = lr;
    net->layers = (Layer**)malloc(net->num_layers * sizeof(Layer*));

    for (int i = 0; i < net->num_layers; i++) {
        net->layers[i] = (Layer*)malloc(sizeof(Layer));
        int n_in = topology[i];
        int n_out = topology[i+1];

        net->layers[i]->W = mat_init(n_in, n_out);
        mat_randomize(net->layers[i]->W);
        net->layers[i]->b = mat_init(1, n_out);

        net->layers[i]->dW = mat_init(n_in, n_out);
        net->layers[i]->db = mat_init(1, n_out);

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

        // 1. Z = X * W
        mat_mul(current_input, l->W, l->Z);

        // 2. Sumar Bias
        const int rows = l->Z->rows;
        const int cols = l->Z->cols;
        float* restrict z_ptr = l->Z->data;
        const float* restrict b_ptr = l->b->data;

        for(int r = 0; r < rows; r++) {
            for(int c = 0; c < cols; c++) {
                z_ptr[r * cols + c] += b_ptr[c];
            }
        }

        // 3. Activación
        memcpy(l->A->data, l->Z->data, rows * cols * sizeof(float));

        if (i == net->num_layers - 1) {
            apply_softmax(l->A);
        } else {
            apply_sigmoid(l->A);
        }

        current_input = l->A;
    }
    return current_input;
}

void net_backward(Network* net, Matrix* input, Matrix* target) {

    for (int i = net->num_layers - 1; i >= 0; i--) {
        Layer* l = net->layers[i];
        Matrix* prev_A = (i == 0) ? input : net->layers[i-1]->A;

        const int size = l->delta->rows * l->delta->cols;

        if (i == net->num_layers - 1) {
            // Capa Salida: delta = A - Target
            float* restrict delta = l->delta->data;
            const float* restrict a = l->A->data;
            const float* restrict t = target->data;
            
            for(int k = 0; k < size; k++) {
                delta[k] = a[k] - t[k];
            }
        } else {
            // Capa Oculta - CUELLO DE BOTELLA PRINCIPAL
            Layer* next_l = net->layers[i+1];
            float* restrict err = l->delta->data;
            const float* restrict W = next_l->W->data;
            const float* restrict D = next_l->delta->data;
            const float* restrict A_data = l->A->data;
            
            const int batch = next_l->delta->rows;
            const int curr_neurons = next_l->W->rows;
            const int next_neurons = next_l->delta->cols;

            // Propagación de error - PARALELIZAR por batch (filas independientes)
            // batch=256, curr_neurons=500, next_neurons=10
            // 256 * 500 * 10 = 1.28M operaciones - vale la pena!
            #pragma omp parallel for schedule(static) if(batch >= 128 && curr_neurons >= 100)
            for (int r = 0; r < batch; r++) {
                const float* restrict D_row = &D[r * next_neurons];
                const float* restrict A_row = &A_data[r * curr_neurons];
                float* restrict err_row = &err[r * curr_neurons];
                
                for (int c = 0; c < curr_neurons; c++) {
                    float sum = 0.0f;
                    const float* restrict W_row = &W[c * next_neurons];
                    
                    #pragma omp simd reduction(+:sum)
                    for (int k = 0; k < next_neurons; k++) {
                        sum += D_row[k] * W_row[k];
                    }
                    
                    // Aplicar derivada sigmoid inline
                    const float a_val = A_row[c];
                    err_row[c] = sum * a_val * (1.0f - a_val);
                }
            }
        }

        // Gradientes dW
        mat_mul_AtB(prev_A, l->delta, l->dW);

        // Gradientes db
        const int db_cols = l->db->cols;
        const int delta_rows = l->delta->rows;
        float* restrict db = l->db->data;
        const float* restrict delta = l->delta->data;
        
        for(int c = 0; c < db_cols; c++) {
            float sum = 0.0f;
            for(int r = 0; r < delta_rows; r++) {
                sum += delta[r * db_cols + c];
            }
            db[c] = sum;
        }
    }
}

void net_update(Network* net, int batch_size) {
    const float scalar = net->learning_rate / batch_size;

    for (int i = 0; i < net->num_layers; i++) {
        Layer* l = net->layers[i];
        
        const int w_size = l->W->rows * l->W->cols;
        float* restrict W = l->W->data;
        const float* restrict dW = l->dW->data;

        for (int j = 0; j < w_size; j++) {
            W[j] -= dW[j] * scalar;
        }

        const int b_size = l->b->cols;
        float* restrict b = l->b->data;
        const float* restrict db_data = l->db->data;
        
        for (int j = 0; j < b_size; j++) {
            b[j] -= db_data[j] * scalar;
        }
    }
}