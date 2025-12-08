#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "network/network.h"
#include "common/mnist.h"

// Función auxiliar para copiar un pedazo de datos (Batch)
void fill_batch(Matrix* dest, Matrix* src, int start_row, int batch_size) {
    if (start_row + batch_size > src->rows) {
        batch_size = src->rows - start_row;
    }

    // Copiamos usando memcpy por fila (más rápido)
    const int cols = src->cols;
    for (int i = 0; i < batch_size; i++) {
        memcpy(&dest->data[i * cols], 
               &src->data[(start_row + i) * cols], 
               cols * sizeof(float));
    }
}

int main() {
    srand(time(NULL));
    
    // Mostrar configuración OpenMP
    printf("=== MLP CON OPENMP ===\n");
    printf("Threads OpenMP: %d\n", omp_get_max_threads());

    // 1. Cargar Datos de Entrenamiento
    const char* TRAIN_IMG = "../../Dataset/archive/train-images.idx3-ubyte";
    const char* TRAIN_LBL = "../../Dataset/archive/train-labels.idx1-ubyte";
    const char* TEST_IMG = "../../Dataset/archive/t10k-images.idx3-ubyte";
    const char* TEST_LBL = "../../Dataset/archive/t10k-labels.idx1-ubyte";

    printf("Cargando dataset MNIST...\n");
    Matrix* X_train_full = mnist_load_images(TRAIN_IMG);
    Matrix* Y_train_full = mnist_load_labels(TRAIN_LBL);
    Matrix* X_test = mnist_load_images(TEST_IMG);
    Matrix* Y_test = mnist_load_labels(TEST_LBL);

    if (X_train_full == NULL || Y_train_full == NULL || X_test == NULL || Y_test == NULL) {
        printf("Error: Fallo al cargar los datos.\n");
        return 1;
    }

    printf("Datos cargados: %d train, %d test\n", X_train_full->rows, X_test->rows);

    // 2. Configuración
    // CLAVE: batch_size grande para justificar OpenMP
    // Con batch_size=32, el overhead de threads mata el rendimiento
    // Con batch_size=256+, empezamos a ver beneficios
    int batch_size = 256;  // AUMENTADO para mejor paralelismo
    int epochs = 10;
    float learning_rate = 0.1f;
    int topology[] = {784, 500, 10};

    printf("Batch size: %d\n", batch_size);
    printf("Inicializando red neural...\n");
    
    Network* net = net_create(topology, 3, learning_rate, batch_size);
    Matrix* X_batch = mat_init(batch_size, 784);
    Matrix* Y_batch = mat_init(batch_size, 10);

    // 3. Entrenamiento
    printf("--- Iniciando Entrenamiento ---\n");
    double start_total = omp_get_wtime();

    int total_samples = X_train_full->rows;
    int num_batches = total_samples / batch_size;

    for (int e = 0; e < epochs; e++) {
        float epoch_loss = 0.0f;
        double start_epoch = omp_get_wtime();

        for (int b = 0; b < num_batches; b++) {
            int start_row = b * batch_size;

            fill_batch(X_batch, X_train_full, start_row, batch_size);
            fill_batch(Y_batch, Y_train_full, start_row, batch_size);

            Matrix* predictions = net_forward(net, X_batch);
            net_backward(net, X_batch, Y_batch);
            net_update(net, batch_size);

            // Loss
            for(int k = 0; k < batch_size * 10; k++) {
                float diff = predictions->data[k] - Y_batch->data[k];
                epoch_loss += diff * diff;
            }
        }

        epoch_loss /= total_samples;
        double time_epoch = omp_get_wtime() - start_epoch;
        printf("Epoch %d/%d - Loss: %.6f - Tiempo: %.2fs\n",
               e+1, epochs, epoch_loss, time_epoch);
    }

    double time_total = omp_get_wtime() - start_total;
    printf("\n=== ENTRENAMIENTO FINALIZADO ===\n");
    printf("Tiempo Total: %.4f segundos\n", time_total);

    // 4. Evaluación en Test Set (procesando en batches)
    printf("\nEvaluando modelo en conjunto de Test...\n");
    int correct = 0;
    int test_samples = X_test->rows;
    int test_batches = (test_samples + batch_size - 1) / batch_size;  // Redondeo hacia arriba
    
    Matrix* X_test_batch = mat_init(batch_size, 784);
    
    for (int b = 0; b < test_batches; b++) {
        int start_row = b * batch_size;
        int current_batch_size = batch_size;
        if (start_row + batch_size > test_samples) {
            current_batch_size = test_samples - start_row;
        }
        
        // Llenar batch de test
        fill_batch(X_test_batch, X_test, start_row, current_batch_size);
        
        // Forward pass
        Matrix* predictions = net_forward(net, X_test_batch);
        
        // Evaluar predicciones de este batch
        for (int i = 0; i < current_batch_size; i++) {
            int global_idx = start_row + i;
            
            // Encontrar índice predicho (argmax)
            int pred_idx = 0;
            float max_val = predictions->data[i * 10];
            for (int j = 1; j < 10; j++) {
                if (predictions->data[i * 10 + j] > max_val) {
                    max_val = predictions->data[i * 10 + j];
                    pred_idx = j;
                }
            }
            
            // Encontrar índice real (argmax de one-hot)
            int real_idx = 0;
            for (int j = 0; j < 10; j++) {
                if (Y_test->data[global_idx * 10 + j] > 0.5f) {
                    real_idx = j;
                    break;
                }
            }
            
            if (pred_idx == real_idx) correct++;
        }
    }
    
    float accuracy = (float)correct / test_samples * 100.0f;
    printf("Precision (Accuracy): %.2f%%\n", accuracy);

    // 5. Resumen para Informe
    printf("\n============================================================\n");
    printf("RESUMEN PARA INFORME TECNICO\n");
    printf("============================================================\n");
    printf("Escenario: C Paralelo (OpenMP)\n");
    printf("Threads:   %d\n", omp_get_max_threads());
    printf("Batch Size: %d\n", batch_size);
    printf("Tiempo Total: %.4f s\n", time_total);
    printf("Accuracy:  %.2f%%\n", accuracy);
    printf("============================================================\n");

    // 6. Limpieza
    mat_free(X_train_full);
    mat_free(Y_train_full);
    mat_free(X_test);
    mat_free(Y_test);
    mat_free(X_batch);
    mat_free(Y_batch);
    mat_free(X_test_batch);
    
    return 0;
}