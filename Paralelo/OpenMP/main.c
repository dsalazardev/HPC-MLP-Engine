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

    // 1. Cargar Datos
    const char* TRAIN_IMG = "../../Dataset/archive/train-images.idx3-ubyte";
    const char* TRAIN_LBL = "../../Dataset/archive/train-labels.idx1-ubyte";

    printf("Cargando dataset MNIST...\n");
    Matrix* X_train_full = mnist_load_images(TRAIN_IMG);
    Matrix* Y_train_full = mnist_load_labels(TRAIN_LBL);

    if (X_train_full == NULL || Y_train_full == NULL) {
        printf("Error: Fallo al cargar los datos.\n");
        return 1;
    }

    printf("Datos cargados: %d imagenes\n", X_train_full->rows);

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

    // 4. Limpieza
    mat_free(X_train_full);
    mat_free(Y_train_full);
    mat_free(X_batch);
    mat_free(Y_batch);
    
    return 0;
}