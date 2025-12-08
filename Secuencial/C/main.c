#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network/network.h"
#include "common/mnist.h"

// Función auxiliar para copiar un pedazo de datos (Batch)
// Copia 'batch_size' filas desde 'src' (empezando en 'start_row') hacia 'dest'
void fill_batch(Matrix* dest, Matrix* src, int start_row, int batch_size) {
    // Validar que no nos salgamos del array
    if (start_row + batch_size > src->rows) {
        batch_size = src->rows - start_row; // Ajuste para el último batch si sobra
    }

    // Copiamos fila por fila
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < src->cols; j++) {
            // src index: (fila_actual + i) * ancho + col
            // dest index: i * ancho + col
            dest->data[i * dest->cols + j] = src->data[(start_row + i) * src->cols + j];
        }
    }
}

int main() {
    srand(time(NULL));

    printf("=== ESCENARIO 1b: C SECUENCIAL (MNIST REAL) ===\n");

    // 1. Cargar Datos Reales (MNIST)

    // Rutas relativas asumiendo que estás en /Secuencial/C
    const char* TRAIN_IMG = "../../Dataset/archive/train-images.idx3-ubyte";
    const char* TRAIN_LBL = "../../Dataset/archive/train-labels.idx1-ubyte";

    printf("Cargando dataset MNIST...\n");
    Matrix* X_train_full = mnist_load_images(TRAIN_IMG);
    Matrix* Y_train_full = mnist_load_labels(TRAIN_LBL);

    if (X_train_full == NULL || Y_train_full == NULL) {
        printf("Error: Fallo al cargar los datos. Revisa las rutas.\n");
        return 1;
    }

    printf("Datos cargados: %d imagenes\n", X_train_full->rows);

    // 2. Configuración de la Red

    int batch_size = 32;
    int epochs = 10;
    float learning_rate = 0.1f;
    int topology[] = {784, 500, 10}; // 784 Inputs -> 128 Hidden -> 10 Outputs

    printf("Inicializando red neural...\n");
    Network* net = net_create(topology, 3, learning_rate, batch_size);
    // Matrices temporales para el Batch (reutilizables)
    // Esto evita hacer malloc/free miles de veces dentro del bucle
    Matrix* X_batch = mat_init(batch_size, 784);
    Matrix* Y_batch = mat_init(batch_size, 10);

    // 3. Bucle de Entrenamiento

    printf("--- Iniciando Entrenamiento ---\n");
    clock_t start_total = clock();

    int total_samples = X_train_full->rows;
    int num_batches = total_samples / batch_size;

    for (int e = 0; e < epochs; e++) {
        float epoch_loss = 0.0f;
        clock_t start_epoch = clock();

        // Iterar sobre todo el dataset en bloques (batches)
        for (int b = 0; b < num_batches; b++) {
            int start_row = b * batch_size;

            // A. Llenar el batch actual con datos
            fill_batch(X_batch, X_train_full, start_row, batch_size);
            fill_batch(Y_batch, Y_train_full, start_row, batch_size);

            // B. Forward Pass
            Matrix* predictions = net_forward(net, X_batch);

            // C. Backward Pass (Calcula gradientes)
            net_backward(net, X_batch, Y_batch);

            // D. Update Weights (Aplica gradientes)
            net_update(net, batch_size);

            // E. Calcular Loss (MSE simple para monitoreo)
            // Sumamos el error cuadrado de este batch
            for(int k=0; k < batch_size * 10; k++) {
                float diff = predictions->data[k] - Y_batch->data[k];
                epoch_loss += diff * diff;
            }
        }

        // Promedio del loss por epoch
        epoch_loss /= total_samples;

        double time_epoch = (double)(clock() - start_epoch) / CLOCKS_PER_SEC;
        printf("Epoch %d/%d - Loss: %.6f - Tiempo: %.2fs\n",
               e+1, epochs, epoch_loss, time_epoch);
    }

    clock_t end_total = clock();
    double time_spent = (double)(end_total - start_total) / CLOCKS_PER_SEC;
    printf("\n=== ENTRENAMIENTO FINALIZADO ===\n");
    printf("Tiempo Total: %.4f segundos\n", time_spent);

    // 4. Limpieza de Memoria (Vital en C)

    mat_free(X_train_full);
    mat_free(Y_train_full);
    mat_free(X_batch);
    mat_free(Y_batch);
    // net_free(net); // Implementar si tienes la función, o dejar que el OS limpie al salir.
    
    return 0;
}