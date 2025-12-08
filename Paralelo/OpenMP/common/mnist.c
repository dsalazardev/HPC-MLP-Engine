#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "mnist.h"

// Macro de seguridad HPC: Verifica que fread lea exactamente los elementos pedidos.
// Si falla, el programa se detiene (Assertion Failure).
#define CHECK_READ(call, expected) assert((call) == (expected))

// Función auxiliar para voltear Endianness (Big Endian -> Little Endian)
// Los archivos MNIST vienen en formato Big Endian (no nativo para Intel/AMD).
uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0x000000ff) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val << 24) & 0xff000000);
}

Matrix* mnist_load_images(const char* filename) {
    FILE* f = fopen(filename, "rb"); // "rb" es vital para lectura binaria
    if (!f) {
        printf("ERROR CRITICO: No se pudo abrir el archivo de imagenes: %s\n", filename);
        exit(1);
    }

    uint32_t magic, num_imgs, rows, cols;

    // Leemos encabezados validando la lectura con CHECK_READ
    CHECK_READ(fread(&magic, sizeof(uint32_t), 1, f), 1);
    CHECK_READ(fread(&num_imgs, sizeof(uint32_t), 1, f), 1);
    CHECK_READ(fread(&rows, sizeof(uint32_t), 1, f), 1);
    CHECK_READ(fread(&cols, sizeof(uint32_t), 1, f), 1);

    // Corregimos Endianness
    magic = swap_endian(magic);
    num_imgs = swap_endian(num_imgs);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic != 2051) {
        printf("ERROR: Magic number incorrecto en imagenes. Leido: %d (Esperado: 2051)\n", magic);
        fclose(f);
        exit(1);
    }

    printf("Cargando %d imagenes de %dx%d...\n", num_imgs, rows, cols);

    int inputs_per_img = rows * cols; // 784 pixels
    Matrix* m = mat_init(num_imgs, inputs_per_img);

    // Buffer temporal para leer bytes (unsigned char 0-255)
    unsigned char* buffer = (unsigned char*)malloc(inputs_per_img * sizeof(unsigned char));
    if (!buffer) {
        printf("ERROR: Fallo asignacion de memoria para buffer.\n");
        exit(1);
    }

    for (int i = 0; i < num_imgs; i++) {
        // Leemos 784 bytes de una sola imagen de golpe
        CHECK_READ(fread(buffer, sizeof(unsigned char), inputs_per_img, f), (size_t)inputs_per_img);

        // Normalizamos a float 0.0-1.0 y guardamos en la matriz
        for (int j = 0; j < inputs_per_img; j++) {
            m->data[i * m->cols + j] = (float)buffer[j] / 255.0f;
        }
    }

    free(buffer);
    fclose(f);
    return m;
}

Matrix* mnist_load_labels(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("ERROR CRITICO: No se pudo abrir el archivo de etiquetas: %s\n", filename);
        exit(1);
    }

    uint32_t magic, num_labels;

    // Leemos encabezados validando lectura
    CHECK_READ(fread(&magic, sizeof(uint32_t), 1, f), 1);
    CHECK_READ(fread(&num_labels, sizeof(uint32_t), 1, f), 1);

    magic = swap_endian(magic);
    num_labels = swap_endian(num_labels);

    if (magic != 2049) {
        printf("ERROR: Magic number incorrecto en etiquetas. Leido: %d (Esperado: 2049)\n", magic);
        fclose(f);
        exit(1);
    }

    printf("Cargando %d etiquetas...\n", num_labels);

    // Creamos matriz para One-Hot Encoding (N x 10)
    // mat_init usa calloc, así que todo inicia en 0.0
    Matrix* m = mat_init(num_labels, 10);

    unsigned char label;
    for (int i = 0; i < num_labels; i++) {
        // Leemos 1 byte (la etiqueta 0-9)
        CHECK_READ(fread(&label, sizeof(unsigned char), 1, f), 1);
        
        // One-Hot: Ponemos un 1.0 en la columna correspondiente
        if (label < 10) {
            m->data[i * 10 + label] = 1.0f;
        } else {
            printf("Advertencia: Etiqueta %d fuera de rango en indice %d\n", label, i);
        }
    }

    fclose(f);
    return m;
}