#!/bin/bash

# 1. Limpiar ejecutable anterior
rm -f mlp_openmp

echo "==============================================="
echo " Compilando versión OPENMP (AVX + 4 Hilos) "
echo "==============================================="

# 2. Compilar
# AGREGADO VITAL: -fopenmp (Activa los #pragma)
# CAMBIO DE NOMBRE: -o mlp_openmp (Para no confundir con el secuencial)
gcc main.c common/mnist.c network/network.c linalg/linalg.c -o mlp_openmp -lm -O3 -march=native -ffast-math -funroll-loops -fopenmp -Wall

# 3. Verificar compilación
if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa."

    # --- CONFIGURACIÓN DE HILOS ---
    # Para Batch Size pequeño (32), usar todos los hilos crea mucho "ruido" (overhead).
    # 4 hilos suele ser el equilibrio perfecto entre velocidad y gestión.
    export OMP_NUM_THREADS=4

    echo "Ejecutando con OMP_NUM_THREADS=$OMP_NUM_THREADS ..."
    ./mlp_openmp
else
    echo "❌ Error en la compilación."
fi