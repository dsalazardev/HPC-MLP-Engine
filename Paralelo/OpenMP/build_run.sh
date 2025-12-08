#!/bin/bash
rm -f mlp_openmp

echo "Compilando versión OPENMP..."

# AGREGAMOS EL FLAG: -fopenmp
# Mantenemos las optimizaciones secuenciales (-O3, -march=native, etc)
gcc main.c common/mnist.c network/network.c linalg/linalg.c -o mlp_openmp -lm -O3 -march=native -ffast-math -funroll-loops -fopenmp -Wall

if [ $? -eq 0 ]; then
    echo "Ejecutando con todos los hilos..."
    # Opcional: Definir hilos explícitamente si quieres probar escalabilidad
    # export OMP_NUM_THREADS=4
    ./mlp_openmp
else
    echo "Error en la compilación."
fi