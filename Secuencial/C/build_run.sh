#!/bin/bash

# Limpiar ejecutable anterior
rm -f mlp_secuencial

echo "Compilando con Optimización EXTREMA (AVX + FastMath)..."

# AGREGAMOS: -march=native -ffast-math -funroll-loops
gcc main.c common/mnist.c network/network.c linalg/linalg.c -o mlp_secuencial -lm -O3 -march=native -ffast-math -funroll-loops -Wall

if [ $? -eq 0 ]; then
    echo "Compilación exitosa. Ejecutando..."
    ./mlp_secuencial
else
    echo "Error en la compilación."
fi