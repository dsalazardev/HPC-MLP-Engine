#!/bin/bash
# medir_speedup.sh
echo "Hilos,Tiempo(s)" > openmp_speedup.csv

for hilos in 1 2 4 8 16
do
    export OMP_NUM_THREADS=$hilos
    echo "Ejecutando con $hilos hilos..."
    tiempo=$(./mlp_openmp | grep "Tiempo Total:" | awk '{print $3}')
    echo "$hilos,$tiempo" >> openmp_speedup.csv
done

echo "CSV generado: openmp_speedup.csv"
