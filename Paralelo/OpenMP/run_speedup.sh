#!/bin/bash
# medir_speedup.sh

echo "Hilos,Tiempo(s),Accuracy(%)" > openmp_speedup.csv

for hilos in 1 2 4 8 16
do
    export OMP_NUM_THREADS=$hilos
    echo "Ejecutando con $hilos hilos..."

    salida=$(./mlp_openmp)

    tiempo=$(echo "$salida" | grep "Tiempo Total:" | awk '{print $3}')
    acc=$(echo "$salida" | grep -E "Accuracy|Precisión" | awk '{print $2}' | tr -d '%')

    echo "$hilos,$tiempo,$acc" >> openmp_speedup.csv
done

echo "CSV generado: openmp_speedup.csv"
