cmake_minimum_required(VERSION 4.0)
project(Proyecto)

set(CMAKE_CXX_STANDARD 14)

add_executable(Proyecto
        Secuencial/C/linalg/linalg.h
        Secuencial/C/linalg/linalg.c
        Secuencial/C/network/network.c
        Secuencial/C/network/network.h
        Secuencial/C/main.c
        Secuencial/C/main.h
        Secuencial/C/common/mnist.c
        Secuencial/C/common/mnist.h
        Paralelo/OpenMP/network/network.c
        Paralelo/OpenMP/network/network.h
        Paralelo/OpenMP/main.c
        Paralelo/OpenMP/main.h
        Paralelo/OpenMP/common/mnist.c
        Paralelo/OpenMP/common/mnist.h
        Paralelo/OpenMP/linalg/linalg.h
        Paralelo/OpenMP/linalg/linalg.c
)
