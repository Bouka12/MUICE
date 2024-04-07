CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(practica3)
ENABLE_LANGUAGE(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -Wall")
set(-DNMC_VECTORIZE_TARGET=AVX2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")



add_executable(practica3_Numero-e-Identificador-de-los-Hilos practica3_Numero-e-Identificador-de-los-Hilos.cpp)
add_executable(practica3_Variables_threadprivate practica3_Variables_threadprivate.cpp)
add_executable(practica3_Construccion practica3_Construccion.cpp)
add_executable(practica3_Construccion_2 practica3_Construccion_2.cpp)
add_executable(practica3_paraleliza_practica1 practica3_paraleliza_practica1.cpp)
