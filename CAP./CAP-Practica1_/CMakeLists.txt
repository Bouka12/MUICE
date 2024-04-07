CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(practica1)
ENABLE_LANGUAGE(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -Wall")
set(-DNMC_VECTORIZE_TARGET=AVX2)



add_executable(practica1_vectors practica1_vectors.cpp)
add_executable(practica1_matrices practica1_matrices.cpp)
 
