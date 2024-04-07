#include <iostream>
#include <vector>
#include <random>
#include<chrono>

// Función para generar vectores A, B y C de tamaño N
void generateVectors(int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C) {
    // Inicializar generador de números aleatorios
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0, 100.0); // Ajustar rango según sea necesario

    // Redimensionar vectores al tamaño N
    A.resize(N);
    B.resize(N);
    C.resize(N);

    // Generar valores aleatorios para los vectores A, B y C
    for (int i = 0; i < N; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
        C[i] = dis(gen);
    }
}

// Función para realizar la operación A * B += C
void vectorMultiplyAdd(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C) {
    int N = A.size(); // Suponiendo que todos los vectores tienen el mismo tamaño
    for (int i = 0; i < N; ++i) {
        A[i] = A[i] * B[i] + C[i];
    }
}

int main() {
    int N;
    std::cout << "Ingrese el tamaño N para los vectores: ";
    std::cin >> N;

    if (N <= 0) {
        std::cerr << "El tamaño N debe ser un entero positivo." << std::endl;
        return 1;
    }

    // Declarar nuestros vectores A, B y C
    std::vector<float> A, B, C;

    // Generar vectores A, B y C
    generateVectors(N, A, B, C);

    // Start the timer
    auto start = std::chrono::steady_clock::now();

    // Realizar la operación A * B += C
    vectorMultiplyAdd(A, B, C);

    // STop the timer
    auto end = std::chrono::steady_clock::now();

    // Imprimir el resultado
    std::cout << "Resultado de A * B += C:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;

    // Compute the duration and print it
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout<<"Time taken for multiplication: "<< duration.count()<< " milliseconds"<< std::endl;


    return 0;
}
