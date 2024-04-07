#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Function to generate matrices A, B, and C with sizes m x k, k x n, and m x n respectively
void generateMatrices(int m, int k, int n, std::vector<std::vector<float>>& A, std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0, 100.0); // Adjust range as needed

    // Resize matrices to appropriate sizes
    A.resize(m, std::vector<float>(k));
    B.resize(k, std::vector<float>(n));
    C.resize(m, std::vector<float>(n));

    // Generate random values for matrices A, B, and C
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            A[i][j] = dis(gen);
        }
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            B[i][j] = dis(gen);
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = dis(gen);
        }
    }
}

// Function to perform matrix multiplication: A * B += C
void matrixMultiplyAdd(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C) {
    int m = A.size();
    int k = B.size();
    int n = B[0].size();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A[i][l] * B[l][j];
            }
            C[i][j] += sum;
        }
    }
}

int main() {
    int m, k, n;
    std::cout << "Enter the size for matrices A(m x k), B(k x n), and C(m x n), specifically m, k, and n: ";
    std::cin >> m >> k >> n;

    if (m <= 0 || k <= 0 || n <= 0) {
        std::cerr << "Matrix dimensions must be positive integers." << std::endl;
        return 1;
    }

    // Declare our matrices A, B, and C
    std::vector<std::vector<float>> A, B, C;

    // Generate matrices A, B, and C
    generateMatrices(m, k, n, A, B, C);

    // Start the timer
    auto start = std::chrono::steady_clock::now();

    // Perform matrix multiplication A * B += C
    matrixMultiplyAdd(A, B, C);

    // Stop the timer
    auto end = std::chrono::steady_clock::now();

    // Print the result
    std::cout << "Result of A * B += C:" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Compute the duration and print it
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken for multiplication: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
