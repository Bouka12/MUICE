#include <omp.h>
#include <iostream>
#include <vector>

int main() {
    int N;
    double beta;

    // User input for N and beta
    std::cout << "Enter the size of the vectors (N): ";
    std::cin >> N;
    std::cout << "Enter the value of beta: ";
    std::cin >> beta;

    // Check if N is even, if not, decrement by 1
    if (N % 2 != 0) {
        N--;
        std::cout << "N adjusted to " << N << " to ensure even number of elements." << std::endl;
    }

    // Declare vectors A, B, and C
    std::vector<double> A(N), B(N), C(N);

    // Initialize vectors A and B with some values (for demonstration)
    for (int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Compute start time
    double start_time = omp_get_wtime();

    // Parallelize the sum of even elements of vectors A and B into vector C
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            C[i] = A[i] + B[i] + C[i] * beta;
        }
    }

    // Compute end time
    double end_time = omp_get_wtime();
    /*
    // Output the results (for demonstration)
    std::cout << "Result vector C:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }
    */

    // Compute and output the elapsed time
    std::cout << "Elapsed time: " << end_time - start_time << " seconds" << std::endl;

    return 0;
}
