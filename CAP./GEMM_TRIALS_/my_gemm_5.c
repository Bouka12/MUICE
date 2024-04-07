#include "include/my_gemm.h"
#include <omp.h>
#include <xmmintrin.h>

void my_gemm(float *A, float *B, float *C, size_t m, size_t n, size_t k) {

    //-> PUT HERE YOUR GEMM CODE
    // Number of floats that can be loaded into an SSE register at once
    const size_t simd_width = 4;

    // Perform the multiplication using SAXPY operation
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // Initialize the result vector to zero
            __m128 result = _mm_setzero_ps();

            // Perform SAXPY operation
            for (size_t x = 0; x < k; ++x) {
                // Load elements from A and B
                __m128 a = _mm_loadu_ps(&A[i * k + x]);
                __m128 b = _mm_loadu_ps(&B[x * n + j]);

                // Multiply each element of A by the corresponding element of B and add to result
                result = _mm_add_ps(_mm_mul_ps(a, b), result);
            }

            // Store the result in the output matrix C
            _mm_storeu_ps(&C[i * n + j], result);
        }
    }
    //--------------------------------------------------------------------

}

