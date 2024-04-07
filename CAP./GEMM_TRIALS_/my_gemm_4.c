#include "include/my_gemm.h"
#include <omp.h>
#include <emmintrin.h>

void my_gemm(float *A, float *B, float *C, size_t m, size_t n, size_t k) {

    //-> PUT HERE YOUR GEMM CODE
    // Treat matrices as vectors and use vectorized operations
    // Reshape A and B to m x (k/4) and (k/4) x n respectively to match the SIMD register size (4 floats)
    // Perform the multiplication using SIMD instructions

    // Number of floats that can be loaded into an SSE register at once
    const size_t simd_width = 4;

    // Reshape A to m x (k/4)
    size_t m_reshape = m;
    size_t k_reshape = k / simd_width;
    
    // Reshape B to (k/4) x n
    size_t n_reshape = n;
    k_reshape = k / simd_width;

    // Perform the multiplication
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < m_reshape; ++i) {
        for (size_t j = 0; j < n_reshape; ++j) {
            // Initialize sum register to zero
            __m128 sum = _mm_setzero_ps();

            // Perform dot product using SIMD instructions
            for (size_t x = 0; x < k_reshape; ++x) {
                // Load 4 floats from A and B
                __m128 a = _mm_loadu_ps(&A[i * k_reshape * simd_width + x * simd_width]);
                __m128 b = _mm_loadu_ps(&B[x * n_reshape * simd_width + j * simd_width]);

                // Multiply-add operation
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }

            // Store the result in the output matrix C
            _mm_storeu_ps(&C[i * n_reshape * simd_width + j * simd_width], sum);
        }
    }
    //--------------------------------------------------------------------

}

