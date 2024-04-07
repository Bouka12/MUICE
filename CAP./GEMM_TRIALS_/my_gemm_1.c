#include "include/my_gemm.h"

void my_gemm(float *A, float *B, float  *C, size_t m, size_t n, size_t k) {
  
  //-> PUT HERE YOUR GEMM CODE
  //--------------------------------------------------------------------
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            __m128 sum = _mm_setzero_ps(); // Initialize sum register to zero

            for (size_t x = 0; x < k; x += 4) {
                __m128 a = _mm_loadu_ps(&A[i * k + x]); // Load 4 floats from A
                __m128 b = _mm_loadu_ps(&B[x * n + j]); // Load 4 floats from B

                sum = _mm_add_ps(sum, _mm_mul_ps(a, b)); // Multiply-add operation
            }

            // Sum all elements in the SSE register
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));

            // Store the result in the output matrix C
            _mm_store_ss(&C[i * n + j], sum);
        }
    }
  //--------------------------------------------------------------------

}

