#include <stdio.h>
#include <pthread.h>
#include <immintrin.h>
#include <xmmintrin.h> // For SSE intrinsics
#include <time.h>
#include <stdlib.h>
#include <cpuid.h> // GCC/Clang


#ifndef ALIGNMENT_SIZE
    #define ALIGNMENT_SIZE 8
#endif

int supports_avx() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (ecx & (1 << 28)) != 0; // Check for AVX bit
}

int supports_avx2() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 5)) != 0; // Check for AVX2 bit
}

int supports_avx512() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 16)) != 0; // Check for AVX-512 bit
}

float dot_ps_single_thread(float* a, float* b, size_t size) {
    float acc = 0;
    for (int i = 0; i < size; i++) {
        acc += a[i] * b[i];
    }

    return acc;
}

float dot_ps_128(float* a, float* b, size_t size) {
    // We will be able to process 4 elements at a time
    __m128 sum_acc = _mm_setzero_ps(); // initialize sum_acc with 0s: 0 | 0 | 0 | 0
    size_t i = 0;
    size_t pack_size = 4;
    size_t full_packs = (size / pack_size) * pack_size;
    size_t remainder = (size - full_packs);

    for(; i < full_packs; i += pack_size) {
        __m128 a_vec = _mm_load_ps(a + i);
        __m128 b_vec = _mm_load_ps(b + i);

        __m128 prod = _mm_mul_ps(a_vec, b_vec);
        sum_acc = _mm_add_ps(sum_acc, prod);
    }

    float* result = aligned_alloc(8, pack_size * sizeof(float));
    _mm_store_ps(result, sum_acc);

    float final_result = 0;

    for (int i = 0; i < pack_size; i++) {
        final_result += result[i];
    }

    // handle the remaining part: Can be done either via remainder loop or masked intrinsic operation
    for (; i < size; i++) {
        final_result += a[i] * b[i];
    }

    return final_result;
}

float dot_ps_128_mm_dp(float* a, float* b, size_t size) {
    // We will be able to process 4 elements at a time
    float result = 0;
    size_t i = 0;
    size_t pack_size = 4;
    size_t full_packs = (size / pack_size) * pack_size;
    size_t remainder = (size - full_packs);

    for(; i < full_packs; i += pack_size) {
        __m128 a_vec = _mm_load_ps(a + i);
        __m128 b_vec = _mm_load_ps(b + i);

        float partial_dot = _mm_cvtss_f32(_mm_dp_ps(a_vec, b_vec, 0xFF)); // Does the dot product already
        result += partial_dot;
    }

    // handle the remaining part: Can be done either via remainder loop or masked intrinsic operation
    // for (; i < size; i++) {
    //     result += a[i] * b[i];
    // }
    
    if (remainder > 0) {
        __m128 mask;
        switch (remainder) {
            case 1:
                mask = _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, -1));
                break;
            case 2:
                mask = _mm_castsi128_ps(_mm_set_epi32(0, 0, -1, -1));
                break;
            case 3:
                mask = _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1));
                break;
            default:
                mask = _mm_setzero_ps();
        }

        __m128 a_vec = _mm_load_ps(&a[full_packs]);
        __m128 b_vec = _mm_load_ps(&b[full_packs]);

        // Apply mask
        a_vec = _mm_and_ps(a_vec, mask);
        b_vec = _mm_and_ps(b_vec, mask);

        float partial_dot = _mm_cvtss_f32(_mm_dp_ps(a_vec, b_vec, 0xFF)); // Does the dot product already
        result += partial_dot;
    }


    return result;
}

float dot_ps_256(float* a, float* b, size_t size) {
    __m256 sum_acc = _mm256_setzero_ps();
    size_t i = 0;
    size_t pack_size = 8;
    size_t full_packs = (size / pack_size) * pack_size;

    for (; i < full_packs; i += pack_size) {
        __m256 a_vec = _mm256_load_ps(a + i);
        __m256 b_vec = _mm256_load_ps(b + i);

        __m256 prod = _mm256_mul_ps(a_vec, b_vec);
        sum_acc = _mm256_add_ps(sum_acc, prod);
    }

    float* result = aligned_alloc(ALIGNMENT_SIZE, pack_size * sizeof(float)); // Make sure this is cache aligned, otherwise it throws segfault
    _mm256_store_ps(result, sum_acc);

    float final_result = 0;
    for (int j = 0; j < pack_size; j++) {
        final_result += result[j];
    }

    // Handle remaining elements
    for (; i < size; i++) {
        final_result += a[i] * b[i];
    }

    return final_result;
}

#ifdef __AVX512F__
float dot_ps_512(float* a, float* b, size_t size) {
    __m512 sum_acc = _mm512_setzero_ps();
    size_t i = 0;
    size_t pack_size = 16;
    size_t full_packs = (size / pack_size) * pack_size;

    for (; i < full_packs; i += pack_size) {
        __m512 a_vec = _mm512_load_ps(a + i);
        __m512 b_vec = _mm512_load_ps(b + i);

        __m512 prod = _mm512_mul_ps(a_vec, b_vec);
        sum_acc = _mm512_add_ps(sum_acc, prod);
    }

    float* result = aligned_alloc(ALIGNMENT_SIZE, pack_size);
    _mm512_store_ps(result, sum_acc);

    float final_result = 0;
    for (int j = 0; j < pack_size; j++) {
        final_result += result[j];
    }

    // Handle remaining elements
    for (; i < size; i++) {
        final_result += a[i] * b[i];
    }

    return final_result;
}
#endif

/*
Transpose helps get rid of cache misses
*/
void transpose_matrix(float* B, float* B_t, int K, int N) {
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            B_t[j * K + i] = B[i * N + j];
        }
    }
}

void matmul_single_thread(float* A, float* B_t, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; k += 1) {
                sum += A[i * K + k] + B_t[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_avx_optimized(float* A, float* B_t, float* C, int M, int K, int N) {
    float* temp = aligned_alloc(ALIGNMENT_SIZE, 8);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < K; k += 8) {
                __m256 a = _mm256_load_ps(&A[i * K + k]);   // A[i][k:k+7]
                __m256 b = _mm256_load_ps(&B_t[j * K + k]); // B_t[j][k:k+7]
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            _mm256_store_ps(temp, sum);
            float c_sum = temp[0] + temp[1] + temp[2] + temp[3] +
                          temp[4] + temp[5] + temp[6] + temp[7];
            C[i * N + j] = c_sum;
        }
    }
}

void matmul_avx_optimized_loop_unrolling(float* A, float* B_t, float* C, int M, int K, int N) {
    float* temp = aligned_alloc(ALIGNMENT_SIZE, 8 * sizeof(float));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            for (int k = 0; k < K; k += 16) {
                __m256 a1 = _mm256_load_ps(&A[i * K + k]);   // A[i][k:k+7]
                __m256 b1 = _mm256_load_ps(&B_t[j * K + k]); // B_t[j][k:k+7]
                sum1 = _mm256_fmadd_ps(a1, b1, sum1);

                __m256 a2 = _mm256_load_ps(&A[i * K + k + 8]);   // A[i][k:k+7]
                __m256 b2 = _mm256_load_ps(&B_t[j * K + k + 8]); // B_t[j][k:k+7]
                sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            }
            sum2 = _mm256_add_ps(sum1, sum2);
            _mm256_store_ps(temp, sum2);
            float c_sum = temp[0] + temp[1] + temp[2] + temp[3] +
                          temp[4] + temp[5] + temp[6] + temp[7];
            C[i * N + j] = c_sum;
        }
    }
}

#ifdef __AVX512F__
void matmul_avx_512_optimized_loop_unrolling(float* A, float* B_t, float* C, int M, int K, int N) {
    float* temp = aligned_alloc(ALIGNMENT_SIZE, 16 * sizeof(float));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            for (int k = 0; k < K; k += 32) {
                __m512 a1 = _mm512_load_ps(&A[i * K + k]);   // A[i][k:k+7]
                __m512 b1 = _mm512_load_ps(&B_t[j * K + k]); // B_t[j][k:k+7]
                sum1 = _mm512_fmadd_ps(a1, b1, sum1);

                __m512 a2 = _mm512_load_ps(&A[i * K + k + 16]);   // A[i][k:k+7]
                __m512 b2 = _mm512_load_ps(&B_t[j * K + k + 16]); // B_t[j][k:k+7]
                sum2 = _mm512_fmadd_ps(a2, b2, sum2);
            }
            sum2 = _mm512_add_ps(sum1, sum2);
            _mm512_store_ps(temp, sum2);
            float c_sum = temp[0] + temp[1] + temp[2] + temp[3] +
                          temp[4] + temp[5] + temp[6] + temp[7] +
                          temp[8] + temp[9] + temp[10] + temp[11] +
                          temp[12] + temp[13] + temp[14] + temp[15];
            C[i * N + j] = c_sum;
        }
    }
}
#endif

int main() {
    // Test data
    size_t length = 5000000; // Large vector size for benchmarking
    float* a = aligned_alloc(ALIGNMENT_SIZE, length * sizeof(float));
    
    for (size_t i = 0; i < length; i++)
        a[i] = 1.0;

    // Benchmark dot_ps_single_thread
    clock_t start = clock();
    float result1 = dot_ps_single_thread(a, a, length);
    clock_t end = clock();
    double time1 = (double)(end - start) / CLOCKS_PER_SEC;
    printf("dot_ps_single_thread result: %f, time: %f seconds\n", result1, time1);

    // Benchmark dot_ps_128
    start = clock();
    float result2 = dot_ps_128(a, a, length);
    end = clock();
    double time2 = (double)(end - start) / CLOCKS_PER_SEC;
    printf("dot_ps_128 result: %f, time: %f seconds\n", result2, time2);

    // Benchmark dot_ps_128_mm_pd
    start = clock();
    float result3 = dot_ps_128_mm_dp(a, a, length);
    end = clock();
    double time3 = (double)(end - start) / CLOCKS_PER_SEC;
    printf("dot_ps_128_mm_pd result: %f, time: %f seconds\n", result3, time3);

    // Benchmark dot_ps_256
    start = clock();
    float result4 = dot_ps_256(a, a, length);
    end = clock();
    double time4 = (double)(end - start) / CLOCKS_PER_SEC;
    printf("dot_ps_256 result: %f, time: %f seconds\n", result4, time4);

    // Benchmark dot_ps_512
    #ifdef __AVX512F__
    start = clock();
    float result5 = dot_ps_512(a, a, length);
    end = clock();
    double time5 = (double)(end - start) / CLOCKS_PER_SEC;
    printf("dot_ps_512 result: %f, time: %f seconds\n", result5, time5);
    #endif

    // Example matrix dimensions
    int M = 1536, K = 1536, N = 1536; // A: M x K, B: K x N, C: M x N
    float* A = aligned_alloc(ALIGNMENT_SIZE, M * K * sizeof(float));
    float* B = aligned_alloc(ALIGNMENT_SIZE, K * N * sizeof(float));
    float* B_t = aligned_alloc(ALIGNMENT_SIZE, K * N * sizeof(float)); // Transposed B
    float* C = aligned_alloc(ALIGNMENT_SIZE, M * N * sizeof(float));

    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Populate matrices A and B with random floating-point values
    for (int i = 0; i < M * K; ++i) {
        A[i] = (float)rand() / RAND_MAX; // Random float between 0 and 1
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = (float)rand() / RAND_MAX; // Random float between 0 and 1
    }

    // Transpose B into B_t
    start = clock();
    transpose_matrix(B, B_t, K, N);
    end = clock();
    printf("transpose time: %f seconds\n", ((double)(end - start) / CLOCKS_PER_SEC));

    start = clock();
    matmul_single_thread(A, B_t, C, M, K, N);
    end = clock();
    printf("matmul single thread time: %f seconds\n", ((double)(end - start) / CLOCKS_PER_SEC));

    start = clock();
    matmul_avx_optimized(A, B_t, C, M, K, N);
    end = clock();
    printf("matmul avx time: %f seconds\n", ((double)(end - start) / CLOCKS_PER_SEC));

    start = clock();
    matmul_avx_optimized_loop_unrolling(A, B_t, C, M, K, N);
    end = clock();
    printf("matmul avx unrolled loop time: %f seconds\n", ((double)(end - start) / CLOCKS_PER_SEC));

    #ifdef __AVX512F__
    start = clock();
    matmul_avx_512_optimized_loop_unrolling(A, B_t, C, M, K, N);
    end = clock();
    printf("matmul avx512 unrolled loop time: %f seconds\n", ((double)(end - start) / CLOCKS_PER_SEC));
    #endif

    return 0;
}