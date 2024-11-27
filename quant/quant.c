#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stddef.h>

#include "../utils/utils.c"

#define ALIGNMENT_SIZE 64
#define ARRAY_SIZE 100000

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

struct QCS {
    float s;
    float z;
};

void compare_quantized_data(float* original, float* dequantized, int size) {
    float mae = 0.0f; // Mean Absolute Error
    float mse = 0.0f; // Mean Squared Error
    float max_error = 0.0f; // Maximum Absolute Error

    // Compute metrics
    for (int i = 0; i < size; ++i) {
        float error = fabs(original[i] - dequantized[i]);
        mae += error;
        mse += error * error;
        if (error > max_error) {
            max_error = error;
        }
    }

    mae /= size;
    mse /= size;

    // Print metrics
    printf("Comparison Metrics:\n");
    printf("Mean Absolute Error (MAE): %f\n", mae);
    printf("Mean Squared Error (MSE): %f\n", mse);
    printf("Maximum Absolute Error: %f\n", max_error);

    // Print a few values for visual inspection
    printf("\nSample Comparison (Original vs Dequantized):\n");
    for (int i = 0; i < 10 && i < size; ++i) { // Print up to 10 values
        printf("Original: %f, Dequantized: %f, Difference: %f\n",
               original[i], dequantized[i], fabs(original[i] - dequantized[i]));
    }
}

int8_t clip_to_int8(int32_t value) {
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

float dot_ps_single_thread(float* a, float* b, size_t size) {
    float acc = 0;
    for (int i = 0; i < size; i++) {
        acc += a[i] * b[i];
    }

    return acc;
}

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

    float* result = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), pack_size);
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

// Quantized dot product
int8_t quantized_dot_product(
    const int8_t *A,
    const int8_t *B,
    size_t length,
    struct QCS consts_x,
    struct QCS consts_y,
    struct QCS consts_xy
) {
    int32_t dot = 0;

    // Compute dot product with zero-point offsets
    for (size_t i = 0; i < length; i++) {
        dot += (int32_t)(A[i]) * (int32_t)(B[i]);
    }

    // Rescale the dot product
    float scale_product = consts_xy.s / (consts_x.s * consts_y.s);
    int32_t quantized_result = (int32_t)roundf(scale_product * dot);

    // Clip to int8 range
    return clip_to_int8(quantized_result);
}

int8_t quantized_dot_product_vnni_opt(
    const int8_t *A,
    const int8_t *B,
    size_t length,
    struct QCS consts_x,
    struct QCS consts_y,
    struct QCS consts_xy
) {
    size_t pack_size = 64;
    int full_packs = (length / pack_size) * pack_size;
    size_t aligned_length = length / 64 * 64;
    __m512i dot = _mm512_setzero_epi32(); // Accumulator for dot product

    for (size_t i = 0; i < full_packs; i += pack_size) {
        // Load 64 bytes (8-bit integers) from A and B
        __m512i a = _mm512_load_si512((const __m512i *)&A[i]);
        __m512i b = _mm512_load_si512((const __m512i *)&B[i]);

        // Compute 8-bit dot product and accumulate into 32-bit integers
        dot = _mm512_dpbusd_epi32(dot, a, b);
    }

    // Reduce the accumulated result across lanes
    int32_t dot_product = _mm512_reduce_add_epi32(dot);

    // Handle remaining elements if length is not a multiple of 64
    for (size_t i = aligned_length; i < length; i++) {
        dot_product += (int32_t)A[i] * (int32_t)B[i];
    }

    // Rescale the dot product
    float scale_product = consts_xy.s / (consts_x.s * consts_y.s);
    int32_t quantized_result = (int32_t)roundf(scale_product * dot_product);

    // Clip to int8 range
    return clip_to_int8(quantized_result);
}

// Given float buffer -> generate quantization constants to be used to quantize to 8 bit integers
struct QCS get_quantization_constants_int8(float* data, size_t size) {
    float max = 0;

    for (int i = 0; i < size; i++) {
        if (fabs(data[i]) > max)
            max = data[i];
    }

    float scale = ((float)INT8_MAX / max);

    struct QCS quant_consts = { scale, 0.0f };
    return quant_consts;
}

int8_t* quantize_int8t(float* data, struct QCS qcs, size_t size) {
    int8_t* quantized_data = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(int8_t), size);

    for (int i = 0; i < size; i++) {
        quantized_data[i] = clip_to_int8(round(qcs.s * data[i]));
    }
    return quantized_data;
}

float* dequantize_int8t(int8_t* data, struct QCS qcs, size_t size) {
    float* dequantized_data = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), size);

    for (int i = 0; i < size; i++) {
        dequantized_data[i] = data[i] / qcs.s;
    }
    return dequantized_data;
}

int main() {
    int M = 1536, K = 1536, N = 1536; // A: M x K, B: K x N, C: M x N
    float* A = aligned_alloc(ALIGNMENT_SIZE, M * K * sizeof(float));
    float* B = aligned_alloc(ALIGNMENT_SIZE, K * N * sizeof(float));

    if (A == NULL || B == NULL) {
        perror("Memory allocation failed");
        return 1;
    }

    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Populate matrices A and B with random floating-point values
    for (int i = 0; i < M * K; ++i) {
        A[i] = (float)rand() / RAND_MAX; // Random float between 0 and 1
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = (float)rand() / RAND_MAX; // Random float between 0 and 1
    }

    struct QCS consts = get_quantization_constants_int8(A, M * K);

    int8_t* a_int8 = quantize_int8t(A, consts, M * K);
    float* a_float32 = dequantize_int8t(a_int8, consts, M * K);

    printf("Quantization constants are %f", consts.s);
    compare_quantized_data(A, a_float32, M * K);

    free(A);
    free(B);
    free(a_int8);
    free(a_float32);

    size_t length = 3;
    float* X = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), length);
    float* Y = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), length);

    for (int i = 0; i < length; i++) {
        X[i] = 0.5f;
    }

    for (int i = 0; i < length; i++) {
        Y[i] = 0.5f;
    }

    // Quantization parameters
    struct QCS consts_x = get_quantization_constants_int8(X, length);
    struct QCS consts_y = get_quantization_constants_int8(Y, length);
    struct QCS consts_xy = {10.0f, 0.0f};

    // Compute quantized dot product
    int8_t result = quantized_dot_product_vnni_opt(
        quantize_int8t(X, consts_x, length),
        quantize_int8t(Y, consts_y, length),
        length,
        consts_x,
        consts_y,
        consts_xy
    );

    printf("Quantized dot product: %d\n", result);
    printf("DEQuantized dot product: %f\n", *dequantize_int8t(((int8_t[1]){result}), consts_xy, 1));

    // Speed test
    float *X1 = (float *)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), ARRAY_SIZE);
    float *X2 = (float *)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), ARRAY_SIZE);

    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        X1[i] = ((float)rand() / (RAND_MAX)) / 1000;
        X2[i] = ((float)rand() / (RAND_MAX)) / 1000;
    }

    struct timespec start, end;

    // Benchmark regular dot product
    clock_gettime(CLOCK_MONOTONIC, &start);
    float result_float = dot_ps_single_thread(X1, X2, ARRAY_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_float = get_time_diff(start, end);
    printf("Scalar dot product result: %f, Time: %.6f seconds\n", result_float, time_float);

    // Benchmark float32 avx512 dot product
    clock_gettime(CLOCK_MONOTONIC, &start);
    float result_avx512 = dot_ps_512(X1, X2, ARRAY_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_avx512 = get_time_diff(start, end);
    printf("AVX512 dot product result: %f, Time: %.6f seconds\n", result_avx512, time_avx512);


    // Benchmark VNNI dot product
    struct QCS q_x1 = get_quantization_constants_int8(X1, ARRAY_SIZE);
    struct QCS q_x2 = get_quantization_constants_int8(X2, ARRAY_SIZE);
    struct QCS q_x3 = { (float)INT8_MAX / fabs(result_float), 0.0f };

    int8_t* X1_q = quantize_int8t(X1, q_x1, ARRAY_SIZE);
    int8_t* X2_q = quantize_int8t(X2, q_x2, ARRAY_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &start);
    int8_t result_vnni = quantized_dot_product_vnni_opt(X1_q, X2_q, ARRAY_SIZE, q_x1, q_x2, q_x3);

    clock_gettime(CLOCK_MONOTONIC, &end);
    float dequant_result = *dequantize_int8t(((int8_t[1]){result_vnni}), q_x3, 1);
    double time_vnni = get_time_diff(start, end);
    printf("VNNI dot product result: %f, Time: %.6f seconds\n", dequant_result, time_vnni);

    // Print speedup
    printf("Speedup: %.2fx\n", time_avx512 / time_vnni);

    return 0;
}