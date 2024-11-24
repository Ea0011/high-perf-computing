#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../utils/utils.c"

#define ALIGNMENT_SIZE 8 // Align data using 8 bytes

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
        X[i] = 1.0f;
    }

    for (int i = 0; i < length; i++) {
        Y[i] = 1.0f;
    }

    // Quantization parameters
    struct QCS consts_x = {127.0f, 0.0f};
    struct QCS consts_y = {127.0f, 0.0f};
    struct QCS consts_xy = {10.0f, 0.0f};

    // Compute quantized dot product
    int8_t result = quantized_dot_product(
        quantize_int8t(X, consts_x, length),
        quantize_int8t(Y, consts_y, length),
        length,
        consts_x,
        consts_y,
        consts_xy
    );

    printf("Quantized dot product: %d\n", result);
    printf("DEQuantized dot product: %f\n", *dequantize_int8t(((int8_t[1]){result}), consts_xy, 1));
    return 0;
}