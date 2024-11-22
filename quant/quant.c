#include <math.h>
#include <float.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ALIGNMENT_SIZE 8

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


struct QuantizationConstants {
    float s;
    float z;
};

// Given float buffer -> generate quantization constants to be used to quantize to 8 bit integers
struct QuantizationConstants get_quantization_constants_int8(float* data, size_t size) {
    float max = 0;

    for (int i = 0; i < size; i++) {
        if (fabs(data[i]) > max)
            max = data[i];
    }

    float scale = ((float)INT8_MAX / max);

    struct QuantizationConstants quant_consts = { scale, 0.0f };
    return quant_consts;
}

int8_t* quantize_int8t(float* data, float scale, size_t size) {
    int8_t* quantized_data = aligned_alloc(ALIGNMENT_SIZE, size * sizeof(int8_t));

    for (int i = 0; i < size; i++) {
        quantized_data[i] = round(scale * data[i]);
    }
    return quantized_data;
}

float* dequantize_int8t(int8_t* data, float scale, size_t size) {
    float* dequantized_data = aligned_alloc(ALIGNMENT_SIZE, size * sizeof(float));

    for (int i = 0; i < size; i++) {
        dequantized_data[i] = data[i] / scale;
    }
    return dequantized_data;
}

int main() {
    int M = 1536, K = 1536, N = 1536; // A: M x K, B: K x N, C: M x N
    float* A = aligned_alloc(ALIGNMENT_SIZE, M * K * sizeof(float));
    float* B = aligned_alloc(ALIGNMENT_SIZE, K * N * sizeof(float));

    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Populate matrices A and B with random floating-point values
    for (int i = 0; i < M * K; ++i) {
        A[i] = (float)rand() / RAND_MAX; // Random float between 0 and 1
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = (float)rand() / RAND_MAX; // Random float between 0 and 1
    }

    struct QuantizationConstants consts = get_quantization_constants_int8(A, M * K);

    int8_t* a_int8 = quantize_int8t(A, consts.s, M * K);
    float* a_float23 = dequantize_int8t(a_int8, consts.s, M * K);

    printf("Quantization constants are %f", consts.s);
    compare_quantized_data(A, a_float23, M * K);

    return 0;
}