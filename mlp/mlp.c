#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include "../utils/utils.c"

#define ALIGNMENT_SIZE 32

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
    float* temp = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), 8);
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
    float* temp = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), 8);
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

// ReLU activation function
void relu(float* input, int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = input[i] > 0 ? input[i] : 0;
    }
}

// Initialize matrix with random values
void initialize_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// MLP implementation
void mlp_forward(
    float* input,
    float* input_layer_weights,
    float* weights,
    float* biases,
    float* output_layer_weights,
    float* output,
    int batch_size,
    int num_layers,
    int input_dim,
    int output_dim,
    int hidden_size,
    int use_avx
) {
    float* x = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), batch_size * hidden_size);
    float* temp = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), batch_size * hidden_size);

    // Input layer transformation
    if (use_avx) {
        matmul_avx_optimized_loop_unrolling(input, input_layer_weights, x, batch_size, input_dim, hidden_size);
    } else {
        matmul_single_thread(input, input_layer_weights, x, batch_size, input_dim, hidden_size);
    }

    relu(x, batch_size * hidden_size);

    // Hidden layers
    for (int l = 0; l < num_layers; ++l) {
        int layer_offset = l * hidden_size * hidden_size;
        int bias_offset = l * hidden_size;

        if (use_avx) {
            matmul_avx_optimized_loop_unrolling(x, &weights[layer_offset], temp, batch_size, hidden_size, hidden_size);
        } else {
            matmul_single_thread(x, &weights[layer_offset], temp, batch_size, hidden_size, hidden_size);
        }

        // Add biases and apply activation function
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                temp[i * hidden_size + j] += biases[bias_offset + j];
            }
        }
        relu(temp, batch_size * hidden_size);

        // Swap x and temp pointers for next layer
        float* swap = x;
        x = temp;
        temp = swap;
    }

    // Output layer transformation
    if (use_avx) {
        matmul_avx_optimized_loop_unrolling(x, output_layer_weights, output, batch_size, hidden_size, output_dim);
    } else {
        matmul_single_thread(x, output_layer_weights, output, batch_size, hidden_size, output_dim);
    }

    // Free temporary buffer
    free(temp);
}

// Measure execution time
double measure_execution_time(
    void (*mlp_func)(
        float*, float*, float*, float*, float*, float*, 
        int, int, int, int, int, int
    ), 
    float* input,
    float* embed_layer,
    float* weights,
    float* biases,
    float* output_layer,
    float* output,
    int batch_size,
    int num_layers,
    int input_dim,
    int output_dim,
    int hidden_size,
    int use_avx
) {
    clock_t start = clock();
    mlp_func(
        input,
        embed_layer,
        weights,
        biases,
        output_layer,
        output,
        batch_size,
        num_layers,
        input_dim,
        output_dim,
        hidden_size,
        use_avx
    );
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}


int main() {
    // Define layer sizes
    int B = 128;
    int input_dim = 256;
    int output_dim = 128;
    int hidden_size = 2048;
    int num_layers = 32;

    // Allocate memory
    float* input = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), B * input_dim);

    // Network parameters
    float* embed_layer = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), input_dim * hidden_size);
    float* weights = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), num_layers * hidden_size * hidden_size);
    float* biases = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), num_layers * hidden_size);
    float* output_layer = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), hidden_size * output_dim);

    float* output = aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), B * output_dim);

    initialize_matrix(input, B, input_dim);
    for (int l = 0; l < num_layers; l++) {
        int layer_offset = l * hidden_size * hidden_size;
        int bias_offset = l * hidden_size;

        initialize_matrix(&weights[layer_offset], hidden_size, hidden_size);
        initialize_matrix(&biases[bias_offset], 1, hidden_size);
    }
    initialize_matrix(output_layer, B, output_dim);

    double avx_time = measure_execution_time(
        mlp_forward,
        input,
        embed_layer,
        weights,
        biases,
        output_layer,
        output,
        B,
        num_layers,
        input_dim,
        output_dim,
        hidden_size,
        1
    );

    double single_thread_time = measure_execution_time(
        mlp_forward,
        input,
        embed_layer,
        weights,
        biases,
        output_layer,
        output,
        B,
        num_layers,
        input_dim,
        output_dim,
        hidden_size,
        0
    );

    printf("AVX-optimized MLP execution time: %f seconds\n", avx_time);
    printf("Single-threaded MLP execution time: %f seconds\n", single_thread_time);

    free(input);
    free(weights);
    free(biases);
    free(output);

    return 0;
}
