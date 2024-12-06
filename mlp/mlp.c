#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include "../utils/utils.c"
#include <omp.h>

#define ALIGNMENT_SIZE 8


double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts); // Use CLOCK_MONOTONIC for wall-clock time
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


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

void matmul_parallel_simd(float* A, float* B_t, float* C, float* bias, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        #pragma omp parallel for num_threads(6) schedule(static)
        for (int j = 0; j < N; ++j) {
            float sum = bias[j];

            #pragma omp simd reduction(+: sum)
            for (int k = 0; k < K; k += 1) {
                sum += A[i * K + k] + B_t[j * K + k];
            }
            C[i * N + j] = (sum) > 0 ? (sum) : 0;
        }
    }
}

void matmul_avx_optimized(float* A, float* B_t, float* C, int M, int K, int N) {
    float* temp = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), 8);
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
    float* temp = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), 8);
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

void fused_matmul_bias_relu_avx(float* A, float* B_t, float* C, float* bias, int M, int K, int N) {
    float* temp = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), 8);
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
                        temp[4] + temp[5] + temp[6] + temp[7] + bias[j];
            C[i * N + j] = c_sum > 0 ? c_sum : 0;
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

void mlp(
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
    int hidden_size
) {
    float* x = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), batch_size * hidden_size);
    float* temp = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), batch_size * hidden_size);

    matmul_single_thread(input, input_layer_weights, x, batch_size, input_dim, hidden_size);
    for (int l = 0; l < num_layers; ++l) {
        int layer_offset = l * hidden_size * hidden_size;
        int bias_offset = l * hidden_size;

        matmul_single_thread(x, &weights[layer_offset], temp, batch_size, hidden_size, hidden_size);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                temp[i * hidden_size + j] += biases[bias_offset + j];
            }
        }
        relu(temp, batch_size * hidden_size);
        float* swap = x;
        x = temp;
        temp = swap;
    }

    matmul_single_thread(x, output_layer_weights, output, batch_size, hidden_size, output_dim);
    free(temp);
}

void mlp_multithread_simd(
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
    int hidden_size
) {
    float* x = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), batch_size * hidden_size);
    float* temp = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), batch_size * hidden_size);

    matmul_single_thread(input, input_layer_weights, x, batch_size, input_dim, hidden_size);
    for (int l = 0; l < num_layers; ++l) {
        int layer_offset = l * hidden_size * hidden_size;
        int bias_offset = l * hidden_size;

        matmul_parallel_simd(x, &weights[layer_offset], temp, &biases[bias_offset], batch_size, hidden_size, hidden_size);
        float* swap = x;
        x = temp;
        temp = swap;
    }

    matmul_single_thread(x, output_layer_weights, output, batch_size, hidden_size, output_dim);
    free(temp);
}

void mlp_avx(
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
    int hidden_size
) {
    float* x = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), batch_size * hidden_size);
    float* temp = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), batch_size * hidden_size);

    matmul_avx_optimized_loop_unrolling(input, input_layer_weights, x, batch_size, input_dim, hidden_size);
    for (int l = 0; l < num_layers; ++l) {
        int layer_offset = l * hidden_size * hidden_size;
        int bias_offset = l * hidden_size;

        fused_matmul_bias_relu_avx(x, &weights[layer_offset], temp, &biases[bias_offset], batch_size, hidden_size, hidden_size);
        float* swap = x;
        x = temp;
        temp = swap;
    }
    matmul_avx_optimized_loop_unrolling(x, output_layer_weights, output, batch_size, hidden_size, output_dim);
    free(temp);
}

// Measure execution time
double measure_execution_time(
    void (*mlp_func)(
        float*, float*, float*, float*, float*, float*, 
        int, int, int, int, int
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
    int hidden_size
) {
    double start = get_time();
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
        hidden_size
    );
    double end = get_time();
    return end - start;
}


int main() {
    // Define layer sizes
    int B = 1;
    int input_dim = 768;
    int output_dim = 32000;
    int hidden_size = 2048;
    int num_layers = 32;

    // Allocate memory
    float* input = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), B * input_dim);

    // Network parameters
    float* embed_layer = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), input_dim * hidden_size);
    float* weights = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), num_layers * hidden_size * hidden_size);
    float* biases = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), num_layers * hidden_size);
    float* output_layer = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), hidden_size * output_dim);

    float* output = (float*)aligned_allocate_buffer(ALIGNMENT_SIZE, sizeof(float), B * output_dim);

    initialize_matrix(input, B, input_dim);
    for (int l = 0; l < num_layers; l++) {
        int layer_offset = l * hidden_size * hidden_size;
        int bias_offset = l * hidden_size;

        initialize_matrix(&weights[layer_offset], hidden_size, hidden_size);
        initialize_matrix(&biases[bias_offset], 1, hidden_size);
    }
    initialize_matrix(output_layer, B, output_dim);

    double avx_time = measure_execution_time(
        mlp_avx,
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
        hidden_size
    );

    double single_thread_time = measure_execution_time(
        mlp,
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
        hidden_size
    );

    double parallel_simd_time = measure_execution_time(
        mlp_multithread_simd,
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
        hidden_size
    );

    printf("AVX-optimized MLP execution time: %f seconds\n", avx_time);
    printf("Single-threaded MLP execution time: %f seconds\n", single_thread_time);
    printf("Multi-threaded MLP execution time: %f seconds\n", parallel_simd_time);

    free(input);
    free(weights);
    free(biases);
    free(output);

    return 0;
}
