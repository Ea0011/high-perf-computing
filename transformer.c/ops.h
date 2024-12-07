/*
 Tensor Operations in Plain C commonly used in transformer architecture
*/

#include <stdlib.h>
// #include <omp.h>
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include "utils.h"


void embedding_lookup(float* E, int index, float* out, int d) {
    /*
        Lookup the embedding for the given index.
        E is a matrix of shape (n, d), where n is the number of embeddings.
        The embedding for the index-th element is stored in out.
    */
    memcpy(out, E + index * d, d * sizeof(float));
}

void vector_sum(float* X1, float* X2, int d) {
    /*
        Sum of two vectors X1 and X2, storing the result in X1.
    */
    for (int i = 0; i < d; i++) {
        X1[i] += X2[i];
    }
}

void matmul(float* X, float* W, float* out, int n, int d, int h) {
    /*
        Matrix multiplication of X and W, storing the result in out.
        X has shape (n, d), W has shape (d, h), and out has shape (n, h).
    */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < h; j++) {
            float sum = 0;
            for (int k = 0; k < d; k++) {
                sum += X[i * d + k] * W[k * h + j];
            }
            out[i * h + j] = sum;
        }
    }
}

void gelu(float* X, int n) {
    /*
        GELU activation function applied to X.
        GELU is approximated via xσ(1.702x)
    */
    for (int i = 0; i < n; i++) {
        X[i] = 0.5 * X[i] * (1 + tanhf(0.79788456 * (X[i] + 0.044715 * X[i] * X[i] * X[i])));
    }
}

void softmax(float* X, int n) {
    /*
        Softmax function applied to X.
    */
    float max = X[0];
    for (int i = 1; i < n; i++) {
        if (X[i] > max) {
            max = X[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < n; i++) {
        X[i] = expf(X[i] - max);
        sum += X[i];
    }
    for (int i = 0; i < n; i++) {
        X[i] /= sum;
    }
}

void layernorm(float* X, float alpha, float betta, float eps, int n) {
    /*
        Layer normalization applied to X.
    */
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += X[i];
    }
    float mean = sum / n;
    sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (X[i] - mean) * (X[i] - mean);
    }
    float std = sqrtf(sum / n + eps);
    for (int i = 0; i < n; i++) {
        X[i] = alpha * (X[i] - mean) / std + betta;
    }
}

int multinomial_sample(float* X, int n) {
    /*
        Sample from the multinomial distribution defined by X.
        X is a probability distribution over n categories.
    */
    float r = (float)rand() / RAND_MAX;

    for (int i = 0; i < n; i++) {
        r -= X[i];
        if (r <= 0) {
            return i;
        }
    }

    return n - 1;
}

int sample_argmax(float* X, int n) {
    /*
        Sample the index of the maximum value in X.
    */
    int max_idx = 0;
    float max_val = X[0];
    for (int i = 1; i < n; i++) {
        if (X[i] > max_val) {
            max_val = X[i];
            max_idx = i;
        }
    }
    return max_idx;
}


// -------- Transformer Operations --------
void single_head_attention(
    float* Q, // Matrix of shape (n, dim_head)
    float* K, // Matrix of shape (n, dim_head)
    float* V, // Matrix of shape (n, dim_head)
    float* attn_matrix, // Output of attention, matrix of shape (n, n)
    float* out, // Output of head attention, matrix of shape (n, dim_head)
    int n,     // Sequence length
    int dim_head // Dimension of the head
) {
    matmul(Q, K, attn_matrix, n, dim_head, n);

    float scale = 1.0f / sqrtf(dim_head);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            attn_matrix[i * n + j] *= scale;
        }
    }
    for (int i = 0; i < n; i++) {
        softmax(attn_matrix + i * n, n);
    }

    matmul(attn_matrix, V, out, n, n, dim_head);
}


void mlp(
    float* X,
    float* W_up,
    float* W_down,
    float* out_up,
    float* out_down,
    int n,
    int d_model,
    int hidden_size
) {
    /*
        MLP layer applied to X.
    */
    matmul(X, W_up, out_up, n, d_model, hidden_size);
    gelu(out_up, n * hidden_size);
    matmul(out_up, W_down, out_down, n, hidden_size, d_model);
}