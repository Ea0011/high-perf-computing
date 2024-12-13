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

void fused_matmul_bias_transpose(float* X, float* W, float* b, float* out, int n, int d, int h) {
    /*
        Matrix multiplication of X and W^T, storing the result in out.
        X has shape (n, d), W has shape (h, d), and out has shape (n, h).
    */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < h; j++) {
            float sum = b[j];
            for (int k = 0; k < d; k++) {
                sum += X[i * d + k] * W[j * d + k];
            }
            out[i * h + j] = sum;
        }
    }
}

void matmul_transpose(float* X, float* W, float* out, int n, int d, int h) {
    /*
        Matrix multiplication of X and W^T, storing the result in out.
        X has shape (n, d), W has shape (h, d), and out has shape (n, h).
    */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < h; j++) {
            float sum = 0;
            for (int k = 0; k < d; k++) {
                sum += X[i * d + k] * W[j * d + k];
            }
            out[i * h + j] = sum;
        }
    }
}

void fused_matmul_bias(float* X, float* W, float* b, float* out, int n, int d, int h) {
    /*
        Matrix multiplication of X and W, storing the result in out.
        Fuses bias addition in one function
        X has shape (n, d), W has shape (d, h), and out has shape (n, h).
    */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < h; j++) {
            float sum = b[j];
            for (int k = 0; k < d; k++) {
                sum += X[i * d + k] * W[k * h + j];
            }
            out[i * h + j] = sum;
        }
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

void relu(float* X, int n) {
    /*
        ReLU activation function applied to X.
    */
    for (int i = 0; i < n; i++) {
        X[i] = X[i] > 0 ? X[i] : 0;
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

void layernorm(float* X, float* alpha, float* betta, float* out, int n) {
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
    float std = sqrtf((sum / n) + 1e-5);
    for (int i = 0; i < n; i++) {
        out[i] = alpha[i] * (X[i] - mean) / (std) + betta[i];
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
    float* Q, // Matrix of shape (1, dim_head)
    float* K, // Matrix of shape (n, dim_head)
    float* V, // Matrix of shape (n, dim_head)
    float* attn_matrix, // Output of attention, matrix of shape (1, n)
    float* out, // Output of head attention, matrix of shape (n, dim_head)
    int pos,     // Sequence length
    int dim_head, // Dimension of the head
    int causal
) {
    matmul_transpose(Q, K, attn_matrix, 1, dim_head, pos);

    // for this to work the cache needs to be stored as to easily retrieve (pos, dim_head) values from the cache
    // Example struct to store kv cache
   
    // array[layer][head][pos][dimension]

    float scale = 1.0f / sqrtf(dim_head);
    for (int i = 0; i < pos; i++) {
        attn_matrix[i] *= scale;
    }
   
    softmax(attn_matrix, pos);

    // linearly combine V with attn_matrix into out
    for (int i = 0; i < dim_head; i++) {
        float sum = 0;
        for (int j = 0; j < pos; j++) {
            sum += attn_matrix[j] * V[j * dim_head + i];
        }
        out[i] = sum;
    }
}


void mlp(
    float* X,
    float* W_up,
    float* b_up,
    float* W_down,
    float* b_down,
    float* out_up,
    float* out_down,
    int n,
    int d_model,
    int hidden_size
) {
    /*
        MLP layer applied to X.
    */
    fused_matmul_bias_transpose(X, W_up, b_up, out_up, n, d_model, hidden_size);
    gelu(out_up, hidden_size);
    fused_matmul_bias_transpose(out_up, W_down, b_down, out_down, n, hidden_size, d_model);
}