#include "ops.h"


int test_attn() {
    int n = 2; // Sequence length
    int dim_head = 3; // Dimension of head

    // Input matrices Q, K, and V (2x3 for this test case)
    float Q[] = {1.0, 0.0, 1.0, 
                 0.0, 1.0, 1.0};
    float K[] = {1.0, 0.0, 
                 0.0, 1.0,
                 1.0, 1.0};
    float V[] = {1.0, 2.0, 3.0, 
                 4.0, 5.0, 6.0};

    float expected_attn_matrix[] = {0.6405, 0.359, 
                                    0.359, 0.6405};

    float expected_out[] = {2.07862757, 3.07862757, 4.07862757, 
                          2.92137243, 3.92137243, 4.92137243};

    // Allocate memory for the attention matrix and output
    float attn_matrix[n * n];
    float out[n * dim_head];

    // Call the single head attention function
    single_head_attention(Q, K, V, attn_matrix, out, n, dim_head, 0);

    // Test cases to check
    for (int i = 0; i < n * n; i++) {
        if (fabs(attn_matrix[i] - expected_attn_matrix[i]) > 1e-3) {
            printf("Test failed for attn_matrix[%d]: expected %f, got %f\n", i, expected_attn_matrix[i], attn_matrix[i]);
            return 1;
        }
    }

    for (int i = 0; i < n * dim_head; i++) {
        if (fabs(out[i] - expected_out[i]) > 1e-3) {
            printf("Test failed for out[%d]: expected %f, got %f\n", i, expected_out[i], out[i]);
            return 1;
        }
    }

    return 0;
}

int test_softmax() {
    // Test case for softmax
    float X[4] = {1, 2, 3, 4};
    softmax(X, 4, 1.0);

    float expected[4] = {0.0321, 0.0871, 0.2369, 0.6439};

    for (int i = 0; i < 4; i++) {
        if (fabs(X[i] - expected[i]) > 1e-3) {
            printf("Test failed for softmax[%d]: expected %f, got %f\n", i, expected[i], X[i]);
            return 1;
        }
    }

    return 0;
}

int test_matmul() {
    // Test case for matmul
    float X[4] = {1, 2, 3, 4};
    float W[4] = {1, 2, 3, 4};
    float out[1] = {0};

    matmul(X, W, out, 1, 4, 1);

    if (fabs(out[0] - 30) > 1e-3) {
        printf("Test failed for matmul: expected 30, got %f\n", out[0]);
        return 1;
    }

    return 0;
}

int test_vector_sum() {
    // Test case for vector_sum
    float X[4] = {1, 2, 3, 4};
    float expected[] = {2, 4, 6, 8};
    vector_sum(X, X, 4);

    for (int i = 0; i < 4; i++) {
        if (fabs(X[i] - expected[i]) > 1e-5) {
            printf("Test failed for vector_sum[%d]: expected %f, got %f\n", i, 2 * (i + 1), X[i]);
            return 1;
        }
    }

    return 0;
}

int test_gelu() {
    // Test case for gelu
    float X[4] = {1, 2, 3, 4};
    gelu(X, 4);

    float expected[4] = {0.8413, 1.9545, 2.9960, 3.9999};

    for (int i = 0; i < 4; i++) {
        if (fabs(X[i] - expected[i]) > 1e-3) {
            printf("Test failed for gelu[%d]: expected %f, got %f\n", i, expected[i], X[i]);
            return 1;
        }
    }

    return 0;
}

int test_embedding_lookup() {
    // Test case for embedding_lookup
    float E[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    float out[1] = {0};

    embedding_lookup(E, 4, out, 1);

    float expected[1] = {5};

    if (fabs(out[0] - expected[0]) > 1e-3) {
        printf("Test failed for embedding_lookup: expected 5, got %f\n", out[0]);
        return 1;
    }

    return 0;
}

int main() {
    char* test_names[] = {
        "test_attn",
        "test_matmul",
        "test_softmax",
        "test_vector_sum",
        "test_gelu",
        "test_embedding_lookup"
    };
    int (*tests[])() = {
        test_attn,
        test_matmul,
        test_softmax,
        test_vector_sum,
        test_gelu,
        test_embedding_lookup
    };

    for (int i = 0; i < (sizeof(tests)) / (sizeof(void (*))); i++) {
        if (tests[i]()) {
            printf("%s failed\n", test_names[i]);
        } else {
            printf("%s passed\n", test_names[i]);
        }
    }

    return 0;
}
