#include "ops.h"
#include "stdio.h"

int main() {
    // testing matmul

    float X[4] = {1, 2, 3, 4};
    float W[4] = {1, 2, 3, 4};
    float out[1] = {0};

    matmul(X, W, out, 1, 4, 1);

    // print out
    for (int i = 0; i < 1; i++) {
        printf("%f\n", out[i]);
    }

    // testing softmax

    softmax(X, 4);

    // print X
    for (int i = 0; i < 4; i++) {
        printf("%f\n", X[i]);
    }

    // check if sum is 1
    float sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += X[i];
    }
    printf("sum: %f\n", sum);
    return 0;
}