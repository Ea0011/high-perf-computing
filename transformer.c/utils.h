#include <stdio.h>

// Function to print a matrix
void print_matrix(const char* name, float* matrix, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%0.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}