#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>

// Helper function to calculate the length of a UTF-8 character
int utf8_char_length(unsigned char c) {
    if ((c & 0x80) == 0x00) return 1; // 1-byte ASCII
    if ((c & 0xE0) == 0xC0) return 2; // 2-byte UTF-8
    if ((c & 0xF0) == 0xE0) return 3; // 3-byte UTF-8
    if ((c & 0xF8) == 0xF0) return 4; // 4-byte UTF-8
    return -1; // Invalid UTF-8
}

// Updated read_vocab function
char** read_vocab(char* path, int vocab_size) {
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        printf("Failed to open file\n");
        exit(1);
    }

    // Determine file size
    fseek(file, 0L, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0L, SEEK_SET); // Reset to beginning of file

    // Allocate memory for vocab array
    char** vocab = (char**)malloc(vocab_size * sizeof(char*));

    // Memory-map the file
    void* ptr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (ptr == MAP_FAILED) {
        printf("Failed to mmap file\n");
        fclose(file);
        exit(1);
    }

    // Read the total vocab size
    int total_size_bytes = *((int*)ptr);
    int bytes_read = 0;
    ptr += sizeof(int);

    while (bytes_read < total_size_bytes) {
        // Read token size and index
        int num_bytes = *((int*)ptr);
        ptr += sizeof(int);
        int token_idx = *((int*)ptr);
        ptr += sizeof(int);

        // Allocate memory for the token
        char* current_token = (char*)calloc(num_bytes + 1, sizeof(char)); // +1 for null terminator

        // Copy UTF-8 bytes into the token string
        int byte_offset = 0;
        for (int i = 0; i < num_bytes; ) {
            int char_len = utf8_char_length(((unsigned char*)ptr)[byte_offset]);
            if (char_len == -1) {
                printf("Invalid UTF-8 character detected\n");
                free(current_token);
                munmap(ptr, file_size);
                fclose(file);
                exit(1);
            }

            // Copy character bytes
            for (int j = 0; j < char_len; ++j) {
                current_token[i++] = ((char*)ptr)[byte_offset++];
            }
        }

        // Add null terminator and store token in vocab
        current_token[num_bytes] = '\0';
        vocab[token_idx] = current_token;

        // Move pointer forward
        ptr += num_bytes;
        bytes_read += num_bytes;
    }

    // Clean up
    fclose(file);
    return vocab;
}


char* decode(char** vocab, int token_idx) {
    return vocab[token_idx];
}
