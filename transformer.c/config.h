#include <stdio.h>

struct ModelConfig {
    int num_layers;
    int hidden_size;
    int d_model;
    int num_heads;
    int vocab_size;
    int max_context_len;
};

typedef struct ModelConfig ModelConfig;

ModelConfig read_config_from_file(char* fname) {
    ModelConfig config;
    FILE* file = fopen(fname, "r");
    if (file == NULL) {
        printf("Error readin the config file\n");
        return config;
    }
    fscanf(
        file,
        "%d %d %d %d %d %d",
        &config.num_layers,
        &config.hidden_size,
        &config.d_model,
        &config.num_heads,
        &config.vocab_size,
        &config.max_context_len
    );
    fclose(file);
    return config;
};