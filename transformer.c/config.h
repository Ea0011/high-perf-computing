#include <stdio.h>

struct ModelConfig {
    unsigned long long num_layers;
    unsigned long long hidden_size;
    unsigned long long d_model;
    unsigned long long num_heads;
    unsigned long long head_dim;
    unsigned long long vocab_size;
    unsigned long long max_context_len;
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
        "%lld %lld %lld %lld %lld %lld %lld",
        &config.num_layers,
        &config.hidden_size,
        &config.d_model,
        &config.num_heads,
        &config.head_dim,
        &config.vocab_size,
        &config.max_context_len
    );
    fclose(file);
    return config;
};