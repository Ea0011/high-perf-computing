#include "stdio.h"
#include "model.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "tokenizer.h"
#include <stdlib.h>

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void measure_tok_s(ModelConfig cfg, Model* model, int num_tokens, int num_runs) {
    RunState* s = initialize_runstate(cfg);
    char** vocab = read_vocab("vocab.bin", cfg.vocab_size);
    double tok_s = 0;

    for (int i = 0; i < num_runs; i++) {
        long time_start = time_in_ms();
        for (int j = 0; j < num_tokens; j++) {
            int next_token = forward(model, s, cfg);
            s->token_idx = next_token;
            s->position++;
        }
        long time_end = time_in_ms();
        double total_time = time_end - time_start;
        double tokens_per_second = num_tokens / (total_time / 1000);
        tok_s += tokens_per_second;
        printf("Tokens per second: %f\n", tokens_per_second);
    }

    printf("Average tokens per second: %f\n", tok_s / num_runs);
}

void generate(ModelConfig cfg, Model* model) {
    RunState* s = initialize_runstate(cfg);
    char** vocab = read_vocab("vocab.bin", cfg.vocab_size);

    double tokens_per_second = 0;
    double total_tokens = 0;
    double total_time = 0;

    long time_start = time_in_ms();
    for (;s->position < cfg.max_context_len;) {
        int next_token = forward(model, s, cfg);
        if (next_token == 50256) {
            break;
        }

        char* decoded_token = decode(vocab, s->token_idx, next_token);
        printf("%s", decoded_token);
        fflush(stdout); // Ensure the output is flushed immediately

        s->token_idx = next_token;
        s->position++;
    }
    long time_end = time_in_ms();
    total_time = time_end - time_start;
    total_tokens = s->position;
    tokens_per_second = total_tokens / (total_time / 1000);
    printf("Total tokens: %f\n", total_tokens);
    printf("Total time: %f\n", total_time);
    printf("Tokens per second: %f\n", tokens_per_second);
}

int main() {
    ModelConfig cfg = read_config_from_file("gpt2_xl.bin");
    Model* model = (Model*)malloc(sizeof(Model));

    mmap_model_from_checkpoint(model, cfg, "./models/c_model_xxl.bin");
    // generate(cfg, model);
    measure_tok_s(cfg, model, 100, 10);
    return 0;
}