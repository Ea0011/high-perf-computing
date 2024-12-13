#include "stdio.h"
#include "model.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(ModelConfig cfg, Model* model) {
    RunState* s = initialize_runstate(cfg);

    double tokens_per_second = 0;
    double total_tokens = 0;
    double total_time = 0;

    long time_start = time_in_ms();
    for (;s->position < cfg.max_context_len;) {
        int next_token = forward(model, s, cfg);
        s->token_idx = next_token;
        s->position++;

        // print current token id and position
        printf("Position %d: %d\n", s->position, s->token_idx);
        if (s->token_idx == -1) { // EOS Token ideally
            break;
        }
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
    ModelConfig cfg = read_config_from_file("gpt2_small.bin");
    Model* model = (Model*)malloc(sizeof(Model));

    // radom_init_model_from_config(model, cfg, 0.02);
    mmap_model_from_checkpoint(model, cfg, "./models/c_model.bin");
    generate(cfg, model);
    return 0;
}