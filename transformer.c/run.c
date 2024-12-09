#include "stdio.h"
#include "model.h"
#include <stdio.h>

void generate(ModelConfig cfg, Model* model) {
    RunState* s = initialize_runstate(cfg);
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
}

int main() {
    ModelConfig cfg = read_config_from_file("gpt2_small.bin");
    Model* model = (Model*)malloc(sizeof(Model));

    radom_init_model_from_config(model, cfg);
    generate(cfg, model);
    return 0;
}