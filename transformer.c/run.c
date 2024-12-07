#include "ops.h"
#include "stdio.h"
#include "model.h"
#include <stdio.h>

int main() {
    ModelConfig cfg = read_config_from_file("gpt2_small.bin");
    Model* model = (Model*)malloc(sizeof(Model));

    zero_init_model_from_config(model, cfg);

    // go over all the model and verify the initialization
    printf("%f\n", model->Embedding[0]);
    printf("%f\n", model->Embedding[cfg.vocab_size * cfg.d_model]);

    int layer = 0;
    for (; layer < cfg.num_layers; layer++) {
        printf("%f\n", model->Blocks[layer].AttnBlock->attn_norm_alpha[0]);
        printf("%f\n", model->Blocks[layer].AttnBlock->attn_norm_alpha[cfg.d_model]);

        printf("%f\n", model->Blocks[layer].AttnBlock->wq[0]);
        printf("%f\n", model->Blocks[layer].AttnBlock->wq[cfg.num_heads * cfg.d_model * cfg.head_dim]);
    }
    return 0;
}