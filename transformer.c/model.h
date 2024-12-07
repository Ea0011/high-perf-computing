#include "config.h"
#include <stdlib.h>

typedef struct {
    float* attn_norm_alpha;
    float* attn_norm_betta;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
} AttentionBlock;

typedef struct {
    float* ffn_norm_alpha;
    float* ffn_norm_betta;
    float* w_up;
    float* w_down;
} FeedForwardBlock;

typedef struct {
    AttentionBlock* AttnBlock;
    FeedForwardBlock* FFNBlock;
} TransformerBlock;

typedef struct {
    float* Embedding;
    float* PositionalEncoding;
    TransformerBlock* Blocks;
    float* UnEmbedding;
} Model;

void zero_init_model_from_config(Model* model, ModelConfig cfg) {
    model->Embedding = (float*)calloc(cfg.vocab_size * cfg.d_model, sizeof(float));
    model->PositionalEncoding = (float*)calloc(cfg.max_context_len * cfg.d_model, sizeof(float));
    model->UnEmbedding = (float*)calloc(cfg.d_model * cfg.vocab_size, sizeof(float));
    model->Blocks = (TransformerBlock*)calloc(cfg.num_layers, sizeof(TransformerBlock));

    for (int i = 0; i < cfg.num_layers; i++) {
        model->Blocks[i].AttnBlock = (AttentionBlock*)calloc(1, sizeof(AttentionBlock));
        model->Blocks[i].FFNBlock = (FeedForwardBlock*)calloc(1, sizeof(FeedForwardBlock));

        model->Blocks[i].AttnBlock->attn_norm_alpha = (float*)calloc(cfg.d_model, sizeof(float));
        model->Blocks[i].AttnBlock->attn_norm_betta = (float*)calloc(cfg.d_model, sizeof(float));

        model->Blocks[i].AttnBlock->wq = (float*)calloc(cfg.num_heads * cfg.d_model * cfg.head_dim, sizeof(float));
        model->Blocks[i].AttnBlock->wk = (float*)calloc(cfg.num_heads * cfg.d_model * cfg.head_dim, sizeof(float));
        model->Blocks[i].AttnBlock->wv = (float*)calloc(cfg.num_heads * cfg.d_model * cfg.head_dim, sizeof(float));

        model->Blocks[i].AttnBlock->wo = (float*)calloc(cfg.d_model * cfg.d_model, sizeof(float));
    }
}