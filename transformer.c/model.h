#include "config.h"
#include <stdlib.h>
#include "ops.h"

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

typedef struct {
    float* k_cache; // the key cache for all layers
    float* v_cache; // the value cache for all layers
    float* x; // hold the intermediate state the the current layer
    float* x_norm; // hold the layer norm of the current layer
    float* attn; // holds attn matrix at the current layer
    float* logits; // holds the logits at the current time stamp
    float* probas; // holds the probabilities at the current time stamp
    int position; // holds the current position in the input sequence
    int token_idx; // holds the current token index
} RunState;

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

void forward(
    Model* model,
    RunState* s,
    ModelConfig cfg
) {
    embedding_lookup(model->Embedding, s->token_idx, s->x, cfg.d_model);
    vector_sum(s->x, model->PositionalEncoding + s->position * cfg.d_model, cfg.d_model);
    for (int i = 0; i < cfg.num_layers; i++) {
        layernorm(s->x, model->Blocks[i].AttnBlock->attn_norm_alpha, model->Blocks[i].AttnBlock->attn_norm_betta, cfg.d_model);
        single_head_attention(
            s->x,
            model->Blocks[i].AttnBlock->wq,
            model->Blocks[i].AttnBlock->wk,
            model->Blocks[i].AttnBlock->wv,
            s->attn,
            s->x,
            cfg.d_model,
            cfg.head_dim,
            1
        );
        layernorm(s->x, model->Blocks[i].FFNBlock->ffn_norm_alpha, model->Blocks[i].FFNBlock->ffn_norm_betta, cfg.d_model);
        mlp(
            s->x,
            model->Blocks[i].FFNBlock->w_up,
            model->Blocks[i].FFNBlock->w_down,
            s->x,
            s->x,
            cfg.d_model,
            cfg.hidden_size
        );
    }
    matmul(s->x, model->UnEmbedding, s->logits, 1, cfg.d_model, cfg.vocab_size);
    int next_token = sample_argmax(s->logits, cfg.vocab_size);
    s->token_idx = next_token;
}