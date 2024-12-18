#include "config.h"
#include <stdlib.h>
#include "ops.h"
#include <sys/mman.h>

typedef struct {
    float* attn_norm_alpha;
    float* attn_norm_betta;
    float* wq;
    float* wq_bias;
    float* wk;
    float* wk_bias;
    float* wv;
    float* wv_bias;
    float* wo;
    float* wo_bias;
} AttentionBlock;

typedef struct {
    float* ffn_norm_alpha;
    float* ffn_norm_betta;
    float* w_up;
    float* w_up_bias;
    float* w_down;
    float* w_down_bias;
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
    float* UnEmbeddingLNAlpha;
    float* UnEmbeddingLNBetta;
} Model;

typedef struct {
    float* k_cache; // the key cache for all layers
    float* v_cache; // the value cache for all layers
    float* x; // hold the intermediate state the the current layer
    float* x_norm; // hold the layer norm of the current layer
    float* x_ffn_down; // hold the intermediate state of the FFN layer
    float* x_ffn_up; // hold the output of the FFN layer
    float* q; // holds the query matrix at the current layer
    float* k; // holds the key matrix at the current layer
    float* v; // holds the value matrix at the current layer
    float* attn_out; // holds the attention output at the current layer
    float* attn_weights; // holds attn matrix at the current layer
    float* logits; // holds the logits at the current time stamp
    float* probas; // holds the probabilities at the current time stamp
    int position; // holds the current position in the input sequence
    int token_idx; // holds the current token index
} RunState;

void zero_init_model_from_config(Model* model, ModelConfig cfg) {
    model->Embedding = (float*)calloc(cfg.vocab_size * cfg.d_model, sizeof(float));
    model->PositionalEncoding = (float*)calloc(cfg.max_context_len * cfg.d_model, sizeof(float));
    model->UnEmbedding = (float*)calloc(cfg.d_model * cfg.vocab_size, sizeof(float));
    model->UnEmbeddingLNAlpha = (float*)calloc(cfg.d_model, sizeof(float));
    model->UnEmbeddingLNBetta = (float*)calloc(cfg.d_model, sizeof(float));
    model->Blocks = (TransformerBlock*)calloc(cfg.num_layers, sizeof(TransformerBlock));

    for (int i = 0; i < cfg.num_layers; i++) {
        model->Blocks[i].AttnBlock = (AttentionBlock*)calloc(1, sizeof(AttentionBlock));
        model->Blocks[i].FFNBlock = (FeedForwardBlock*)calloc(1, sizeof(FeedForwardBlock));

        model->Blocks[i].AttnBlock->attn_norm_betta = (float*)calloc(cfg.d_model, sizeof(float));
        model->Blocks[i].AttnBlock->attn_norm_alpha = (float*)calloc(cfg.d_model, sizeof(float));
        model->Blocks[i].FFNBlock->ffn_norm_alpha = (float*)calloc(cfg.d_model, sizeof(float));
        model->Blocks[i].FFNBlock->ffn_norm_betta = (float*)calloc(cfg.d_model, sizeof(float));

        model->Blocks[i].AttnBlock->wq = (float*)calloc(cfg.num_heads * cfg.d_model * cfg.head_dim, sizeof(float));
        model->Blocks[i].AttnBlock->wq_bias = (float*)calloc(cfg.num_heads * cfg.head_dim, sizeof(float));

        model->Blocks[i].AttnBlock->wk = (float*)calloc(cfg.num_heads * cfg.d_model * cfg.head_dim, sizeof(float));
        model->Blocks[i].AttnBlock->wk_bias = (float*)calloc(cfg.num_heads * cfg.head_dim, sizeof(float));

        model->Blocks[i].AttnBlock->wv = (float*)calloc(cfg.num_heads * cfg.d_model * cfg.head_dim, sizeof(float));
        model->Blocks[i].AttnBlock->wv_bias = (float*)calloc(cfg.num_heads * cfg.head_dim, sizeof(float));

        model->Blocks[i].AttnBlock->wo = (float*)calloc(cfg.d_model * cfg.d_model, sizeof(float));
        model->Blocks[i].AttnBlock->wo_bias = (float*)calloc(cfg.d_model, sizeof(float));

        model->Blocks[i].FFNBlock->w_up = (float*)calloc(cfg.d_model * cfg.hidden_size, sizeof(float));
        model->Blocks[i].FFNBlock->w_up_bias = (float*)calloc(cfg.hidden_size, sizeof(float));

        model->Blocks[i].FFNBlock->w_down = (float*)calloc(cfg.hidden_size * cfg.d_model, sizeof(float));
        model->Blocks[i].FFNBlock->w_down_bias = (float*)calloc(cfg.d_model, sizeof(float));
    }
}

void radom_init_model_from_config(Model* model, ModelConfig cfg, float scale) {
    model->Embedding = (float*)calloc(cfg.vocab_size * cfg.d_model, sizeof(float));
    for (int i = 0; i < cfg.vocab_size * cfg.d_model; i++) {
        model->Embedding[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }

    model->PositionalEncoding = (float*)calloc(cfg.max_context_len * cfg.d_model, sizeof(float));
    for (int i = 0; i < cfg.max_context_len * cfg.d_model; i++) {
        model->PositionalEncoding[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }

    model->UnEmbedding = (float*)calloc(cfg.d_model * cfg.vocab_size, sizeof(float));
    model->UnEmbeddingLNAlpha = (float*)calloc(cfg.d_model, sizeof(float));
    model->UnEmbeddingLNBetta = (float*)calloc(cfg.d_model, sizeof(float));
    for (int i = 0; i < cfg.d_model * cfg.vocab_size; i++) {
        model->UnEmbedding[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }

    model->Blocks = (TransformerBlock*)calloc(cfg.num_layers, sizeof(TransformerBlock));
    for (int i = 0; i < cfg.num_layers; i++) {
        model->Blocks[i].AttnBlock = (AttentionBlock*)calloc(1, sizeof(AttentionBlock));
        model->Blocks[i].FFNBlock = (FeedForwardBlock*)calloc(1, sizeof(FeedForwardBlock));

        model->Blocks[i].AttnBlock->attn_norm_betta = (float*)calloc(cfg.d_model, sizeof(float));
        model->Blocks[i].AttnBlock->attn_norm_alpha = (float*)calloc(cfg.d_model, sizeof(float));
        model->Blocks[i].FFNBlock->ffn_norm_alpha = (float*)calloc(cfg.d_model, sizeof(float));
        model->Blocks[i].FFNBlock->ffn_norm_betta = (float*)calloc(cfg.d_model, sizeof(float));

        for (int j = 0; j < cfg.d_model; j++) {
            model->Blocks[i].AttnBlock->attn_norm_alpha[j] = 1.0;
            model->Blocks[i].AttnBlock->attn_norm_betta[j] = 0.0;
            model->Blocks[i].FFNBlock->ffn_norm_alpha[j] = 1.0;
            model->Blocks[i].FFNBlock->ffn_norm_betta[j] = 0.0;
        }

        model->Blocks[i].AttnBlock->wq = (float*)calloc(cfg.num_heads * cfg.d_model * cfg.head_dim, sizeof(float));
        model->Blocks[i].AttnBlock->wq_bias = (float*)calloc(cfg.num_heads * cfg.head_dim, sizeof(float));

        model->Blocks[i].AttnBlock->wk = (float*)calloc(cfg.num_heads * cfg.d_model * cfg.head_dim, sizeof(float));
        model->Blocks[i].AttnBlock->wk_bias = (float*)calloc(cfg.num_heads * cfg.head_dim, sizeof(float));

        model->Blocks[i].AttnBlock->wv = (float*)calloc(cfg.num_heads * cfg.d_model * cfg.head_dim, sizeof(float));
        model->Blocks[i].AttnBlock->wv_bias = (float*)calloc(cfg.num_heads * cfg.head_dim, sizeof(float));

        model->Blocks[i].AttnBlock->wo = (float*)calloc(cfg.d_model * cfg.d_model, sizeof(float));
        model->Blocks[i].AttnBlock->wo_bias = (float*)calloc(cfg.d_model, sizeof(float));

        for (int j = 0; j < cfg.num_heads * cfg.d_model * cfg.head_dim; j++) {
            model->Blocks[i].AttnBlock->wq[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
            model->Blocks[i].AttnBlock->wk[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
            model->Blocks[i].AttnBlock->wv[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }

        for (int j = 0; j < cfg.d_model * cfg.d_model; j++) {
            model->Blocks[i].AttnBlock->wo[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }

        model->Blocks[i].FFNBlock->w_up = (float*)calloc(cfg.d_model * cfg.hidden_size, sizeof(float));
        model->Blocks[i].FFNBlock->w_up_bias = (float*)calloc(cfg.hidden_size, sizeof(float));

        model->Blocks[i].FFNBlock->w_down = (float*)calloc(cfg.hidden_size * cfg.d_model, sizeof(float));
        model->Blocks[i].FFNBlock->w_down_bias = (float*)calloc(cfg.d_model, sizeof(float));

        for (int j = 0; j < cfg.d_model * cfg.hidden_size; j++) {
            model->Blocks[i].FFNBlock->w_up[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }

        for (int j = 0; j < cfg.hidden_size * cfg.d_model; j++) {
            model->Blocks[i].FFNBlock->w_down[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
    }

    // check if allocations went smoothly
    if (model->Embedding == NULL || model->PositionalEncoding == NULL || model->UnEmbedding == NULL || model->Blocks == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    };

    for (int i = 0; i < cfg.num_layers; i++) {
        if (model->Blocks[i].AttnBlock == NULL || model->Blocks[i].FFNBlock == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
    }

    for (int i = 0; i < cfg.num_layers; i++) {
        if (model->Blocks[i].AttnBlock->attn_norm_alpha == NULL || model->Blocks[i].AttnBlock->attn_norm_betta == NULL || model->Blocks[i].FFNBlock->ffn_norm_alpha == NULL || model->Blocks[i].FFNBlock->ffn_norm_betta == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
    }

    for (int i = 0; i < cfg.num_layers; i++) {
        if (model->Blocks[i].AttnBlock->wq == NULL || model->Blocks[i].AttnBlock->wk == NULL || model->Blocks[i].AttnBlock->wv == NULL || model->Blocks[i].AttnBlock->wo == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
    }

    for (int i = 0; i < cfg.num_layers; i++) {
        if (model->Blocks[i].FFNBlock->w_up == NULL || model->Blocks[i].FFNBlock->w_down == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
    }
}

void mmap_model_from_checkpoint(Model* model, ModelConfig cfg, const char* path) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        printf("Failed to open file\n");
        exit(1);
    }

    fseek(file, 0L, SEEK_END);
    long file_size = ftell(file);

    zero_init_model_from_config(model, cfg);

    float* weight_ptr = (float*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    float* weight_ptr_copy = weight_ptr;

    if (weight_ptr == MAP_FAILED) {
        printf("Failed to map file\n");
        exit(1);
    }

    // start setting model weight pointers to their mapped weights

    // Embedding matrices
    model->Embedding = weight_ptr;
    weight_ptr += cfg.vocab_size * cfg.d_model;
    model->PositionalEncoding = weight_ptr;
    weight_ptr += cfg.max_context_len * cfg.d_model;

    // Transformer blocks

    /*
        GPT2 weights are stored in the following order:
        - Attention LayerNorm alpha
        - Attention LayerNorm betta
        - Attention wq
        - Attention wk
        - Attention wv
        - Attention wk bias
        - Attention wq bias
        - Attention wv bias
        - Attention wo
        - Attention wo bias
        - FFN LayerNorm alpha
        - FFN LayerNorm betta
        - FFN w_up
        - FFN w_up bias
        - FFN w_down
        - FFN w_down bias
    */
    for (int i = 0; i < cfg.num_layers; i++) {
        // Attn LayerNorm
        model->Blocks[i].AttnBlock->attn_norm_alpha = weight_ptr;
        weight_ptr += cfg.d_model;
        model->Blocks[i].AttnBlock->attn_norm_betta = weight_ptr;
        weight_ptr += cfg.d_model;

        // Attention Block
        model->Blocks[i].AttnBlock->wq = weight_ptr;
        weight_ptr += cfg.num_heads * cfg.d_model * cfg.head_dim;
        model->Blocks[i].AttnBlock->wk = weight_ptr;
        weight_ptr += cfg.num_heads * cfg.d_model * cfg.head_dim;
        model->Blocks[i].AttnBlock->wv = weight_ptr;
        weight_ptr += cfg.num_heads * cfg.d_model * cfg.head_dim;
        model->Blocks[i].AttnBlock->wq_bias = weight_ptr;
        weight_ptr += cfg.num_heads * cfg.head_dim;
        model->Blocks[i].AttnBlock->wk_bias = weight_ptr;
        weight_ptr += cfg.num_heads * cfg.head_dim;
        model->Blocks[i].AttnBlock->wv_bias = weight_ptr;
        weight_ptr += cfg.num_heads * cfg.head_dim;
        model->Blocks[i].AttnBlock->wo = weight_ptr;
        weight_ptr += cfg.d_model * cfg.d_model;
        model->Blocks[i].AttnBlock->wo_bias = weight_ptr;
        weight_ptr += cfg.d_model;

        // FFN Layer Norm
        model->Blocks[i].FFNBlock->ffn_norm_alpha = weight_ptr;
        weight_ptr += cfg.d_model;
        model->Blocks[i].FFNBlock->ffn_norm_betta = weight_ptr;
        weight_ptr += cfg.d_model;

        // FFN Block
        model->Blocks[i].FFNBlock->w_up = weight_ptr;
        weight_ptr += cfg.d_model * cfg.hidden_size;
        model->Blocks[i].FFNBlock->w_up_bias = weight_ptr;
        weight_ptr += cfg.hidden_size;
        model->Blocks[i].FFNBlock->w_down = weight_ptr;
        weight_ptr += cfg.hidden_size * cfg.d_model;
        model->Blocks[i].FFNBlock->w_down_bias = weight_ptr;
        weight_ptr += cfg.d_model;
    }

    // UnEmbedding Layer Norm
    model->UnEmbeddingLNAlpha = weight_ptr;
    weight_ptr += cfg.d_model;
    model->UnEmbeddingLNBetta = weight_ptr;
    weight_ptr += cfg.d_model;

    // Unembedding
    model->UnEmbedding = weight_ptr;
    weight_ptr += cfg.d_model  * cfg.vocab_size;

    printf("Model loaded\n");
    fclose(file);
}


RunState* initialize_runstate(ModelConfig cfg) {
    RunState* s = (RunState*)calloc(1, sizeof(RunState));

    // Current token and position
    s->token_idx = 50256; // EOS token
    s->position = 0;

    // Intermediate states
    s->x = (float*)calloc(cfg.d_model, sizeof(float));
    s->x_norm = (float*)calloc(cfg.d_model, sizeof(float));
    s->x_ffn_down = (float*)calloc(cfg.d_model, sizeof(float));
    s->x_ffn_up = (float*)calloc(cfg.hidden_size, sizeof(float)); // GPT2 uses 4x hidden size for the FFN layer

    // Attention cache, (layer, head, position, dim_head)
    s->k_cache = (float*)calloc(cfg.num_layers * cfg.num_heads * cfg.max_context_len * cfg.head_dim, sizeof(float));
    s->v_cache = (float*)calloc(cfg.num_layers * cfg.num_heads * cfg.max_context_len * cfg.head_dim, sizeof(float));

    // Attention matrices
    s->q = (float*)calloc(cfg.head_dim * cfg.num_heads, sizeof(float));
    s->k = (float*)calloc(cfg.head_dim * cfg.num_heads, sizeof(float));
    s->v = (float*)calloc(cfg.head_dim * cfg.num_heads, sizeof(float));
    s->attn_weights = (float*)calloc(cfg.max_context_len, sizeof(float));
    s->attn_out = (float*)calloc(cfg.d_model, sizeof(float));

    // Output
    s->logits = (float*)calloc(cfg.vocab_size, sizeof(float));
    s->probas = (float*)calloc(cfg.vocab_size, sizeof(float));

    // Check if allocations were successful
    if (s->x == NULL || s->x_norm == NULL || s->k_cache == NULL || s->v_cache == NULL || s->logits == NULL || s->probas == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    return s;
}

int forward(
    Model* model,
    RunState* s,
    ModelConfig cfg
) {
    embedding_lookup(model->Embedding, s->token_idx, s->x, cfg.d_model);
    vector_sum(s->x, model->PositionalEncoding + s->position * cfg.d_model, cfg.d_model);
    for (int l = 0; l < cfg.num_layers; l++) {
        layernorm(
            s->x,
            model->Blocks[l].AttnBlock->attn_norm_alpha,
            model->Blocks[l].AttnBlock->attn_norm_betta,
            s->x_norm,
            cfg.d_model
        );

        // MHA
        for (int h = 0; h < cfg.num_heads; h++) {
            fused_matmul_bias_transpose(
                s->x_norm,
                model->Blocks[l].AttnBlock->wq + h * cfg.d_model * cfg.head_dim,
                model->Blocks[l].AttnBlock->wq_bias + h * cfg.head_dim,
                s->q + h * cfg.head_dim,
                1,
                cfg.d_model,
                cfg.head_dim
            );
            fused_matmul_bias_transpose(
                s->x_norm,
                model->Blocks[l].AttnBlock->wk + h * cfg.d_model * cfg.head_dim,
                model->Blocks[l].AttnBlock->wk_bias + h * cfg.head_dim,
                s->k + h * cfg.head_dim,
                1,
                cfg.d_model,
                cfg.head_dim
            );
            fused_matmul_bias_transpose(
                s->x_norm,
                model->Blocks[l].AttnBlock->wv + h * cfg.d_model * cfg.head_dim,
                model->Blocks[l].AttnBlock->wv_bias + h * cfg.head_dim,
                s->v + h * cfg.head_dim,
                1,
                cfg.d_model,
                cfg.head_dim
            );

            // Put the values and keys into the cache (layer, head, position, dim_head)
            memcpy(s->k_cache + l * cfg.num_heads * cfg.max_context_len * cfg.head_dim + h * cfg.max_context_len * cfg.head_dim + s->position * cfg.head_dim, s->k + h * cfg.head_dim, cfg.head_dim * sizeof(float));
            memcpy(s->v_cache + l * cfg.num_heads * cfg.max_context_len * cfg.head_dim + h * cfg.max_context_len * cfg.head_dim + s->position * cfg.head_dim, s->v + h * cfg.head_dim, cfg.head_dim * sizeof(float));
            
            single_head_attention(
                s->q + h * cfg.head_dim,
                s->k_cache + l * cfg.num_heads * cfg.max_context_len * cfg.head_dim + h * cfg.max_context_len * cfg.head_dim,
                s->v_cache + l * cfg.num_heads * cfg.max_context_len * cfg.head_dim + h * cfg.max_context_len * cfg.head_dim,
                s->attn_weights,
                s->attn_out + h * cfg.head_dim,
                s->position + 1,
                cfg.head_dim,
                1
            );
        }

        // Output projection
        fused_matmul_bias_transpose(s->attn_out, model->Blocks[l].AttnBlock->wo, model->Blocks[l].AttnBlock->wo_bias, s->x_norm, 1, cfg.d_model, cfg.d_model);

        // Skip connection
        vector_sum(s->x, s->x_norm, cfg.d_model);

        layernorm(
            s->x,
            model->Blocks[l].FFNBlock->ffn_norm_alpha,
            model->Blocks[l].FFNBlock->ffn_norm_betta,
            s->x_norm,
            cfg.d_model
        );

        // FFN
        mlp(
            s->x_norm,
            model->Blocks[l].FFNBlock->w_up,
            model->Blocks[l].FFNBlock->w_up_bias,
            model->Blocks[l].FFNBlock->w_down,
            model->Blocks[l].FFNBlock->w_down_bias,
            s->x_ffn_up,
            s->x_ffn_down,
            1,
            cfg.d_model,
            cfg.hidden_size
        );

        // Skip connection
        vector_sum(s->x, s->x_ffn_down, cfg.d_model);
    }

    // Final layer norm
    layernorm(
        s->x,
        model->UnEmbeddingLNAlpha,
        model->UnEmbeddingLNBetta,
        s->x_norm,
        cfg.d_model
    );

    // Unembedding and Sample
    matmul_transpose(s->x_norm, model->UnEmbedding, s->logits, 1, cfg.d_model, cfg.vocab_size);
    softmax(s->logits, cfg.vocab_size, 0.2);
    int next_token = multinomial_sample(s->logits, cfg.vocab_size);
    
    return next_token;
}