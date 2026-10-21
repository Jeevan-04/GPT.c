#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stddef.h>
#include "../rng/rng.h"

typedef struct
{
    size_t vocab_size;
    size_t embedding_dim;
    float *weights;
} Embedding;

int embedding_init
(
    Embedding *embedding,
    size_t vocab_size,
    size_t embedding_dim
);

void embedding_free(Embedding *embedding);

float *embedding_get(Embedding *embedding, size_t token_id);

void embedding_zero(Embedding *embedding);

void embedding_randomize(Embedding *embedding, float limit, RNG *rng);

#endif // EMBEDDING_H