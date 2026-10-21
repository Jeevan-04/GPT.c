#include "embedding.h"

#include <stdint.h>
#include <stdlib.h>

int embedding_init(Embedding *embedding, size_t vocab_size, size_t embedding_dim)
{

    if(embedding==NULL || vocab_size==0 || embedding_dim==0)
        return -1;

    if(vocab_size > SIZE_MAX / embedding_dim)
        return -1;

    size_t total_elements = vocab_size * embedding_dim;

    if(total_elements > SIZE_MAX / sizeof(float))
        return -1;

    embedding->vocab_size = vocab_size;
    embedding->embedding_dim = embedding_dim;
    embedding->weights = NULL;

    embedding->weights = malloc(total_elements * sizeof(float));

    if(embedding->weights == NULL)
    {
        embedding->vocab_size = 0;
        embedding->embedding_dim = 0;
        return -1;
    }

    return 0;
}

void embedding_free(Embedding *embedding)
{
    if(embedding == NULL)
        return;
    
    free(embedding->weights);

    embedding->weights = NULL;
    embedding->vocab_size = 0;
    embedding->embedding_dim = 0;
}

float *embedding_get(Embedding *embedding, size_t token_id)
{
    if(embedding == NULL || embedding->weights == NULL|| token_id >= embedding->vocab_size)
        return NULL;
    
    size_t offset = token_id * embedding->embedding_dim;

    return &embedding->weights[offset];
}

void embedding_zero(Embedding *embedding)
{
    if(embedding == NULL || embedding->weights == NULL)
        return;
    
    size_t total_elements = embedding->vocab_size * embedding->embedding_dim;

    for(size_t i=0; i < total_elements; i++)
    {
        embedding->weights[i] = 0.0f;
    }
}

void embedding_randomize(Embedding *embedding, float limit, RNG *rng)
{
    if (embedding == NULL || embedding->weights == NULL || rng == NULL || limit <= 0.0f)
        return;

    size_t total_elements = embedding->vocab_size * embedding->embedding_dim;

    for (size_t i = 0; i < total_elements; i++)
    {
        double r = (double)rng_next(rng) / 4294967296.0;

        embedding->weights[i] = (float)((2.0 * r - 1.0) * (double)limit);
    }
}