#ifndef RNG_H
#define RNG_H

#include <stdint.h>

typedef struct
{
    uint64_t state;
    uint64_t increment;
} RNG;

void rng_init(RNG *rng, uint64_t seed);
uint32_t rng_next(RNG *rng);

#endif // RNG_H