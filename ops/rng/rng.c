#include <stddef.h>
#include "rng.h"

uint32_t rng_next(RNG *rng)
{
    if(rng == NULL)
        return 0;

    uint64_t oldstate = rng->state;

    rng->state = oldstate * UINT64_C(6364136223846793005) + rng->increment;

    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);

    uint32_t rotation = (uint32_t)(oldstate >> 59u);

    return(xorshifted >> rotation) | (xorshifted << ((-(int32_t)rotation) & 31));
}

void rng_init(RNG *rng, uint64_t seed)
{
    if (rng == NULL)
        return;

    rng->state = 0;
    rng->increment = UINT64_C(1442695040888963407);

    rng_next(rng);
    rng->state += seed;
    rng_next(rng);
}