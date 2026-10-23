# PCG32 pseudorandom number generator

The RNG module provides deterministic pseudorandom values for model
initialization and reproducible experiments. It implements the PCG32
XSH-RR output function using a 64-bit internal state and 32-bit results.

Implementation:

- [`rng.h`](./rng.h)
- [`rng.c`](./rng.c)

> This generator is not cryptographically secure. Do not use it for passwords,
> cryptographic keys, tokens, or security-sensitive decisions.

## API

```c
typedef struct
{
    uint64_t state;
    uint64_t increment;
} RNG;

void rng_init(RNG *rng, uint64_t seed);
uint32_t rng_next(RNG *rng);
```

| Field or function | Description |
| --- | --- |
| `state` | Current 64-bit generator state. |
| `increment` | Stream increment used by the state transition. |
| `rng_init()` | Initializes a generator from a seed. |
| `rng_next()` | Advances the generator and returns a `uint32_t`. |

## Usage

```c
RNG rng;
rng_init(&rng, 123);

uint32_t value = rng_next(&rng);
```

The same seed produces the same sequence on every run, provided the same
implementation and unsigned integer widths are used. A `NULL` pointer passed
to either function is handled safely: `rng_init()` does nothing and
`rng_next()` returns `0`.

## How it works

### State transition

Each call saves the old state and advances the generator with a linear
congruential step:

```text
s[n+1] = (6364136223846793005 × s[n] + increment) mod 2^64
```

Unsigned 64-bit arithmetic in C wraps modulo `2^64`.

### Output permutation

The output is derived from the old state in three stages:

1. XOR and shift the state to produce a 32-bit value.
2. Use the top five state bits as a rotation amount.
3. Rotate the 32-bit value to produce the result.

The relevant operations in the implementation are:

```c
uint32_t xorshifted =
    (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);

uint32_t rotation = (uint32_t)(oldstate >> 59u);

return (xorshifted >> rotation)
     | (xorshifted << ((-(int32_t)rotation) & 31));
```

The rotation is always in `[0, 31]`, so the expression performs a valid
32-bit rotate-right operation.

### Seed initialization

`rng_init()` sets a fixed stream increment and performs a short warm-up
sequence:

1. Set the state to zero.
2. Set the increment to `1442695040888963407`.
3. Advance once.
4. Add the caller's seed to the state.
5. Advance again.

This ensures that the seed is incorporated before values are returned to the
caller.

## Generating floating-point values

The embedding module converts the 32-bit output to a value in `[0, 1)`:

```c
double r = (double)rng_next(rng) / 4294967296.0;
```

It then maps the result to a symmetric interval:

```c
float weight = (float)((2.0 * r - 1.0) * (double)limit);
```

This produces values in `[-limit, limit)`, subject to floating-point
representation.

## Complexity

Each call to `rng_next()` performs a fixed number of arithmetic and bitwise
operations:

- Time: `O(1)`
- Additional storage: `O(1)`
- Output range: `0` through `UINT32_MAX`

The generator is appropriate for reproducible initialization and ordinary
simulation. It does not provide cryptographic security, entropy from the
operating system, or protection against prediction.
