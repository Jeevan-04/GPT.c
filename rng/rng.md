# Random Number Generator (RNG)

## 1. Concept

Neural networks need initial parameter values. Our embedding layer uses a **pseudorandom number generator (PRNG)** to initialize its weights.

A PRNG generates a sequence of numbers from an initial value called a *seed*.

- Same seed → same sequence.
- Different seeds → generally different sequences.

This makes experiments reproducible.

Our implementation uses **PCG32 (Permuted Congruential Generator)**, which maintains a 64-bit internal state and produces 32-bit pseudorandom outputs.

> This is a deterministic PRNG, not a cryptographically secure random-number generator.

---

## 2. Structure: `rng.h`

```c
typedef struct
{
    uint64_t state;
    uint64_t increment;
} RNG;
```

- `state`: The generator's current 64-bit internal state.
- `increment`: A constant added during each state transition.

We use `uint64_t` because the algorithm requires unsigned 64-bit arithmetic.

The header also declares:

```c
void rng_init(RNG *rng, uint64_t seed);
uint32_t rng_next(RNG *rng);
```

`rng_init()` initializes the generator, while `rng_next()` advances its state and returns a 32-bit result.

---

## 3. State Initialization: `rng_init()`

Our implementation:

```c
rng->state = 0;
rng->increment = UINT64_C(1442695040888963407);

rng_next(rng);
rng->state += seed;
rng_next(rng);
```

The generator is initialized in three stages:

1. Start with a zero state and set the increment.
2. Advance the generator once.
3. Add the seed to the state and advance it again.

This sequence initializes the internal state before producing values for the caller.

`UINT64_C(...)` specifies an integer constant suitable for 64-bit unsigned arithmetic.

The function checks whether `rng` is `NULL` before accessing its fields.

---

## 4. Generating a Number: `rng_next()`

### Step 1: Save the old state

```c
uint64_t oldstate = rng->state;
```

We preserve the current state because it is used both to calculate the next state and to generate the output.

### Step 2: Update the state

```c
rng->state =
    oldstate * UINT64_C(6364136223846793005)
    + rng->increment;
```

Mathematically:

\[
\boxed{s_{n+1}=(a s_n+c)\bmod 2^{64}}
\]

where:

- \(s_n\) is the current state.
- \(a=6364136223846793005\).
- \(c\) is the increment.
- \(s_{n+1}\) is the next state.

Unsigned 64-bit arithmetic in C wraps modulo \(2^{64}\).

The state transition is deterministic: the same current state and increment produce the same next state.

### Step 3: Mix bits from the old state

```c
uint32_t xorshifted =
    (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
```

This combines a shifted version of the state with the original state using XOR (`^`), then shifts the result again.

The operations rearrange and mix bits to produce a 32-bit intermediate value.

The cast to `uint32_t` retains the low 32 bits of the result.

### Step 4: Calculate the rotation amount

```c
uint32_t rotation = (uint32_t)(oldstate >> 59u);
```

Shifting right by 59 positions extracts the top five bits of the old 64-bit state.

Therefore:

\[
0\leq rotation<32
\]

### Step 5: Rotate the bits

```c
return (xorshifted >> rotation)
     | (xorshifted << ((-(int32_t)rotation) & 31));
```

This performs a **32-bit rotate-right** operation.

Unlike an ordinary shift, a rotation moves bits that leave one end of the value back to the other end.

The expression:

```c
(-(int32_t)rotation) & 31
```

calculates the complementary shift amount for the 32-bit rotation.

The bitwise OR (`|`) combines the two shifted values.

Together, these operations implement the output permutation used by PCG32's XSH-RR variant.

The function returns the resulting `uint32_t`.

If the input pointer is `NULL`, our implementation returns zero.

---

## 5. Using the RNG for Embeddings

In `embedding_randomize()`, we call:

```c
rng_next(rng);
```

This produces an integer in the range:

\[
0\leq x\leq 2^{32}-1
\]

We convert it to a floating-point value:

\[
r=\frac{x}{2^{32}}
\]

Therefore:

\[
0\leq r<1
\]

Then we map it to the desired initialization interval:

\[
\boxed{w=(2r-1)L}
\]

where \(L>0\) is the initialization limit.

This connects our RNG module to the embedding module.

---

## 6. Complexity and Limitations

Each call to `rng_next()` performs a fixed number of arithmetic and bitwise operations.

Therefore:

\[
\boxed{\text{Time complexity}=O(1)}
\]

The generator stores only its state and increment:

\[
\boxed{\text{State storage}=O(1)}
\]

PCG32 is appropriate for reproducible model initialization and ordinary simulation tasks. It should not be used for passwords, cryptographic keys, or security-sensitive randomness.

---

## Summary

Our RNG module implements PCG32 using:

- A 64-bit internal state.
- A fixed increment.
- A deterministic state transition.
- Bitwise mixing and rotation to generate 32-bit outputs.
- Seed-based initialization for reproducibility.

The embedding module uses these outputs to initialize its weight matrix with pseudorandom floating-point values.