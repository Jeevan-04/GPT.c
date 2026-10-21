# Embedding layer

The embedding layer maps discrete token IDs to dense floating-point vectors.
It stores the vectors in a single contiguous `float` array and exposes
operations for allocation, lookup, initialization, and cleanup.

Implementation:

- [`embedding.h`](./embedding.h)
- [`embedding.c`](./embedding.c)

## API

```c
typedef struct
{
    size_t vocab_size;
    size_t embedding_dim;
    float *weights;
} Embedding;

int embedding_init(
    Embedding *embedding,
    size_t vocab_size,
    size_t embedding_dim
);

void embedding_free(Embedding *embedding);
float *embedding_get(Embedding *embedding, size_t token_id);
void embedding_zero(Embedding *embedding);
void embedding_randomize(Embedding *embedding, float limit, RNG *rng);
```

### Structure fields

| Field | Meaning |
| --- | --- |
| `vocab_size` | Number of token IDs represented by the table, `V`. |
| `embedding_dim` | Number of values in each token vector, `D`. |
| `weights` | Contiguous storage for all `V × D` `float` values. |

## Data layout

For a vocabulary of size `V` and vectors of dimension `D`, the weight matrix
has shape:

```text
weights[V][D]
```

The implementation stores that matrix as one flat array. The value at row
`token_id` and column `dimension` is located at:

```text
weights[token_id * embedding_dim + dimension]
```

The table contains `V × D` values and requires:

```text
V × D × sizeof(float)
```

bytes of storage.

## Usage

Always start with a zero-initialized structure:

```c
Embedding embedding = {0};
RNG rng;

if (embedding_init(&embedding, 100, 8) != 0)
{
    /* Allocation or input validation failed. */
    return 1;
}

rng_init(&rng, 123);
embedding_randomize(&embedding, 0.05f, &rng);

float *token_vector = embedding_get(&embedding, 2);
if (token_vector != NULL)
{
    /* token_vector points to 8 values in this example. */
}

embedding_free(&embedding);
```

`embedding_get()` returns a pointer into the table; it does not allocate or
copy a vector. Do not use that pointer after `embedding_free()` or after the
embedding has otherwise been released.

## Initialization and error handling

`embedding_init()` returns:

- `0` on success.
- `-1` when the embedding pointer is `NULL`, either dimension is zero, the
  requested allocation would overflow `size_t`, or memory allocation fails.

The implementation checks both multiplication steps before allocating:

1. `vocab_size * embedding_dim`
2. `total_elements * sizeof(float)`

`malloc()` does not initialize the allocated values. Call `embedding_zero()` or
`embedding_randomize()` before reading the weights.

## Operations

### `embedding_get()`

Returns the first value of the row for `token_id`. A `NULL` pointer is returned
when the embedding is invalid, has no allocated weights, or `token_id` is
outside the range `[0, vocab_size)`.

Lookup is `O(1)`.

### `embedding_zero()`

Sets every weight to `0.0f`:

```text
W[token_id][dimension] = 0
```

The operation is `O(V × D)`. Zero initialization is useful for tests and
resetting a table, but is generally not a suitable neural-network
initialization strategy because it can preserve symmetry between parameters.

### `embedding_randomize()`

Generates each weight using the supplied [`RNG`](../rng/rng.h) and maps the
result to the interval `[-limit, limit)`:

```text
r = rng_next(rng) / 2^32
w = (2r - 1) × limit
```

The function does nothing when the embedding, RNG, or weight storage is
invalid, or when `limit` is not positive. Reusing the same RNG seed produces
the same sequence of weights.

### `embedding_free()`

Releases the weight storage and resets all structure fields to zero. It is safe
to call with `NULL` and safe to call more than once.

Do not call `embedding_init()` on a structure that already owns allocated
weights. Call `embedding_free()` first to avoid leaking the existing
allocation.

## Complexity

| Operation | Time | Additional storage |
| --- | ---: | ---: |
| `embedding_init()` | `O(V × D)` allocation size | `O(V × D)` |
| `embedding_get()` | `O(1)` | `O(1)` |
| `embedding_zero()` | `O(V × D)` | `O(1)` |
| `embedding_randomize()` | `O(V × D)` | `O(1)` |
| `embedding_free()` | `O(1)` | `O(1)` |

The layer currently stores and initializes parameters, but it does not train
them or compute gradients.
