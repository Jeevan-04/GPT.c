# Embedding Layer

## 1. Concept

A tokenizer converts text into token IDs. An embedding layer maps each token ID to a vector of floating-point numbers.

For example, if our vocabulary contains 100 tokens and each embedding has 8 dimensions:

\[
V=100,\qquad D=8
\]

The embedding matrix is:

\[
W\in\mathbb{R}^{V\times D}
\]

Therefore:

\[
W\in\mathbb{R}^{100\times8}
\]

Each row represents one token, and each column represents one dimension of its vector.

The total number of parameters is:

\[
N=V\times D=100\times8=800
\]

**Important:** Token IDs are identifiers, not meaningful numerical representations. The embedding layer provides trainable vector representations for them.

---

## 2. Structure: `embedding.h`

```c
typedef struct
{
    size_t vocab_size;
    size_t embedding_dim;
    float *weights;
} Embedding;
```

- `vocab_size`: Number of tokens \(V\).
- `embedding_dim`: Number of dimensions per token \(D\).
- `weights`: Pointer to the embedding matrix stored in contiguous memory.

We store the matrix in one flat array rather than allocating each row separately.

### Memory indexing

The mathematical matrix element \(W_{t,j}\) is stored at:

\[
\boxed{\text{index}=tD+j}
\]

where \(t\) is the token ID and \(j\) is the embedding dimension index.

For example, if \(D=8\), token 2 begins at index:

\[
2\times8=16
\]

Its eight values occupy indices 16 through 23.

---

## 3. Memory Allocation: `embedding_init()`

```c
size_t total_elements = vocab_size * embedding_dim;

embedding->weights =
    malloc(total_elements * sizeof(float));
```

The total number of elements is:

\[
N=VD
\]

The required memory is:

\[
\boxed{M=VD\times\operatorname{sizeof}(\texttt{float})}
\]

For 100 tokens with 8 dimensions each, assuming 4-byte floats:

\[
M=100\times8\times4=3200\text{ bytes}
\]

We check for zero dimensions and integer overflow before calculating the allocation size.

`malloc()` allocates memory but **does not initialize its values**. Therefore, we must zero or randomize the weights before reading them.

The function returns `0` on success and `-1` on failure.

---

## 4. Token Lookup: `embedding_get()`

```c
size_t offset = token_id * embedding->embedding_dim;

return &embedding->weights[offset];
```

Mathematically, retrieving a token embedding means selecting one row:

\[
\boxed{e_t=W_{t,:}}
\]

The offset is:

\[
\boxed{o=tD}
\]

The function returns a pointer to the beginning of that row. It does not create a copy of the vector.

We check that the token ID is valid:

\[
0\leq t<V
\]

If the embedding is invalid or the token ID is out of range, the function returns `NULL`.

---

## 5. Zero Initialization: `embedding_zero()`

```c
size_t total_elements =
    embedding->vocab_size * embedding->embedding_dim;

for (size_t i = 0; i < total_elements; i++)
{
    embedding->weights[i] = 0.0f;
}
```

This sets every parameter to zero:

\[
\boxed{W_{t,j}=0}
\]

for all valid rows and columns.

Time complexity:

\[
O(VD)
\]

Zero initialization is useful for testing and resetting weights. However, initializing every neural-network weight to zero is generally unsuitable for training because it can cause symmetry problems.

---

## 6. Random Initialization: `embedding_randomize()`

Our implementation uses the RNG module to generate pseudorandom values.

First, we convert a 32-bit unsigned integer into a floating-point value:

```c
double r = (double)rng_next(rng) / 4294967296.0;
```

Since:

\[
4294967296=2^{32}
\]

we obtain:

\[
0\leq r<1
\]

Next, we map the value into a symmetric interval:

```c
embedding->weights[i] =
    (float)((2.0 * r - 1.0) * (double)limit);
```

The mathematical transformation is:

\[
\boxed{w=(2r-1)L}
\]

where \(L\) is the positive limit.

Therefore:

\[
-L\leq w<L
\]

For example, if \(L=0.05\):

\[
-0.05\leq w<0.05
\]

Using the same RNG seed produces the same sequence of values, making our initialization reproducible.

This is a simple initialization strategy; later, we can implement initialization rules appropriate to the complete transformer architecture.

---

## 7. Memory Cleanup: `embedding_free()`

```c
free(embedding->weights);

embedding->weights = NULL;
embedding->vocab_size = 0;
embedding->embedding_dim = 0;
```

`free()` releases dynamically allocated memory.

We then reset the pointer and dimensions so the structure no longer represents an allocated embedding.

Our module supports safely freeing a zero-initialized structure and freeing the same structure again after it has been reset.

A new embedding should be initialized like this:

```c
Embedding embedding = {0};
```

Do not call `embedding_init()` again on an embedding that already owns allocated memory. Free it first.

---

## Summary

The embedding layer implements:

\[
\boxed{\text{Token ID }t\longrightarrow W_{t,:}}
\]

It stores a trainable matrix of shape \(V\times D\) and provides functions to allocate, retrieve, initialize, and free its weights.

Currently, the module initializes and retrieves parameters; **it does not yet train the embeddings**.