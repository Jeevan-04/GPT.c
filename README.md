# GPT.c

GPT.c is an experimental implementation of a Generative Pre-trained Transformer
written from scratch in C. The project is being built one small, independently
tested module at a time.

The current implementation includes:

- A PCG32-based pseudorandom number generator.
- A trainable token-embedding table.
- Tests for allocation, lookup, initialization, cleanup, and error handling.

The transformer itself is still under development. The checklist below tracks
the planned components.

## Project status

| Component | Status |
| --- | --- |
| [`rng/`](./rng/) — PCG32 pseudorandom number generator | Complete |
| [`embedding/`](./embedding/) — token embeddings | Complete |
| `tensor/` — tensor and matrix operations | Planned |
| `tokenizer/` — tokenization and vocabulary | Planned |
| `positional_encoding/` — positional encoding | Planned |
| `linear/` — linear transformations | Planned |
| `layer_norm/` — layer normalization | Planned |
| `attention/` — scaled dot-product attention | Planned |
| `multi_head_attention/` — causal multi-head self-attention | Planned |
| `feed_forward/` — feed-forward network and GELU | Planned |
| `transformer_block/` — transformer blocks and residual connections | Planned |
| `gpt/` — complete GPT model | Planned |
| `softmax/` and `generation/` — output probabilities and text generation | Planned |
| `loss/`, `autograd/`, `optimizer/`, and `training/` | Planned |
| `tests/`, `benchmarks/`, and Apple Accelerate support | Planned |

## Architecture

The target architecture is a decoder-only GPT-style transformer:

```text
Input token IDs
      │
      ├── Token embeddings
      └── Positional encodings
      │
      ▼
Masked multi-head self-attention
      │
      ▼
Feed-forward network
      │
      └── Repeated transformer blocks
      │
      ▼
Output projection → Softmax → Next-token probabilities
```

The diagram describes the intended architecture. Only the RNG and embedding
layers are implemented at this stage.

## Build and run

The repository currently uses a small C test program in [`main.c`](./main.c).
Build it with any C11-compatible compiler:

```sh
cc -std=c11 -Wall -Wextra -Wpedantic \
  main.c embedding/embedding.c rng/rng.c \
  -lm -o embedding_test
```

Run the tests:

```sh
./embedding_test
```

The test program exits with a non-zero status if a test fails.

## Modules

- [`embedding/embedding.h`](./embedding/embedding.h) and
  [`embedding/embedding.c`](./embedding/embedding.c) implement the embedding
  table.
- [`embedding/embedding.md`](./embedding/embedding.md) documents the data
  layout, API, initialization, and memory ownership rules.
- [`rng/rng.h`](./rng/rng.h) and [`rng/rng.c`](./rng/rng.c) implement the
  deterministic PCG32 generator.
- [`rng/rng.md`](./rng/rng.md) documents the generator and its intended use.

## Scope and limitations

This project is educational and incomplete. The current modules do not provide
tokenization, model training, automatic differentiation, inference, or text
generation. The RNG is deterministic but is **not** suitable for passwords,
cryptographic keys, or other security-sensitive purposes.

## License

No license has been added to this repository yet. Until a license is provided,
please treat the source as all-rights-reserved.
