# GPT.c

GPT implemented from scratch in C, covering the core architecture of a Generative Pre-trained Transformer—from token embeddings and positional encoding to self-attention, transformer blocks, and autoregressive text generation.

Each component is implemented as an independent C module, with its own header, source file, and documentation. The modules are tested individually before being integrated into the complete architecture.

## Architecture & Progress

```text
                      OUTPUT PROBABILITIES
                               ^
                               |
                         +-----------+
                         |  Softmax  |  [ ]
                         +-----------+
                               ^
                               |
                         +-----------+
                         |  Linear   |  [ ]
                         |  (Output) |
                         +-----------+
                               ^
                               |
              +================================+
              |       TRANSFORMER BLOCK         |
              |              N x                |
              |                                 |
              |   +-------------------------+   |
              |   |   Layer Normalization   |   | [ ]
              |   +-------------------------+   |
              |                ^                |
              |                |                |
              |   +-------------------------+   |
              |   | Feed-Forward Network    |   | [ ]
              |   | Linear -> GELU ->       |   |
              |   | Linear                  |   |
              |   +-------------------------+   |
              |                ^                |
              |          Residual Add           | [ ]
              |                ^                |
              |   +-------------------------+   |
              |   |   Layer Normalization   |   | [ ]
              |   +-------------------------+   |
              |                ^                |
              |                |                |
              |   +-------------------------+   |
              |   | Masked Multi-Head       |   | [ ]
              |   | Self-Attention           |   |
              |   +-------------------------+   |
              |                ^                |
              |          Residual Add           | [ ]
              +================|================+
                               |
                      +-------------------+
                      | Token Embeddings  |  [x]
                      | embedding/        |
                      +-------------------+
                               +
                      +-------------------+
                      | Positional        |  [ ]
                      | Encoding          |
                      +-------------------+
                               ^
                               |
                      +-------------------+
                      |   Input Tokens    |  [ ]
                      +-------------------+
```


**Note:** This diagram is a simplified GPT architecture. The residual connections and normalization order are schematic; we will define their exact mathematical operations when implementing the transformer block.

## Module Checklist

### Core Infrastructure
- [x] `rng/` — Pseudorandom Number Generator (PCG32)
- [ ] `tensor/` — Tensor and Matrix Operations
- [ ] `tokenizer/` — Tokenization and Vocabulary

### Input Representation
- [x] `embedding/` — Token Embeddings
- [ ] `positional_encoding/` — Positional Encoding

### Transformer
- [ ] `linear/` — Linear Transformation
- [ ] `layer_norm/` — Layer Normalization
- [ ] `attention/` — Scaled Dot-Product Attention
- [ ] `multi_head_attention/` — Multi-Head Causal Self-Attention
- [ ] `feed_forward/` — Feed-Forward Network and GELU
- [ ] `transformer_block/` — Transformer Block and Residual Connections
- [ ] `gpt/` — Complete GPT Architecture

### Output
- [ ] `output/` — Language Model Output Projection
- [ ] `softmax/` — Probability Distribution
- [ ] `generation/` — Autoregressive Text Generation and Sampling

### Training
- [ ] `loss/` — Cross-Entropy Loss
- [ ] `autograd/` — Automatic Differentiation and Backpropagation
- [ ] `optimizer/` — Adam Optimizer
- [ ] `training/` — Training Loop and Batching
- [ ] `checkpoint/` — Model Saving and Loading

### Testing and Optimization
- [ ] `tests/` — Unit and Integration Tests
- [ ] `benchmarks/` — Performance Benchmarks
- [ ] `accelerate/` — Apple Accelerate Integration

## Current Status

**Completed:** `rng/`, `embedding/`

**Next module:** `tensor/`