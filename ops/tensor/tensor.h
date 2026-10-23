#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct
{
    size_t ndim;
    size_t *shape;
    size_t size;
    float *data;
} Tensor;

typedef enum
{
    TENSOR_OP_ADD,
    TENSOR_OP_SUB,
    TENSOR_OP_MUL
} TensorBinaryOp;

int tensor_init(Tensor *tensor, size_t ndim, const size_t *shape);

void tensor_free(Tensor *tensor);

void tensor_zero(Tensor *tensor);

int tensor_copy(Tensor *dst, const Tensor *src);

int tensor_get(const Tensor *tensor, const size_t *indices, float *value);

int tensor_set(Tensor *tensor, const size_t *indices, float value);

int tensor_add(const Tensor *a, const Tensor *b, Tensor *out);

int tensor_sub(const Tensor *a, const Tensor *b, Tensor *out);

int tensor_mul(const Tensor *a, const Tensor *b, Tensor *out);

int tensor_scale(const Tensor *a, float scalar, Tensor *out);

#endif