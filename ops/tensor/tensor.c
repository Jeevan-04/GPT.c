#include "tensor.h"

#include <stdint.h>
#include <stdlib.h>

static int tensor_validate(const Tensor *tensor)
{
    if (tensor == NULL || tensor->ndim == 0 || tensor->shape == NULL || tensor->data == NULL || tensor->size == 0)
        return -1;

    size_t expected_size = 1;

    for (size_t i = 0; i < tensor->ndim; i++)
    {
        if (tensor->shape[i] == 0 ||
            expected_size > SIZE_MAX / tensor->shape[i])
            return -1;

        expected_size *= tensor->shape[i];
    }

    return expected_size == tensor->size ? 0 : -1;
}

int tensor_init(Tensor *tensor, size_t ndim, const size_t *shape)
{
    if (tensor == NULL || ndim == 0 || shape == NULL)
        return -1;

    size_t size = 1;

    for (size_t i = 0; i < ndim; i++)
    {
        if (shape[i] == 0 || size > SIZE_MAX / shape[i])
            return -1;

        size *= shape[i];
    }

    if (ndim > SIZE_MAX / sizeof(size_t) ||
        size > SIZE_MAX / sizeof(float))
        return -1;

    size_t *tensor_shape = malloc(ndim * sizeof(size_t));

    if (tensor_shape == NULL)
        return -1;

    float *data = malloc(size * sizeof(float));

    if (data == NULL)
    {
        free(tensor_shape);
        return -1;
    }

    for (size_t i = 0; i < ndim; i++)
        tensor_shape[i] = shape[i];

    tensor->ndim = ndim;
    tensor->shape = tensor_shape;
    tensor->size = size;
    tensor->data = data;

    return 0;
}

void tensor_free(Tensor *tensor)
{
    if (tensor == NULL)
        return;

    free(tensor->shape);
    free(tensor->data);

    tensor->ndim = 0;
    tensor->shape = NULL;
    tensor->size = 0;
    tensor->data = NULL;
}

void tensor_zero(Tensor *tensor)
{
    if (tensor_validate(tensor) != 0)
        return;

    for (size_t i = 0; i < tensor->size; i++)
        tensor->data[i] = 0.0f;
}

int tensor_copy(Tensor *dst, const Tensor *src)
{
    if (dst == NULL || src == NULL || dst == src)
        return -1;

    if (tensor_validate(src) != 0)
        return -1;

    Tensor temp = {0};

    if (tensor_init(&temp, src->ndim, src->shape) != 0)
        return -1;

    for (size_t i = 0; i < src->size; i++)
        temp.data[i] = src->data[i];

    tensor_free(dst);
    *dst = temp;

    return 0;
}

static int tensor_offset( const Tensor *tensor, const size_t *indices, size_t *offset)
{
    if (indices == NULL || offset == NULL ||
        tensor_validate(tensor) != 0)
        return -1;

    size_t result = 0;

    for (size_t i = 0; i < tensor->ndim; i++)
    {
        if (indices[i] >= tensor->shape[i])
            return -1;

        result = result * tensor->shape[i] + indices[i];
    }

    *offset = result;

    return 0;
}

int tensor_get( const Tensor *tensor, const size_t *indices, float *value)
{
    if (value == NULL)
        return -1;

    size_t offset;

    if (tensor_offset(tensor, indices, &offset) != 0)
        return -1;

    *value = tensor->data[offset];

    return 0;
}

int tensor_set( Tensor *tensor, const size_t *indices, float value)
{
    size_t offset;

    if (tensor_offset(tensor, indices, &offset) != 0)
        return -1;

    tensor->data[offset] = value;

    return 0;
}

static int tensor_binary_op( const Tensor *a, const Tensor *b, Tensor *out, TensorBinaryOp operation)
{
    if (a == NULL || b == NULL || out == NULL ||
        out == a || out == b)
        return -1;

    if (operation != TENSOR_OP_ADD && operation != TENSOR_OP_SUB && operation != TENSOR_OP_MUL)
        return -1;

    if (tensor_validate(a) != 0 || tensor_validate(b) != 0)
        return -1;

    if (a->ndim != b->ndim || a->size != b->size)
        return -1;

    for (size_t i = 0; i < a->ndim; i++)
    {
        if (a->shape[i] != b->shape[i])
            return -1;
    }

    Tensor temp = {0};

    if (tensor_init(&temp, a->ndim, a->shape) != 0)
        return -1;

    for (size_t i = 0; i < a->size; i++)
    {
        switch (operation)
        {
        case TENSOR_OP_ADD:
            temp.data[i] = a->data[i] + b->data[i];
            break;

        case TENSOR_OP_SUB:
            temp.data[i] = a->data[i] - b->data[i];
            break;

        case TENSOR_OP_MUL:
            temp.data[i] = a->data[i] * b->data[i];
            break;
        }
    }

    tensor_free(out);
    *out = temp;

    return 0;
}

int tensor_add(const Tensor *a, const Tensor *b, Tensor *out)
{
    return tensor_binary_op(a, b, out, TENSOR_OP_ADD);
}

int tensor_sub(const Tensor *a, const Tensor *b, Tensor *out)
{
    return tensor_binary_op(a, b, out, TENSOR_OP_SUB);
}

int tensor_mul(const Tensor *a, const Tensor *b, Tensor *out)
{
    return tensor_binary_op(a, b, out, TENSOR_OP_MUL);
}

int tensor_scale(const Tensor *a, float scalar, Tensor *out)
{
    if (a == NULL || out == NULL || a == out)
        return -1;

    if (tensor_validate(a) != 0)
        return -1;

    Tensor temp = {0};

    if (tensor_init(&temp, a->ndim, a->shape) != 0)
        return -1;

    for (size_t i = 0; i < a->size; i++)
        temp.data[i] = a->data[i] * scalar;

    tensor_free(out);
    *out = temp;

    return 0;
}