#include "matmul.h"

#include <Accelerate/Accelerate.h>
#include <limits.h>
#include <stdint.h>

static int matmul_validate(const Tensor *tensor)
{
    if (tensor == NULL || tensor->ndim != 2 || tensor->shape == NULL ||
        tensor->data == NULL || tensor->shape[0] == 0 || tensor->shape[1] == 0)
        return -1;

    if (tensor->shape[0] > SIZE_MAX / tensor->shape[1])
        return -1;

    if (tensor->shape[0] * tensor->shape[1] != tensor->size)
        return -1;

    return 0;
}

int matmul(const Tensor *a, const Tensor *b, Tensor *out)
{
    if (a == NULL || b == NULL || out == NULL ||
        out == a || out == b)
        return -1;

    if (matmul_validate(a) != 0 ||
        matmul_validate(b) != 0)
        return -1;

    size_t m = a->shape[0];
    size_t k = a->shape[1];
    size_t n = b->shape[1];

    if (k != b->shape[0])
        return -1;

    if (m > INT_MAX || n > INT_MAX || k > INT_MAX)
        return -1;

    size_t output_shape[] = {m, n};
    Tensor result = {0};

    if (tensor_init(&result, 2, output_shape) != 0)
        return -1;

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k,
        1.0f, a->data, (int)k, b->data, (int)n, 0.0f, result.data, (int)n);

    tensor_free(out);
    *out = result;

    return 0;
}