#include "transpose.h"

#include <stdint.h>

int tensor_transpose(const Tensor *input, Tensor *out)
{
    if (input == NULL || out == NULL || input == out)
        return -1;

    if (input->ndim != 2 ||
        input->shape == NULL ||
        input->data == NULL)
        return -1;

    size_t rows = input->shape[0];
    size_t cols = input->shape[1];

    if (rows == 0 || cols == 0)
        return -1;

    if (rows > SIZE_MAX / cols)
        return -1;

    if (rows * cols != input->size)
        return -1;

    size_t output_shape[] = {cols, rows};
    Tensor result = {0};

    if (tensor_init(&result, 2, output_shape) != 0)
        return -1;

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            result.data[j * rows + i] = input->data[i * cols + j];
        }
    }

    tensor_free(out);
    *out = result;

    return 0;
}