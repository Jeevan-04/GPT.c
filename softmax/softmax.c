#include "softmax.h"

#include <math.h>
#include <stdint.h>

int softmax(const Tensor *input, size_t axis, Tensor *out)
{
    if (input == NULL || out == NULL || input == out)
        return -1;

    if (input->ndim == 0 ||
        input->shape == NULL ||
        input->data == NULL ||
        axis >= input->ndim)
        return -1;

    size_t expected_size = 1;

    for (size_t i = 0; i < input->ndim; i++)
    {
        if (input->shape[i] == 0 ||
            expected_size > SIZE_MAX / input->shape[i])
            return -1;

        expected_size *= input->shape[i];
    }

    if (expected_size != input->size)
        return -1;

    size_t axis_size = input->shape[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;

    for (size_t i = 0; i < axis; i++)
        outer_size *= input->shape[i];

    for (size_t i = axis + 1; i < input->ndim; i++)
        inner_size *= input->shape[i];

    Tensor result = {0};

    if (tensor_init(&result, input->ndim, input->shape) != 0)
        return -1;

    for (size_t outer = 0; outer < outer_size; outer++)
    {
        for (size_t inner = 0; inner < inner_size; inner++)
        {
            size_t base = outer * axis_size * inner_size + inner;
            float max_value = input->data[base];

            for (size_t j = 0; j < axis_size; j++)
            {
                float value = input->data[base + j * inner_size];

                if (!isfinite(value))
                {
                    tensor_free(&result);
                    return -1;
                }

                if (value > max_value)
                    max_value = value;
            }

            double sum = 0.0;

            for (size_t j = 0; j < axis_size; j++)
            {
                size_t index = base + j * inner_size;
                double exponent =
                    exp((double)input->data[index] - max_value);

                result.data[index] = (float)exponent;
                sum += exponent;
            }

            if (!isfinite(sum) || sum <= 0.0)
            {
                tensor_free(&result);
                return -1;
            }

            for (size_t j = 0; j < axis_size; j++)
            {
                size_t index = base + j * inner_size;
                result.data[index] =
                    (float)(result.data[index] / sum);
            }
        }
    }

    tensor_free(out);
    *out = result;

    return 0;
}