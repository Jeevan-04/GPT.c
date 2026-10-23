#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../ops/tensor/tensor.h"

int softmax(const Tensor *input, size_t axis, Tensor *out);

#endif