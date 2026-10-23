#ifndef MATMUL_H
#define MATMUL_H

#include "../tensor/tensor.h"

int matmul(const Tensor *a, const Tensor *b, Tensor *out);

#endif //MATMUL_H