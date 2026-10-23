#include <stdio.h>

#include "ops/rng/rng.h"
#include "embedding/embedding.h"
#include "ops/tensor/tensor.h"
#include "ops/matmul/matmul.h"
#include "ops/transpose/transpose.h"
#include "softmax/softmax.h"

#define VOCAB_SIZE 100
#define EMBEDDING_DIM 8
#define RANDOM_LIMIT 0.05f

static void print_tensor(const char *name, const Tensor *t)
{
    printf("\n--- %s ---\n", name);
    printf("ndim: %zu\n", t->ndim);
    printf("shape: {");

    for (size_t i = 0; i < t->ndim; i++)
    {
        printf("%zu", t->shape[i]);

        if (i + 1 < t->ndim)
            printf(", ");
    }

    printf("}\n");
    printf("size: %zu\n", t->size);

    if (t->ndim != 2)
    {
        for (size_t i = 0; i < t->size; i++)
            printf("data[%zu] = %f\n", i, t->data[i]);

        return;
    }

    for (size_t i = 0; i < t->shape[0]; i++)
    {
        printf("Row %zu: [", i);

        for (size_t j = 0; j < t->shape[1]; j++)
        {
            size_t indices[] = {i, j};
            float value;

            if (tensor_get(t, indices, &value) != 0)
            {
                printf("ERROR: tensor_get failed\n");
                return;
            }

            printf("% .6f", value);

            if (j + 1 < t->shape[1])
                printf(", ");
        }

        printf("]\n");
    }
}

static int set_tensor_data(Tensor *tensor, const float *values)
{
    if (tensor == NULL || values == NULL)
        return -1;

    for (size_t i = 0; i < tensor->size; i++)
        tensor->data[i] = values[i];

    return 0;
}

int main(void)
{
    int status = 1;

    RNG rng;
    rng_init(&rng, 123);

    Embedding embedding = {0};
    Tensor input = {0};
    Tensor copy = {0};
    Tensor a = {0};
    Tensor b = {0};
    Tensor c = {0};
    Tensor sub_result = {0};
    Tensor mul_result = {0};
    Tensor scale_result = {0};
    Tensor softmax_input = {0};
    Tensor softmax_vector_output = {0};
    Tensor softmax_matrix_output = {0};

    const char *tokens[] = {"I", "am", "Jeevan"};
    size_t token_ids[] = {17, 42, 91};

    printf("====================================\n");
    printf("       GPT.c INTEGRATED WALKTHROUGH\n");
    printf("====================================\n");

    printf("\n[STEP 1] Initialize RNG\n");
    printf("RNG initialized with seed 123.\n");

    printf("\n[STEP 2] Initialize embedding table\n");

    if (embedding_init(&embedding, VOCAB_SIZE, EMBEDDING_DIM) != 0)
    {
        printf("Embedding initialization failed.\n");
        goto cleanup;
    }

    printf("Vocabulary size: %zu\n", embedding.vocab_size);
    printf("Embedding dimension: %zu\n", embedding.embedding_dim);

    printf("\n[STEP 3] Randomize embedding weights\n");

    embedding_randomize(&embedding, RANDOM_LIMIT, &rng);

    printf("Weights initialized in [-%.3f, +%.3f).\n",
           RANDOM_LIMIT, RANDOM_LIMIT);

    printf("\n[STEP 4] Define input tokens\n");

    for (size_t i = 0; i < 3; i++)
        printf("Token: %-8s Token ID: %zu\n", tokens[i], token_ids[i]);

    printf("\n[STEP 5] Retrieve embedding vectors\n");

    for (size_t i = 0; i < 3; i++)
    {
        float *vector = embedding_get(&embedding, token_ids[i]);

        if (vector == NULL)
        {
            printf("Embedding lookup failed for %s.\n", tokens[i]);
            goto cleanup;
        }

        printf("%s -> [", tokens[i]);

        for (size_t j = 0; j < EMBEDDING_DIM; j++)
        {
            printf("% .6f", vector[j]);

            if (j + 1 < EMBEDDING_DIM)
                printf(", ");
        }

        printf("]\n");
    }

    printf("\n[STEP 6] Create tensor from input embeddings\n");

    size_t input_shape[] = {3, EMBEDDING_DIM};

    if (tensor_init(&input, 2, input_shape) != 0)
    {
        printf("Tensor initialization failed.\n");
        goto cleanup;
    }

    tensor_zero(&input);

    printf("Tensor shape: {3, %d}\n", EMBEDDING_DIM);

    printf("\n[STEP 7] Transfer embeddings into tensor\n");

    for (size_t i = 0; i < 3; i++)
    {
        float *vector = embedding_get(&embedding, token_ids[i]);

        if (vector == NULL)
        {
            printf("Embedding lookup failed.\n");
            goto cleanup;
        }

        for (size_t j = 0; j < EMBEDDING_DIM; j++)
        {
            size_t indices[] = {i, j};

            if (tensor_set(&input, indices, vector[j]) != 0)
            {
                printf("Tensor write failed.\n");
                goto cleanup;
            }
        }
    }

    print_tensor("Input embedding tensor", &input);

    printf("\n[STEP 8] Read one tensor element\n");

    size_t indices[] = {1, 2};
    float value;

    if (tensor_get(&input, indices, &value) != 0)
    {
        printf("Tensor read failed.\n");
        goto cleanup;
    }

    printf("Row 1, column 2: %.6f\n", value);

    printf("\n[STEP 9] Copy the tensor\n");

    if (tensor_copy(&copy, &input) != 0)
    {
        printf("Tensor copy failed.\n");
        goto cleanup;
    }

    printf("Original data address: %p\n", (void *)input.data);
    printf("Copy data address:     %p\n", (void *)copy.data);
    printf("Independent storage:   %s\n",
           input.data != copy.data ? "YES" : "NO");

    print_tensor("Copied tensor", &copy);

    printf("\n[STEP 10] Initialize tensors for arithmetic\n");

    size_t arithmetic_shape[] = {2, 3};

    if (tensor_init(&a, 2, arithmetic_shape) != 0 ||
        tensor_init(&b, 2, arithmetic_shape) != 0)
    {
        printf("Input tensor allocation failed.\n");
        goto cleanup;
    }

    const float values_a[] = {1, 2, 3, 4, 5, 6};
    const float values_b[] = {10, 20, 30, 40, 50, 60};

    if (set_tensor_data(&a, values_a) != 0 ||
        set_tensor_data(&b, values_b) != 0)
    {
        printf("Tensor data initialization failed.\n");
        goto cleanup;
    }

    print_tensor("Tensor A", &a);
    print_tensor("Tensor B", &b);

    printf("\n[STEP 11] Tensor addition\n");

    if (tensor_add(&a, &b, &c) != 0)
    {
        printf("Tensor addition failed.\n");
        goto cleanup;
    }

    print_tensor("A + B", &c);

    printf("\n[STEP 12] Tensor subtraction\n");

    if (tensor_sub(&a, &b, &sub_result) != 0)
    {
        printf("Tensor subtraction failed.\n");
        goto cleanup;
    }

    print_tensor("A - B", &sub_result);

    printf("\n[STEP 13] Elementwise tensor multiplication\n");

    if (tensor_mul(&a, &b, &mul_result) != 0)
    {
        printf("Tensor multiplication failed.\n");
        goto cleanup;
    }

    print_tensor("A * B", &mul_result);

    printf("\n[STEP 14] Scalar multiplication\n");

    if (tensor_scale(&a, 0.5f, &scale_result) != 0)
    {
        printf("Tensor scaling failed.\n");
        goto cleanup;
    }

    print_tensor("A * 0.5", &scale_result);

    status = 0;

    printf("\n[STEP 15] Matrix multiplication\n");

    Tensor ma = {0};
    Tensor mb = {0};
    Tensor mc = {0};

    size_t ma_shape[] = {2, 3};
    size_t mb_shape[] = {3, 2};

    const float ma_values[] = {
        1, 2, 3,
        4, 5, 6};

    const float mb_values[] = {
        7, 8,
        9, 10,
        11, 12};

    if (tensor_init(&ma, 2, ma_shape) != 0 ||
        tensor_init(&mb, 2, mb_shape) != 0)
    {
        printf("Matrix allocation failed.\n");
        tensor_free(&ma);
        tensor_free(&mb);
        goto cleanup;
    }

    for (size_t i = 0; i < ma.size; i++)
        ma.data[i] = ma_values[i];

    for (size_t i = 0; i < mb.size; i++)
        mb.data[i] = mb_values[i];

    print_tensor("Matrix A", &ma);
    print_tensor("Matrix B", &mb);

    if (matmul(&ma, &mb, &mc) != 0)
    {
        printf("Matrix multiplication failed.\n");
        tensor_free(&ma);
        tensor_free(&mb);
        tensor_free(&mc);
        goto cleanup;
    }

    print_tensor("A x B", &mc);

    printf("\nSTEP 16: MATRIX TRANSPOSE\n");

    Tensor original = {0};
    Tensor transposed = {0};

    size_t original_shape[] = {2, 3};

    float original_data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};

    if (tensor_init(&original, 2, original_shape) != 0)
        goto cleanup;

    if (set_tensor_data(&original, original_data) != 0)
        goto cleanup;

    print_tensor("Original matrix", &original);

    if (tensor_transpose(&original, &transposed) != 0)
    {
        tensor_free(&original);
        tensor_free(&transposed);
        goto cleanup;
    }

    print_tensor("Transposed matrix", &transposed);

    printf("\nSTEP 17: SOFTMAX\n");

    size_t vector_shape[] = {3};
    float vector_values[] = {2.0f, 1.0f, 0.0f};

    if (tensor_init(&softmax_input, 1, vector_shape) != 0)
    {
        printf("Softmax input allocation failed.\n");
        goto cleanup;
    }

    if (set_tensor_data(&softmax_input, vector_values) != 0)
    {
        printf("Softmax input initialization failed.\n");
        goto cleanup;
    }

    if (softmax(&softmax_input, 0, &softmax_vector_output) != 0)
    {
        printf("Vector softmax failed.\n");
        goto cleanup;
    }

    print_tensor("Vector softmax", &softmax_vector_output);

    tensor_free(&softmax_input);

    size_t softmax_matrix_shape[] = {2, 3};

    float softmax_matrix_values[] = {
        1.0f, 2.0f, 3.0f,
        1.0f, 1.0f, 1.0f};

    if (tensor_init(&softmax_input, 2, softmax_matrix_shape) != 0)
    {
        printf("Softmax matrix allocation failed.\n");
        goto cleanup;
    }

    if (set_tensor_data(&softmax_input, softmax_matrix_values) != 0)
    {
        printf("Softmax matrix initialization failed.\n");
        goto cleanup;
    }

    if (softmax(&softmax_input, 1, &softmax_matrix_output) != 0)
    {
        printf("Matrix softmax failed.\n");
        goto cleanup;
    }

    print_tensor("Row-wise matrix softmax", &softmax_matrix_output);

cleanup:
    tensor_free(&scale_result);
    tensor_free(&mul_result);
    tensor_free(&sub_result);
    tensor_free(&c);
    tensor_free(&b);
    tensor_free(&a);
    tensor_free(&copy);
    tensor_free(&input);
    embedding_free(&embedding);
    tensor_free(&ma);
    tensor_free(&mb);
    tensor_free(&mc);
    tensor_free(&original);
    tensor_free(&transposed);
    tensor_free(&softmax_input);
    tensor_free(&softmax_vector_output);
    tensor_free(&softmax_matrix_output);

    if (status == 0)
    {
        printf("\n====================================\n");
        printf("       WALKTHROUGH COMPLETE\n");
        printf("====================================\n");
    }

    return status;
}