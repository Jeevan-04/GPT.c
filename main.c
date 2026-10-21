#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "embedding/embedding.h"
#include "rng/rng.h"

#define TEST_VOCAB_SIZE 100
#define TEST_EMBEDDING_DIM 8
#define RANDOM_LIMIT 0.05f

static int test_count = 0;
static int passed_count = 0;

static void report_test(const char *name, int passed)
{
    test_count++;

    if (passed)
    {
        passed_count++;
        printf("[PASS] %s\n", name);
    }
    else
    {
        printf("[FAIL] %s\n", name);
    }
}

static int embeddings_equal(const Embedding *a, const Embedding *b)
{
    if (a == NULL || b == NULL ||
        a->weights == NULL || b->weights == NULL ||
        a->vocab_size != b->vocab_size ||
        a->embedding_dim != b->embedding_dim)
    {
        return 0;
    }

    size_t count = a->vocab_size * a->embedding_dim;

    for (size_t i = 0; i < count; i++)
    {
        if (a->weights[i] != b->weights[i])
            return 0;
    }

    return 1;
}

static int all_weights_equal(const Embedding *embedding, float value)
{
    if (embedding == NULL || embedding->weights == NULL)
        return 0;

    size_t count = embedding->vocab_size * embedding->embedding_dim;

    for (size_t i = 0; i < count; i++)
    {
        if (embedding->weights[i] != value)
            return 0;
    }

    return 1;
}

int main(void)
{
    printf("====================================\n");
    printf("     EMBEDDING MODULE TESTS\n");
    printf("====================================\n\n");

    /* Test 1: Reject a NULL embedding pointer. */
    report_test(
        "Reject NULL embedding during initialization",
        embedding_init(NULL, TEST_VOCAB_SIZE, TEST_EMBEDDING_DIM) != 0);

    /* Test 2: Reject zero vocabulary size. */
    Embedding invalid_vocab = {0};

    report_test(
        "Reject zero vocabulary size",
        embedding_init(&invalid_vocab, 0, TEST_EMBEDDING_DIM) != 0);

    embedding_free(&invalid_vocab);

    /* Test 3: Reject zero embedding dimension. */
    Embedding invalid_dim = {0};

    report_test(
        "Reject zero embedding dimension",
        embedding_init(&invalid_dim, TEST_VOCAB_SIZE, 0) != 0);

    embedding_free(&invalid_dim);

    /* Test 4: Reject multiplication overflow. */
    Embedding overflow_embedding = {0};

    report_test(
        "Reject overflowing embedding dimensions",
        embedding_init(&overflow_embedding, SIZE_MAX, 2) != 0);

    embedding_free(&overflow_embedding);

    /* Test 5: Initialize a valid embedding. */
    Embedding embedding = {0};

    int init_result = embedding_init(
        &embedding,
        TEST_VOCAB_SIZE,
        TEST_EMBEDDING_DIM);

    report_test("Initialize valid embedding", init_result == 0);

    if (init_result != 0)
    {
        printf("\nCannot continue: embedding allocation failed.\n");
        return 1;
    }

    size_t total_elements =
        embedding.vocab_size * embedding.embedding_dim;

    /* Test 6: Verify the dimensions. */
    report_test(
        "Store correct vocabulary size and embedding dimension",
        embedding.vocab_size == TEST_VOCAB_SIZE &&
            embedding.embedding_dim == TEST_EMBEDDING_DIM);

    /* Test 7: Reject an invalid token ID. */
    report_test(
        "Reject token ID equal to vocabulary size",
        embedding_get(&embedding, TEST_VOCAB_SIZE) == NULL);

    /* Test 8: Reject a NULL embedding during lookup. */
    report_test(
        "Reject NULL embedding during lookup",
        embedding_get(NULL, 0) == NULL);

    /* Test 9: Reject lookup when weights are NULL. */
    Embedding empty_embedding = {0};

    report_test(
        "Reject lookup on an empty embedding",
        embedding_get(&empty_embedding, 0) == NULL);

    /* Test 10: Zero all embedding weights. */
    embedding_zero(&embedding);

    report_test(
        "Zero every embedding weight",
        all_weights_equal(&embedding, 0.0f));

    /* Test 11: Zeroing a NULL embedding should be safe. */
    embedding_zero(NULL);
    report_test("Safely handle NULL in embedding_zero", 1);

    /* Test 12: Initialize the RNG. */
    RNG rng;
    rng_init(&rng, 123);

    /* Test 13: Randomize embedding weights. */
    embedding_randomize(&embedding, RANDOM_LIMIT, &rng);

    int all_in_range = 1;
    float min_weight = embedding.weights[0];
    float max_weight = embedding.weights[0];

    for (size_t i = 0; i < total_elements; i++)
    {
        float weight = embedding.weights[i];

        if (!isfinite(weight) ||
            weight < -RANDOM_LIMIT ||
            weight > RANDOM_LIMIT)
        {
            all_in_range = 0;
        }

        if (weight < min_weight)
            min_weight = weight;

        if (weight > max_weight)
            max_weight = weight;
    }

    report_test("Randomize embedding weights within range", all_in_range);

    printf("    Minimum weight: %.8f\n", min_weight);
    printf("    Maximum weight: %.8f\n", max_weight);

    /* Test 14: Invalid randomization arguments should not change weights. */
    Embedding before_invalid_randomization = {0};

    int copy_init_result = embedding_init(
        &before_invalid_randomization,
        TEST_VOCAB_SIZE,
        TEST_EMBEDDING_DIM);

    if (copy_init_result != 0)
    {
        report_test("Invalid randomization arguments preserve weights", 0);
    }
    else
    {
        for (size_t i = 0; i < total_elements; i++)
        {
            before_invalid_randomization.weights[i] =
                embedding.weights[i];
        }

        embedding_randomize(&embedding, 0.0f, &rng);

        report_test(
            "Zero randomization limit preserves weights",
            embeddings_equal(&embedding, &before_invalid_randomization));

        embedding_free(&before_invalid_randomization);
    }

    /* Test 15: A NULL RNG should not change weights. */
    Embedding before_null_rng = {0};

    int null_rng_copy_result = embedding_init(
        &before_null_rng,
        TEST_VOCAB_SIZE,
        TEST_EMBEDDING_DIM);

    if (null_rng_copy_result != 0)
    {
        report_test("NULL RNG preserves embedding weights", 0);
    }
    else
    {
        for (size_t i = 0; i < total_elements; i++)
            before_null_rng.weights[i] = embedding.weights[i];

        embedding_randomize(&embedding, RANDOM_LIMIT, NULL);

        report_test(
            "NULL RNG preserves embedding weights",
            embeddings_equal(&embedding, &before_null_rng));

        embedding_free(&before_null_rng);
    }

    /* Test 16: Same seed should produce identical embeddings. */
    Embedding embedding_a = {0};
    Embedding embedding_b = {0};

    int init_a = embedding_init(
        &embedding_a,
        TEST_VOCAB_SIZE,
        TEST_EMBEDDING_DIM);

    int init_b = embedding_init(
        &embedding_b,
        TEST_VOCAB_SIZE,
        TEST_EMBEDDING_DIM);

    if (init_a != 0 || init_b != 0)
    {
        report_test("Same seed produces identical embeddings", 0);
    }
    else
    {
        RNG rng_a;
        RNG rng_b;

        rng_init(&rng_a, 456);
        rng_init(&rng_b, 456);

        embedding_randomize(&embedding_a, RANDOM_LIMIT, &rng_a);
        embedding_randomize(&embedding_b, RANDOM_LIMIT, &rng_b);

        report_test(
            "Same seed produces identical embeddings",
            embeddings_equal(&embedding_a, &embedding_b));
    }

    embedding_free(&embedding_a);
    embedding_free(&embedding_b);

    /* Test 17: Different seeds should produce different embeddings. */
    Embedding embedding_c = {0};
    Embedding embedding_d = {0};

    int init_c = embedding_init(
        &embedding_c,
        TEST_VOCAB_SIZE,
        TEST_EMBEDDING_DIM);

    int init_d = embedding_init(
        &embedding_d,
        TEST_VOCAB_SIZE,
        TEST_EMBEDDING_DIM);

    if (init_c != 0 || init_d != 0)
    {
        report_test("Different seeds produce different embeddings", 0);
    }
    else
    {
        RNG rng_c;
        RNG rng_d;

        rng_init(&rng_c, 111);
        rng_init(&rng_d, 222);

        embedding_randomize(&embedding_c, RANDOM_LIMIT, &rng_c);
        embedding_randomize(&embedding_d, RANDOM_LIMIT, &rng_d);

        report_test(
            "Different seeds produce different embeddings",
            !embeddings_equal(&embedding_c, &embedding_d));
    }

    embedding_free(&embedding_c);
    embedding_free(&embedding_d);

    /* Test 18: Retrieve a valid token vector. */
    float *token_vector = embedding_get(&embedding, 5);

    report_test("Retrieve valid token embedding", token_vector != NULL);

    /* Test 19: Verify the returned vector has the expected values. */
    int vector_valid = token_vector != NULL;

    if (vector_valid)
    {
        for (size_t i = 0; i < embedding.embedding_dim; i++)
        {
            if (!isfinite(token_vector[i]) ||
                token_vector[i] < -RANDOM_LIMIT ||
                token_vector[i] > RANDOM_LIMIT)
            {
                vector_valid = 0;
                break;
            }
        }
    }

    report_test("Retrieved token vector contains valid weights", vector_valid);

    /* Test 20: Free an empty embedding safely. */
    Embedding empty_to_free = {0};
    embedding_free(&empty_to_free);

    report_test("Safely free an empty embedding", 1);

    /* Display the main embedding before freeing it. */
    printf("\n====================================\n");
    printf("     EMBEDDING SUMMARY\n");
    printf("====================================\n");

    printf("Vocabulary size: %zu\n", embedding.vocab_size);
    printf("Embedding dimension: %zu\n", embedding.embedding_dim);
    printf("Total parameters: %zu\n", total_elements);

    printf("Token 5 embedding:\n");

    token_vector = embedding_get(&embedding, 5);

    if (token_vector != NULL)
    {
        for (size_t i = 0; i < embedding.embedding_dim; i++)
            printf("%.8f ", token_vector[i]);

        printf("\n");
    }

    /* Test 21: Free the main embedding and verify its reset state. */
    embedding_free(&embedding);

    report_test(
        "Free embedding and reset structure",
        embedding.weights == NULL &&
            embedding.vocab_size == 0 &&
            embedding.embedding_dim == 0);

    /* Test 22: Freeing the already-freed structure is safe. */
    embedding_free(&embedding);
    report_test("Safely free an embedding twice", 1);

    printf("\n====================================\n");
    printf("Tests passed: %d / %d\n", passed_count, test_count);
    printf("====================================\n");

    return passed_count == test_count ? 0 : 1;
}