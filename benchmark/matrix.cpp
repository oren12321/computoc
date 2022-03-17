#include <benchmark/benchmark.h>

#include <math/core/matrix.hpp>

static void BM_Matrix_DefaultCtor(benchmark::State& state)
{
    for (auto _ : state)
    {
        math::core::matrix m(1, 1, 0);
    }
}

BENCHMARK(BM_Matrix_DefaultCtor);

