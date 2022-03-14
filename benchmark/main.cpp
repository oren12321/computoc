#include <benchmark/benchmark.h>

#include <la/la.hpp>

static void BM_Matrix_DefaultCtor(benchmark::State& state)
{
    for (auto _ : state)
    {
        la::matrix<int, 1, 1> m{};
    }
}

BENCHMARK(BM_Matrix_DefaultCtor);

BENCHMARK_MAIN();

