#include <benchmark/benchmark.h>

#include <math/core/matrix.hpp>

static void BM_Matrix_DefaultCtor(benchmark::State& state)
{
    for (auto _ : state)
    {
        math::core::matrix m(state.range(0), state.range(0), 0);
    }
}

BENCHMARK(BM_Matrix_DefaultCtor)->Arg(1)->RangeMultiplier(2)->Range(2, 1024);

