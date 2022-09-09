#include <gtest/gtest.h>

#include <computoc/computoc.h>

TEST(Memoc_test, use_all_namespaces)
{
    using namespace computoc;
    using namespace computoc::algorithms;
    SUCCEED();
}