#include <gtest/gtest.h>

#include <computoc/computoc.h>

TEST(Computoc_test, use_all_namespaces)
{
    using namespace computoc;
    SUCCEED();
}