#include <gtest/gtest.h>

#include <memoc/utils.h>

TEST(Aux_test, an_object_can_be_constructed_and_destructed_at_specified_address)
{
    using namespace memoc::utils;

    struct Test {
        Test() = default;
        Test(int a, int b, bool* constructed, bool* destructed)
            : a_(a), b_(b), constructed_(constructed), destructed_(destructed)
        {
            *constructed_ = true;
        }
        ~Test()
        {
            *destructed_ = true;
        }
        int a_{ 0 };
        int b_{ 0 };
        bool* constructed_{ nullptr };
        bool* destructed_{ nullptr };
    };

    Test t{};
    EXPECT_EQ(0, t.a_);
    EXPECT_EQ(0, t.b_);

    bool constructed = false;
    bool destructed = false;
    Test* tp = construct_at(&t, 1, 2, &constructed, &destructed);
    EXPECT_EQ(1, t.a_);
    EXPECT_EQ(2, t.b_);
    EXPECT_EQ(1, tp->a_);
    EXPECT_EQ(2, tp->b_);
    EXPECT_TRUE(constructed);
    EXPECT_FALSE(destructed);
    destruct_at(&t);
    EXPECT_TRUE(destructed);
}