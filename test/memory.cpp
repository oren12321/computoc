#include <gtest/gtest.h>

#include <type_traits>

#include <math/core/memory.h>

// Block tests

TEST(Block_test, is_empty_when_deafult_initalized)
{
    using namespace math::core::memory;

    Block b{};

    EXPECT_EQ(nullptr, b.p);
    EXPECT_EQ(0, b.s);
    EXPECT_TRUE(b.empty());
}

TEST(Block_test, can_be_of_specific_type)
{
    using namespace math::core::memory;

    Typed_block<int> b{};

    EXPECT_EQ(nullptr, b.p);
    EXPECT_EQ(0, b.s);
    EXPECT_TRUE(b.empty());

    bool valid_buffer_type = std::is_same<int, typename std::remove_pointer<decltype(b.p)>::type>();
    EXPECT_TRUE(valid_buffer_type);
}

TEST(Aux_test, an_object_can_be_constructed_at_specified_address)
{
    using namespace math::core::memory::aux;

    struct Test {
        Test(int a = 0, int b = 0) : a_(a), b_(b) {}
        int a_{ 0 };
        int b_{ 0 };
    };

    Test t{};
    EXPECT_EQ(0, t.a_);
    EXPECT_EQ(0, t.b_);

    Test* tp = construct_at(&t, 1, 2);
    EXPECT_EQ(1, t.a_);
    EXPECT_EQ(2, t.b_);
    EXPECT_EQ(1, tp->a_);
    EXPECT_EQ(2, tp->b_);
}
