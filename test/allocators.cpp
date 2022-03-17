#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <array>

#include <math/core/allocators.h>

// Block tests

TEST(Block_test, is_empty_when_deafult_initalized)
{
    using namespace math::core::allocators;

    Block b{};

    EXPECT_EQ(nullptr, b.p);
    EXPECT_EQ(0, b.s);
    EXPECT_TRUE(b.empty());
}

// Malloc_allocator tests

class Malloc_allocator_test : public ::testing::Test {
protected:
    using Allocator = math::core::allocators::Malloc_allocator;
    std::unique_ptr<Allocator> allocator_ = std::make_unique<Allocator>();
};

TEST_F(Malloc_allocator_test, not_owns_an_empty_block)
{
    using namespace math::core::allocators;

    EXPECT_FALSE(allocator_->owns(Block{}));
}

TEST_F(Malloc_allocator_test, allocates_and_deallocates_memory_successfully)
{
    using namespace math::core::allocators;

    const Block::Size_type s{ 1 };

    Block b = allocator_->allocate(s);
    EXPECT_NE(nullptr, b.p);
    EXPECT_EQ(1, b.s);

    EXPECT_TRUE(allocator_->owns(b));

    allocator_->deallocate(&b);
    EXPECT_TRUE(b.empty());
}

// Stack_allocator tests

class Stack_allocator_test : public ::testing::Test {
protected:
    static constexpr std::size_t size_ = 16;

    using Allocator = math::core::allocators::Stack_allocator<size_>;
    std::unique_ptr<Allocator> allocator_ = std::make_unique<Allocator>();
};

TEST_F(Stack_allocator_test, not_owns_an_empty_block)
{
    using namespace math::core::allocators;

    EXPECT_FALSE(allocator_->owns(Block{}));
}

TEST_F(Stack_allocator_test, allocates_and_deallocates_an_arbitrary_in_range_memory_successfully)
{
    using namespace math::core::allocators;

    const Block::Size_type size_in_range{ size_ / 2 };

    Block b = allocator_->allocate(size_in_range);
    EXPECT_NE(nullptr, b.p);
    EXPECT_EQ(size_in_range, b.s);

    EXPECT_TRUE(allocator_->owns(b));

    allocator_->deallocate(&b);
    EXPECT_TRUE(b.empty());
}

TEST_F(Stack_allocator_test, allocates_and_deallocates_all_available_memory_successfully)
{
    using namespace math::core::allocators;

    Block b = allocator_->allocate(size_);
    EXPECT_NE(nullptr, b.p);
    EXPECT_EQ(size_, b.s);

    EXPECT_TRUE(allocator_->owns(b));

    allocator_->deallocate(&b);
    EXPECT_TRUE(b.empty());
}

TEST_F(Stack_allocator_test, reuses_the_same_memory_if_deallocating_between_two_allocations)
{
    using namespace math::core::allocators;

    const Block::Size_type size_in_range{ size_ / 2 };

    Block b1 = allocator_->allocate(size_in_range);
    EXPECT_NE(nullptr, b1.p);
    EXPECT_EQ(size_in_range, b1.s);
    EXPECT_TRUE(allocator_->owns(b1));

    Block b1_copy{ b1 };

    allocator_->deallocate(&b1);
    EXPECT_TRUE(b1.empty());

    Block b2 = allocator_->allocate(size_in_range);
    EXPECT_NE(nullptr, b2.p);
    EXPECT_EQ(size_in_range, b2.s);
    EXPECT_TRUE(allocator_->owns(b2));
    EXPECT_EQ(b1_copy.p, b2.p);
    EXPECT_EQ(b1_copy.s, b2.s);

    allocator_->deallocate(&b2);
    EXPECT_TRUE(b2.empty());
}

TEST_F(Stack_allocator_test, fails_to_allocate_memory_bigger_than_memory_size)
{
    using namespace math::core::allocators;

    Block b = allocator_->allocate(size_ + 1);
    EXPECT_TRUE(b.empty());
}

// Free_list_allocator tests

class Free_list_allocator_test : public ::testing::Test {
protected:
    static constexpr std::size_t min_size_ = 16;
    static constexpr std::size_t max_size_ = 32;
    static constexpr std::size_t max_list_size_ = 2;
    using Parent = math::core::allocators::Malloc_allocator;

    using Allocator = math::core::allocators::Free_list_allocator<Parent, min_size_, max_size_, max_list_size_>;
    std::unique_ptr<Allocator> allocator_ = std::make_unique<Allocator>();
};

TEST_F(Free_list_allocator_test, not_owns_an_empty_block)
{
    using namespace math::core::allocators;

    EXPECT_FALSE(allocator_->owns(Block{}));
}

TEST_F(Free_list_allocator_test, allocates_and_deallocates_an_arbitrary_not_in_range_memory_successfully_using_parent_allocator)
{
    using namespace math::core::allocators;

    const Block::Size_type size_out_of_range{ max_size_ + 1 };

    Block b = allocator_->allocate(size_out_of_range);
    EXPECT_NE(nullptr, b.p);
    EXPECT_EQ(size_out_of_range, b.s);

    EXPECT_TRUE(allocator_->owns(b));

    allocator_->deallocate(&b);
    EXPECT_TRUE(b.empty());
}

TEST_F(Free_list_allocator_test, reuses_the_same_memory_if_deallocating_in_memory_range)
{
    using namespace math::core::allocators;

    const Block::Size_type size_in_range{ min_size_ + (max_size_ - min_size_) / 2 };

    std::array<Block, max_list_size_> saved_blocks{};

    for (int i = 0; i < max_list_size_; ++i)
    {
        Block b = allocator_->allocate(size_in_range);
        EXPECT_NE(nullptr, b.p);
        EXPECT_EQ(size_in_range, b.s);
        EXPECT_TRUE(allocator_->owns(b));

        saved_blocks[i] = b;
    }

    for (std::array<Block, max_list_size_>::const_reverse_iterator it = saved_blocks.rbegin(); it != saved_blocks.rend(); ++it)
    {
        Block b_copy{ *it };
        allocator_->deallocate(&b_copy);
        EXPECT_TRUE(b_copy.empty());
    }

    for (const auto& saved_block : saved_blocks)
    {
        Block b = allocator_->allocate(size_in_range);
        EXPECT_NE(nullptr, b.p);
        EXPECT_EQ(size_in_range, b.s);
        EXPECT_TRUE(allocator_->owns(b));
        EXPECT_EQ(saved_block.p, b.p);
        EXPECT_EQ(saved_block.s, b.s);
    }

    // Test cleanup since free list does not actually release memory
    Parent p{};
    for (auto& saved_block : saved_blocks)
    {
        p.deallocate(&saved_block);
        EXPECT_TRUE(saved_block.empty());
    }
}
