#include <gtest/gtest.h>

#include <computoc/linear_algebra.h>
#include <computoc/matrix.h>

TEST(LA_test, matrix_have_minor)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20 };
    computoc::Dims dims{ 4, 5 };
    Integer_matrix mat{ dims, data };

    computoc::Inds pivot{ 2, 1 };
    Integer_matrix mmat{ computoc::minor(mat, pivot) };

    const int rdata[] = {
        1, 3, 4, 5,
        6, 8, 9, 10,
        16, 18, 19, 20 };
    const std::size_t rn = 3;
    const std::size_t rm = 4;
    Integer_matrix rmat{ {rn, rm}, rdata };

    EXPECT_EQ(rmat, mmat);
    EXPECT_THROW(computoc::minor(Integer_matrix{}, computoc::Inds{}), std::invalid_argument);
    EXPECT_THROW(computoc::minor(Integer_matrix{ {1, 1, 2} }, computoc::Inds{}), std::invalid_argument);
    EXPECT_THROW(computoc::minor(mat, { 0, dims.m }), std::out_of_range);
    EXPECT_THROW(computoc::minor(mat, { dims.n, 0 }), std::out_of_range);
}



