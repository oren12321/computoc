#include <gtest/gtest.h>

#include <math/core/matrix.h>

TEST(Matrix_test, can_be_initialized_with_valid_size_and_data)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int data[] = { 0 };
    EXPECT_NO_THROW((Integer_matrix{ {1, 1}, data }));

    EXPECT_THROW((Integer_matrix{ {0, 0}, data }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 0}, data }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {0, 1}, data }), std::invalid_argument);

    const int* null_data = nullptr;
    EXPECT_THROW((Integer_matrix{ {1, 1}, null_data }), std::invalid_argument);
}

TEST(Matrix_test, can_be_initialized_with_valid_size_and_value)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int value{ 0 };
    EXPECT_NO_THROW((Integer_matrix{ {1, 1}, value }));

    EXPECT_THROW((Integer_matrix{ {0, 0}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 0}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {0, 1}, value }), std::invalid_argument);
}

TEST(Matrix_test, can_return_its_size)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int value{ 0 };
    const std::size_t n = 1;
    const std::size_t m = 2;
    Integer_matrix mat{ {n, m}, value };

    math::core::types::Dimensions d{ mat.dimensions() };
    EXPECT_EQ(n, d.n);
    EXPECT_EQ(m, d.m);
}

TEST(Matrix_test, have_read_write_access_to_its_cells)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int data[] = {
        1, 2, 3,
        4, 5, 6 };
    const std::size_t n = 2;
    const std::size_t m = 3;
    Integer_matrix mat{ {n, m}, data };

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            EXPECT_EQ(mat(i, j), data[i * m + j]);
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            mat(i, j) = 0;
            EXPECT_EQ(mat(i, j), 0);
        }
    }

    EXPECT_THROW(mat(n, 0), std::out_of_range);
    EXPECT_THROW(mat(0, m), std::out_of_range);
    EXPECT_THROW(mat(n, m), std::out_of_range);
}

TEST(Matrix_test, can_be_compared_with_another_matrix)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int data1[] = {
        1, 2, 3,
        4, 5, 6 };
    const std::size_t n1 = 2;
    const std::size_t m1 = 3;
    Integer_matrix mat1{ {n1, m1}, data1 };
    Integer_matrix mat2{ {n1, m1}, data1 };

    EXPECT_EQ(mat1, mat2);

    const int data2[] = {
        1, 2, 3,
        4, 5, 5 };
    Integer_matrix mat3{ {m1, n1}, data1 };
    Integer_matrix mat4{ {n1, m1}, data2 };

    EXPECT_NE(mat1, mat3);
    EXPECT_NE(mat1, mat4);
}

TEST(Matrix_test, can_return_slice)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int data[] = {
    1, 2, 3,
    4, 5, 6 };
    const std::size_t n = 2;
    const std::size_t m = 3;
    Integer_matrix mat{ {n, m}, data };

    const int sdata1[] = { 1, 2, 3 };
    const std::size_t sn1 = 1;
    const std::size_t sm1 = 3;
    Integer_matrix smat1{ {sn1, sm1}, sdata1 };
    EXPECT_EQ(mat.get_slice(0, 0, smat1.dimensions()), smat1);

    const int sdata2[] = {
        2,
        5 };
    const std::size_t sn2 = 2;
    const std::size_t sm2 = 1;
    Integer_matrix smat2{ {sn2, sm2}, sdata2 };
    EXPECT_EQ(mat.get_slice(0, 1, smat2.dimensions()), smat2);

    EXPECT_THROW(mat.get_slice(n, 0, { 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat.get_slice(0, m, { 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat.get_slice(n, m, { 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat.get_slice(0, 0, { n + 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat.get_slice(0, 0, { 1, m + 1 }), std::out_of_range);
    EXPECT_THROW(mat.get_slice(0, 0, { n + 1, m + 1 }), std::out_of_range);
}

TEST(Matrix_test, can_write_into_slice)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int data[] = {
    1, 2, 3,
    4, 5, 6 };
    const std::size_t n = 2;
    const std::size_t m = 3;
    Integer_matrix mat{ {n, m}, data };

    const int sdata[] = {
        0,
        0 };
    const std::size_t sn = 2;
    const std::size_t sm = 1;
    Integer_matrix smat{ {sn, sm}, sdata };
    mat.set_slice(0, 1, smat);

    const int rdata[] = {
        1, 0, 3,
        4, 0, 6 };
    const std::size_t rn = 2;
    const std::size_t rm = 3;
    Integer_matrix rmat{ {rn, m}, rdata };

    EXPECT_EQ(mat, rmat);

    EXPECT_THROW(smat.set_slice(0, 0, mat), std::out_of_range);
}

TEST(Matrix_test, can_be_added_with_another_matrix)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int data[] = {
        1, 2, 3,
        4, 5, 6 };
    const std::size_t n = 2;
    const std::size_t m = 3;
    Integer_matrix mat{ {n, m}, data };

    mat += mat + mat;

    const int rdata[] = {
        3, 6, 9,
        12, 15, 18 };
    Integer_matrix rmat{ {n, m}, rdata };

    EXPECT_EQ(mat, rmat);
}

TEST(Matrix_test, can_be_subtracted_from_another_matrix)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int data[] = {
        1, 2, 3,
        4, 5, 6 };
    const std::size_t n = 2;
    const std::size_t m = 3;
    Integer_matrix mat{ {n, m}, data };
    Integer_matrix rmat{ mat };

    mat -= mat - mat;

    EXPECT_EQ(mat, rmat);
}

TEST(Matrix_test, can_be_multiplied_by_a_constant)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int data[] = {
        1, 2, 3,
        4, 5, 6 };
    const std::size_t n = 2;
    const std::size_t m = 3;
    Integer_matrix mat{ {n, m}, data };

    const int rdata[] = {
        2, 4, 6,
        8, 10, 12 };
    Integer_matrix rmat{ {n, m}, rdata };

    EXPECT_EQ(mat * 2, rmat);
    EXPECT_EQ(2 * mat, rmat);

    mat *= 2;
    EXPECT_EQ(mat, rmat);
}

TEST(Matrix_test, can_be_multiplied_by_another_matrix)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int data1[] = {
        1, 2, 3,
        4, 5, 6 };
    const std::size_t n1 = 2;
    const std::size_t m1 = 3;
    Integer_matrix mat1{ {n1, m1}, data1 };

    const int data2[] = {
        1, 4,
        2, 5,
        3, 6 };
    const std::size_t n2 = 3;
    const std::size_t m2 = 2;
    Integer_matrix mat2{ {n2, m2}, data2 };

    const int rdata[] = {
        14, 32,
        32, 77};
    const std::size_t rn = 2;
    const std::size_t rm = 2;
    Integer_matrix rmat{ {rn, rm}, rdata };

    EXPECT_EQ(mat1 * mat2, rmat);
    
    mat1 *= mat2;
    EXPECT_EQ(mat1, rmat);
}

TEST(Matrix_test, can_be_transposed)
{
    using Integer_matrix = math::core::types::Matrix<int>;

    const int data[] = {
    1, 2, 3,
    4, 5, 6 };
    const std::size_t n = 2;
    const std::size_t m = 3;
    Integer_matrix mat{ {n, m}, data };

    const int rdata[] = {
        1, 4,
        2, 5,
        3, 6 };
    const std::size_t rn = 3;
    const std::size_t rm = 2;
    Integer_matrix rmat{ {rn, rm}, rdata };

    EXPECT_EQ(mat.transposed(), rmat);

    mat.transpose();
    EXPECT_EQ(mat, rmat);
}

