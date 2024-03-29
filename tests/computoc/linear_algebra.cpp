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

        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,
        26, 27, 28, 29, 30 };
    computoc::Dims dims{ 3, 5, 2 };
    Integer_matrix mat{ dims, data };

    Integer_matrix mmat1{ computoc::excluded(mat, {0, 0}) };
    const int rdata1[] = {
        7, 8, 9, 10,
        12, 13, 14, 15,
        22, 23, 24, 25,
        27, 28, 29, 30 };
    computoc::Dims rdims1{ 2, 4, 2 };
    Integer_matrix rmat1{ rdims1, rdata1 };
    EXPECT_EQ(rmat1, mmat1);

    Integer_matrix mmat2{ computoc::excluded(mat, {0, dims.m - 1}) };
    const int rdata2[] = {
        6, 7, 8, 9,
        11, 12, 13, 14,
        21, 22, 23, 24,
        26, 27, 28, 29 };
    computoc::Dims rdims2{ 2, 4, 2 };
    Integer_matrix rmat2{ rdims2, rdata2 };
    EXPECT_EQ(rmat2, mmat2);

    Integer_matrix mmat3{ computoc::excluded(mat, {dims.n - 1, 0}) };
    const int rdata3[] = {
        2, 3, 4, 5,
        7, 8, 9, 10,
        17, 18, 19, 20,
        22, 23, 24, 25 };
    computoc::Dims rdims3{ 2, 4, 2 };
    Integer_matrix rmat3{ rdims3, rdata3 };
    EXPECT_EQ(rmat3, mmat3);

    Integer_matrix mmat4{ computoc::excluded(mat, {dims.n - 1, dims.m - 1}) };
    const int rdata4[] = {
        1, 2, 3, 4,
        6, 7, 8, 9,
        16, 17, 18, 19,
        21, 22, 23, 24 };
    computoc::Dims rdims4{ 2, 4, 2 };
    Integer_matrix rmat4{ rdims4, rdata4 };
    EXPECT_EQ(rmat4, mmat4);

    Integer_matrix mmat5{ computoc::excluded(mat, {0, 2}) };
    const int rdata5[] = {
         6, 7, 9, 10,
        11, 12, 14, 15,
        21, 22, 24, 25,
        26, 27, 29, 30 };
    computoc::Dims rdims5{ 2, 4, 2 };
    Integer_matrix rmat5{ rdims5, rdata5 };
    EXPECT_EQ(rmat5, mmat5);

    Integer_matrix mmat6{ computoc::excluded(mat, {1, 0}) };
    const int rdata6[] = {
        2, 3, 4, 5,
        12, 13, 14, 15,
        17, 18, 19, 20,
        27, 28, 29, 30 };
    computoc::Dims rdims6{ 2, 4, 2 };
    Integer_matrix rmat6{ rdims6, rdata6 };
    EXPECT_EQ(rmat6, mmat6);

    Integer_matrix mmat7{ computoc::excluded(mat, {1, 2}) };
    const int rdata7[] = {
        1, 2, 4, 5,
        11, 12, 14, 15,
        16, 17, 19, 20,
        26, 27, 29, 30 };
    computoc::Dims rdims7{ 2, 4, 2 };
    Integer_matrix rmat7{ rdims7, rdata7 };
    EXPECT_EQ(rmat7, mmat7);

    Integer_matrix mmat8{ computoc::excluded(mat, {1, dims.m - 1}) };
    const int rdata8[] = {
        1, 2, 3, 4,
        11, 12, 13, 14,
        16, 17, 18, 19,
        26, 27, 28, 29 };
    computoc::Dims rdims8{ 2, 4, 2 };
    Integer_matrix rmat8{ rdims8, rdata8 };
    EXPECT_EQ(rmat8, mmat8);

    Integer_matrix mmat9{ computoc::excluded(mat, {dims.n - 1, 2}) };
    const int rdata9[] = {
        1, 2, 4, 5,
        6, 7, 9, 10,
        16, 17, 19, 20,
        21, 22, 24, 25 };
    computoc::Dims rdims9{ 2, 4, 2 };
    Integer_matrix rmat9{ rdims9, rdata9 };
    EXPECT_EQ(rmat9, mmat9);

    EXPECT_THROW(computoc::excluded(Integer_matrix{}, computoc::Inds{}), std::invalid_argument);
    EXPECT_THROW(computoc::excluded(Integer_matrix{ {1, 1, 2} }, computoc::Inds{}), std::invalid_argument);
    EXPECT_THROW(computoc::excluded(Integer_matrix{ {1, 2, 2} }, computoc::Inds{}), std::invalid_argument);
    EXPECT_THROW(computoc::excluded(Integer_matrix{ {2, 1, 2} }, computoc::Inds{}), std::invalid_argument);
    EXPECT_THROW(computoc::excluded(mat, { 0, dims.m }), std::out_of_range);
    EXPECT_THROW(computoc::excluded(mat, { dims.n, 0 }), std::out_of_range);
}

TEST(LA_test, matrix_have_positive)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
    1, 2, 3,
    4, 5, 6 };
    computoc::Dims dims{ 1, 3, 2 };
    Integer_matrix mat{ dims, data };

    EXPECT_EQ(mat, +mat);
}

TEST(LA_test, can_negate_matrix)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
    1, 2, 3,
    4, 5, 6 };
    computoc::Dims dims{ 1, 3, 2 };
    Integer_matrix mat{ dims, data };

    const int rdata[] = {
    -1, -2, -3,
    -4, -5, -6 };
    computoc::Dims rdims{ 1, 3, 2 };
    Integer_matrix rmat{ rdims, rdata };

    EXPECT_EQ(rmat, -mat);
}

TEST(LA_test, matrix_can_be_added_with_another_matrix)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2, 3,
        4, 5, 6 };
    Integer_matrix mat{ {1, 3, 2}, data };

    mat += mat + mat;

    const int rdata[] = {
        3, 6, 9,
        12, 15, 18 };
    Integer_matrix rmat{ {1, 3, 2}, rdata };

    EXPECT_EQ(mat, rmat);

    EXPECT_THROW((mat + Integer_matrix{ {1, 1}, 0 }), std::invalid_argument);
}

TEST(LA_test, matrix_can_be_subtracted_from_another_matrix)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2, 3,
        4, 5, 6 };
    Integer_matrix mat{ {1, 3, 2}, data };
    Integer_matrix rmat{ mat };

    mat -= mat - mat;

    EXPECT_EQ(mat, rmat);

    EXPECT_THROW((mat - Integer_matrix{ {1, 1}, 0 }), std::invalid_argument);
}

TEST(LA_test, matrix_can_be_multiplied_by_a_constant)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2, 3,
        4, 5, 6 };
    Integer_matrix mat{ {1, 3, 2}, data };

    const int rdata[] = {
        2, 4, 6,
        8, 10, 12 };
    Integer_matrix rmat{ {1, 3, 2}, rdata };

    EXPECT_EQ(mat * 2, rmat);
    EXPECT_EQ(2 * mat, rmat);

    mat *= 2;
    EXPECT_EQ(mat, rmat);
}

TEST(LA_test, matrix_can_be_multiplied_by_another_matrix)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data1[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12};
    Integer_matrix mat1{ {2, 3, 2}, data1 };

    const int data2[] = {
        1, 4,
        2, 5,
        3, 6,
        7, 10,
        8, 11,
        9, 12};
    Integer_matrix mat2{ {3, 2, 2}, data2 };

    const int rdata[] = {
        14, 32,
        32, 77,
        194, 266,
        266, 365};
    Integer_matrix rmat{ {2, 2, 2}, rdata };

    EXPECT_EQ(mat1 * mat2, rmat);

    mat1 *= mat2;
    EXPECT_EQ(mat1, rmat);

    EXPECT_THROW(mat2 * mat2, std::invalid_argument);
}

TEST(LA_test, matrix_can_be_transposed)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12};
    Integer_matrix mat{ {2, 3, 2}, data };

    const int rdata[] = {
        1, 4,
        2, 5,
        3, 6,
        7, 10,
        8, 11,
        9, 12 };
    Integer_matrix rmat{ {3, 2, 2}, rdata };

    EXPECT_EQ(transposed(mat), rmat);
}

TEST(LA_test, matrix_have_determinant_if_squared)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32 };
    Integer_matrix mat{ {4, 4, 2}, data };

    const int rdata1[] = {
        1,
        17 };
    Integer_matrix rmat1{ {1, 1, 2}, rdata1 };
    EXPECT_EQ(rmat1, computoc::determinant(mat({ 0, 0, 0 }, {1, 1, 2})));

    const int rdata2[] = {
        -4,
        -4 };
    Integer_matrix rmat2{ {1, 1, 2}, rdata2 };
    EXPECT_EQ(rmat2, computoc::determinant(mat({ 0, 0, 0 }, { 2, 2, 2 })));

    const int rdata3[] = {
        0,
        0};
    Integer_matrix rmat3{ {1, 1, 2}, rdata3 };
    EXPECT_EQ(rmat3, computoc::determinant(mat));

    EXPECT_THROW(computoc::determinant(Integer_matrix{}), std::invalid_argument);

    EXPECT_THROW(computoc::determinant(Integer_matrix{ {1, 2}, 0 }), std::invalid_argument);
    EXPECT_THROW(computoc::determinant(Integer_matrix{ {2, 1}, 0 }), std::invalid_argument);
}

TEST(LA_test, matrix_have_inverse_if_squared_and_no_zero_determinant)
{
    using Double_matrix = computoc::Matrix<double>;

    const double data[] = {
        1, 2,
        5, 6,
        9, 10,
        13, 14 };
    Double_matrix mat{ {2, 2, 2}, data };

    const double inv_data1[] = {
        -1.5, 0.5,
        1.25, -0.25,
        -3.5, 2.5,
        3.25, -2.25 };
    Double_matrix inv_mat1{ {2, 2, 2}, inv_data1 };
    EXPECT_EQ(inv_mat1, computoc::inversed(mat));

    const double unit_data2[] = {
        1, 0,
        0, 1,
        1, 0,
        0, 1 };
    Double_matrix unit_mat2{ {2, 2, 2}, unit_data2 };
    EXPECT_EQ(unit_mat2, inv_mat1 * mat);

    EXPECT_THROW(computoc::inversed(Double_matrix{ {1, 2}, 0.0 }), std::invalid_argument);
    EXPECT_THROW(computoc::inversed(Double_matrix{ {2, 1}, 0.0 }), std::invalid_argument);

    const double data2[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16 };
    Double_matrix mat2{ {4, 4}, data2 };
    EXPECT_THROW(computoc::inversed(mat2), std::invalid_argument);
}

TEST(LA_test, can_add_multiply_and_swap_matrix_rows)
{
    using Double_matrix = computoc::Matrix<double>;

    const double data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16 };
    Double_matrix mat{ {2, 4, 2}, data };

    computoc::multiply_row(mat, 0, 2.0);
    const double rdata1[] = {
        2, 4, 6, 8,
        5, 6, 7, 8,
        18, 20, 22, 24,
        13, 14, 15, 16 };
    Double_matrix rmat1{ {2, 4, 2}, rdata1 };
    EXPECT_EQ(rmat1, mat);

    computoc::add_to_row(mat, 0, 1, 0.5);
    const double rdata2[] = {
        2, 4, 6, 8,
        6, 8, 10, 12,
        18, 20, 22, 24,
        22, 24, 26, 28 };
    Double_matrix rmat2{ {2, 4, 2}, rdata2 };
    EXPECT_EQ(rmat2, mat);

    computoc::swap_rows(mat, 0, 1);
    const double rdata3[] = {
        6, 8, 10, 12,
        2, 4, 6, 8,
        22, 24, 26, 28,
        18, 20, 22, 24 };
    Double_matrix rmat3{ {2, 4, 2}, rdata3 };
    EXPECT_EQ(rmat3, mat);

    EXPECT_THROW(computoc::multiply_row(mat, 3, 1.0), std::out_of_range);

    EXPECT_THROW(computoc::add_to_row(mat, 3, 0, 1.0), std::out_of_range);
    EXPECT_THROW(computoc::add_to_row(mat, 0, 3, 1.0), std::out_of_range);

    EXPECT_THROW(computoc::swap_rows(mat, 3, 0), std::out_of_range);
    EXPECT_THROW(computoc::swap_rows(mat, 0, 3), std::out_of_range);
}

TEST(LA_test, matrix_have_reduced_row_echelon_form)
{
    using Double_matrix = computoc::Matrix<double>;

    const double data[] = {
        0.25, 0.5, 1, 5.75,
        1, 1, 1, 7,
        4, 2, 1, 2,
    
        6, 10, 4, 22,
        1, 1, 1, 3,
        2, 4, 1, 8 };
    Double_matrix mat{ {3, 4, 2}, data };

    const double rdata[] = {
        1, 0, 0, -5,
        0, 1, 0, 10,
        0, 0, 1, 2,
    
        1, 0, 1.5, 2,
        0, 1, -0.5, 1,
        0, 0, 0, 0 };
    Double_matrix rmat{ {3, 4 ,2}, rdata };

    EXPECT_EQ(rmat, computoc::reduced_row_echelon_form(mat));
}
