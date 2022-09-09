#include <gtest/gtest.h>

#include <computoc/matrix.h>

TEST(Matrix_test, can_be_initialized_with_valid_size_and_data)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = { 0, 0, 0 };
    EXPECT_NO_THROW((Integer_matrix{ {1, 1}, data }));
    EXPECT_NO_THROW((Integer_matrix{ {1, 3}, data }));
    EXPECT_NO_THROW((Integer_matrix{ {3, 1}, data }));
    EXPECT_NO_THROW((Integer_matrix{ {1, 1, 3}, data }));
    EXPECT_NO_THROW((Integer_matrix{ {1, 1, 3}, data }));

    EXPECT_THROW((Integer_matrix{ {0, 0}, data }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 0}, data }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {0, 1}, data }), std::invalid_argument);

    EXPECT_THROW((Integer_matrix{ {0, 0, 1}, data }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 0, 1}, data }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {0, 1, 1}, data }), std::invalid_argument);

    EXPECT_THROW((Integer_matrix{ {0, 0, 0}, data }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 0, 0}, data }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {0, 1, 0}, data }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 1, 0}, data }), std::invalid_argument);
}

TEST(Matrix_test, can_be_initialized_with_valid_size_and_value)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int value{ 0 };
    EXPECT_NO_THROW((Integer_matrix{ {1, 1}, value }));
    EXPECT_NO_THROW((Integer_matrix{ {1, 3}, value }));
    EXPECT_NO_THROW((Integer_matrix{ {3, 1}, value }));
    EXPECT_NO_THROW((Integer_matrix{ {1, 1, 3}, value }));
    EXPECT_NO_THROW((Integer_matrix{ {1, 1, 3}, value }));

    EXPECT_THROW((Integer_matrix{ {0, 0}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 0}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {0, 1}, value }), std::invalid_argument);

    EXPECT_THROW((Integer_matrix{ {0, 0, 1}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 0, 1}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {0, 1, 1}, value }), std::invalid_argument);

    EXPECT_THROW((Integer_matrix{ {0, 0, 0}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 0, 0}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {0, 1, 0}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 1, 0}, value }), std::invalid_argument);
}

TEST(Matrix_test, can_return_its_size)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int value{ 0 };
    const computoc::Dims dims = { 1, 2, 3 };
    Integer_matrix mat{ dims, value };

    EXPECT_EQ(dims, mat.dims());
}

TEST(Matrix_test, have_read_write_access_to_its_cells)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2, 
        3, 4,
        5, 6 };

    computoc::Dims dims1d{ 6, 1 };
    Integer_matrix mat1d{ dims1d, data };
    for (std::size_t i = 0; i < dims1d.n; ++i) {
        EXPECT_EQ(mat1d({ i }), data[i]);
    }
    for (std::size_t i = 0; i < dims1d.n; ++i) {
        mat1d({ i }) = 0;
        EXPECT_EQ(mat1d({ i }), 0);
    }

    EXPECT_THROW(mat1d({ dims1d.n, 0, 0 }), std::out_of_range);
    EXPECT_THROW(mat1d({ 0, dims1d.m, 0 }), std::out_of_range);
    EXPECT_THROW(mat1d({ 0, 0, 1 }), std::out_of_range);

    computoc::Dims dims2d{ 3, 2 };
    Integer_matrix mat2d{ dims2d, data };
    for (std::size_t i = 0; i < dims2d.n; ++i) {
        for (std::size_t j = 0; j < dims2d.m; ++j) {
            EXPECT_EQ(mat2d({ i, j }), data[i * dims2d.m + j]);
        }
    }
    for (std::size_t i = 0; i < dims2d.n; ++i) {
        for (std::size_t j = 0; j < dims2d.m; ++j) {
            mat2d({ i, j }) = 0;
            EXPECT_EQ(mat2d({ i, j }), 0);
        }
    }

    EXPECT_THROW(mat2d({ dims2d.n, 0, 0 }), std::out_of_range);
    EXPECT_THROW(mat2d({ 0, dims2d.m, 0 }), std::out_of_range);
    EXPECT_THROW(mat2d({ 0, 0, 1 }), std::out_of_range);

    computoc::Dims dims3d{ 1, 2, 3 };
    Integer_matrix mat3d{ dims3d, data };
    for (std::size_t k = 0; k < dims3d.p; ++k) {
        for (std::size_t i = 0; i < dims3d.n; ++i) {
            for (std::size_t j = 0; j < dims3d.m; ++j) {
                EXPECT_EQ(mat3d({ i, j, k }), data[k * (dims3d.n * dims3d.m) + i * dims3d.m + j]);
            }
        }
    }
    for (std::size_t k = 0; k < dims3d.p; ++k) {
        for (std::size_t i = 0; i < dims3d.n; ++i) {
            for (std::size_t j = 0; j < dims3d.m; ++j) {
                mat3d({ i, j, k }) = 0;
                EXPECT_EQ(mat3d({ i, j, k }), 0);
            }
        }
    }

    EXPECT_THROW(mat3d({ dims3d.n, 0, 0 }), std::out_of_range);
    EXPECT_THROW(mat3d({ 0, dims3d.m, 0 }), std::out_of_range);
    EXPECT_THROW(mat3d({ 0, 0, dims3d.p }), std::out_of_range);
}

TEST(Matrix_test, can_be_compared_with_another_matrix)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data1[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims1{ 1, 2, 3 };
    Integer_matrix mat1{ dims1, data1 };
    Integer_matrix mat2{ dims1, data1 };

    EXPECT_EQ(mat1, mat2);

    computoc::Dims dims2{ 3, 2 };
    Integer_matrix mat3{ dims2, data1 };

    EXPECT_NE(mat1, mat3);

    const int data2[] = {
        1, 2,
        3, 4,
        5, 5 };
    Integer_matrix mat4{ dims1, data2 };
    Integer_matrix mat5{ dims2, data2 };

    EXPECT_NE(mat1, mat4);
    EXPECT_NE(mat1, mat5);
}

TEST(Matrix_test, can_return_slice)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
    1, 2,
    3, 4,
    5, 6 };

    computoc::Dims dims1{6, 1};
    Integer_matrix mat1{ dims1, data };

    const int sdata1[] = { 1, 2, 3 };
    computoc::Dims sdims1{ 3, 1 };
    Integer_matrix smat1{ sdims1, sdata1 };
    EXPECT_EQ(mat1({ 0, 0 }, smat1.dims()), smat1);

    EXPECT_THROW(mat1({ 0, 0 }, { 0, dims1.m }), std::invalid_argument);
    EXPECT_THROW(mat1({ 0, 0 }, { dims1.n, 0 }), std::invalid_argument);
    EXPECT_THROW(mat1({ 0, 0 }, { 1, dims1.m + 1 }), std::out_of_range);
    EXPECT_THROW(mat1({ 0, 0 }, { dims1.n + 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat1({ dims1.n, 0 }, { 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat1({ 0, dims1.m }, { 1, 1 }), std::out_of_range);

    computoc::Dims dims2{ 3, 2 };
    Integer_matrix mat2{ dims2, data };

    const int sdata2[] = {
        1, 2,
        3, 4 };
    computoc::Dims sdims2{ 2, 2 };
    Integer_matrix smat2{ sdims2, sdata2 };
    EXPECT_EQ(mat2({ 0, 0 }, smat2.dims()), smat2);

    EXPECT_THROW(mat2({ 0, 0 }, { 0, dims2.m }), std::invalid_argument);
    EXPECT_THROW(mat2({ 0, 0 }, { dims2.n, 0 }), std::invalid_argument);
    EXPECT_THROW(mat2({ 0, 0 }, { 1, dims2.m + 1 }), std::out_of_range);
    EXPECT_THROW(mat2({ 0, 0 }, { dims2.n + 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat2({ dims2.n, 0 }, { 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat2({ 0, dims2.m }, { 1, 1 }), std::out_of_range);

    computoc::Dims dims3{ 1, 2, 3 };
    Integer_matrix mat3{ dims3, data };

    const int sdata3[] = {
        1, 2,
        3, 4 };
    computoc::Dims sdims3{ 1, 2, 2 };
    Integer_matrix smat3{ sdims3, sdata3 };
    EXPECT_EQ(mat3({ 0, 0 }, smat3.dims()), smat3);

    EXPECT_THROW(mat3({ 0, 0, 0 }, { 0, dims3.m, dims3.p }), std::invalid_argument);
    EXPECT_THROW(mat3({ 0, 0, 0 }, { dims3.n, 0, dims3.p }), std::invalid_argument);
    EXPECT_THROW(mat3({ 0, 0, 0 }, { dims3.n, dims3.m, 0 }), std::invalid_argument);
    EXPECT_THROW(mat3({ 0, 0, 0 }, { 1, dims3.m + 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat3({ 0, 0, 0 }, { dims3.n + 1, 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat3({ 0, 0, 0 }, { 1, 1, dims3.p + 1 }), std::out_of_range);
    EXPECT_THROW(mat3({ dims3.n, 0, 0 }, { 1, 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat3({ 0, dims3.m, 0 }, { 1, 1, 1 }), std::out_of_range);
    EXPECT_THROW(mat3({ 0, 0, dims3.p }, { 1, 1, 1 }), std::out_of_range);
}

TEST(Matrix_test, copy_by_reference)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims{ 1, 2, 3 };
    Integer_matrix mat{ dims, data };

    Integer_matrix cmat1{ mat };
    cmat1({ 0, 0, 2 }) = 0;
    const int rdata1[] = {
        1, 2,
        3, 4,
        0, 6 };
    Integer_matrix rmat1{ dims, rdata1 };
    EXPECT_EQ(rmat1, cmat1);

    Integer_matrix cmat2{};
    cmat2 = cmat1;
    cmat1({ 0, 0, 0 }) = 0;
    const int rdata2[] = {
        0, 2,
        3, 4,
        0, 6 };
    Integer_matrix rmat2{ dims, rdata2 };
    EXPECT_EQ(rmat2, cmat2);
    EXPECT_THROW(cmat2({ 0, 0, 0 }, { 1, 1, 1 }) = cmat1, std::runtime_error);
}

TEST(Matrix_test, move_by_reference)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims{ 1, 2, 3 };
    Integer_matrix smat{ dims, data };

    Integer_matrix mat{ dims, data };
    Integer_matrix cmat1{ std::move(mat) };
    EXPECT_EQ(smat, cmat1);
    EXPECT_TRUE(is_empty(mat.dims()));

    Integer_matrix cmat2{};
    cmat2 = std::move(cmat1);
    EXPECT_EQ(smat, cmat2);
    EXPECT_TRUE(is_empty(cmat1.dims()));
    EXPECT_THROW(cmat2({ 0, 0, 0 }, { 1, 1, 1 }) = std::move(smat), std::runtime_error);
}

TEST(Matrix_test, copy_matrix)
{
    using Integer_matrix = computoc::Matrix<int>;

    Integer_matrix empty_mat{};
    Integer_matrix cempty_mat{ empty_mat.copy() };
    EXPECT_EQ(empty_mat, cempty_mat);

    const int data[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims{ 1, 2, 3 };
    Integer_matrix smat{ dims, data };

    Integer_matrix cmat{ smat.copy() };
    EXPECT_EQ(cmat, smat);
    cmat({ 0, 0 }) = 0;
    EXPECT_NE(cmat, smat);

    Integer_matrix csubmat{ smat({0, 0, 0}, {1, 1, 1}).copy() };
    EXPECT_EQ(smat({ 0, 0, 0 }, { 1, 1, 1 }), csubmat);
    csubmat({ 0, 0 }) = 5;
    EXPECT_NE(smat({ 0, 0, 0 }, { 1, 1, 1 }), csubmat);
}

TEST(Matrix_test, copy_from)
{
    using Integer_matrix = computoc::Matrix<int>;

    Integer_matrix empty_mat{};
    Integer_matrix cempty_mat{};
    cempty_mat.copy_from(Integer_matrix{});
    EXPECT_EQ(empty_mat, cempty_mat);

    const int data1[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims1{ 1, 2, 3 };
    Integer_matrix mat1{ dims1, data1 };

    const int data2[] = {
        2, 4,
        6, 8,
        10, 12 };
    computoc::Dims dims2{ 1, 2, 3 };
    Integer_matrix mat2{ dims2, data2 };
    EXPECT_NE(mat1, mat2);
    mat1.copy_from(mat2);
    EXPECT_EQ(mat1, mat2);

    const int data3[] = {
        10, 12,
        6, 8,
        10, 12 };
    computoc::Dims dims3{ 1, 2, 3 };
    Integer_matrix mat3{ dims3, data3 };
    mat2({ 0, 0, 0 }, { 1, 2, 1 }).copy_from(mat2({ 0, 0, 2 }, { 1, 2, 1 }));
    EXPECT_EQ(mat3, mat2);

    EXPECT_THROW(mat2({ 0, 0, 0 }, { 1, 2, 1 }).copy_from(mat2), std::invalid_argument);
}

TEST(Matrix_test, copy_to)
{
    using Integer_matrix = computoc::Matrix<int>;

    Integer_matrix empty_mat{};
    Integer_matrix cempty_mat{};
    empty_mat.copy_to(cempty_mat);
    EXPECT_EQ(empty_mat, cempty_mat);

    const int data1[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims1{ 1, 2, 3 };
    Integer_matrix mat1{ dims1, data1 };

    const int data2[] = {
        2, 4,
        6, 8,
        10, 12 };
    computoc::Dims dims2{ 1, 2, 3 };
    Integer_matrix mat2{ dims2, data2 };
    EXPECT_NE(mat1, mat2);
    mat2.copy_to(mat1);
    EXPECT_EQ(mat1, mat2);

    const int data3[] = {
        10, 12,
        6, 8,
        10, 12 };
    computoc::Dims dims3{ 1, 2, 3 };
    Integer_matrix mat3{ dims3, data3 };
    mat2({ 0, 0, 2 }, { 1, 2, 1 }).copy_to(mat2({ 0, 0, 0 }, { 1, 2, 1 }));
    EXPECT_EQ(mat3, mat2);
    EXPECT_THROW(mat2({ 0, 0, 2 }, { 1, 2, 1 }).copy_to(mat2({ 0, 0, 0 }, { 1, 2, 2 })), std::runtime_error);

    mat3({ 0, 0, 0 }, { 1, 2, 1 }).copy_to(mat2);
    EXPECT_EQ(mat3({ 0, 0, 0 }, { 1, 2, 1 }), mat2);
}

TEST(Matrix_test, reshape)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data1[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims1{ 1, 2, 3 };
    Integer_matrix mat1{ dims1, data1 };

    const int data2[] = {1, 2, 3, 4, 5, 6 };
    computoc::Dims dims2{ 1, 6 };
    Integer_matrix mat2{ dims2, data2 };
    
    Integer_matrix rmat1{ mat1.reshape({1, 6}) };
    EXPECT_EQ(mat2, rmat1);

    const int data3[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims3{ 1, 2, 3 };
    Integer_matrix mat3{ dims3, data3 };

    Integer_matrix rmat2{ mat1.reshaped({1, 2, 3}) };
    EXPECT_NE((computoc::Dims{ 1, 2, 3 }), mat1.dims());
    EXPECT_EQ(mat3, rmat2);
    
    EXPECT_THROW(mat2({ 0, 0 }, { 1, 2 }).reshape({}), std::runtime_error);
    EXPECT_THROW(Integer_matrix{}.reshape({}), std::runtime_error);
    EXPECT_THROW(mat2.reshape({ 1, 1 }), std::invalid_argument);
}

TEST(Matrix_test, resize)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data1[] = { 1, 2, 3, 4, 5, 6 };
    computoc::Dims dims1{ 1, 6 };
    Integer_matrix mat1{ dims1, data1 };

    const int data2[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims2{ 1, 2, 3 };
    Integer_matrix mat2{ dims2, data2 };

    mat1.resize({ 1, 2, 3 });
    EXPECT_EQ(mat2, mat1);

    const int data3[] = { 1, 2 };
    computoc::Dims dims3{ 1, 2 };
    Integer_matrix mat3{ dims3, data3 };

    mat1.resize({ 1, 2 });
    EXPECT_EQ(mat3, mat1);

    const int data4[] = { 1, 2, 3, 4 };
    computoc::Dims dims4{ 1, 4 };
    Integer_matrix mat4{ dims4, data4 };

    mat1.resize({ 1, 4 });
    EXPECT_NE(mat4, mat1);
    EXPECT_EQ(mat4({ 0, 0 }), mat1({ 0, 0 }));
    EXPECT_EQ(mat4({ 0, 1 }), mat1({ 0, 1 }));

    Integer_matrix mat5{ mat1.resized({1, 2}) };
    EXPECT_NE(mat1.dims(), mat5.dims());
    EXPECT_EQ(mat3, mat5);

    EXPECT_THROW(mat1({ 0, 0, 0 }, { 1, 1, 1 }).resize({}), std::runtime_error);
}

/*
TEST(Matrix_test, can_be_initialized_with_valid_size_and_data)
{
    using Integer_matrix = computoc::Matrix<int>;

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
    using Integer_matrix = computoc::Matrix<int>;

    const int value{ 0 };
    EXPECT_NO_THROW((Integer_matrix{ {1, 1}, value }));

    EXPECT_THROW((Integer_matrix{ {0, 0}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {1, 0}, value }), std::invalid_argument);
    EXPECT_THROW((Integer_matrix{ {0, 1}, value }), std::invalid_argument);
}

TEST(Matrix_test, can_return_its_size)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int value{ 0 };
    const std::size_t n = 1;
    const std::size_t m = 2;
    Integer_matrix mat{ {n, m}, value };

    computoc::Dimensions d{ mat.dimensions() };
    EXPECT_EQ(n, d.n);
    EXPECT_EQ(m, d.m);
}

TEST(Matrix_test, have_read_write_access_to_its_cells)
{
    using Integer_matrix = computoc::Matrix<int>;

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
    using Integer_matrix = computoc::Matrix<int>;

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
    using Integer_matrix = computoc::Matrix<int>;

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

TEST(Matrix_test, can_return_slice_by_pivot)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20 };
    const std::size_t n = 4;
    const std::size_t m = 5;
    Integer_matrix mat{ {n, m}, data };

    const std::size_t pi = 2;
    const std::size_t pj = 1;
    Integer_matrix slice{ mat.get_slice(pi, pj) };

    const int rdata[] = {
        1, 3, 4, 5,
        6, 8, 9, 10,
        16, 18, 19, 20 };
    const std::size_t rn = 3;
    const std::size_t rm = 4;
    Integer_matrix rmat{ {rn, rm}, rdata };

    EXPECT_EQ(rmat, slice);
    EXPECT_THROW(mat.get_slice(n, 0), std::out_of_range);
    EXPECT_THROW(mat.get_slice(0, m), std::out_of_range);
    EXPECT_THROW(mat.get_slice(n, m), std::out_of_range);
}

TEST(Matrix_test, can_write_into_slice)
{
    using Integer_matrix = computoc::Matrix<int>;

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
    Integer_matrix rmat{ {rn, rm}, rdata };

    EXPECT_EQ(mat, rmat);

    EXPECT_THROW(smat.set_slice(0, 0, mat), std::out_of_range);
}

TEST(Matrix_test, can_negate)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
    1, 2, 3,
    4, 5, 6 };
    const std::size_t n = 2;
    const std::size_t m = 3;
    Integer_matrix mat{ {n, m}, data };

    const int rdata[] = {
    -1, -2, -3,
    -4, -5, -6 };
    const std::size_t rn = 2;
    const std::size_t rm = 3;
    Integer_matrix rmat{ {rn, rm}, rdata };

    EXPECT_EQ(rmat, -mat);
}

TEST(Matrix_test, have_positive)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
    1, 2, 3,
    4, 5, 6 };
    const std::size_t n = 2;
    const std::size_t m = 3;
    Integer_matrix mat{ {n, m}, data };

    EXPECT_EQ(mat, +mat);
}

TEST(Matrix_test, can_be_added_with_another_matrix)
{
    using Integer_matrix = computoc::Matrix<int>;

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

    EXPECT_THROW((mat + Integer_matrix{{1, 1}, 0}), std::invalid_argument);
}

TEST(Matrix_test, can_be_subtracted_from_another_matrix)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2, 3,
        4, 5, 6 };
    const std::size_t n = 2;
    const std::size_t m = 3;
    Integer_matrix mat{ {n, m}, data };
    Integer_matrix rmat{ mat };

    mat -= mat - mat;

    EXPECT_EQ(mat, rmat);

    EXPECT_THROW((mat - Integer_matrix{{1, 1}, 0}), std::invalid_argument);
}

TEST(Matrix_test, can_be_multiplied_by_a_constant)
{
    using Integer_matrix = computoc::Matrix<int>;

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
    using Integer_matrix = computoc::Matrix<int>;

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

    EXPECT_THROW(mat2 * mat2, std::invalid_argument);
}

TEST(Matrix_test, can_be_transposed)
{
    using Integer_matrix = computoc::Matrix<int>;

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

TEST(Matrix_test, have_determinant_if_squared)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16 };
    const std::size_t n = 4;
    Integer_matrix mat{ {n, n}, data };

    EXPECT_EQ(1, computoc::determinant(mat.get_slice(0, 0, { 1, 1 })));
    EXPECT_EQ(-4, computoc::determinant(mat.get_slice(0, 0, { 2, 2 })));
    EXPECT_EQ(0, computoc::determinant(mat.get_slice(0, 0, { 3, 3 })));
    EXPECT_EQ(0, computoc::determinant(mat));

    EXPECT_THROW(computoc::determinant(Integer_matrix{ {1, 2}, 0 }), std::invalid_argument);
    EXPECT_THROW(computoc::determinant(Integer_matrix{ {2, 1}, 0 }), std::invalid_argument);
}

TEST(Matrix_test, have_inverse_if_squared_and_zero_determinant)
{
    using Double_matrix = computoc::Matrix<double>;

    const double data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16 };
    const std::size_t n = 4;
    Double_matrix mat{ {n, n}, data };

    const double inv_data1[] = {
        -1.5, 0.5,
        1.25, -0.25 };
    const std::size_t in1 = 2;
    Double_matrix inv_mat1{ {in1, in1}, inv_data1 };
    EXPECT_EQ(inv_mat1, computoc::inverse(mat.get_slice(0, 0, { 2, 2 })));

    const double unit_data2[] = {
    1, 0,
    0, 1 };
    const std::size_t un1 = 2;
    Double_matrix unit_mat2{ {un1, un1}, unit_data2 };
    EXPECT_EQ(unit_mat2, inv_mat1 * mat.get_slice(0, 0, { 2, 2 }));

    EXPECT_THROW(computoc::inverse(Double_matrix{ {1, 2}, 0.0 }), std::invalid_argument);
    EXPECT_THROW(computoc::inverse(Double_matrix{ {2, 1}, 0.0 }), std::invalid_argument);
    EXPECT_THROW(computoc::inverse(mat), std::invalid_argument);
}

TEST(Matrix_test, can_be_merged)
{
    using Integer_matrix = computoc::Matrix<int>;

    const int data1[] = {
        1, 2, 3,
        4, 5, 6 };
    const std::size_t n1 = 2;
    const std::size_t m1 = 3;
    Integer_matrix mat1{ {n1, m1}, data1 };

    const int data2[] = {
        7, 8, 9,
        10, 11, 12 };
    const std::size_t n2 = 2;
    const std::size_t m2 = 3;
    Integer_matrix mat2{ {n2, m2}, data2 };

    const int hmerged_data[] = {
        1, 2, 3, 7, 8, 9,
        4, 5, 6, 10, 11, 12 };
    const std::size_t hn = 2;
    const std::size_t hm = 6;
    Integer_matrix hmerged{ {hn, hm}, hmerged_data };
    EXPECT_EQ(hmerged, computoc::merge_horizontal(mat1, mat2));
    EXPECT_THROW(computoc::merge_horizontal(Integer_matrix{ {1, 1}, 0 }, Integer_matrix{ {2, 1}, 0 }), std::invalid_argument);

    const int vmerged_data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12 };
    const std::size_t vn = 4;
    const std::size_t vm = 3;
    Integer_matrix vmerged{ {vn, vm}, vmerged_data };
    EXPECT_EQ(vmerged, computoc::merge_vertical(mat1, mat2));
    EXPECT_THROW(computoc::merge_vertical(Integer_matrix{ {1, 1}, 0 }, Integer_matrix{ {1, 2}, 0 }), std::invalid_argument);
}

TEST(Matrix_test, can_add_multiply_and_swap_rows)
{
    using Double_matrix = computoc::Matrix<double>;

    const double data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8 };
    const std::size_t n = 2;
    const std::size_t m = 4;
    Double_matrix mat{ {n, m}, data };

    mat.multiply_row(0, 2);
    const double rdata1[] = {
        2, 4, 6, 8,
        5, 6, 7, 8 };
    const std::size_t rn1 = 2;
    const std::size_t rm1 = 4;
    Double_matrix rmat1{ {rn1, rm1}, rdata1 };
    EXPECT_EQ(rmat1, mat);

    mat.add_row(0, 1, 0.5);
    const double rdata2[] = {
        2, 4, 6, 8,
        6, 8, 10, 12 };
    const std::size_t rn2 = 2;
    const std::size_t rm2 = 4;
    Double_matrix rmat2{ {rn2, rm2}, rdata2 };
    EXPECT_EQ(rmat2, mat);

    mat.swap_rows(0, 1);
    const double rdata3[] = {
        6, 8, 10, 12,
        2, 4, 6, 8 };
    const std::size_t rn3 = 2;
    const std::size_t rm3 = 4;
    Double_matrix rmat3{ {rn3, rm3}, rdata3 };
    EXPECT_EQ(rmat3, mat);

    EXPECT_THROW(mat.multiply_row(n + 1, 1), std::out_of_range);

    EXPECT_THROW(mat.add_row(n + 1, 0, 1), std::out_of_range);
    EXPECT_THROW(mat.add_row(0, n + 1, 1), std::out_of_range);

    EXPECT_THROW(mat.swap_rows(n + 1, 0), std::out_of_range);
    EXPECT_THROW(mat.swap_rows(0, n + 1), std::out_of_range);
}

//TEST(Matrix_test, have_row_echelon_form)
//{
//    using Double_matrix = computoc::Matrix<double>;
//
//    const double data[] = {
//        0.25, 0.5, 1, 5.75,
//        1, 1, 1, 7,
//        4, 2, 1, 2 };
//    const std::size_t n = 3;
//    const std::size_t m = 4;
//    Double_matrix mat{ {n, m}, data };
//
//    const double rdata[] = {
//        1, 2, 4, 23,
//        0, 1, 3, 16,
//        0, 0, 3, 6 };
//    const std::size_t rn = 3;
//    const std::size_t rm = 4;
//    Double_matrix rmat{ {rn, rm}, rdata };
//
//    EXPECT_EQ(rmat, computoc::row_echelon_form(mat));
//}

TEST(Matrix_test, have_reduced_row_echelon_form)
{
    using Double_matrix = computoc::Matrix<double>;

    const double data1[] = {
        0.25, 0.5, 1, 5.75, 
        1, 1, 1, 7,
        4, 2, 1, 2 };
    const std::size_t n1 = 3;
    const std::size_t m1 = 4;
    Double_matrix mat1{ {n1, m1}, data1 };

    const double rdata1[] = {
        1, 0, 0, -5,
        0, 1, 0, 10,
        0, 0, 1, 2 };
    const std::size_t rn1 = 3;
    const std::size_t rm1 = 4;
    Double_matrix rmat1{ {rn1, rm1}, rdata1 };

    EXPECT_EQ(rmat1, computoc::reduced_row_echelon_form(mat1));

    const double data2[] = {
        6, 10, 4, 22,
        1, 1, 1, 3,
        2, 4, 1, 8 };
    const std::size_t n2 = 3;
    const std::size_t m2 = 4;
    Double_matrix mat2{ {n2, m2}, data2 };

    const double rdata2[] = {
        1, 0, 1.5, 2,
        0, 1, -0.5, 1,
        0, 0, 0, 0 };
    const std::size_t rn2 = 3;
    const std::size_t rm2 = 4;
    Double_matrix rmat2{ {rn2, rm2}, rdata2 };

    EXPECT_EQ(rmat2, computoc::reduced_row_echelon_form(mat2));
}
*/
