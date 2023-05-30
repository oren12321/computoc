#include <gtest/gtest.h>

#include <computoc/matrix.h>

TEST(Dims_test, can_be_empty)
{
    using namespace computoc;

    EXPECT_TRUE(empty(Dims{0, 0, 0}));
    EXPECT_TRUE(empty(Dims{1, 0, 0}));
    EXPECT_TRUE(empty(Dims{0, 1, 0}));
    EXPECT_TRUE(empty(Dims{0, 0, 1}));
    EXPECT_FALSE(empty(Dims{1, 1, 1}));
}

TEST(Dims_test, is_comparable)
{
    using namespace computoc;

    EXPECT_EQ((Dims{ 1, 1, 1 }), (Dims{ 1, 1, 1 }));
    EXPECT_NE((Dims{ 1, 1, 1 }), (Dims{ 0, 1, 1 }));
    EXPECT_NE((Dims{ 1, 1, 1 }), (Dims{ 1, 0, 1 }));
    EXPECT_NE((Dims{ 1, 1, 1 }), (Dims{ 1, 1, 0 }));
}

TEST(Dims_test, have_product)
{
    using namespace computoc;

    EXPECT_EQ(1 * 2 * 3, product(Dims{ 1, 2, 3 }));
}


TEST(Step_test, is_comparable)
{
    using namespace computoc;

    EXPECT_EQ((Step{ 1, 1 }), (Step{ 1, 1 }));
    EXPECT_NE((Step{ 1, 1 }), (Step{ 0, 1 }));
    EXPECT_NE((Step{ 1, 1 }), (Step{ 1, 0 }));
}


TEST(Inds_test, is_comparable)
{
    using namespace computoc;

    EXPECT_EQ((Inds{ 1, 1, 1 }), (Inds{ 1, 1, 1 }));
    EXPECT_NE((Inds{ 1, 1, 1 }), (Inds{ 0, 1, 1 }));
    EXPECT_NE((Inds{ 1, 1, 1 }), (Inds{ 1, 0, 1 }));
    EXPECT_NE((Inds{ 1, 1, 1 }), (Inds{ 1, 1, 0 }));
}

TEST(Inds_test, can_be_converted_to_buff_index)
{
    using namespace computoc;

    EXPECT_EQ(6 + 3 * 5 + 1 * 4 + 2, to_buff_index(Inds{ 1, 2, 3 }, Step{ 4, 5 }, 6));
}

TEST(Inds_test, can_check_if_inside_dimensions)
{
    using namespace computoc;

    Dims dims{ 1, 2, 3 };

    EXPECT_TRUE(is_inside(Inds{}, dims));
    EXPECT_TRUE(is_inside(Inds{ 0, 1, 2 }, dims));
    EXPECT_FALSE(is_inside(Inds{ 1, 1, 2 }, dims));
    EXPECT_FALSE(is_inside(Inds{ 1, 3, 3 }, dims));
    EXPECT_FALSE(is_inside(Inds{ 1, 2, 4 }, dims));
}


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

TEST(Matrix_test, can_return_its_header_and_data)
{
    using Integer_matrix = computoc::Matrix<int>;

    Integer_matrix emat{};
    const Integer_matrix::Header& ehdr{ emat.header() };

    EXPECT_EQ(computoc::Dims{}, ehdr.dims);
    EXPECT_EQ(computoc::Step{}, ehdr.step);
    EXPECT_EQ(0, ehdr.offset);
    EXPECT_FALSE(ehdr.is_submatrix);
    EXPECT_FALSE(emat.data());

    const int value{ 0 };
    const computoc::Dims dims = { 1, 2, 3 };
    Integer_matrix mat{ dims, value };
    const Integer_matrix::Header& hdr{ mat.header() };

    EXPECT_EQ(dims, hdr.dims);
    EXPECT_EQ((computoc::Step{2, 2}), hdr.step);
    EXPECT_EQ(0, hdr.offset);
    EXPECT_FALSE(hdr.is_submatrix);
    EXPECT_TRUE(mat.data());
    for (std::size_t i = 0; i < computoc::product(dims); ++i) {
        EXPECT_EQ(0, mat.data()[i]);
    }
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
    EXPECT_EQ(mat1({ 0, 0 }, smat1.header().dims), smat1);

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
    EXPECT_EQ(mat2({ 0, 0 }, smat2.header().dims), smat2);

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
    EXPECT_EQ(mat3({ 0, 0 }, smat3.header().dims), smat3);

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
    EXPECT_TRUE(empty(mat.header().dims));

    Integer_matrix cmat2{};
    cmat2 = std::move(cmat1);
    EXPECT_EQ(smat, cmat2);
    EXPECT_TRUE(empty(cmat1.header().dims));
    EXPECT_THROW(cmat2({ 0, 0, 0 }, { 1, 1, 1 }) = std::move(smat), std::runtime_error);
}

TEST(Matrix_test, clone)
{
    using Integer_matrix = computoc::Matrix<int>;

    Integer_matrix empty_mat{};
    Integer_matrix cempty_mat{ computoc::clone(empty_mat) };
    EXPECT_EQ(empty_mat, cempty_mat);

    const int data[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims{ 1, 2, 3 };
    Integer_matrix smat{ dims, data };

    Integer_matrix cmat{ computoc::clone(smat) };
    EXPECT_EQ(cmat, smat);
    cmat({ 0, 0 }) = 0;
    EXPECT_NE(cmat, smat);

    Integer_matrix csubmat{ computoc::clone(smat({0, 0, 0}, {1, 1, 1})) };
    EXPECT_EQ(smat({ 0, 0, 0 }, { 1, 1, 1 }), csubmat);
    csubmat({ 0, 0 }) = 5;
    EXPECT_NE(smat({ 0, 0, 0 }, { 1, 1, 1 }), csubmat);
}

TEST(Matrix_test, copy)
{
    using Integer_matrix = computoc::Matrix<int>;

    { // backward cases - copy from other matrix to created matrix
        Integer_matrix empty_mat{};
        Integer_matrix cempty_mat{};
        computoc::copy(Integer_matrix{}, cempty_mat);
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
        computoc::copy(mat2, mat1);
        EXPECT_EQ(mat1, mat2);

        const int data3[] = {
            10, 12,
            6, 8,
            10, 12 };
        computoc::Dims dims3{ 1, 2, 3 };
        Integer_matrix mat3{ dims3, data3 };
        computoc::copy(mat2({ 0, 0, 2 }, { 1, 2, 1 }), mat2({ 0, 0, 0 }, { 1, 2, 1 }));
        EXPECT_EQ(mat3, mat2);

        EXPECT_THROW(computoc::copy(mat2, mat2({ 0, 0, 0 }, { 1, 2, 1 })), std::runtime_error);
    }

    { // forward cases - copy from other matrix to created matrix
        Integer_matrix empty_mat{};
        Integer_matrix cempty_mat{};
        computoc::copy(empty_mat, cempty_mat);
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
        computoc::copy(mat2, mat1);
        EXPECT_EQ(mat1, mat2);

        const int data3[] = {
            10, 12,
            6, 8,
            10, 12 };
        computoc::Dims dims3{ 1, 2, 3 };
        Integer_matrix mat3{ dims3, data3 };
        computoc::copy(mat2({ 0, 0, 2 }, { 1, 2, 1 }), mat2({ 0, 0, 0 }, { 1, 2, 1 }));
        EXPECT_EQ(mat3, mat2);
        EXPECT_THROW(computoc::copy(mat2({ 0, 0, 2 }, { 1, 2, 1 }), mat2({ 0, 0, 0 }, { 1, 2, 2 })), std::runtime_error);

        computoc::copy(mat3({ 0, 0, 0 }, { 1, 2, 1 }), mat2);
        EXPECT_EQ(mat3({ 0, 0, 0 }, { 1, 2, 1 }), mat2);
    }
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

    const int data2[] = { 1, 2, 3, 4, 5, 6 };
    computoc::Dims dims2{ 1, 6 };
    Integer_matrix mat2{ dims2, data2 };

    Integer_matrix rmat1{ computoc::reshaped(mat1, {1, 6}) };
    EXPECT_EQ(mat2, rmat1);

    const int data3[] = {
        1, 2,
        3, 4,
        5, 6 };
    computoc::Dims dims3{ 1, 2, 3 };
    Integer_matrix mat3{ dims3, data3 };

    Integer_matrix rmat2{ computoc::reshaped(mat1, {1, 2, 3}) };
    EXPECT_EQ(mat3, rmat2);

    EXPECT_THROW(computoc::reshaped(mat2({ 0, 0 }, { 1, 2 }), {}), std::runtime_error);
    EXPECT_THROW(computoc::reshaped(Integer_matrix{}, {}), std::runtime_error);
    EXPECT_THROW(computoc::reshaped(mat2, { 1, 1 }), std::invalid_argument);
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

    mat1 = computoc::resized(mat1, { 1, 2, 3 });
    EXPECT_EQ(mat2, mat1);

    const int data3[] = { 1, 2 };
    computoc::Dims dims3{ 1, 2 };
    Integer_matrix mat3{ dims3, data3 };

    mat1 = computoc::resized(mat1, { 1, 2 });
    EXPECT_EQ(mat3, mat1);

    const int data4[] = { 1, 2, 3, 4 };
    computoc::Dims dims4{ 1, 4 };
    Integer_matrix mat4{ dims4, data4 };

    mat1 = computoc::resized(mat1, { 1, 4 });
    //EXPECT_NE(mat4, mat1);
    EXPECT_EQ(mat4({ 0, 0 }), mat1({ 0, 0 }));
    EXPECT_EQ(mat4({ 0, 1 }), mat1({ 0, 1 }));

    Integer_matrix mat5{ computoc::resized(mat1, {1, 2}) };
    EXPECT_NE(mat1.header().dims, mat5.header().dims);
    EXPECT_EQ(mat3, mat5);

    EXPECT_THROW(computoc::resized(mat1({ 0, 0, 0 }, { 1, 1, 1 }), {}), std::runtime_error);
}

/*
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

*/
