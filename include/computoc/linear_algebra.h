#ifndef COMPUTOC_LINEAR_ALGEBRA_H
#define COMPUTOC_LINEAR_ALGEBRA_H

#include <cstddef>
#include <stdexcept>

#include <memoc/allocators.h>
#include <memoc/buffers.h>
#include <erroc/errors.h>
#include <computoc/concepts.h>
#include <computoc/math.h>
#include <computoc/matrix.h>

namespace computoc {
    namespace details {
        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> excluded(const Matrix<T, Internal_buffer, Internal_allocator>& mat, const Inds& pivot)
        {
            ERROC_EXPECT(!empty(mat), std::invalid_argument, "minor for empty matrix is invalid");
            ERROC_EXPECT(is_inside(pivot, mat.header().dims), std::out_of_range, "pivot is not in matrix dimensions");
            ERROC_EXPECT(mat.header().dims.n > 1 && mat.header().dims.m > 1, std::invalid_argument, "operation is undefined for 1x1 matrix");

            Matrix<T, Internal_buffer, Internal_allocator> mmat{ {mat.header().dims.n - 1, mat.header().dims.m - 1, mat.header().dims.p} };

            if (pivot.i == 0 && pivot.j == 0) {
                copy(mat({ 1 ,1, 0 }, { mmat.header().dims.n, mmat.header().dims.m, mat.header().dims.p }), mmat); // BR
                return mmat;
            }

            if (pivot.i == mat.header().dims.n - 1 && pivot.j == mat.header().dims.m - 1) {
                copy(mat({ 0, 0, 0 }, { mmat.header().dims.n, mmat.header().dims.m, mat.header().dims.p }), mmat); // UL
                return mmat;
            }

            if (pivot.i == 0 && pivot.j == mat.header().dims.m - 1) {
                copy(mat({ 1, 0, 0 }, { mmat.header().dims.n, mmat.header().dims.m, mat.header().dims.p }), mmat); // BL
                return mmat;
            }

            if (pivot.i == mat.header().dims.n - 1 && pivot.j == 0) {
                copy(mat({ 0, 1, 0 }, { mmat.header().dims.n, mmat.header().dims.m, mat.header().dims.p }), mmat); // UR
                return mmat;
            }

            if (pivot.i == 0) { // 0 < pivot.j < mat.header().dims.m - 1
                copy(mat({ 1, 0, 0 }, { mmat.header().dims.n, pivot.j, mat.header().dims.p }), mmat({ 0, 0, 0 }, { mmat.header().dims.n, pivot.j, mat.header().dims.p })); // BL
                copy(mat({ 1 ,pivot.j + 1, 0 }, { mmat.header().dims.n, mmat.header().dims.m - pivot.j, mat.header().dims.p }), mmat({ 0, pivot.j, 0 }, { mmat.header().dims.n, mmat.header().dims.m - pivot.j, mat.header().dims.p })); // BR
                return mmat;
            }

            if (pivot.j == 0) { // 0 < pivot.i < mat.header.dims.n - 1
                copy(mat({ 0, 1, 0 }, { pivot.i, mmat.header().dims.m, mat.header().dims.p }), mmat({ 0, 0, 0 }, { pivot.i, mmat.header().dims.m, mat.header().dims.p })); // UR
                copy(mat({ pivot.i + 1 , 1, 0 }, { mmat.header().dims.n - pivot.i, mmat.header().dims.m, mat.header().dims.p }), mmat({ pivot.i, 0, 0 }, { mmat.header().dims.n - pivot.i, mmat.header().dims.m, mat.header().dims.p })); // BR
                return mmat;
            }

            if (pivot.i == mat.header().dims.n - 1) { // 0 < pivot.j < mat.header().dims.m - 1
                copy(mat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p }), mmat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p })); // UL
                copy(mat({ 0, pivot.j + 1, 0 }, { pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p }), mmat({ 0, pivot.j, 0 }, { pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p })); // UR
                return mmat;
            }

            if (pivot.j == mat.header().dims.m - 1) { // 0 < pivot.i < mat.header().dims.n - 1
                copy(mat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p }), mmat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p })); // UL
                copy(mat({ pivot.i + 1, 0, 0 }, { mmat.header().dims.n - pivot.i, pivot.j, mat.header().dims.p }), mmat({ pivot.i, 0, 0 }, { mmat.header().dims.n - pivot.i, pivot.j, mat.header().dims.p })); // BL
                return mmat;
            }

            copy(mat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p }), mmat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p })); // UL
            copy(mat({ pivot.i + 1, 0, 0 }, { mmat.header().dims.n - pivot.i, pivot.j, mat.header().dims.p }), mmat({ pivot.i, 0, 0 }, { mmat.header().dims.n - pivot.i, pivot.j, mat.header().dims.p })); // BL
            copy(mat({ 0, pivot.j + 1, 0 }, { pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p }), mmat({ 0, pivot.j, 0 }, { pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p })); // UR
            copy(mat({ pivot.i + 1 ,pivot.j + 1, 0 }, { mmat.header().dims.n - pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p }), mmat({ pivot.i, pivot.j, 0 }, { mmat.header().dims.n - pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p })); // BR
            return mmat;
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator+(const Matrix<T, Internal_buffer, Internal_allocator>& mat)
        {
            return clone(mat);
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator-(const Matrix<T, Internal_buffer, Internal_allocator>& mat)
        {
            Matrix<T, Internal_buffer, Internal_allocator> nmat{ mat.header().dims };

            for (std::size_t k = 0; k < mat.header().dims.p; ++k) {
                for (std::size_t i = 0; i < mat.header().dims.n; ++i) {
                    for (std::size_t j = 0; j < mat.header().dims.m; ++j) {
                        nmat({ i, j, k }) = -mat({ i, j, k });
                    }
                }
            }

            return nmat;
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator>& operator+=(Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            ERROC_EXPECT(lhs.header().dims == rhs.header().dims, std::invalid_argument, "matrix should have same dimensions");

            for (std::size_t k = 0; k < lhs.header().dims.p; ++k) {
                for (std::size_t i = 0; i < lhs.header().dims.n; ++i) {
                    for (std::size_t j = 0; j < lhs.header().dims.m; ++j) {
                        lhs({ i, j, k }) += rhs({ i, j, k });
                    }
                }
            }

            return lhs;
        }
        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator+(const Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            ERROC_EXPECT(lhs.header().dims == rhs.header().dims, std::invalid_argument, "matrix should have same dimensions");

            Matrix<T, Internal_buffer, Internal_allocator> addition{ clone(lhs) };
            addition += rhs;

            return addition;
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator>& operator-=(Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            ERROC_EXPECT(lhs.header().dims == rhs.header().dims, std::invalid_argument, "matrix should have same dimensions");

            for (std::size_t k = 0; k < lhs.header().dims.p; ++k) {
                for (std::size_t i = 0; i < lhs.header().dims.n; ++i) {
                    for (std::size_t j = 0; j < lhs.header().dims.m; ++j) {
                        lhs({ i, j, k }) -= rhs({ i, j, k });
                    }
                }
            }

            return lhs;
        }
        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator-(const Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            ERROC_EXPECT(lhs.header().dims == rhs.header().dims, std::invalid_argument, "matrix should have same dimensions");

            Matrix<T, Internal_buffer, Internal_allocator> subtraction{ clone(lhs) };
            subtraction += rhs;

            return subtraction;
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator>& operator*=(Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            ERROC_EXPECT(lhs.header().dims.m == rhs.header().dims.n && lhs.header().dims.p == rhs.header().dims.p, std::invalid_argument, "matrices dimensions are invalid for multiplication");

            Matrix<T, Internal_buffer> multiplication{ {lhs.header().dims.n, rhs.header().dims.m, rhs.header().dims.p}, T{} };

            for (std::size_t t = 0; t < lhs.header().dims.p; ++t) {

                for (std::size_t i = 0; i < multiplication.header().dims.n; ++i) {
                    for (std::size_t k = 0; k < lhs.header().dims.m; ++k) {
                        for (std::size_t j = 0; j < multiplication.header().dims.m; ++j) {
                            multiplication({ i, j, t }) += lhs({ i, k, t }) * rhs({ k, j, t });
                        }
                    }
                }

            }

            lhs = std::move(multiplication);
            return lhs;
        }
        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator*(const Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            Matrix<T, Internal_buffer, Internal_allocator> multiplication{ clone(lhs) };
            multiplication *= rhs;

            return multiplication;
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator>& operator*=(Matrix<T, Internal_buffer, Internal_allocator>& lhs, const T& rhs)
        {
            for (std::size_t k = 0; k < lhs.header().dims.p; ++k) {
                for (std::size_t i = 0; i < lhs.header().dims.n; ++i) {
                    for (std::size_t j = 0; j < lhs.header().dims.m; ++j) {
                        lhs({ i, j, k }) *= rhs;
                    }
                }
            }

            return lhs;
        }
        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator*(const Matrix<T, Internal_buffer, Internal_allocator>& lhs, const T& rhs)
        {
            Matrix<T, Internal_buffer, Internal_allocator> multiplication{ clone(lhs) };
            multiplication *= rhs;

            return multiplication;
        }
        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator*(const T& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            return rhs * lhs;
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> transposed(const Matrix<T, Internal_buffer, Internal_allocator>& mat)
        {
            Matrix<T, Internal_buffer, Internal_allocator> tmat{ {mat.header().dims.m, mat.header().dims.n, mat.header().dims.p} };

            for (std::size_t k = 0; k < tmat.header().dims.p; ++k) {
                for (std::size_t i = 0; i < tmat.header().dims.n; ++i) {
                    for (std::size_t j = 0; j < tmat.header().dims.m; ++j) {
                        tmat({ i, j, k }) = mat({ j, i, k });
                    }
                }
            }

            return tmat;
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline T determinant2d_recursive(const Matrix<T, Internal_buffer, Internal_allocator>& mat, std::size_t k)
        {
            std::size_t n = mat.header().dims.n;

            if (n == 1) {
                return mat({ 0, 0, k });
            }

            if (n == 2) {
                return mat({ 0, 0, k }) * mat({ 1, 1, k }) - mat({ 0, 1, k }) * mat({ 1, 0, k });
            }

            int sign = T{ 1 };
            T d{ 0 };
            for (std::size_t j = 0; j < n; ++j) {
                T p{ mat({0, j, k}) };
                if (p != T{ 0 }) {
                    d += sign * p * determinant2d_recursive<T, Internal_buffer, Internal_allocator>(excluded(mat, { 0, j }), k);
                }
                sign *= T{ -1 };
            }
            return d;
        }
        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> determinant(const Matrix<T, Internal_buffer, Internal_allocator>& mat)
        {
            ERROC_EXPECT(!empty(mat), std::invalid_argument, "no determinant for emtpy matrix");
            ERROC_EXPECT(mat.header().dims.m == mat.header().dims.n, std::invalid_argument, "not squared matrix");

            Matrix<T, Internal_buffer, Internal_allocator> det{ {1, 1, mat.header().dims.p} };

            for (std::size_t k = 0; k < mat.header().dims.p; ++k) {
                det({ 0, 0, k }) = determinant2d_recursive(mat, k);
            }

            return det;
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> inversed(const Matrix<T, Internal_buffer, Internal_allocator>& mat)
        {
            ERROC_EXPECT(!empty(mat), std::invalid_argument, "no determinant for emtpy matrix");
            ERROC_EXPECT(mat.header().dims.m == mat.header().dims.n, std::invalid_argument, "not squared matrix");

            std::size_t n = mat.header().dims.n;

            Matrix<T, Internal_buffer, Internal_allocator> d{ determinant(mat) };
            for (std::size_t k = 0; k < mat.header().dims.p; ++k) {
                ERROC_EXPECT(d({ 0, 0, k }) != T{ 0 }, std::invalid_argument, "zero determinant");
            }

            Matrix<T, Internal_buffer, Internal_allocator> inv{ mat.header().dims };

            for (std::size_t i = 0; i < n; ++i) {
                T sign = (i + 1) % 2 == 0 ? T{ -1 } : T{ 1 };
                for (std::size_t j = 0; j < n; ++j) {
                    copy(sign * determinant(excluded(mat, { i, j })), inv({ i, j, 0 }, { 1, 1, mat.header().dims.p }));
                    sign *= T{ -1 };
                }
            }

            for (std::size_t k = 0; k < mat.header().dims.p; ++k) {
                Matrix<T, Internal_buffer, Internal_allocator> slice{ inv({ 0, 0, k }, {inv.header().dims.n, inv.header().dims.m, 1}) };
                slice *= T{ 1 / d({0, 0, k}) };
            }

            return transposed(inv);
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> swap_rows(Matrix<T, Internal_buffer, Internal_allocator>& mat, std::size_t ri1, std::size_t ri2)
        {
            ERROC_EXPECT(ri1 < mat.header().dims.n && ri2 < mat.header().dims.n, std::out_of_range, "out of range indices");

            for (std::size_t k = 0; k < mat.header().dims.p; ++k) {
                for (std::size_t j = 0; j < mat.header().dims.m; ++j) {
                    T tmp{ mat({ri1, j, k}) };
                    mat({ ri1, j, k }) = mat({ ri2, j, k });
                    mat({ ri2, j, k }) = tmp;
                }
            }

            return mat;
        }
        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> swap_rows(Matrix<T, Internal_buffer, Internal_allocator>&& mat, std::size_t ri1, std::size_t ri2)
        {
            return swap_rows<T, Internal_buffer, Internal_allocator>(std::ref(mat), ri1, ri2);
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> add_to_row(Matrix<T, Internal_buffer, Internal_allocator>& mat, std::size_t sri, std::size_t dri, const T& factor = T{1})
        {
            ERROC_EXPECT(sri < mat.header().dims.n&& dri < mat.header().dims.n, std::out_of_range, "out of range indices");

            for (std::size_t k = 0; k < mat.header().dims.p; ++k) {
                for (std::size_t j = 0; j < mat.header().dims.m; ++j) {
                    mat({ dri, j, k }) += (factor * mat({ sri, j, k }));
                }
            }

            return mat;
        }
        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> add_to_row(Matrix<T, Internal_buffer, Internal_allocator>&& mat, std::size_t sri, std::size_t dri, const T& factor = T{ 1 })
        {
            return add_to_row<T, Internal_buffer, Internal_allocator>(std::ref(mat), sri, dri, factor);
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> multiply_row(Matrix<T, Internal_buffer, Internal_allocator>& mat, std::size_t ri, const T& factor)
        {
            ERROC_EXPECT(ri < mat.header().dims.n, std::out_of_range, "out of range indices");

            for (std::size_t k = 0; k < mat.header().dims.p; ++k) {
                for (std::size_t j = 0; j < mat.header().dims.m; ++j) {
                    mat({ ri, j, k }) *= factor;
                }
            }

            return mat;
        }
        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> multiply_row(Matrix<T, Internal_buffer, Internal_allocator>&& mat, std::size_t ri, const T& factor)
        {
            return multiply_row<T, Internal_buffer, Internal_allocator>(std::ref(mat), ri, factor);
        }

        template <Number T, typename Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> reduced_row_echelon_form(Matrix<T, Internal_buffer, Internal_allocator>& mat)
        {
            Matrix<T, Internal_buffer, Internal_allocator> rref_mat{ mat };

            std::size_t r = mat.header().dims.n > mat.header().dims.m ? mat.header().dims.m : mat.header().dims.n;

            for (std::size_t t = 0; t < mat.header().dims.p; ++t) {

                for (std::size_t k = 0; k < r; ++k) {
                    if (rref_mat({ k, k, t }) != T{ 0 }) {
                        for (std::size_t i = k + 1; i < mat.header().dims.n; ++i) {
                            if (rref_mat({ i, k, t }) != T{ 0 }) {
                                swap_rows(rref_mat({ 0, 0, t }, { mat.header().dims.n, mat.header().dims.m, 1 }), k, i);
                            }
                        }
                    }
                    if (rref_mat({ k, k, t }) != T{ 0 }) {
                        multiply_row(rref_mat({ 0, 0, t }, { mat.header().dims.n, mat.header().dims.m, 1 }), k, T{ 1 } / rref_mat({ k, k, t }));
                        for (std::size_t i = 0; i < mat.header().dims.n; ++i) {
                            if (i != k) {
                                add_to_row(rref_mat({ 0, 0, t }, { mat.header().dims.n, mat.header().dims.m, 1 }), k, i, -rref_mat({ i, k, t }));
                            }
                        }
                    }
                }

            }

            return rref_mat;
        }
    }

    using details::excluded;
    using details::transposed;
    using details::determinant;
    using details::inversed;
    using details::swap_rows;
    using details::add_to_row;
    using details::multiply_row;
    using details::reduced_row_echelon_form;
}

#endif // COMPUTOC_LINEAR_ALGEBRA_H