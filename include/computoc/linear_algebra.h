#ifndef COMPUTOC_LINEAR_ALGEBRA_H
#define COMPUTOC_LINEAR_ALGEBRA_H

#include <cstddef>
#include <stdexcept>

#include <memoc/allocators.h>
#include <memoc/buffers.h>
#include <computoc/errors.h>
#include <computoc/concepts.h>
#include <computoc/algorithms.h>
#include <computoc/matrix.h>

namespace computoc {
    namespace details {
        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> excluded(const Matrix<T, Internal_buffer, Internal_allocator>& mat, const Inds& pivot)
        {
            COMPUTOC_THROW_IF_FALSE(!is_empty(mat), std::invalid_argument, "minor for empty matrix is invalid");
            COMPUTOC_THROW_IF_FALSE(is_inside(pivot, mat.header().dims), std::out_of_range, "pivot is not in matrix dimensions");
            COMPUTOC_THROW_IF_FALSE(mat.header().dims.n > 1 && mat.header().dims.m > 1, std::invalid_argument, "operation is undefined for 1x1 matrix");

            Matrix<T, Internal_buffer, Internal_allocator> mmat{ {mat.header().dims.n - 1, mat.header().dims.m - 1, mat.header().dims.p} };

            if (pivot.i == 0 && pivot.j == 0) {
                copy_to(mat({ 1 ,1, 0 }, { mmat.header().dims.n, mmat.header().dims.m, mat.header().dims.p }), mmat); // BR
                return mmat;
            }

            if (pivot.i == mat.header().dims.n - 1 && pivot.j == mat.header().dims.m - 1) {
                copy_to(mat({ 0, 0, 0 }, { mmat.header().dims.n, mmat.header().dims.m, mat.header().dims.p }), mmat); // UL
                return mmat;
            }

            if (pivot.i == 0 && pivot.j == mat.header().dims.m - 1) {
                copy_to(mat({ 1, 0, 0 }, { mmat.header().dims.n, mmat.header().dims.m, mat.header().dims.p }), mmat); // BL
                return mmat;
            }

            if (pivot.i == mat.header().dims.n - 1 && pivot.j == 0) {
                copy_to(mat({ 0, 1, 0 }, { mmat.header().dims.n, mmat.header().dims.m, mat.header().dims.p }), mmat); // UR
                return mmat;
            }

            if (pivot.i == 0) { // 0 < pivot.j < mat.header().dims.m - 1
                copy_to(mat({ 1, 0, 0 }, { mmat.header().dims.n, pivot.j, mat.header().dims.p }), mmat({ 0, 0, 0 }, { mmat.header().dims.n, pivot.j, mat.header().dims.p })); // BL
                copy_to(mat({ 1 ,pivot.j + 1, 0 }, { mmat.header().dims.n, mmat.header().dims.m - pivot.j, mat.header().dims.p }), mmat({ 0, pivot.j, 0 }, { mmat.header().dims.n, mmat.header().dims.m - pivot.j, mat.header().dims.p })); // BR
                return mmat;
            }

            if (pivot.j == 0) { // 0 < pivot.i < mat.header.dims.n - 1
                copy_to(mat({ 0, 1, 0 }, { pivot.i, mmat.header().dims.m, mat.header().dims.p }), mmat({ 0, 0, 0 }, { pivot.i, mmat.header().dims.m, mat.header().dims.p })); // UR
                copy_to(mat({ pivot.i + 1 , 1, 0 }, { mmat.header().dims.n - pivot.i, mmat.header().dims.m, mat.header().dims.p }), mmat({ pivot.i, 0, 0 }, { mmat.header().dims.n - pivot.i, mmat.header().dims.m, mat.header().dims.p })); // BR
                return mmat;
            }

            if (pivot.i == mat.header().dims.n - 1) { // 0 < pivot.j < mat.header().dims.m - 1
                copy_to(mat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p }), mmat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p })); // UL
                copy_to(mat({ 0, pivot.j + 1, 0 }, { pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p }), mmat({ 0, pivot.j, 0 }, { pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p })); // UR
                return mmat;
            }

            if (pivot.j == mat.header().dims.m - 1) { // 0 < pivot.i < mat.header().dims.n - 1
                copy_to(mat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p }), mmat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p })); // UL
                copy_to(mat({ pivot.i + 1, 0, 0 }, { mmat.header().dims.n - pivot.i, pivot.j, mat.header().dims.p }), mmat({ pivot.i, 0, 0 }, { mmat.header().dims.n - pivot.i, pivot.j, mat.header().dims.p })); // BL
                return mmat;
            }

            copy_to(mat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p }), mmat({ 0, 0, 0 }, { pivot.i, pivot.j, mat.header().dims.p })); // UL
            copy_to(mat({ pivot.i + 1, 0, 0 }, { mmat.header().dims.n - pivot.i, pivot.j, mat.header().dims.p }), mmat({ pivot.i, 0, 0 }, { mmat.header().dims.n - pivot.i, pivot.j, mat.header().dims.p })); // BL
            copy_to(mat({ 0, pivot.j + 1, 0 }, { pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p }), mmat({ 0, pivot.j, 0 }, { pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p })); // UR
            copy_to(mat({ pivot.i + 1 ,pivot.j + 1, 0 }, { mmat.header().dims.n - pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p }), mmat({ pivot.i, pivot.j, 0 }, { mmat.header().dims.n - pivot.i, mmat.header().dims.m - pivot.j, mat.header().dims.p })); // BR
            return mmat;
        }

        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator+(const Matrix<T, Internal_buffer, Internal_allocator>& mat)
        {
            return copy_of(mat);
        }

        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
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

        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator>& operator+=(Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.header().dims == rhs.header().dims, std::invalid_argument, "matrix should have same dimensions");

            for (std::size_t k = 0; k < lhs.header().dims.p; ++k) {
                for (std::size_t i = 0; i < lhs.header().dims.n; ++i) {
                    for (std::size_t j = 0; j < lhs.header().dims.m; ++j) {
                        lhs({ i, j, k }) += rhs({ i, j, k });
                    }
                }
            }

            return lhs;
        }
        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator+(const Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {        
            COMPUTOC_THROW_IF_FALSE(lhs.header().dims == rhs.header().dims, std::invalid_argument, "matrix should have same dimensions");

            Matrix<T, Internal_buffer, Internal_allocator> addition{ copy_of(lhs) };
            addition += rhs;

            return addition;
        }

        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator>& operator-=(Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.header().dims == rhs.header().dims, std::invalid_argument, "matrix should have same dimensions");

            for (std::size_t k = 0; k < lhs.header().dims.p; ++k) {
                for (std::size_t i = 0; i < lhs.header().dims.n; ++i) {
                    for (std::size_t j = 0; j < lhs.header().dims.m; ++j) {
                        lhs({ i, j, k }) -= rhs({ i, j, k });
                    }
                }
            }

            return lhs;
        }
        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator-(const Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.header().dims == rhs.header().dims, std::invalid_argument, "matrix should have same dimensions");

            Matrix<T, Internal_buffer, Internal_allocator> subtraction{ copy_of(lhs) };
            subtraction += rhs;

            return subtraction;
        }

        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator>& operator*=(Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.header().dims.m == rhs.header().dims.n && lhs.header().dims.p == rhs.header().dims.p, std::invalid_argument, "matrices dimensions are invalid for multiplication");

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
        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator*(const Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            Matrix<T, Internal_buffer, Internal_allocator> multiplication{ copy_of(lhs) };
            multiplication *= rhs;

            return multiplication;
        }

        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
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
        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator*(const Matrix<T, Internal_buffer, Internal_allocator>& lhs, const T& rhs)
        {
            Matrix<T, Internal_buffer, Internal_allocator> multiplication{ copy_of(lhs) };
            multiplication *= rhs;

            return multiplication;
        }
        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> operator*(const T& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            return rhs * lhs;
        }
    }

    using details::excluded;
}

#endif // COMPUTOC_LINEAR_ALGEBRA_H