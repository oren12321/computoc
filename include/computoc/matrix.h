#ifndef COMPUTOC_TYPES_MATRIX_H
#define COMPUTOC_TYPES_MATRIX_H

#include <cstddef>
#include <stdexcept>
#include <utility>

//#include <iostream>
//#include <ostream>

#include <memoc/allocators.h>
#include <memoc/buffers.h>
#include <computoc/errors.h>
#include <computoc/algorithms.h>
#include <computoc/concepts.h>

namespace computoc::types {
    namespace details {
        // Every matrix with size less or equal to 9 will be allocated on stack
        template <typename T>
        using Matrix_buffer = memoc::buffers::Typed_buffer<T, memoc::buffers::Fallback_buffer<
            memoc::buffers::Stack_buffer<9 * sizeof(T)>,
            memoc::buffers::Allocated_buffer<memoc::allocators::Malloc_allocator, true>>>;

        struct Dimensions {
            std::size_t n{ 0 };
            std::size_t m{ 0 };
        };

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer = Matrix_buffer<T>>
        class Matrix {
        public:
            Matrix(Dimensions dims, const T* data)
                : dims_{ dims }, buff_(dims.n* dims.m, data)
            {
                COMPUTOC_THROW_IF_FALSE(dims_.n* dims_.m != 0, std::invalid_argument, "non-positive dimensions (n_ = %d, m_ = %d)", dims_.n, dims_.m);
                COMPUTOC_THROW_IF_FALSE(data, std::invalid_argument, "data is null");
                COMPUTOC_THROW_IF_FALSE(buff_.usable(), std::runtime_error, "internal buffer failed");
            }

            Matrix(Dimensions dims, const T& value)
                : dims_{ dims }, buff_(dims.n* dims.m)
            {
                COMPUTOC_THROW_IF_FALSE(dims_.n* dims_.m != 0, std::invalid_argument, "non-positive dimensions (n_ = %d, m_ = %d)", dims_.n, dims_.m);
                COMPUTOC_THROW_IF_FALSE(buff_.usable(), std::runtime_error, "internal buffer failed");

                for (std::size_t i = 0; i < buff_.data().s; ++i) {
                    buff_.data().p[i] = value;
                }
            }

            Dimensions dimensions() const
            {
                return dims_;
            }

            const T& operator()(std::size_t i, std::size_t j) const
            {
                COMPUTOC_THROW_IF_FALSE(i < dims_.n&& j < dims_.m, std::out_of_range, "out of range indices (i = %d, j = %d)", i, j);
                return buff_.data().p[i * dims_.m + j];
            }

            T& operator()(std::size_t i, std::size_t j)
            {
                COMPUTOC_THROW_IF_FALSE(i < dims_.n&& j < dims_.m, std::out_of_range, "out of range indices (i = %d, j = %d)", i, j);
                return buff_.data().p[i * dims_.m + j];
            }

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend bool operator==(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            Matrix<T, Internal_buffer> get_slice(std::size_t si, std::size_t sj, Dimensions dims) const
            {
                COMPUTOC_THROW_IF_FALSE(si + dims.n <= dims_.n && sj + dims.m <= dims_.m, std::out_of_range, "out of range indices (si = %d, sj = %d, dims.n = %d, dims.m = %d)", si, sj, dims.n, dims.m);

                Matrix<T, Internal_buffer> slice{ dims, T{} };

                for (std::size_t i = 0; i < dims.n; ++i) {
                    for (std::size_t j = 0; j < dims.m; ++j) {
                        slice.buff_.data().p[i * dims.m + j] = buff_.data().p[(si + i) * dims_.m + sj + j];
                    }
                }
                return slice;
            }

            Matrix<T, Internal_buffer> get_slice(std::size_t pi, std::size_t pj) const
            {
                COMPUTOC_THROW_IF_FALSE(pi < dims_.n&& pj < dims_.m, std::out_of_range, "out of range indices (pi = %d, pj = %d)", pi, pj);

                Matrix<T, Internal_buffer> slice{ {dims_.n - 1, dims_.m - 1}, T{} };

                for (std::size_t i = 0; i < pi; ++i) {
                    for (std::size_t j = 0; j < pj; ++j) {
                        slice.buff_.data().p[i * slice.dims_.m + j] = buff_.data().p[i * dims_.m + j];
                    }
                }

                for (std::size_t i = 0; i < pi; ++i) {
                    for (std::size_t j = pj + 1; j < dims_.m; ++j) {
                        slice.buff_.data().p[i * slice.dims_.m + (j - 1)] = buff_.data().p[i * dims_.m + j];
                    }
                }

                for (std::size_t i = pi + 1; i < dims_.n; ++i) {
                    for (std::size_t j = 0; j < pj; ++j) {
                        slice.buff_.data().p[(i - 1) * slice.dims_.m + j] = buff_.data().p[i * dims_.m + j];
                    }
                }

                for (std::size_t i = pi + 1; i < dims_.n; ++i) {
                    for (std::size_t j = pj + 1; j < dims_.m; ++j) {
                        slice.buff_.data().p[(i - 1) * slice.dims_.m + (j - 1)] = buff_.data().p[i * dims_.m + j];
                    }
                }

                return slice;
            }

            Matrix<T, Internal_buffer>& set_slice(std::size_t si, std::size_t sj, const Matrix<T, Internal_buffer>& mat)
            {
                COMPUTOC_THROW_IF_FALSE(si + mat.dims_.n <= dims_.n && sj + mat.dims_.m <= dims_.m, std::out_of_range, "out of range indices (si = %d, sj = %d, mat.dims_.n = %d, mat.dims_.m = %d)", si, sj, mat.dims_.n, mat.dims_.m);

                for (std::size_t i = 0; i < mat.dims_.n; ++i) {
                    for (std::size_t j = 0; j < mat.dims_.m; ++j) {
                        buff_.data().p[(si + i) * dims_.m + sj + j] = mat.buff_.data().p[i * mat.dims_.m + j];
                    }
                }
                return *this;
            }

            Matrix<T, Internal_buffer> operator-()
            {
                Matrix<T, Internal_buffer> neg{ { dims_.n, dims_.m }, T{} };
                for (std::size_t i = 0; i < buff_.data().s; ++i) {
                    neg.buff_.data().p[i] = -buff_.data().p[i];
                }
                return neg;
            }

            Matrix<T, Internal_buffer> operator+()
            {
                return *this;
            }

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> operator+(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            Matrix<T, Internal_buffer>& operator+=(const Matrix<T, Internal_buffer>& other)
            {
                COMPUTOC_THROW_IF_FALSE(dims_.n == other.dims_.n && dims_.m == other.dims_.m, std::invalid_argument, "matrices size mismatch (dims_.n = %d, other.dims_.n = %d, dims_.m = %d, other.dims_.m = %d)", dims_.n, other.dims_.n, dims_.m, other.dims_.m);

                for (std::size_t i = 0; i < buff_.data().s; ++i) {
                    buff_.data().p[i] += other.buff_.data().p[i];
                }
                return *this;
            }

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> operator-(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            Matrix<T, Internal_buffer>& operator-=(const Matrix<T, Internal_buffer>& other)
            {
                COMPUTOC_THROW_IF_FALSE(dims_.n == other.dims_.n && dims_.m == other.dims_.m, std::invalid_argument, "matrices size mismatch (dims_.n = %d, other.dims_.n = %d, dims_.m = %d, other.dims_.m = %d)", dims_.n, other.dims_.n, dims_.m, other.dims_.m);

                for (std::size_t i = 0; i < buff_.data().s; ++i) {
                    buff_.data().p[i] -= other.buff_.data().p[i];
                }
                return *this;
            }

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> operator*(const Matrix<T_o, Internal_buffer_o>& lhs, const T_o& rhs);

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> operator*(const T_o& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            Matrix<T, Internal_buffer>& operator*=(const T& other)
            {
                for (std::size_t i = 0; i < buff_.data().s; ++i) {
                    buff_.data().p[i] *= other;
                }
                return *this;
            }

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> operator*(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            Matrix<T, Internal_buffer>& operator*=(const Matrix<T, Internal_buffer>& other)
            {
                COMPUTOC_THROW_IF_FALSE(dims_.m == other.dims_.n, std::invalid_argument, "matrices size mismatch (dims_.m = %d, other.dims_.n = %d)", dims_.m, other.dims_.n);

                Matrix<T, Internal_buffer> multiplication{ {dims_.n, other.dims_.m}, T{} };

                for (std::size_t i = 0; i < multiplication.dims_.n; ++i) {
                    for (std::size_t k = 0; k < dims_.m; ++k) {
                        for (std::size_t j = 0; j < multiplication.dims_.m; ++j) {
                            multiplication.buff_.data().p[i * multiplication.dims_.m + j] += buff_.data().p[i * dims_.m + k] * other.buff_.data().p[k * other.dims_.m + j];
                        }
                    }
                }
                *this = std::move(multiplication);
                return *this;
            }

            Matrix<T, Internal_buffer> transposed() const
            {
                Matrix<T, Internal_buffer> tmat{ {dims_.m, dims_.n}, T{} };

                for (std::size_t i = 0; i < tmat.dims_.n; ++i) {
                    for (std::size_t j = 0; j < tmat.dims_.m; ++j) {
                        tmat.buff_.data().p[i * tmat.dims_.m + j] = buff_.data().p[j * dims_.m + i];
                    }
                }
                return tmat;
            }

            Matrix<T, Internal_buffer>& transpose()
            {
                Matrix<T, Internal_buffer> tmat{ {dims_.m, dims_.n}, T{} };

                for (std::size_t i = 0; i < tmat.dims_.n; ++i) {
                    for (std::size_t j = 0; j < tmat.dims_.m; ++j) {
                        tmat.buff_.data().p[i * tmat.dims_.m + j] = buff_.data().p[j * dims_.m + i];
                    }
                }
                *this = std::move(tmat);
                return *this;
            }

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend T_o determinant(const Matrix<T_o, Internal_buffer_o>& mat);

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> inverse(const Matrix<T_o, Internal_buffer_o>& mat);

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> merge_horizontal(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> merge_vertical(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            Matrix<T, Internal_buffer>& multiply_row(std::size_t i, const T& factor)
            {
                COMPUTOC_THROW_IF_FALSE(i < dims_.n, std::out_of_range, "out of range index (i = %d)", i);

                for (std::size_t j = 0; j < dims_.m; ++j) {
                    buff_.data().p[i * dims_.m + j] *= factor;
                }
                return *this;
            }

            Matrix<T, Internal_buffer>& add_row(std::size_t si, std::size_t di, const T& factor)
            {
                COMPUTOC_THROW_IF_FALSE(si < dims_.n&& di < dims_.n, std::out_of_range, "out of range indices (si = %d, di = %d)", si, di);

                for (std::size_t j = 0; j < dims_.m; ++j) {
                    buff_.data().p[di * dims_.m + j] += factor * buff_.data().p[si * dims_.m + j];
                }
                return *this;
            }

            Matrix<T, Internal_buffer>& swap_rows(std::size_t i1, std::size_t i2)
            {
                COMPUTOC_THROW_IF_FALSE(i1 < dims_.n&& i2 < dims_.n, std::out_of_range, "out of range indices (i1 = %d, i2 = %d)", i1, i2);

                for (std::size_t j = 0; j < dims_.m; ++j) {
                    T temp{ buff_.data().p[i1 * dims_.m + j] };
                    buff_.data().p[i1 * dims_.m + j] = buff_.data().p[i2 * dims_.m + j];
                    buff_.data().p[i2 * dims_.m + j] = temp;
                }
                return *this;
            }

            //template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            //friend Matrix<T_o, Internal_buffer_o> row_echelon_form(const Matrix<T_o, Internal_buffer_o>& mat);

            template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> reduced_row_echelon_form(const Matrix<T_o, Internal_buffer_o>& mat);

        private:
            Dimensions dims_{};
            Internal_buffer buff_{};
        };

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline bool operator==(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            if (lhs.dims_.n != rhs.dims_.n || lhs.dims_.m != rhs.dims_.m) {
                return false;
            }

            for (std::size_t i = 0; i < lhs.buff_.data().s; ++i) {
                if (!algorithms::is_equal(lhs.buff_.data().p[i], rhs.buff_.data().p[i])) {
                    return false;
                }
            }
            return true;
        }

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> operator+(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.dims_.n == rhs.dims_.n && lhs.dims_.m == rhs.dims_.m, std::invalid_argument, "matrices size mismatch (lhs.dims_.n = %d, rhs.dims_.n = %d, lhs.dims_.m = %d, rhs.dims_.m = %d)", lhs.dims_.n, rhs.dims_.n, lhs.dims_.m, rhs.dims_.m);

            Matrix<T, Internal_buffer> sum{ lhs };

            for (std::size_t i = 0; i < sum.buff_.data().s; ++i) {
                sum.buff_.data().p[i] += rhs.buff_.data().p[i];
            }
            return sum;
        }

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> operator-(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.dims_.n == rhs.dims_.n && lhs.dims_.m == rhs.dims_.m, std::invalid_argument, "matrices size mismatch (lhs.dims_.n = %d, rhs.dims_.n = %d, lhs.dims_.m = %d, rhs.dims_.m = %d)", lhs.dims_.n, rhs.dims_.n, lhs.dims_.m, rhs.dims_.m);

            Matrix<T, Internal_buffer> subtraction{ lhs };

            for (std::size_t i = 0; i < subtraction.buff_.data().s; ++i) {
                subtraction.buff_.data().p[i] -= rhs.buff_.data().p[i];
            }
            return subtraction;
        }

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> operator*(const Matrix<T, Internal_buffer>& lhs, const T& rhs)
        {
            Matrix<T, Internal_buffer> multiplication{ lhs };

            for (std::size_t i = 0; i < multiplication.buff_.data().s; ++i) {
                multiplication.buff_.data().p[i] *= rhs;
            }
            return multiplication;
        }

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> operator*(const T& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            return operator*(rhs, lhs);
        }

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> operator*(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.dims_.m == rhs.dims_.n, std::invalid_argument, "matrices size mismatch (lhs.dims_.m = %d, rhs.dims_.n = %d)", lhs.dims_.m, rhs.dims_.n);

            Matrix<T, Internal_buffer> multiplication{ {lhs.dims_.n, rhs.dims_.m}, T{} };

            for (std::size_t i = 0; i < multiplication.dims_.n; ++i) {
                for (std::size_t k = 0; k < lhs.dims_.m; ++k) {
                    for (std::size_t j = 0; j < multiplication.dims_.m; ++j) {
                        multiplication.buff_.data().p[i * multiplication.dims_.m + j] += lhs.buff_.data().p[i * lhs.dims_.m + k] * rhs.buff_.data().p[k * rhs.dims_.m + j];
                    }
                }
            }
            return multiplication;
        }

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline T determinant(const Matrix<T, Internal_buffer>& mat)
        {
            COMPUTOC_THROW_IF_FALSE(mat.dims_.n == mat.dims_.m, std::invalid_argument, "not square matrix");

            std::size_t n = mat.dims_.n;

            if (n == 1) {
                return mat.buff_.data().p[0];
            }

            if (n == 2) {
                return mat.buff_.data().p[0] * mat.buff_.data().p[3] - mat.buff_.data().p[1] * mat.buff_.data().p[2];
            }

            int sign = T{ 1 };
            T d{ 0 };
            for (std::size_t j = 0; j < n; ++j) {
                T p{ mat.buff_.data().p[j] };
                if (!algorithms::is_equal(p, T{ 0 })) {
                    d += sign * p * determinant<T, Internal_buffer>(mat.get_slice(0, j));
                }
                sign *= T{ -1 };
            }
            return d;
        }

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> inverse(const Matrix<T, Internal_buffer>& mat)
        {
            COMPUTOC_THROW_IF_FALSE(mat.dims_.n == mat.dims_.m, std::invalid_argument, "not square matrix");
            std::size_t n = mat.dims_.n;

            T d{ determinant(mat) };
            COMPUTOC_THROW_IF_FALSE(!algorithms::is_equal(d, T{ 0 }), std::invalid_argument, "zero determinant");

            Matrix<T, Internal_buffer> inv(mat.dims_, T{});

            for (std::size_t i = 0; i < n; ++i) {
                T sign = (i + 1) % 2 == 0 ? T{ -1 } : T{ 1 };
                for (std::size_t j = 0; j < n; ++j) {
                    inv.buff_.data().p[i * n + j] = sign * determinant(mat.get_slice(i, j));
                    sign *= T{ -1 };
                }
            }

            inv.transpose();

            return T{ 1 / d } *inv;
        }

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> merge_horizontal(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.dims_.n == rhs.dims_.n, std::invalid_argument, "dimensions mismatch (lhs.dims_.n = %d, rhs.dims_.n = %d)", lhs.dims_.n, rhs.dims_.n);

            Matrix<T, Internal_buffer> merged{ {lhs.dims_.n, lhs.dims_.m + rhs.dims_.m}, T{} };

            merged.set_slice(0, 0, lhs);
            merged.set_slice(0, lhs.dims_.m, rhs);

            return merged;
        }

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> merge_vertical(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.dims_.m == rhs.dims_.m, std::invalid_argument, "dimensions mismatch (lhs.dims_.m = %d, rhs.dims_.m = %d)", lhs.dims_.m, rhs.dims_.m);

            Matrix<T, Internal_buffer> merged{ {lhs.dims_.n + rhs.dims_.n, lhs.dims_.m}, T{} };

            merged.set_slice(0, 0, lhs);
            merged.set_slice(lhs.dims_.n, 0, rhs);

            return merged;
        }

        //template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        //inline Matrix<T, Internal_buffer> row_echelon_form(const Matrix<T, Internal_buffer>& mat)
        //{
        //    Matrix<T, Internal_buffer> rref_mat{ mat };

        //    std::size_t r = mat.dims_.n > mat.dims_.m ? mat.dims_.m : mat.dims_.n;

        //    for (std::size_t k = 0; k < r - 1; ++k) {
        //        if (algorithms::is_equal(rref_mat(k, k), T{ 0 })) {
        //            for (std::size_t i = k + 1; i < mat.dims_.n; ++i) {
        //                if (!algorithms::is_equal(rref_mat(i, k), T{ 0 })) {
        //                    rref_mat.swap_rows(k, i);
        //                }
        //            }
        //        }
        //        if (!algorithms::is_equal(rref_mat(k, k), T{ 0 })) {
        //            rref_mat.multiply_row(k, T{ 1 } / rref_mat(k, k));
        //            for (std::size_t i = k + 1; i < mat.dims_.n; ++i) {
        //                if (i != k) {
        //                    rref_mat.add_row(k, i, -rref_mat(i, k));
        //                }
        //            }
        //        }
        //    }

        //    return rref_mat;
        //}

        template <concepts::Arithmetic T, memoc::buffers::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> reduced_row_echelon_form(const Matrix<T, Internal_buffer>& mat)
        {
            Matrix<T, Internal_buffer> rref_mat{ mat };

            std::size_t r = mat.dims_.n > mat.dims_.m ? mat.dims_.m : mat.dims_.n;

            for (std::size_t k = 0; k < r; ++k) {
                if (algorithms::is_equal(rref_mat(k, k), T{ 0 })) {
                    for (std::size_t i = k + 1; i < mat.dims_.n; ++i) {
                        if (!algorithms::is_equal(rref_mat(i, k), T{ 0 })) {
                            rref_mat.swap_rows(k, i);
                        }
                    }
                }
                if (!algorithms::is_equal(rref_mat(k, k), T{ 0 })) {
                    rref_mat.multiply_row(k, T{ 1 } / rref_mat(k, k));
                    for (std::size_t i = 0; i < mat.dims_.n; ++i) {
                        if (i != k) {
                            rref_mat.add_row(k, i, -rref_mat(i, k));
                        }
                    }
                }
            }

            return rref_mat;
        }

        //template <concepts::Arithmetic T_o, memoc::buffers::Buffer<T_o> Internal_buffer_o>
        //std::ostream& operator<<(std::ostream& os, const Matrix<T_o, Internal_buffer_o>& mat)
        //{
        //    for (std::size_t i = 0; i < mat.dimensions().n; ++i) {
        //        for (std::size_t j = 0; j < mat.dimensions().m; ++j) {
        //            std::cout << mat(i, j) << " ";
        //        }
        //        std::cout << "\n";
        //    }
        //    return os;
        //}
    }

    using details::Dimensions;
    using details::Matrix;
}

namespace computoc {
    using types::details::determinant;
    using types::details::inverse;
    using types::details::merge_horizontal;
    using types::details::merge_vertical;
    using types::details::reduced_row_echelon_form;
    //using types::details::row_echelon_form;
}

#endif // COMPUTOC_TYPE_MATRIX_H

