#ifndef MATH_TYPES_MATRIX_H
#define MATH_TYPES_MATRIX_H

#include <cstddef>
#include <type_traits>
#include <stdexcept>
#include <utility>

//#include <iostream>
//#include <ostream>

#include <math/core/allocators.h>
#include <math/core/buffers.h>
#include <math/core/utils.h>

namespace math::core::types {
    // Every matrix with size less or equal to 9 will be allocated on stack
    template <typename T>
    using Matrix_buffer = math::core::buffers::Typed_buffer<T, math::core::buffers::Fallback_buffer<
        math::core::buffers::Stack_buffer<9 * sizeof(T)>,
        math::core::buffers::Allocated_buffer<math::core::allocators::Malloc_allocator, true>>>;

    template <typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;

    struct Dimensions {
        std::size_t n{ 0 };
        std::size_t m{ 0 };
    };

    template <Arithmetic T, math::core::buffers::T_buffer<T> Internal_buffer = Matrix_buffer<T>>
    class Matrix {
    public:
        Matrix(Dimensions dims, const T* data)
            : dims_{ dims }, buff_(dims.n* dims.m, data)
        {
            CORE_EXPECT(dims_.n* dims_.m != 0, std::invalid_argument, "non-positive dimensions (n_ = %d, m_ = %d)", dims_.n, dims_.m);
            CORE_EXPECT(data, std::invalid_argument, "data is null");
            CORE_EXPECT(buff_.usable(), std::runtime_error, "internal buffer failed");
        }

        Matrix(Dimensions dims, const T& value)
            : dims_{ dims }, buff_(dims.n* dims.m)
        {
            CORE_EXPECT(dims_.n* dims_.m != 0, std::invalid_argument, "non-positive dimensions (n_ = %d, m_ = %d)", dims_.n, dims_.m);
            CORE_EXPECT(buff_.usable(), std::runtime_error, "internal buffer failed");

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
            CORE_EXPECT(i < dims_.n&& j < dims_.m, std::out_of_range, "out of range indices (i = %d, j = %d)", i, j);
            return buff_.data().p[i * dims_.m + j];
        }

        T& operator()(std::size_t i, std::size_t j)
        {
            CORE_EXPECT(i < dims_.n&& j < dims_.m, std::out_of_range, "out of range indices (i = %d, j = %d)", i, j);
            return buff_.data().p[i * dims_.m + j];
        }

        template <Arithmetic T_o, math::core::buffers::T_buffer<T_o> Internal_buffer_o>
        friend bool operator==(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

        Matrix<T, Internal_buffer> get_slice(std::size_t si, std::size_t sj, Dimensions dims) const
        {
            CORE_EXPECT(si + dims.n <= dims_.n && sj + dims.m <= dims_.m, std::out_of_range, "out of range indices (si = %d, sj = %d, dims.n = %d, dims.m = %d)", si, sj, dims.n, dims.m);

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
            CORE_EXPECT(pi < dims_.n&& pj < dims_.m, std::out_of_range, "out of range indices (pi = %d, pj = %d)", pi, pj);

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
            CORE_EXPECT(si + mat.dims_.n <= dims_.n && sj + mat.dims_.m <= dims_.m, std::out_of_range, "out of range indices (si = %d, sj = %d, mat.dims_.n = %d, mat.dims_.m = %d)", si, sj, mat.dims_.n, mat.dims_.m);

            for (std::size_t i = 0; i < mat.dims_.n; ++i) {
                for (std::size_t j = 0; j < mat.dims_.m; ++j) {
                    buff_.data().p[(si + i) * dims_.m + sj + j] = mat.buff_.data().p[i * mat.dims_.m + j];
                }
            }
            return *this;
        }

        template <Arithmetic T_o, math::core::buffers::T_buffer<T_o> Internal_buffer_o>
        friend Matrix<T_o, Internal_buffer_o> operator+(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

        Matrix<T, Internal_buffer>& operator+=(const Matrix<T, Internal_buffer>& other)
        {
            CORE_EXPECT(dims_.n == other.dims_.n && dims_.m == other.dims_.m, std::invalid_argument, "matrices size mismatch (dims_.n = %d, other.dims_.n = %d, dims_.m = %d, other.dims_.m = %d)", dims_.n, other.dims_.n, dims_.m, other.dims_.m);

            for (std::size_t i = 0; i < buff_.data().s; ++i) {
                buff_.data().p[i] += other.buff_.data().p[i];
            }
            return *this;
        }

        template <Arithmetic T_o, math::core::buffers::T_buffer<T_o> Internal_buffer_o>
        friend Matrix<T_o, Internal_buffer_o> operator-(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

        Matrix<T, Internal_buffer>& operator-=(const Matrix<T, Internal_buffer>& other)
        {
            CORE_EXPECT(dims_.n == other.dims_.n && dims_.m == other.dims_.m, std::invalid_argument, "matrices size mismatch (dims_.n = %d, other.dims_.n = %d, dims_.m = %d, other.dims_.m = %d)", dims_.n, other.dims_.n, dims_.m, other.dims_.m);

            for (std::size_t i = 0; i < buff_.data().s; ++i) {
                buff_.data().p[i] -= other.buff_.data().p[i];
            }
            return *this;
        }

        template <Arithmetic T_o, math::core::buffers::T_buffer<T_o> Internal_buffer_o>
        friend Matrix<T_o, Internal_buffer_o> operator*(const Matrix<T_o, Internal_buffer_o>& lhs, const T_o& rhs);

        template <Arithmetic T_o, math::core::buffers::T_buffer<T_o> Internal_buffer_o>
        friend Matrix<T_o, Internal_buffer_o> operator*(const T_o& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

        Matrix<T, Internal_buffer>& operator*=(const T& other)
        {
            for (std::size_t i = 0; i < buff_.data().s; ++i) {
                buff_.data().p[i] *= other;
            }
            return *this;
        }

        template <Arithmetic T_o, math::core::buffers::T_buffer<T_o> Internal_buffer_o>
        friend Matrix<T_o, Internal_buffer_o> operator*(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

        Matrix<T, Internal_buffer>& operator*=(const Matrix<T, Internal_buffer>& other)
        {
            CORE_EXPECT(dims_.m == other.dims_.n, std::invalid_argument, "matrices size mismatch (dims_.m = %d, other.dims_.n = %d)", dims_.m, other.dims_.n);

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

    private:
        Dimensions dims_{};
        Internal_buffer buff_{};
    };

    template <Arithmetic T, math::core::buffers::T_buffer<T> Internal_buffer>
    inline bool operator==(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
    {
        if (lhs.dims_.n != rhs.dims_.n || lhs.dims_.m != rhs.dims_.m) {
            return false;
        }

        for (std::size_t i = 0; i < lhs.buff_.data().s; ++i) {
            if (lhs.buff_.data().p[i] != rhs.buff_.data().p[i]) {
                return false;
            }
        }
        return true;
    }

    template <Arithmetic T, math::core::buffers::T_buffer<T> Internal_buffer>
    inline Matrix<T, Internal_buffer> operator+(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
    {
        CORE_EXPECT(lhs.dims_.n == rhs.dims_.n && lhs.dims_.m == rhs.dims_.m, std::invalid_argument, "matrices size mismatch (lhs.dims_.n = %d, rhs.dims_.n = %d, lhs.dims_.m = %d, rhs.dims_.m = %d)", lhs.dims_.n, rhs.dims_.n, lhs.dims_.m, rhs.dims_.m);

        Matrix<T, Internal_buffer> sum{lhs};

        for (std::size_t i = 0; i < sum.buff_.data().s; ++i) {
            sum.buff_.data().p[i] += rhs.buff_.data().p[i];
        }
        return sum;
    }

    template <Arithmetic T, math::core::buffers::T_buffer<T> Internal_buffer>
    inline Matrix<T, Internal_buffer> operator-(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
    {
        CORE_EXPECT(lhs.dims_.n == rhs.dims_.n && lhs.dims_.m == rhs.dims_.m, std::invalid_argument, "matrices size mismatch (lhs.dims_.n = %d, rhs.dims_.n = %d, lhs.dims_.m = %d, rhs.dims_.m = %d)", lhs.dims_.n, rhs.dims_.n, lhs.dims_.m, rhs.dims_.m);

        Matrix<T, Internal_buffer> subtraction{ lhs };

        for (std::size_t i = 0; i < subtraction.buff_.data().s; ++i) {
            subtraction.buff_.data().p[i] -= rhs.buff_.data().p[i];
        }
        return subtraction;
    }

    template <Arithmetic T, math::core::buffers::T_buffer<T> Internal_buffer>
    inline Matrix<T, Internal_buffer> operator*(const Matrix<T, Internal_buffer>& lhs, const T& rhs)
    {
        Matrix<T, Internal_buffer> multiplication{ lhs };

        for (std::size_t i = 0; i < multiplication.buff_.data().s; ++i) {
            multiplication.buff_.data().p[i] *= rhs;
        }
        return multiplication;
    }

    template <Arithmetic T, math::core::buffers::T_buffer<T> Internal_buffer>
    inline Matrix<T, Internal_buffer> operator*(const T& lhs, const Matrix<T, Internal_buffer>& rhs)
    {
        return operator*(rhs, lhs);
    }

    template <Arithmetic T, math::core::buffers::T_buffer<T> Internal_buffer>
    inline Matrix<T, Internal_buffer> operator*(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
    {
        CORE_EXPECT(lhs.dims_.m == rhs.dims_.n, std::invalid_argument, "matrices size mismatch (lhs.dims_.m = %d, rhs.dims_.n = %d)", lhs.dims_.m, rhs.dims_.n);

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

    //template <Arithmetic T_o, math::core::buffers::T_buffer<T_o> Internal_buffer_o>
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

#endif // MATH_TYPE_MATRIX_H

