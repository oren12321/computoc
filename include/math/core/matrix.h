#ifndef MATH_TYPES_MATRIX_H
#define MATH_TYPES_MATRIX_H

#include <cstddef>
#include <type_traits>
#include <stdexcept>

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

        Dimensions dimensions()
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

        Matrix<T, Internal_buffer>& set_slice(std::size_t si, std::size_t ei, std::size_t sj, std::size_t ej, const Matrix<T, Internal_buffer> mat)
        {
            CORE_EXPECT(si <= ei && ei < dims_.n&& sj <= ej && ej < dims_.m, std::out_of_range, "out of range indices (si = %d, ei = %d, sj = %d, ej = %d)", si, ei, sj, ej);

            std::size_t sliced_n = ei - si + 1;
            std::size_t sliced_m = ej - sj + 1;

            CORE_EXPECT(sliced_n == mat.dims_.n && sliced_m == mat.dims_.m, std::invalid_argument, "input mat size and slice size mismatch (slice_n = %d, input_n = %d, slice_m = %d, input_m = %d)", sliced_n, mat.dims_.n, sliced_m, mat.dims_.m);

            for (std::size_t i = 0; i < sliced_n; ++i) {
                for (std::size_t j = 0; j < sliced_m; ++j) {
                    buff_.data().p[(si + i) * dims_.m + sj + j] = mat.buff_.data().p[i * mat.dims_.m + j];
                }
            }
            return *this;
        }

        Matrix<T, Internal_buffer> get_slice(std::size_t si, std::size_t ei, std::size_t sj, std::size_t ej) const
        {
            CORE_EXPECT(si <= ei && ei < dims_.n&& sj <= ej && ej < dims_.m, std::out_of_range, "out of range indices (si = %d, ei = %d, sj = %d, ej = %d)", si, ei, sj, ej);

            std::size_t sliced_n = ei - si + 1;
            std::size_t sliced_m = ej - sj + 1;

            Matrix<T, Internal_buffer> sliced{ {sliced_n, sliced_m}, T{} };

            for (std::size_t i = 0; i < sliced_n; ++i) {
                for (std::size_t j = 0; j < sliced_m; ++j) {
                    sliced.buff_.data().p[i * sliced_m + j] = buff_.data().p[(si + i) * dims_.m + sj + j];
                }
            }
            return sliced;
        }

        template <Arithmetic T_o, math::core::buffers::T_buffer<T_o> Internal_Buffer_o>
        friend bool operator==(const Matrix<T_o, Internal_Buffer_o>& lhs, const Matrix<T_o, Internal_Buffer_o>& rhs);

    private:
        Dimensions dims_{};
        Internal_buffer buff_{};
    };

    template <Arithmetic T, math::core::buffers::T_buffer<T> Internal_Buffer>
    inline bool operator==(const Matrix<T, Internal_Buffer>& lhs, const Matrix<T, Internal_Buffer>& rhs)
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
}

#endif // MATH_TYPE_MATRIX_H

