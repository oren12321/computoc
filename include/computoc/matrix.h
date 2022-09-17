#ifndef COMPUTOC_TYPES_MATRIX_H
#define COMPUTOC_TYPES_MATRIX_H

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <ostream>

//#include <iostream>
//#include <ostream>

#include <memoc/allocators.h>
#include <memoc/buffers.h>
#include <computoc/errors.h>
#include <computoc/algorithms.h>

#include <memoc/pointers.h>

namespace computoc {
    namespace details {
        struct Dims {
            std::size_t n{ 0 }, m{ 0 }, p{ 1 };
        };

        inline bool is_empty(const Dims& dims)
        {
            return (dims.n * dims.m * dims.p == 0);
        }

        inline bool operator==(const Dims& lhs, const Dims& rhs)
        {
            return (lhs.n == rhs.n && lhs.m == rhs.m && lhs.p == rhs.p);
        }

        inline std::size_t product(const Dims& dims)
        {
            return (dims.n * dims.m * dims.p);
        }

        
        struct Step {
            std::size_t right{ 0 };
            std::size_t in{ 0 };
        };

        inline Step to_step(const Dims& dims)
        {
            return { dims.m, dims.n * dims.m };
        }

        inline bool operator==(const Step& lhs, const Step& rhs)
        {
            return (lhs.right == rhs.right && lhs.in == rhs.in);
        }


        struct Inds {
            std::size_t i{ 0 }, j{ 0 }, k{ 0 };
        };

        inline bool is_inside(const Inds& inds, const Dims& dims)
        {
            return (inds.i < dims.n&& inds.j < dims.m&& inds.k < dims.p);
        }

        inline std::size_t to_buff_index(const Inds& inds, const Step& step, std::size_t offset = 0)
        {
            return (offset + inds.k * step.in + inds.i * step.right + inds.j);
        }

        inline bool operator==(const Inds& lhs, const Inds& rhs)
        {
            return (lhs.i == rhs.i && lhs.j == rhs.j && lhs.k == rhs.k);
        }

        
        // Every matrix with size less or equal to 9 will be allocated on stack
        using Matrix_allocator = memoc::Malloc_allocator;

        template <typename T>
        using Matrix_buffer = memoc::Typed_buffer<T, memoc::Fallback_buffer<
            memoc::Stack_buffer<9 * sizeof(T)>,
            memoc::Allocated_buffer<Matrix_allocator, true>>>;

        template <typename T, memoc::Buffer<T> Internal_buffer = Matrix_buffer<T>, memoc::Allocator Internal_allocator = Matrix_allocator>
        class Matrix {
        public:
            struct Header {
                Dims dims{};
                Step step{};
                std::size_t offset{ 0 };
                bool is_submatrix{ false };
            };

            Matrix() = default;

            Matrix(Matrix<T, Internal_buffer, Internal_allocator>&& other)
                : hdr_{ other.hdr_ }, buffsp_{ other.buffsp_ }
            {
                other.hdr_ = Header{};
                other.buffsp_.reset();
            }
            Matrix<T, Internal_buffer, Internal_allocator>& operator=(Matrix<T, Internal_buffer, Internal_allocator>&& other)
            {
                COMPUTOC_THROW_IF_FALSE(!hdr_.is_submatrix, std::runtime_error, "move assignment to submatrix is undefined");

                if (this == &other) {
                    return *this;
                }

                hdr_ = other.hdr_;
                buffsp_ = other.buffsp_;

                other.hdr_ = Header{};
                other.buffsp_.reset();

                return *this;
            }

            Matrix(const Matrix<T, Internal_buffer, Internal_allocator>& other) = default;
            Matrix<T, Internal_buffer, Internal_allocator>& operator=(const Matrix<T, Internal_buffer, Internal_allocator>& other)
            {
                COMPUTOC_THROW_IF_FALSE(!hdr_.is_submatrix, std::runtime_error, "copy assignemnt to submatrix is undefined");

                if (this == &other) {
                    return *this;
                }

                hdr_ = other.hdr_;
                buffsp_ = other.buffsp_;

                return *this;
            }

            virtual ~Matrix() = default;

            Matrix(const Dims& dims, const T* data = nullptr)
                : hdr_{ dims, to_step(dims) }, buffsp_(memoc::make_shared<Internal_buffer, Internal_allocator>(product(dims), data))
            {
                COMPUTOC_THROW_IF_FALSE(!is_empty(hdr_.dims), std::invalid_argument, "zero matrix dimensions");
                COMPUTOC_THROW_IF_FALSE(buffsp_ && buffsp_->usable(), std::runtime_error, "internal buffer failed");
            }

            Matrix(const Dims& dims, const T& value)
                : hdr_{ dims, to_step(dims) }, buffsp_(memoc::make_shared<Internal_buffer, Internal_allocator>(product(dims)))
            {
                COMPUTOC_THROW_IF_FALSE(!is_empty(hdr_.dims), std::invalid_argument, "zero matrix dimensions");
                COMPUTOC_THROW_IF_FALSE(buffsp_&& buffsp_->usable(), std::runtime_error, "internal buffer failed");

                for (std::size_t i = 0; i < buffsp_->data().s; ++i) {
                    buffsp_->data().p[i] = value;
                }
            }

            const Header& header() const
            {
                return hdr_;
            }

            T* data() const
            {
                return (buffsp_ ? buffsp_->data().p : nullptr);
            }

            const T& operator()(const Inds& inds) const
            {
                COMPUTOC_THROW_IF_FALSE(is_inside(inds, hdr_.dims), std::out_of_range, "out of range indices");
                return buffsp_->data().p[to_buff_index(inds, hdr_.step, hdr_.offset)];
            }

            T& operator()(const Inds& inds)
            {
                COMPUTOC_THROW_IF_FALSE(is_inside(inds, hdr_.dims), std::out_of_range, "out of range indices");
                return buffsp_->data().p[to_buff_index(inds, hdr_.step, hdr_.offset)];
            }

            template <typename T_o, memoc::Buffer<T_o> Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend bool operator==(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& lhs, const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& rhs);

            Matrix<T, Internal_buffer, Internal_allocator> operator()(const Inds& inds, const Dims& dims) const
            {
                COMPUTOC_THROW_IF_FALSE(!is_empty(dims), std::invalid_argument, "zero matrix dimensions");

                Inds max_inds{ inds.i + dims.n - 1, inds.j + dims.m - 1, inds.k + dims.p - 1 };
                COMPUTOC_THROW_IF_FALSE(is_inside(max_inds, hdr_.dims), std::out_of_range, "out of range submatrix");

                Matrix<T, Internal_buffer, Internal_allocator> slice{};
                slice.hdr_ = { dims, hdr_.step, to_buff_index(inds, hdr_.step, hdr_.offset), true };
                slice.buffsp_ = buffsp_;

                return slice;
            }

            template <typename T_o, memoc::Buffer<T_o> Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend Matrix<T_o, Internal_buffer_o, Internal_allocator_o> copy_to(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& src, Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& dst);
            template <typename T_o, memoc::Buffer<T_o> Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend Matrix<T_o, Internal_buffer_o, Internal_allocator_o> copy_to(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& src, Matrix<T_o, Internal_buffer_o, Internal_allocator_o>&& dst);

            template <typename T_o, memoc::Buffer<T_o> Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend Matrix<T_o, Internal_buffer_o, Internal_allocator_o> copy_of(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& mat);

            template <typename T_o, memoc::Buffer<T_o> Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend Matrix<T_o, Internal_buffer_o, Internal_allocator_o> reshaped(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& mat, const Dims& new_dims);

            template <typename T_o, memoc::Buffer<T_o> Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend Matrix<T_o, Internal_buffer_o, Internal_allocator_o> resized(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& mat, const Dims& new_dims);

        private:
            Header hdr_{};
            memoc::Shared_ptr<Internal_buffer, Internal_allocator> buffsp_{ nullptr };
        };

        template <typename T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline bool operator==(const Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            if (lhs.hdr_.dims != rhs.hdr_.dims) {
                return false;
            }

            for (std::size_t k = 0; k < lhs.hdr_.dims.p; ++k) {
                for (std::size_t i = 0; i < lhs.hdr_.dims.n; ++i) {
                    for (std::size_t j = 0; j < lhs.hdr_.dims.m; ++j) {
                        if (!is_equal(lhs({ i, j, k }), rhs({ i, j, k }))) {
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        template <typename T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> copy_to(const Matrix<T, Internal_buffer, Internal_allocator>& src, Matrix<T, Internal_buffer, Internal_allocator>& dst)
        {
            if (src.hdr_.dims != dst.hdr_.dims) {
                COMPUTOC_THROW_IF_FALSE(!dst.hdr_.is_submatrix, std::runtime_error, "unable to reallocate submatrix");
                dst.hdr_ = { src.hdr_.dims, src.hdr_.step, 0, false };
                dst.buffsp_ = memoc::make_shared<Internal_buffer, Internal_allocator>(product(src.hdr_.dims));
            }

            for (std::size_t k = 0; k < src.hdr_.dims.p; ++k) {
                for (std::size_t i = 0; i < src.hdr_.dims.n; ++i) {
                    for (std::size_t j = 0; j < src.hdr_.dims.m; ++j) {
                        dst({ i, j, k }) = src({ i, j, k });
                    }
                }
            }

            return dst;
        }
        template <typename T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> copy_to(const Matrix<T, Internal_buffer, Internal_allocator>& src, Matrix<T, Internal_buffer, Internal_allocator>&& dst)
        {
            return copy_to(src, dst);
        }

        template <typename T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> copy_of(const Matrix<T, Internal_buffer, Internal_allocator>& mat)
        {
            Matrix<T, Internal_buffer, Internal_allocator> clone{};
            clone.hdr_ = { mat.hdr_.dims, mat.hdr_.step, 0, false };
            if (mat.buffsp_) {
                clone.buffsp_ = memoc::make_shared<Internal_buffer, Internal_allocator>(product(mat.hdr_.dims));
                for (std::size_t k = 0; k < mat.hdr_.dims.p; ++k) {
                    for (std::size_t i = 0; i < mat.hdr_.dims.n; ++i) {
                        for (std::size_t j = 0; j < mat.hdr_.dims.m; ++j) {
                            clone({ i, j, k }) = mat({ i, j, k });
                        }
                    }
                }
            }
            return clone;
        }

        template <typename T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> reshaped(const Matrix<T, Internal_buffer, Internal_allocator>& mat, const Dims& new_dims)
        {
            COMPUTOC_THROW_IF_FALSE(!mat.hdr_.is_submatrix, std::runtime_error, "reshaping submatrix is undefined");
            COMPUTOC_THROW_IF_FALSE(mat.buffsp_, std::runtime_error, "matrix should not be empty");
            COMPUTOC_THROW_IF_FALSE(product(new_dims) == product(mat.hdr_.dims), std::invalid_argument, "reshaped matrix should have the same amount of cells as the original");

            Matrix<T, Internal_buffer, Internal_allocator> rmat{ mat };

            rmat.hdr_.dims = new_dims;
            rmat.hdr_.step = to_step(new_dims);

            return rmat;
        }

        template <typename T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> resized(const Matrix<T, Internal_buffer, Internal_allocator>& mat, const Dims& new_dims)
        {
            COMPUTOC_THROW_IF_FALSE(!mat.hdr_.is_submatrix, std::runtime_error, "resize for sub matrix is undefined");

            if (mat.hdr_.dims == new_dims) {
                return mat;
            }

            if (!mat.buffsp_) {
                return mat;
            }

            if (product(new_dims) == product(mat.hdr_.dims)) {
                return reshaped(mat, new_dims);
            }

            if (product(new_dims) < product(mat.hdr_.dims)) {
                return Matrix<T, Internal_buffer, Internal_allocator>{ new_dims, mat.buffsp_->data().p };
            }

            Matrix<T, Internal_buffer, Internal_allocator> rmat{ new_dims };
            for (std::size_t i = 0; i < mat.buffsp_->data().s; ++i) {
                rmat.buffsp_->data().p[i] = mat.buffsp_->data().p[i];
            }
            return rmat;
        }

        template <typename T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline bool is_empty(const Matrix<T, Internal_buffer, Internal_allocator>& mat)
        {
            return (!mat.data() || is_empty(mat.header().dims));
        }

        /*
        template <Arithmetic T, memoc::Buffer<T> Internal_buffer = Matrix_buffer<T>>
        class Matrix {
        public:
            template <Arithmetic T_o, memoc::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> operator*(const Matrix<T_o, Internal_buffer_o>& lhs, const T_o& rhs);

            template <Arithmetic T_o, memoc::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> operator*(const T_o& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            Matrix<T, Internal_buffer>& operator*=(const T& other)
            {
                for (std::size_t i = 0; i < buff_.data().s; ++i) {
                    buff_.data().p[i] *= other;
                }
                return *this;
            }

            template <Arithmetic T_o, memoc::Buffer<T_o> Internal_buffer_o>
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

            template <Arithmetic T_o, memoc::Buffer<T_o> Internal_buffer_o>
            friend T_o determinant(const Matrix<T_o, Internal_buffer_o>& mat);

            template <Arithmetic T_o, memoc::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> inverse(const Matrix<T_o, Internal_buffer_o>& mat);

            template <Arithmetic T_o, memoc::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> merge_horizontal(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            template <Arithmetic T_o, memoc::Buffer<T_o> Internal_buffer_o>
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

            //template <Arithmetic T_o, memoc::Buffer<T_o> Internal_buffer_o>
            //friend Matrix<T_o, Internal_buffer_o> row_echelon_form(const Matrix<T_o, Internal_buffer_o>& mat);

            template <Arithmetic T_o, memoc::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> reduced_row_echelon_form(const Matrix<T_o, Internal_buffer_o>& mat);

        private:
            Dimensions dims_{};
            Internal_buffer buff_{};
        };

        template <Arithmetic T, memoc::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> operator*(const Matrix<T, Internal_buffer>& lhs, const T& rhs)
        {
            Matrix<T, Internal_buffer> multiplication{ lhs };

            for (std::size_t i = 0; i < multiplication.buff_.data().s; ++i) {
                multiplication.buff_.data().p[i] *= rhs;
            }
            return multiplication;
        }

        template <Arithmetic T, memoc::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> operator*(const T& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            return operator*(rhs, lhs);
        }

        template <Arithmetic T, memoc::Buffer<T> Internal_buffer>
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

        template <Arithmetic T, memoc::Buffer<T> Internal_buffer>
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
                if (!is_equal(p, T{ 0 })) {
                    d += sign * p * determinant<T, Internal_buffer>(mat.get_slice(0, j));
                }
                sign *= T{ -1 };
            }
            return d;
        }

        template <Arithmetic T, memoc::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> inverse(const Matrix<T, Internal_buffer>& mat)
        {
            COMPUTOC_THROW_IF_FALSE(mat.dims_.n == mat.dims_.m, std::invalid_argument, "not square matrix");
            std::size_t n = mat.dims_.n;

            T d{ determinant(mat) };
            COMPUTOC_THROW_IF_FALSE(!is_equal(d, T{ 0 }), std::invalid_argument, "zero determinant");

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

        template <Arithmetic T, memoc::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> merge_horizontal(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.dims_.n == rhs.dims_.n, std::invalid_argument, "dimensions mismatch (lhs.dims_.n = %d, rhs.dims_.n = %d)", lhs.dims_.n, rhs.dims_.n);

            Matrix<T, Internal_buffer> merged{ {lhs.dims_.n, lhs.dims_.m + rhs.dims_.m}, T{} };

            merged.set_slice(0, 0, lhs);
            merged.set_slice(0, lhs.dims_.m, rhs);

            return merged;
        }

        template <Arithmetic T, memoc::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> merge_vertical(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            COMPUTOC_THROW_IF_FALSE(lhs.dims_.m == rhs.dims_.m, std::invalid_argument, "dimensions mismatch (lhs.dims_.m = %d, rhs.dims_.m = %d)", lhs.dims_.m, rhs.dims_.m);

            Matrix<T, Internal_buffer> merged{ {lhs.dims_.n + rhs.dims_.n, lhs.dims_.m}, T{} };

            merged.set_slice(0, 0, lhs);
            merged.set_slice(lhs.dims_.n, 0, rhs);

            return merged;
        }

        //template <Arithmetic T, memoc::Buffer<T> Internal_buffer>
        //inline Matrix<T, Internal_buffer> row_echelon_form(const Matrix<T, Internal_buffer>& mat)
        //{
        //    Matrix<T, Internal_buffer> rref_mat{ mat };

        //    std::size_t r = mat.dims_.n > mat.dims_.m ? mat.dims_.m : mat.dims_.n;

        //    for (std::size_t k = 0; k < r - 1; ++k) {
        //        if (is_equal(rref_mat(k, k), T{ 0 })) {
        //            for (std::size_t i = k + 1; i < mat.dims_.n; ++i) {
        //                if (!is_equal(rref_mat(i, k), T{ 0 })) {
        //                    rref_mat.swap_rows(k, i);
        //                }
        //            }
        //        }
        //        if (!is_equal(rref_mat(k, k), T{ 0 })) {
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

        template <Arithmetic T, memoc::Buffer<T> Internal_buffer>
        inline Matrix<T, Internal_buffer> reduced_row_echelon_form(const Matrix<T, Internal_buffer>& mat)
        {
            Matrix<T, Internal_buffer> rref_mat{ mat };

            std::size_t r = mat.dims_.n > mat.dims_.m ? mat.dims_.m : mat.dims_.n;

            for (std::size_t k = 0; k < r; ++k) {
                if (is_equal(rref_mat(k, k), T{ 0 })) {
                    for (std::size_t i = k + 1; i < mat.dims_.n; ++i) {
                        if (!is_equal(rref_mat(i, k), T{ 0 })) {
                            rref_mat.swap_rows(k, i);
                        }
                    }
                }
                if (!is_equal(rref_mat(k, k), T{ 0 })) {
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

        //template <Arithmetic T_o, memoc::Buffer<T_o> Internal_buffer_o>
        //std::ostream& operator<<(std::ostream& os, const Matrix<T_o, Internal_buffer_o>& mat)
        //{
        //    for (std::size_t i = 0; i < mat.dimensions().n; ++i) {
        //        for (std::size_t j = 0; j < mat.dimensions().m; ++j) {
        //            std::cout << mat(i, j) << " ";
        //        }
        //        std::cout << "\n";
        //    }
        //    return os;
        //}*/
    }



    /*using details::Dimensions;
    using details::Matrix;*/

    using details::Dims;
    using details::is_empty;
    using details::product;

    using details::Step;
    using details::to_step;

    using details::Inds;
    using details::is_inside;
    using details::to_buff_index;

    using details::Matrix;
    using details::copy_of;
    using details::copy_to;
    using details::reshaped;
    using details::resized;
}

/*namespace computoc {
    using details::determinant;
    using details::inverse;
    using details::merge_horizontal;
    using details::merge_vertical;
    using details::reduced_row_echelon_form;
    //using details::row_echelon_form;
}*/

#endif // COMPUTOC_TYPE_MATRIX_H
