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
#include <erroc/errors.h>
#include <computoc/math.h>

#include <memoc/pointers.h>

namespace computoc {
    namespace details {
        struct Dims {
            std::size_t n{ 0 }, m{ 0 }, p{ 1 };
        };

        inline bool empty(const Dims& dims)
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
            memoc::Allocated_buffer<Matrix_allocator>>>;

        template <typename T, memoc::Buffer Internal_buffer = Matrix_buffer<T>, memoc::Allocator Internal_allocator = Matrix_allocator>
            requires std::is_same_v<T, typename decltype(memoc::block(Internal_buffer()))::Type>
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
                ERROC_EXPECT(!hdr_.is_submatrix, std::runtime_error, "move assignment to submatrix is undefined");

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
                ERROC_EXPECT(!hdr_.is_submatrix, std::runtime_error, "copy assignemnt to submatrix is undefined");

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
                ERROC_EXPECT(!empty(hdr_.dims), std::invalid_argument, "zero matrix dimensions");
                ERROC_EXPECT(buffsp_ && !memoc::empty(*buffsp_), std::runtime_error, "internal buffer failed");
            }

            Matrix(const Dims& dims, const T& value)
                : hdr_{ dims, to_step(dims) }, buffsp_(memoc::make_shared<Internal_buffer, Internal_allocator>(product(dims)))
            {
                ERROC_EXPECT(!empty(hdr_.dims), std::invalid_argument, "zero matrix dimensions");
                ERROC_EXPECT(buffsp_&& !memoc::empty(*buffsp_), std::runtime_error, "internal buffer failed");

                for (std::size_t i = 0; i < memoc::size(*buffsp_); ++i) {
                    memoc::data(*buffsp_)[i] = value;
                }
            }

            const Header& header() const
            {
                return hdr_;
            }

            T* data() const
            {
                return (buffsp_ ? memoc::data(*buffsp_) : nullptr);
            }

            const T& operator()(const Inds& inds) const
            {
                ERROC_EXPECT(is_inside(inds, hdr_.dims), std::out_of_range, "out of range indices");
                return memoc::data(*buffsp_)[to_buff_index(inds, hdr_.step, hdr_.offset)];
            }

            T& operator()(const Inds& inds)
            {
                ERROC_EXPECT(is_inside(inds, hdr_.dims), std::out_of_range, "out of range indices");
                return memoc::data(*buffsp_)[to_buff_index(inds, hdr_.step, hdr_.offset)];
            }

            template <typename T_o, memoc::Buffer Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend bool operator==(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& lhs, const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& rhs);

            Matrix<T, Internal_buffer, Internal_allocator> operator()(const Inds& inds, const Dims& dims) const
            {
                ERROC_EXPECT(!empty(dims), std::invalid_argument, "zero matrix dimensions");

                Inds max_inds{ inds.i + dims.n - 1, inds.j + dims.m - 1, inds.k + dims.p - 1 };
                ERROC_EXPECT(is_inside(max_inds, hdr_.dims), std::out_of_range, "out of range submatrix");

                Matrix<T, Internal_buffer, Internal_allocator> slice{};
                slice.hdr_ = { dims, hdr_.step, to_buff_index(inds, hdr_.step, hdr_.offset), true };
                slice.buffsp_ = buffsp_;

                return slice;
            }

            template <typename T_o, memoc::Buffer Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend Matrix<T_o, Internal_buffer_o, Internal_allocator_o> copy(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& src, Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& dst);
            template <typename T_o, memoc::Buffer Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend Matrix<T_o, Internal_buffer_o, Internal_allocator_o> copy(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& src, Matrix<T_o, Internal_buffer_o, Internal_allocator_o>&& dst);

            template <typename T_o, memoc::Buffer Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend Matrix<T_o, Internal_buffer_o, Internal_allocator_o> clone(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& mat);

            template <typename T_o, memoc::Buffer Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend Matrix<T_o, Internal_buffer_o, Internal_allocator_o> reshaped(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& mat, const Dims& new_dims);

            template <typename T_o, memoc::Buffer Internal_buffer_o, memoc::Allocator Internal_allocator_o>
            friend Matrix<T_o, Internal_buffer_o, Internal_allocator_o> resized(const Matrix<T_o, Internal_buffer_o, Internal_allocator_o>& mat, const Dims& new_dims);

        private:
            Header hdr_{};
            memoc::Shared_ptr<Internal_buffer, Internal_allocator> buffsp_{ nullptr };
        };

        template <typename T, memoc::Buffer Internal_buffer, memoc::Allocator Internal_allocator>
        inline bool operator==(const Matrix<T, Internal_buffer, Internal_allocator>& lhs, const Matrix<T, Internal_buffer, Internal_allocator>& rhs)
        {
            if (lhs.hdr_.dims != rhs.hdr_.dims) {
                return false;
            }

            for (std::size_t k = 0; k < lhs.hdr_.dims.p; ++k) {
                for (std::size_t i = 0; i < lhs.hdr_.dims.n; ++i) {
                    for (std::size_t j = 0; j < lhs.hdr_.dims.m; ++j) {
                        if (lhs({ i, j, k }) != rhs({ i, j, k })) {
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        template <typename T, memoc::Buffer Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> copy(const Matrix<T, Internal_buffer, Internal_allocator>& src, Matrix<T, Internal_buffer, Internal_allocator>& dst)
        {
            if (src.hdr_.dims != dst.hdr_.dims) {
                ERROC_EXPECT(!dst.hdr_.is_submatrix, std::runtime_error, "unable to reallocate submatrix");
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
        template <typename T, memoc::Buffer Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> copy(const Matrix<T, Internal_buffer, Internal_allocator>& src, Matrix<T, Internal_buffer, Internal_allocator>&& dst)
        {
            return copy(src, dst);
        }

        template <typename T, memoc::Buffer Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> clone(const Matrix<T, Internal_buffer, Internal_allocator>& mat)
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

        template <typename T, memoc::Buffer Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> reshaped(const Matrix<T, Internal_buffer, Internal_allocator>& mat, const Dims& new_dims)
        {
            ERROC_EXPECT(!mat.hdr_.is_submatrix, std::runtime_error, "reshaping submatrix is undefined");
            ERROC_EXPECT(mat.buffsp_, std::runtime_error, "matrix should not be empty");
            ERROC_EXPECT(product(new_dims) == product(mat.hdr_.dims), std::invalid_argument, "reshaped matrix should have the same amount of cells as the original");

            Matrix<T, Internal_buffer, Internal_allocator> rmat{ mat };

            rmat.hdr_.dims = new_dims;
            rmat.hdr_.step = to_step(new_dims);

            return rmat;
        }

        template <typename T, memoc::Buffer Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> resized(const Matrix<T, Internal_buffer, Internal_allocator>& mat, const Dims& new_dims)
        {
            ERROC_EXPECT(!mat.hdr_.is_submatrix, std::runtime_error, "resize for sub matrix is undefined");

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
                return Matrix<T, Internal_buffer, Internal_allocator>{ new_dims, memoc::data(*mat.buffsp_) };
            }

            Matrix<T, Internal_buffer, Internal_allocator> rmat{ new_dims };
            for (std::size_t i = 0; i < memoc::size(*mat.buffsp_); ++i) {
                memoc::data(*rmat.buffsp_)[i] = memoc::data(*mat.buffsp_)[i];
            }
            return rmat;
        }

        template <typename T, memoc::Buffer Internal_buffer, memoc::Allocator Internal_allocator>
        inline bool empty(const Matrix<T, Internal_buffer, Internal_allocator>& mat)
        {
            return (!mat.data() || empty(mat.header().dims));
        }

        /*
        template <Numeric T, memoc::Buffer Internal_buffer = Matrix_buffer<T>>
        class Matrix {
        public:
            template <Numeric T_o, memoc::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> merge_horizontal(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            template <Numeric T_o, memoc::Buffer<T_o> Internal_buffer_o>
            friend Matrix<T_o, Internal_buffer_o> merge_vertical(const Matrix<T_o, Internal_buffer_o>& lhs, const Matrix<T_o, Internal_buffer_o>& rhs);

            //template <Numeric T_o, memoc::Buffer<T_o> Internal_buffer_o>
            //friend Matrix<T_o, Internal_buffer_o> row_echelon_form(const Matrix<T_o, Internal_buffer_o>& mat);

        private:
            Dimensions dims_{};
            Internal_buffer buff_{};
        };

        template <Numeric T, memoc::Buffer Internal_buffer>
        inline Matrix<T, Internal_buffer> merge_horizontal(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            ERROC_EXPECT(lhs.dims_.n == rhs.dims_.n, std::invalid_argument, "dimensions mismatch (lhs.dims_.n = %d, rhs.dims_.n = %d)", lhs.dims_.n, rhs.dims_.n);

            Matrix<T, Internal_buffer> merged{ {lhs.dims_.n, lhs.dims_.m + rhs.dims_.m}, T{} };

            merged.set_slice(0, 0, lhs);
            merged.set_slice(0, lhs.dims_.m, rhs);

            return merged;
        }

        template <Numeric T, memoc::Buffer Internal_buffer>
        inline Matrix<T, Internal_buffer> merge_vertical(const Matrix<T, Internal_buffer>& lhs, const Matrix<T, Internal_buffer>& rhs)
        {
            ERROC_EXPECT(lhs.dims_.m == rhs.dims_.m, std::invalid_argument, "dimensions mismatch (lhs.dims_.m = %d, rhs.dims_.m = %d)", lhs.dims_.m, rhs.dims_.m);

            Matrix<T, Internal_buffer> merged{ {lhs.dims_.n + rhs.dims_.n, lhs.dims_.m}, T{} };

            merged.set_slice(0, 0, lhs);
            merged.set_slice(lhs.dims_.n, 0, rhs);

            return merged;
        }

        //template <Numeric T, memoc::Buffer Internal_buffer>
        //inline Matrix<T, Internal_buffer> row_echelon_form(const Matrix<T, Internal_buffer>& mat)
        //{
        //    Matrix<T, Internal_buffer> rref_mat{ mat };

        //    std::size_t r = mat.dims_.n > mat.dims_.m ? mat.dims_.m : mat.dims_.n;

        //    for (std::size_t k = 0; k < r - 1; ++k) {
        //        if (equal(rref_mat(k, k), T{ 0 })) {
        //            for (std::size_t i = k + 1; i < mat.dims_.n; ++i) {
        //                if (!equal(rref_mat(i, k), T{ 0 })) {
        //                    rref_mat.swap_rows(k, i);
        //                }
        //            }
        //        }
        //        if (!equal(rref_mat(k, k), T{ 0 })) {
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

        //template <Numeric T_o, memoc::Buffer<T_o> Internal_buffer_o>
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
    using details::empty;
    using details::product;

    using details::Step;
    using details::to_step;

    using details::Inds;
    using details::is_inside;
    using details::to_buff_index;

    using details::Matrix;
    using details::clone;
    using details::copy;
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
