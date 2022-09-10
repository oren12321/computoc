#ifndef COMPUTOC_LINEAR_ALGEBRA_H
#define COMPUTOC_LINEAR_ALGEBRA_H

#include <memoc/allocators.h>
#include <memoc/buffers.h>
#include <computoc/errors.h>
#include <computoc/concepts.h>
#include <computoc/algorithms.h>
#include <computoc/matrix.h>

namespace computoc {
    namespace details {
        template <Number T, memoc::Buffer<T> Internal_buffer, memoc::Allocator Internal_allocator>
        inline Matrix<T, Internal_buffer, Internal_allocator> minor(const Matrix<T, Internal_buffer, Internal_allocator>& mat, const Inds& pivot)
        {
            COMPUTOC_THROW_IF_FALSE(!is_empty(mat.header().dims), std::invalid_argument, "no minor for empty matrix - invalid dimensions");
            COMPUTOC_THROW_IF_FALSE(mat.header().dims.p == 1, std::invalid_argument, "matrix should have two dimensions");

            COMPUTOC_THROW_IF_FALSE(is_inside(pivot, mat.header().dims), std::out_of_range, "pivot is not in matrix dimensions");

            Matrix<T, Internal_buffer, Internal_allocator> mmat{ {mat.header().dims.n - 1, mat.header().dims.m - 1} };

            copy_to(mat({ 0, 0 }, { pivot.i, pivot.j }), mmat({ 0, 0 }, { pivot.i, pivot.j })); // UL
            copy_to(mat({ pivot.i + 1, 0 }, { mat.header().dims.n - (pivot.i + 1), pivot.j }), mmat({ pivot.i, 0 }, { mat.header().dims.n - (pivot.i + 1), pivot.j })); // BL
            copy_to(mat({ 0, pivot.j + 1 }, { pivot.i, mat.header().dims.m - (pivot.j + 1) }), mmat({ 0, pivot.j }, { pivot.i, mat.header().dims.m - (pivot.j + 1) })); // UR
            copy_to(mat({ pivot.i + 1 ,pivot.j + 1 }, { mat.header().dims.n - (pivot.i + 1),mat.header().dims.m - (pivot.j + 1) }), mmat({ pivot.i, pivot.j }, { mat.header().dims.n - (pivot.i + 1),mat.header().dims.m - (pivot.j + 1) })); // BR

            return mmat;
        }
    }

    using details::minor;
}

#endif // COMPUTOC_LINEAR_ALGEBRA_H