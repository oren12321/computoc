#include <cstddef>
#include <utility>
#include <ostream>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <memory>
#include <cstring>
#include <concepts>

namespace la
{

    template <typename T>
    concept arithmetic = std::integral<T> or std::floating_point<T>;

    template <arithmetic T, std::size_t N, std::size_t M, class Allocator = std::allocator<T>>
    class matrix
    {
    public:
        matrix()
        {
            _data = _allocator.allocate(N * M);
        }

        matrix(T value)
        {
            _data = _allocator.allocate(N * M);
            std::fill_n(_data, N * M, value);
        }

        matrix(const matrix<T, N, M, Allocator>& other)
        {
            _allocator = other._allocator;
            _data = _allocator.allocate(N * M);
            std::memcpy(_data, other._data, N * M * sizeof(T));
        }

        matrix<T, N, M, Allocator>& operator=(const matrix<T, N, M, Allocator>& other)
        {
            std::cout << "copy assignment\n";
            if (this == &other)
            {
                return *this;
            }
            _allocator = other._allocator;
            if (!_data)
            {
                _data = _allocator.allocate(N * M);
            }
            std::memcpy(_data, other._data, N * M * sizeof(T));
            return *this;
        }

        matrix(matrix<T, N, M, Allocator>&& other)
        {
            _allocator = std::move(other._allocator);
            _data = other._data;
            other._data = nullptr;
        }

        matrix<T, N, M, Allocator>& operator=(matrix<T, N, M, Allocator>&& other)
        {
            if (this == &other)
            {
                return *this;
            }
            if (_data)
            {
                _allocator.deallocate(_data, N * M);
            }
            _allocator = std::move(other._allocator);
            _data = other._data;
            other._data = nullptr;
            return *this;
        }

        matrix(std::initializer_list<T> data)
        {
            if (data.size() != N * M)
            {
                std::stringstream err;
                err << "received data size(" << data.size() << ")  must be equal to matrix size(" << N * M << ")";
                throw std::invalid_argument(err.str());
            }
            _data = _allocator.allocate(N * M);
            std::memcpy(_data, data.begin(), N * M * sizeof(T));
        }

        virtual ~matrix()
        {
            if (_data)
            {
                _allocator.deallocate(_data, N * M);
            }
        }

        T& operator()(std::size_t r, std::size_t c)
        {
            if (r < 0 || r >= N)
            {
                std::stringstream err;
                err << "'r == " << r << "' is out of range - must be in [0, " << N << ")";
                throw std::out_of_range(err.str());
            }
            if (c < 0 || c >= M)
            {
                std::stringstream err;
                err << "'c == " << c << "' is out of range - must be in [0, " << M << ")";
                throw std::out_of_range(err.str());
            }
            return _data[r * M + c];
        }

        const T& operator()(std::size_t r, std::size_t c) const
        {
            if (r < 0 || r >= N)
            {
                std::stringstream err;
                err << "'r == " << r << "' is out of range - must be in [0, " << N << ")";
                throw std::out_of_range(err.str());
            }
            if (c < 0 || c >= M)
            {
                std::stringstream err;
                err << "'c == " << c << "' is out of range - must be in [0, " << M << ")";
                throw std::out_of_range(err.str());
            }
            return _data[r * M + c];
        }

        friend std::ostream& operator<<(std::ostream& os, const la::matrix<T, N, M, Allocator>& m);
    private:
        T* _data;
        Allocator _allocator;
    };

}

template <typename T, std::size_t N, std::size_t M, class Allocator>
std::ostream& operator<<(std::ostream& os, const la::matrix<T, N, M, Allocator>& m)
{
    for (std::size_t r = 0; r < N; ++r)
    {
        for (std::size_t c = 0; c < M; ++c)
        {
            os << std::fixed << std::setprecision(4) << m._data[r * M + c];
            if (c < M - 1)
            {
                os << " ";
            }
        }
        if (r < N - 1)
        {
            os << "\n";
        }
    }
    return os;
}

