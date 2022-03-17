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

template <typename T>
concept arithmetic = std::integral<T> or std::floating_point<T>;

namespace math::core
{

    template <arithmetic T, class Allocator = std::allocator<T>>
    class matrix
    {
    public:
        matrix(std::size_t n, std::size_t m, const T* data) : N(n), M(m)
        {
            if (!data)
            {
                throw std::invalid_argument("received data is null");
            }
            _data = _allocator.allocate(N * M);
            std::memcpy(_data, data, N * M * sizeof(T));
        }

        matrix(std::size_t n, std::size_t m, T value) : N(n), M(m)
        {
            _data = _allocator.allocate(N * M);
            std::fill_n(_data, N * M, value);
        }

        matrix(std::size_t n, std::size_t m, std::initializer_list<T> data) : N(n), M(m)
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

        matrix(const matrix& other) : N(other.N), M(other.M)
        {
            _allocator = other._allocator;
            _data = _allocator.allocate(N * M);
            std::memcpy(_data, other._data, N * M * sizeof(T));
        }

        matrix& operator=(const matrix& other)
        {
            if (this == &other)
            {
                return *this;
            }
            N = other.N;
            M = other.M;
            _allocator = other._allocator;
            if (!_data)
            {
                _data = _allocator.allocate(N * M);
            }
            std::memcpy(_data, other._data, N * M * sizeof(T));
            return *this;
        }

        matrix(matrix&& other) : N(other.N), M(other.M)
        {
            _allocator = std::move(other._allocator);
            _data = other._data;
            other._data = nullptr;
        }

        matrix& operator=(matrix&& other)
        {
            if (this == &other)
            {
                return *this;
            }
            N = other.N;
            M = other.M;
            if (_data)
            {
                _allocator.deallocate(_data, N * M);
            }
            _allocator = std::move(other._allocator);
            _data = other._data;
            other._data = nullptr;
            return *this;
        }

        virtual ~matrix()
        {
            if (_data)
            {
                _allocator.deallocate(_data, N * M);
            }
        }

        const T& operator()(std::size_t i, std::size_t j) const
        {
            if (i < 0 || i >= N)
            {
                std::stringstream err;
                err << "'r == " << i << "' is out of range - must be in [0, " << N << ")";
                throw std::out_of_range(err.str());
            }
            if (j < 0 || j >= M)
            {
                std::stringstream err;
                err << "'c == " << j << "' is out of range - must be in [0, " << M << ")";
                throw std::out_of_range(err.str());
            }
            return _data[i * M + j];
        }

        matrix operator()(std::size_t si, std::size_t ei, std::size_t sj, std::size_t ej)
        {
            std::size_t nN = ei - si + 1;
            std::size_t nM = ej - sj + 1;

            matrix m(nN, nM, static_cast<T>(0));

            for (std::size_t i = 0; i < nN; ++i)
            {
                std::memcpy(m._data + i * nM, _data + (si + i) * M + sj, nM * sizeof(T));
            }

            return m;
        }

    public:
        const std::size_t N, M;

        template <typename T_, class Allocator_>
        friend std::ostream& operator<<(std::ostream& os, const matrix<T_, Allocator_>& m);

    private:
        T* _data;
        Allocator _allocator;
    };

}

template <typename T, class Allocator>
std::ostream& operator<<(std::ostream& os, const math::core::matrix<T, Allocator>& m)
{
    for (std::size_t r = 0; r < m.N; ++r)
    {
        for (std::size_t c = 0; c < m.M; ++c)
        {
            os << std::fixed << std::setprecision(4) << m._data[r * m.M + c];
            if (c < m.M - 1)
            {
                os << " ";
            }
        }
        if (r < m.N - 1)
        {
            os << "\n";
        }
    }
    return os;
}

