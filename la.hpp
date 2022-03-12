#include <cstddef>
#include <utility>
#include <ostream>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <memory>
#include <cstring>

namespace la
{

template <typename T, std::size_t N, std::size_t M, class Allocator = std::allocator<T>,
          typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class matrix
{
public:
    matrix()
    {
        _data = _alloc.allocate(N * M);
    }

    matrix(std::initializer_list<T> data)
    {
        if (data.size() != N * M)
        {
            std::stringstream err;
            err << "received data size(" << data.size() << ")  must be equal to matrix size(" << N * M << ")";
            throw std::invalid_argument(err.str());
        }
        _data = _alloc.allocate(N * M);
        std::memcpy(_data, data.begin(), N * M * sizeof(T));
    }

    virtual ~matrix()
    {
        if (_data) _alloc.deallocate(_data, N * M);
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

private:
    T* _data;
    Allocator _alloc;
};

}

template <typename T, std::size_t N, std::size_t M, class Allocator>
std::ostream& operator<<(std::ostream& os, const la::matrix<T, N, M, Allocator>& m)
{
    for (std::size_t r = 0; r < N; ++r)
    {
        for (std::size_t c = 0; c < M; ++c)
        {
            os << std::fixed << std::setprecision(4) << m(r, c);
            if (c < M - 1) os << " ";
        }
        if (r < N - 1) os << "\n";
    }
    return os;
}

