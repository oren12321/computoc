# enumoc

A one header library which provide an automatic generation of `enum` with a `to_string` function for it.

Usage:

```cpp
#include <enumoc/enumoc.h>

ENUMOC_GENERATE(<namespace_name>::<namespace_name>::..., <enum_name>,
    <field1_name>, <field2_name>, ...);
```

For example the following macro:

```cpp
ENUMOC_GENERATE(math_utils, Error,
    out_of_range,
    divide_by_zero);
```

Is expended to:

```cpp
namespace math_utils {
  namespace details {
    enum class Error {
      out_of_range,
      divide_by_zero
    };
    [[nodiscard]] inline constexpr const char* to_string(Error) noexcept { ... }
  }
  using details::Error;
  using details::to_string;
};
```

e.g the string representation of `divide_by_zero` is `"math_utils::Error::divide_by_zero"`. 








