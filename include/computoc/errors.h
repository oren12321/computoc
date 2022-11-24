#ifndef COMPUTOC_ERRORS_H
#define COMPUTOC_ERRORS_H

#include <memoc/errors.h>

#define COMPUTOC_THROW_IF_FALSE(condition,exception_type,...) MEMOC_THROW_IF_FALSE(condition,exception_type,__VA_ARGS__)
#define COMPUTOCPP_THROW_IF_FALSE(condition,exception_type,...) MEMOCPP_THROW_IF_FALSE(condition,exception_type,__VA_ARGS__)

#endif // COMPUTOC_ERRORS_H
