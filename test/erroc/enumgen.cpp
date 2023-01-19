#include <gtest/gtest.h>

#include <erroc/enumgen.h>

GENERATE_ENUM(Errors,
    division_by_zero,
    invalid_argument,
    file_not_found,
    runtime_error);

TEST(Enumgen_test, generate_enum_with_matching_strings)
{
    EXPECT_EQ(4, static_cast<int>(Errors::size));

    EXPECT_STREQ("Errors:: (division_by_zero)", to_string(Errors::division_by_zero));
    EXPECT_STREQ("Errors:: (invalid_argument)", to_string(Errors::invalid_argument));
    EXPECT_STREQ("Errors:: (file_not_found)", to_string(Errors::file_not_found));
    EXPECT_STREQ("Errors:: (runtime_error)", to_string(Errors::runtime_error));

    Errors error{ Errors::division_by_zero };

    EXPECT_STREQ("Errors:: (division_by_zero)", to_string(error));
}
