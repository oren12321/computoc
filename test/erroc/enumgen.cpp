#include <gtest/gtest.h>

#include <erroc/enumgen.h>

TEST(Enumgen_test, generate_enum_with_matching_strings)
{
    GENERATE_ENUM(Errors,
        division_by_zero,
        invalid_argument,
        file_not_found,
        runtime_error);

    EXPECT_EQ(4, static_cast<int>(Errors::size));

    EXPECT_STREQ("division_by_zero", ENUM_TO_CSTRING(Errors, division_by_zero));
    EXPECT_STREQ("invalid_argument", ENUM_TO_CSTRING(Errors, invalid_argument));
    EXPECT_STREQ("file_not_found", ENUM_TO_CSTRING(Errors, file_not_found));
    EXPECT_STREQ("runtime_error", ENUM_TO_CSTRING(Errors, runtime_error));
}
