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

    EXPECT_STREQ("division_by_zero", enum_to_string(Errors::division_by_zero));
    EXPECT_STREQ("invalid_argument", enum_to_string(Errors::invalid_argument));
    EXPECT_STREQ("file_not_found", enum_to_string(Errors::file_not_found));
    EXPECT_STREQ("runtime_error", enum_to_string(Errors::runtime_error));

    Errors error{ Errors::division_by_zero };

    EXPECT_STREQ("division_by_zero", enum_to_string(error));
}
