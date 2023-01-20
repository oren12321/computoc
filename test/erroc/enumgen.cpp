#include <gtest/gtest.h>

#include <erroc/enumgen.h>

ENUMOC_GENERATE(some_namespace, An_enum,
    first_field,
    another_field);

TEST(Enumoc_generate, generates_enum_and_to_string_method_for_it)
{
    EXPECT_EQ(2, static_cast<int>(some_namespace::An_enum::size));

    EXPECT_STREQ("some_namespace::An_enum:: (first_field)", to_string(some_namespace::An_enum::first_field));
    EXPECT_STREQ("some_namespace::An_enum:: (another_field)", to_string(some_namespace::An_enum::another_field));

    some_namespace::An_enum field{ some_namespace::An_enum::first_field };

    EXPECT_STREQ("some_namespace::An_enum:: (first_field)", to_string(field));
}
