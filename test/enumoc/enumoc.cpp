#include <gtest/gtest.h>

#include <enumoc/enumoc.h>

ENUMOC_GENERATE(ns1::ns2, An_enum,
    first_field,
    another_field);

TEST(Enumoc_generate, generates_enum_and_to_string_method_for_it)
{
    EXPECT_EQ(2, static_cast<int>(ns1::ns2::An_enum::size));

    EXPECT_STREQ("ns1::ns2::An_enum:: (first_field)", to_string(ns1::ns2::An_enum::first_field));
    EXPECT_STREQ("ns1::ns2::An_enum:: (another_field)", to_string(ns1::ns2::An_enum::another_field));

    ns1::ns2::An_enum field{ ns1::ns2::An_enum::first_field };

    EXPECT_STREQ("ns1::ns2::An_enum:: (first_field)", to_string(field));
}
