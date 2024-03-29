#include <gtest/gtest.h>

#include <stdexcept>
#include <regex>
#include <sstream>
#include <string>

#include <erroc/errors.h>

class Erroc_expect : public testing::Test {
protected:
    using Selected_exception = std::runtime_error;

    const bool true_condition_{ true };
    const bool false_condition_{ false };

    const std::regex error_message_pattern_regex_ = std::regex("^'.+' failed on '.+' at line:[0-9]+@.+@.+ with message: .+$");
};

TEST_F(Erroc_expect, not_throw_exception_if_condition_is_true)
{
    EXPECT_NO_THROW(ERROC_EXPECT(true_condition_, Selected_exception));
}

TEST_F(Erroc_expect, throw_exception_if_condition_is_false)
{
    EXPECT_THROW(ERROC_EXPECT(false_condition_, Selected_exception), Selected_exception);
}

TEST_F(Erroc_expect, throws_an_exception_with_specific_format)
{
    try {
        ERROC_EXPECT(false_condition_, Selected_exception, "some message with optional %d value", 0);
        FAIL();
    }
    catch (const Selected_exception& ex) {
        EXPECT_TRUE(std::regex_match(ex.what(), error_message_pattern_regex_));
    }
}

class Errocpp_expect : public Erroc_expect {
};

TEST_F(Errocpp_expect, not_throw_exception_if_condition_is_true)
{
#ifdef __unix__
    EXPECT_NO_THROW((ERROCPP_EXPECT(true_condition_, Selected_exception)));
#elif defined(_WIN32) || defined(_WIN64)
    EXPECT_NO_THROW(ERROCPP_EXPECT(true_condition_, Selected_exception));
#endif
}

TEST_F(Errocpp_expect, throw_exception_if_condition_is_false)
{
#ifdef __unix__
    EXPECT_THROW((ERROCPP_EXPECT(false_condition_, Selected_exception)), Selected_exception);
#elif defined(_WIN32) || defined(_WIN64)
    EXPECT_THROW(ERROCPP_EXPECT(false_condition_, Selected_exception), Selected_exception);
#endif
}

TEST_F(Errocpp_expect, throws_an_exception_with_specific_format)
{
    try {
        ERROCPP_EXPECT(false_condition_, Selected_exception, << "some message with optional " << 0 << " value");
        FAIL();
    }
    catch (const Selected_exception& ex) {
        EXPECT_TRUE(std::regex_match(ex.what(), error_message_pattern_regex_));
    }
}

TEST(Expected_test, can_have_either_value_or_error_of_any_types)
{
    using namespace erroc;

    enum class Errors {
        division_by_zero
    };

    auto divide = [](int a, int b) -> Expected<int, Errors> {
        if (b == 0) {
            return Unexpected{ Errors::division_by_zero };
        }
        return a / b;
    };

    {
        auto result = divide(2, 1);
        EXPECT_TRUE(result);
        EXPECT_EQ(2, result.value());
        EXPECT_THROW((void)result.error(), std::runtime_error);
        EXPECT_EQ(2, result.value_or(-1));
    }

    {
        auto result = divide(2, 0);
        EXPECT_FALSE(result);
        EXPECT_EQ(Errors::division_by_zero, result.error());
        EXPECT_THROW((void)result.value(), std::runtime_error);
        EXPECT_EQ(-1, result.value_or(-1));
    }
}

TEST(Expected_test, can_be_copied)
{
    using namespace erroc;

    {
        Expected<int, double> result = 1;
        Expected<int, double> copied_result(result);

        EXPECT_TRUE(result);
        EXPECT_EQ(1, result.value());

        EXPECT_TRUE(copied_result);
        EXPECT_EQ(1, copied_result.value());
        
        copied_result = result;

        EXPECT_TRUE(copied_result);
        EXPECT_EQ(1, copied_result.value());
    }

    {
        Expected<int, double> result = Unexpected(1.1);
        Expected<int, double> copied_result(result);

        EXPECT_FALSE(result);
        EXPECT_EQ(1.1, result.error());

        EXPECT_FALSE(copied_result);
        EXPECT_EQ(1.1, copied_result.error());

        copied_result = result;

        EXPECT_FALSE(copied_result);
        EXPECT_EQ(1.1, copied_result.error());
    }
}

TEST(Expected_test, can_be_moved)
{
    using namespace erroc;

    {
        Expected<int, double> result = 1;
        Expected<int, double> moved_result(std::move(result));

        EXPECT_TRUE(moved_result);
        EXPECT_EQ(1, moved_result.value());
        EXPECT_THROW((void)moved_result.error(), std::runtime_error);

        Expected<int, double> other_result = 2;
        moved_result = std::move(other_result);

        EXPECT_TRUE(moved_result);
        EXPECT_EQ(2, moved_result.value());
        EXPECT_THROW((void)moved_result.error(), std::runtime_error);
    }

    {
        Expected<int, double> result = Unexpected(1.1);
        Expected<int, double> moved_result(result);

        EXPECT_FALSE(moved_result);
        EXPECT_EQ(1.1, moved_result.error());
        EXPECT_THROW((void)moved_result.value(), std::runtime_error);

        Expected<int, double> other_result = Unexpected(2.2);
        moved_result = other_result;

        EXPECT_FALSE(moved_result);
        EXPECT_EQ(2.2, moved_result.error());
        EXPECT_THROW((void)moved_result.value(), std::runtime_error);
    }
}

TEST(Expected_test, have_monadic_oprations)
{
    using namespace erroc;

    enum class Errors {
        division_by_zero,
        division_failed
    };

    enum class Other_errors {
        division_by_zero,
        division_failed
    };

    auto divide = [](double a, double b) -> Expected<double, Errors> {
        if (b == 0) {
            return Unexpected{ Errors::division_by_zero };
        }
        return a / b;
    };

    auto increment = [](double a) { return a + 1; };
    auto to_int = [](double a) { return static_cast<int>(a); };

    auto same = [](Errors) { return Errors::division_failed; };
    auto seem = [](Errors a) { return static_cast<Other_errors>(a); };

    {
        auto result = divide(2, 4)
            .and_then(increment)
            .and_then(to_int);

        EXPECT_TRUE(result);
        EXPECT_EQ(1, result.value());
    }

    {
        auto result = divide(2, 0)
            .or_else(same)
            .or_else(seem);

        EXPECT_FALSE(result);
        EXPECT_EQ(Other_errors::division_failed, result.error());
    }

    std::stringstream ss{};
    auto report_value = [&ss](double a) { ss << "result = " << a; };
    auto report_failure = [&ss](Errors) { ss << "operation failed: "; };
    auto report_reason = [&ss](Errors) { ss << "division by zero"; };

    {
        ss.str("");

        auto result = divide(2, 4)
            .and_then(increment)
            .and_then(report_value)
            .or_else(report_failure)
            .or_else(report_reason);

        EXPECT_TRUE(result);
        EXPECT_EQ(1.5, result.value());

        EXPECT_EQ("result = 1.5", ss.str());
    }

    {
        ss.str("");

        auto result = divide(2, 0)
            .and_then(increment)
            .and_then(report_value)
            .or_else(report_failure)
            .or_else(report_reason);

        EXPECT_FALSE(result);
        EXPECT_EQ(Errors::division_by_zero, result.error());

        EXPECT_EQ("operation failed: division by zero", ss.str());
    }
}

TEST(Expected_test, can_have_either_value_or_not)
{
    using namespace erroc;

    auto whole_divide = [](int a, int b) -> Expected<int> {
        if (a % b != 0) {
            return Unexpected();
        }
        return a / b;
    };

    {
        auto result = whole_divide(4, 2);
        EXPECT_TRUE(result);
        EXPECT_EQ(2, result.value());
        EXPECT_EQ(2, result.value_or(-1));
    }

    {
        auto result = whole_divide(3, 2);
        EXPECT_FALSE(result);
        EXPECT_THROW((void)result.value(), std::runtime_error);
        EXPECT_EQ(-1, result.value_or(-1));
    }
}

TEST(Expected_test, can_be_compared)
{
    using namespace erroc;
    Expected<int, double> opt1 = 110;
    Expected<char, float> opt2 = 'n';

    Expected<char, double> nopt1 = Unexpected(1.);
    Expected<int, float> nopt2 = Unexpected(1.f);

    {
        EXPECT_TRUE(opt1 == opt2);
        EXPECT_TRUE(nopt1 == nopt2);
        EXPECT_FALSE(opt1 == nopt1);
        EXPECT_FALSE(opt2 == nopt2);
    }

    {
        EXPECT_FALSE(opt1 != opt2);
        EXPECT_FALSE(nopt1 != nopt2);
        EXPECT_TRUE(opt1 != nopt1);
        EXPECT_TRUE(opt2 != nopt2);
    }

    {
        EXPECT_FALSE(opt1 < opt2);
        EXPECT_FALSE(nopt1 < nopt2);
        EXPECT_FALSE(opt1 < nopt1);
        EXPECT_FALSE(opt2 < nopt2);
    }

    {
        EXPECT_TRUE(opt1 <= opt2);
        EXPECT_TRUE(nopt1 <= nopt2);
        EXPECT_FALSE(opt1 <= nopt1);
        EXPECT_FALSE(opt2 <= nopt2);
    }

    {
        EXPECT_FALSE(opt1 > opt2);
        EXPECT_FALSE(nopt1 > nopt2);
        EXPECT_FALSE(opt1 > nopt1);
        EXPECT_FALSE(opt2 > nopt2);
    }

    {
        EXPECT_TRUE(opt1 >= opt2);
        EXPECT_TRUE(nopt1 >= nopt2);
        EXPECT_FALSE(opt1 >= nopt1);
        EXPECT_FALSE(opt2 >= nopt2);
    }
}

TEST(Expected_test, destroys_active_union_value)
{
    using namespace erroc;

    class Dummy {
    public:
        Dummy(bool& destroyed)
            : destoryed_(destroyed)
        {
        }
        ~Dummy()
        { 
            destoryed_ = true;
        }
        bool& destoryed_;
    };

    bool destroyed{ false };

    {
        {
            Expected<Dummy> opt{ Dummy(destroyed) };
            destroyed = false;
        }
        EXPECT_TRUE(destroyed);
    }

    {
        {
            Expected<int, Dummy> opt{ Unexpected(Dummy(destroyed)) };
            destroyed = false;
        }
        EXPECT_TRUE(destroyed);
    }
}

namespace ns1 {
    namespace details {
        enum class Enum {
            field,
        };
        const char* Enum_strings[] {
            "field",
        };
        [[nodiscard]] const char* to_string(Enum code) noexcept {
            return Enum_strings[static_cast<int>(code)];
        }
    }
    using details::Enum;
    using details::to_string;
}

TEST(Expected_test, if_error_have_to_string_implementation_it_is_printed_when_there_is_no_value)
{
    using namespace erroc;

    Expected<int, ns1::Enum> result = Unexpected(ns1::Enum::field);

    try {
        (void)result.value();
        FAIL();
    }
    catch (const std::runtime_error& ex) {
        std::string str = ex.what();
        EXPECT_NE(std::string::npos, str.find("'field'"));
    }
}
