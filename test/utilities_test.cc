#include <gtest/gtest.h>
#include "utilities.h"

TEST(UtilitiesTests, RemoveFileExtensionTest) {
    EXPECT_EQ(remove_file_extension("foo.inp"), "foo");
    EXPECT_EQ(remove_file_extension("baz.foo.inp"), "baz.foo");
    EXPECT_EQ(remove_file_extension("../examples/foo.inp"), "foo");
}

