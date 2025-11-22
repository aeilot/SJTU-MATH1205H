cmake_minimum_required(VERSION 4.0)
project(Matrix)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

enable_testing()

FetchContent_MakeAvailable(googletest)

add_executable(Matrix test.cpp
        lib/matrix.h)

target_link_libraries(
        Matrix
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(Matrix)
