cmake_minimum_required(VERSION 3.20)
project(PentaCoreTests)

include(GoogleTest)

# Test sources
set(TEST_SOURCES
    harmony_test.cpp
    groove_test.cpp
    osc_test.cpp
    rt_memory_test.cpp
)

add_executable(penta_tests ${TEST_SOURCES})

target_link_libraries(penta_tests PRIVATE
    penta_core
    gtest
    gtest_main
)

# Discover tests for CTest
gtest_discover_tests(penta_tests)
