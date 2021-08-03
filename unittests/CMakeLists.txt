cmake_minimum_required(VERSION 3.15.0)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

link_libraries(gtest_main)
include_directories(${PROJECT_SOURCE_DIR})

# Add each test file as executable
add_executable(
  state_test
  state_test.cpp
  ../src/state.cpp
  ../src/smt.cpp
  ../src/value.cpp
  ../src/abstractops.cpp
  ../src/memory.cpp
)
add_executable(
  value_test
  value_test.cpp
  ../src/value.cpp
  ../src/smt.cpp
  ../src/abstractops.cpp
  ../src/memory.cpp
)
include(GoogleTest)
# Add each executable to gtest

gtest_discover_tests(state_test)
gtest_discover_tests(value_test)
