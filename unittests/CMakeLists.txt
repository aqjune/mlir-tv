cmake_minimum_required(VERSION 3.15.0)

# Try using libc when building googletest if possible
if(USE_LIBC)
    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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
add_dependencies(state_test ${PROJECT_LIB})
target_include_directories(state_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(state_test PRIVATE gtest_main ${PROJECT_LIB})

add_executable(
  value_test
  value_test.cpp
  ../src/value.cpp
  ../src/smt.cpp
  ../src/abstractops.cpp
  ../src/memory.cpp
)
add_dependencies(value_test ${PROJECT_LIB})
target_include_directories(value_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(value_test PRIVATE gtest_main ${PROJECT_LIB})

include(GoogleTest)
# Add each executable to gtest
gtest_discover_tests(state_test)
gtest_discover_tests(value_test)
