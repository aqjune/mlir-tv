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

link_directories(${IREE_BUILD_DIR}/third_party/llvm-project/llvm/lib)
link_directories(${Z3_LIBRARY_DIR})
link_libraries(
    MLIRStandard MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces MLIRViewLikeInterface MLIRInferTypeOpInterface
    MLIRIR MLIRDialect MLIRDialectUtils MLIRLinalg MLIRAffine MLIRMemRef
    MLIRMemRefUtils MLIRTensor MLIRParser MLIRSupport MLIRInferTypeOpInterface
    LLVMSupport LLVMDemangle z3 pthread m curses)

# Add each test file as executable
add_executable(
  smt_test
  smt_test.cpp
)
add_executable(
  state_test
  state_test.cpp
)
add_executable(
  tensor_test
  tensor_test.cpp
)
add_executable(
  vcgen_test
  vcgen_test.cpp
)

include(GoogleTest)
# Add each executable to gtest
gtest_discover_tests(smt_test)
gtest_discover_tests(state_test)
gtest_discover_tests(tensor_test)
gtest_discover_tests(vcgen_test)
