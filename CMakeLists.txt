cmake_minimum_required(VERSION 3.15.0)
project(mlir-tv VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 17)

set(IREE_DIR "" CACHE STRING "IREE source top-level directory")
set(IREE_BUILD_DIR "" CACHE STRING "IREE build top-level directory")
set(Z3_DIR "" CACHE STRING "Z3 installation top-level directory")
set(SMT_SWITCH_DIR "" CACHE STRING "smt-switch installation top-level directory")

set(Z3_INC_DIR "${Z3_DIR}/include")
set(Z3_LIB_DIR "${Z3_DIR}/lib")
set(SMT_SWITCH_INC_DIR "${SMT_SWITCH_DIR}/include")
set(SMT_SWITCH_LIB_DIR "${SMT_SWITCH_DIR}/lib")

# Libraries can be found in IREE_BUILD_DIR/third_party/llvm-project/llvm/lib
link_directories(${IREE_BUILD_DIR}/third_party/llvm-project/llvm/lib)
link_directories(${Z3_LIB_DIR})
link_directories(${SMT_SWITCH_LIB_DIR})
link_libraries(
    MLIRViewLikeInterface MLIRInferTypeOpInterface
    MLIRIR MLIRDialect MLIRDialectUtils MLIRLinalg MLIRViewLikeInterface MLIRAffine MLIRMemRef MLIRShape MLIRMath
    MLIRControlFlowInterfaces MLIRStandard MLIRSideEffectInterfaces MLIRMemRefUtils MLIRTensor MLIRParser MLIRSupport MLIRInferTypeOpInterface
    LLVMSupport LLVMDemangle z3 smt-switch smt-switch-z3 smt-switch-cvc4 pthread m curses)

# IREE include directories
include_directories(${IREE_DIR}/third_party/llvm-project/llvm/include ${IREE_DIR}/third_party/llvm-project/mlir/include)
include_directories(${IREE_BUILD_DIR}/third_party/llvm-project/llvm/include ${IREE_BUILD_DIR}/third_party/llvm-project/llvm/tools/mlir/include)

# Z3 include directories
include_directories(${Z3_INC_DIR})

# smt-switch include 
include_directories(${SMT_SWITCH_INC_DIR})

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/abstractops.cpp
    src/memory.cpp
    src/smt.cpp
    src/state.cpp
    src/value.cpp
    src/vcgen.cpp)

target_link_options(${PROJECT_NAME} PUBLIC -fPIC)
target_compile_options(${PROJECT_NAME} PUBLIC -fno-rtti)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
add_subdirectory(${PROJECT_SOURCE_DIR}/unittests)
