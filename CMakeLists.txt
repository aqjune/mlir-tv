cmake_minimum_required(VERSION 3.15.0)
project(mlir-tv VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 17)

set(MLIR_DIR "" CACHE STRING "MLIR installation top-level directory")
set(Z3_DIR "" CACHE STRING "Z3 installation top-level directory")

set(MLIR_INC_DIR "${MLIR_DIR}/include")
set(MLIR_LIB_DIR "${MLIR_DIR}/lib")
set(Z3_INC_DIR "${Z3_DIR}/include")
set(Z3_LIB_DIR "${Z3_DIR}/lib")

link_directories(${MLIR_LIB_DIR})
link_directories(${Z3_LIB_DIR})

link_libraries(
    MLIRViewLikeInterface MLIRInferTypeOpInterface
    MLIRIR MLIRDialect MLIRDialectUtils MLIRLinalg MLIRViewLikeInterface MLIRAffine MLIRMemRef MLIRShape MLIRMath
    MLIRControlFlowInterfaces MLIRStandard MLIRSideEffectInterfaces MLIRMemRefUtils MLIRTensor MLIRParser MLIRSupport MLIRInferTypeOpInterface
    LLVMSupport LLVMDemangle z3 pthread m curses)

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

include_directories(${MLIR_INC_DIR})
include_directories(${Z3_INC_DIR})

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
add_subdirectory(${PROJECT_SOURCE_DIR}/unittests)
