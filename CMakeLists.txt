cmake_minimum_required(VERSION 3.15.0)
project(iree-tv VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 17)

set(IREE_DIR "~/src/iree" CACHE STRING "IREE source top-level directory")
set(IREE_BUILD_DIR "~/src/iree-build" CACHE STRING "IREE build top-level directory")
set(Z3_INCLUDE_DIR "~/src/z3" CACHE STRING "Z3 include directory")
set(Z3_LIBRARY_DIR "~/src/z3" CACHE STRING "Z3 lib directory")

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/vcgen.cpp
    src/state.cpp
    src/tensor.cpp
    src/smt.cpp)

#target_link_options(${PROJECT_NAME} PUBLIC -lz3 -ltinfo -lLLVMSupport -lMLIRStandard -lMLIRControlFlowInterfaces -lMLIRSideEffectInterfaces -lMLIRViewLikeInterface -lMLIRInferTypeOpInterface -lMLIRIR -lMLIRDialect -lMLIRDialectUtils -lMLIRLinalg -lMLIRAffine -lMLIRMemRef -lMLIRMemRefUtils -lMLIRTensor -lMLIRParser -lMLIRSupport -lpthread -lm -fPIC)
target_link_libraries(${PROJECT_NAME}
    MLIRStandard MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces MLIRViewLikeInterface MLIRInferTypeOpInterface
    MLIRIR MLIRDialect MLIRDialectUtils MLIRLinalg MLIRAffine MLIRMemRef
    MLIRMemRefUtils MLIRTensor MLIRParser MLIRSupport MLIRInferTypeOpInterface
    LLVMSupport LLVMDemangle z3 pthread m curses)
target_link_options(${PROJECT_NAME} PUBLIC -fPIC)
target_compile_options(${PROJECT_NAME} PUBLIC -fno-rtti)
target_link_directories(${PROJECT_NAME} PUBLIC ${IREE_BUILD_DIR}/third_party/llvm-project/llvm/lib)
target_link_directories(${PROJECT_NAME} PUBLIC ${Z3_LIBRARY_DIR})

# IREE include directories
include_directories(${IREE_DIR}/third_party/llvm-project/llvm/include ${IREE_DIR}/third_party/llvm-project/mlir/include)
include_directories(${IREE_BUILD_DIR}/third_party/llvm-project/llvm/include ${IREE_BUILD_DIR}/third_party/llvm-project/llvm/tools/mlir/include)

# Z3 API directories
include_directories(${Z3_INCLUDE_DIR})
