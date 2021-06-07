cmake_minimum_required(VERSION 3.15.0)
project(iree-tv VERSION 0.1.0)

# >>> modify directories below according to your environment <<<
set(IREE_DIR "~/src/iree") # IREE source top-level directory
set(IREE_BUILD_DIR "~/src/iree-build") # built IREE top-level directory
set(Z3_DIR "~/src/z3") # Z3 source top-level directory

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_directories(${PROJECT_NAME} PUBLIC ${IREE_DIR}/third_party/llvm-project/llvm/lib ${IREE_BUILD_DIR}/third_party/llvm-project/llvm/lib)
target_link_options(${PROJECT_NAME} PUBLIC -lz3 -ltinfo -lLLVMSupport -lMLIRStandard -lMLIRControlFlowInterfaces -lMLIRSideEffectInterfaces -lMLIRViewLikeInterface -lMLIRInferTypeOpInterface -lMLIRIR -lMLIRDialect -lMLIRDialectUtils -lMLIRLinalg -lMLIRAffine -lMLIRMemRef -lMLIRMemRefUtils -lMLIRTensor -lMLIRParser -lMLIRSupport -lpthread -lm -fPIC)
target_compile_options(${PROJECT_NAME} PUBLIC -fno-rtti)

# IREE include directories
include_directories(${IREE_DIR}/third_party/llvm-project/llvm/include ${IREE_DIR}/third_party/llvm-project/mlir/include)
include_directories(${IREE_BUILD_DIR}/third_party/llvm-project/llvm/include ${IREE_BUILD_DIR}/third_party/llvm-project/llvm/tools/mlir/include)

# Z3 API directories
include_directories(${Z3_DIR}/src/api ${Z3_DIR}/src/api/c++)