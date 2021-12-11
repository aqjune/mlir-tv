cmake_minimum_required(VERSION 3.15.0)
project(mlir-tv VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 17)

string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

set(MLIR_DIR CACHE PATH "MLIR installation top-level directory")
set(Z3_DIR CACHE PATH "Z3 installation top-level directory")
set(CVC5_DIR CACHE PATH "CVC5 installation top-level directory")
option(USE_LIBC "Use libc++ in case the MLIR (and CVC5) is linked against libc++")

set(MLIR_INC_DIR "${MLIR_DIR}/include")
set(MLIR_LIB_DIR "${MLIR_DIR}/lib")
set(Z3_INC_DIR "${Z3_DIR}/include")
set(Z3_LIB_DIR "${Z3_DIR}/lib")
set(CVC5_INC_DIR "${CVC5_DIR}/include")
set(CVC5_LIB_DIR "${CVC5_DIR}/lib")

# /============================================================/
# 1. Build object files to check warnings/errors before linking
# /============================================================/

set(PROJECT_OBJ "mlirtvobj")
add_library(${PROJECT_OBJ} OBJECT
    src/abstractops.cpp
    src/analysis.cpp
    src/debug.cpp
    src/encode.cpp
    src/memory.cpp
    src/print.cpp
    src/smt.cpp
    src/state.cpp
    src/utils.cpp
    src/value.cpp
    src/vcgen.cpp)

# Check MLIR headers and include if available
if(NOT MLIR_DIR)
    message(FATAL_ERROR "path to MLIR must be provided!")
elseif(NOT EXISTS ${MLIR_INC_DIR})
    message(FATAL_ERROR "cannot find MLIR include directory!")
else()
    target_include_directories(${PROJECT_OBJ} PUBLIC ${MLIR_INC_DIR})
endif()

# Check if at least one solver is available
if(NOT Z3_DIR AND NOT CVC5_DIR)
    message(FATAL_ERROR "path to at least one of the solvers must be provided!")
endif()

# Check Z3 header and include if available
if(Z3_DIR)
    if(NOT EXISTS ${Z3_INC_DIR})
        message(FATAL_ERROR "cannot find Z3 include directory!")
    else()
        target_include_directories(${PROJECT_OBJ} PUBLIC ${Z3_INC_DIR})
        target_compile_definitions(${PROJECT_OBJ} PUBLIC SOLVER_Z3)
    endif()
endif()

# Check CVC5 header and include if available
if(CVC5_DIR)
    if(NOT EXISTS ${CVC5_INC_DIR})
        message(FATAL_ERROR "cannot find CVC5 include directory!")
    else()
        target_include_directories(${PROJECT_OBJ} PUBLIC ${CVC5_INC_DIR})
        target_compile_definitions(${PROJECT_OBJ} PUBLIC SOLVER_CVC5)
    endif()
endif()

# Warn about unused variables
target_compile_options(${PROJECT_OBJ} PUBLIC -Wunused-variable)

# Try using libc if possible
if(USE_LIBC)
    target_compile_options(${PROJECT_OBJ} PUBLIC -stdlib=libc++)
endif()

# Get magic_enum using CPM
include(cmake/CPM.cmake)
CPMAddPackage(
    NAME magic_enum
    GITHUB_REPOSITORY Neargye/magic_enum
    GIT_TAG v0.7.3
)
# Well, magic_enum does not copy the include directory into its build directory :(
# Manual registration is needed.
target_include_directories(${PROJECT_OBJ} PUBLIC ${magic_enum_SOURCE_DIR}/include)

# /============================================================/
# 2. Build libmlirtv
# /============================================================/

set(PROJECT_LIB "mlirtv")
add_library(${PROJECT_LIB} STATIC)
target_link_libraries(${PROJECT_LIB} PUBLIC ${PROJECT_OBJ})

# Check MLIR libraries and link if possible
if(NOT EXISTS ${MLIR_LIB_DIR})
    message(FATAL_ERROR "cannot find MLIR library directory!")
else()
    target_link_directories(${PROJECT_LIB} PUBLIC ${MLIR_LIB_DIR})
    list(APPEND LIB_LIST
        MLIRViewLikeInterface MLIRInferTypeOpInterface MLIRControlFlowInterfaces MLIRSideEffectInterfaces
        MLIRIR MLIRDialect MLIRDialectUtils MLIRLinalg MLIRAffine MLIRMemRef
        MLIRShape MLIRMath MLIRSparseTensor MLIRSCF MLIRArithmetic MLIRBufferization
        MLIRStandard MLIRMemRefUtils MLIRTensor MLIRTosa MLIRQuant MLIRParser MLIRSupport
        LLVMSupport LLVMDemangle pthread m curses)
    if (APPLE) # Apple LLD does not support 'group' flags
        target_link_libraries(${PROJECT_LIB} PUBLIC ${LIB_LIST})
    else()
        target_link_libraries(${PROJECT_LIB} PUBLIC "-Wl,--start-group" ${LIB_LIST} "-Wl,--end-group")
    endif()
endif()

# Check Z3 library and link if possible
if(Z3_DIR)
    if(NOT EXISTS ${Z3_LIB_DIR})
        message(FATAL_ERROR "cannot find Z3 library directory!")
    else()
        target_link_directories(${PROJECT_LIB} PUBLIC ${Z3_LIB_DIR})
        target_link_libraries(${PROJECT_LIB} PUBLIC z3)
    endif()
endif()

# Check CVC5 library and link if possible
if(CVC5_DIR)
    if(NOT EXISTS ${CVC5_LIB_DIR})
        message(FATAL_ERROR "cannot find CVC5 library directory!")
    else()
        target_link_directories(${PROJECT_LIB} PUBLIC ${CVC5_LIB_DIR})
        target_link_libraries(${PROJECT_LIB} PUBLIC cvc5)
    endif()
endif()

# Try using libc if possible
if(USE_LIBC)
    target_link_options(${PROJECT_LIB} PUBLIC -stdlib=libc++)
endif()

# /============================================================/
# 3. Build binaries
# /============================================================/

# Build executable
add_executable(${PROJECT_NAME} src/main.cpp)
add_dependencies(${PROJECT_NAME} ${PROJECT_LIB})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LIB})
target_compile_options(${PROJECT_NAME} PRIVATE -fno-rtti)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
# Reactivate this after unit tests are updated to use the new SMT wrapper classes
# add_subdirectory(${PROJECT_SOURCE_DIR}/unittests)
