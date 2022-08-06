cmake_minimum_required(VERSION 3.15.0)
project(mlir-tv VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 17)

string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

set(MLIR_DIR CACHE PATH "MLIR installation top-level directory, followed by /lib/cmake/mlir")
set(Z3_DIR CACHE PATH "Z3 installation top-level directory")
set(CVC5_DIR CACHE PATH "CVC5 installation top-level directory")
option(USE_LIBC "Use libc++ in case the MLIR (and CVC5) is linked against libc++")

set(Z3_INC_DIR "${Z3_DIR}/include")
set(Z3_LIB_DIR "${Z3_DIR}/lib")
set(CVC5_INC_DIR "${CVC5_DIR}/include")
set(CVC5_LIB_DIR "${CVC5_DIR}/lib")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

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

# Check MLIR and LLVM headers and include
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Check if at least one solver is available
if(NOT Z3_DIR AND NOT CVC5_DIR)
    message(FATAL_ERROR "path to at least one of the solvers must be provided!")
endif()

# Check Z3 header and include if available
if(Z3_DIR)
    if(NOT EXISTS ${Z3_INC_DIR})
        message(FATAL_ERROR "cannot find Z3 include directory!")
    endif()
    target_include_directories(${PROJECT_OBJ} PUBLIC ${Z3_INC_DIR})
    target_compile_definitions(${PROJECT_OBJ} PUBLIC SOLVER_Z3)
endif()

# Check CVC5 header and include if available
if(CVC5_DIR)
    if(NOT EXISTS ${CVC5_INC_DIR})
        message(FATAL_ERROR "cannot find CVC5 include directory!")
    endif()
    target_include_directories(${PROJECT_OBJ} PUBLIC ${CVC5_INC_DIR})
    target_compile_definitions(${PROJECT_OBJ} PUBLIC SOLVER_CVC5)
endif()

# Warn about unused variables
target_compile_options(${PROJECT_OBJ} PUBLIC -Wunused-variable)
# Using cl::opt requires this
target_compile_options(${PROJECT_OBJ} PUBLIC -fno-rtti)

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

# Check MLIR libraries and link 
link_directories(${LLVM_BUILD_LIBRARY_DIR})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIB_LIST ${dialect_libs})
target_link_libraries(${PROJECT_LIB} PUBLIC ${LIB_LIST} pthread m curses)
llvm_update_compile_flags(${PROJECT_LIB})

# Check Z3 library and link if possible
if(Z3_DIR)
    if(NOT EXISTS ${Z3_LIB_DIR})
        message(FATAL_ERROR "cannot find Z3 library directory!")
    endif()
    target_link_directories(${PROJECT_LIB} PUBLIC ${Z3_LIB_DIR})
    target_link_libraries(${PROJECT_LIB} PUBLIC z3)
endif()

# Check CVC5 library and link if possible
if(CVC5_DIR)
    if(NOT EXISTS ${CVC5_LIB_DIR})
        message(FATAL_ERROR "cannot find CVC5 library directory!")
    endif()
    target_link_directories(${PROJECT_LIB} PUBLIC ${CVC5_LIB_DIR})
    target_link_libraries(${PROJECT_LIB} PUBLIC cvc5)
endif()

# Try using libc if possible
if(USE_LIBC)
    target_link_options(${PROJECT_LIB} PUBLIC -stdlib=libc++)
endif()


# /============================================================/
# 3. Build executable
# /============================================================/

# Build executable
add_executable(${PROJECT_NAME} src/main.cpp)
add_dependencies(${PROJECT_NAME} ${PROJECT_LIB})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LIB})

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
# Reactivate this after unit tests are updated to use the new SMT wrapper classes
# add_subdirectory(${PROJECT_SOURCE_DIR}/unittests)
