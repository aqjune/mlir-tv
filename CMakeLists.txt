cmake_minimum_required(VERSION 3.13.0)
project(mlir-tv VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

# CMake usually does decent job in finding out the proper dependency packages,
# so for most of the times you won't have to care about the <NAME>_ROOT variables.
# If you are not satisfied with the CMake's choices, you may use these variables
# to manually override them with the MLIR, Z3, or cvc5 you prefer.
set(MLIR_ROOT CACHE PATH "MLIR installation top-level directory")
set(Z3_ROOT CACHE PATH "Z3 installation top-level directory")
set(cvc5_ROOT CACHE PATH "cvc5 installation top-level directory")
option(USE_LIBC "Use libc++ in case the MLIR (and cvc5) is linked against libc++")
option(USE_Z3 "Use Z3 solver backend")
option(USE_cvc5 "Use cvc5 solver backend")

if ((NOT USE_Z3) AND (NOT USE_cvc5))
    message(FATAL_ERROR "No solver was specified; please add -DUSE_Z3=ON or -DUSE_cvc5=ON")
endif()

if(USE_Z3)
    find_package(Z3 4.8.13)
    if(Z3_FOUND)
        add_compile_definitions(SOLVER_Z3)
        message(STATUS "Found Z3 ${Z3_VERSION} from ${Z3_DIR}/Z3Config.cmake")
    else()
        message(FATAL_ERROR "Could not locate Z3")
    endif()
endif()

if(USE_cvc5)
    find_package(cvc5 0.0.3)
    if(cvc5_FOUND)
        set(CVC5_LIBRARY cvc5::cvc5-shared)
        get_property(CVC5_INCLUDE_DIRS TARGET ${CVC5_LIBRARY} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        add_compile_definitions(SOLVER_CVC5)
        message(STATUS "Found cvc5 ${cvc5_VERSION} from ${cvc5_DIR}/cvc5Config.cmake")
    else()
        message(FATAL_ERROR "Could not locate cvc5")
    endif()
endif()

find_package(MLIR REQUIRED)
# MLIR_VERSION does not exist, so use LLVM_VERSION instead
# If package MLIR is found, package LLVM must have been found in the process
message(STATUS "Found MLIR ${LLVM_VERSION} from ${MLIR_DIR}/MLIRConfig.cmake")
include(AddLLVM)

# /============================================================/
# 1. Build object files to check warnings/errors before linking
# /============================================================/

set(PROJECT_OBJ "mlirtvobj")
add_library(${PROJECT_OBJ} OBJECT
    src/abstractops.cpp
    src/analysis.cpp
    src/debug.cpp
    src/encode.cpp
    src/function.cpp
    src/memory.cpp
    src/print.cpp
    src/smt.cpp
    src/state.cpp
    src/utils.cpp
    src/value.cpp
    src/vcgen.cpp)

# Check MLIR and LLVM headers and include
target_include_directories(${PROJECT_OBJ} PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(${PROJECT_OBJ} PUBLIC ${MLIR_INCLUDE_DIRS})

if(Z3_FOUND)
    target_include_directories(${PROJECT_OBJ} PUBLIC ${Z3_CXX_INCLUDE_DIRS})
endif()
if(cvc5_FOUND)
    target_include_directories(${PROJECT_OBJ} PUBLIC ${CVC5_INCLUDE_DIRS})
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
add_library(${PROJECT_LIB})
target_link_libraries(${PROJECT_LIB} PUBLIC ${PROJECT_OBJ})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIB_LIST ${dialect_libs})
target_link_libraries(${PROJECT_LIB} PUBLIC ${LIB_LIST} pthread m curses)
llvm_update_compile_flags(${PROJECT_LIB})

if(Z3_FOUND)
    target_link_libraries(${PROJECT_LIB} PRIVATE ${Z3_LIBRARIES})
endif()
if(cvc5_FOUND)
    target_link_libraries(${PROJECT_LIB} PRIVATE ${CVC5_LIBRARY})
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
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LIB})

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
# Reactivate this after unit tests are updated to use the new SMT wrapper classes
# add_subdirectory(${PROJECT_SOURCE_DIR}/unittests)
