cmake_minimum_required(VERSION 3.15.0)
project(mlir-tv VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 17)

set(MLIR_DIR CACHE PATH "MLIR installation top-level directory")
set(Z3_DIR CACHE PATH "Z3 installation top-level directory")
option(USE_LIBC "Use libc++ in case the MLIR is linked against libc++")

set(MLIR_INC_DIR "${MLIR_DIR}/include")
set(MLIR_LIB_DIR "${MLIR_DIR}/lib")
set(Z3_INC_DIR "${Z3_DIR}/include")
set(Z3_LIB_DIR "${Z3_DIR}/lib")

# Build library first
set(PROJECT_LIB "mlirtv")
add_library(${PROJECT_LIB} STATIC
    src/abstractops.cpp
    src/encode.cpp
    src/memory.cpp
    src/print.cpp
    src/smt.cpp
    src/state.cpp
    src/value.cpp
    src/vcgen.cpp)

# Check for MLIR installation
if(NOT MLIR_DIR)
    message(FATAL_ERROR "path to MLIR must be provided!")
elseif(NOT EXISTS ${MLIR_INC_DIR} OR NOT EXISTS ${MLIR_LIB_DIR})
    message(FATAL_ERROR "the provided path doesn't seem to be a valid MLIR directory")
else()
    target_include_directories(${PROJECT_LIB} PUBLIC ${MLIR_INC_DIR})
    target_link_directories(${PROJECT_LIB} PUBLIC ${MLIR_LIB_DIR})

    list(APPEND LIB_LIST
        MLIRViewLikeInterface MLIRInferTypeOpInterface MLIRControlFlowInterfaces MLIRSideEffectInterfaces
        MLIRIR MLIRDialect MLIRDialectUtils MLIRLinalg MLIRAffine MLIRMemRef MLIRShape MLIRMath
        MLIRStandard MLIRMemRefUtils MLIRTensor MLIRParser MLIRSupport
        LLVMSupport LLVMDemangle pthread m curses)
    if (APPLE) # Apple LLD does not support 'group' flags
        target_link_libraries(${PROJECT_LIB} PUBLIC ${LIB_LIST})
    else()
        target_link_libraries(${PROJECT_LIB} PUBLIC "-Wl,--start-group" ${LIB_LIST} "-Wl,--end-group")
    endif()
endif()

# Check if at least one solver is available
if(NOT Z3_DIR)
    message(FATAL_ERROR "path to at least one of the solvers must be provided!")
endif()

# Check for Z3 installation
if(Z3_DIR)
    if(NOT EXISTS ${Z3_INC_DIR} OR NOT EXISTS ${Z3_LIB_DIR})
        message(FATAL_ERROR "the provided path doesn't seem to be a valid Z3 directory")
    else()
        target_include_directories(${PROJECT_LIB} PUBLIC ${Z3_INC_DIR})
        target_link_directories(${PROJECT_LIB} PUBLIC ${Z3_LIB_DIR})
        target_link_libraries(${PROJECT_LIB} PUBLIC z3)
    endif()
endif()

# Try using libc if possible
if(USE_LIBC)
    target_compile_options(${PROJECT_LIB} PUBLIC -stdlib=libc++)
    target_link_options(${PROJECT_LIB} PUBLIC -stdlib=libc++)
endif()

# Build executable
add_executable(${PROJECT_NAME} src/main.cpp)
add_dependencies(${PROJECT_NAME} ${PROJECT_LIB})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LIB})
target_compile_options(${PROJECT_NAME} PRIVATE -fno-rtti)

# Build interpreter
add_executable(mlir-interp src/interp.cpp)
add_dependencies(mlir-interp ${PROJECT_LIB})
target_link_libraries(mlir-interp PRIVATE ${PROJECT_LIB})
target_compile_options(mlir-interp PRIVATE -fno-rtti)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
add_subdirectory(${PROJECT_SOURCE_DIR}/unittests)
