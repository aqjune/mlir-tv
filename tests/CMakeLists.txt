cmake_minimum_required(VERSION 3.15.0)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_test(NAME Passes-conv2d-to-img2col
    COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/passes.py "${CMAKE_CURRENT_BINARY_DIR}" -v --param pass=conv2d-to-img2col)
add_test(NAME Passes-fold-memref-subview-op
    COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/passes.py "${CMAKE_CURRENT_BINARY_DIR}" -v --param pass=fold-memref-subview-op)
add_test(NAME Passes-fold-tensor-extract-op
    COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/passes.py "${CMAKE_CURRENT_BINARY_DIR}" -v --param pass=fold-tensor-extract-op)
add_test(NAME Passes-fusion-tensor
    COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/passes.py "${CMAKE_CURRENT_BINARY_DIR}" -v --param pass=fusion-tensor)
add_test(NAME Passes-llvmcpu-plan-conv-loop-order
    COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/passes.py "${CMAKE_CURRENT_BINARY_DIR}" -v --param pass=llvmcpu-plan-conv-loop-order)
