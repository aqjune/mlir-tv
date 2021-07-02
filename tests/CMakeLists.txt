cmake_minimum_required(VERSION 3.15.0)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
configure_file(lit.cfg.py.in lit.cfg.py @ONLY)

add_test(NAME test-passes
    COMMAND ${PYTHON_BIN} ${PROJECT_SOURCE_DIR}/tests/passes.py "${CMAKE_CURRENT_BINARY_DIR}" -v)