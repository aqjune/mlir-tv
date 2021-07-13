cmake_minimum_required(VERSION 3.15.0)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

# https://stackoverflow.com/questions/7787823/cmake-how-to-get-the-name-of-all-subdirectories-of-a-directory
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(PASSES "${PROJECT_SOURCE_DIR}/tests/opts")
foreach(PASS_NAME ${PASSES})
  add_test(NAME Opts-${PASS_NAME}
    COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/passes.py "${CMAKE_CURRENT_BINARY_DIR}" -v --param pass=${PASS_NAME} --param root=opts)
endforeach()

add_test(NAME Litmus
    COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/passes.py "${CMAKE_CURRENT_BINARY_DIR}" -v --param pass=linalg-litmus --param root=litmus)
