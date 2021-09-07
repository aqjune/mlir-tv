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

SUBDIRLIST(PASSES "${PROJECT_SOURCE_DIR}/tests/long-opts")
foreach(PASS_NAME ${PASSES})
  add_test(NAME Longopts-${PASS_NAME}
    COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/passes.py "${CMAKE_CURRENT_BINARY_DIR}" -v --param pass=${PASS_NAME} --param root=long-opts)
endforeach()

SUBDIRLIST(PASSES "${PROJECT_SOURCE_DIR}/tests/litmus")
foreach(PASS_NAME ${PASSES})
  add_test(NAME Litmus-${PASS_NAME}
    COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/passes.py "${CMAKE_CURRENT_BINARY_DIR}" -v --param pass=${PASS_NAME} --param root=litmus)
endforeach()
