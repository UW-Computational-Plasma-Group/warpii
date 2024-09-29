find_package(Doxygen)


#
# Set up the doxygen target
#

set(_doxygen_input ${CMAKE_SOURCE_DIR}/README.md)
list(APPEND _doxygen_input
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_SOURCE_DIR}/src/
    ${CMAKE_SOURCE_DIR}/docs/)

file(GLOB _doxygen_depend
    ${CMAKE_SOURCE_DIR}/include/**/*.h
    ${CMAKE_SOURCE_DIR}/src/**/*.h
    ${CMAKE_SOURCE_DIR}/src/**/*.cc
    )

set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/doxyfile")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/docs/doxyfile.in" ${doxyfile} @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/docs/doxylayout.in.xml"
    "${CMAKE_CURRENT_BINARY_DIR}/doxylayout.xml" COPYONLY)

string(REPLACE ";" " " _doxygen_input_string "${_doxygen_input}")
file(APPEND "${doxyfile}"
  "
INPUT=${_doxygen_input_string}
  "
  )

add_custom_target(doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile} 
    > ${CMAKE_BINARY_DIR}/doxygen.log 2>&1 # doxygen be quiet
    COMMAND ${CMAKE_COMMAND} -E echo "-- Documentation is available at ${CMAKE_CURRENT_BINARY_DIR}/html/index.html"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${DOXYGEN_EXECUTABLE} ${doxyfile} ${_doxygen_depend}
    COMMENT "Generating documentation with Doxygen"
    VERBATIM)

#
# Commands to build params html pages
#

set(app_params_pages "")
set(params_doc_script "${CMAKE_SOURCE_DIR}/docs/scripts/build_params_doc_pages.sh")
foreach(app five_moment)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/${app}_params_copy.html
        COMMAND ${params_doc_script} ${app} ${CMAKE_SOURCE_DIR}
        DEPENDS warpii doxygen ${params_doc_script}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating param doc pages for ${app}"
        VERBATIM)
    list(APPEND app_params_pages "${CMAKE_CURRENT_BINARY_DIR}/html/${app}_params_copy.html")
endforeach()

add_custom_target(documentation DEPENDS doxygen ${app_params_pages})

add_test(
    NAME DoxygenTest
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target documentation)
