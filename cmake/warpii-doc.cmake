find_package(Doxygen)

set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/doxyfile")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/docs/doxyfile.in"
    ${doxyfile} @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/docs/doxylayout.in.xml"
    "${CMAKE_CURRENT_BINARY_DIR}/doxylayout.xml" COPYONLY)

set(docs_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/docs")
set(docs_dest_dir "${CMAKE_CURRENT_BINARY_DIR}/docs")

add_custom_target(copy-readme
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/README.md" "${CMAKE_CURRENT_BINARY_DIR}/README.md"
    COMMENT "Copying README.md to build tree")

add_custom_target(copy-docs
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${docs_src_dir} ${docs_dest_dir}
    COMMENT "Copying docs/ directory to build tree")

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
    DEPENDS ${DOXYGEN_EXECUTABLE} ${doxyfile} ${docs_src_dir}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating documentation with Doxygen"
    VERBATIM)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/param_doc_pages
    COMMAND ${docs_dest_dir}/scripts/build_params_doc_pages.sh
    DEPENDS ${DOXYGEN_EXECUTABLE} ${doxyfile} ${docs_src_dir} ${docs_src_dir}/scripts/build_params_doc_pages.sh ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating param doc pages"
    VERBATIM)

add_custom_target(documentation
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/param_doc_pages ${CMAKE_CURRENT_BINARY_DIR}/html/index.html)

add_dependencies(documentation copy-docs copy-readme)

add_test(
    NAME DoxygenTest
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target documentation)
