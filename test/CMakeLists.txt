include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  EXCLUDE_FROM_ALL
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

set(TEST_SOURCES
    input_test.cc
    timestepper_test.cc
    euler_test.cc
    conservation_test.cc
    #current_drive_test.cc (Disabled until we can select which implicit solver to use)
    dof_utils_test.cc
    plasma_waves_test.cc
    shock_capturing_fv_test.cc
    utilities_test.cc
    phmaxwell_test.cc
    bc_extension_test.cc
    species_func_test.cc
    cell_evaluators_test.cc
    snowplow_test.cc
    entropy_stability_test.cc
    rk_test.cc
    # Add more test files here as needed
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE}
        euler_test_helpers.cc five_moment_test_helpers.cc)
    target_link_libraries(${TEST_NAME} GTest::gtest_main libwarpii)
    target_link_libraries(${TEST_NAME} ${MPI_CXX_LIBRARIES})
    target_include_directories(${TEST_NAME} PRIVATE "${CMAKE_SOURCE_DIR}")
    deal_ii_setup_target(${TEST_NAME})
    gtest_discover_tests(${TEST_NAME})
endforeach()

add_test(
    NAME HelpTest
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run-help.sh" "${CMAKE_BINARY_DIR}/warpii"
)
add_test(
    NAME FPETest
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/fpe-enabled-test.sh" "${CMAKE_BINARY_DIR}/warpii"
)
add_test(
    NAME CompileExtensionTest
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compile-extension.sh" "${CMAKE_BINARY_DIR}/extensions/Makefile.example"
)
add_test(
    NAME Pragmas
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/headers-start-with-pragma-once.sh"
)
set_tests_properties(Pragmas PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_subdirectory(doctests)

set(EXAMPLE_PAIRS
    "five-moment/freestream_1d.inp 0.01"
    "five-moment/freestream_2d_diagonal.inp 0.01"
    "five-moment/langmuir_sheath.inp 0.01"
    "five-moment/langmuir_wave.inp 0.01"
    "five-moment/shu_osher_shock_1d.inp 0.01"
    "five-moment/snowplow.inp 0.001"
    "five-moment/sod_shocktube.inp 0.01"
    "five-moment/source_outflow.inp 0.01"
    "phmaxwell/lightprop_periodic_1d.inp 0.01"
    )

foreach(PAIR ${EXAMPLE_PAIRS})
    string(REPLACE ";" " " PAIR_SPACED "${PAIR}")
    separate_arguments(PAIR_SPLIT UNIX_COMMAND "${PAIR_SPACED}")
    list(GET PAIR_SPLIT 0 EXAMPLE)
    list(GET PAIR_SPLIT 1 T_END)
    add_test(
        NAME Example__${EXAMPLE}
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run-example-as-regression-test.sh" "${CMAKE_BINARY_DIR}/warpii" "${CMAKE_SOURCE_DIR}/examples/${EXAMPLE}" "${T_END}"
        )
endforeach()

