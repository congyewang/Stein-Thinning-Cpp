cmake_minimum_required(VERSION 3.10)
project(stein_thinning_test CXX)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Try to find out OpenBLAS
find_package(BLAS)

# Search for OpenBLAS library files
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
set(OPENBLAS_LIB_DIR "${PARENT_DIR}/external/openblas/*/build/lib")
file(GLOB OPENBLAS_LIB_FILES_A "${OPENBLAS_LIB_DIR}/libopenblas.a")
file(GLOB OPENBLAS_LIB_FILES_SO "${OPENBLAS_LIB_DIR}/libopenblas.so")

if(BLAS_FOUND)
    message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
elseif(EXISTS ${OPENBLAS_LIB_FILES_A})
    message(STATUS "OpenBLAS found: ${OPENBLAS_LIB_FILES_A}")
    set(BLAS_LIBRARIES ${OPENBLAS_LIB_FILES_A})
elseif(EXISTS ${OPENBLAS_LIB_FILES_SO})
    message(STATUS "OpenBLAS found: ${OPENBLAS_LIB_FILES_SO}")
    set(BLAS_LIBRARIES ${OPENBLAS_LIB_FILES_SO})
else()
    message(STATUS "OpenBLAS not found. Downloading and building OpenBLAS...")

    # Set path and url of OpenBLAS
    set(OPENBLAS_VERSION "0.3.28")
    set(OPENBLAS_URL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz")
    set(OPENBLAS_DIR "${PARENT_DIR}/external/openblas")
    set(OPENBLAS_SRC_DIR "${OPENBLAS_DIR}/OpenBLAS-${OPENBLAS_VERSION}")
    set(OPENBLAS_BUILD_DIR "${OPENBLAS_SRC_DIR}/build")
    set(OPENBLAS_LIB_DIR "${OPENBLAS_BUILD_DIR}/lib")
    set(OPENBLAS_TAR "${OPENBLAS_DIR}/openblas.tar.gz")

    # Download OpenBLAS
    file(DOWNLOAD ${OPENBLAS_URL} ${OPENBLAS_TAR})

    # Extract OpenBLAS
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${OPENBLAS_TAR}
        WORKING_DIRECTORY ${OPENBLAS_DIR}
    )

    # Ensure that the extracted source directory exists
    if(NOT EXISTS ${OPENBLAS_SRC_DIR})
        message(FATAL_ERROR "OpenBLAS source directory not found after extraction.")
    endif()

    # Remove the downloaded tar.gz file
    file(REMOVE ${OPENBLAS_TAR})
    message(STATUS "Removed downloaded OpenBLAS tarball: ${OPENBLAS_TAR}")

    # Create build directory
    file(MAKE_DIRECTORY ${OPENBLAS_BUILD_DIR})

    # Build OpenBLAS
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${OPENBLAS_SRC_DIR} -B ${OPENBLAS_BUILD_DIR} -G ${CMAKE_GENERATOR}
        WORKING_DIRECTORY ${OPENBLAS_BUILD_DIR}
    )

    execute_process(
        COMMAND ${CMAKE_MAKE_PROGRAM}
        WORKING_DIRECTORY ${OPENBLAS_BUILD_DIR}
    )

    # Set OPENBLAS
    set(BLAS_LIBRARIES
        ${OPENBLAS_LIB_DIR}/libopenblas.a
    )

    # Include OpenBLAS directory
    include_directories(${OPENBLAS_LIB_DIR})

endif()

set(SOURCES
    ${PARENT_DIR}/src/kernel/kernel.cpp
    ${PARENT_DIR}/src/kernel/kmat.tpp
    ${PARENT_DIR}/src/kernel/stein.tpp
    ${PARENT_DIR}/src/kernel/thinning.cpp
    ${PARENT_DIR}/src/kernel/utils.cpp
    test.cpp
)
set(HEADERS
    ${PARENT_DIR}/src/kernel/kernel.h
    ${PARENT_DIR}/src/kernel/thinning.h
    ${PARENT_DIR}/src/kernel/utils.h
)

include_directories(${PARENT_DIR}/src)
include_directories(${PARENT_DIR}/external/armadillo/include)

aux_source_directory(src SRCS)

enable_testing()

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES} GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
