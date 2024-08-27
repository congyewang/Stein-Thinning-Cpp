cmake_minimum_required(VERSION 3.10)

project(stein_thinning CXX)
set(CMAKE_CXX_STANDARD 14)

# Try to find out OpenBLAS
find_package(BLAS)

# Search for OpenBLAS library files
set(OPENBLAS_LIB_DIR "${CMAKE_SOURCE_DIR}/external/openblas/*/build/lib")
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
    set(OPENBLAS_DIR "${CMAKE_SOURCE_DIR}/external/openblas")
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
    ${CMAKE_SOURCE_DIR}/src/kernel.cpp
    ${CMAKE_SOURCE_DIR}/src/thinning.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)
set(HEADERS
    ${CMAKE_SOURCE_DIR}/include/Stein-Thinning-Cpp/kernel.h
    ${CMAKE_SOURCE_DIR}/include/Stein-Thinning-Cpp/thinning.h
    ${CMAKE_SOURCE_DIR}/include/Stein-Thinning-Cpp/utils.h
    ${CMAKE_SOURCE_DIR}/include/Stein-Thinning-Cpp/kmat.tpp
    ${CMAKE_SOURCE_DIR}/include/Stein-Thinning-Cpp/stein.tpp
)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include/Stein-Thinning-Cpp)
include_directories(external/armadillo/include)

aux_source_directory(src SRCS)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})
