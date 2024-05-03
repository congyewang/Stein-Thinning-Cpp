cmake_minimum_required(VERSION 3.10)

project(test CXX)

set(CMAKE_CXX_STANDARD 14)
set(SOURCES
    ../src/kernel/kernel.cpp
    ../src/kernel/kmat.tpp
    ../src/kernel/stein.tpp
    ../src/kernel/thinning.cpp
    ../src/kernel/utils.cpp
    test.cpp
)
set(HEADERS
    ../src/kernel/kernel.h
    ../src/kernel/thinning.h
    ../src/kernel/utils.h
)
set(OPENBLAS
  ${CMAKE_SOURCE_DIR}/../include/openblas/lib/libopenblas.a

)

include_directories(../src)
include_directories(../include/armadillo/include ../include/openblas/lib)

aux_source_directory(src SRCS)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${OPENBLAS})

