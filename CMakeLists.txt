cmake_minimum_required(VERSION 3.10)

project(main CXX)

set(CMAKE_CXX_STANDARD 14)
set(SOURCES
    src/kernel/kernel.cpp
    src/kernel/kmat.tpp
    src/main.cpp
)
set(HEADERS
    src/kernel/kernel.h
)

include_directories(include include/armadillo/include)

link_directories(include/armadillo/lib)

aux_source_directory(src SRCS)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} libopenblas.a)
