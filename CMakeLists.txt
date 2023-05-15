cmake_minimum_required(VERSION 3.10)

project(main CXX)

set(CMAKE_CXX_STANDARD 14)

include_directories(include include/armadillo/include)

link_directories(include/armadillo/lib)

aux_source_directory(src SRCS)

add_executable(${PROJECT_NAME} ${SRCS} src/main.cpp src/kernel.h)

target_link_libraries(${PROJECT_NAME} libopenblas.a)

