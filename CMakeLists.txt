cmake_minimum_required(VERSION 3.8)
project(mnist)

set(CMAKE_CXX_STANDARD 20)

aux_source_directory(src SRC)

set(SOURCE_FILES main.cpp ${SRC})
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
