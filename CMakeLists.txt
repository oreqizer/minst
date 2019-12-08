cmake_minimum_required(VERSION 3.8)
project(mnist)

set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

aux_source_directory(src SRC)

set(SOURCE_FILES main.cpp ${SRC})
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

FILE(COPY data/mnist_train_labels.csv DESTINATION "${CMAKE_BINARY_DIR}/data")
FILE(COPY data/mnist_train_vectors.csv DESTINATION "${CMAKE_BINARY_DIR}/data")
FILE(COPY data/mnist_test_labels.csv DESTINATION "${CMAKE_BINARY_DIR}/data")
FILE(COPY data/mnist_test_vectors.csv DESTINATION "${CMAKE_BINARY_DIR}/data")
