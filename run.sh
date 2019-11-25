#!/bin/bash

# start time
SECONDS=0
START_TIME=$(date)
echo "$START_TIME"

# remove previous binary
rm magnificence

# compile
echo "Compiling magnificence..."
g++ main.cpp src/*.cpp -O3 -std=c++17 -o magnificence
echo -e "Finished compiling.\n"

# train the model
echo "Current time is $(date)"
./magnificence
