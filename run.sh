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

# check predictions
echo -e "Train predictions:"
java -jar evaluator.jar trainPredictions data/mnist_train_labels.csv 10
cat Results

echo -e "\n\nActual test predictions:"
java -jar evaluator.jar actualTestPredictions data/mnist_test_labels.csv 10
cat Results

# print total time
echo "Start time: $START_TIME"
echo "Current time: $(date)"
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."
