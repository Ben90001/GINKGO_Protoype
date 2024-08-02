#!/bin/bash

rm -rf ./build
mkdir ./build
cd ./build
cmake ..
make #VERBOSE=1
echo "Running the program: "
./gko-evaluate-solvers
